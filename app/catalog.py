"""
Enhanced Catalog loader and FAISS semantic-search index.

Improvements over v1:
  - Text normalization before embedding (lowercase, remove punctuation)
    for better similarity on technical terms.
  - Hybrid search: semantic FAISS + keyword boosting for precision.
  - Disk-cached embeddings for fast restarts (< 5 s after first run).
  - Fuzzy name resolution for comparison queries.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent
CATALOG_PATH = _BASE / "data" / "shl_catalog.json"
EMBEDDINGS_CACHE = _BASE / "data" / "embeddings_v2.npy"   # v2 = normalized
CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# Stop words for keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "i", "we", "for", "to", "and", "or", "is", "are",
    "be", "was", "were", "need", "want", "looking", "hiring", "assessment",
    "test", "who", "with", "what", "how", "in", "of", "on", "at", "by",
    "this", "that", "it", "its", "some", "have", "has", "would", "could",
    "should", "do", "does", "did", "please", "help", "me", "my", "our",
    "their", "also", "about", "from", "find", "get", "give", "show",
    "recommend", "suggestion", "looking", "role", "position", "job",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _normalize_for_embedding(text: str) -> str:
    """Lowercase + remove punctuation for better embedding similarity on tech terms."""
    text = text.lower()
    # Replace punctuation with space (keeps word boundaries for "C#" → "c ")
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _item_embedding_text(item: dict) -> str:
    """Normalized composite text used for FAISS embedding."""
    name = item.get("name", "")
    desc = item.get("description", "")
    levels = " ".join(item.get("job_levels", []))
    keys = " ".join(item.get("keys", []))
    langs = " ".join(item.get("languages", []))
    duration = item.get("duration", "")
    raw = f"{name} {name} {desc} {levels} {keys} {langs} {duration}"
    # Name is doubled to boost its weight in similarity
    return _normalize_for_embedding(raw)


def _primary_code(keys: list[str]) -> str:
    for k in keys:
        if k in KEY_TO_CODE:
            return KEY_TO_CODE[k]
    return "K"


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful lowercase keywords from query text."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# Compact snippet for LLM context
# ---------------------------------------------------------------------------
def item_to_snippet(item: dict) -> str:
    keys = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    langs = item.get("languages", [])
    lang_str = ", ".join(langs[:6])
    if len(langs) > 6:
        lang_str += f" (+{len(langs)-6} more)"
    duration = item.get("duration", "N/A") or "N/A"
    desc = (item.get("description", "") or "")[:300]
    remote = item.get("remote", "")
    adaptive = item.get("adaptive", "")
    return (
        f"Name: {item['name']}\n"
        f"URL: {item['link']}\n"
        f"Type: {keys}\n"
        f"Job Levels: {levels}\n"
        f"Duration: {duration}\n"
        f"Languages: {lang_str}\n"
        f"Remote: {remote} | Adaptive: {adaptive}\n"
        f"Description: {desc}"
    )


# ---------------------------------------------------------------------------
# Main index class
# ---------------------------------------------------------------------------
class CatalogIndex:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self._index: faiss.IndexFlatIP | None = None
        self._model: SentenceTransformer | None = None
        self._name_map: dict[str, dict] = {}   # lowercase name → item
        self._url_map: dict[str, dict] = {}    # url → item

    # ------------------------------------------------------------------
    def load(self) -> None:
        self._ensure_catalog()

        logger.info("Loading catalog from %s", CATALOG_PATH)
        with open(CATALOG_PATH, encoding="utf-8") as f:
            raw_text = f.read()

        try:
            raw: list[dict] = json.loads(raw_text)
        except json.JSONDecodeError:
            raw: list[dict] = json.loads(raw_text, strict=False)

        self.items = [it for it in raw if it.get("status") == "ok"]
        logger.info("Catalog items (status=ok): %d", len(self.items))

        # Build lookup maps
        self._name_map = {it["name"].lower(): it for it in self.items}
        self._url_map = {it["link"]: it for it in self.items}

        # Load embedding model
        logger.info("Loading sentence-transformer model ...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build or restore embeddings
        if EMBEDDINGS_CACHE.exists():
            logger.info("Loading cached embeddings from %s", EMBEDDINGS_CACHE)
            embeddings = np.load(str(EMBEDDINGS_CACHE)).astype(np.float32)
            if embeddings.shape[0] != len(self.items):
                logger.warning("Cache size mismatch — rebuilding")
                embeddings = self._build_embeddings()
        else:
            embeddings = self._build_embeddings()

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(
            "FAISS index ready (%d vectors, dim=%d) [normalized embeddings]",
            len(self.items), dim,
        )

    def _build_embeddings(self) -> np.ndarray:
        logger.info("Building normalized embeddings for %d items ...", len(self.items))
        texts = [_item_embedding_text(it) for it in self.items]
        embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
        )
        emb = np.array(embeddings, dtype=np.float32)
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_CACHE), emb)
        logger.info("Embeddings saved to %s", EMBEDDINGS_CACHE)
        return emb

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Pure semantic search via FAISS."""
        if self._index is None or self._model is None:
            raise RuntimeError("CatalogIndex not loaded.")
        q_norm = _normalize_for_embedding(query)
        q_emb = self._model.encode([q_norm], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, top_k)
        results = []
        for i, score in zip(indices[0], scores[0]):
            if 0 <= i < len(self.items):
                item = dict(self.items[i])
                item["_score"] = float(score)
                results.append(item)
        return results

    def search_hybrid(self, query: str, top_k: int = 15) -> list[dict]:
        """
        Hybrid search: semantic FAISS + keyword boosting.

        1. Get top-30 candidates from FAISS.
        2. Boost scores for items whose text contains query keywords.
        3. Return re-ranked top_k.

        This improves precision for specific skills (e.g. "Java", "Finance").
        """
        candidates = self.search(query, top_k=min(top_k * 2, 30))
        keywords = _extract_keywords(query)

        if not keywords:
            return candidates[:top_k]

        scored: list[tuple[float, dict]] = []
        for item in candidates:
            base_score = item.get("_score", 0.0)
            # Check keywords against the raw (unnormalized) display text
            item_text = (
                f"{item.get('name','')} {item.get('description','')} "
                f"{' '.join(item.get('keys',[]))} {' '.join(item.get('job_levels',[]))}"
            ).lower()
            # Keyword match bonus: 0.05 per matched keyword (capped at 0.25)
            hits = sum(1 for kw in keywords if kw in item_text)
            keyword_bonus = min(hits * 0.05, 0.25)
            scored.append((base_score + keyword_bonus, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    def get_by_name(self, name: str) -> dict | None:
        """Exact (case-insensitive) lookup."""
        return self._name_map.get(name.lower())

    def get_by_names(self, names: list[str]) -> list[dict]:
        """Lookup multiple items — exact then fuzzy."""
        found: list[dict] = []
        seen: set[str] = set()
        for name in names:
            item = self._name_map.get(name.lower())
            if item and item["name"] not in seen:
                found.append(item)
                seen.add(item["name"])
                continue
            # Fuzzy: look for substring match
            nl = name.lower()
            for k, v in self._name_map.items():
                if (nl in k or k in nl) and v["name"] not in seen:
                    found.append(v)
                    seen.add(v["name"])
                    break
        return found

    def resolve_recommendation(self, name: str, url: str) -> dict | None:
        """Find catalog item by name (exact then fuzzy), fallback to URL."""
        item = self._name_map.get(name.lower())
        if item:
            return item
        # Fuzzy name match
        nl = name.lower()
        for k, v in self._name_map.items():
            if nl in k or k in nl:
                return v
        # URL fallback
        return self._url_map.get(url)

    def primary_code(self, item: dict) -> str:
        return _primary_code(item.get("keys", []))

    # ------------------------------------------------------------------
    def _ensure_catalog(self) -> None:
        if CATALOG_PATH.exists():
            logger.info("Catalog file found: %s", CATALOG_PATH)
            return
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading catalog from %s ...", CATALOG_URL)
        urllib.request.urlretrieve(CATALOG_URL, str(CATALOG_PATH))
        logger.info("Catalog downloaded (%d bytes)", CATALOG_PATH.stat().st_size)


catalog = CatalogIndex()
