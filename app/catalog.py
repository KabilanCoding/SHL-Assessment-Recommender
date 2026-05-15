"""
Memory-optimized Catalog loader for Render Free tier (512MB RAM limit).

Key change: Uses Google's text-embedding-004 API instead of
loading sentence-transformers + torch locally (saves ~1.2GB RAM).

Embeddings are cached to disk so the API is only called once (first run).
Subsequent starts load from disk: fast and zero RAM overhead.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent
CATALOG_PATH = _BASE / "data" / "shl_catalog.json"
EMBEDDINGS_CACHE = _BASE / "data" / "embeddings_gemini.npy"
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

_STOP_WORDS = {
    "a", "an", "the", "i", "we", "for", "to", "and", "or", "is", "are",
    "be", "was", "were", "need", "want", "looking", "hiring", "assessment",
    "test", "who", "with", "what", "how", "in", "of", "on", "at", "by",
    "this", "that", "it", "its", "some", "have", "has", "would", "could",
    "should", "do", "does", "did", "please", "help", "me", "my", "our",
    "their", "also", "about", "from", "find", "get", "give", "show",
    "recommend", "suggestion", "role", "position", "job",
}

_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 3072  # gemini-embedding-001 output dimension


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _item_text(item: dict) -> str:
    """Composite text for embedding — name doubled for weight."""
    name = item.get("name", "")
    desc = item.get("description", "") or ""
    levels = " ".join(item.get("job_levels", []))
    keys = " ".join(item.get("keys", []))
    return _normalize(f"{name} {name} {desc} {levels} {keys}")


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def _primary_code(keys: list[str]) -> str:
    for k in keys:
        if k in KEY_TO_CODE:
            return KEY_TO_CODE[k]
    return "K"


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
    return (
        f"Name: {item['name']}\n"
        f"URL: {item['link']}\n"
        f"Type: {keys}\n"
        f"Job Levels: {levels}\n"
        f"Duration: {duration}\n"
        f"Languages: {lang_str}\n"
        f"Description: {desc}"
    )


# ---------------------------------------------------------------------------
# Google Embedding API (text-embedding-004)
# ---------------------------------------------------------------------------
def _embed_via_api(texts: list[str], api_key: str, batch_size: int = 50) -> np.ndarray:
    """
    Embed a list of texts using Google's gemini-embedding-001 API.
    Free, zero local RAM, 3072-dim output.
    Auto-retries on 429 with 65-second cooldown.
    """
    import time
    from google import genai

    client = genai.Client(api_key=api_key)
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_num = i // batch_size + 1
        logger.info("Embedding batch %d/%d (%d texts)...", batch_num, total_batches, len(batch))

        while True:
            try:
                result = client.models.embed_content(
                    model=_EMBED_MODEL,
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break  # success — move to next batch
            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    logger.warning("Rate limited on embedding batch %d — waiting 65s...", batch_num)
                    time.sleep(65)
                    continue
                raise  # non-429 errors propagate immediately

    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main index class
# ---------------------------------------------------------------------------
class CatalogIndex:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self._index: faiss.IndexFlatIP | None = None
        self._name_map: dict[str, dict] = {}
        self._url_map: dict[str, dict] = {}

    def load(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")

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

        self._name_map = {it["name"].lower(): it for it in self.items}
        self._url_map = {it["link"]: it for it in self.items}

        # Build or restore embeddings
        if EMBEDDINGS_CACHE.exists():
            logger.info("Loading cached embeddings from %s", EMBEDDINGS_CACHE)
            embeddings = np.load(str(EMBEDDINGS_CACHE)).astype(np.float32)
            if embeddings.shape[0] != len(self.items):
                logger.warning("Cache size mismatch — rebuilding")
                embeddings = self._build_embeddings(api_key)
        else:
            embeddings = self._build_embeddings(api_key)

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info("FAISS index ready (%d vectors, dim=%d)", len(self.items), dim)

    def _build_embeddings(self, api_key: str) -> np.ndarray:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY required to build embeddings")
        logger.info("Building embeddings via text-embedding-004 API...")
        texts = [_item_text(it) for it in self.items]
        emb = _embed_via_api(texts, api_key)
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_CACHE), emb)
        logger.info("Saved %d embeddings to %s", len(emb), EMBEDDINGS_CACHE)
        return emb

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string via the API."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        from google import genai
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model=_EMBED_MODEL,
            contents=[_normalize(query)],
        )
        vec = np.array([result.embeddings[0].values], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 15) -> list[dict]:
        if self._index is None:
            raise RuntimeError("CatalogIndex not loaded.")
        q_emb = self._embed_query(query)
        scores, indices = self._index.search(q_emb, top_k)
        results = []
        for i, score in zip(indices[0], scores[0]):
            if 0 <= i < len(self.items):
                item = dict(self.items[i])
                item["_score"] = float(score)
                results.append(item)
        return results

    def search_hybrid(self, query: str, top_k: int = 15) -> list[dict]:
        """Semantic search + keyword score boosting."""
        candidates = self.search(query, top_k=min(top_k * 2, 30))
        keywords = _extract_keywords(query)
        if not keywords:
            return candidates[:top_k]

        scored: list[tuple[float, dict]] = []
        for item in candidates:
            base = item.get("_score", 0.0)
            item_text = (
                f"{item.get('name','')} {item.get('description','')} "
                f"{' '.join(item.get('keys',[]))} {' '.join(item.get('job_levels',[]))}"
            ).lower()
            hits = sum(1 for kw in keywords if kw in item_text)
            scored.append((base + min(hits * 0.05, 0.25), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def get_by_name(self, name: str) -> dict | None:
        return self._name_map.get(name.lower())

    def get_by_names(self, names: list[str]) -> list[dict]:
        found: list[dict] = []
        seen: set[str] = set()
        for name in names:
            item = self._name_map.get(name.lower())
            if item and item["name"] not in seen:
                found.append(item)
                seen.add(item["name"])
                continue
            nl = name.lower()
            for k, v in self._name_map.items():
                if (nl in k or k in nl) and v["name"] not in seen:
                    found.append(v)
                    seen.add(v["name"])
                    break
        return found

    def resolve_recommendation(self, name: str, url: str) -> dict | None:
        item = self._name_map.get(name.lower())
        if item:
            return item
        nl = name.lower()
        for k, v in self._name_map.items():
            if nl in k or k in nl:
                return v
        return self._url_map.get(url)

    def primary_code(self, item: dict) -> str:
        return _primary_code(item.get("keys", []))

    def _ensure_catalog(self) -> None:
        if CATALOG_PATH.exists():
            logger.info("Catalog file found: %s", CATALOG_PATH)
            return
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading catalog from %s ...", CATALOG_URL)
        urllib.request.urlretrieve(CATALOG_URL, str(CATALOG_PATH))
        logger.info("Catalog downloaded (%d bytes)", CATALOG_PATH.stat().st_size)


catalog = CatalogIndex()
