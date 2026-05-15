"""Quick catalog v2 verification script"""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from app.catalog import catalog

catalog.load()
print(f"Catalog: {len(catalog.items)} items")
print(f"FAISS index: {catalog._index.ntotal} vectors")

# Test hybrid search
results = catalog.search_hybrid("Java developer stakeholder communication", top_k=5)
print("\nHybrid search (Java developer):")
for r in results:
    name = r["name"]
    score = r.get("_score", 0)
    print(f"  {name}  [score={score:.3f}]")

# Test comparison lookup
items = catalog.get_by_names(["OPQ32r", "Global Skills Assessment"])
print(f"\nComparison lookup: found {len(items)} items")
for it in items:
    print(f"  -> {it['name']}")
