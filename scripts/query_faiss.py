"""Query the FAISS index and print results.

Run from PowerShell like:
  python scripts\query_faiss.py --index-dir data/faiss_index --query "large living room" --top-k 5
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import backend` works when running this script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/faiss_index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        from backend.utils.embedding_loader import FinnishFloorPlanEmbeddingLoader
        from backend.utils.faiss_index import FaissIndexManager
    except Exception as e:
        print("Failed to import modules:", e)
        raise

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"Index directory not found: {index_dir}. Build it first with scripts/build_faiss_index.py")
        return

    loader = FinnishFloorPlanEmbeddingLoader('enhanced_multimodal_rag_store')
    mgr = FaissIndexManager(loader)
    try:
        mgr.load(str(index_dir))
    except Exception as e:
        print("Failed to load FAISS index:", e)
        return

    results = mgr.search(args.query, top_k=args.top_k)
    if not results:
        print("No results")
        return

    for i, r in enumerate(results, start=1):
        meta = r.get('metadata') or {}
        print(f"{i}. score={r['score']:.4f} index={r['index']} plan_id={meta.get('plan_id')} room_type={meta.get('room_type')} text={meta.get('text')}")


if __name__ == '__main__':
    main()
