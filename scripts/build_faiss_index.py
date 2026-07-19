"""CLI to build a FAISS index from the existing Finnish floorplan embeddings.

Example:
  python scripts/build_faiss_index.py --embeddings-dir enhanced_multimodal_rag_store --out-dir data/faiss_index --embedding-type composite
"""
import argparse
from pathlib import Path
import logging
import sys

# Ensure project root is on sys.path so `import backend` works when running this script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True, help="Path to embeddings folder (where plan_mapping.json, consolidated_metadata.json and *_embeddings.npy live)")
    parser.add_argument("--out-dir", required=True, help="Directory to write the faiss index and metadata")
    parser.add_argument("--embedding-type", default="composite", help="Which embedding array to use (composite|text|architectural|spatial|visual)")
    args = parser.parse_args()

    # Import locally to avoid requiring FAISS at module import time
    try:
        from backend.utils.embedding_loader import FinnishFloorPlanEmbeddingLoader
        from backend.utils.faiss_index import FaissIndexManager
    except Exception as e:
        logger.error("Failed to import project modules: %s", e)
        raise

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        raise SystemExit(f"Embeddings directory not found: {embeddings_dir}")

    logger.info("Loading embeddings from %s", embeddings_dir)
    loader = FinnishFloorPlanEmbeddingLoader(str(embeddings_dir))

    mgr = FaissIndexManager(loader, embedding_type=args.embedding_type)
    logger.info("Building FAISS index (embedding_type=%s)", args.embedding_type)
    mgr.build_index()

    out_dir = Path(args.out_dir)
    mgr.save(str(out_dir))
    logger.info("FAISS index built and saved to %s", out_dir)


if __name__ == '__main__':
    main()
