from typing import List, Dict, Any, Union
import numpy as np
from pathlib import Path
import logging

from .faiss_index import FaissIndexManager
from .embedding_loader import FinnishFloorPlanEmbeddingLoader

logger = logging.getLogger(__name__)


class FaissClient:
    """Simple programmatic facade around FaissIndexManager for other modules.

    Example:
        client = FaissClient(embeddings_dir='enhanced_multimodal_rag_store', index_dir='data/faiss_index')
        client.ensure_index()  # loads existing index or builds it
        results = client.search('kitchen near dining', top_k=5)
    """

    def __init__(self, embeddings_dir: str = 'enhanced_multimodal_rag_store', index_dir: str = 'data/faiss_index', embedding_type: str = 'composite'):
        self.embeddings_dir = Path(embeddings_dir)
        self.index_dir = Path(index_dir)
        self.embedding_type = embedding_type

        # Lazy init
        self.loader: FinnishFloorPlanEmbeddingLoader | None = None
        self.mgr: FaissIndexManager | None = None

    def _init_loader_and_mgr(self):
        if self.loader is None:
            self.loader = FinnishFloorPlanEmbeddingLoader(str(self.embeddings_dir))
        if self.mgr is None:
            self.mgr = FaissIndexManager(self.loader, embedding_type=self.embedding_type)

    def ensure_index(self, build_if_missing: bool = True) -> None:
        """Load existing index from disk or build it from embeddings.

        If `build_if_missing` is True, will build the index automatically when missing.
        """
        self._init_loader_and_mgr()

        try:
            self.mgr.load(str(self.index_dir))
            logger.info("FAISS index loaded from %s", self.index_dir)
        except Exception:
            if not build_if_missing:
                raise
            logger.info("FAISS index not found, building...")
            self.mgr.build_index()
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.mgr.save(str(self.index_dir))

    def search(self, query: Union[str, np.ndarray], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search by query text or a pre-computed embedding vector.

        - If `query` is a string, it will be encoded via the loader's embedding model.
        - If `query` is a numpy array, it will be used directly.
        Returns a list of dicts with keys `index`, `score`, and `metadata`.
        """
        self._init_loader_and_mgr()
        if self.mgr is None or self.mgr.index is None:
            self.ensure_index(build_if_missing=True)

        # If text, delegate to FaissIndexManager which handles encoding
        if isinstance(query, str):
            return self.mgr.search(query, top_k=top_k)

        # If a vector is provided, run query directly against FAISS index
        if isinstance(query, np.ndarray):
            vec = query.astype('float32')
            # normalize for cosine similarity
            faiss = __import__('faiss')
            faiss.normalize_L2(vec.reshape(1, -1))
            D, I = self.mgr.index.search(vec.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                meta = None
                if 0 <= int(idx) < len(self.mgr.meta):
                    meta = self.mgr.meta[int(idx)]
                results.append({'index': int(idx), 'score': float(score), 'metadata': meta})
            return results

        raise ValueError('Query must be either str or numpy.ndarray')
