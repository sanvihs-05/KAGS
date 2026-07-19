from pathlib import Path
import json
import numpy as np
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import faiss
except Exception as e:
    faiss = None
    logger.warning("faiss not available: %s", e)


class FaissIndexManager:
    """Builds, saves and queries a FAISS index from embeddings loaded by
    `FinnishFloorPlanEmbeddingLoader`.

    Usage:
      - Create loader = FinnishFloorPlanEmbeddingLoader(path)
      - mgr = FaissIndexManager(loader)
      - mgr.build_index(embedding_type='composite')
      - mgr.save(index_dir)
      - mgr.load(index_dir)
      - mgr.search(query, top_k=10)
    """

    def __init__(self, loader, embedding_type: str = 'composite'):
        self.loader = loader
        self.embedding_type = embedding_type
        self.index = None
        self.index_dim = None
        self.index_count = 0
        self.meta: List[Dict[str, Any]] = []

    def build_index(self, normalize: bool = True) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install with `pip install faiss-cpu`.")

        if self.embedding_type not in self.loader.embeddings:
            raise ValueError(f"Embedding type '{self.embedding_type}' not found in loader")

        embs = self.loader.embeddings[self.embedding_type]
        if embs is None or len(embs) == 0:
            raise ValueError("No embeddings available to build index")

        embs = np.asarray(embs, dtype='float32')
        self.index_dim = embs.shape[1]

        # Use inner-product on normalized vectors to perform cosine similarity
        if normalize:
            faiss.normalize_L2(embs)

        self.index = faiss.IndexFlatIP(self.index_dim)
        self.index.add(embs)
        self.index_count = self.index.ntotal

        # Build metadata for each vector (minimal set)
        self.meta = []
        if self.loader.metadata:
            for m in self.loader.metadata:
                # keep only a few fields to keep the metadata small
                self.meta.append({
                    'plan_id': m.get('plan_id'),
                    'room_type': m.get('room_type'),
                    'text': m.get('text'),
                    'translated': m.get('translated'),
                    'function': m.get('function'),
                    'has_visual_features': m.get('has_visual_features', False)
                })
        logger.info("Built FAISS index (dim=%s, count=%s)", self.index_dim, self.index_count)

    def save(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise RuntimeError("Index not built")

        idx_file = out_path / 'faiss_index.index'
        meta_file = out_path / 'faiss_index_meta.json'

        faiss.write_index(self.index, str(idx_file))

        meta = {
            'embedding_type': self.embedding_type,
            'index_dim': self.index_dim,
            'index_count': self.index_count,
            'metadata': self.meta
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info("Saved FAISS index to %s", out_path)

    def load(self, index_dir: str) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install with `pip install faiss-cpu`.")

        idx_file = Path(index_dir) / 'faiss_index.index'
        meta_file = Path(index_dir) / 'faiss_index_meta.json'

        if not idx_file.exists() or not meta_file.exists():
            raise FileNotFoundError("Index files not found in %s" % index_dir)

        self.index = faiss.read_index(str(idx_file))

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.embedding_type = meta.get('embedding_type', self.embedding_type)
        self.index_dim = meta.get('index_dim')
        self.index_count = meta.get('index_count')
        self.meta = meta.get('metadata', [])

        logger.info("Loaded FAISS index (dim=%s, count=%s)", self.index_dim, self.index_count)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built or loaded")

        q_emb = self.loader.encode_query(query)
        q_emb = np.asarray(q_emb, dtype='float32')

        # normalize for cosine
        faiss.normalize_L2(q_emb.reshape(1, -1))

        D, I = self.index.search(q_emb.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if int(idx) < 0 or int(idx) >= len(self.meta):
                meta = None
            else:
                meta = self.meta[int(idx)]

            results.append({
                'index': int(idx),
                'score': float(score),
                'metadata': meta
            })
        return results
