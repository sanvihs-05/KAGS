FAISS index builder
===================

This folder contains a small script to build a FAISS index from the precomputed embeddings in `enhanced_multimodal_rag_store`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
# on Windows you may prefer: python -m pip install faiss-cpu
```

Build an index:

```powershell
python scripts/build_faiss_index.py --embeddings-dir enhanced_multimodal_rag_store --out-dir data/faiss_index --embedding-type composite
```

Query the index (example):

```python
from backend.utils.embedding_loader import FinnishFloorPlanEmbeddingLoader
from backend.utils.faiss_index import FaissIndexManager

loader = FinnishFloorPlanEmbeddingLoader('enhanced_multimodal_rag_store')
mgr = FaissIndexManager(loader)
mgr.load('data/faiss_index')
results = mgr.search('large living room', top_k=5)
for r in results:
    print(r)
```

Notes:
- FAISS returns inner-product scores when using normalized vectors; results are comparable to cosine similarity.
- If you prefer a managed vector DB, consider building a Chroma collection instead.
