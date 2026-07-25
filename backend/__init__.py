"""FBSL-KAGS backend package.

OpenMP guard (must run before torch / faiss import their native runtimes):
faiss links libomp and torch links libiomp5. When both are loaded in one
process — which happens as soon as RAG retrieval is active (SentenceTransformer
+ faiss search in the same run) — their OpenMP runtimes collide and the process
segfaults with `OMP: Error #15`. Setting this before any heavy import is the
documented, safe-for-inference workaround. `setdefault` respects an explicit
override from the environment.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
