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

# Load .env from the project root, if present.
#
# `python-dotenv` was already a declared dependency but was never called, so an
# API key had to be exported by hand in whichever shell launched the server.
# Miss that and the encoder silently falls back to the rule-based parser, which
# always succeeds and returns a generic room programme — the failure looks like
# the system ignoring the brief rather than a missing credential.
#
# Loading here rather than in main.py covers every entry point that imports the
# package: the API server, the scripts, and the tests. `override=False` keeps a
# variable already set in the environment winning over the file.
try:
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except Exception:  # dotenv absent or unreadable — env vars still work
    pass
