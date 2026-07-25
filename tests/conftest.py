"""Shared pytest setup for the FBSL-KAGS suite.

Sets the OpenMP guard (torch/faiss runtime collision) before any heavy import
and puts the repo root on sys.path so `import backend...` resolves.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
