"""Demonstration script: run ResearchAgent.research_node using FAISS-backed retrieval

Usage:
  python scripts\demo_research_agent.py
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.vector_store import VectorStoreManager
from backend.agents.research_agent import ResearchAgent
from backend.core.fbsl_models import FBSLLayoutNode, Function


def main():
    print("Initializing VectorStoreManager (may take a moment)...")
    vsm = VectorStoreManager()

    # Create ResearchAgent
    agent = ResearchAgent(vsm)

    # Grab a sample composite embedding from the loader if available
    sample_embedding = None
    try:
        loader = vsm.finnish_embeddings
        if loader and 'composite' in loader.embeddings:
            emb = loader.embeddings['composite']
            if len(emb) > 0:
                sample_embedding = np.array(emb[0])
    except Exception as e:
        print("Could not fetch sample embedding from loader:", e)

    if sample_embedding is None:
        # Fallback: random vector (not ideal, but shows pipeline)
        print("No precomputed embedding found — using random vector fallback")
        sample_embedding = np.random.randn(384).astype('float32')

    # Construct a minimal FBSL node with one function that contains the embedding
    node = FBSLLayoutNode()
    func = Function(name="demo_function", description="Demo function embedding")
    func.embedding = sample_embedding
    node.add_function(func)

    print("Running research... (FAISS will be used if available)")
    findings = agent.research_node(node, depth=5)

    print("Found precedents:")
    for i, p in enumerate(findings.get('similar_spaces', [])[:10], start=1):
        print(f"{i}. similarity={p.get('similarity'):.4f} plan={p.get('plan_id')} room_type={p.get('room_type')} text={p.get('translated_text')}")


if __name__ == '__main__':
    main()
