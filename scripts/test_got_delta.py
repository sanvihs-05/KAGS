"""Test delta-based stopping for GraphOfThoughtsEngine

This script monkeypatches `GraphOfThoughtsEngine.expand_node` to return children
with controlled composite_score values to verify that generate_thought_graph stops
when improvements fall below delta for the configured patience.

Run:
  py scripts\test_got_delta.py
"""
import sys
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from backend.core.graph_of_thoughts import GraphOfThoughtsEngine
from backend.core.fbsl_models import FBSLLayoutNode


def make_child_with_score(parent: FBSLLayoutNode, score: float) -> FBSLLayoutNode:
    import copy
    child = copy.deepcopy(parent)
    child.node_id = f"child_{int(score*1000)}"
    child.composite_score = float(score)
    return child


async def fake_expand_node(self, node, strategies):
    """Return children with diminishing improvements to trigger delta stopping.
    Sequence of best scores simulated: 0.50 -> 0.60 -> 0.605 -> 0.606 (tiny improvements)
    With delta=0.01 and patience=2 stopping should occur after small improvements.
    """
    # Use engine-level step counter to output a controlled sequence of best scores
    if not hasattr(self, '_fake_steps'):
        self._fake_steps = 0

    # Predefined best scores sequence (diminishing improvements)
    seq = [0.60, 0.605, 0.606, 0.6062, 0.60625]
    idx = min(self._fake_steps, len(seq) - 1)
    target = seq[idx]

    # create two children at the target score (duplicates to populate breadth)
    children = [make_child_with_score(node, target), make_child_with_score(node, target)]

    self._fake_steps += 1
    return children


async def run_test():
    engine = GraphOfThoughtsEngine(max_depth=10, breadth=2)

    # Add a root with initial composite score 0.50
    root = FBSLLayoutNode()
    root.node_id = 'root'
    root.composite_score = 0.5
    engine._add_node_to_graph(root)

    # Monkeypatch expand_node
    engine.expand_node = fake_expand_node.__get__(engine, GraphOfThoughtsEngine)

    # Run generator with delta-based stopping: delta=0.01, patience=2
    graph = await engine.generate_thought_graph(root, expansion_strategies=['layout'], delta=0.01, patience=2)

    stats = engine.get_graph_statistics()
    print('Graph stats after generation:')
    print(f"  total_nodes: {stats['total_nodes']}")
    print(f"  total_edges: {stats['total_edges']}")

    # Print best composite score seen
    best = max((n.composite_score for n in engine.node_registry.values()), default=0.0)
    print(f"Best composite score observed: {best}")


if __name__ == '__main__':
    asyncio.run(run_test())
