"""Test script for GraphOfThoughtsEngine.aggregate_nodes

Creates several FBSLLayoutNode samples with varying composite scores
and structural materials to exercise the compatibility and aggregation
logic implemented in backend/core/graph_of_thoughts.py

Run:
  python scripts\test_got_aggregation.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Ensure backend package path is importable so modules using `from core...` work
sys.path.insert(0, str(ROOT / 'backend'))

from backend.core.graph_of_thoughts import GraphOfThoughtsEngine
from backend.core.fbsl_models import FBSLLayoutNode, Function, Structure


def make_node(name: str, composite: float, material: str) -> FBSLLayoutNode:
    node = FBSLLayoutNode()
    node.node_id = name
    node.composite_score = composite

    # Add a simple structure with material to influence compatibility
    struct = Structure(name=f"struct_{name}", material_type=material)
    node.structures[struct.structure_id] = struct

    # Add a simple function as placeholder
    func = Function(name=f"func_{name}")
    node.functions[func.function_id] = func

    return node


def main():
    engine = GraphOfThoughtsEngine(max_depth=2, breadth=2)

    # Create nodes: 3 'concrete' (high scores) and 1 'wood' (lower score)
    nA = make_node('A', 0.90, 'concrete')
    nB = make_node('B', 0.88, 'concrete')
    nC = make_node('C', 0.70, 'wood')
    nD = make_node('D', 0.85, 'concrete')

    # Add nodes into engine registry and graph
    for n in (nA, nB, nC, nD):
        engine._add_node_to_graph(n)

    node_ids = [nA.node_id, nB.node_id, nC.node_id, nD.node_id]

    print("Node qualities:")
    for nid in node_ids:
        node = engine.node_registry[nid]
        print(f"  {nid}: composite_score={node.composite_score}")

    print('\nRunning aggregation (top_k=3, compatibility_threshold=0.5, selection_metric="composite")')
    aggregated = engine.aggregate_nodes(node_ids, top_k=3, compatibility_threshold=0.5, selection_metric='composite')

    print('\nAggregation result:')
    print('  aggregated.node_id:', aggregated.node_id)
    print('  aggregated.metadata.description:', aggregated.metadata.get('description'))
    print('  aggregated.metadata.aggregated_from:', aggregated.metadata.get('aggregated_from'))

    print('\nAggregated functions count:', len(aggregated.functions))
    print('Aggregated behaviors count:', len(aggregated.behaviors))
    print('Aggregated structures count:', len(aggregated.structures))

if __name__ == '__main__':
    main()
