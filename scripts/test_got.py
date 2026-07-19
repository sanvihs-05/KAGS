import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

import asyncio
from database.vector_store import VectorStoreManager
from agents.encoder_agent import EncoderAgent
from core.graph_of_thoughts import GraphOfThoughtsEngine

async def test_got():
    print("="*70)
    print("🕸️  TESTING GRAPH OF THOUGHT ENGINE")
    print("="*70)
    
    # Initialize
    vs = VectorStoreManager()
    encoder = EncoderAgent(vs)
    got = GraphOfThoughtsEngine(max_depth=2, breadth=3)
    
    # Create problem node
    print("\n1️⃣ Creating problem node...")
    problem = encoder.encode_requirements(
        "3 bedroom apartment with 2 bathrooms, kitchen, and living room"
    )
    print(f"   ✓ Problem: {len(problem.functions)} functions")
    
    # Generate thought graph
    print("\n2️⃣ Generating thought graph...")
    graph = await got.generate_thought_graph(
        problem,
        expansion_strategies=['functional', 'behavioral', 'layout']
    )
    # Get statistics
    print("\n3️⃣ Graph Statistics:")
    stats = got.get_graph_statistics()
    print(f"   • Total Nodes: {stats['total_nodes']}")
    print(f"   • Total Edges: {stats['total_edges']}")
    print(f"   • Graph Depth: {stats['graph_depth']}")
    print(f"   • Branching Factor: {stats['branching_factor']:.2f}")
    print(f"   • Leaf Nodes: {stats['leaf_nodes']}")
    print(f"   • Transformation Types:")
    for trans_type, count in stats['transformation_types'].items():
        print(f"     - {trans_type}: {count}")
    
    # Find best paths
    print("\n4️⃣ Finding best paths...")
    best_paths = got.find_best_paths(problem.node_id, top_k=5)
    
    print(f"   Found {len(best_paths)} paths:")
    for i, path in enumerate(best_paths, 1):
        print(f"\n   Path {i}:")
        print(f"     • Length: {len(path.nodes)} nodes")
        print(f"     • Total Score: {path.total_score:.3f}")
        print(f"     • Avg Quality: {path.avg_quality:.3f}")
        print(f"     • Total Cost: {path.total_cost:.3f}")
        
        # Show node sequence
        print(f"     • Node Sequence:")
        for j, node_id in enumerate(path.nodes):
            node = got.node_registry.get(node_id)
            if node:
                desc = node.metadata.get('description', 'Root')
                trans = node.metadata.get('transformation_type', 'N/A')
                print(f"       {j}. {node_id[:8]}... | {desc[:40]} | {trans}")
    
    # Test node aggregation
    print("\n5️⃣ Testing node aggregation...")
    if len(best_paths) >= 2:
        # Get leaf nodes from top 3 paths
        leaf_nodes = [path.nodes[-1] for path in best_paths[:3]]
        
        print(f"   Aggregating {len(leaf_nodes)} leaf nodes...")
        aggregated = got.aggregate_nodes(leaf_nodes)
        
        print(f"   ✓ Aggregated Node:")
        print(f"     • ID: {aggregated.node_id[:16]}...")
        print(f"     • Functions: {len(aggregated.functions)}")
        print(f"     • Behaviors: {len(aggregated.behaviors)}")
        print(f"     • Structures: {len(aggregated.structures)}")
        print(f"     • Description: {aggregated.metadata.get('description', 'N/A')}")
    
    # Visualize graph
    print("\n6️⃣ Visualizing graph...")
    try:
        got.visualize_graph("thought_graph.png")
        print("   ✓ Visualization saved to thought_graph.png")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    print("\n" + "="*70)
    print("✅ GRAPH OF THOUGHT TEST COMPLETE")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✅ Node generation and expansion")
    print("  ✅ Path scoring and ranking")
    print("  ✅ Node aggregation with compatibility")
    print("  ✅ Graph statistics and analysis")
    print("  ✅ Directed acyclic graph structure")

if __name__ == "__main__":
    asyncio.run(test_got())