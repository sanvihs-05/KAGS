"""
Test Pareto Optimizer with Existing Prototypes

This script tests the Pareto optimality implementation using the 5 prototypes
from the Sustainable Family Home test case.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.pareto_optimizer import ParetoOptimizer
from backend.core.fbsl_models import FBSLLayoutNode, NodeType
import uuid


def create_test_prototypes():
    """Create 5 test prototypes with scores matching the demo"""
    
    prototypes = []
    
    # DP-1: Compact Efficiency
    dp1 = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    dp1.functional_score = 0.89
    dp1.behavioral_score = 0.88
    dp1.structural_score = 0.86
    dp1.layout_score = 0.92
    dp1.sustainability_score = 0.82
    dp1.composite_score = 0.876
    prototypes.append(dp1)
    
    # DP-2: Solar Passive (Best overall)
    dp2 = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    dp2.functional_score = 0.88
    dp2.behavioral_score = 0.92
    dp2.structural_score = 0.89
    dp2.layout_score = 0.87
    dp2.sustainability_score = 0.95
    dp2.composite_score = 0.892
    prototypes.append(dp2)
    
    # DP-3: Family Interaction
    dp3 = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    dp3.functional_score = 0.93
    dp3.behavioral_score = 0.89
    dp3.structural_score = 0.85
    dp3.layout_score = 0.88
    dp3.sustainability_score = 0.80
    dp3.composite_score = 0.885
    prototypes.append(dp3)
    
    # DP-4: Privacy-Focused
    dp4 = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    dp4.functional_score = 0.91
    dp4.behavioral_score = 0.90
    dp4.structural_score = 0.87
    dp4.layout_score = 0.83
    dp4.sustainability_score = 0.78
    dp4.composite_score = 0.868
    prototypes.append(dp4)
    
    # DP-5: Flexible Adaptable
    dp5 = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    dp5.functional_score = 0.90
    dp5.behavioral_score = 0.88
    dp5.structural_score = 0.84
    dp5.layout_score = 0.87
    dp5.sustainability_score = 0.81
    dp5.composite_score = 0.879
    prototypes.append(dp5)
    
    return prototypes


def test_pareto_optimizer():
    """Test all Pareto optimizer functionality"""
    
    print("=" * 80)
    print("PARETO OPTIMIZER TEST")
    print("=" * 80)
    print()
    
    # Create test prototypes
    prototypes = create_test_prototypes()
    print(f"✓ Created {len(prototypes)} test prototypes")
    print()
    
    # Initialize optimizer
    optimizer = ParetoOptimizer()
    print("✓ Initialized Pareto Optimizer")
    print()
    
    # Test 1: Dominance Checking
    print("TEST 1: Dominance Checking")
    print("-" * 80)
    
    # DP-2 should NOT dominate DP-3 (DP-3 has higher functional score)
    dominates_23 = optimizer.check_dominance(prototypes[1], prototypes[2])
    print(f"DP-2 dominates DP-3: {dominates_23} (Expected: False)")
    
    # DP-3 should NOT dominate DP-2 (DP-2 has higher sustainability)
    dominates_32 = optimizer.check_dominance(prototypes[2], prototypes[1])
    print(f"DP-3 dominates DP-2: {dominates_32} (Expected: False)")
    
    print()
    
    # Test 2: Pareto Frontier Identification
    print("TEST 2: Pareto Frontier Identification")
    print("-" * 80)
    
    pareto_frontier = optimizer.identify_pareto_frontier(prototypes)
    print(f"Pareto frontier size: {len(pareto_frontier)}/{len(prototypes)}")
    print(f"Expected: All 5 prototypes should be non-dominated")
    print()
    
    # Test 3: Trade-off Characterization
    print("TEST 3: Trade-off Characterization")
    print("-" * 80)
    
    for i, prototype in enumerate(pareto_frontier, 1):
        analysis = optimizer.characterize_trade_offs(prototype, pareto_frontier)
        print(f"\nDP-{i}:")
        print(f"  Composite: {analysis['composite_score']:.3f}")
        if analysis['champions']:
            print(f"  Champions: {', '.join(analysis['champions'])}")
        if analysis['trade_offs']:
            print(f"  Trade-offs:")
            for dim, cost in analysis['trade_offs'][:3]:  # Top 3
                print(f"    - {dim}: -{cost:.3f}")
    
    print()
    
    # Test 4: Full Trade-off Report
    print("TEST 4: Full Trade-off Report")
    print("-" * 80)
    
    report = optimizer.generate_trade_off_report(pareto_frontier)
    print(report)
    print()
    
    # Test 5: Pareto Statistics
    print("TEST 5: Pareto Statistics")
    print("-" * 80)
    
    stats = optimizer.get_pareto_statistics(pareto_frontier)
    print(f"Frontier size: {stats['size']}")
    print(f"Champion counts:")
    for dim, count in stats['champion_counts'].items():
        print(f"  {dim}: {count}")
    print(f"Average trade-off cost: {stats['avg_trade_off_cost']:.3f}")
    print()
    
    # Test 6: Pareto-Preserving Selection
    print("TEST 6: Pareto-Preserving Selection")
    print("-" * 80)
    
    selected = optimizer.pareto_preserving_selection(prototypes, target_count=3)
    print(f"Selected {len(selected)} solutions (target: 3)")
    for i, sol in enumerate(selected, 1):
        print(f"  {i}. Composite: {sol.composite_score:.3f}")
    print()
    
    print("=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_pareto_optimizer()
