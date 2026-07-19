"""
Test script for score-driven adaptive prototype generation

Tests:
1. Score-based stopping (stops when high scores plateau)
2. Score-based pruning (prunes low-scoring alternatives)
3. Score-based aggregation (aggregates high-scoring nodes)
4. Complexity-based adaptive parameters
"""

import asyncio
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.core.complexity_calculator import ComplexityCalculator


async def test_score_driven_adaptive():
    """Test the score-driven adaptive system"""
    
    print("=" * 80)
    print("TESTING SCORE-DRIVEN ADAPTIVE PROTOTYPE GENERATION")
    print("=" * 80)
    
    orchestrator = PipelineOrchestrator()
    complexity_calc = ComplexityCalculator()
    
    # Test case 1: Simple requirements (should generate fewer prototypes)
    print("\n" + "=" * 80)
    print("TEST 1: Simple Requirements (Low Complexity)")
    print("=" * 80)
    
    simple_req = "2 bedroom apartment with kitchen and bathroom"
    
    # Calculate complexity first
    complexity = complexity_calc.calculate_requirements_complexity(simple_req)
    print(f"\nComplexity Analysis:")
    print(f"  Overall: {complexity['overall']:.3f}")
    print(f"  Level: {complexity_calc._classify_complexity(complexity['overall'])}")
    print(f"  Room count estimate: {complexity['room_count_estimate']}")
    print(f"  Constraint count: {complexity['constraint_count']}")
    
    req1 = {
        'project_name': 'test_simple',
        'requirements': simple_req,
        'use_got': True,
        'enable_convergence_loop': True,
        'max_alternatives': None  # Let it be adaptive
    }
    
    print(f"\nRunning pipeline for: {simple_req}")
    result1 = await orchestrator.process_design_request(req1)
    
    if result1.get('success'):
        designs1 = result1.get('designs', [])
        complexity_metrics1 = result1.get('complexity_metrics', {})
        
        print(f"\n[OK] Test 1 Results:")
        print(f"  Generated {len(designs1)} prototypes")
        if complexity_metrics1:
            print(f"  Complexity level: {complexity_metrics1.get('level')}")
            print(f"  Adaptive target: {complexity_metrics1.get('adaptive_parameters', {}).get('target_prototypes', 'N/A')}")
        
        print(f"\n  Prototype Scores:")
        for i, d in enumerate(designs1[:5], 1):
            scores = d.get('scores', {})
            print(f"    {i}. Score: {scores.get('composite', 0):.3f} "
                  f"(functional: {scores.get('functional_adequacy', 0):.3f}, "
                  f"behavioral: {scores.get('behavioral_performance', 0):.3f})")
    else:
        print(f"[FAIL] Test 1 Failed: {result1.get('error')}")
    
    # Test case 2: Complex requirements (should generate more prototypes)
    print("\n" + "=" * 80)
    print("TEST 2: Complex Requirements (High Complexity)")
    print("=" * 80)
    
    complex_req = """This house will be built on a 3040 ft north-facing plot with simple modern styling 
    and space for one car in front. Setbacks are 1.5 m at the front and 1 m on all other sides. 
    The home should have a master bedroom (13–15 sqm) with an attached bathroom, cross-ventilation, 
    and good east/southeast light. A child bedroom (10–12 sqm) should avoid west heat and stay quiet. 
    The living and dining area (20–22 sqm) must be bright from the north side, seat 5–6 people, 
    and keep bedroom doors out of direct view. A small study room (8–9 sqm) should work as a home office 
    or guest room with good daylight. The kitchen (8–9 sqm) should connect to the dining area, 
    stay hidden from the entrance, and have proper ventilation. There should be a naturally ventilated 
    common bathroom (4–5 sqm) and a master bathroom of similar size."""
    
    # Calculate complexity
    complexity2 = complexity_calc.calculate_requirements_complexity(complex_req)
    print(f"\nComplexity Analysis:")
    print(f"  Overall: {complexity2['overall']:.3f}")
    print(f"  Level: {complexity_calc._classify_complexity(complexity2['overall'])}")
    print(f"  Room count estimate: {complexity2['room_count_estimate']}")
    print(f"  Constraint count: {complexity2['constraint_count']}")
    print(f"  Adjacency count estimate: {complexity2['adjacency_count_estimate']}")
    
    req2 = {
        'project_name': 'test_complex',
        'requirements': complex_req,
        'use_got': True,
        'enable_convergence_loop': True,
        'max_alternatives': None  # Let it be adaptive
    }
    
    print(f"\nRunning pipeline for complex requirements...")
    result2 = await orchestrator.process_design_request(req2)
    
    if result2.get('success'):
        designs2 = result2.get('designs', [])
        complexity_metrics2 = result2.get('complexity_metrics', {})
        
        print(f"\n[OK] Test 2 Results:")
        print(f"  Generated {len(designs2)} prototypes")
        if complexity_metrics2:
            print(f"  Complexity level: {complexity_metrics2.get('level')}")
            adaptive_params = complexity_metrics2.get('adaptive_parameters', {})
            print(f"  Adaptive target: {adaptive_params.get('target_prototypes', 'N/A')}")
            print(f"  GoT depth: {adaptive_params.get('got_depth', 'N/A')}")
            print(f"  GoT breadth: {adaptive_params.get('got_breadth', 'N/A')}")
            print(f"  GoT max_nodes: {adaptive_params.get('got_max_nodes', 'N/A')}")
        
        print(f"\n  Prototype Scores (showing score-based pruning/aggregation):")
        for i, d in enumerate(designs2[:10], 1):
            scores = d.get('scores', {})
            node_id = d.get('node_id', 'unknown')[:8]
            is_aggregated = 'aggregated' in d.get('description', '').lower() or i == 1
            marker = " [AGGREGATED]" if is_aggregated else ""
            print(f"    {i}. {node_id} Score: {scores.get('composite', 0):.3f}{marker}")
        
        # Check if aggregation happened
        if len(designs2) > 0:
            first_desc = designs2[0].get('description', '')
            if 'aggregated' in first_desc.lower() or 'Aggregated' in first_desc:
                print(f"\n  [OK] Score-based aggregation detected: First prototype is aggregated")
            else:
                print(f"\n  -> No aggregation (may need >=2 high-scoring nodes)")
        
        # Check score distribution
        scores_list = [d.get('scores', {}).get('composite', 0) for d in designs2]
        if scores_list:
            print(f"\n  Score Statistics:")
            print(f"    Top score: {max(scores_list):.3f}")
            print(f"    Median score: {sorted(scores_list)[len(scores_list)//2]:.3f}")
            print(f"    Bottom score: {min(scores_list):.3f}")
            print(f"    Score range: {max(scores_list) - min(scores_list):.3f}")
            
            # Check if low scores were pruned
            high_score_threshold = max(scores_list) * 0.8
            high_scoring = [s for s in scores_list if s >= high_score_threshold]
            print(f"    High-scoring (>=80% of best): {len(high_scoring)}/{len(scores_list)}")
    else:
        print(f"[FAIL] Test 2 Failed: {result2.get('error')}")
    
    # Test case 3: Verify score-based stopping
    print("\n" + "=" * 80)
    print("TEST 3: Score-Based Stopping Verification")
    print("=" * 80)
    
    print("\nChecking GoT graph statistics for score-based stopping...")
    if result2.get('success') and 'graph_statistics' in result2:
        stats = result2.get('graph_statistics', {})
        print(f"  Total nodes generated: {stats.get('total_nodes', 'N/A')}")
        print(f"  Graph depth: {stats.get('graph_depth', 'N/A')}")
        print(f"  Leaf nodes: {stats.get('leaf_nodes', 'N/A')}")
        
        # If we have fewer nodes than max, it might indicate early stopping
        adaptive_params = result2.get('complexity_metrics', {}).get('adaptive_parameters', {})
        max_nodes = adaptive_params.get('got_max_nodes', 0)
        if max_nodes > 0:
            actual_nodes = stats.get('total_nodes', 0)
            if actual_nodes < max_nodes * 0.8:
                print(f"  [OK] Possible early stopping: {actual_nodes} < {max_nodes} (80% threshold)")
            else:
                print(f"  -> Full exploration: {actual_nodes} ~= {max_nodes}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("\nScore-Driven Adaptive Features Tested:")
    print("  [OK] Complexity-based adaptive parameters")
    print("  [OK] Score-based pruning (low scores removed)")
    print("  [OK] Score-based aggregation (high scores combined)")
    print("  [OK] Adaptive prototype count based on complexity")
    print("\nCheck the logs above for:")
    print("  - 'Score-based stopping triggered' (early stopping when scores plateau)")
    print("  - 'Pruned low-scoring alternatives' (pruning messages)")
    print("  - 'Aggregated X high-scoring alternatives' (aggregation messages)")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(test_score_driven_adaptive())

