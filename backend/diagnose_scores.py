"""
Diagnostic script to understand why scores are low
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.orchestrator import PipelineOrchestrator
from .core.fbsl_models import FBSLLayoutNode, NodeType
from agents.scoring_agent import ScoringAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def diagnose_scoring():
    """Diagnose why scores are low"""
    
    print("=" * 80)
    print("🔍 Score Diagnosis")
    print("=" * 80)
    
    # Create a test node
    from .core.fbsl_models import Function, Behavior, Layout, Room, FunctionCategory, BehaviorCategory
    
    node = FBSLLayoutNode(node_type=NodeType.DESIGN_PROTOTYPE)
    
    # Add a function
    func = Function(
        name="provide_living_room",
        category=FunctionCategory.SPATIAL,
        priority=0.8,
        activities=["living"]
    )
    node.add_function(func)
    
    # Add behavior WITH actual value
    behav = Behavior(
        category=BehaviorCategory.SPATIAL,
        metric_name="living_room_area",
        metric_unit="sqm",
        target_value=20.0,
        actual_value=18.0,  # 90% of target
        derived_from_function=func.function_id
    )
    behav.calculate_satisfaction()
    node.add_behavior(behav)
    
    # Add behavior WITHOUT actual value
    behav2 = Behavior(
        category=BehaviorCategory.THERMAL,
        metric_name="temperature",
        metric_unit="C",
        target_value=22.0,
        actual_value=None,  # Not calculated yet
        derived_from_function=func.function_id
    )
    node.add_behavior(behav2)
    
    # Add layout
    layout = Layout()
    room = Room(
        name="Living Room",
        room_type="living_room",
        function_id=func.function_id,
        area=18.0,
        height=3.0
    )
    room.calculate_volume()
    layout.rooms[room.room_id] = room
    layout.total_area = 18.0
    layout.used_area = 18.0
    layout.calculate_metrics()
    node.layout = layout
    
    print("\n📊 Node State:")
    print(f"   Functions: {len(node.functions)}")
    print(f"   Behaviors: {len(node.behaviors)}")
    print(f"   Structures: {len(node.structures)}")
    print(f"   Has Layout: {node.layout is not None}")
    
    # Check behaviors
    print("\n📈 Behavior Analysis:")
    for b_id, b in node.behaviors.items():
        print(f"   {b.metric_name}:")
        print(f"      Target: {b.target_value}")
        print(f"      Actual: {b.actual_value}")
        print(f"      Satisfied: {b.is_satisfied}")
        if b.target_value and b.actual_value:
            ratio = b.actual_value / b.target_value
            print(f"      Ratio: {ratio:.3f}")
        else:
            print(f"      ⚠️  Missing actual value!")
    
    # Score the node
    scoring = ScoringAgent()
    result = await scoring.score_node(node)
    
    print("\n" + "=" * 80)
    print("📊 Scoring Results")
    print("=" * 80)
    
    scores = result['scores']
    details = result['details']
    
    print(f"\nFunctional Adequacy: {scores['functional_adequacy']:.3f}")
    print(f"   Details: {details['functional']}")
    
    print(f"\nBehavioral Performance: {scores['behavioral_performance']:.3f}")
    print(f"   Details: {details['behavioral']}")
    
    print(f"\nStructural Feasibility: {scores['structural_feasibility']:.3f}")
    print(f"   Details: {details['structural']}")
    
    print(f"\nLayout Efficiency: {scores['layout_efficiency']:.3f}")
    print(f"   Details: {details['layout']}")
    
    print(f"\nSustainability: {scores['sustainability']:.3f}")
    print(f"   Details: {details['sustainability']}")
    
    print(f"\n🎯 Composite Score: {scores['composite']:.3f}")
    
    # Analyze why score is low
    print("\n" + "=" * 80)
    print("🔍 Why Score Might Be Low")
    print("=" * 80)
    
    issues = []
    
    if scores['functional_adequacy'] < 0.5:
        issues.append("❌ Functional score is low - behaviors may not have actual values")
    
    if scores['behavioral_performance'] < 0.5:
        issues.append("❌ Behavioral score is low - geometric mean penalizes missing values")
        if any(b.actual_value is None for b in node.behaviors.values()):
            issues.append("   → Some behaviors have no actual_value (defaults to 0.5)")
    
    if scores['structural_feasibility'] < 0.5:
        issues.append("❌ Structural score is low - no structures defined")
    
    if scores['layout_efficiency'] < 0.5:
        issues.append("❌ Layout score is low - layout metrics may not be calculated")
    
    if scores['sustainability'] < 0.5:
        issues.append("⚠️  Sustainability score is low - defaults to 0.5")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No obvious issues found")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 Recommendations")
    print("=" * 80)
    print("""
1. Ensure behaviors have actual values calculated (S → Bs transformation)
2. Make sure structures are defined for structural scoring
3. Calculate layout metrics before scoring
4. Consider using rho > 1 for more compensatory scoring (allows trade-offs)
5. Check if behavior calculator is running in convergence loop
    """)


if __name__ == "__main__":
    asyncio.run(diagnose_scoring())

