"""Quick import smoke test"""
import traceback
import sys

modules = [
    ("backend.core.fbsl_models", "FBSLLayoutNode"),
    ("backend.core.graph_of_thoughts", "GraphOfThoughtsEngine"),
    ("backend.core.behavior_calculator", "BehaviorCalculator"),
    ("backend.core.spatial_algorithms", "AStarPathfinder"),
    ("backend.agents.encoder_agent", "EncoderAgent"),
    ("backend.agents.generalizer_agent", "GeneralizerAgent"),
    ("backend.agents.research_agent", "ResearchAgent"),
    ("backend.agents.scoring_agent", "ScoringAgent"),
    ("backend.agents.refinement_agent", "RefinementAgent"),
    ("backend.agents.layout_agent", "LayoutGenerationAgent"),
    ("backend.visualization.enhanced_layout", "EnhancedLayoutVisualizer"),
    ("backend.visualization.improved_layout_visualizer", "ImprovedLayoutVisualizer"),
    ("backend.pipeline.orchestrator", "PipelineOrchestrator"),
]

results = []
for mod_path, class_name in modules:
    try:
        mod = __import__(mod_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        results.append((mod_path, class_name, "OK"))
        print(f"  OK  {mod_path}.{class_name}")
    except Exception as e:
        results.append((mod_path, class_name, str(e)))
        print(f"  FAIL {mod_path}.{class_name}")
        traceback.print_exc()
        print()

passed = sum(1 for r in results if r[2] == "OK")
total = len(results)
print(f"\n{passed}/{total} imports succeeded")
if passed < total:
    sys.exit(1)
