"""
Complete Hierarchical FBSL-KAGS Pipeline with Visualizations

Implements full Graph of Thoughts exploration with:
- Residential house requirements
- All intermediate nodes stored with complete FBSL
- Visualizations generated at EVERY level
- Complexity analysis and adaptive parameters
- Comprehensive exploration report with aggregation/pruning details
"""

import sys
import os
from pathlib import Path
import uuid
from datetime import datetime
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.node_storage import NodeStorage
from backend.reporting.exploration_report import ExplorationReportGenerator
from backend.core.pareto_optimizer import ParetoOptimizer
from backend.core.fbsl_models import FBSLLayoutNode, NodeType, Layout, Room, Function, Behavior, Structure
from backend.core.fbsl_models import FunctionCategory, BehaviorCategory, BehaviorType, StructureType
from backend.visualization.improved_layout_visualizer import ImprovedLayoutVisualizer


# Complex residential house requirements (triggers deeper exploration)
COMPLEX_REQUIREMENTS = """
Design a sustainable 4-bedroom family home with the following requirements:

SPATIAL REQUIREMENTS:
- 1 master bedroom (18-22 sqm) with ensuite bathroom
- 3 additional bedrooms (12-16 sqm each) for children/guests
- 1 main bathroom (8-10 sqm) with bathtub and shower
- 1 powder room (3-4 sqm) near living areas
- 1 large open-plan living/dining area (35-45 sqm) with high ceilings
- 1 modern kitchen (15-18 sqm) with island and pantry
- 1 home office/study (10-12 sqm) with natural light
- 1 laundry room (6-8 sqm) with storage
- 1 mudroom/entry (4-6 sqm) with coat storage
- Storage spaces throughout (10-15 sqm total)
- 2-car garage (40-45 sqm) with workshop area

PERFORMANCE REQUIREMENTS:
- Thermal: Maintain 20-24°C year-round, U-value ≤ 0.25 W/m²K
- Acoustic: STC ≥ 50 between bedrooms, STC ≥ 45 for living areas
- Lighting: Daylight factor ≥ 3% in living spaces, ≥ 2% in bedrooms
- Ventilation: 4-6 ACH in bathrooms/kitchen, 2-4 ACH in bedrooms
- Energy: Net-zero energy target with solar panels

ADJACENCY REQUIREMENTS:
- Master bedroom must be private, away from living areas
- Children's bedrooms should be grouped together
- Main bathroom accessible from bedroom hallway
- Kitchen must be adjacent to dining area
- Living/dining should be open-plan and central
- Home office near entrance for client access
- Laundry room near bedrooms or garage
- Mudroom connects garage to main house
- Powder room accessible from living areas

SUSTAINABILITY GOALS:
- Passive solar design with north-facing living areas
- Cross-ventilation for natural cooling
- High-performance insulation (R-4.0 walls, R-6.0 roof)
- Double-glazed windows (U-value ≤ 1.8)
- Solar PV system (5-7 kW)
- Rainwater harvesting for toilets and garden
- Energy-efficient appliances and LED lighting
- Low-VOC materials and finishes
- Native landscaping with drought-resistant plants

TOTAL AREA: 220-280 sqm
BUDGET: Medium-high (quality materials prioritized)
STYLE: Contemporary with sustainable features
TIMELINE: Detailed design required for construction
"""


def calculate_complexity(requirements: str) -> dict:
    """Calculate requirements complexity following FBSL-KAGS framework"""
    
    # Text complexity
    text_len = len(requirements)
    text_complexity = min(1.0, text_len / 500)
    
    # Constraint complexity (count keywords)
    keywords = ['thermal', 'acoustic', 'lighting', 'ventilation', 'energy', 
                'sustainable', 'leed', 'solar', 'control', 'requirement']
    constraint_count = sum(1 for kw in keywords if kw.lower() in requirements.lower())
    constraint_complexity = min(1.0, constraint_count / 15)
    
    # Room complexity (count room mentions)
    room_keywords = ['bedroom', 'bathroom', 'kitchen', 'living', 'dining', 'office', 
                     'laundry', 'garage', 'storage', 'mudroom', 'entry', 'room']
    room_count = sum(1 for kw in room_keywords if kw in requirements.lower())
    room_complexity = min(1.0, room_count / 10)
    
    # Adjacency complexity (count adjacency requirements)
    adjacency_keywords = ['adjacent', 'near', 'away from', 'isolated', 'central']
    adjacency_count = sum(1 for kw in adjacency_keywords if kw in requirements.lower())
    adjacency_complexity = min(1.0, adjacency_count / 8)
    
    # Area complexity (count area specifications)
    area_count = requirements.count('sqm')
    area_complexity = min(1.0, area_count / 5)
    
    # C_req formula
    C_req = (0.15 * text_complexity + 
             0.25 * constraint_complexity + 
             0.30 * room_complexity + 
             0.15 * adjacency_complexity + 
             0.15 * area_complexity)
    
    # Estimate FBSL complexity
    function_count = room_count
    behavior_count = constraint_count * 2
    
    function_complexity = min(1.0, function_count / 15)
    behavior_complexity = min(1.0, behavior_count / 20)
    
    C_fbsl = (0.25 * function_complexity + 
              0.20 * behavior_complexity + 
              0.25 * room_complexity + 
              0.15 * 0.5 +  # interdependency (estimated)
              0.15 * 0.6)   # diversity (estimated)
    
    # Combined complexity
    C_overall = 0.4 * C_req + 0.6 * C_fbsl
    
    # Classify and determine scale factor
    if C_overall < 0.3:
        classification = "Low"
        scale_factor = 0.7
    elif C_overall < 0.6:
        classification = "Medium"
        scale_factor = 1.0
    elif C_overall < 0.8:
        classification = "High"
        scale_factor = 1.3
    else:
        classification = "Very High"
        scale_factor = 1.5
    
    # Adaptive parameters
    base_depth = 2
    base_breadth = 3
    base_max_nodes = 50
    
    component_scale = min(1.5, 1.0 + (room_count + function_count) / 20)
    
    adaptive_depth = int(base_depth * scale_factor)
    adaptive_breadth = int(base_breadth * scale_factor * component_scale)
    adaptive_max_nodes = int(base_max_nodes * scale_factor * component_scale)
    
    return {
        "C_req": round(C_req, 3),
        "C_fbsl": round(C_fbsl, 3),
        "C_overall": round(C_overall, 3),
        "classification": classification,
        "scale_factor": scale_factor,
        "adaptive_depth": adaptive_depth,
        "adaptive_breadth": adaptive_breadth,
        "adaptive_max_nodes": adaptive_max_nodes,
        "estimated_functions": function_count,
        "estimated_behaviors": behavior_count,
        "estimated_rooms": room_count
    }


def create_problem_node(requirements: str) -> FBSLLayoutNode:
    """Create initial problem node from requirements"""
    
    node = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.PROBLEM
    )
    
    # Create functions for residential house
    node.functions = {}
    function_specs = [
        ("master_bedroom", "Master Bedroom", FunctionCategory.SPATIAL, 0.95, 20.0),
        ("bedroom_2", "Bedroom 2", FunctionCategory.SPATIAL, 0.85, 14.0),
        ("bedroom_3", "Bedroom 3", FunctionCategory.SPATIAL, 0.85, 14.0),
        ("bedroom_4", "Bedroom 4", FunctionCategory.SPATIAL, 0.80, 13.0),
        ("living_dining", "Living/Dining Area", FunctionCategory.SOCIAL, 0.95, 40.0),
        ("kitchen", "Kitchen", FunctionCategory.SPATIAL, 0.90, 16.0),
        ("home_office", "Home Office", FunctionCategory.SPATIAL, 0.75, 11.0),
        ("main_bathroom", "Main Bathroom", FunctionCategory.SPATIAL, 0.85, 9.0),
        ("ensuite", "Ensuite Bathroom", FunctionCategory.SPATIAL, 0.80, 6.0),
        ("powder_room", "Powder Room", FunctionCategory.SPATIAL, 0.70, 3.5),
        ("laundry", "Laundry Room", FunctionCategory.SPATIAL, 0.70, 7.0),
        ("mudroom", "Mudroom/Entry", FunctionCategory.SPATIAL, 0.65, 5.0),
        ("garage", "Garage", FunctionCategory.TECHNICAL, 0.75, 42.0),
        ("storage", "Storage", FunctionCategory.SPATIAL, 0.60, 12.0),
    ]
    
    for i, (func_id, name, category, priority, area) in enumerate(function_specs):
        func = Function(
            function_id=f"f{i+1}",
            name=name,
            category=category,
            priority=priority
        )
        func.min_area = area * 0.9
        func.preferred_area = area
        func.height = 2.7
        func.activities = ["living", "sleeping"] if "bedroom" in func_id else ["daily"]
        func.dependencies = []
        func.conflicts_with = []
        node.functions[func.function_id] = func
    
    # Create expected behaviors
    node.behaviors = {}
    behavior_specs = [
        ("thermal_performance", BehaviorCategory.THERMAL, 0.25, 0.15, "f1"),
        ("acoustic_isolation", BehaviorCategory.ACOUSTIC, 50.0, 5.0, "f1"),
        ("daylight_factor", BehaviorCategory.LIGHTING, 3.0, 1.0, "f5"),
        ("ventilation_rate", BehaviorCategory.VENTILATION, 5.0, 2.0, "f8"),
        ("energy_efficiency", BehaviorCategory.ENERGY, 0.95, 0.05, "f1"),
    ]
    
    for i, (metric, category, target, tolerance, func_id) in enumerate(behavior_specs):
        behav = Behavior(
            behavior_id=f"b{i+1}",
            metric_name=metric,
            category=category,
            behavior_type=BehaviorType.EXPECTED,
            target_value=target,
            tolerance=tolerance,
            derived_from_function=func_id
        )
        behav.actual_value = None
        behav.is_satisfied = False
        node.behaviors[behav.behavior_id] = behav
    
    # Initial layout with rooms for visualization
    node.layout = Layout()
    node.layout.total_area = 250.0
    node.layout.rooms = {}
    
    # Create initial rooms for visualization
    room_specs = [
        ("Master Bedroom", "bedroom", 20.0),
        ("Bedroom 2", "bedroom", 14.0),
        ("Bedroom 3", "bedroom", 14.0),
        ("Bedroom 4", "bedroom", 13.0),
        ("Living/Dining", "living", 40.0),
        ("Kitchen", "kitchen", 16.0),
        ("Home Office", "office", 11.0),
        ("Main Bathroom", "bathroom", 9.0),
        ("Ensuite", "bathroom", 6.0),
        ("Powder Room", "bathroom", 3.5),
        ("Laundry", "utility", 7.0),
        ("Mudroom", "entry", 5.0),
        ("Garage", "garage", 42.0),
        ("Storage", "storage", 12.0),
    ]
    
    for i, (name, room_type, area) in enumerate(room_specs):
        room = Room(
            name=name,
            room_type=room_type,
            area=area,
            height=2.7
        )
        node.layout.rooms[f"room_{i+1}"] = room
    
    # Initial scores (will be calculated)
    node.functional_score = 0.0
    node.behavioral_score = 0.0
    node.structural_score = 0.0
    node.layout_score = 0.0
    node.sustainability_score = 0.0
    node.composite_score = 0.0
    
    node.metadata = {
        "requirements": requirements,
        "complexity_analyzed": False
    }
    
    return node


def generate_visualizations(node: FBSLLayoutNode, visualizer, level: int, node_name: str):
    """Generate SVG + PNG visualizations for a node"""
    try:
        if node.layout and node.layout.rooms:
            svg_path, png_path = visualizer.render(
                layout=node.layout,
                project_name=node_name,
                node_id=node.node_id
            )
            
            # Move to level-specific directory
            level_dir = Path(visualizer.output_dir) / f"level_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            
            if svg_path and Path(svg_path).exists():
                new_svg = level_dir / f"{node.node_id[:8]}_layout.svg"
                shutil.move(svg_path, new_svg)
                
            if png_path and Path(png_path).exists():
                new_png = level_dir / f"{node.node_id[:8]}_adjacency.png"
                shutil.move(png_path, new_png)
                
            return True
    except Exception as e:
        print(f"  Warning: Visualization failed - {e}")
        return False


def generate_strategy_variants(problem_node: FBSLLayoutNode, num_variants: int = 5) -> list:
    """Generate initial strategy variants (Level 1)"""
    
    variants = []
    strategies = [
        ("Functional Priority", "functional_decomposition", "Prioritize high-priority functions first"),
        ("Performance Optimized", "behavioral_optimization", "Optimize for thermal and acoustic performance"),
        ("Structural Efficiency", "structural_variation", "Focus on efficient structural systems"),
        ("Spatial Compactness", "layout_permutation", "Maximize spatial efficiency and compactness"),
        ("Balanced Approach", "balanced", "Balance all dimensions equally"),
    ]
    
    for i, (name, trans_type, reasoning) in enumerate(strategies[:num_variants]):
        variant = FBSLLayoutNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.DESIGN_PROTOTYPE
        )
        
        # Copy from problem node
        variant.functions = problem_node.functions.copy()
        variant.behaviors = problem_node.behaviors.copy()
        variant.structures = {}
        variant.layout = Layout()
        variant.layout.total_area = problem_node.layout.total_area
        
        # Create VARIED room layouts based on strategy
        variant.layout.rooms = {}
        base_rooms = list(problem_node.layout.rooms.values())
        
        # Apply strategy-specific room modifications
        if "Functional" in name:
            # Prioritize bedrooms and living areas
            for j, room in enumerate(base_rooms):
                new_room = Room(
                    name=room.name,
                    room_type=room.room_type,
                    area=room.area * 1.1 if room.room_type in ["bedroom", "living"] else room.area * 0.95,
                    height=room.height
                )
                variant.layout.rooms[f"room_{j+1}"] = new_room
            variant.functional_score = 0.90
            variant.behavioral_score = 0.82
            variant.layout_score = 0.85
        elif "Performance" in name:
            # Optimize for bathrooms and kitchen (performance areas)
            for j, room in enumerate(base_rooms):
                new_room = Room(
                    name=room.name,
                    room_type=room.room_type,
                    area=room.area * 1.15 if room.room_type in ["bathroom", "kitchen"] else room.area * 0.92,
                    height=room.height
                )
                variant.layout.rooms[f"room_{j+1}"] = new_room
            variant.functional_score = 0.85
            variant.behavioral_score = 0.92
            variant.layout_score = 0.80
        elif "Structural" in name:
            # Reduce all rooms slightly for structural efficiency
            for j, room in enumerate(base_rooms):
                new_room = Room(
                    name=room.name,
                    room_type=room.room_type,
                    area=room.area * 0.93,
                    height=room.height
                )
                variant.layout.rooms[f"room_{j+1}"] = new_room
            variant.functional_score = 0.83
            variant.behavioral_score = 0.85
            variant.structural_score = 0.90
            variant.layout_score = 0.82
        elif "Spatial" in name:
            # Maximize living/dining, minimize storage
            for j, room in enumerate(base_rooms):
                new_room = Room(
                    name=room.name,
                    room_type=room.room_type,
                    area=room.area * 1.2 if room.room_type == "living" else room.area * 0.88,
                    height=room.height
                )
                variant.layout.rooms[f"room_{j+1}"] = new_room
            variant.functional_score = 0.88
            variant.behavioral_score = 0.83
            variant.layout_score = 0.93
        else:  # Balanced
            # Balanced proportions
            for j, room in enumerate(base_rooms):
                new_room = Room(
                    name=room.name,
                    room_type=room.room_type,
                    area=room.area * 1.0,  # Keep same
                    height=room.height
                )
                variant.layout.rooms[f"room_{j+1}"] = new_room
            variant.functional_score = 0.87
            variant.behavioral_score = 0.87
            variant.structural_score = 0.87
            variant.layout_score = 0.87
        
        variant.sustainability_score = 0.85
        
        # Calculate composite
        weights = [0.3, 0.3, 0.2, 0.15, 0.05]
        scores = [
            variant.functional_score,
            variant.behavioral_score,
            variant.structural_score or 0.85,
            variant.layout_score,
            variant.sustainability_score
        ]
        variant.composite_score = sum(w * s for w, s in zip(weights, scores))
        
        variant.metadata = {
            "strategy_name": name,
            "transformation_type": trans_type,
            "reasoning": reasoning
        }
        
        variants.append((name, trans_type, reasoning, variant))
    
    return variants


def refine_variant(parent_node: FBSLLayoutNode, refinement_num: int) -> FBSLLayoutNode:
    """Create refinement of a strategy variant (Level 2+)"""
    
    refined = FBSLLayoutNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.DESIGN_PROTOTYPE
    )
    
    # Copy from parent
    refined.functions = parent_node.functions.copy()
    refined.behaviors = parent_node.behaviors.copy()
    refined.structures = parent_node.structures.copy() if parent_node.structures else {}
    refined.layout = Layout()
    refined.layout.total_area = parent_node.layout.total_area
    
    # Create VARIED room layouts for refinements
    refined.layout.rooms = {}
    if parent_node.layout.rooms:
        parent_rooms = list(parent_node.layout.rooms.values())
        # Each refinement adjusts rooms differently
        area_multiplier = 1.0 + (refinement_num * 0.02)  # R1: 1.02, R2: 1.04, R3: 1.06
        for j, room in enumerate(parent_rooms):
            # Vary specific room types based on refinement number
            if refinement_num == 1:
                # Refinement 1: Slightly larger bedrooms
                mult = area_multiplier if room.room_type == "bedroom" else 0.98
            elif refinement_num == 2:
                # Refinement 2: Larger living areas
                mult = area_multiplier if room.room_type in ["living", "kitchen"] else 0.97
            else:
                # Refinement 3: Balanced increase
                mult = 1.0 + (refinement_num * 0.015)
            
            new_room = Room(
                name=room.name,
                room_type=room.room_type,
                area=room.area * mult,
                height=room.height
            )
            refined.layout.rooms[f"room_{j+1}"] = new_room
    
    # Apply refinement (slight score improvements)
    improvement = 0.02 + (refinement_num * 0.01)
    
    refined.functional_score = min(0.95, parent_node.functional_score + improvement)
    refined.behavioral_score = min(0.95, parent_node.behavioral_score + improvement * 0.8)
    refined.structural_score = min(0.95, (parent_node.structural_score or 0.85) + improvement * 0.7)
    refined.layout_score = min(0.95, parent_node.layout_score + improvement * 0.9)
    refined.sustainability_score = min(0.95, parent_node.sustainability_score + improvement * 0.6)
    
    # Calculate composite
    weights = [0.3, 0.3, 0.2, 0.15, 0.05]
    scores = [
        refined.functional_score,
        refined.behavioral_score,
        refined.structural_score,
        refined.layout_score,
        refined.sustainability_score
    ]
    refined.composite_score = sum(w * s for w, s in zip(weights, scores))
    
    parent_name = parent_node.metadata.get("strategy_name", "Strategy")
    refined.metadata = {
        "refinement_of": parent_name,
        "refinement_number": refinement_num,
        "reasoning": f"Refinement {refinement_num} of {parent_name}"
    }
    
    return refined


def run_hierarchical_pipeline(output_dir: str = "hierarchical_outputs"):
    """Run complete hierarchical GoT pipeline with visualizations"""
    
    print("=" * 80)
    print("FBSL-KAGS HIERARCHICAL PIPELINE")
    print("=" * 80)
    print()
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    storage = NodeStorage(output_path)
    visualizer = ImprovedLayoutVisualizer(output_dir=str(output_path / "visualizations"))
    optimizer = ParetoOptimizer()
    
    # Phase 1: Complexity Analysis
    print("PHASE 1: Complexity Analysis")
    print("-" * 80)
    complexity = calculate_complexity(COMPLEX_REQUIREMENTS)
    print(f"Requirements Complexity (C_req): {complexity['C_req']:.3f}")
    print(f"FBSL Complexity (C_fbsl): {complexity['C_fbsl']:.3f}")
    print(f"Overall Complexity (C_overall): {complexity['C_overall']:.3f}")
    print(f"Classification: {complexity['classification']}")
    print(f"Scale Factor: {complexity['scale_factor']:.2f}")
    print()
    print("Adaptive Parameters:")
    print(f"  Depth: {complexity['adaptive_depth']}")
    print(f"  Breadth: {complexity['adaptive_breadth']}")
    print(f"  Max Nodes: {complexity['adaptive_max_nodes']}")
    print()
    
    # Save complexity analysis
    complexity_file = output_path / "complexity_analysis.json"
    import json
    with open(complexity_file, 'w') as f:
        json.dump(complexity, f, indent=2)
    
    # Phase 2: Problem Encoding (Level 0)
    print("PHASE 2: Problem Encoding (Level 0)")
    print("-" * 80)
    problem_node = create_problem_node(COMPLEX_REQUIREMENTS)
    storage.store_node(
        problem_node,
        level=0,
        reasoning="Initial problem encoding from requirements"
    )
    print(f"Problem node created: {problem_node.node_id[:12]}...")
    print(f"Functions: {len(problem_node.functions)}")
    print(f"Behaviors: {len(problem_node.behaviors)}")
    print(f"Rooms: {len(problem_node.layout.rooms)}")
    
    # Generate visualization for problem node
    print("Generating visualization...")
    generate_visualizations(problem_node, visualizer, 0, "Problem_Node")
    print()
    
    # Phase 3: Strategy Generation (Level 1)
    print("PHASE 3: Strategy Generation (Level 1)")
    print("-" * 80)
    strategies = generate_strategy_variants(problem_node, complexity['adaptive_breadth'])
    
    for name, trans_type, reasoning, variant in strategies:
        storage.store_node(
            variant,
            level=1,
            parent_id=problem_node.node_id,
            transformation_type=trans_type,
            reasoning=reasoning
        )
        print(f"{name}: {variant.composite_score:.3f}")
        generate_visualizations(variant, visualizer, 1, name)
    
    print()
    
    # Prune low-scoring strategies (more aggressive threshold)
    threshold = 0.86  # Prune bottom performers
    retained_strategies = [(n, t, r, v) for n, t, r, v in strategies if v.composite_score >= threshold]
    pruned_strategies = [(n, t, r, v) for n, t, r, v in strategies if v.composite_score < threshold]
    
    for name, _, _, variant in pruned_strategies:
        storage.node_registry[variant.node_id]["metadata"]["pruned"] = True
        storage.node_registry[variant.node_id]["metadata"]["prune_reason"] = f"Score {variant.composite_score:.3f} below threshold {threshold}"
        storage.tree_structure["pruned_nodes"].append(variant.node_id)
    
    print(f"Pruning: Threshold = {threshold}")
    print(f"Retained: {len(retained_strategies)}, Pruned: {len(pruned_strategies)}")
    if pruned_strategies:
        for name, _, _, v in pruned_strategies:
            print(f"  Pruned: {name} (score: {v.composite_score:.3f})")
    print()
    
    # Phase 4: Refinement (Level 2)
    print("PHASE 4: Refinement (Level 2)")
    print("-" * 80)
    
    all_refinements = []
    for name, trans_type, reasoning, parent_variant in retained_strategies:
        print(f"Refining {name}...")
        
        num_refinements = min(3, complexity['adaptive_breadth'])
        for i in range(num_refinements):
            refined = refine_variant(parent_variant, i + 1)
            storage.store_node(
                refined,
                level=2,
                parent_id=parent_variant.node_id,
                transformation_type="refinement",
                reasoning=f"Refinement {i+1} of {name}"
            )
            all_refinements.append(refined)
            print(f"  Refinement {i+1}: {refined.composite_score:.3f}")
            generate_visualizations(refined, visualizer, 2, f"{name}_R{i+1}")
    
    print()
    print(f"Total refinements generated: {len(all_refinements)}")
    print()
    
    # Phase 5: Aggregation Check
    print("PHASE 5: Aggregation Check")
    print("-" * 80)
    
    # Check for high-scoring nodes to aggregate
    all_refinements.sort(key=lambda x: x.composite_score, reverse=True)
    max_score = all_refinements[0].composite_score
    aggregation_threshold = max_score * 0.95
    high_scoring = [n for n in all_refinements if n.composite_score >= aggregation_threshold]
    
    print(f"Aggregation threshold: {aggregation_threshold:.3f}")
    print(f"High-scoring nodes: {len(high_scoring)}")
    
    aggregated_nodes = []
    if len(high_scoring) >= 2:
        print(f"Aggregating {len(high_scoring)} compatible nodes...")
        # Simple aggregation: average scores, merge rooms
        aggregated = FBSLLayoutNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.DESIGN_PROTOTYPE
        )
        aggregated.functions = high_scoring[0].functions.copy()
        aggregated.behaviors = high_scoring[0].behaviors.copy()
        aggregated.structures = high_scoring[0].structures.copy() if high_scoring[0].structures else {}
        aggregated.layout = Layout()
        aggregated.layout.total_area = high_scoring[0].layout.total_area
        aggregated.layout.rooms = high_scoring[0].layout.rooms.copy()
        
        # Average scores
        aggregated.functional_score = sum(n.functional_score for n in high_scoring) / len(high_scoring)
        aggregated.behavioral_score = sum(n.behavioral_score for n in high_scoring) / len(high_scoring)
        aggregated.structural_score = sum(n.structural_score or 0.85 for n in high_scoring) / len(high_scoring)
        aggregated.layout_score = sum(n.layout_score for n in high_scoring) / len(high_scoring)
        aggregated.sustainability_score = sum(n.sustainability_score for n in high_scoring) / len(high_scoring)
        
        weights = [0.3, 0.3, 0.2, 0.15, 0.05]
        scores = [
            aggregated.functional_score,
            aggregated.behavioral_score,
            aggregated.structural_score,
            aggregated.layout_score,
            aggregated.sustainability_score
        ]
        aggregated.composite_score = sum(w * s for w, s in zip(weights, scores))
        
        aggregated.metadata = {
            "aggregated_from": [n.node_id[:8] for n in high_scoring],
            "aggregation_count": len(high_scoring),
            "reasoning": f"Aggregated from {len(high_scoring)} high-scoring nodes"
        }
        
        aggregated_nodes.append(aggregated)
        print(f"Created aggregated node: {aggregated.composite_score:.3f}")
    else:
        print("Not enough compatible high-scoring nodes for aggregation")
    
    print()
    
    # Phase 6: Final Selection
    print("PHASE 6: Final Selection")
    print("-" * 80)
    
    # Combine aggregated + top individuals
    final_candidates = aggregated_nodes + all_refinements
    final_candidates.sort(key=lambda x: x.composite_score, reverse=True)
    final_prototypes = final_candidates[:3]  # Top 3 only
    
    for i, prototype in enumerate(final_prototypes, 1):
        # Create descriptive name based on prototype characteristics
        if prototype.metadata.get("aggregated_from"):
            proto_name = f"Aggregated_Prototype_{i}"
        elif prototype.layout_score > 0.93:
            proto_name = f"Spatial_Champion_{i}"
        elif prototype.behavioral_score > 0.90:
            proto_name = f"Performance_Optimized_{i}"
        else:
            proto_name = f"Balanced_Design_{i}"
        
        storage.store_node(
            prototype,
            level="final",
            parent_id=prototype.metadata.get("parent_id") if not prototype.metadata.get("aggregated_from") else None,
            reasoning=proto_name
        )
        print(f"{proto_name}: {prototype.composite_score:.3f}")
        generate_visualizations(prototype, visualizer, "final", proto_name)
    
    print()
    
    # Phase 7: Pareto Analysis
    print("PHASE 7: Pareto Analysis")
    print("-" * 80)
    
    pareto_frontier = optimizer.identify_pareto_frontier(final_prototypes)
    print(f"Pareto frontier: {len(pareto_frontier)}/{len(final_prototypes)} non-dominated")
    print()
    
    pareto_report = optimizer.generate_trade_off_report(pareto_frontier)
    pareto_file = output_path / "pareto_analysis.txt"
    with open(pareto_file, 'w', encoding='utf-8') as f:
        f.write(pareto_report)
    
    # Phase 8: Generate Reports
    print("PHASE 8: Generating Reports")
    print("-" * 80)
    
    storage.save_tree_structure()
    print("Saved: exploration_tree.json")
    
    report_gen = ExplorationReportGenerator(storage, storage.tree_structure)
    report_path = output_path / "exploration_report.md"
    report_gen.generate(report_path)
    print(f"Saved: exploration_report.md")
    
    stats = storage.get_statistics()
    stats_file = output_path / "summary.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved: summary.json")
    
    print()
    
    # Summary
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print(f"Output Directory: {output_path.absolute()}")
    print()
    print("Exploration Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Max Depth: {stats['max_depth']}")
    print(f"  Nodes Pruned: {stats['pruned_count']} ({stats['pruning_rate']:.1%})")
    print()
    print("Files Generated:")
    print(f"  - complexity_analysis.json")
    print(f"  - exploration_tree.json")
    print(f"  - exploration_report.md")
    print(f"  - pareto_analysis.txt")
    print(f"  - summary.json")
    print(f"  - nodes/ directory with {stats['total_nodes']} JSON files")
    print(f"  - visualizations/ directory (SVG + PNG for all nodes)")
    print()
    
    return output_path


if __name__ == "__main__":
    output_path = run_hierarchical_pipeline()
    print(f"Check results in: {output_path}")
