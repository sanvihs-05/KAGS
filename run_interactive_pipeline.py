"""
Interactive FBSL-KAGS Pipeline

Takes user requirements as input and generates complete design prototypes
with visual outputs (SVG floor plans and PNG adjacency graphs).
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.pareto_optimizer import ParetoOptimizer
from backend.core.fbsl_models import FBSLLayoutNode, NodeType, Layout, Room
from backend.visualization.improved_layout_visualizer import ImprovedLayoutVisualizer
import uuid


def get_user_requirements():
    """Get design requirements from user"""
    
    print("=" * 80)
    print("FBSL-KAGS INTERACTIVE PIPELINE")
    print("=" * 80)
    print()
    print("Please provide your design requirements:")
    print()
    
    # Project name
    project_name = input("Project Name (e.g., 'Modern Family Home'): ").strip()
    if not project_name:
        project_name = "Custom_Project"
    
    # Number of prototypes
    while True:
        try:
            num_prototypes = input("Number of design alternatives to generate (2-5): ").strip()
            num_prototypes = int(num_prototypes) if num_prototypes else 3
            if 2 <= num_prototypes <= 5:
                break
            print("Please enter a number between 2 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Total area
    while True:
        try:
            total_area = input("Total area in sqm (e.g., 200): ").strip()
            total_area = float(total_area) if total_area else 200.0
            if total_area > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Number of rooms
    while True:
        try:
            num_rooms = input("Number of rooms (3-10): ").strip()
            num_rooms = int(num_rooms) if num_rooms else 4
            if 3 <= num_rooms <= 10:
                break
            print("Please enter a number between 3 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    # Design priorities
    print()
    print("Design Priorities (rate 1-5, where 5 is highest priority):")
    priorities = {}
    
    for dimension in ["Functionality", "Performance", "Structure", "Layout", "Sustainability"]:
        while True:
            try:
                priority = input(f"  {dimension} (1-5): ").strip()
                priority = int(priority) if priority else 3
                if 1 <= priority <= 5:
                    priorities[dimension.lower()] = priority / 5.0  # Normalize to 0-1
                    break
                print("  Please enter a number between 1 and 5")
            except ValueError:
                print("  Please enter a valid number")
    
    print()
    return {
        "project_name": project_name,
        "num_prototypes": num_prototypes,
        "total_area": total_area,
        "num_rooms": num_rooms,
        "priorities": priorities
    }


def generate_prototypes_from_requirements(requirements):
    """Generate design prototypes based on user requirements"""
    
    prototypes = []
    num_prototypes = requirements["num_prototypes"]
    total_area = requirements["total_area"]
    num_rooms = requirements["num_rooms"]
    priorities = requirements["priorities"]
    
    # Design strategies
    strategies = [
        ("Compact_Efficient", {"layout": 0.95, "sustainability": 0.80}),
        ("Sustainable_Focus", {"sustainability": 0.95, "performance": 0.90}),
        ("Functional_Priority", {"functionality": 0.95, "layout": 0.85}),
        ("Balanced_Design", {"functionality": 0.88, "performance": 0.88, "layout": 0.88}),
        ("Structural_Robust", {"structure": 0.92, "performance": 0.88}),
    ]
    
    for i in range(num_prototypes):
        strategy_name, strategy_scores = strategies[i]
        
        # Create prototype
        prototype = FBSLLayoutNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.DESIGN_PROTOTYPE
        )
        
        # Calculate scores based on strategy and user priorities
        base_scores = {
            "functional_score": 0.85,
            "behavioral_score": 0.85,
            "structural_score": 0.85,
            "layout_score": 0.85,
            "sustainability_score": 0.85
        }
        
        # Apply strategy modifiers
        for key, value in strategy_scores.items():
            score_key = f"{key}_score" if key != "performance" else "behavioral_score"
            if score_key in base_scores:
                base_scores[score_key] = value
        
        # Apply user priorities (slight adjustment)
        for key, priority in priorities.items():
            score_key = f"{key}_score" if key != "performance" else "behavioral_score"
            if score_key in base_scores:
                base_scores[score_key] = base_scores[score_key] * 0.7 + priority * 0.3
        
        # Set scores
        prototype.functional_score = base_scores["functional_score"]
        prototype.behavioral_score = base_scores["behavioral_score"]
        prototype.structural_score = base_scores["structural_score"]
        prototype.layout_score = base_scores["layout_score"]
        prototype.sustainability_score = base_scores["sustainability_score"]
        
        # Calculate composite score
        weights = [0.3, 0.3, 0.2, 0.15, 0.05]
        scores = [
            prototype.functional_score,
            prototype.behavioral_score,
            prototype.structural_score,
            prototype.layout_score,
            prototype.sustainability_score
        ]
        prototype.composite_score = sum(w * s for w, s in zip(weights, scores))
        
        # Create layout
        prototype.layout = Layout()
        prototype.layout.total_area = total_area * (0.9 + i * 0.05)  # Slight variation
        
        # Generate rooms
        room_types = ["living", "kitchen", "bedroom", "bathroom", "office", "dining", "storage"]
        rooms = {}
        
        area_per_room = prototype.layout.total_area / num_rooms
        
        for j in range(num_rooms):
            room_type = room_types[j % len(room_types)]
            room_name = f"{room_type.title()} {j+1}" if j >= len(room_types) else room_type.title()
            
            room = Room(
                name=room_name,
                room_type=room_type,
                area=area_per_room * (0.8 + (j % 3) * 0.2),  # Variation
                height=2.7
            )
            rooms[f"room_{j+1}"] = room
        
        prototype.layout.rooms = rooms
        
        prototypes.append((strategy_name, prototype))
    
    return prototypes


def run_interactive_pipeline():
    """Run complete interactive pipeline"""
    
    # Step 1: Get user input
    requirements = get_user_requirements()
    
    project_name = requirements["project_name"].replace(" ", "_")
    
    print("=" * 80)
    print("GENERATING DESIGN PROTOTYPES...")
    print("=" * 80)
    print()
    
    # Step 2: Generate prototypes
    print("STEP 1: Generating Prototypes")
    print("-" * 80)
    prototypes = generate_prototypes_from_requirements(requirements)
    print(f"✓ Generated {len(prototypes)} design alternatives")
    print()
    
    # Step 3: Pareto analysis
    print("STEP 2: Pareto Analysis")
    print("-" * 80)
    optimizer = ParetoOptimizer()
    solutions = [p[1] for p in prototypes]
    pareto_frontier = optimizer.identify_pareto_frontier(solutions)
    print(f"✓ Identified {len(pareto_frontier)} non-dominated solutions")
    print()
    
    # Step 4: Trade-off analysis
    print("STEP 3: Trade-off Analysis")
    print("-" * 80)
    for name, prototype in prototypes:
        analysis = optimizer.characterize_trade_offs(prototype, pareto_frontier)
        print(f"{name}: {analysis['composite_score']:.3f}")
        if analysis['champions']:
            print(f"  Champions: {', '.join(analysis['champions'])}")
    print()
    
    # Step 5: Create output directory
    output_path = Path("pipeline_outputs") / project_name
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path = output_path / "visualizations"
    viz_path.mkdir(exist_ok=True)
    
    # Step 6: Generate visualizations
    print("STEP 4: Generating Visualizations")
    print("-" * 80)
    
    visualizer = ImprovedLayoutVisualizer(output_dir=str(viz_path))
    
    for name, prototype in prototypes:
        try:
            # Generate floor plan and adjacency graph
            svg_path, png_path = visualizer.render(
                layout=prototype.layout,
                project_name=name,
                node_id=prototype.node_id
            )
            print(f"✓ {name}:")
            print(f"  - Floor plan: {Path(svg_path).name if svg_path else 'N/A'}")
            print(f"  - Adjacency graph: {Path(png_path).name if png_path else 'N/A'}")
        except Exception as e:
            print(f"✗ {name}: Visualization failed - {e}")
    
    print()
    
    # Step 7: Save results
    print("STEP 5: Saving Results")
    print("-" * 80)
    
    # Save Pareto report
    report = optimizer.generate_trade_off_report(pareto_frontier)
    report_path = output_path / "pareto_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: pareto_analysis.txt")
    
    # Save prototype data
    for i, (name, prototype) in enumerate(prototypes, 1):
        data = {
            "name": name,
            "scores": {
                "functional": round(prototype.functional_score, 3),
                "behavioral": round(prototype.behavioral_score, 3),
                "structural": round(prototype.structural_score, 3),
                "layout": round(prototype.layout_score, 3),
                "sustainability": round(prototype.sustainability_score, 3),
                "composite": round(prototype.composite_score, 3)
            },
            "layout": {
                "total_area": round(prototype.layout.total_area, 1),
                "rooms": [
                    {
                        "name": r.name,
                        "type": r.room_type,
                        "area": round(r.area, 1)
                    }
                    for r in prototype.layout.rooms.values()
                ]
            }
        }
        
        data_path = output_path / f"prototype_{i}.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: prototype_{i}.json")
    
    # Save requirements
    req_path = output_path / "requirements.json"
    with open(req_path, 'w', encoding='utf-8') as f:
        json.dump(requirements, f, indent=2)
    print(f"Saved: requirements.json")
    
    print()
    
    # Summary
    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print(f"Project: {requirements['project_name']}")
    print(f"Output Location: {output_path.absolute()}")
    print()
    print("Generated Files:")
    print(f"  - {len(prototypes)} SVG floor plans")
    print(f"  - {len(prototypes)} PNG adjacency graphs")
    print(f"  - {len(prototypes)} prototype JSON files")
    print(f"  - 1 Pareto analysis report")
    print(f"  - 1 requirements file")
    print()
    print(f"Total: {len(prototypes) * 4 + 2} files")
    print()
    print(f"Check visualizations in: {viz_path}")
    print()
    
    return output_path


if __name__ == "__main__":
    try:
        output_path = run_interactive_pipeline()
        print("=" * 80)
        print("SUCCESS!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
