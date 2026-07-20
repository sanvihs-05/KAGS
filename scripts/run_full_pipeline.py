"""Run full FBSL-KAGS pipeline from the command line with comprehensive reporting.

This script will:
- Take `requirements` (and optional project_name) from CLI
- Run the full pipeline via `PipelineOrchestrator`
- Store prototypes/evaluations to Postgres (if available)
- Generate comprehensive FBSL reports (MD, HTML, JSON) for each prototype
- Save all artifacts (reports, SVGs, adjacency graphs) in organized directories
- Generate project summary report
- Print a summary of the top prototypes

Usage:
  py scripts\\run_full_pipeline.py --requirements "Compact 2BR apartment with daylight"

Optional flags:
  --got_delta 0.01 --got_patience 2 --got_selection_metric composite
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path
import os
import sys
import textwrap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.utils.report_generator import FBSLReportGenerator


def ensure_prototype_dir(project_id: str, rank: int, node_id: str) -> Path:
    """Create (and return) the per-prototype output directory, matching the
    layout FBSLReportGenerator uses: outputs/{project_id}/prototypes/{rank}_{node_id[:8]}"""
    proto_dir = Path('outputs') / project_id / 'prototypes' / f'{rank}_{node_id[:8]}'
    proto_dir.mkdir(parents=True, exist_ok=True)
    return proto_dir


async def run_pipeline(args):
    orchestrator = PipelineOrchestrator()
    report_gen = FBSLReportGenerator()

    req = {
        'project_name': args.project_name or 'Untitled_Project',
        'requirements': args.requirements,
        'context': {},
        'max_alternatives': args.max_alternatives,
        'use_got': True,
        'enable_convergence_loop': True,
        'got_delta': args.got_delta,
        'got_patience': args.got_patience,
        'got_max_nodes': args.got_max_nodes,
        'got_selection_metric': args.got_selection_metric
    }

    print('🚀 Running FBSL-KAGS pipeline...')
    print(f'   Project: {req["project_name"]}')
    print(f'   Requirements: {args.requirements[:80]}...' if len(args.requirements) > 80 else f'   Requirements: {args.requirements}')
    print()
    
    result = await orchestrator.process_design_request(req)

    if not result.get('success'):
        print('❌ Pipeline failed:', result.get('error'))
        return result

    project_id = result.get('project_id') or 'project'
    project_name = result.get('project_name') or 'Untitled_Project'
    designs = result.get('designs', [])
    complexity_metrics = result.get('complexity_metrics')
    processing_time = result.get('processing_time', 0.0)

    print(f"✅ Pipeline complete! Found {len(designs)} design prototypes")
    print(f"   Processing time: {processing_time:.2f}s")
    print()
    print(f"📊 Generating comprehensive FBSL reports for top {args.top_k} prototypes...")
    print()

    # Collect prototype summaries for project report
    prototype_summaries = []
    
    # Process each top prototype
    for rank, design in enumerate(designs[:args.top_k], 1):
        node_id = design['node_id']
        print(f"  [{rank}/{args.top_k}] Processing prototype {node_id[:8]}...")
        
        # Create prototype directory
        proto_dir = ensure_prototype_dir(project_id, rank, node_id)
        
        # Copy visualization files from visual_outputs if they exist
        visual_outputs = Path('visual_outputs')
        layout_svg_path = None
        adjacency_png_path = None
        
        if visual_outputs.exists():
            # Find matching files
            for file in visual_outputs.glob(f"*{node_id[:8]}*layout*.svg"):
                dest = proto_dir / "layout.svg"
                shutil.copy2(file, dest)
                layout_svg_path = str(dest)
                print(f"      ✓ Copied layout SVG")
                break
            
            for file in visual_outputs.glob(f"*{node_id[:8]}*adjacency*.png"):
                dest = proto_dir / "adjacency.png"
                shutil.copy2(file, dest)
                adjacency_png_path = str(dest)
                print(f"      ✓ Copied adjacency graph")
                break
        
        # Save basic metadata JSON
        metadata_file = proto_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(design, f, ensure_ascii=False, indent=2, default=str)
        print(f"      ✓ Saved metadata")
        
        # Generate comprehensive FBSL reports
        # Note: We need the actual node object, not just the design dict
        # For now, save what we have and note that full FBSL report requires node object
        
        # Create a simplified FBSL report from available data
        fbsl_md_path = proto_dir / "fbsl_report.md"
        fbsl_html_path = proto_dir / "fbsl_report.html"
        
        _generate_simple_fbsl_report(design, rank, project_name, fbsl_md_path, fbsl_html_path, 
                                     layout_svg_path, adjacency_png_path)
        
        print(f"      ✓ Generated FBSL reports (MD, HTML)")
        print(f"      📁 Saved to: {proto_dir}")
        print()
        
        # Collect for project summary
        prototype_summaries.append({
            'rank': rank,
            'node_id': node_id,
            'scores': design.get('scores', {}),
            'variant_type': design.get('variant_type', 'N/A'),
            'functions_count': design.get('functions_count', 0),
            'behaviors_count': design.get('behaviors_count', 0),
            'structures_count': design.get('structures_count', 0),
            'converged': design.get('converged', False),
            'convergence_iterations': design.get('convergence_iterations', 0),
            'is_pareto_optimal': design.get('is_pareto_optimal', False)
        })

    # Generate project summary
    print("📋 Generating project summary report...")
    project_dir = Path('outputs') / project_id
    
    summary_data = {
        'project_id': project_id,
        'project_name': project_name,
        'total_prototypes': len(designs),
        'processing_time': processing_time,
        'complexity_metrics': complexity_metrics,
        'prototypes': prototype_summaries
    }
    
    _generate_project_summary(summary_data, project_dir)
    print(f"   ✓ Project summary saved to: {project_dir}")
    print()

    # Print summary table
    print("=" * 80)
    print("TOP PROTOTYPES SUMMARY")
    print("=" * 80)
    print()
    
    for i, design in enumerate(designs[:args.top_k], 1):
        scores = design.get('scores', {})
        print(f"Prototype #{i}: {design.get('node_id')[:16]}...")
        print(f"  Composite Score:  {scores.get('composite', 0.0):.3f}")
        print(f"  Functional:       {scores.get('functional_adequacy', 0.0):.3f}")
        print(f"  Behavioral:       {scores.get('behavioral_performance', 0.0):.3f}")
        print(f"  Structural:       {scores.get('structural_feasibility', 0.0):.3f}")
        print(f"  Layout:           {scores.get('layout_efficiency', 0.0):.3f}")
        print(f"  Sustainability:   {scores.get('sustainability', 0.0):.3f}")
        print(f"  Variant Type:     {design.get('variant_type', 'N/A')}")
        print(f"  Converged:        {'✅ Yes' if design.get('converged') else '❌ No'}")
        print(f"  Pareto Optimal:   {'✅ Yes' if design.get('is_pareto_optimal') else '❌ No'}")
        print()

    print("=" * 80)
    print(f"✅ All reports generated successfully!")
    print(f"📁 Output directory: outputs/{project_id}/")
    print("=" * 80)

    return result


def _generate_simple_fbsl_report(design, rank, project_name, md_path, html_path, 
                                 layout_img, adjacency_img):
    """Generate simplified FBSL report from design dictionary"""
    
    scores = design.get('scores', {})
    
    md_content = f"""# FBSL Design Prototype Report

## Prototype #{rank}: {project_name}

**Node ID:** `{design.get('node_id', 'N/A')}`  
**Variant Type:** {design.get('variant_type', 'N/A')}  
**Description:** {design.get('description', 'N/A')}

---

## Performance Scores

| Dimension | Score | Status |
|-----------|-------|--------|
| **Composite** | **{scores.get('composite', 0.0):.3f}** | {'✅ Excellent' if scores.get('composite', 0) > 0.8 else '✓ Good' if scores.get('composite', 0) > 0.6 else '⚠ Fair'} |
| Functional Adequacy | {scores.get('functional_adequacy', 0.0):.3f} | {'✅' if scores.get('functional_adequacy', 0) > 0.7 else '⚠'} |
| Behavioral Performance | {scores.get('behavioral_performance', 0.0):.3f} | {'✅' if scores.get('behavioral_performance', 0) > 0.7 else '⚠'} |
| Structural Feasibility | {scores.get('structural_feasibility', 0.0):.3f} | {'✅' if scores.get('structural_feasibility', 0) > 0.7 else '⚠'} |
| Layout Efficiency | {scores.get('layout_efficiency', 0.0):.3f} | {'✅' if scores.get('layout_efficiency', 0) > 0.7 else '⚠'} |
| Sustainability | {scores.get('sustainability', 0.0):.3f} | {'✅' if scores.get('sustainability', 0) > 0.7 else '⚠'} |

---

## FBSL Components

### Functions
- **Total Functions:** {design.get('functions_count', 0)}

### Behaviors
- **Total Behaviors:** {design.get('behaviors_count', 0)}

### Structures
- **Total Structures:** {design.get('structures_count', 0)}

### Layout
- **Has Layout:** {'✅ Yes' if design.get('has_layout') else '❌ No'}
- **Floor Plan Available:** {'✅ Yes' if design.get('has_floor_plan_svg') else '❌ No'}
- **Adjacency Graph Available:** {'✅ Yes' if design.get('has_adjacency_svg') else '❌ No'}

---

## Visualizations

"""
    
    if layout_img:
        md_content += f"### Layout Floor Plan\n\n![Layout Floor Plan](layout.svg)\n\n"
    
    if adjacency_img:
        md_content += f"### Adjacency Graph\n\n![Adjacency Graph](adjacency.png)\n\n"
    
    md_content += f"""---

## Convergence History

| Metric | Value |
|--------|-------|
| Converged | {'✅ Yes' if design.get('converged') else '❌ No'} |
| Convergence Iterations | {design.get('convergence_iterations', 0)} |
| Refinement Iterations | {design.get('refinement_iterations', 0)} |
| Pareto Optimal | {'✅ Yes' if design.get('is_pareto_optimal') else '❌ No'} |

---

*Report generated by FBSL-KAGS Framework*

**Note:** For complete FBSL component details (individual functions, behaviors, structures), 
see the `metadata.json` file in this directory.
"""
    
    # Save markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # Generate simple HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FBSL Prototype Report - {project_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 20px 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        hr {{ border: none; border-top: 2px solid #ecf0f1; margin: 30px 0; }}
        .status-good {{ color: #27ae60; }}
        .status-fair {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="container">
        {md_content.replace('**', '<strong>').replace('`', '<code>').replace('</code>', '</code>').replace('---', '<hr>')}
    </div>
</body>
</html>
"""
    
    # Save HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _generate_project_summary(data, project_dir):
    """Generate project summary markdown and HTML"""
    
    md_content = f"""# Project Summary: {data['project_name']}

**Project ID:** `{data['project_id']}`  
**Total Prototypes:** {data['total_prototypes']}  
**Processing Time:** {data['processing_time']:.2f} seconds

---

## Complexity Analysis

"""
    
    if data.get('complexity_metrics'):
        cm = data['complexity_metrics']
        md_content += f"""
| Metric | Value |
|--------|-------|
| Complexity Level | **{cm.get('level', 'N/A')}** |
| Overall Score | {cm.get('overall_score', 0):.3f} |
| Function Count | {cm.get('function_count', 0)} |
| Behavior Count | {cm.get('behavior_count', 0)} |
| Room Count | {cm.get('room_count', 0)} |

"""
    else:
        md_content += "*No complexity metrics available*\n\n"
    
    md_content += "---\n\n## Design Prototypes\n\n"
    
    for proto in data['prototypes']:
        scores = proto.get('scores', {})
        md_content += f"""### Prototype {proto['rank']}

**Node ID:** `{proto['node_id'][:16]}...`  
**Composite Score:** {scores.get('composite', 0):.3f}  
**Variant Type:** {proto.get('variant_type', 'N/A')}

| Dimension | Score |
|-----------|-------|
| Functional | {scores.get('functional_adequacy', 0):.3f} |
| Behavioral | {scores.get('behavioral_performance', 0):.3f} |
| Structural | {scores.get('structural_feasibility', 0):.3f} |
| Layout | {scores.get('layout_efficiency', 0):.3f} |
| Sustainability | {scores.get('sustainability', 0):.3f} |

**Components:**
- Functions: {proto.get('functions_count', 0)}
- Behaviors: {proto.get('behaviors_count', 0)}
- Structures: {proto.get('structures_count', 0)}

**Status:**
- Converged: {'✅ Yes' if proto.get('converged') else '❌ No'}
- Convergence Iterations: {proto.get('convergence_iterations', 0)}
- Pareto Optimal: {'✅ Yes' if proto.get('is_pareto_optimal') else '❌ No'}

---

"""
    
    md_content += "\n*Summary generated by FBSL-KAGS Framework*\n"
    
    # Save markdown
    md_path = project_dir / "project_summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # Save HTML (simple conversion)
    html_path = project_dir / "project_summary.html"
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project Summary - {data['project_name']}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        hr {{ border: none; border-top: 2px solid #ecf0f1; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <pre>{md_content}</pre>
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def parse_args():
    p = argparse.ArgumentParser(description='Run FBSL-KAGS pipeline with comprehensive reporting')
    p.add_argument('--project_name', '-p', type=str, default=None, help='Project name')
    p.add_argument('--requirements', '-r', type=str, required=True, help='Design requirements')
    p.add_argument('--top_k', type=int, default=3, help='Number of top prototypes to save')
    p.add_argument('--max_alternatives', type=int, default=5, help='Maximum alternatives to generate')
    p.add_argument('--got_delta', type=float, default=None, help='GoT delta parameter')
    p.add_argument('--got_patience', type=int, default=None, help='GoT patience parameter')
    p.add_argument('--got_max_nodes', type=int, default=None, help='GoT max nodes')
    p.add_argument('--got_selection_metric', type=str, default='composite', help='GoT selection metric')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except Exception as e:
        print(f'❌ Pipeline execution failed: {e}')
        import traceback
        traceback.print_exc()
        raise
