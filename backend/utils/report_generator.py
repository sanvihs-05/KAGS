"""
FBSL Report Generator - Comprehensive reporting for design prototypes

Generates human-readable reports consolidating Functions, Behaviors, Structures,
and Layout components for FBSL design prototypes.

Output formats:
- Markdown (.md) - Human-readable text format
- HTML (.html) - Rich formatted with embedded images
- JSON (.json) - Machine-readable complete data
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


class FBSLReportGenerator:
    """Generates comprehensive FBSL reports for design prototypes"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Base directory for report outputs (default: outputs/)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_prototype_report(
        self,
        node: Any,
        scores: Dict[str, Any],
        rank: int,
        project_id: str,
        project_name: str,
        layout_image_path: Optional[str] = None,
        adjacency_image_path: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Generate complete report for a single prototype
        
        Args:
            node: FBSLLayoutNode instance
            scores: Score dictionary from scoring agent
            rank: Prototype rank (1, 2, 3, ...)
            project_id: Project identifier
            project_name: Human-readable project name
            layout_image_path: Path to layout image (PNG or SVG)
            adjacency_image_path: Path to adjacency graph image
        
        Returns:
            Dictionary with paths to generated reports
        """
        # Create prototype directory
        prototype_dir = self.output_dir / project_id / "prototypes" / f"{rank}_{node.node_id[:8]}"
        prototype_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports in all formats
        report_data = self._prepare_report_data(
            node, scores, rank, project_name,
            layout_image_path, adjacency_image_path
        )
        
        # Save JSON
        json_path = prototype_dir / "fbsl_data.json"
        self._save_json_report(report_data, json_path)
        
        # Save Markdown
        md_path = prototype_dir / "fbsl_report.md"
        self._save_markdown_report(report_data, md_path)
        
        # Save HTML
        html_path = prototype_dir / "fbsl_report.html"
        self._save_html_report(report_data, html_path)
        
        logger.info(f"Generated reports for prototype {rank} in {prototype_dir}")
        
        return {
            'json': json_path,
            'markdown': md_path,
            'html': html_path,
            'directory': prototype_dir
        }
    
    def generate_project_summary(
        self,
        project_id: str,
        project_name: str,
        prototypes: List[Dict[str, Any]],
        complexity_metrics: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0
    ) -> Dict[str, Path]:
        """
        Generate project-level summary report
        
        Args:
            project_id: Project identifier
            project_name: Human-readable project name
            prototypes: List of prototype data dictionaries
            complexity_metrics: Complexity analysis results
            processing_time: Total processing time in seconds
        
        Returns:
            Dictionary with paths to summary reports
        """
        project_dir = self.output_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            'project_id': project_id,
            'project_name': project_name,
            'generated_at': datetime.now().isoformat(),
            'total_prototypes': len(prototypes),
            'processing_time': processing_time,
            'complexity_metrics': complexity_metrics,
            'prototypes': prototypes
        }
        
        # Save Markdown summary
        md_path = project_dir / "project_summary.md"
        self._save_project_summary_markdown(summary_data, md_path)
        
        # Save HTML summary
        html_path = project_dir / "project_summary.html"
        self._save_project_summary_html(summary_data, html_path)
        
        logger.info(f"Generated project summary in {project_dir}")
        
        return {
            'markdown': md_path,
            'html': html_path,
            'directory': project_dir
        }
    
    def _prepare_report_data(
        self,
        node: Any,
        scores: Dict[str, Any],
        rank: int,
        project_name: str,
        layout_image_path: Optional[str],
        adjacency_image_path: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare structured data for report generation"""
        
        # Extract FBSL components
        functions = self._extract_functions(node)
        behaviors = self._extract_behaviors(node)
        structures = self._extract_structures(node)
        layout_metrics = self._extract_layout_metrics(node)
        
        # Extract scores
        score_details = scores.get('scores', {})
        
        # Extract metadata
        metadata = {
            'node_id': node.node_id,
            'rank': rank,
            'project_name': project_name,
            'variant_type': node.metadata.get('variant_type', 'N/A'),
            'description': node.metadata.get('description', 'N/A'),
            'generated_at': datetime.now().isoformat()
        }
        
        # Convergence history
        convergence = node.metadata.get('convergence_history', {})
        refinement = node.metadata.get('refinement_history', {})
        
        return {
            'metadata': metadata,
            'functions': functions,
            'behaviors': behaviors,
            'structures': structures,
            'layout': layout_metrics,
            'scores': score_details,
            'convergence': convergence,
            'refinement': refinement,
            'images': {
                'layout': layout_image_path,
                'adjacency': adjacency_image_path
            }
        }
    
    def _extract_functions(self, node: Any) -> List[Dict[str, Any]]:
        """Extract function data from node"""
        functions = []
        for func_id, func in node.functions.items():
            func_data = {
                'id': func_id,
                'name': func.name,
                'category': func.category.value if hasattr(func.category, 'value') else str(func.category),
                'priority': func.priority,
                'activities': func.activities,
                'spatial_requirements': func.spatial_requirements,
                'temporal_needs': func.temporal_needs
            }
            functions.append(func_data)
        
        # Sort by priority (descending)
        functions.sort(key=lambda x: x['priority'], reverse=True)
        return functions
    
    def _extract_behaviors(self, node: Any) -> List[Dict[str, Any]]:
        """Extract behavior data from node"""
        behaviors = []
        for behav_id, behav in node.behaviors.items():
            behav_data = {
                'id': behav_id,
                'metric_name': behav.metric_name,
                'category': behav.category.value if hasattr(behav.category, 'value') else str(behav.category),
                'target_value': behav.target_value,
                'actual_value': behav.actual_value,
                'unit': getattr(behav, 'unit', ''),
                'tolerance': behav.tolerance,
                'is_satisfied': behav.is_satisfied,
                'satisfaction_degree': getattr(behav, 'satisfaction_degree', 0.0),
                'source_function': behav.source_function
            }
            behaviors.append(behav_data)
        
        # Sort by satisfaction (unsatisfied first, then by category)
        behaviors.sort(key=lambda x: (x['is_satisfied'], x['category']))
        return behaviors
    
    def _extract_structures(self, node: Any) -> List[Dict[str, Any]]:
        """Extract structure data from node"""
        structures = []
        for struct_id, struct in node.structures.items():
            struct_data = {
                'id': struct_id,
                'name': struct.name,
                'type': struct.type.value if hasattr(struct.type, 'value') else str(struct.type),
                'material': struct.material,
                'properties': struct.properties,
                'dimensions': getattr(struct, 'dimensions', {}),
                'load_bearing': getattr(struct, 'load_bearing', False)
            }
            structures.append(struct_data)
        
        # Sort by type
        structures.sort(key=lambda x: x['type'])
        return structures
    
    def _extract_layout_metrics(self, node: Any) -> Dict[str, Any]:
        """Extract layout metrics from node"""
        if not node.layout:
            return {}
        
        layout = node.layout
        metrics = {
            'total_area': getattr(layout, 'total_area', 0.0),
            'used_area': getattr(layout, 'used_area', 0.0),
            'room_count': len(layout.rooms) if layout.rooms else 0,
            'compactness': getattr(layout, 'compactness', 0.0),
            'circulation_efficiency': getattr(layout, 'circulation_efficiency', 0.0),
            'adjacency_satisfaction': getattr(layout, 'adjacency_satisfaction', 0.0)
        }
        
        # Room details
        rooms = []
        if layout.rooms:
            for room_id, room in layout.rooms.items():
                room_data = {
                    'id': room_id,
                    'name': room.name,
                    'type': room.room_type,
                    'area': room.area,
                    'height': getattr(room, 'height', 0.0),
                    'volume': getattr(room, 'volume', 0.0)
                }
                rooms.append(room_data)
        
        metrics['rooms'] = rooms
        return metrics
    
    def _save_json_report(self, data: Dict[str, Any], path: Path):
        """Save JSON report"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_markdown_report(self, data: Dict[str, Any], path: Path):
        """Save Markdown report"""
        md_content = self._generate_markdown_content(data)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _save_html_report(self, data: Dict[str, Any], path: Path):
        """Save HTML report"""
        html_content = self._generate_html_content(data)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_markdown_content(self, data: Dict[str, Any]) -> str:
        """Generate markdown report content"""
        meta = data['metadata']
        functions = data['functions']
        behaviors = data['behaviors']
        structures = data['structures']
        layout = data['layout']
        scores = data['scores']
        convergence = data['convergence']
        images = data['images']
        
        md = f"""# FBSL Design Prototype Report

## Prototype #{meta['rank']}: {meta['project_name']}

**Node ID:** `{meta['node_id']}`  
**Variant Type:** {meta['variant_type']}  
**Description:** {meta['description']}  
**Generated:** {meta['generated_at']}

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

## 1. Functions (F)

Total Functions: **{len(functions)}**

"""
        
        # Functions table
        if functions:
            md += "| Priority | Name | Category | Activities | Spatial Requirements |\n"
            md += "|----------|------|----------|------------|---------------------|\n"
            for func in functions:
                activities = ', '.join(func['activities'][:3]) if func['activities'] else 'N/A'
                spatial = f"{func['spatial_requirements'].get('min_area', 'N/A')} m²" if func['spatial_requirements'] else 'N/A'
                md += f"| {func['priority']:.2f} | {func['name']} | {func['category']} | {activities} | {spatial} |\n"
        else:
            md += "*No functions defined*\n"
        
        md += "\n---\n\n## 2. Behaviors (B)\n\n"
        md += f"Total Behaviors: **{len(behaviors)}**  \n"
        
        # Behavior satisfaction summary
        satisfied_count = sum(1 for b in behaviors if b['is_satisfied'])
        md += f"Satisfied: **{satisfied_count}/{len(behaviors)}** ({satisfied_count/len(behaviors)*100:.1f}%)\n\n" if behaviors else ""
        
        # Behaviors table
        if behaviors:
            md += "| Metric | Category | Target | Actual | Satisfied | Degree |\n"
            md += "|--------|----------|--------|--------|-----------|--------|\n"
            for behav in behaviors:
                status = '✅' if behav['is_satisfied'] else '❌'
                target = f"{behav['target_value']:.2f} {behav['unit']}" if behav['unit'] else f"{behav['target_value']:.2f}"
                actual = f"{behav['actual_value']:.2f} {behav['unit']}" if behav['unit'] and behav['actual_value'] else f"{behav['actual_value']:.2f}"
                degree = f"{behav['satisfaction_degree']:.2f}"
                md += f"| {behav['metric_name']} | {behav['category']} | {target} | {actual} | {status} | {degree} |\n"
        else:
            md += "*No behaviors defined*\n"
        
        md += "\n---\n\n## 3. Structures (S)\n\n"
        md += f"Total Structures: **{len(structures)}**\n\n"
        
        # Structures by type
        if structures:
            struct_by_type = {}
            for struct in structures:
                stype = struct['type']
                if stype not in struct_by_type:
                    struct_by_type[stype] = []
                struct_by_type[stype].append(struct)
            
            for stype, structs in sorted(struct_by_type.items()):
                md += f"### {stype}\n\n"
                md += "| Name | Material | Properties |\n"
                md += "|------|----------|------------|\n"
                for struct in structs:
                    props = ', '.join([f"{k}: {v}" for k, v in list(struct['properties'].items())[:2]]) if struct['properties'] else 'N/A'
                    md += f"| {struct['name']} | {struct['material']} | {props} |\n"
                md += "\n"
        else:
            md += "*No structures defined*\n"
        
        md += "\n---\n\n## 4. Layout (L)\n\n"
        
        if layout:
            md += f"""### Metrics

| Metric | Value |
|--------|-------|
| Total Area | {layout.get('total_area', 0):.2f} m² |
| Used Area | {layout.get('used_area', 0):.2f} m² |
| Room Count | {layout.get('room_count', 0)} |
| Compactness | {layout.get('compactness', 0):.3f} |
| Circulation Efficiency | {layout.get('circulation_efficiency', 0):.3f} |
| Adjacency Satisfaction | {layout.get('adjacency_satisfaction', 0):.3f} |

### Rooms

"""
            if layout.get('rooms'):
                md += "| Name | Type | Area (m²) | Height (m) | Volume (m³) |\n"
                md += "|------|------|-----------|------------|-------------|\n"
                for room in layout['rooms']:
                    md += f"| {room['name']} | {room['type']} | {room['area']:.2f} | {room['height']:.2f} | {room['volume']:.2f} |\n"
            else:
                md += "*No rooms defined*\n"
        else:
            md += "*No layout data available*\n"
        
        # Add images if available
        md += "\n---\n\n## 5. Visualizations\n\n"
        
        if images.get('layout'):
            md += f"### Layout Floor Plan\n\n![Layout Floor Plan]({images['layout']})\n\n"
        
        if images.get('adjacency'):
            md += f"### Adjacency Graph\n\n![Adjacency Graph]({images['adjacency']})\n\n"
        
        # Convergence info
        if convergence:
            md += f"""---

## Convergence History

| Metric | Value |
|--------|-------|
| Converged | {'✅ Yes' if convergence.get('converged') else '❌ No'} |
| Iterations | {convergence.get('iterations', 0)} |
| Initial Score | {convergence.get('initial_score', 0):.3f} |
| Final Score | {convergence.get('final_score', 0):.3f} |
| Improvement | {convergence.get('score_improvement', 0):.3f} |

"""
        
        md += "\n---\n\n*Report generated by FBSL-KAGS Framework*\n"
        
        return md
    
    def _generate_html_content(self, data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        md_content = self._generate_markdown_content(data)
        
        # Simple HTML wrapper with styling
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FBSL Prototype Report - {data['metadata']['project_name']}</title>
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
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .score-excellent {{ color: #27ae60; font-weight: bold; }}
        .score-good {{ color: #f39c12; font-weight: bold; }}
        .score-fair {{ color: #e74c3c; font-weight: bold; }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
"""
        
        # Convert markdown tables and formatting to HTML
        # Simple conversion - replace markdown with HTML
        html_body = md_content
        
        # Convert headers
        html_body = html_body.replace('# ', '<h1>').replace('\n\n', '</h1>\n\n', 1)
        html_body = html_body.replace('## ', '<h2>').replace('\n\n', '</h2>\n\n')
        html_body = html_body.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')
        
        # Convert bold
        import re
        html_body = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_body)
        
        # Convert code blocks
        html_body = re.sub(r'`(.*?)`', r'<code>\1</code>', html_body)
        
        # Convert images
        html_body = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1" />', html_body)
        
        # Convert horizontal rules
        html_body = html_body.replace('---', '<hr>')
        
        # Convert line breaks
        html_body = html_body.replace('\n\n', '<br><br>')
        
        html += html_body
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def _save_project_summary_markdown(self, data: Dict[str, Any], path: Path):
        """Save project summary in markdown format"""
        md = f"""# Project Summary: {data['project_name']}

**Project ID:** `{data['project_id']}`  
**Generated:** {data['generated_at']}  
**Total Prototypes:** {data['total_prototypes']}  
**Processing Time:** {data['processing_time']:.2f} seconds

---

## Complexity Analysis

"""
        
        if data.get('complexity_metrics'):
            cm = data['complexity_metrics']
            md += f"""
| Metric | Value |
|--------|-------|
| Complexity Level | **{cm.get('level', 'N/A')}** |
| Overall Score | {cm.get('overall_score', 0):.3f} |
| Function Count | {cm.get('function_count', 0)} |
| Behavior Count | {cm.get('behavior_count', 0)} |
| Room Count | {cm.get('room_count', 0)} |

"""
        else:
            md += "*No complexity metrics available*\n\n"
        
        md += "---\n\n## Design Prototypes\n\n"
        
        for i, proto in enumerate(data['prototypes'], 1):
            scores = proto.get('scores', {})
            md += f"""### Prototype {i}

**Node ID:** `{proto.get('node_id', 'N/A')[:16]}...`  
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
        
        md += "\n*Summary generated by FBSL-KAGS Framework*\n"
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)
    
    def _save_project_summary_html(self, data: Dict[str, Any], path: Path):
        """Save project summary in HTML format"""
        # Read markdown and convert to HTML (simplified)
        md_path = path.parent / "project_summary.md"
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Summary - {data['project_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
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
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
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
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
