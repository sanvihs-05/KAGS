"""
Exploration Report Generator

Generates comprehensive markdown report showing the complete Graph of Thoughts
exploration tree with reasoning for each node.
"""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ExplorationReportGenerator:
    """
    Generates detailed exploration report showing GoT tree and node reasoning
    """
    
    def __init__(self, node_storage, tree_structure: Dict):
        self.storage = node_storage
        self.tree = tree_structure
        self.nodes = node_storage.node_registry
    
    def generate(self, output_path: Path):
        """Generate complete exploration report"""
        
        report_lines = []
        
        # Header
        report_lines.extend(self._generate_header())
        
        # Overview
        report_lines.extend(self._generate_overview())
        
        # Exploration Tree Diagram
        report_lines.extend(self._generate_tree_diagram())
        
        # Node Details (all levels)
        report_lines.extend(self._generate_node_details())
        
        # Final Prototypes Summary
        report_lines.extend(self._generate_final_summary())
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _generate_header(self) -> List[str]:
        return [
            "# Graph of Thoughts Exploration Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ]
    
    def _generate_overview(self) -> List[str]:
        stats = self.storage.get_statistics()
        
        lines = [
            "## Exploration Overview",
            "",
            f"- **Total Nodes Explored:** {stats['total_nodes']}",
            f"- **Maximum Depth:** {stats['max_depth']}",
            f"- **Nodes Pruned:** {stats['pruned_count']} ({stats['pruning_rate']:.1%})",
            ""
        ]
        
        # Nodes by level
        lines.append("**Nodes by Level:**")
        for level in sorted(stats['nodes_by_level'].keys(), key=lambda x: int(x) if x.isdigit() else 999):
            count = stats['nodes_by_level'][level]
            lines.append(f"- Level {level}: {count} nodes")
        
        lines.extend(["", "---", ""])
        return lines
    
    def _generate_tree_diagram(self) -> List[str]:
        lines = [
            "## Exploration Tree",
            "",
            "```"
        ]
        
        root_id = self.tree["root"]
        if root_id:
            lines.extend(self._build_tree_recursive(root_id, "", True))
        
        lines.extend(["```", "", "---", ""])
        return lines
    
    def _build_tree_recursive(self, node_id: str, prefix: str, is_last: bool) -> List[str]:
        """Recursively build tree diagram"""
        lines = []
        
        node_data = self.nodes.get(node_id)
        if not node_data:
            return lines
        
        # Node info
        level = node_data.get("level", "?")
        scores = node_data.get("scores", {})
        composite = scores.get("composite", 0.0)
        pruned = node_data.get("metadata", {}).get("pruned", False)
        
        # Format node line
        connector = "└─ " if is_last else "├─ "
        status = "[PRUNED]" if pruned else "[RETAINED]"
        node_name = node_data.get("reasoning", f"Node {node_id[:8]}")[:50]
        
        line = f"{prefix}{connector}L{level}: {node_name} [Score: {composite:.3f}] {status}"
        lines.append(line)
        
        # Get children
        rel = self.tree["relationships"].get(node_id, {})
        children = rel.get("children", [])
        
        # Recurse for children
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(self._build_tree_recursive(child_id, child_prefix, is_last_child))
        
        return lines
    
    def _generate_node_details(self) -> List[str]:
        lines = [
            "## Node Details",
            ""
        ]
        
        # Group by level
        levels = sorted(self.tree["levels"].keys(), key=lambda x: int(x) if x.isdigit() else 999)
        
        for level in levels:
            node_ids = self.tree["levels"][level]
            
            lines.append(f"### Level {level}")
            lines.append("")
            
            for node_id in node_ids:
                lines.extend(self._format_node_detail(node_id))
                lines.append("")
        
        return lines
    
    def _format_node_detail(self, node_id: str) -> List[str]:
        """Format detailed node information"""
        node_data = self.nodes.get(node_id)
        if not node_data:
            return []
        
        lines = []
        
        # Header
        reasoning = node_data.get("reasoning", f"Node {node_id[:8]}")
        lines.append(f"#### {reasoning}")
        lines.append("")
        
        # Basic info
        lines.append(f"**Node ID:** `{node_id[:12]}...`")
        lines.append(f"**Parent:** `{node_data.get('parent_id', 'None')[:12] if node_data.get('parent_id') else 'None'}...`")
        
        trans_type = node_data.get("transformation_type")
        if trans_type:
            lines.append(f"**Transformation:** {trans_type}")
        
        # Scores
        scores = node_data.get("scores", {})
        lines.append("")
        lines.append("**Scores:**")
        lines.append(f"- Functional: {scores.get('functional', 0.0):.3f}")
        lines.append(f"- Behavioral: {scores.get('behavioral', 0.0):.3f}")
        lines.append(f"- Structural: {scores.get('structural', 0.0):.3f}")
        lines.append(f"- Layout: {scores.get('layout', 0.0):.3f}")
        lines.append(f"- Sustainability: {scores.get('sustainability', 0.0):.3f}")
        lines.append(f"- **Composite: {scores.get('composite', 0.0):.3f}**")
        
        # FBSL Summary
        functions = node_data.get("functions", [])
        behaviors = node_data.get("behaviors", [])
        structures = node_data.get("structures", [])
        layout = node_data.get("layout")
        
        lines.append("")
        lines.append("**FBSL Summary:**")
        lines.append(f"- Functions: {len(functions)}")
        lines.append(f"- Behaviors: {len(behaviors)}")
        lines.append(f"- Structures: {len(structures)}")
        if layout:
            rooms = layout.get("rooms", [])
            total_area = layout.get("total_area", 0.0)
            lines.append(f"- Layout: {total_area:.1f} sqm, {len(rooms)} rooms")
        
        # Pruning info
        pruned = node_data.get("metadata", {}).get("pruned", False)
        if pruned:
            reason = node_data.get("metadata", {}).get("prune_reason", "Score below threshold")
            lines.append("")
            lines.append(f"**Pruned:** {reason}")
        
        # Visualizations
        level = node_data.get("level")
        lines.append("")
        lines.append("**Visualizations:**")
        lines.append(f"- Floor Plan: `visualizations/level_{level}/{node_id[:8]}_layout.svg`")
        lines.append(f"- Adjacency Graph: `visualizations/level_{level}/{node_id[:8]}_adjacency.png`")
        
        lines.append("")
        lines.append("---")
        
        return lines
    
    def _generate_final_summary(self) -> List[str]:
        lines = [
            "",
            "## Final Prototypes Summary",
            ""
        ]
        
        # Get final level nodes
        final_nodes = self.tree["levels"].get("final", [])
        
        if not final_nodes:
            lines.append("No final prototypes selected.")
            return lines
        
        # Sort by composite score
        final_data = [(nid, self.nodes[nid]) for nid in final_nodes if nid in self.nodes]
        final_data.sort(key=lambda x: x[1].get("scores", {}).get("composite", 0.0), reverse=True)
        
        for i, (node_id, node_data) in enumerate(final_data, 1):
            scores = node_data.get("scores", {})
            reasoning = node_data.get("reasoning", f"Prototype {i}")
            
            lines.append(f"### Prototype {i}: {reasoning}")
            lines.append("")
            lines.append(f"**Composite Score:** {scores.get('composite', 0.0):.3f}")
            lines.append("")
            lines.append("**Dimension Scores:**")
            lines.append(f"- Functional: {scores.get('functional', 0.0):.3f}")
            lines.append(f"- Behavioral: {scores.get('behavioral', 0.0):.3f}")
            lines.append(f"- Structural: {scores.get('structural', 0.0):.3f}")
            lines.append(f"- Layout: {scores.get('layout', 0.0):.3f}")
            lines.append(f"- Sustainability: {scores.get('sustainability', 0.0):.3f}")
            lines.append("")
            
            # Path from root
            path = self.storage.get_node_path(node_id)
            lines.append("**Evolution Path:**")
            lines.append("```")
            for j, path_node_id in enumerate(path):
                path_node = self.nodes.get(path_node_id)
                if path_node:
                    level = path_node.get("level")
                    reason = path_node.get("reasoning", f"Node {path_node_id[:8]}")[:40]
                    lines.append(f"L{level}: {reason}")
                    if j < len(path) - 1:
                        lines.append("  ↓")
            lines.append("```")
            lines.append("")
        
        return lines
