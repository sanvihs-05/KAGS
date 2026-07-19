"""
Quick Integration Guide for Improved Layout Visualizer

The improved visualizer has been created in:
backend/visualization/improved_layout_visualizer.py

To use it in your pipeline:
"""

# Option 1: Direct usage in scripts
from backend.visualization.improved_layout_visualizer import ImprovedLayoutVisualizer

visualizer = ImprovedLayoutVisualizer(output_dir="visual_outputs")

# Render layout (returns tuple of paths)
svg_path, adjacency_path = visualizer.render(
    layout=layout_object,  # Your Layout object
    project_name="my_project",
    node_id="node_123"
)

print(f"Generated SVG: {svg_path}")
print(f"Generated Adjacency Graph: {adjacency_path}")


# Option 2: Replace in layout_agent.py
# Find line ~68 where EnhancedLayoutVisualizer is initialized
# Replace:
#   self.enhanced_visualizer = EnhancedLayoutVisualizer()
# With:
#   from ..visualization.improved_layout_visualizer import ImprovedLayoutVisualizer
#   self.improved_visualizer = ImprovedLayoutVisualizer()

# Then update the render call (around line 194):
# Replace:
#   enhanced_outputs = self.enhanced_visualizer.render(layout, project_name, node.node_id)
# With:
#   svg_path, adj_path = self.improved_visualizer.render(layout, project_name, node.node_id)
#   enhanced_outputs = {'svg_floor_plan': svg_path, 'adjacency_graph': adj_path}


# Key Features of Improved Visualizer:
# 
# 1. Smart Room Placement (CompactRoomPlacer)
#    - Prioritizes important rooms (living room, kitchen, hallways)
#    - Places rooms based on critical adjacencies
#    - Creates compact, understandable layouts
#
# 2. Strict Adjacency Detection
#    - Rooms must actually touch (within 1ft tolerance)
#    - Requires 30% edge overlap to count as adjacent
#    - No false adjacencies from distant rooms
#
# 3. Clean Adjacency Graphs (4 panels)
#    - Panel 1: Spatial layout with actual room positions
#    - Panel 2: Graph topology with edge types (critical/preferred/spatial/bridge)
#    - Panel 3: Connectivity analysis with satisfaction scores
#    - Panel 4: Performance metrics bar chart
#
# 4. Edge Classification
#    - Critical (green): Must-have adjacencies (kitchen-dining, bedroom-bathroom)
#    - Preferred (blue): Nice-to-have adjacencies
#    - Spatial (gray): Rooms that happen to touch
#    - Bridge (red dashed): Added to ensure graph connectivity
#
# 5. Clean SVG Floor Plans
#    - Compact layout with minimal margins
#    - Color-coded rooms by type
#    - Room labels and areas
#    - Professional styling
