# utils/visualization.py
"""
Visualization Utilities: Generate SVG floor plans and graphs
- SVG floor plan generation from Layout objects
- Adjacency graph visualization
- Performance metrics charts
- Interactive HTML exports
"""

import logging
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Polygon, Point, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not available, some features will be limited")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, graph visualization disabled")


class SVGFloorPlanGenerator:
    """
    Generate SVG floor plans from Layout objects
    
    Features:
    - Room polygons with labels
    - Circulation paths
    - Color-coded by function
    - Dimension annotations
    - Legend
    """
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        scale: float = 20.0,  # pixels per meter
        margin: int = 50
    ):
        """
        Initialize SVG generator
        
        Args:
            width: SVG canvas width in pixels
            height: SVG canvas height in pixels
            scale: Scale factor (pixels per meter)
            margin: Margin around floor plan in pixels
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.margin = margin
        
        # Color palette for room types
        self.room_colors = {
            'bedroom': '#E8F4F8',
            'living': '#FFF4E6',
            'kitchen': '#F0F8E8',
            'bathroom': '#F8E8F4',
            'dining': '#FFF8E8',
            'circulation': '#F0F0F0',
            'storage': '#E8E8F0',
            'default': '#FFFFFF'
        }
        
        logger.info(f"✓ SVG Generator initialized ({width}x{height}px, scale={scale})")
    
    def generate_floor_plan(
        self,
        layout,
        title: str = "Floor Plan",
        show_dimensions: bool = True,
        show_circulation: bool = True,
        show_legend: bool = True
    ) -> str:
        """
        Generate complete SVG floor plan
        
        Args:
            layout: Layout object with rooms and circulation
            title: Plan title
            show_dimensions: Show room dimensions
            show_circulation: Show circulation paths
            show_legend: Show room legend
        
        Returns:
            SVG string
        """
        logger.info(f"Generating floor plan: {title}")
        
        # Create SVG root
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}'
        })
        
        # Add styles
        self._add_styles(svg)
        
        # Add title
        self._add_title(svg, title)
        
        # Calculate bounds and transform
        bounds = self._calculate_bounds(layout)
        transform = self._calculate_transform(bounds)
        
        # Create main group with transform
        main_group = ET.SubElement(svg, 'g', {
            'transform': transform
        })
        
        # Draw rooms
        if hasattr(layout, 'room_polygons') and layout.room_polygons:
            self._draw_rooms(main_group, layout.room_polygons, layout.rooms)
        elif hasattr(layout, 'rooms') and layout.rooms:
            self._draw_rooms_from_specs(main_group, layout.rooms)
        
        # Draw circulation paths
        if show_circulation and hasattr(layout, 'circulation_paths'):
            self._draw_circulation(main_group, layout.circulation_paths)
        
        # Add dimensions
        if show_dimensions and hasattr(layout, 'rooms'):
            self._add_dimensions(main_group, layout.rooms)
        
        # Add legend
        if show_legend and hasattr(layout, 'rooms'):
            self._add_legend(svg, layout.rooms)
        
        # Add metrics overlay
        self._add_metrics_overlay(svg, layout)
        
        return self._prettify_svg(svg)
    
    def _add_styles(self, svg: ET.Element):
        """Add CSS styles to SVG"""
        style = ET.SubElement(svg, 'style')
        style.text = """
            .room { 
                stroke: #333; 
                stroke-width: 2; 
                fill-opacity: 0.9;
            }
            .room:hover { 
                fill-opacity: 1.0; 
                stroke-width: 3;
            }
            .room-label { 
                font-family: Arial, sans-serif; 
                font-size: 14px; 
                text-anchor: middle; 
                fill: #333;
                font-weight: bold;
            }
            .room-area { 
                font-family: Arial, sans-serif; 
                font-size: 11px; 
                text-anchor: middle; 
                fill: #666;
            }
            .circulation { 
                stroke: #4A90E2; 
                stroke-width: 3; 
                stroke-dasharray: 8,4; 
                fill: none;
                opacity: 0.6;
            }
            .dimension-line { 
                stroke: #999; 
                stroke-width: 1; 
                fill: none;
            }
            .dimension-text { 
                font-family: Arial, sans-serif; 
                font-size: 10px; 
                fill: #666;
            }
            .title { 
                font-family: Arial, sans-serif; 
                font-size: 24px; 
                font-weight: bold; 
                fill: #333;
            }
            .legend-item { 
                font-family: Arial, sans-serif; 
                font-size: 12px; 
                fill: #333;
            }
        """
    
    def _add_title(self, svg: ET.Element, title: str):
        """Add title to SVG"""
        text = ET.SubElement(svg, 'text', {
            'x': str(self.width // 2),
            'y': '30',
            'class': 'title',
            'text-anchor': 'middle'
        })
        text.text = title
    
    def _calculate_bounds(self, layout) -> Dict[str, float]:
        """Calculate layout bounds"""
        if hasattr(layout, 'bounds') and layout.bounds:
            return layout.bounds
        
        # Calculate from rooms
        if hasattr(layout, 'room_polygons') and layout.room_polygons:
            all_coords = []
            for poly in layout.room_polygons.values():
                if SHAPELY_AVAILABLE:
                    bounds = poly.bounds
                    all_coords.extend([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
            
            if all_coords:
                xs = [c[0] for c in all_coords]
                ys = [c[1] for c in all_coords]
                return {
                    'min_x': min(xs),
                    'min_y': min(ys),
                    'max_x': max(xs),
                    'max_y': max(ys)
                }
        
        # Default bounds
        return {'min_x': 0, 'min_y': 0, 'max_x': 50, 'max_y': 50}
    
    def _calculate_transform(self, bounds: Dict[str, float]) -> str:
        """Calculate SVG transform to fit bounds"""
        # Calculate layout dimensions
        layout_width = bounds['max_x'] - bounds['min_x']
        layout_height = bounds['max_y'] - bounds['min_y']
        
        # Calculate available space
        available_width = self.width - 2 * self.margin
        available_height = self.height - 2 * self.margin - 100  # Extra for title
        
        # Calculate scale to fit
        scale_x = available_width / max(layout_width, 1)
        scale_y = available_height / max(layout_height, 1)
        scale = min(scale_x, scale_y, self.scale)
        
        # Calculate translation
        translate_x = self.margin - bounds['min_x'] * scale
        translate_y = self.margin + 50 - bounds['min_y'] * scale
        
        return f'translate({translate_x}, {translate_y}) scale({scale}, {scale})'
    
    def _draw_rooms(
        self,
        group: ET.Element,
        room_polygons: Dict,
        room_specs: Dict
    ):
        """Draw room polygons"""
        for room_id, polygon in room_polygons.items():
            if not SHAPELY_AVAILABLE:
                continue
            
            # Get room info
            room = room_specs.get(room_id)
            room_name = room.name if room else room_id
            room_type = self._get_room_type(room_name)
            color = self.room_colors.get(room_type, self.room_colors['default'])
            
            # Get polygon coordinates
            coords = list(polygon.exterior.coords)
            points = ' '.join([f'{x},{y}' for x, y in coords])
            
            # Draw polygon
            poly = ET.SubElement(group, 'polygon', {
                'points': points,
                'fill': color,
                'class': 'room'
            })
            poly.set('data-room-id', room_id)
            
            # Add label
            centroid = polygon.centroid
            label = ET.SubElement(group, 'text', {
                'x': str(centroid.x),
                'y': str(centroid.y - 0.5),
                'class': 'room-label'
            })
            label.text = room_name
            
            # Add area
            area_text = ET.SubElement(group, 'text', {
                'x': str(centroid.x),
                'y': str(centroid.y + 0.5),
                'class': 'room-area'
            })
            area_text.text = f'{polygon.area:.1f} m²'
    
    def _draw_rooms_from_specs(self, group: ET.Element, rooms: Dict):
        """Draw rooms from specifications (without polygons)"""
        x_offset = 0
        y_offset = 0
        max_height = 0
        
        for room_id, room in rooms.items():
            width = room.width if hasattr(room, 'width') else 5.0
            length = room.length if hasattr(room, 'length') else 5.0
            
            # Simple grid layout
            if x_offset + width > 30:
                x_offset = 0
                y_offset += max_height + 2
                max_height = 0
            
            room_type = self._get_room_type(room.name)
            color = self.room_colors.get(room_type, self.room_colors['default'])
            
            # Draw rectangle
            rect = ET.SubElement(group, 'rect', {
                'x': str(x_offset),
                'y': str(y_offset),
                'width': str(width),
                'height': str(length),
                'fill': color,
                'class': 'room'
            })
            
            # Add label
            label = ET.SubElement(group, 'text', {
                'x': str(x_offset + width/2),
                'y': str(y_offset + length/2),
                'class': 'room-label'
            })
            label.text = room.name
            
            x_offset += width + 2
            max_height = max(max_height, length)
    
    def _draw_circulation(self, group: ET.Element, circulation_paths: List):
        """Draw circulation paths"""
        for path in circulation_paths:
            if not hasattr(path, 'path_points'):
                continue
            
            points = ' '.join([f'{x},{y}' for x, y in path.path_points])
            
            polyline = ET.SubElement(group, 'polyline', {
                'points': points,
                'class': 'circulation'
            })
    
    def _add_dimensions(self, group: ET.Element, rooms: Dict):
        """Add dimension annotations"""
        # Simplified: just add overall dimensions
        pass
    
    def _add_legend(self, svg: ET.Element, rooms: Dict):
        """Add room legend"""
        legend_x = self.width - 200
        legend_y = 100
        
        # Legend background
        ET.SubElement(svg, 'rect', {
            'x': str(legend_x - 10),
            'y': str(legend_y - 10),
            'width': '190',
            'height': str(len(rooms) * 25 + 30),
            'fill': 'white',
            'stroke': '#ccc',
            'stroke-width': '1'
        })
        
        # Legend title
        title = ET.SubElement(svg, 'text', {
            'x': str(legend_x),
            'y': str(legend_y),
            'class': 'legend-item',
            'font-weight': 'bold'
        })
        title.text = 'Rooms'
        
        # Legend items
        y = legend_y + 20
        for room_id, room in rooms.items():
            room_type = self._get_room_type(room.name)
            color = self.room_colors.get(room_type, self.room_colors['default'])
            
            # Color box
            ET.SubElement(svg, 'rect', {
                'x': str(legend_x),
                'y': str(y - 10),
                'width': '15',
                'height': '15',
                'fill': color,
                'stroke': '#333'
            })
            
            # Room name
            text = ET.SubElement(svg, 'text', {
                'x': str(legend_x + 20),
                'y': str(y),
                'class': 'legend-item'
            })
            text.text = f'{room.name} ({room.area:.0f}m²)'
            
            y += 25
    
    def _add_metrics_overlay(self, svg: ET.Element, layout):
        """Add performance metrics overlay"""
        metrics_x = 20
        metrics_y = self.height - 120
        
        # Metrics background
        ET.SubElement(svg, 'rect', {
            'x': str(metrics_x),
            'y': str(metrics_y),
            'width': '250',
            'height': '100',
            'fill': 'white',
            'stroke': '#ccc',
            'stroke-width': '1',
            'opacity': '0.9'
        })
        
        # Metrics title
        title = ET.SubElement(svg, 'text', {
            'x': str(metrics_x + 10),
            'y': str(metrics_y + 20),
            'class': 'legend-item',
            'font-weight': 'bold'
        })
        title.text = 'Performance Metrics'
        
        # Add metrics
        y = metrics_y + 40
        metrics = [
            ('Total Area', f'{layout.total_area:.1f} m²' if hasattr(layout, 'total_area') else 'N/A'),
            ('Space Utilization', f'{layout.space_utilization_ratio*100:.1f}%' if hasattr(layout, 'space_utilization_ratio') else 'N/A'),
            ('Circulation Efficiency', f'{layout.circulation_efficiency*100:.1f}%' if hasattr(layout, 'circulation_efficiency') else 'N/A'),
            ('Compactness', f'{layout.compactness_score*100:.1f}%' if hasattr(layout, 'compactness_score') else 'N/A')
        ]
        
        for metric_name, metric_value in metrics[:3]:  # Show top 3
            text = ET.SubElement(svg, 'text', {
                'x': str(metrics_x + 10),
                'y': str(y),
                'class': 'dimension-text'
            })
            text.text = f'{metric_name}: {metric_value}'
            y += 18
    
    def _get_room_type(self, room_name: str) -> str:
        """Determine room type from name"""
        name_lower = room_name.lower()
        if 'bedroom' in name_lower or 'bed' in name_lower:
            return 'bedroom'
        elif 'living' in name_lower or 'lounge' in name_lower:
            return 'living'
        elif 'kitchen' in name_lower:
            return 'kitchen'
        elif 'bath' in name_lower or 'toilet' in name_lower:
            return 'bathroom'
        elif 'dining' in name_lower:
            return 'dining'
        elif 'hall' in name_lower or 'corridor' in name_lower:
            return 'circulation'
        elif 'storage' in name_lower or 'closet' in name_lower:
            return 'storage'
        else:
            return 'default'
    
    def _prettify_svg(self, elem: ET.Element) -> str:
        """Return prettified SVG string"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_to_file(self, svg_string: str, filename: str):
        """Save SVG to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.info(f"✓ Saved floor plan to: {filename}")


class AdjacencyGraphVisualizer:
    """Visualize adjacency graphs using NetworkX with native SVG generation"""
    
    def __init__(self, width: int = 800, height: int = 600):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for graph visualization")
        self.width = width
        self.height = height
        self.node_colors = {
            'bedroom': '#E8F4F8',
            'living': '#FFF4E6',
            'kitchen': '#F0F8E8',
            'bathroom': '#F8E8F4',
            'dining': '#FFF8E8',
            'circulation': '#F0F0F0',
            'storage': '#E8E8F0',
            'default': '#FFFFFF'
        }
        logger.info("✓ Adjacency Graph Visualizer initialized")
    
    def generate_adjacency_svg(
        self,
        adjacency_graph: nx.Graph,
        title: str = "Adjacency Graph",
        show_weights: bool = True,
        layout_type: str = "spring"
    ) -> str:
        """
        Generate SVG visualization of adjacency graph
        
        Args:
            adjacency_graph: NetworkX graph to visualize
            title: Graph title
            show_weights: Show edge weights/labels
            layout_type: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        
        Returns:
            SVG string
        """
        logger.info(f"Generating adjacency graph: {title}")
        
        if adjacency_graph.number_of_nodes() == 0:
            return self._generate_empty_graph_svg(title)
        
        # Calculate layout positions
        pos = self._calculate_layout(adjacency_graph, layout_type)
        
        # Normalize positions to SVG coordinates
        normalized_pos = self._normalize_positions(pos)
        
        # Create SVG
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}'
        })
        
        # Add styles
        self._add_graph_styles(svg)
        
        # Add title
        self._add_graph_title(svg, title, adjacency_graph)
        
        # Create main drawing group
        main_group = ET.SubElement(svg, 'g', {'id': 'graph-main'})
        
        # Draw edges first (so they appear behind nodes)
        self._draw_edges(main_group, adjacency_graph, normalized_pos, show_weights)
        
        # Draw nodes
        self._draw_nodes(main_group, adjacency_graph, normalized_pos)
        
        # Add legend
        self._add_graph_legend(svg, adjacency_graph)
        
        return self._prettify_svg(svg)
    
    def _calculate_layout(self, graph: nx.Graph, layout_type: str) -> Dict:
        """Calculate node positions using specified layout algorithm"""
        if layout_type == "spring":
            return nx.spring_layout(graph, k=2, iterations=50, seed=42)
        elif layout_type == "circular":
            return nx.circular_layout(graph)
        elif layout_type == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(graph)
            except:
                return nx.spring_layout(graph, seed=42)
        else:
            return nx.spring_layout(graph, seed=42)
    
    def _normalize_positions(self, pos: Dict) -> Dict:
        """Normalize graph positions to SVG coordinate space"""
        if not pos:
            return {}
        
        # Extract all coordinates
        coords = list(pos.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        # Calculate bounds
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Margins
        margin = 80
        legend_space = 200
        title_space = 60
        
        # Available drawing area
        draw_width = self.width - 2 * margin - legend_space
        draw_height = self.height - 2 * margin - title_space
        
        # Normalize to drawing area
        normalized = {}
        for node, (x, y) in pos.items():
            # Normalize to [0, 1]
            norm_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
            norm_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5
            
            # Scale to SVG coordinates
            svg_x = margin + norm_x * draw_width
            svg_y = margin + title_space + norm_y * draw_height
            
            normalized[node] = (svg_x, svg_y)
        
        return normalized
    
    def _add_graph_styles(self, svg: ET.Element):
        """Add CSS styles for graph elements"""
        style = ET.SubElement(svg, 'style')
        style.text = """
            .graph-node {
                stroke: #333;
                stroke-width: 2;
                cursor: pointer;
            }
            .graph-node:hover {
                stroke: #E74C3C;
                stroke-width: 3;
            }
            .graph-node-label {
                font-family: Arial, sans-serif;
                font-size: 12px;
                font-weight: bold;
                text-anchor: middle;
                fill: #333;
                pointer-events: none;
            }
            .graph-edge {
                stroke: #999;
                stroke-width: 2;
                fill: none;
            }
            .graph-edge-strong {
                stroke: #4A90E2;
                stroke-width: 3;
            }
            .graph-edge-label {
                font-family: Arial, sans-serif;
                font-size: 10px;
                fill: #666;
                text-anchor: middle;
            }
            .graph-title {
                font-family: Arial, sans-serif;
                font-size: 20px;
                font-weight: bold;
                fill: #333;
            }
            .graph-subtitle {
                font-family: Arial, sans-serif;
                font-size: 12px;
                fill: #666;
            }
            .graph-legend-title {
                font-family: Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
                fill: #333;
            }
            .graph-legend-item {
                font-family: Arial, sans-serif;
                font-size: 11px;
                fill: #333;
            }
        """
    
    def _add_graph_title(self, svg: ET.Element, title: str, graph: nx.Graph):
        """Add title and graph statistics"""
        # Main title
        title_elem = ET.SubElement(svg, 'text', {
            'x': str(self.width // 2),
            'y': '25',
            'class': 'graph-title',
            'text-anchor': 'middle'
        })
        title_elem.text = title
        
        # Statistics subtitle
        subtitle = ET.SubElement(svg, 'text', {
            'x': str(self.width // 2),
            'y': '45',
            'class': 'graph-subtitle',
            'text-anchor': 'middle'
        })
        subtitle.text = f'{graph.number_of_nodes()} rooms, {graph.number_of_edges()} adjacencies'
    
    def _draw_edges(
        self,
        group: ET.Element,
        graph: nx.Graph,
        pos: Dict,
        show_weights: bool
    ):
        """Draw graph edges"""
        for u, v, data in graph.edges(data=True):
            if u not in pos or v not in pos:
                continue
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Determine edge class based on weight/importance
            weight = data.get('weight', 1.0)
            edge_class = 'graph-edge-strong' if weight > 0.7 else 'graph-edge'
            
            # Draw edge line
            line = ET.SubElement(group, 'line', {
                'x1': str(x1),
                'y1': str(y1),
                'x2': str(x2),
                'y2': str(y2),
                'class': edge_class
            })
            
            # Add edge label if weights should be shown
            if show_weights and 'weight' in data:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                label = ET.SubElement(group, 'text', {
                    'x': str(mid_x),
                    'y': str(mid_y - 5),
                    'class': 'graph-edge-label'
                })
                label.text = f'{weight:.2f}'
    
    def _draw_nodes(self, group: ET.Element, graph: nx.Graph, pos: Dict):
        """Draw graph nodes"""
        for node, node_data in graph.nodes(data=True):
            if node not in pos:
                continue
            
            x, y = pos[node]
            
            # Determine node color
            room_name = node_data.get('name', str(node))
            room_type = self._get_room_type(room_name)
            color = self.node_colors.get(room_type, self.node_colors['default'])
            
            # Determine node size
            node_size = node_data.get('area', 20.0)
            radius = min(30, max(15, node_size * 0.3))
            
            # Draw node circle
            circle = ET.SubElement(group, 'circle', {
                'cx': str(x),
                'cy': str(y),
                'r': str(radius),
                'fill': color,
                'class': 'graph-node'
            })
            circle.set('data-node-id', str(node))
            
            # Add node label
            label = ET.SubElement(group, 'text', {
                'x': str(x),
                'y': str(y + 4),
                'class': 'graph-node-label'
            })
            label.text = room_name if len(room_name) < 15 else room_name[:12] + '...'
    
    def _add_graph_legend(self, svg: ET.Element, graph: nx.Graph):
        """Add legend showing room types"""
        legend_x = self.width - 180
        legend_y = 80
        
        # Get unique room types
        room_types = set()
        for node, data in graph.nodes(data=True):
            room_name = data.get('name', str(node))
            room_type = self._get_room_type(room_name)
            room_types.add(room_type)
        
        # Legend background
        legend_height = len(room_types) * 25 + 40
        ET.SubElement(svg, 'rect', {
            'x': str(legend_x - 10),
            'y': str(legend_y - 10),
            'width': '170',
            'height': str(legend_height),
            'fill': 'white',
            'stroke': '#ccc',
            'stroke-width': '1',
            'rx': '5'
        })
        
        # Legend title
        title = ET.SubElement(svg, 'text', {
            'x': str(legend_x),
            'y': str(legend_y + 5),
            'class': 'graph-legend-title'
        })
        title.text = 'Room Types'
        
        # Legend items
        y = legend_y + 25
        for room_type in sorted(room_types):
            color = self.node_colors.get(room_type, self.node_colors['default'])
            
            # Color circle
            ET.SubElement(svg, 'circle', {
                'cx': str(legend_x + 8),
                'cy': str(y - 3),
                'r': '6',
                'fill': color,
                'stroke': '#333',
                'stroke-width': '1'
            })
            
            # Type label
            text = ET.SubElement(svg, 'text', {
                'x': str(legend_x + 20),
                'y': str(y),
                'class': 'graph-legend-item'
            })
            text.text = room_type.title()
            
            y += 25
    
    def _get_room_type(self, room_name: str) -> str:
        """Determine room type from name"""
        name_lower = room_name.lower()
        if 'bedroom' in name_lower or 'bed' in name_lower:
            return 'bedroom'
        elif 'living' in name_lower or 'lounge' in name_lower:
            return 'living'
        elif 'kitchen' in name_lower:
            return 'kitchen'
        elif 'bath' in name_lower or 'toilet' in name_lower:
            return 'bathroom'
        elif 'dining' in name_lower:
            return 'dining'
        elif 'hall' in name_lower or 'corridor' in name_lower:
            return 'circulation'
        elif 'storage' in name_lower or 'closet' in name_lower:
            return 'storage'
        else:
            return 'default'
    
    def _generate_empty_graph_svg(self, title: str) -> str:
        """Generate SVG for empty graph"""
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(self.width),
            'height': str(self.height)
        })
        
        text = ET.SubElement(svg, 'text', {
            'x': str(self.width // 2),
            'y': str(self.height // 2),
            'text-anchor': 'middle',
            'font-size': '16',
            'fill': '#999'
        })
        text.text = f"{title}: No nodes to display"
        
        return ET.tostring(svg, encoding='unicode')
    
    def _prettify_svg(self, elem: ET.Element) -> str:
        """Return prettified SVG string"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_to_file(self, svg_string: str, filename: str):
        """Save adjacency graph SVG to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.info(f"✓ Saved adjacency graph to: {filename}")


def generate_html_report(
    layout,
    scores: Dict,
    svg_floor_plan: str,
    title: str = "Design Report"
) -> str:
    """
    Generate interactive HTML report with floor plan and metrics
    
    Args:
        layout: Layout object
        scores: Score dictionary
        svg_floor_plan: SVG floor plan string
        title: Report title
    
    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4A90E2;
                padding-bottom: 10px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #4A90E2;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                color: #333;
                margin-top: 5px;
            }}
            .floor-plan {{
                margin: 30px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Composite Score</div>
                    <div class="metric-value">{scores.get('composite', 0)*100:.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Functional Adequacy</div>
                    <div class="metric-value">{scores.get('functional_adequacy', 0)*100:.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Behavioral Performance</div>
                    <div class="metric-value">{scores.get('behavioral_performance', 0)*100:.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Layout Efficiency</div>
                    <div class="metric-value">{scores.get('layout_efficiency', 0)*100:.0f}%</div>
                </div>
            </div>
            
            <div class="floor-plan">
                {svg_floor_plan}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html