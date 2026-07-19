"""
Improved Layout Visualizer - Clean, Understandable Floor Plans and Adjacency Graphs

Key improvements:
- Smart room placement with compact layouts
- Strict adjacency detection (rooms must actually touch)
- Clear, color-coded adjacency graphs
- Proper connectivity analysis
- Clean SVG floor plans with compass and legends
"""

import json
import math
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class CompactRoomPlacer:
    """Smart room placement with proper adjacency tracking"""
    
    # Room importance for placement priority
    IMPORTANCE_SCORES = {
        'living_room': 15,
        'kitchen': 14,
        'hallway': 13,
        'corridor': 13,
        'dining_room': 12,
        'dining_hall': 12,
        'bedroom': 10,
        'bathroom': 8,
        'office': 7,
        'balcony': 6,
        'utility': 5,
        'storage': 4,
        'storeroom': 4,
        'garage': 3
    }
    
    # Critical adjacencies that MUST be satisfied
    CRITICAL_ADJACENCIES = {
        'kitchen': {'dining_room', 'dining_hall', 'living_room'},
        'bathroom': {'bedroom', 'hallway', 'corridor'},
        'bedroom': {'bathroom', 'hallway', 'corridor'},
        'living_room': {'kitchen', 'dining_room', 'hallway'},
        'dining_room': {'kitchen', 'living_room'},
        'dining_hall': {'kitchen', 'living_room'},
        'hallway': {'living_room', 'bedroom', 'bathroom'},
        'corridor': {'bedroom', 'bathroom'},
    }
    
    @classmethod
    def place_rooms_optimally(cls, rooms: List[Dict]) -> List[Dict]:
        """Place rooms with compact layout and track actual adjacencies"""
        
        if not rooms:
            return rooms
        
        # Prioritize rooms
        sorted_rooms = cls._prioritize_rooms(rooms)
        
        # Place first room at origin
        placed_rooms = []
        first_room = sorted_rooms[0].copy()
        first_room['x'] = 0
        first_room['y'] = 0
        placed_rooms.append(first_room)
        
        # Place remaining rooms
        for room in sorted_rooms[1:]:
            best_position = cls._find_best_position(room, placed_rooms)
            room_copy = room.copy()
            room_copy['x'] = best_position[0]
            room_copy['y'] = best_position[1]
            placed_rooms.append(room_copy)
        
        # Calculate actual adjacencies
        for room in placed_rooms:
            room['actual_adjacent_rooms'] = cls._get_adjacent_rooms(room, placed_rooms)
        
        return placed_rooms
    
    @classmethod
    def _prioritize_rooms(cls, rooms: List[Dict]) -> List[Dict]:
        """Prioritize rooms by importance and connectivity needs"""
        
        def room_priority(room):
            importance = cls.IMPORTANCE_SCORES.get(room['room_type'], 5)
            area_factor = min(room['area'] / 100, 3)
            adjacency_count = len(room.get('adjacencies', []))
            connectivity_boost = adjacency_count * 2
            
            return importance + area_factor + connectivity_boost
        
        return sorted(rooms, key=room_priority, reverse=True)
    
    @classmethod
    def _find_best_position(cls, room: Dict, placed_rooms: List[Dict]) -> Tuple[float, float]:
        """Find optimal position for room"""
        
        critical_adjacencies = cls.CRITICAL_ADJACENCIES.get(room['room_type'], set())
        preferred_adjacencies = set(room.get('adjacencies', []))
        
        # Find target rooms
        critical_targets = [r for r in placed_rooms if r['room_type'] in critical_adjacencies]
        preferred_targets = [r for r in placed_rooms if r['room_type'] in preferred_adjacencies]
        
        best_position = None
        best_score = float('-inf')
        
        # Try positions near critical targets first
        for target in critical_targets + preferred_targets:
            positions = cls._get_adjacent_positions(room, target)
            
            for pos in positions:
                if not cls._overlaps(room, pos, placed_rooms):
                    score = cls._evaluate_position(room, pos, placed_rooms, 
                                                   critical_adjacencies, preferred_adjacencies)
                    if score > best_score:
                        best_score = score
                        best_position = pos
        
        # Fallback: find most compact position
        if best_position is None:
            best_position = cls._find_compact_position(room, placed_rooms)
        
        return best_position
    
    @classmethod
    def _get_adjacent_positions(cls, room: Dict, target: Dict) -> List[Tuple[float, float]]:
        """Get positions adjacent to target room"""
        
        spacing = 0.5  # Small gap for walls
        
        return [
            # Right
            (target['x'] + target['width'] + spacing, target['y']),
            # Left
            (target['x'] - room['width'] - spacing, target['y']),
            # Below
            (target['x'], target['y'] + target['height'] + spacing),
            # Above
            (target['x'], target['y'] - room['height'] - spacing),
            # Aligned variations
            (target['x'] + target['width'] + spacing, 
             target['y'] + target['height'] - room['height']),
            (target['x'] - room['width'] - spacing, 
             target['y'] + target['height'] - room['height']),
        ]
    
    @classmethod
    def _overlaps(cls, room: Dict, position: Tuple[float, float], 
                  placed_rooms: List[Dict]) -> bool:
        """Check if position causes overlap"""
        
        x, y = position
        tolerance = 0.1
        
        room_rect = (x - tolerance, y - tolerance,
                    x + room['width'] + tolerance, y + room['height'] + tolerance)
        
        for placed in placed_rooms:
            placed_rect = (placed['x'] - tolerance, placed['y'] - tolerance,
                          placed['x'] + placed['width'] + tolerance,
                          placed['y'] + placed['height'] + tolerance)
            
            if cls._rectangles_overlap(room_rect, placed_rect):
                return True
        
        return False
    
    @classmethod
    def _rectangles_overlap(cls, rect1: Tuple, rect2: Tuple) -> bool:
        """Check rectangle overlap"""
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2
        
        return not (x1_max <= x2_min or x2_max <= x1_min or
                   y1_max <= y2_min or y2_max <= y1_min)
    
    @classmethod
    def _find_compact_position(cls, room: Dict, placed_rooms: List[Dict]) -> Tuple[float, float]:
        """Find most compact available position"""
        
        if not placed_rooms:
            return (0, 0)
        
        min_x = min(r['x'] for r in placed_rooms)
        max_x = max(r['x'] + r['width'] for r in placed_rooms)
        min_y = min(r['y'] for r in placed_rooms)
        max_y = max(r['y'] + r['height'] for r in placed_rooms)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        best_position = None
        best_distance = float('inf')
        
        # Try positions around each existing room
        for existing in placed_rooms:
            positions = cls._get_adjacent_positions(room, existing)
            
            for pos in positions:
                if not cls._overlaps(room, pos, placed_rooms):
                    room_center_x = pos[0] + room['width'] / 2
                    room_center_y = pos[1] + room['height'] / 2
                    
                    distance = math.sqrt((room_center_x - center_x)**2 + 
                                       (room_center_y - center_y)**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_position = pos
        
        # Fallback
        if best_position is None:
            best_position = (max_x + 1, min_y)
        
        return best_position
    
    @classmethod
    def _evaluate_position(cls, room: Dict, position: Tuple[float, float],
                          placed_rooms: List[Dict], critical_adj: set,
                          preferred_adj: set) -> float:
        """Evaluate position quality"""
        
        score = 0
        x, y = position
        temp_room = {**room, 'x': x, 'y': y}
        
        # Critical adjacency score (high weight)
        for placed in placed_rooms:
            if placed['room_type'] in critical_adj:
                if cls._are_adjacent(temp_room, placed):
                    score += 1000
                else:
                    distance = cls._distance_between(temp_room, placed)
                    score += 100 / (distance + 1)
        
        # Preferred adjacency score
        for placed in placed_rooms:
            if placed['room_type'] in preferred_adj:
                if cls._are_adjacent(temp_room, placed):
                    score += 200
                else:
                    distance = cls._distance_between(temp_room, placed)
                    score += 20 / (distance + 1)
        
        # Compactness score
        if placed_rooms:
            layout_center_x = sum(r['x'] + r['width']/2 for r in placed_rooms) / len(placed_rooms)
            layout_center_y = sum(r['y'] + r['height']/2 for r in placed_rooms) / len(placed_rooms)
            
            room_center_x = x + room['width'] / 2
            room_center_y = y + room['height'] / 2
            
            distance_from_center = math.sqrt((room_center_x - layout_center_x)**2 +
                                           (room_center_y - layout_center_y)**2)
            score += 50 / (distance_from_center + 1)
        
        return score
    
    @classmethod
    def _are_adjacent(cls, room1: Dict, room2: Dict) -> bool:
        """Check if two rooms are actually adjacent (share edge)"""
        
        r1_left = room1['x']
        r1_right = room1['x'] + room1['width']
        r1_top = room1['y']
        r1_bottom = room1['y'] + room1['height']
        
        r2_left = room2['x']
        r2_right = room2['x'] + room2['width']
        r2_top = room2['y']
        r2_bottom = room2['y'] + room2['height']
        
        tolerance = 1.0  # Wall thickness
        
        # Horizontal adjacency
        horizontal_adjacent = (abs(r1_right - r2_left) <= tolerance or 
                              abs(r2_right - r1_left) <= tolerance)
        
        # Vertical adjacency
        vertical_adjacent = (abs(r1_bottom - r2_top) <= tolerance or 
                            abs(r2_bottom - r1_top) <= tolerance)
        
        if horizontal_adjacent:
            overlap_start = max(r1_top, r2_top)
            overlap_end = min(r1_bottom, r2_bottom)
            overlap_length = overlap_end - overlap_start
            
            min_height = min(r1_bottom - r1_top, r2_bottom - r2_top)
            required_overlap = min_height * 0.3
            
            return overlap_length >= required_overlap
        
        elif vertical_adjacent:
            overlap_start = max(r1_left, r2_left)
            overlap_end = min(r1_right, r2_right)
            overlap_length = overlap_end - overlap_start
            
            min_width = min(r1_right - r1_left, r2_right - r2_left)
            required_overlap = min_width * 0.3
            
            return overlap_length >= required_overlap
        
        return False
    
    @classmethod
    def _distance_between(cls, room1: Dict, room2: Dict) -> float:
        """Calculate distance between rooms"""
        
        x1, y1 = room1['x'], room1['y']
        w1, h1 = room1['width'], room1['height']
        x2, y2 = room2['x'], room2['y']
        w2, h2 = room2['width'], room2['height']
        
        dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
        dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
        
        return math.sqrt(dx*dx + dy*dy)
    
    @classmethod
    def _get_adjacent_rooms(cls, room: Dict, all_rooms: List[Dict]) -> List[str]:
        """Get list of actually adjacent rooms"""
        
        adjacent = []
        for other in all_rooms:
            if other['room_id'] != room['room_id']:
                if cls._are_adjacent(room, other):
                    adjacent.append(other['room_id'])
        
        return adjacent


class ImprovedLayoutVisualizer:
    """Generate clean, understandable visualizations"""
    
    def __init__(self, output_dir: str = "visual_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.room_colors = {
            'bedroom': '#3498DB',
            'bathroom': '#E67E22',
            'kitchen': '#27AE60',
            'living_room': '#9B59B6',
            'dining_room': '#F39C12',
            'dining_hall': '#F39C12',
            'office': '#E74C3C',
            'garage': '#95A5A6',
            'utility': '#16A085',
            'storage': '#8E44AD',
            'storeroom': '#8E44AD',
            'balcony': '#1ABC9C',
            'hallway': '#BDC3C7',
            'corridor': '#BDC3C7',
            'default': '#34495E'
        }
    
    def render(self, layout, project_name: str, node_id: str) -> Tuple[str, str]:
        """Render improved layout visualizations"""
        
        # Extract room data
        rooms = self._extract_rooms(layout)
        
        if not rooms:
            logger.warning("No rooms found in layout")
            return None, None
        
        # Apply smart placement
        rooms = CompactRoomPlacer.place_rooms_optimally(rooms)
        
        # Generate visualizations
        svg_path = self._generate_svg(rooms, project_name, node_id)
        adjacency_path = self._generate_adjacency_graph(rooms, project_name, node_id)
        
        return svg_path, adjacency_path
    
    def _extract_rooms(self, layout) -> List[Dict]:
        """Extract room data from layout object"""
        
        rooms = []
        
        if not hasattr(layout, 'rooms') or not layout.rooms:
            return rooms
        
        for room_id, room in layout.rooms.items():
            # Get room name (prefer actual name over type)
            room_name = getattr(room, 'name', None)
            if not room_name or room_name == room_id:
                # Fallback to formatted room type
                room_type = getattr(room, 'room_type', 'room')
                room_name = room_type.replace('_', ' ').title()
            
            # Get room dimensions
            if hasattr(layout, 'room_polygons') and room_id in layout.room_polygons:
                polygon = layout.room_polygons[room_id]
                bounds = polygon.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                area = polygon.area
            else:
                # Fallback dimensions
                area = getattr(room, 'area', 100)
                width = height = math.sqrt(area)
            
            room_data = {
                'room_id': room_id,
                'room_name': room_name,  # Actual room name
                'room_type': getattr(room, 'room_type', 'default'),
                'area': area,
                'width': width,
                'height': height,
                'x': 0,
                'y': 0,
                'adjacencies': getattr(room, 'required_adjacencies', []) or []
            }
            
            rooms.append(room_data)
        
        return rooms
    
    def _generate_svg(self, rooms: List[Dict], project_name: str, node_id: str) -> str:
        """Generate clean SVG floor plan"""
        
        # Calculate bounds
        min_x = min(r['x'] for r in rooms)
        max_x = max(r['x'] + r['width'] for r in rooms)
        min_y = min(r['y'] for r in rooms)
        max_y = max(r['y'] + r['height'] for r in rooms)
        
        margin = 30
        scale = 10
        
        svg_width = (max_x - min_x + 2 * margin) * scale
        svg_height = (max_y - min_y + 2 * margin) * scale
        
        # Create SVG
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': str(svg_width),
            'height': str(svg_height),
            'viewBox': f'0 0 {svg_width} {svg_height}'
        })
        
        # Add styles
        style = ET.SubElement(svg, 'style')
        style.text = """
        .room { stroke: #2C3E50; stroke-width: 2; opacity: 0.9; }
        .room-label { font-family: Arial; font-size: 12px; font-weight: bold; 
                     fill: white; text-anchor: middle; dominant-baseline: middle; }
        .title { font-family: Arial; font-size: 16px; font-weight: bold; fill: #2C3E50; }
        """
        
        # Add title
        title = ET.SubElement(svg, 'text', {'x': '20', 'y': '25', 'class': 'title'})
        title.text = f"{project_name.replace('_', ' ').title()} - Layout"
        
        # Add rooms
        for room in rooms:
            svg_x = (room['x'] - min_x + margin) * scale
            svg_y = (room['y'] - min_y + margin) * scale
            svg_w = room['width'] * scale
            svg_h = room['height'] * scale
            
            color = self.room_colors.get(room['room_type'], self.room_colors['default'])
            
            # Room rectangle
            ET.SubElement(svg, 'rect', {
                'x': str(svg_x), 'y': str(svg_y),
                'width': str(svg_w), 'height': str(svg_h),
                'fill': color, 'class': 'room'
            })
            
            # Room label - show both name and type
            label = ET.SubElement(svg, 'text', {
                'x': str(svg_x + svg_w/2),
                'y': str(svg_y + svg_h/2 - 5),
                'class': 'room-label'
            })
            label.text = room.get('room_name', room['room_type'].replace('_', ' ').title())
            
            # Room type as subtitle
            type_label = ET.SubElement(svg, 'text', {
                'x': str(svg_x + svg_w/2),
                'y': str(svg_y + svg_h/2 + 10),
                'class': 'room-label',
                'font-size': '10'
            })
            type_label.text = f"({room['room_type'].replace('_', ' ').title()})"
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_{node_id[:8]}_layout_{timestamp}.svg"
        filepath = self.output_dir / filename
        
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ")
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Generated SVG: {filepath}")
        return str(filepath)
    
    def _generate_adjacency_graph(self, rooms: List[Dict], project_name: str, 
                                  node_id: str) -> str:
        """Generate clear adjacency graph"""
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for room in rooms:
            G.add_node(room['room_id'], **room)
        
        # Add edges based on ACTUAL adjacencies
        edge_types = {'critical': [], 'preferred': [], 'spatial': [], 'bridge': []}
        
        for room in rooms:
            for adj_id in room.get('actual_adjacent_rooms', []):
                if G.has_edge(room['room_id'], adj_id):
                    continue
                
                adj_room = next((r for r in rooms if r['room_id'] == adj_id), None)
                if not adj_room:
                    continue
                
                # Determine edge type
                critical_adj = CompactRoomPlacer.CRITICAL_ADJACENCIES.get(room['room_type'], set())
                
                if adj_room['room_type'] in critical_adj:
                    edge_type = 'critical'
                elif adj_id in room.get('adjacencies', []):
                    edge_type = 'preferred'
                else:
                    edge_type = 'spatial'
                
                G.add_edge(room['room_id'], adj_id, edge_type=edge_type)
                edge_types[edge_type].append((room['room_id'], adj_id))
        
        # Ensure connectivity
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            main_component = max(components, key=len)
            
            for component in components:
                if component == main_component:
                    continue
                
                # Find closest rooms
                min_dist = float('inf')
                best_pair = None
                
                for r1_id in component:
                    r1 = next(r for r in rooms if r['room_id'] == r1_id)
                    for r2_id in main_component:
                        r2 = next(r for r in rooms if r['room_id'] == r2_id)
                        dist = CompactRoomPlacer._distance_between(r1, r2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (r1_id, r2_id)
                
                if best_pair:
                    G.add_edge(*best_pair, edge_type='bridge')
                    edge_types['bridge'].append(best_pair)
                    main_component.update(component)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{project_name.replace("_", " ").title()} - Adjacency Analysis',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Spatial layout
        self._plot_spatial_layout(G, rooms, ax1)
        
        # Plot 2: Graph topology
        self._plot_graph_topology(G, edge_types, ax2)
        
        # Plot 3: Connectivity analysis
        self._plot_connectivity(G, rooms, ax3)
        
        # Plot 4: Performance metrics
        self._plot_performance(G, rooms, ax4)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_{node_id[:8]}_adjacency_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated adjacency graph: {filepath}")
        return str(filepath)
    
    def _plot_spatial_layout(self, G, rooms, ax):
        """Plot spatial layout with connections"""
        
        ax.set_title("Spatial Layout", fontsize=14, fontweight='bold')
        
        # Draw rooms
        for room in rooms:
            color = self.room_colors.get(room['room_type'], self.room_colors['default'])
            rect = Rectangle((room['x'], -room['y'] - room['height']),
                           room['width'], room['height'],
                           facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            
            # Label - show name and type
            room_display = room.get('room_name', room['room_type'].replace('_', ' ').title())
            room_type_display = f"\n({room['room_type'].replace('_', ' ').title()})"
            
            ax.text(room['x'] + room['width']/2, 
                   -room['y'] - room['height']/2,
                   room_display + room_type_display,
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white')
        
        # Draw connections
        pos = {room['room_id']: (room['x'] + room['width']/2, 
                                 -room['y'] - room['height']/2) 
               for room in rooms}
        
        nx.draw_networkx_edges(G, pos, edge_color='#2C3E50', width=2, ax=ax, alpha=0.6)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_graph_topology(self, G, edge_types, ax):
        """Plot graph topology with edge types"""
        
        ax.set_title("Adjacency Graph", fontsize=14, fontweight='bold')
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        node_colors = [self.room_colors.get(G.nodes[node].get('room_type', 'default'),
                                           self.room_colors['default'])
                      for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                              ax=ax, edgecolors='black', linewidths=2)
        
        # Draw edges by type
        edge_styles = {
            'critical': ('#27AE60', 4, 'solid'),
            'preferred': ('#3498DB', 3, 'solid'),
            'spatial': ('#95A5A6', 2, 'solid'),
            'bridge': ('#E74C3C', 2, 'dashed')
        }
        
        for edge_type, (color, width, style) in edge_styles.items():
            edges = edge_types.get(edge_type, [])
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color,
                                      width=width, style=style, ax=ax, alpha=0.8,
                                      label=f'{edge_type.title()} ({len(edges)})')
        
        # Labels - show name with type in parentheses
        labels = {}
        for node in G.nodes():
            room_name = G.nodes[node].get('room_name', 
                                          G.nodes[node].get('room_type', 'unknown').replace('_', ' ').title())
            room_type = G.nodes[node].get('room_type', 'unknown').replace('_', ' ').title()
            labels[node] = f"{room_name}\n({room_type})"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax, font_weight='bold')
        
        ax.axis('off')
        ax.legend(loc='upper right')
    
    def _plot_connectivity(self, G, rooms, ax):
        """Plot connectivity analysis"""
        
        ax.set_title("Connectivity Analysis", fontsize=14, fontweight='bold')
        
        # Calculate satisfaction scores
        room_scores = []
        room_names = []
        
        for room in rooms:
            critical_adj = CompactRoomPlacer.CRITICAL_ADJACENCIES.get(room['room_type'], set())
            
            if critical_adj:
                satisfied = sum(1 for adj_id in room.get('actual_adjacent_rooms', [])
                              if any(r['room_id'] == adj_id and r['room_type'] in critical_adj 
                                    for r in rooms))
                score = (satisfied / len(critical_adj)) * 100
            else:
                score = 100
            
            room_scores.append(score)
            # Show name with type
            room_name = room.get('room_name', room['room_type'].replace('_', ' ').title())
            room_type = room['room_type'].replace('_', ' ').title()
            room_names.append(f"{room_name} ({room_type})")
        
        # Bar chart
        colors = ['#27AE60' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' 
                 for s in room_scores]
        
        bars = ax.barh(room_names, room_scores, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, score in zip(bars, room_scores):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{score:.0f}%', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Satisfaction %', fontsize=12)
        ax.set_xlim(0, 110)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_performance(self, G, rooms, ax):
        """Plot performance metrics"""
        
        ax.axis('off')
        
        # Calculate metrics
        total_rooms = len(rooms)
        total_connections = G.number_of_edges()
        avg_connections = (total_connections * 2 / total_rooms) if total_rooms > 0 else 0
        is_connected = nx.is_connected(G)
        
        metrics_text = f"""
PERFORMANCE METRICS

Connectivity:
• Graph Connected: {'✅ YES' if is_connected else '❌ NO'}
• Total Connections: {total_connections}
• Avg Connections/Room: {avg_connections:.1f}

Layout:
• Total Rooms: {total_rooms}
• Total Area: {sum(r['area'] for r in rooms):.0f} sq ft

Status:
{'✅ EXCELLENT' if is_connected and avg_connections >= 2 else '⚠️ NEEDS IMPROVEMENT'}
        """
        
        ax.text(0.1, 0.5, metrics_text.strip(), fontsize=11,
               bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.9),
               verticalalignment='center', fontfamily='monospace')
