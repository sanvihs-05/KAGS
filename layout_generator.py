"""
Fixed FBS Architectural Layout Generator - Proper Adjacency Graph Generation
Fixes disconnected adjacency graphs and ensures proper connectivity visualization
"""

import json
import math
import os
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, Arrow, FancyArrowPatch
import numpy as np
from pathlib import Path

# SVG generation imports
import xml.etree.ElementTree as ET
from xml.dom import minidom

@dataclass
class VisualConfig:
    """Configuration for visual generation"""
    # Colors
    wall_color: str = "#2C3E50"
    wall_width: float = 0.2
    
    # Room colors by type
    room_colors: Dict[str, str] = None
    
    # Text styling
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16
    
    # Layout
    margin: float = 50
    scale: float = 10  # pixels per foot
    
    # Grid
    show_grid: bool = True
    grid_color: str = "#ECF0F1"
    grid_spacing: float = 5  # feet
    
    # Dimensions
    show_dimensions: bool = True
    dimension_color: str = "#E74C3C"
    
    # Directional indicators
    show_directions: bool = True
    compass_size: float = 40
    direction_arrow_color: str = "#F39C12"
    sun_color: str = "#F1C40F"
    wind_color: str = "#3498DB"
    
    # Adjacency graph
    node_size: float = 1000
    edge_width: float = 2
    node_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.room_colors is None:
            self.room_colors = {
                'bedroom': '#3498DB',      # Blue
                'bathroom': '#E67E22',     # Orange
                'kitchen': '#27AE60',      # Green
                'living_room': '#9B59B6',  # Purple
                'dining_room': '#F39C12',  # Yellow
                'dining_hall': '#F39C12',  # Yellow
                'office': '#E74C3C',       # Red
                'garage': '#95A5A6',       # Gray
                'utility': '#16A085',      # Teal
                'storage': '#8E44AD',      # Purple
                'storeroom': '#8E44AD',    # Purple
                'balcony': '#1ABC9C',      # Turquoise
                'hallway': '#BDC3C7',      # Light Gray
                'corridor': '#BDC3C7',     # Light Gray
                'default': '#34495E'       # Dark Gray
            }
        
        if self.node_colors is None:
            self.node_colors = self.room_colors.copy()

class CompactRoomPlacer:
    """Improved room placement with compact layout and proper adjacency tracking"""
    
    @staticmethod
    def place_rooms_optimally(rooms: List[Dict]) -> List[Dict]:
        """Place rooms with compact layout and track actual adjacencies"""
        
        if not rooms:
            return rooms
        
        # Sort rooms by priority (key rooms first, then by area)
        sorted_rooms = CompactRoomPlacer._prioritize_rooms_smartly(rooms)
        
        # Start with the most important room at origin
        placed_rooms = []
        first_room = sorted_rooms[0].copy()
        first_room['x'] = 0
        first_room['y'] = 0
        placed_rooms.append(first_room)
        
        # Place remaining rooms with compact strategy
        for room in sorted_rooms[1:]:
            best_position = CompactRoomPlacer._find_compact_position(room, placed_rooms)
            room_copy = room.copy()
            room_copy['x'] = best_position[0]
            room_copy['y'] = best_position[1]
            placed_rooms.append(room_copy)
        
        # Calculate actual adjacencies based on final positions
        for room in placed_rooms:
            room['actual_adjacent_rooms'] = CompactRoomPlacer._get_actual_adjacent_rooms(room, placed_rooms)
        
        return placed_rooms
    
    @staticmethod
    def _get_actual_adjacent_rooms(room: Dict, all_rooms: List[Dict]) -> List[str]:
        """Get list of rooms that are actually adjacent to this room"""
        adjacent_rooms = []
        
        for other_room in all_rooms:
            if other_room['room_id'] != room['room_id']:
                if CompactRoomPlacer._are_rooms_adjacent(room, other_room):
                    adjacent_rooms.append(other_room['room_id'])
        
        return adjacent_rooms
    
    @staticmethod
    def _prioritize_rooms_smartly(rooms: List[Dict]) -> List[Dict]:
        """Smart prioritization focusing on key rooms and connectivity"""
        
        # Define room importance with connectivity focus
        importance_scores = {
            'living_room': 15,      # Highest - central hub
            'kitchen': 14,          # High - needs dining/living connection
            'hallway': 13,          # High - connectivity essential
            'corridor': 13,         # High - connectivity essential
            'dining_room': 12,      # High - needs kitchen/living
            'dining_hall': 12,      # High - needs kitchen/living
            'bedroom': 10,          # Medium-high - needs bathroom/hallway
            'bathroom': 8,          # Medium - needs bedroom/hallway
            'office': 7,            # Medium - flexible placement
            'balcony': 6,           # Medium-low - needs living/bedroom
            'utility': 5,           # Low - flexible but near kitchen
            'storage': 4,           # Low - flexible
            'storeroom': 4,         # Low - flexible
            'garage': 3             # Lowest - often separate
        }
        
        def room_priority(room):
            importance = importance_scores.get(room['room_type'], 5)
            area_factor = min(room['area'] / 100, 3)  # Cap area influence
            adjacency_count = len(room.get('adjacencies', []))
            
            # Boost score for rooms with many desired adjacencies
            connectivity_boost = adjacency_count * 2
            
            return importance + area_factor + connectivity_boost
        
        return sorted(rooms, key=room_priority, reverse=True)
    
    @staticmethod
    def _find_compact_position(room: Dict, placed_rooms: List[Dict]) -> Tuple[float, float]:
        """Find position that maximizes compactness while respecting critical adjacencies"""
        
        # Get critical adjacencies (must-have connections)
        critical_adjacencies = CompactRoomPlacer._get_critical_adjacencies(room)
        preferred_adjacencies = set(room.get('adjacencies', []))
        
        # Find target rooms for critical adjacencies
        critical_targets = [r for r in placed_rooms 
                          if r['room_type'] in critical_adjacencies]
        
        # Find target rooms for preferred adjacencies
        preferred_targets = [r for r in placed_rooms 
                           if r['room_type'] in preferred_adjacencies]
        
        best_position = None
        best_score = float('-inf')
        
        # Strategy 1: Try positions near critical adjacency targets
        if critical_targets:
            for target in critical_targets:
                positions = CompactRoomPlacer._get_adjacent_positions_compact(room, target)
                for pos in positions:
                    if not CompactRoomPlacer._position_overlaps(room, pos, placed_rooms):
                        score = CompactRoomPlacer._evaluate_position_comprehensive(
                            room, pos, placed_rooms, critical_adjacencies, preferred_adjacencies)
                        if score > best_score:
                            best_score = score
                            best_position = pos
        
        # Strategy 2: Try positions near preferred targets if no critical position found
        if best_position is None and preferred_targets:
            for target in preferred_targets:
                positions = CompactRoomPlacer._get_adjacent_positions_compact(room, target)
                for pos in positions:
                    if not CompactRoomPlacer._position_overlaps(room, pos, placed_rooms):
                        score = CompactRoomPlacer._evaluate_position_comprehensive(
                            room, pos, placed_rooms, critical_adjacencies, preferred_adjacencies)
                        if score > best_score:
                            best_score = score
                            best_position = pos
        
        # Strategy 3: Find most compact available position
        if best_position is None:
            best_position = CompactRoomPlacer._find_most_compact_position(room, placed_rooms)
        
        return best_position
    
    @staticmethod
    def _get_critical_adjacencies(room: Dict) -> set:
        """Define critical adjacencies that should be prioritized"""
        
        room_type = room['room_type']
        
        critical_rules = {
            'kitchen': {'dining_room', 'dining_hall', 'living_room'},  # Kitchen MUST connect to social areas
            'bathroom': {'bedroom', 'hallway', 'corridor'},            # Bathroom MUST connect to access
            'bedroom': {'bathroom', 'hallway', 'corridor'},            # Bedroom MUST have access
            'living_room': {'kitchen', 'dining_room', 'hallway'},      # Living room MUST be central
            'dining_room': {'kitchen', 'living_room'},                 # Dining MUST connect to kitchen/living
            'dining_hall': {'kitchen', 'living_room'},                 # Dining hall MUST connect to kitchen/living
            'hallway': {'living_room', 'bedroom', 'bathroom'},         # Hallway MUST connect key areas
            'corridor': {'bedroom', 'bathroom'},                       # Corridor MUST connect bedrooms/baths
        }
        
        return critical_rules.get(room_type, set())
    
    @staticmethod
    def _get_adjacent_positions_compact(room: Dict, target_room: Dict) -> List[Tuple[float, float]]:
        """Get compact adjacent positions with minimal spacing"""
        
        positions = []
        spacing = 0.5  # Small gap between rooms for walls
        
        # Primary positions (direct adjacency)
        positions.extend([
            # Right of target
            (target_room['x'] + target_room['width'] + spacing, target_room['y']),
            # Left of target  
            (target_room['x'] - room['width'] - spacing, target_room['y']),
            # Below target
            (target_room['x'], target_room['y'] + target_room['height'] + spacing),
            # Above target
            (target_room['x'], target_room['y'] - room['height'] - spacing),
        ])
        
        # Secondary positions (aligned edges for better adjacency)
        positions.extend([
            # Right of target, aligned with bottom
            (target_room['x'] + target_room['width'] + spacing, 
             target_room['y'] + target_room['height'] - room['height']),
            # Left of target, aligned with bottom
            (target_room['x'] - room['width'] - spacing, 
             target_room['y'] + target_room['height'] - room['height']),
            # Below target, aligned with right
            (target_room['x'] + target_room['width'] - room['width'], 
             target_room['y'] + target_room['height'] + spacing),
            # Above target, aligned with right
            (target_room['x'] + target_room['width'] - room['width'], 
             target_room['y'] - room['height'] - spacing),
        ])
        
        return positions
    
    @staticmethod
    def _position_overlaps(room: Dict, position: Tuple[float, float], placed_rooms: List[Dict]) -> bool:
        """Check overlap with minimal tolerance for walls"""
        
        x, y = position
        tolerance = 0.1  # Small tolerance for wall thickness
        
        room_rect = (x - tolerance, y - tolerance, 
                    x + room['width'] + tolerance, y + room['height'] + tolerance)
        
        for placed_room in placed_rooms:
            placed_rect = (
                placed_room['x'] - tolerance, 
                placed_room['y'] - tolerance,
                placed_room['x'] + placed_room['width'] + tolerance,
                placed_room['y'] + placed_room['height'] + tolerance
            )
            
            if CompactRoomPlacer._rectangles_overlap(room_rect, placed_rect):
                return True
        
        return False
    
    @staticmethod
    def _rectangles_overlap(rect1: Tuple[float, float, float, float], 
                           rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap"""
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2
        
        return not (x1_max <= x2_min or x2_max <= x1_min or 
                   y1_max <= y2_min or y2_max <= y1_min)
    
    @staticmethod
    def _find_most_compact_position(room: Dict, placed_rooms: List[Dict]) -> Tuple[float, float]:
        """Find the most compact available position using fine-grained search"""
        
        if not placed_rooms:
            return (0, 0)
        
        # Calculate tight bounds
        min_x = min(r['x'] for r in placed_rooms)
        max_x = max(r['x'] + r['width'] for r in placed_rooms)
        min_y = min(r['y'] for r in placed_rooms)
        max_y = max(r['y'] + r['height'] for r in placed_rooms)
        
        # Try positions around existing rooms first
        best_position = None
        best_compactness = float('inf')
        
        # Generate candidate positions around each existing room
        for existing_room in placed_rooms:
            positions = CompactRoomPlacer._get_adjacent_positions_compact(room, existing_room)
            
            for pos in positions:
                if not CompactRoomPlacer._position_overlaps(room, pos, placed_rooms):
                    # Calculate compactness (distance from layout center)
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    room_center_x = pos[0] + room['width'] / 2
                    room_center_y = pos[1] + room['height'] / 2
                    
                    distance = math.sqrt((room_center_x - center_x)**2 + (room_center_y - center_y)**2)
                    
                    if distance < best_compactness:
                        best_compactness = distance
                        best_position = pos
        
        # Fallback: place to the right if no good position found
        if best_position is None:
            best_position = (max_x + 1, min_y)
        
        return best_position
    
    @staticmethod
    def _evaluate_position_comprehensive(room: Dict, position: Tuple[float, float], 
                                       placed_rooms: List[Dict], critical_adjacencies: set,
                                       preferred_adjacencies: set) -> float:
        """Comprehensive position evaluation"""
        
        score = 0
        x, y = position
        
        # Critical adjacency score (high weight)
        for placed_room in placed_rooms:
            if placed_room['room_type'] in critical_adjacencies:
                # Check if this position would create adjacency
                temp_room = {**room, 'x': x, 'y': y}
                if CompactRoomPlacer._are_rooms_adjacent(temp_room, placed_room):
                    score += 1000  # Very high score for critical adjacencies
                else:
                    distance = CompactRoomPlacer._distance_between_rooms(
                        (x, y, room['width'], room['height']),
                        (placed_room['x'], placed_room['y'], placed_room['width'], placed_room['height'])
                    )
                    score += 100 / (distance + 1)
        
        # Preferred adjacency score (medium weight)
        for placed_room in placed_rooms:
            if placed_room['room_type'] in preferred_adjacencies:
                temp_room = {**room, 'x': x, 'y': y}
                if CompactRoomPlacer._are_rooms_adjacent(temp_room, placed_room):
                    score += 200  # Medium score for preferred adjacencies
                else:
                    distance = CompactRoomPlacer._distance_between_rooms(
                        (x, y, room['width'], room['height']),
                        (placed_room['x'], placed_room['y'], placed_room['width'], placed_room['height'])
                    )
                    score += 20 / (distance + 1)
        
        # Compactness score (layout efficiency)
        if placed_rooms:
            layout_center_x = sum(r['x'] + r['width']/2 for r in placed_rooms) / len(placed_rooms)
            layout_center_y = sum(r['y'] + r['height']/2 for r in placed_rooms) / len(placed_rooms)
            
            room_center_x = x + room['width'] / 2
            room_center_y = y + room['height'] / 2
            
            distance_from_center = math.sqrt((room_center_x - layout_center_x)**2 + 
                                           (room_center_y - layout_center_y)**2)
            score += 50 / (distance_from_center + 1)
        
        return score
    
    @staticmethod
    def _distance_between_rooms(room1_bounds: Tuple[float, float, float, float],
                               room2_bounds: Tuple[float, float, float, float]) -> float:
        """Calculate minimum distance between two rooms"""
        
        x1, y1, w1, h1 = room1_bounds
        x2, y2, w2, h2 = room2_bounds
        
        # Calculate edge-to-edge distance
        dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
        dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
        
        return math.sqrt(dx*dx + dy*dy)
    
    @staticmethod
    def _are_rooms_adjacent(room1: Dict, room2: Dict) -> bool:
        """Check if two rooms are adjacent - must share a common edge"""
        
        # Room 1 bounds
        r1_left = room1['x']
        r1_right = room1['x'] + room1['width']
        r1_top = room1['y']
        r1_bottom = room1['y'] + room1['height']
        
        # Room 2 bounds  
        r2_left = room2['x']
        r2_right = room2['x'] + room2['width']
        r2_top = room2['y']
        r2_bottom = room2['y'] + room2['height']
        
        # Very strict tolerance - rooms must actually touch or be separated by thin walls only
        tolerance = 1.0  # feet (for wall thickness)
        
        # Check for horizontal adjacency (rooms side by side)
        horizontal_adjacent = (
            (abs(r1_right - r2_left) <= tolerance or abs(r2_right - r1_left) <= tolerance)
        )
        
        # Check for vertical adjacency (rooms above/below each other)  
        vertical_adjacent = (
            (abs(r1_bottom - r2_top) <= tolerance or abs(r2_bottom - r1_top) <= tolerance)
        )
        
        if horizontal_adjacent:
            # Must have significant vertical overlap to be truly adjacent
            overlap_start = max(r1_top, r2_top)
            overlap_end = min(r1_bottom, r2_bottom)
            overlap_length = overlap_end - overlap_start
            
            # Require at least 30% overlap of the smaller room's height
            min_height = min(r1_bottom - r1_top, r2_bottom - r2_top)
            required_overlap = min_height * 0.3
            
            return overlap_length >= required_overlap
        
        elif vertical_adjacent:
            # Must have significant horizontal overlap to be truly adjacent  
            overlap_start = max(r1_left, r2_left)
            overlap_end = min(r1_right, r2_right)
            overlap_length = overlap_end - overlap_start
            
            # Require at least 30% overlap of the smaller room's width
            min_width = min(r1_right - r1_left, r2_right - r2_left)
            required_overlap = min_width * 0.3
            
            return overlap_length >= required_overlap
        
        return False


class DataProcessor:
    """Process the input JSON data to extract layout information"""
    
    @staticmethod
    def extract_layout_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract layout data from the enhanced JSON format"""
        
        # Get the optimal layout from the nested structure
        optimal_layout = data.get('optimal_layout', {}).get('optimal_layout', {})
        rooms_data = optimal_layout.get('rooms', {})
        
        # Get directional analysis
        directional_analysis = data.get('directional_analysis', {})
        sun_path_analysis = directional_analysis.get('sun_path_analysis', {})
        wind_flow_analysis = directional_analysis.get('wind_flow_analysis', {})
        heat_gain_analysis = directional_analysis.get('heat_gain_analysis', {})
        privacy_analysis = directional_analysis.get('privacy_analysis', {})
        
        # Convert to our expected format
        rooms = []
        room_counter = {}
        
        for room_key, room_info in rooms_data.items():
            room_type = room_info.get('type', room_key)
            
            # Handle room numbering
            if room_type not in room_counter:
                room_counter[room_type] = 1
            else:
                room_counter[room_type] += 1
            
            # Create unique room ID
            if room_counter[room_type] == 1:
                room_id = room_type
            else:
                room_id = f"{room_type}_{room_counter[room_type]}"
            
            # Extract dimensions
            dimensions = room_info.get('dimensions', [10, 10])
            if len(dimensions) >= 2:
                width = float(dimensions[0])
                height = float(dimensions[1])
            else:
                # Calculate from area if dimensions not available
                area = room_info.get('area', 100)
                width = height = math.sqrt(area)
            
            # Extract position (will be recalculated by smart placer)
            position = room_info.get('position', [0, 0])
            if len(position) >= 2:
                x = float(position[0])
                y = float(position[1])
            else:
                x = y = 0
            
            # Get detailed directional data from analysis
            sun_data = sun_path_analysis.get(room_key, {})
            heat_data = heat_gain_analysis.get(room_key, {})
            privacy_data = privacy_analysis.get(room_key, {})
            
            room_data = {
                'room_id': room_id,
                'room_type': room_type,
                'area': room_info.get('area', width * height),
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'natural_light_access': True,
                'adjacencies': [],  # Will be calculated later
                
                # Environmental data
                'sun_exposure_hours': {
                    'summer': sun_data.get('summer_sun_exposure_hours', 6),
                    'winter': sun_data.get('winter_sun_exposure_hours', 4)
                },
                'heat_gain_level': heat_data.get('heat_gain_level', 'medium'),
                'privacy_score': privacy_data.get('privacy_score', 0.5),
                
                # Window information
                'windows': room_info.get('windows', {}),
                'has_windows': bool(room_info.get('windows', {}))
            }
            
            rooms.append(room_data)
        
        # Calculate flexible adjacencies
        for room in rooms:
            room['adjacencies'] = DataProcessor._find_flexible_adjacencies(room, rooms)
        
        # Use compact room placement
        rooms = CompactRoomPlacer.place_rooms_optimally(rooms)
        
        layout_data = {
            'rooms': rooms,
            'total_area': sum(room['area'] for room in rooms),
            'efficiency_score': optimal_layout.get('scores', {}).get('total_score', 0.8),
            'wind_analysis': wind_flow_analysis,
            'directional_analysis': directional_analysis
        }
        
        return layout_data
    
    @staticmethod
    def _find_flexible_adjacencies(room: Dict, all_rooms: List[Dict]) -> List[str]:
        """Find flexible adjacencies with priority system"""
        
        room_type = room['room_type']
        
        # Define flexible adjacency rules (more lenient)
        adjacency_rules = {
            'kitchen': ['dining_room', 'dining_hall', 'living_room', 'utility', 'hallway'],
            'dining_room': ['kitchen', 'living_room', 'hallway'],
            'dining_hall': ['kitchen', 'living_room', 'hallway'],
            'living_room': ['kitchen', 'dining_room', 'dining_hall', 'hallway', 'balcony'],
            'bedroom': ['bathroom', 'hallway', 'corridor', 'balcony'],
            'bathroom': ['bedroom', 'hallway', 'corridor'],
            'office': ['living_room', 'hallway', 'corridor'],
            'storage': ['kitchen', 'utility', 'garage', 'hallway'],
            'storeroom': ['kitchen', 'utility', 'garage', 'hallway'],
            'utility': ['kitchen', 'storage', 'garage'],
            'garage': ['utility', 'storage'],
            'balcony': ['living_room', 'bedroom'],
            'hallway': ['living_room', 'bedroom', 'bathroom', 'kitchen', 'dining_room', 'office'],
            'corridor': ['bedroom', 'bathroom', 'office']
        }
        
        # Get adjacencies for this room type
        preferred_adjacencies = adjacency_rules.get(room_type, [])
        
        # Find which of these room types actually exist
        existing_types = set(r['room_type'] for r in all_rooms if r['room_id'] != room['room_id'])
        
        # Return intersection
        return [rt for rt in preferred_adjacencies if rt in existing_types]


class ImprovedAdjacencyGraphGenerator:
    """Generate improved adjacency graphs with proper connectivity visualization"""
    
    def __init__(self, config=None):
        self.config = config or VisualConfig()
        self.output_dir = "visual_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_adjacency_graph(self, layout_data: Dict[str, Any], fbs_data: Dict[str, Any] = None, 
                                project_name: str = "house_design") -> str:
        """Generate proper adjacency graph with connected components"""
        
        rooms = layout_data['rooms']
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (rooms) with attributes
        for room in rooms:
            G.add_node(
                room['room_id'],
                room_type=room['room_type'],
                area=room['area'],
                x=room['x'],
                y=room['y'],
                width=room['width'],
                height=room['height'],
                natural_light=room.get('natural_light_access', False),
                heat_gain_level=room.get('heat_gain_level', 'medium'),
                privacy_score=room.get('privacy_score', 0.5),
                has_windows=room.get('has_windows', False)
            )
        
        # Add edges based on ACTUAL spatial adjacencies
        self._add_actual_adjacency_edges(G, rooms)
        
        # Ensure graph is connected (add minimum spanning connections if needed)
        self._ensure_graph_connectivity(G, rooms)
        
        # Generate visualization
        return self._visualize_improved_graph(G, project_name, layout_data)
    
    def _add_actual_adjacency_edges(self, G: nx.Graph, rooms: List[Dict]):
        """Add edges based on STRICT spatial adjacencies only"""
        
        edges_added = 0
        adjacency_debug = []
        
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms[i+1:], i+1):
                
                # Check if rooms are actually adjacent with strict criteria
                if CompactRoomPlacer._are_rooms_adjacent(room1, room2):
                    
                    # Calculate distance for edge weight
                    distance = CompactRoomPlacer._distance_between_rooms(
                        (room1['x'], room1['y'], room1['width'], room1['height']),
                        (room2['x'], room2['y'], room2['width'], room2['height'])
                    )
                    
                    # Determine edge type and priority
                    edge_type, priority = self._determine_edge_type(room1, room2)
                    
                    # Add edge
                    G.add_edge(
                        room1['room_id'], 
                        room2['room_id'],
                        weight=1.0 / (distance + 0.1),
                        distance=distance,
                        edge_type=edge_type,
                        priority=priority,
                        relationship='spatial_adjacent'
                    )
                    
                    edges_added += 1
                    adjacency_debug.append(f"  {room1['room_type']} ↔ {room2['room_type']} ({edge_type})")
        
        print(f"Added {edges_added} STRICT adjacency edges:")
        for debug_line in adjacency_debug:
            print(debug_line)
        
        # If no edges were added (rooms too spread out), use distance-based fallback
        if edges_added == 0:
            print("⚠️ No strict adjacencies found. Using distance-based fallback...")
            self._add_distance_based_edges(G, rooms)
    
    def _determine_edge_type(self, room1: Dict, room2: Dict) -> Tuple[str, str]:
        """Determine the type and priority of edge between two rooms"""
        
        # Get critical adjacencies for both rooms
        room1_critical = CompactRoomPlacer._get_critical_adjacencies(room1)
        room2_critical = CompactRoomPlacer._get_critical_adjacencies(room2)
        
        # Get preferred adjacencies
        room1_preferred = set(room1.get('adjacencies', []))
        room2_preferred = set(room2.get('adjacencies', []))
        
        # Check if this adjacency satisfies critical requirements
        if (room2['room_type'] in room1_critical or room1['room_type'] in room2_critical):
            return 'critical_satisfied', 'high'
        
        # Check if this adjacency satisfies preferred requirements
        elif (room2['room_type'] in room1_preferred or room1['room_type'] in room2_preferred):
            return 'preferred_satisfied', 'medium'
        
        # Just spatial adjacency
        else:
            return 'spatial_only', 'low'
    
    def _add_distance_based_edges(self, G: nx.Graph, rooms: List[Dict]):
        """Fallback: Add edges based on proximity when no strict adjacencies exist"""
        
        # Calculate all pairwise distances
        room_distances = []
        
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms[i+1:], i+1):
                distance = CompactRoomPlacer._distance_between_rooms(
                    (room1['x'], room1['y'], room1['width'], room1['height']),
                    (room2['x'], room2['y'], room2['width'], room2['height'])
                )
                room_distances.append((distance, room1, room2))
        
        # Sort by distance and add only the closest connections
        room_distances.sort(key=lambda x: x[0])
        
        # Add edges for closest pairs (ensuring each room has at least one connection)
        connected_rooms = set()
        edges_added = 0
        
        for distance, room1, room2 in room_distances:
            # Stop when we have enough connections or distance gets too large
            if edges_added >= len(rooms) or distance > 50:  # 50 feet max
                break
                
            # Prioritize connecting unconnected rooms
            if room1['room_id'] not in connected_rooms or room2['room_id'] not in connected_rooms:
                edge_type, priority = self._determine_edge_type(room1, room2)
                
                G.add_edge(
                    room1['room_id'], 
                    room2['room_id'],
                    weight=1.0 / (distance + 0.1),
                    distance=distance,
                    edge_type='proximity_based',
                    priority='low',
                    relationship='distance_based'
                )
                
                connected_rooms.add(room1['room_id'])
                connected_rooms.add(room2['room_id'])
                edges_added += 1
                
                print(f"  Added proximity edge: {room1['room_type']} ↔ {room2['room_type']} (dist: {distance:.1f}ft)")
    
    def _ensure_graph_connectivity(self, G: nx.Graph, rooms: List[Dict]):
        """Ensure graph is connected by adding minimum necessary edges"""
        
        # Check if graph is already connected
        if nx.is_connected(G):
            print("✅ Graph is already connected")
            return
        
        # Find connected components
        components = list(nx.connected_components(G))
        print(f"⚠️ Found {len(components)} disconnected components")
        
        if len(components) <= 1:
            return
        
        # Connect components by finding closest pairs
        main_component = max(components, key=len)
        bridges_added = 0
        
        for component in components:
            if component == main_component:
                continue
            
            # Find closest rooms between this component and main component
            min_distance = float('inf')
            best_connection = None
            
            for room1_id in component:
                room1 = next(r for r in rooms if r['room_id'] == room1_id)
                
                for room2_id in main_component:
                    room2 = next(r for r in rooms if r['room_id'] == room2_id)
                    
                    distance = CompactRoomPlacer._distance_between_rooms(
                        (room1['x'], room1['y'], room1['width'], room1['height']),
                        (room2['x'], room2['y'], room2['width'], room2['height'])
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_connection = (room1_id, room2_id, room1['room_type'], room2['room_type'])
            
            # Add connecting edge
            if best_connection:
                G.add_edge(
                    best_connection[0],
                    best_connection[1],
                    weight=1.0 / (min_distance + 0.1),
                    distance=min_distance,
                    edge_type='connectivity_bridge',
                    priority='medium',
                    relationship='forced_connection'
                )
                main_component.update(component)
                bridges_added += 1
                print(f"  🌉 Bridge: {best_connection[2]} ↔ {best_connection[3]} (dist: {min_distance:.1f}ft)")
        
        print(f"✅ Added {bridges_added} connectivity bridges. Graph now has {G.number_of_edges()} total edges")
    
    def _visualize_improved_graph(self, G: nx.Graph, project_name: str, layout_data: Dict[str, Any]) -> str:
        """Create improved graph visualization with proper connectivity"""
        
        # Create figure with better layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
        fig.suptitle(f'{project_name.replace("_", " ").title()} - Fixed Spatial Analysis', 
                    fontsize=18, fontweight='bold')
        
        # Graph 1: Critical adjacencies analysis
        self._plot_critical_adjacencies_fixed(G, ax1, layout_data, "Critical Adjacencies Analysis")
        
        # Graph 2: Actual spatial layout with all connections
        self._plot_spatial_layout_fixed(G, ax2, layout_data, "Actual Layout - All Connections")
        
        # Graph 3: Connectivity analysis with satisfaction scores
        self._plot_connectivity_analysis_fixed(G, ax3, layout_data, "Connectivity & Satisfaction Analysis")
        
        # Graph 4: Room adjacency performance metrics
        self._plot_adjacency_performance_fixed(G, ax4, layout_data, "Adjacency Performance Metrics")
        
        # Add comprehensive metrics
        self._add_fixed_metrics(fig, G, layout_data)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_fixed_adjacency_analysis_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Fixed adjacency graph saved: {filepath}")
        return filepath
    
    def _plot_critical_adjacencies_fixed(self, G: nx.Graph, ax, layout_data: Dict[str, Any], title: str):
        """Plot critical adjacency analysis with satisfaction status"""
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No rooms to display', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Use spring layout with better parameters
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Get node colors and sizes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            room_type = G.nodes[node].get('room_type', 'default')
            color = self.config.node_colors.get(room_type, self.config.node_colors['default'])
            node_colors.append(color)
            
            area = G.nodes[node].get('area', 100)
            size = min(2000, max(500, area * 8))
            node_sizes.append(size)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax, alpha=0.8, 
                              edgecolors='black', linewidths=2)
        
        # Draw edges by type
        critical_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('edge_type') == 'critical_satisfied']
        preferred_edges = [(u, v) for u, v, d in G.edges(data=True) 
                          if d.get('edge_type') == 'preferred_satisfied']
        spatial_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get('edge_type') == 'spatial_only']
        bridge_edges = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('edge_type') == 'connectivity_bridge']
        
        # Draw different edge types
        if critical_edges:
            nx.draw_networkx_edges(G, pos, edgelist=critical_edges,
                                 edge_color='#27AE60', width=4, ax=ax, alpha=0.9,
                                 style='solid', label=f'Critical Satisfied ({len(critical_edges)})')
        
        if preferred_edges:
            nx.draw_networkx_edges(G, pos, edgelist=preferred_edges,
                                 edge_color='#3498DB', width=3, ax=ax, alpha=0.8,
                                 style='solid', label=f'Preferred Satisfied ({len(preferred_edges)})')
        
        if spatial_edges:
            nx.draw_networkx_edges(G, pos, edgelist=spatial_edges,
                                 edge_color='#95A5A6', width=2, ax=ax, alpha=0.6,
                                 style='solid', label=f'Spatial Only ({len(spatial_edges)})')
        
        if bridge_edges:
            nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,
                                 edge_color='#E74C3C', width=2, ax=ax, alpha=0.8,
                                 style='dashed', label=f'Connectivity Bridges ({len(bridge_edges)})')
        
        # Add labels
        labels = {node: G.nodes[node].get('room_type', 'unknown').replace('_', ' ').title() 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax, font_weight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right')
    
    def _plot_spatial_layout_fixed(self, G: nx.Graph, ax, layout_data: Dict[str, Any], title: str):
        """Plot spatial layout showing actual room positions and connections"""
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No rooms to display', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Use actual room positions
        pos = {}
        for node in G.nodes():
            x = G.nodes[node].get('x', 0) + G.nodes[node].get('width', 0) / 2
            y = G.nodes[node].get('y', 0) + G.nodes[node].get('height', 0) / 2
            pos[node] = (x, -y)  # Flip Y for correct orientation
        
        # Draw room rectangles
        for node in G.nodes():
            x = G.nodes[node].get('x', 0)
            y = -G.nodes[node].get('y', 0) - G.nodes[node].get('height', 0)
            width = G.nodes[node].get('width', 0)
            height = G.nodes[node].get('height', 0)
            
            room_type = G.nodes[node].get('room_type', 'default')
            color = self.config.room_colors.get(room_type, self.config.room_colors['default'])
            
            rect = Rectangle((x, y), width, height, 
                           linewidth=2, edgecolor='black', 
                           facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Room label
            center_x = x + width / 2
            center_y = y + height / 2
            display_name = room_type.replace('_', ' ').title()
            ax.text(center_x, center_y, display_name, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        
        # Draw all edges
        if G.number_of_edges() > 0:
            # Group edges by type for different colors
            edge_groups = {
                'critical_satisfied': [],
                'preferred_satisfied': [],
                'spatial_only': [],
                'connectivity_bridge': []
            }
            
            for u, v, d in G.edges(data=True):
                edge_type = d.get('edge_type', 'spatial_only')
                if edge_type in edge_groups:
                    edge_groups[edge_type].append((u, v))
            
            # Draw each group with different colors
            colors = {
                'critical_satisfied': '#27AE60',
                'preferred_satisfied': '#3498DB', 
                'spatial_only': '#95A5A6',
                'connectivity_bridge': '#E74C3C'
            }
            
            widths = {
                'critical_satisfied': 4,
                'preferred_satisfied': 3,
                'spatial_only': 2,
                'connectivity_bridge': 2
            }
            
            styles = {
                'critical_satisfied': 'solid',
                'preferred_satisfied': 'solid',
                'spatial_only': 'solid',
                'connectivity_bridge': 'dashed'
            }
            
            for edge_type, edges in edge_groups.items():
                if edges:
                    nx.draw_networkx_edges(G, pos, edgelist=edges,
                                         edge_color=colors[edge_type], 
                                         width=widths[edge_type], 
                                         ax=ax, alpha=0.8,
                                         style=styles[edge_type])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_connectivity_analysis_fixed(self, G: nx.Graph, ax, layout_data: Dict[str, Any], title: str):
        """Plot connectivity analysis with satisfaction scores"""
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No rooms to display', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Calculate satisfaction scores for each room
        room_scores = self._calculate_room_satisfaction_scores(G, layout_data['rooms'])
        
        # Create circular layout for better visualization
        pos = nx.circular_layout(G)
        
        # Color nodes by satisfaction level
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            satisfaction = room_scores.get(node, {}).get('overall_score', 0)
            
            if satisfaction >= 0.8:
                color = '#27AE60'  # Green - high satisfaction
            elif satisfaction >= 0.6:
                color = '#F39C12'  # Orange - medium satisfaction
            elif satisfaction >= 0.4:
                color = '#E67E22'  # Dark orange - low-medium satisfaction
            else:
                color = '#E74C3C'  # Red - low satisfaction
            
            node_colors.append(color)
            
            # Size based on connectivity
            connections = G.degree(node)
            size = min(1500, max(300, connections * 200 + 500))
            node_sizes.append(size)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax, alpha=0.8, 
                              edgecolors='black', linewidths=2)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='#2C3E50', width=1.5, ax=ax, alpha=0.6)
        
        # Add labels with satisfaction scores
        labels = {}
        for node in G.nodes():
            room_type = G.nodes[node].get('room_type', 'unknown')
            satisfaction = room_scores.get(node, {}).get('overall_score', 0)
            connections = G.degree(node)
            labels[node] = f"{room_type.replace('_', ' ').title()}\n{satisfaction:.1%}\n({connections} conn.)"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax, font_weight='bold')
        
        # Add satisfaction legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', 
                      markersize=12, label='High Satisfaction (≥80%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', 
                      markersize=12, label='Medium Satisfaction (60-79%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22', 
                      markersize=12, label='Low-Med Satisfaction (40-59%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                      markersize=12, label='Low Satisfaction (<40%)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _calculate_room_satisfaction_scores(self, G: nx.Graph, rooms: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed satisfaction scores for each room"""
        
        satisfaction_scores = {}
        
        for room in rooms:
            room_id = room['room_id']
            room_type = room['room_type']
            
            # Get required adjacencies
            critical_adjacencies = CompactRoomPlacer._get_critical_adjacencies(room)
            preferred_adjacencies = set(room.get('adjacencies', [])) - critical_adjacencies
            
            # Count satisfied adjacencies
            critical_satisfied = 0
            preferred_satisfied = 0
            
            # Check connections in graph
            for neighbor in G.neighbors(room_id):
                neighbor_type = G.nodes[neighbor].get('room_type', 'unknown')
                
                if neighbor_type in critical_adjacencies:
                    critical_satisfied += 1
                elif neighbor_type in preferred_adjacencies:
                    preferred_satisfied += 1
            
            # Calculate scores
            critical_score = (critical_satisfied / len(critical_adjacencies)) if critical_adjacencies else 1.0
            preferred_score = (preferred_satisfied / len(preferred_adjacencies)) if preferred_adjacencies else 1.0
            
            # Overall score (weighted)
            overall_score = critical_score * 0.7 + preferred_score * 0.3
            
            satisfaction_scores[room_id] = {
                'critical_score': critical_score,
                'preferred_score': preferred_score,
                'overall_score': overall_score,
                'connections': G.degree(room_id)
            }
        
        return satisfaction_scores
    
    def _plot_adjacency_performance_fixed(self, G: nx.Graph, ax, layout_data: Dict[str, Any], title: str):
        """Plot detailed adjacency performance metrics"""
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No rooms to display', ha='center', va='center')
            ax.set_title(title)
            return
        
        rooms = layout_data['rooms']
        room_scores = self._calculate_room_satisfaction_scores(G, rooms)
        
        # Extract data for plotting
        room_names = []
        critical_scores = []
        preferred_scores = []
        connection_counts = []
        
        for room in rooms:
            room_id = room['room_id']
            scores = room_scores.get(room_id, {})
            
            room_names.append(room['room_type'].replace('_', ' ').title())
            critical_scores.append(scores.get('critical_score', 0) * 100)
            preferred_scores.append(scores.get('preferred_score', 0) * 100)
            connection_counts.append(scores.get('connections', 0))
        
        # Create grouped bar chart
        x = np.arange(len(room_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, critical_scores, width, 
                      label='Critical Adjacencies', color='#E74C3C', alpha=0.8)
        bars2 = ax.bar(x, preferred_scores, width, 
                      label='Preferred Adjacencies', color='#3498DB', alpha=0.8)
        bars3 = ax.bar(x + width, [c * 10 for c in connection_counts], width, 
                      label='Connections (×10)', color='#27AE60', alpha=0.8)
        
        # Add value labels on bars
        def add_value_labels(bars, values, is_connection=False):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if is_connection:
                    label = f'{int(value/10)}'
                else:
                    label = f'{value:.0f}%'
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1, critical_scores)
        add_value_labels(bars2, preferred_scores)
        add_value_labels(bars3, [c * 10 for c in connection_counts], True)
        
        ax.set_xlabel('Rooms', fontsize=12)
        ax.set_ylabel('Satisfaction Percentage / Connection Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(room_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 110)
    
    def _add_fixed_metrics(self, fig, G: nx.Graph, layout_data: Dict[str, Any]):
        """Add comprehensive fixed metrics"""
        
        if len(G.nodes()) == 0:
            return
        
        rooms = layout_data['rooms']
        room_scores = self._calculate_room_satisfaction_scores(G, rooms)
        
        # Calculate overall metrics
        total_rooms = len(rooms)
        total_connections = G.number_of_edges()
        avg_connections = total_connections * 2 / total_rooms if total_rooms > 0 else 0
        
        # Satisfaction metrics
        critical_scores = [scores.get('critical_score', 0) for scores in room_scores.values()]
        preferred_scores = [scores.get('preferred_score', 0) for scores in room_scores.values()]
        overall_scores = [scores.get('overall_score', 0) for scores in room_scores.values()]
        
        avg_critical = sum(critical_scores) / len(critical_scores) * 100 if critical_scores else 0
        avg_preferred = sum(preferred_scores) / len(preferred_scores) * 100 if preferred_scores else 0
        avg_overall = sum(overall_scores) / len(overall_scores) * 100 if overall_scores else 0
        
        # Connectivity analysis
        is_connected = nx.is_connected(G)
        components = nx.number_connected_components(G)
        
        # Edge type analysis
        edge_types = {}
        for u, v, d in G.edges(data=True):
            edge_type = d.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Layout compactness
        if rooms:
            x_coords = [room['x'] + room['width']/2 for room in rooms]
            y_coords = [room['y'] + room['height']/2 for room in rooms]
            
            if len(x_coords) > 1:
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                bounding_area = x_range * y_range if x_range > 0 and y_range > 0 else 1
                total_room_area = sum(room['area'] for room in rooms)
                compactness = (total_room_area / bounding_area) * 100 if bounding_area > 0 else 100
            else:
                compactness = 100
        else:
            compactness = 100
        
        # Best and worst performing rooms
        if room_scores:
            best_room = max(room_scores.items(), key=lambda x: x[1].get('overall_score', 0))
            worst_room = min(room_scores.items(), key=lambda x: x[1].get('overall_score', 0))
            
            best_room_type = next(r['room_type'] for r in rooms if r['room_id'] == best_room[0])
            worst_room_type = next(r['room_type'] for r in rooms if r['room_id'] == worst_room[0])
        else:
            best_room_type = worst_room_type = "None"
            best_room = worst_room = (None, {'overall_score': 0})
        
        # Create metrics text
        metrics_text = f"""
FIXED ADJACENCY ANALYSIS RESULTS:

CONNECTIVITY STATUS:
• Graph Connected: {'✅ YES' if is_connected else '❌ NO'}
• Connected Components: {components}
• Total Connections: {total_connections}
• Average Connections/Room: {avg_connections:.1f}

ADJACENCY SATISFACTION:
• Critical Adjacencies: {avg_critical:.1f}%
• Preferred Adjacencies: {avg_preferred:.1f}%
• Overall Satisfaction: {avg_overall:.1f}%

CONNECTION TYPES:
• Critical Satisfied: {edge_types.get('critical_satisfied', 0)}
• Preferred Satisfied: {edge_types.get('preferred_satisfied', 0)}
• Spatial Only: {edge_types.get('spatial_only', 0)}
• Connectivity Bridges: {edge_types.get('connectivity_bridge', 0)}

LAYOUT METRICS:
• Space Compactness: {compactness:.1f}%
• Total Area: {layout_data.get('total_area', 0):.0f} sq ft
• Room Count: {total_rooms}

PERFORMANCE ANALYSIS:
• Best Performer: {best_room_type.replace('_', ' ').title()} ({best_room[1].get('overall_score', 0):.1%})
• Needs Improvement: {worst_room_type.replace('_', ' ').title()} ({worst_room[1].get('overall_score', 0):.1%})

STATUS:
{'✅ EXCELLENT - Well connected layout' if avg_overall >= 80 and is_connected else
 '⚠️ GOOD - Minor improvements possible' if avg_overall >= 60 and is_connected else
 '❌ NEEDS WORK - Connectivity issues detected'}
        """
        
        # Add text box
        fig.text(0.02, 0.02, metrics_text.strip(), fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                verticalalignment='bottom', fontfamily='monospace')


# Keep the SVGFloorPlanGenerator and LayoutAnalysisGenerator classes unchanged
# [Previous SVGFloorPlanGenerator and LayoutAnalysisGenerator code remains the same]

class SVGFloorPlanGenerator:
    """Generate SVG floor plans from layout data with global compass"""
    
    def __init__(self, config: VisualConfig = None):
        self.config = config or VisualConfig()
        self.output_dir = "visual_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_floor_plan(self, layout_data: Dict[str, Any], project_name: str = "house_design") -> str:
        """Generate compact SVG floor plan"""
        
        rooms = layout_data['rooms']
        
        if not rooms:
            print("⚠️ No rooms found in layout data")
            return None
        
        # Calculate bounds with minimal margins
        bounds = self._calculate_bounds(rooms)
        
        # Reduce margins for more compact display
        compact_margin = self.config.margin * 0.6
        
        # Create SVG document
        svg_width = (bounds['max_x'] - bounds['min_x'] + 2 * compact_margin) * self.config.scale
        svg_height = (bounds['max_y'] - bounds['min_y'] + 2 * compact_margin) * self.config.scale
        
        # Create root SVG element
        svg = ET.Element('svg')
        svg.set('width', str(svg_width))
        svg.set('height', str(svg_height))
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('viewBox', f'0 0 {svg_width} {svg_height}')
        
        # Add styles
        self._add_styles(svg)
        
        # Add grid if enabled
        if self.config.show_grid:
            self._add_compact_grid(svg, bounds, compact_margin)
        
        # Add global compass
        if self.config.show_directions:
            self._add_global_compass(svg, bounds, compact_margin)
        
        # Add rooms
        for room in rooms:
            self._add_room(svg, room, bounds, compact_margin)
        
        # Add title and metadata
        self._add_title_block(svg, layout_data, project_name)
        
        # Add legend
        self._add_legend(svg, layout_data)
        
        # Save SVG file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_fixed_floorplan_{timestamp}.svg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write SVG to file
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ", level=0)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        print(f"✅ Fixed SVG floor plan saved: {filepath}")
        return filepath
    
    def _calculate_bounds(self, rooms: List[Dict]) -> Dict[str, float]:
        """Calculate tight layout bounds"""
        min_x = min(room['x'] for room in rooms)
        max_x = max(room['x'] + room['width'] for room in rooms)
        min_y = min(room['y'] for room in rooms)
        max_y = max(room['y'] + room['height'] for room in rooms)
        
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
    
    def _add_styles(self, svg: ET.Element):
        """Add CSS styles to SVG"""
        style = ET.SubElement(svg, 'style')
        style.text = f"""
        .room-rect {{
            stroke: {self.config.wall_color};
            stroke-width: {self.config.wall_width * self.config.scale};
            opacity: 0.9;
        }}
        .room-label {{
            font-family: {self.config.font_family};
            font-size: {self.config.font_size}px;
            font-weight: bold;
            text-anchor: middle;
            dominant-baseline: middle;
            fill: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        .dimension-line {{
            stroke: {self.config.dimension_color};
            stroke-width: 1;
            marker-end: url(#arrowhead);
        }}
        .dimension-text {{
            font-family: {self.config.font_family};
            font-size: {self.config.font_size - 2}px;
            fill: {self.config.dimension_color};
            text-anchor: middle;
        }}
        .grid-line {{
            stroke: {self.config.grid_color};
            stroke-width: 0.5;
            opacity: 0.5;
        }}
        .title-text {{
            font-family: {self.config.font_family};
            font-size: {self.config.title_font_size}px;
            font-weight: bold;
            fill: {self.config.wall_color};
        }}
        .compass {{
            font-family: {self.config.font_family};
            font-size: 14px;
            font-weight: bold;
        }}
        """
    
    def _add_compact_grid(self, svg: ET.Element, bounds: Dict[str, float], margin: float):
        """Add compact grid with reduced spacing"""
        grid_group = ET.SubElement(svg, 'g')
        grid_group.set('class', 'grid')
        
        # Smaller grid spacing for compact layout
        spacing = self.config.grid_spacing / 2
        
        # Vertical lines
        x = bounds['min_x'] - margin
        while x <= bounds['max_x'] + margin:
            line = ET.SubElement(grid_group, 'line')
            line.set('x1', str((x - bounds['min_x'] + margin) * self.config.scale))
            line.set('y1', '0')
            line.set('x2', str((x - bounds['min_x'] + margin) * self.config.scale))
            line.set('y2', str((bounds['max_y'] - bounds['min_y'] + 2 * margin) * self.config.scale))
            line.set('class', 'grid-line')
            x += spacing
        
        # Horizontal lines
        y = bounds['min_y'] - margin
        while y <= bounds['max_y'] + margin:
            line = ET.SubElement(grid_group, 'line')
            line.set('x1', '0')
            line.set('y1', str((y - bounds['min_y'] + margin) * self.config.scale))
            line.set('x2', str((bounds['max_x'] - bounds['min_x'] + 2 * margin) * self.config.scale))
            line.set('y2', str((y - bounds['min_y'] + margin) * self.config.scale))
            line.set('class', 'grid-line')
            y += spacing
    
    def _add_global_compass(self, svg: ET.Element, bounds: Dict[str, float], margin: float):
        """Add global compass in top-right corner"""
        compass_size = self.config.compass_size
        
        # Position in top-right corner
        compass_x = (bounds['max_x'] - bounds['min_x'] + 2 * margin) * self.config.scale - compass_size - 20
        compass_y = 20
        
        compass_group = ET.SubElement(svg, 'g')
        compass_group.set('transform', f'translate({compass_x}, {compass_y})')
        
        # Compass circle background
        circle = ET.SubElement(compass_group, 'circle')
        circle.set('cx', str(compass_size / 2))
        circle.set('cy', str(compass_size / 2))
        circle.set('r', str(compass_size / 2 - 2))
        circle.set('fill', 'white')
        circle.set('stroke', self.config.wall_color)
        circle.set('stroke-width', '2')
        circle.set('opacity', '0.9')
        
        # North arrow
        arrow_points = f"{compass_size/2},{compass_size/2-15} {compass_size/2-5},{compass_size/2-5} {compass_size/2+5},{compass_size/2-5}"
        north_arrow = ET.SubElement(compass_group, 'polygon')
        north_arrow.set('points', arrow_points)
        north_arrow.set('fill', self.config.direction_arrow_color)
        
        # Compass labels
        labels = [
            ('N', compass_size/2, 8, self.config.direction_arrow_color),
            ('S', compass_size/2, compass_size-4, self.config.wall_color),
            ('E', compass_size-4, compass_size/2+3, self.config.wall_color),
            ('W', 4, compass_size/2+3, self.config.wall_color)
        ]
        
        for label, x, y, color in labels:
            text = ET.SubElement(compass_group, 'text')
            text.set('x', str(x))
            text.set('y', str(y))
            text.set('text-anchor', 'middle')
            text.set('class', 'compass')
            text.set('fill', color)
            text.text = label
        
        # Sun indicator (small circle)
        sun = ET.SubElement(compass_group, 'circle')
        sun.set('cx', str(compass_size - 8))
        sun.set('cy', str(8))
        sun.set('r', '3')
        sun.set('fill', self.config.sun_color)
        sun.set('stroke', '#F39C12')
        sun.set('stroke-width', '1')
        
        # Wind indicator (small triangle)
        wind_points = f"8,8 12,4 12,12"
        wind = ET.SubElement(compass_group, 'polygon')
        wind.set('points', wind_points)
        wind.set('fill', self.config.wind_color)
    
    def _add_room(self, svg: ET.Element, room: Dict, bounds: Dict[str, float], margin: float):
        """Add room rectangle with enhanced details"""
        # Calculate SVG coordinates
        svg_x = (room['x'] - bounds['min_x'] + margin) * self.config.scale
        svg_y = (room['y'] - bounds['min_y'] + margin) * self.config.scale
        svg_width = room['width'] * self.config.scale
        svg_height = room['height'] * self.config.scale
        
        # Get room color
        room_type = room['room_type']
        fill_color = self.config.room_colors.get(room_type, self.config.room_colors['default'])
        
        # Room rectangle
        rect = ET.SubElement(svg, 'rect')
        rect.set('x', str(svg_x))
        rect.set('y', str(svg_y))
        rect.set('width', str(svg_width))
        rect.set('height', str(svg_height))
        rect.set('fill', fill_color)
        rect.set('class', 'room-rect')
        
        # Room label
        label_x = svg_x + svg_width / 2
        label_y = svg_y + svg_height / 2
        
        text = ET.SubElement(svg, 'text')
        text.set('x', str(label_x))
        text.set('y', str(label_y))
        text.set('class', 'room-label')
        
        # Format room name
        display_name = room_type.replace('_', ' ').title()
        text.text = display_name
        
        # Add area information
        area_text = ET.SubElement(svg, 'text')
        area_text.set('x', str(label_x))
        area_text.set('y', str(label_y + 15))
        area_text.set('class', 'room-label')
        area_text.set('font-size', str(self.config.font_size - 2))
        area_text.text = f"{room['area']:.0f} sq ft"
        
        # Add window indicators if room has windows
        if room.get('has_windows', False):
            self._add_window_indicators(svg, room, svg_x, svg_y, svg_width, svg_height)
    
    def _add_window_indicators(self, svg: ET.Element, room: Dict, svg_x: float, svg_y: float, 
                             svg_width: float, svg_height: float):
        """Add window indicators to room"""
        windows = room.get('windows', {})
        
        for direction, window_info in windows.items():
            if not isinstance(window_info, dict) or not window_info.get('count', 0):
                continue
            
            window_count = window_info.get('count', 1)
            window_size = 20  # SVG units
            
            # Calculate window positions based on direction
            if direction == 'north':
                start_x = svg_x + (svg_width - window_count * window_size) / 2
                start_y = svg_y
                for i in range(window_count):
                    self._draw_window(svg, start_x + i * window_size, start_y, window_size, 3, 'horizontal')
            
            elif direction == 'south':
                start_x = svg_x + (svg_width - window_count * window_size) / 2
                start_y = svg_y + svg_height - 3
                for i in range(window_count):
                    self._draw_window(svg, start_x + i * window_size, start_y, window_size, 3, 'horizontal')
            
            elif direction == 'east':
                start_x = svg_x + svg_width - 3
                start_y = svg_y + (svg_height - window_count * window_size) / 2
                for i in range(window_count):
                    self._draw_window(svg, start_x, start_y + i * window_size, 3, window_size, 'vertical')
            
            elif direction == 'west':
                start_x = svg_x
                start_y = svg_y + (svg_height - window_count * window_size) / 2
                for i in range(window_count):
                    self._draw_window(svg, start_x, start_y + i * window_size, 3, window_size, 'vertical')
    
    def _draw_window(self, svg: ET.Element, x: float, y: float, width: float, height: float, orientation: str):
        """Draw individual window indicator"""
        window = ET.SubElement(svg, 'rect')
        window.set('x', str(x))
        window.set('y', str(y))
        window.set('width', str(width))
        window.set('height', str(height))
        window.set('fill', '#87CEEB')  # Light blue for windows
        window.set('stroke', '#4682B4')  # Steel blue border
        window.set('stroke-width', '1')
    
    def _add_title_block(self, svg: ET.Element, layout_data: Dict[str, Any], project_name: str):
        """Add title block with project information"""
        title_y = 25
        
        # Project title
        title = ET.SubElement(svg, 'text')
        title.set('x', '20')
        title.set('y', str(title_y))
        title.set('class', 'title-text')
        title.text = f"{project_name.replace('_', ' ').title()} - Fixed Floor Plan"
        
        # Subtitle with key metrics
        subtitle = ET.SubElement(svg, 'text')
        subtitle.set('x', '20')
        subtitle.set('y', str(title_y + 20))
        subtitle.set('font-family', self.config.font_family)
        subtitle.set('font-size', str(self.config.font_size))
        subtitle.set('fill', self.config.wall_color)
        
        total_area = layout_data.get('total_area', 0)
        room_count = len(layout_data.get('rooms', []))
        efficiency = layout_data.get('efficiency_score', 0)
        
        subtitle.text = f"Total Area: {total_area:.0f} sq ft | Rooms: {room_count} | Efficiency: {efficiency:.1%}"
        
        # Timestamp
        timestamp = ET.SubElement(svg, 'text')
        timestamp.set('x', '20')
        timestamp.set('y', str(title_y + 40))
        timestamp.set('font-family', self.config.font_family)
        timestamp.set('font-size', str(self.config.font_size - 2))
        timestamp.set('fill', '#7F8C8D')
        timestamp.text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def _add_legend(self, svg: ET.Element, layout_data: Dict[str, Any]):
        """Add compact legend"""
        rooms = layout_data.get('rooms', [])
        if not rooms:
            return
        
        # Get unique room types
        room_types = list(set(room['room_type'] for room in rooms))
        room_types.sort()
        
        # Legend position (bottom-left)
        legend_x = 20
        legend_y = float(svg.get('height')) - 120
        
        # Legend background
        legend_bg = ET.SubElement(svg, 'rect')
        legend_bg.set('x', str(legend_x - 10))
        legend_bg.set('y', str(legend_y - 25))
        legend_bg.set('width', '200')
        legend_bg.set('height', str(len(room_types) * 20 + 35))
        legend_bg.set('fill', 'white')
        legend_bg.set('stroke', self.config.wall_color)
        legend_bg.set('stroke-width', '1')
        legend_bg.set('opacity', '0.95')
        
        # Legend title
        legend_title = ET.SubElement(svg, 'text')
        legend_title.set('x', str(legend_x))
        legend_title.set('y', str(legend_y - 5))
        legend_title.set('font-family', self.config.font_family)
        legend_title.set('font-size', str(self.config.font_size + 2))
        legend_title.set('font-weight', 'bold')
        legend_title.set('fill', self.config.wall_color)
        legend_title.text = "Room Types"
        
        # Legend items
        for i, room_type in enumerate(room_types):
            item_y = legend_y + 15 + i * 20
            
            # Color square
            color_square = ET.SubElement(svg, 'rect')
            color_square.set('x', str(legend_x))
            color_square.set('y', str(item_y - 8))
            color_square.set('width', '12')
            color_square.set('height', '12')
            color_square.set('fill', self.config.room_colors.get(room_type, self.config.room_colors['default']))
            color_square.set('stroke', self.config.wall_color)
            color_square.set('stroke-width', '0.5')
            
            # Label
            label = ET.SubElement(svg, 'text')
            label.set('x', str(legend_x + 20))
            label.set('y', str(item_y))
            label.set('font-family', self.config.font_family)
            label.set('font-size', str(self.config.font_size - 1))
            label.set('fill', self.config.wall_color)
            label.text = room_type.replace('_', ' ').title()


# Main function to generate all visualizations with fixed adjacency logic
def generate_all_visualizations_fixed(json_file_path: str, project_name: str = None) -> Dict[str, str]:
    """Generate all visualizations from JSON data with fixed adjacency graphs"""
    
    try:
        # Load and process data
        with open(json_file_path, 'r') as file:
            fbs_data = json.load(file)
        
        layout_data = DataProcessor.extract_layout_from_json(fbs_data)
        
        if not project_name:
            project_name = Path(json_file_path).stem
        
        print(f"🏠 Generating FIXED visualizations for: {project_name}")
        print(f"📊 Processing {len(layout_data['rooms'])} rooms...")
        
        # Debug: Print room positions and adjacencies
        print("\n🏗️ Room Layout Summary:")
        for room in layout_data['rooms']:
            adjacent_rooms = room.get('actual_adjacent_rooms', [])
            print(f"  {room['room_type']} at ({room['x']:.1f}, {room['y']:.1f}) "
                  f"[{room['width']:.1f}×{room['height']:.1f}] - "
                  f"Adjacent to: {adjacent_rooms}")
        
        # Initialize generators
        config = VisualConfig()
        svg_generator = SVGFloorPlanGenerator(config)
        adjacency_generator = ImprovedAdjacencyGraphGenerator(config)
        
        # Generate visualizations
        generated_files = {}
        
        # 1. Fixed SVG Floor Plan
        print("\n📐 Generating fixed SVG floor plan...")
        svg_file = svg_generator.generate_floor_plan(layout_data, project_name)
        if svg_file:
            generated_files['svg_floor_plan'] = svg_file
        
        # 2. Fixed Adjacency Graph (main fix)
        print("\n🔗 Generating FIXED adjacency analysis...")
        adjacency_file = adjacency_generator.generate_adjacency_graph(layout_data, fbs_data, project_name)
        if adjacency_file:
            generated_files['adjacency_graph'] = adjacency_file
        
        print(f"\n✅ All FIXED visualizations generated successfully!")
        print(f"📁 Output directory: {Path('visual_outputs').absolute()}")
        
        return generated_files
        
    except Exception as e:
        print(f"❌ Error generating fixed visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


# Example usage
if __name__ == "__main__":
    # Example usage with a JSON file
    json_file = "enhanced_house_design_analysis.json"  # Replace with your JSON file path
    project_name = "fixed_modern_family_home"     # Optional project name
    
    # Generate all fixed visualizations
    files = generate_all_visualizations_fixed(json_file, project_name)
    
    if files:
        print("\nGenerated fixed files:")
        for viz_type, filepath in files.items():
            print(f"  {viz_type}: {filepath}")
    else:
        print("No files were generated. Please check the input data and try again.")