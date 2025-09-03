# layout_optimizer.py
from layout_generator import CompactRoomPlacer, generate_all_visualizations_fixed, DataProcessor
from fbs import EnhancedDirectionalFBSInterface
import numpy as np
import json
import os
import tempfile
from typing import List, Dict, Any
import math

class LayoutOptimizer:
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.fbs_interface = EnhancedDirectionalFBSInterface()

    def optimize_layouts(self, prototypes: List[Dict], requirements: Dict) -> List[Dict]:
        """Iterative layout optimization loop (for flowchart's S→W→W1→S)."""
        if not prototypes:
            print("Warning: No prototypes provided for optimization")
            return []
        
        # Convert prototype to layout format if needed
        layout_data = self._extract_layout_from_prototype(prototypes[0])
        rooms = layout_data.get('rooms', [])
        
        if not rooms:
            print("Warning: No rooms found in prototype")
            return []
        
        best_layouts = []
        prev_performance = 0.0
        iteration = 0
        
        print(f"Starting layout optimization with {len(rooms)} rooms...")
        
        while iteration < self.max_iterations:
            print(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate variants using the CompactRoomPlacer
            try:
                variants = self._generate_layout_variants(rooms, num_variants=3)
                print(f"Generated {len(variants)} layout variants")
                
                if not variants:
                    print("No variants generated, breaking optimization loop")
                    break
                
                # Cross-layout comparison (flowchart U)
                performances = self._compare_layouts(variants, prototypes[0], requirements)
                current_performance = np.mean(performances)
                
                print(f"Performance scores: {[f'{p:.3f}' for p in performances]}")
                print(f"Average performance: {current_performance:.3f}")
                
                # Check convergence (flowchart W1)
                if abs(current_performance - prev_performance) < self.convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break
                
                # Select best (flowchart V)
                best_idx = np.argmax(performances)
                best_layout = {
                    'rooms': variants[best_idx],
                    'performance_score': performances[best_idx],
                    'iteration': iteration + 1
                }
                best_layouts.append(best_layout)
                
                print(f"Best layout performance: {performances[best_idx]:.3f}")
                
                # Generate visualizations for best layout
                self._generate_visualization_for_layout(best_layout, f"optimized_layout_iter_{iteration + 1}")
                
                prev_performance = current_performance
                iteration += 1
                
            except Exception as e:
                print(f"Error in optimization iteration {iteration + 1}: {str(e)}")
                break
        
        if not best_layouts:
            # Fallback: return original layout with optimization applied
            optimized_rooms = CompactRoomPlacer.place_rooms_optimally(rooms)
            best_layouts = [{
                'rooms': optimized_rooms,
                'performance_score': 0.7,
                'iteration': 0
            }]
        
        print(f"Optimization completed. Generated {len(best_layouts)} optimized layouts")
        return best_layouts

    def _extract_layout_from_prototype(self, prototype: Dict) -> Dict[str, Any]:
        """Extract layout data from prototype format"""
        
        # Check if prototype already has rooms in the expected format
        if 'rooms' in prototype and isinstance(prototype['rooms'], list):
            return {'rooms': prototype['rooms']}
        
        # Try to extract from nested structure
        rooms = []
        
        # Look for room data in various possible locations
        room_sources = [
            prototype.get('spatial_layout', {}),
            prototype.get('layout', {}),
            prototype.get('room_layout', {}),
            prototype
        ]
        
        for source in room_sources:
            if isinstance(source, dict):
                # Try to find rooms data
                if 'rooms' in source and isinstance(source['rooms'], (list, dict)):
                    rooms_data = source['rooms']
                    
                    if isinstance(rooms_data, list):
                        rooms = rooms_data
                        break
                    elif isinstance(rooms_data, dict):
                        # Convert dict format to list format
                        rooms = []
                        for room_id, room_info in rooms_data.items():
                            room_dict = {
                                'room_id': room_id,
                                'room_type': room_info.get('type', room_id),
                                'area': room_info.get('area', 100),
                                'width': room_info.get('dimensions', [10, 10])[0] if room_info.get('dimensions') else 10,
                                'height': room_info.get('dimensions', [10, 10])[1] if room_info.get('dimensions') and len(room_info['dimensions']) > 1 else 10,
                                'x': 0,
                                'y': 0,
                                'adjacencies': room_info.get('adjacencies', []),
                                'natural_light_access': room_info.get('natural_light_access', True),
                                'has_windows': room_info.get('has_windows', True)
                            }
                            rooms.append(room_dict)
                        break
        
        # If still no rooms found, create basic rooms from spatial_needs
        if not rooms and 'spatial_needs' in prototype:
            spatial_needs = prototype['spatial_needs']
            if isinstance(spatial_needs, list):
                for i, need in enumerate(spatial_needs):
                    if isinstance(need, dict):
                        room_type = need.get('room_type', need.get('type', f'room_{i}'))
                        area = need.get('area', need.get('size', 100))
                        width = height = math.sqrt(area)
                        
                        room_dict = {
                            'room_id': room_type,
                            'room_type': room_type,
                            'area': area,
                            'width': width,
                            'height': height,
                            'x': 0,
                            'y': 0,
                            'adjacencies': need.get('adjacencies', []),
                            'natural_light_access': need.get('natural_light_access', True),
                            'has_windows': need.get('has_windows', True)
                        }
                        rooms.append(room_dict)
        
        # Final fallback: create basic rooms
        if not rooms:
            default_rooms = [
                {'room_id': 'living_room', 'room_type': 'living_room', 'area': 200, 'width': 15, 'height': 13.33, 'x': 0, 'y': 0, 'adjacencies': []},
                {'room_id': 'kitchen', 'room_type': 'kitchen', 'area': 120, 'width': 12, 'height': 10, 'x': 0, 'y': 0, 'adjacencies': []},
                {'room_id': 'bedroom', 'room_type': 'bedroom', 'area': 150, 'width': 12, 'height': 12.5, 'x': 0, 'y': 0, 'adjacencies': []},
                {'room_id': 'bathroom', 'room_type': 'bathroom', 'area': 50, 'width': 7, 'height': 7.14, 'x': 0, 'y': 0, 'adjacencies': []}
            ]
            rooms = default_rooms
        
        # Ensure all rooms have required attributes
        for room in rooms:
            room.setdefault('natural_light_access', True)
            room.setdefault('has_windows', True)
            room.setdefault('adjacencies', [])
        
        return {'rooms': rooms}

    def _generate_layout_variants(self, rooms: List[Dict], num_variants: int = 3) -> List[List[Dict]]:
        """Generate multiple layout variants with different arrangements"""
        variants = []
        
        for variant_num in range(num_variants):
            try:
                # Create a copy of rooms for this variant
                variant_rooms = [room.copy() for room in rooms]
                
                # Apply different placement strategies for each variant
                if variant_num == 0:
                    # Standard optimal placement
                    placed_rooms = CompactRoomPlacer.place_rooms_optimally(variant_rooms)
                elif variant_num == 1:
                    # Shuffle room priority and place
                    import random
                    random.shuffle(variant_rooms)
                    placed_rooms = CompactRoomPlacer.place_rooms_optimally(variant_rooms)
                else:
                    # Alternative arrangement (e.g., linear layout)
                    placed_rooms = self._create_linear_layout(variant_rooms)
                
                variants.append(placed_rooms)
                
            except Exception as e:
                print(f"Error generating variant {variant_num}: {str(e)}")
                continue
        
        # Ensure we have at least one variant
        if not variants:
            # Fallback: use original optimal placement
            placed_rooms = CompactRoomPlacer.place_rooms_optimally([room.copy() for room in rooms])
            variants.append(placed_rooms)
        
        return variants

    def _create_linear_layout(self, rooms: List[Dict]) -> List[Dict]:
        """Create a linear arrangement of rooms as an alternative layout"""
        placed_rooms = []
        current_x = 0
        
        for room in rooms:
            room_copy = room.copy()
            room_copy['x'] = current_x
            room_copy['y'] = 0
            current_x += room_copy['width'] + 1  # Add 1 foot spacing
            placed_rooms.append(room_copy)
        
        # Calculate actual adjacencies for linear layout
        for room in placed_rooms:
            room['actual_adjacent_rooms'] = CompactRoomPlacer._get_actual_adjacent_rooms(room, placed_rooms)
        
        return placed_rooms

    def _compare_layouts(self, variants: List[List[Dict]], prototype: Dict, requirements: Dict) -> List[float]:
        """Cross-layout FBS comparison (flowchart U)."""
        performances = []
        
        for i, variant in enumerate(variants):
            try:
                # Create a proper layout data structure
                layout_data = {
                    'rooms': variant,
                    'total_area': sum(room.get('area', 0) for room in variant),
                    'efficiency_score': 0.8
                }
                
                # Calculate performance based on multiple criteria
                score = self._calculate_layout_performance(layout_data, requirements)
                performances.append(score)
                
                print(f"Variant {i + 1} performance: {score:.3f}")
                
            except Exception as e:
                print(f"Error evaluating variant {i + 1}: {str(e)}")
                performances.append(0.0)  # Default low score for failed evaluation
        
        return performances

    def _calculate_layout_performance(self, layout_data: Dict, requirements: Dict) -> float:
        """Calculate comprehensive layout performance score"""
        rooms = layout_data['rooms']
        
        if not rooms:
            return 0.0
        
        score_components = {}
        
        # 1. Adjacency satisfaction score (40% weight)
        adjacency_score = self._calculate_adjacency_satisfaction(rooms)
        score_components['adjacency'] = adjacency_score
        
        # 2. Compactness score (25% weight)
        compactness_score = self._calculate_compactness_score(rooms)
        score_components['compactness'] = compactness_score
        
        # 3. Functional efficiency score (20% weight)
        efficiency_score = self._calculate_functional_efficiency(rooms, requirements)
        score_components['efficiency'] = efficiency_score
        
        # 4. Space utilization score (15% weight)
        utilization_score = self._calculate_space_utilization(rooms)
        score_components['utilization'] = utilization_score
        
        # Calculate weighted final score
        weights = {
            'adjacency': 0.40,
            'compactness': 0.25,
            'efficiency': 0.20,
            'utilization': 0.15
        }
        
        final_score = sum(score_components[component] * weights[component] 
                         for component in score_components)
        
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1

    def _calculate_adjacency_satisfaction(self, rooms: List[Dict]) -> float:
        """Calculate how well the layout satisfies desired adjacencies"""
        total_satisfaction = 0.0
        total_requirements = 0
        
        for room in rooms:
            desired_adjacencies = set(room.get('adjacencies', []))
            actual_adjacencies = set(room.get('actual_adjacent_rooms', []))
            
            if desired_adjacencies:
                satisfied = len(desired_adjacencies.intersection(actual_adjacencies))
                total_satisfaction += satisfied / len(desired_adjacencies)
                total_requirements += 1
        
        return total_satisfaction / total_requirements if total_requirements > 0 else 0.5

    def _calculate_compactness_score(self, rooms: List[Dict]) -> float:
        """Calculate how compact the layout is (higher = more compact)"""
        if len(rooms) <= 1:
            return 1.0
        
        # Calculate bounding rectangle
        min_x = min(room['x'] for room in rooms)
        max_x = max(room['x'] + room['width'] for room in rooms)
        min_y = min(room['y'] for room in rooms)
        max_y = max(room['y'] + room['height'] for room in rooms)
        
        bounding_area = (max_x - min_x) * (max_y - min_y)
        total_room_area = sum(room['area'] for room in rooms)
        
        if bounding_area > 0:
            compactness = total_room_area / bounding_area
            return min(1.0, compactness)  # Cap at 1.0
        
        return 0.5

    def _calculate_functional_efficiency(self, rooms: List[Dict], requirements: Dict) -> float:
        """Calculate functional efficiency based on room relationships"""
        # Count beneficial adjacencies (kitchen-dining, bedroom-bathroom, etc.)
        beneficial_pairs = [
            ('kitchen', 'dining_room'),
            ('kitchen', 'dining_hall'),
            ('bedroom', 'bathroom'),
            ('living_room', 'kitchen'),
            ('living_room', 'dining_room')
        ]
        
        room_types = {room['room_type']: room for room in rooms}
        beneficial_adjacencies = 0
        possible_beneficial = 0
        
        for type1, type2 in beneficial_pairs:
            if type1 in room_types and type2 in room_types:
                possible_beneficial += 1
                room1 = room_types[type1]
                room2 = room_types[type2]
                
                if CompactRoomPlacer._are_rooms_adjacent(room1, room2):
                    beneficial_adjacencies += 1
        
        return beneficial_adjacencies / possible_beneficial if possible_beneficial > 0 else 0.5

    def _calculate_space_utilization(self, rooms: List[Dict]) -> float:
        """Calculate how efficiently space is used"""
        if not rooms:
            return 0.0
        
        # Simple metric: ratio of room areas to total layout area
        total_room_area = sum(room['area'] for room in rooms)
        
        # Calculate total layout footprint
        min_x = min(room['x'] for room in rooms)
        max_x = max(room['x'] + room['width'] for room in rooms)
        min_y = min(room['y'] for room in rooms)  
        max_y = max(room['y'] + room['height'] for room in rooms)
        
        layout_area = (max_x - min_x) * (max_y - min_y)
        
        if layout_area > 0:
            utilization = total_room_area / layout_area
            return min(1.0, utilization)
        
        return 0.5

    def _generate_visualization_for_layout(self, layout: Dict, project_name: str):
        """Generate visualization for a specific layout"""
        try:
            # Create temporary JSON file for visualization
            temp_data = {
                'optimal_layout': {
                    'optimal_layout': {
                        'rooms': {
                            room['room_id']: {
                                'type': room['room_type'],
                                'area': room['area'],
                                'dimensions': [room['width'], room['height']],
                                'position': [room['x'], room['y']],
                                'windows': room.get('windows', {}),
                                'adjacencies': room.get('adjacencies', [])
                            }
                            for room in layout['rooms']
                        }
                    }
                },
                'directional_analysis': {}
            }
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(temp_data, temp_file)
                temp_file_path = temp_file.name
            
            # Generate visualization
            generate_all_visualizations_fixed(temp_file_path, project_name)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")