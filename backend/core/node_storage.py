"""
Node Storage System for Hierarchical Graph of Thoughts

Manages storage of all nodes (intermediate and final) with complete FBSL data,
parent-child relationships, and exploration tree structure.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class NodeStorage:
    """
    Manages hierarchical storage of all GoT nodes with complete FBSL data
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.nodes_dir = self.output_dir / "nodes"
        self.viz_dir = self.output_dir / "visualizations"
        
        # Create directory structure
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory registry
        self.node_registry = {}  # node_id → node data
        self.tree_structure = {
            "root": None,
            "levels": {},
            "relationships": {},
            "pruned_nodes": []
        }
        
        logger.info(f"NodeStorage initialized at {output_dir}")
    
    def store_node(self, node, level: int, parent_id: str = None, 
                   transformation_type: str = None, reasoning: str = None,
                   pruned: bool = False, prune_reason: str = None):
        """
        Store complete node data with FBSL
        
        Args:
            node: FBSLLayoutNode instance
            level: Exploration level (0, 1, 2, 3, "final")
            parent_id: Parent node ID
            transformation_type: Type of transformation applied
            reasoning: Reasoning for this design variant
            pruned: Whether node was pruned
            prune_reason: Reason for pruning
        """
        
        # Serialize complete FBSL
        node_data = self.serialize_fbsl(node)
        
        # Add metadata
        node_data["level"] = level
        node_data["parent_id"] = parent_id
        node_data["transformation_type"] = transformation_type
        node_data["reasoning"] = reasoning
        node_data["metadata"]["pruned"] = pruned
        node_data["metadata"]["prune_reason"] = prune_reason
        node_data["metadata"]["timestamp"] = datetime.now().isoformat()
        
        # Store in registry
        self.node_registry[node.node_id] = node_data
        
        # Update tree structure
        if level == 0:
            self.tree_structure["root"] = node.node_id
        
        level_key = str(level)
        if level_key not in self.tree_structure["levels"]:
            self.tree_structure["levels"][level_key] = []
        self.tree_structure["levels"][level_key].append(node.node_id)
        
        # Update relationships
        self.tree_structure["relationships"][node.node_id] = {
            "parent": parent_id,
            "children": [],
            "transformation": transformation_type,
            "pruned": pruned,
            "level": level
        }
        
        if parent_id and parent_id in self.tree_structure["relationships"]:
            self.tree_structure["relationships"][parent_id]["children"].append(node.node_id)
        
        if pruned:
            self.tree_structure["pruned_nodes"].append(node.node_id)
        
        # Save to disk
        level_dir = self.nodes_dir / f"level_{level}"
        level_dir.mkdir(exist_ok=True)
        
        node_file = level_dir / f"node_{node.node_id[:8]}.json"
        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(node_data, f, indent=2)
        
        logger.debug(f"Stored node {node.node_id[:8]} at level {level}")
    
    def serialize_fbsl(self, node) -> Dict[str, Any]:
        """
        Convert FBSL node to complete JSON structure
        
        Returns complete FBSL data including functions, behaviors, structures, layout, scores
        """
        
        data = {
            "node_id": node.node_id,
            "node_type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
            
            # Functions
            "functions": self._serialize_functions(node.functions) if node.functions else [],
            
            # Behaviors
            "behaviors": self._serialize_behaviors(node.behaviors) if node.behaviors else [],
            
            # Structures (always call to generate defaults if empty)
            "structures": self._serialize_structures(node.structures if node.structures is not None else {}),
            
            # Layout
            "layout": self._serialize_layout(node.layout) if node.layout else None,
            
            # Scores
            "scores": {
                "functional": round(node.functional_score, 3) if node.functional_score else 0.0,
                "behavioral": round(node.behavioral_score, 3) if node.behavioral_score else 0.0,
                "structural": round(node.structural_score, 3) if node.structural_score else 0.0,
                "layout": round(node.layout_score, 3) if node.layout_score else 0.0,
                "sustainability": round(node.sustainability_score, 3) if node.sustainability_score else 0.0,
                "composite": round(node.composite_score, 3) if node.composite_score else 0.0
            },
            
            # Metadata
            "metadata": {
                "generation_level": getattr(node, 'generation_level', 0),
                "iteration_number": getattr(node, 'iteration_number', 0),
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Add any additional metadata from node
        if hasattr(node, 'metadata') and node.metadata:
            data["metadata"].update(node.metadata)
        
        return data
    
    def _serialize_functions(self, functions: Dict) -> List[Dict]:
        """Serialize functions to JSON-compatible format"""
        result = []
        for func_id, func in functions.items():
            func_data = {
                "function_id": func.function_id,
                "name": func.name,
                "category": func.category.value if hasattr(func.category, 'value') else str(func.category),
                "priority": round(func.priority, 3),
                "activities": func.activities or [],
                "spatial_requirements": {
                    "min_area": func.min_area,
                    "preferred_area": func.preferred_area,
                    "height": func.height
                } if hasattr(func, 'min_area') else {},
                "temporal_needs": getattr(func, 'temporal_needs', None),
                "dependencies": getattr(func, 'dependencies', []),
                "conflicts_with": getattr(func, 'conflicts_with', [])
            }
            result.append(func_data)
        return result
    
    def _serialize_behaviors(self, behaviors: Dict) -> List[Dict]:
        """Serialize behaviors to JSON-compatible format"""
        result = []
        for behav_id, behav in behaviors.items():
            behav_data = {
                "behavior_id": behav.behavior_id,
                "metric_name": behav.metric_name,
                "category": behav.category.value if hasattr(behav.category, 'value') else str(behav.category),
                "behavior_type": behav.behavior_type.value if hasattr(behav.behavior_type, 'value') else str(behav.behavior_type),
                "target_value": round(behav.target_value, 3) if behav.target_value else None,
                "actual_value": round(behav.actual_value, 3) if behav.actual_value else None,
                "tolerance": round(behav.tolerance, 3) if behav.tolerance else None,
                "is_satisfied": behav.is_satisfied,
                "derived_from_function": behav.derived_from_function,
                "unit": getattr(behav, 'unit', None)
            }
            result.append(behav_data)
        return result
    
    def _serialize_structures(self, structures: Dict) -> List[Dict]:
        """Serialize structures to JSON-compatible format"""
        result = []
        
        # If no structures provided, generate basic building envelope
        if not structures or len(structures) == 0:
            # Generate basic structures for residential building
            basic_structures = [
                {
                    "structure_id": "s1",
                    "name": "External Walls",
                    "type": "WALL",
                    "material_type": "insulated_concrete",
                    "properties": {
                        "u_value": 0.25,
                        "stc_rating": 50,
                        "thickness": 0.30,
                        "r_value": 4.0
                    },
                    "load_bearing": True,
                    "dimensions": {"length": 50.0, "height": 2.7}
                },
                {
                    "structure_id": "s2",
                    "name": "Internal Partitions",
                    "type": "PARTITION",
                    "material_type": "gypsum_board",
                    "properties": {
                        "u_value": 0.0,
                        "stc_rating": 45,
                        "thickness": 0.10
                    },
                    "load_bearing": False,
                    "dimensions": {"length": 80.0, "height": 2.7}
                },
                {
                    "structure_id": "s3",
                    "name": "Roof",
                    "type": "ROOF",
                    "material_type": "insulated_concrete_slab",
                    "properties": {
                        "u_value": 0.16,
                        "stc_rating": 55,
                        "thickness": 0.25,
                        "r_value": 6.0
                    },
                    "load_bearing": True,
                    "dimensions": {"area": 250.0}
                },
                {
                    "structure_id": "s4",
                    "name": "Foundation Slab",
                    "type": "SLAB",
                    "material_type": "reinforced_concrete",
                    "properties": {
                        "u_value": 0.30,
                        "thickness": 0.20
                    },
                    "load_bearing": True,
                    "dimensions": {"area": 250.0}
                },
                {
                    "structure_id": "s5",
                    "name": "Windows",
                    "type": "OPENING",
                    "material_type": "double_glazed",
                    "properties": {
                        "u_value": 1.8,
                        "shgc": 0.6,
                        "vlt": 0.7
                    },
                    "load_bearing": False,
                    "dimensions": {"total_area": 35.0}
                }
            ]
            return basic_structures
        
        # Otherwise serialize provided structures
        for struct_id, struct in structures.items():
            struct_data = {
                "structure_id": struct.structure_id,
                "name": struct.name,
                "type": struct.structure_type.value if hasattr(struct.structure_type, 'value') else str(struct.structure_type),
                "material_type": getattr(struct, 'material_type', None),
                "properties": getattr(struct, 'properties', {}),
                "load_bearing": getattr(struct, 'load_bearing', False),
                "dimensions": getattr(struct, 'dimensions', {})
            }
            result.append(struct_data)
        return result
    
    def _serialize_layout(self, layout) -> Dict:
        """Serialize layout to JSON-compatible format"""
        if not layout:
            return None
        
        layout_data = {
            "total_area": round(layout.total_area, 2) if layout.total_area else 0.0,
            "used_area": round(getattr(layout, 'used_area', 0.0), 2),
            "circulation_area": round(getattr(layout, 'circulation_area', 0.0), 2),
            "rooms": []
        }
        
        if layout.rooms:
            # Generate simple grid-based positions for rooms
            import math
            num_rooms = len(layout.rooms)
            grid_size = math.ceil(math.sqrt(num_rooms))
            
            for idx, (room_id, room) in enumerate(layout.rooms.items()):
                # Calculate grid position
                row = idx // grid_size
                col = idx % grid_size
                
                # Estimate room dimensions from area
                room_width = math.sqrt(room.area)
                room_height = math.sqrt(room.area)
                
                # Position in grid (with spacing)
                spacing = 2.0  # 2m spacing between rooms
                x = col * (room_width + spacing)
                y = row * (room_height + spacing)
                
                # Generate adjacencies based on grid position
                adjacencies = []
                # Left neighbor
                if col > 0:
                    left_idx = idx - 1
                    if left_idx >= 0:
                        adjacencies.append(f"room_{left_idx + 1}")
                # Right neighbor
                if col < grid_size - 1 and idx + 1 < num_rooms:
                    adjacencies.append(f"room_{idx + 2}")
                # Top neighbor
                if row > 0:
                    top_idx = idx - grid_size
                    if top_idx >= 0:
                        adjacencies.append(f"room_{top_idx + 1}")
                # Bottom neighbor
                if row < grid_size - 1:
                    bottom_idx = idx + grid_size
                    if bottom_idx < num_rooms:
                        adjacencies.append(f"room_{bottom_idx + 1}")
                
                room_data = {
                    "room_id": room_id,
                    "name": room.name,
                    "room_type": room.room_type,
                    "area": round(room.area, 2),
                    "height": round(room.height, 2),
                    "function_id": getattr(room, 'function_id', None),
                    "position": {
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "width": round(room_width, 2),
                        "height": round(room_height, 2)
                    },
                    "adjacencies": adjacencies
                }
                layout_data["rooms"].append(room_data)
        
        # Add layout metrics if available
        if hasattr(layout, 'metrics'):
            layout_data["metrics"] = {
                "compactness": round(layout.metrics.get('compactness', 0.0), 3),
                "circulation_efficiency": round(layout.metrics.get('circulation_efficiency', 0.0), 3),
                "adjacency_satisfaction": round(layout.metrics.get('adjacency_satisfaction', 0.0), 3)
            }
        
        return layout_data
    
    def save_tree_structure(self):
        """Save complete tree structure to JSON"""
        tree_file = self.output_dir / "exploration_tree.json"
        with open(tree_file, 'w', encoding='utf-8') as f:
            json.dump(self.tree_structure, f, indent=2)
        logger.info(f"Saved exploration tree to {tree_file}")
    
    def get_node_path(self, node_id: str) -> List[str]:
        """Get path from root to node"""
        path = []
        current = node_id
        
        while current:
            path.insert(0, current)
            rel = self.tree_structure["relationships"].get(current)
            if not rel:
                break
            current = rel["parent"]
        
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        stats = {
            "total_nodes": len(self.node_registry),
            "nodes_by_level": {
                level: len(nodes) 
                for level, nodes in self.tree_structure["levels"].items()
            },
            "pruned_count": len(self.tree_structure["pruned_nodes"]),
            "pruning_rate": len(self.tree_structure["pruned_nodes"]) / max(len(self.node_registry), 1),
            "max_depth": max((int(l) for l in self.tree_structure["levels"].keys() if l.isdigit()), default=0)
        }
        return stats
