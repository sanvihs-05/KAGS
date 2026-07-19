"""
Maps Finnish floor plan embeddings to FBSL (Function-Behavior-Structure-Layout) components

✅ FIXED VERSION: Now creates Layout with Rooms from precedents
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .fbsl_models import (
    FBSLLayoutNode, Function, Behavior, Structure, Room, Layout,
    FunctionCategory, BehaviorCategory, StructureType, NodeType
)

logger = logging.getLogger(__name__)

class FinnishFBSLMapper:
    """
    Maps Finnish floor plan data to FBSL ontology
    
    ✅ CRITICAL FIX: Now creates Layout with Rooms from precedents
    """
    
    def __init__(self, finnish_embeddings):
        self.finnish_embeddings = finnish_embeddings
        
        # Finnish room type to FBSL Function mapping
        self.room_to_function = {
            # Bedrooms
            'mh': {'name': 'provide_sleeping_space', 'category': FunctionCategory.SPATIAL, 
                   'priority': 0.9, 'activities': ['sleeping', 'resting', 'privacy']},
            'bedroom': {'name': 'provide_sleeping_space', 'category': FunctionCategory.SPATIAL,
                       'priority': 0.9, 'activities': ['sleeping', 'resting']},
            
            # Living rooms
            'oh': {'name': 'provide_social_space', 'category': FunctionCategory.SOCIAL,
                   'priority': 0.85, 'activities': ['relaxation', 'entertainment', 'socializing']},
            'olohuone': {'name': 'provide_social_space', 'category': FunctionCategory.SOCIAL,
                        'priority': 0.85, 'activities': ['relaxation', 'socializing']},
            'lasihuone': {'name': 'provide_sunroom', 'category': FunctionCategory.ENVIRONMENTAL,
                         'priority': 0.6, 'activities': ['relaxation', 'natural_light']},
            
            # Kitchens
            'keittiö': {'name': 'provide_food_preparation', 'category': FunctionCategory.TECHNICAL,
                       'priority': 0.95, 'activities': ['cooking', 'food_storage', 'dining']},
            'tupakeittiö': {'name': 'provide_kitchenette', 'category': FunctionCategory.TECHNICAL,
                           'priority': 0.7, 'activities': ['basic_cooking', 'dining']},
            'k': {'name': 'provide_food_preparation', 'category': FunctionCategory.TECHNICAL,
                  'priority': 0.95, 'activities': ['cooking']},
            
            # Bathrooms
            'kh': {'name': 'provide_bathing_facilities', 'category': FunctionCategory.TECHNICAL,
                   'priority': 0.9, 'activities': ['bathing', 'hygiene']},
            'wc': {'name': 'provide_sanitation', 'category': FunctionCategory.TECHNICAL,
                   'priority': 0.85, 'activities': ['sanitation']},
            'wc-pe': {'name': 'provide_toilet_washing', 'category': FunctionCategory.TECHNICAL,
                     'priority': 0.85, 'activities': ['sanitation', 'washing']},
            
            # Utility
            'vh': {'name': 'provide_utility_space', 'category': FunctionCategory.TECHNICAL,
                   'priority': 0.6, 'activities': ['laundry', 'cleaning', 'storage']},
            
            # Storage
            'cl': {'name': 'provide_storage', 'category': FunctionCategory.SPATIAL,
                   'priority': 0.5, 'activities': ['storage']},
            'cwh': {'name': 'provide_clothes_storage', 'category': FunctionCategory.SPATIAL,
                   'priority': 0.55, 'activities': ['clothes_storage', 'dressing']},
            'varasto': {'name': 'provide_storage', 'category': FunctionCategory.SPATIAL,
                       'priority': 0.5, 'activities': ['storage']},
            
            # Circulation
            'eteinen': {'name': 'provide_entry_circulation', 'category': FunctionCategory.SPATIAL,
                       'priority': 0.7, 'activities': ['entry', 'circulation', 'transition']},
            'käytävä': {'name': 'provide_circulation', 'category': FunctionCategory.SPATIAL,
                       'priority': 0.65, 'activities': ['circulation']},
            
            # Special
            'sauna': {'name': 'provide_sauna', 'category': FunctionCategory.SOCIAL,
                     'priority': 0.75, 'activities': ['bathing', 'relaxation', 'wellness']},
            'parveke': {'name': 'provide_outdoor_space', 'category': FunctionCategory.ENVIRONMENTAL,
                       'priority': 0.6, 'activities': ['outdoor_access', 'fresh_air']},
            'ulkotila': {'name': 'provide_outdoor_space', 'category': FunctionCategory.ENVIRONMENTAL,
                        'priority': 0.6, 'activities': ['outdoor_access']},
        }
        
        # Room type to typical behaviors
        self.room_to_behaviors = {
            'bedroom': [
                {'metric': 'min_area', 'value': 9.0, 'unit': 'sqm', 'category': BehaviorCategory.SPATIAL},
                {'metric': 'noise_level', 'value': 35.0, 'unit': 'dB', 'category': BehaviorCategory.ACOUSTIC},
                {'metric': 'privacy_level', 'value': 0.9, 'unit': 'ratio', 'category': BehaviorCategory.SPATIAL},
            ],
            'living_room': [
                {'metric': 'min_area', 'value': 15.0, 'unit': 'sqm', 'category': BehaviorCategory.SPATIAL},
                {'metric': 'daylight_factor', 'value': 2.0, 'unit': '%', 'category': BehaviorCategory.LIGHTING},
                {'metric': 'ventilation_rate', 'value': 0.5, 'unit': 'ACH', 'category': BehaviorCategory.VENTILATION},
            ],
            'kitchen': [
                {'metric': 'min_area', 'value': 6.0, 'unit': 'sqm', 'category': BehaviorCategory.SPATIAL},
                {'metric': 'ventilation_rate', 'value': 10.0, 'unit': 'ACH', 'category': BehaviorCategory.VENTILATION},
                {'metric': 'task_lighting', 'value': 500.0, 'unit': 'lux', 'category': BehaviorCategory.LIGHTING},
            ],
            'bathroom': [
                {'metric': 'min_area', 'value': 2.5, 'unit': 'sqm', 'category': BehaviorCategory.SPATIAL},
                {'metric': 'humidity_control', 'value': 0.8, 'unit': 'ratio', 'category': BehaviorCategory.VENTILATION},
                {'metric': 'water_resistant', 'value': 1.0, 'unit': 'ratio', 'category': BehaviorCategory.STRUCTURAL},
            ],
        }
        
        # Default room areas (sqm) - Finnish building code based
        self.default_areas = {
            'bedroom': 12.0,
            'living_room': 20.0,
            'kitchen': 10.0,
            'bathroom': 6.0,
            'toilet': 4.0,
            'closet': 2.0,
            'storage': 3.0,
            'sauna': 6.0,
            'balcony': 5.0,
            'hallway': 4.0,
            'entry': 3.0,
            'utility': 4.0,
            'room': 10.0
        }
        
        logger.info("✓ Finnish FBSL Mapper initialized")
    
    def create_fbsl_node_from_query(self, query: str, max_rooms: int = 10) -> FBSLLayoutNode:
        """
        Create FBSL node from natural language query using Finnish embeddings
        
        ✅ CRITICAL FIX: Now creates Layout with Rooms from precedents
        
        Args:
            query: User query like "3 bedroom apartment with kitchen and bathroom"
            max_rooms: Maximum number of rooms to include
        
        Returns:
            FBSLLayoutNode with populated functions, behaviors, structures, AND layout with rooms
        """
        # Create root node
        node = FBSLLayoutNode(
            node_type=NodeType.PROBLEM,
            generation_level=0
        )
        
        # ✅ NEW: Create Layout object
        layout = Layout()
        layout.configuration_name = "Finnish Precedent Layout"
        
        # Encode query
        try:
            query_embedding = self.finnish_embeddings.embedding_model.encode(query)
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}, using zero embedding")
            query_embedding = np.zeros(384)
        
        # Search for relevant spaces
        try:
            similar_spaces = self.finnish_embeddings.search_similar_by_embedding(
                query_embedding,
                embedding_type='composite',
                top_k=max_rooms * 3  # Get more to filter
            )
        except Exception as e:
            logger.warning(f"Search failed: {e}, using empty results")
            similar_spaces = []
        
        # Extract room types from query
        query_lower = query.lower()
        requested_rooms = self._extract_requested_rooms(query_lower)
        
        logger.info(f"Query: {query[:50]}...")
        logger.info(f"Requested rooms: {requested_rooms}")
        logger.info(f"Found {len(similar_spaces)} similar spaces")
        
        # Populate FBSL components + rooms
        added_room_types = {}  # room_type -> count
        
        for i, space in enumerate(similar_spaces):
            if len(node.functions) >= max_rooms:
                break
            
            room_type = space.get('room_type', 'room').lower()
            finnish_type = space.get('finnish_type', room_type).lower()
            
            # Map to standard type
            standard_type = self._map_to_standard_type(finnish_type, room_type)
            
            # Check if this room type was requested or is essential
            is_requested = any(req in standard_type or req in room_type for req in requested_rooms)
            is_essential = standard_type in ['bathroom', 'kitchen', 'bedroom', 'living_room']
            similarity = space.get('similarity', 0.0)
            
            should_include = is_requested or (is_essential and similarity > 0.3) or similarity > 0.6
            
            if not should_include:
                continue
            
            # Allow multiple bedrooms/bathrooms, limit others to 1
            if standard_type in ['bedroom', 'bathroom']:
                room_count = added_room_types.get(standard_type, 0)
                if room_count >= 3:
                    continue
                added_room_types[standard_type] = room_count + 1
                room_number = str(room_count + 1)
            else:
                if standard_type in added_room_types:
                    continue
                added_room_types[standard_type] = 1
                room_number = "1"
            
            # Create Function
            function = self._create_function_from_space(space, standard_type)
            node.add_function(function)
            
            # Create Behaviors
            behaviors = self._create_behaviors_from_space(space, function.function_id, standard_type)
            for behavior in behaviors:
                node.add_behavior(behavior)
            
            # Create Structure suggestions
            structures = self._create_structures_from_space(space, function.function_id, standard_type)
            for structure in structures:
                node.add_structure(structure)
            
            # ✅ NEW: Create Room and add to layout
            room_area = space.get('area', self.default_areas.get(standard_type, 10.0))
            if room_area <= 0 or room_area > 1000:  # Sanity check
                room_area = self.default_areas.get(standard_type, 10.0)
            
            room = Room(
                name=f"{standard_type.replace('_', ' ').title()} {room_number}",
                room_type=standard_type,
                room_number=room_number,
                function_id=function.function_id,
                area=room_area,
                height=3.0
            )
            room.calculate_volume()
            layout.rooms[room.room_id] = room
            
            logger.debug(f"Added room: {room.name} ({room_area:.1f} m²)")
        
        # ✅ CRITICAL: Ensure at least one room exists
        if not layout.rooms:
            logger.warning("No rooms created from precedents, adding default living room")
            func = Function(
                name="provide_living_room",
                category=FunctionCategory.SPATIAL,
                description="Living space",
                priority=0.8,
                activities=['living', 'relaxation'],
                spatial_requirements={
                    'preferred_area': 20.0,
                    'min_area': 15.0,
                    'max_area': 30.0
                }
            )
            node.add_function(func)
            
            behav = Behavior(
                category=BehaviorCategory.SPATIAL,
                metric_name="living_room_area",
                metric_unit="sqm",
                target_value=20.0,
                actual_value=18.0,
                tolerance=0.2,
                derived_from_function=func.function_id
            )
            node.add_behavior(behav)
            
            room = Room(
                name="Living Room",
                room_type="living_room",
                room_number="1",
                function_id=func.function_id,
                area=20.0,
                height=3.0
            )
            room.calculate_volume()
            layout.rooms[room.room_id] = room
        
        # ✅ Calculate layout metrics
        layout.total_area = sum(r.area for r in layout.rooms.values())
        layout.used_area = layout.total_area
        layout.calculate_metrics()
        
        # ✅ Assign layout to node
        node.layout = layout
        
        logger.info(
            f"✓ Created FBSL node with {len(node.functions)} functions, "
            f"{len(node.behaviors)} behaviors, {len(node.structures)} structures, "
            f"{len(layout.rooms)} rooms"
        )
        
        return node
    
    def _extract_requested_rooms(self, query: str) -> List[str]:
        """Extract requested room types from query"""
        room_keywords = {
            'bedroom': ['bedroom', 'bed room', 'mh', 'makuuhuone', 'br'],
            'living_room': ['living', 'living room', 'oh', 'olohuone', 'lounge'],
            'kitchen': ['kitchen', 'keittiö', 'k', 'cook'],
            'bathroom': ['bathroom', 'bath', 'kh', 'kylpyhuone', 'wc', 'toilet'],
            'storage': ['storage', 'closet', 'varasto', 'cl'],
            'sauna': ['sauna'],
            'balcony': ['balcony', 'parveke', 'terrace'],
            'study': ['study', 'office', 'workspace'],
            'hallway': ['hallway', 'corridor', 'entry'],
        }
        
        requested = []
        for room_type, keywords in room_keywords.items():
            if any(keyword in query for keyword in keywords):
                requested.append(room_type)
        
        return requested if requested else ['living_room']
    
    def _map_to_standard_type(self, finnish_type: str, room_type: str) -> str:
        """Map Finnish room type to standard type"""
        mapping = {
            'mh': 'bedroom',
            'makuuhuone': 'bedroom',
            'oh': 'living_room',
            'olohuone': 'living_room',
            'lasihuone': 'living_room',
            'keittiö': 'kitchen',
            'tupakeittiö': 'kitchen',
            'k': 'kitchen',
            'kh': 'bathroom',
            'kylpyhuone': 'bathroom',
            'wc': 'bathroom',
            'wc-pe': 'bathroom',
            'vh': 'utility',
            'cl': 'closet',
            'cwh': 'closet',
            'varasto': 'storage',
            'eteinen': 'entry',
            'käytävä': 'hallway',
            'sauna': 'sauna',
            'parveke': 'balcony',
            'ulkotila': 'balcony',
            'bedroom': 'bedroom',
            'living_room': 'living_room',
            'living': 'living_room',
            'kitchen': 'kitchen',
            'bathroom': 'bathroom',
        }
        
        return mapping.get(finnish_type, mapping.get(room_type, room_type))
    
    def _create_function_from_space(self, space: Dict, standard_type: str) -> Function:
        """Create FBSL Function from Finnish space data"""
        room_type = standard_type
        finnish_type = space.get('finnish_type', '').lower()
        
        # Get function template
        func_template = self.room_to_function.get(
            finnish_type,
            self.room_to_function.get(room_type, {
                'name': f'provide_{room_type}',
                'category': FunctionCategory.SPATIAL,
                'priority': 0.5,
                'activities': ['general_use']
            })
        )
        
        preferred_area = self.default_areas.get(room_type, 10.0)
        
        function = Function(
            name=func_template['name'],
            category=func_template['category'],
            description=f"{room_type.replace('_', ' ').title()} space for {', '.join(func_template['activities'])}",
            priority=func_template['priority'],
            activities=func_template['activities'],
            spatial_requirements={
                'min_area': preferred_area * 0.8,
                'preferred_area': preferred_area,
                'max_area': preferred_area * 1.5,
                'height': 3.0
            }
        )
        
        # Store embedding for similarity comparisons
        if 'embedding' in space:
            function.embedding = space['embedding']
        
        return function
    
    def _create_behaviors_from_space(self, space: Dict, function_id: str, standard_type: str) -> List[Behavior]:
        """Create FBSL Behaviors from Finnish space data"""
        
        # Get behavior templates
        behavior_templates = self.room_to_behaviors.get(standard_type, [
            {'metric': 'min_area', 'value': 10.0, 'unit': 'sqm', 'category': BehaviorCategory.SPATIAL}
        ])
        
        behaviors = []
        for template in behavior_templates:
            behavior = Behavior(
                category=template['category'],
                metric_name=template['metric'],
                metric_unit=template['unit'],
                target_value=template['value'],
                actual_value=template['value'] * 0.9,  # Initial estimate
                tolerance=0.2,
                derived_from_function=function_id
            )
            behaviors.append(behavior)
        
        return behaviors
    
    def _create_structures_from_space(self, space: Dict, function_id: str, standard_type: str) -> List[Structure]:
        """Create FBSL Structures from Finnish space data"""
        room_type = standard_type
        
        structures = []
        
        # Basic partition walls
        wall = Structure(
            name=f"{room_type}_partition",
            structure_type=StructureType.PARTITION,
            material_type="gypsum_board",
            dimensions={'thickness': 0.1},
            fire_rating="30min"
        )
        structures.append(wall)
        
        # Special structures based on room type
        if room_type in ['bathroom', 'toilet']:
            waterproof = Structure(
                name=f"{room_type}_waterproofing",
                structure_type=StructureType.WALL,
                material_type="waterproof_membrane",
                dimensions={'thickness': 0.005}
            )
            structures.append(waterproof)
        
        if room_type in ['kitchen']:
            vent = Structure(
                name=f"{room_type}_ventilation",
                structure_type=StructureType.MEP,
                material_type="steel_duct",
                dimensions={'diameter': 0.15}
            )
            structures.append(vent)
        
        return structures
    
    def _estimate_min_area(self, room_type: str) -> float:
        """Estimate minimum area based on Finnish building codes"""
        return self.default_areas.get(room_type, 10.0) * 0.8
    
    def _estimate_preferred_area(self, room_type: str) -> float:
        """Estimate preferred area"""
        return self.default_areas.get(room_type, 10.0)


def demonstrate_finnish_fbsl_mapping():
    """Demonstrate Finnish to FBSL mapping"""
    from backend.database.vector_store import VectorStoreManager
    
    print("🏠 Finnish Floor Plan to FBSL Mapping Demo")
    print("=" * 60)
    
    # Initialize
    vs = VectorStoreManager()
    
    if vs.finnish_embeddings is None:
        print("❌ Finnish embeddings not loaded")
        return
    
    mapper = FinnishFBSLMapper(vs.finnish_embeddings)
    
    # Test queries
    test_queries = [
        "2 bedroom apartment with kitchen and bathroom",
        "studio apartment with kitchenette",
        "3 bedroom house with sauna and balcony",
        "apartment with living room and 2 bathrooms"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        # Create FBSL node
        node = mapper.create_fbsl_node_from_query(query, max_rooms=8)
        
        print(f"✓ Created FBSL Node:")
        print(f"  • Functions: {len(node.functions)}")
        for func_id, func in list(node.functions.items())[:5]:
            print(f"    - {func.name} (priority: {func.priority})")
        
        print(f"  • Behaviors: {len(node.behaviors)}")
        for behav_id, behav in list(node.behaviors.items())[:5]:
            print(f"    - {behav.metric_name}: {behav.target_value} {behav.metric_unit}")
        
        print(f"  • Structures: {len(node.structures)}")
        for struct_id, struct in list(node.structures.items())[:5]:
            print(f"    - {struct.name} ({struct.structure_type.value})")
        
        # ✅ NEW: Show rooms
        print(f"  • Rooms: {len(node.layout.rooms) if node.layout else 0}")
        if node.layout and node.layout.rooms:
            for room_id, room in list(node.layout.rooms.items())[:5]:
                print(f"    - {room.name}: {room.area:.1f} m² (type: {room.room_type})")
    
    print("\n✅ Finnish-FBSL mapping demonstration complete!")


if __name__ == "__main__":
    demonstrate_finnish_fbsl_mapping()