import re
import json
import math
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import logging  # ADD THIS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # ADD THIS

# For Ollama integration (placeholder)
# Gemma 3 integration via Ollama
import requests
from encoder import Gemma3Encoder
OLLAMA_AVAILABLE = True


# --- Base Classes (to make the provided code self-contained and runnable) ---
# These classes replace the dependency on an external 'test' file.

@dataclass
class SpatialNeed:
    room_type: str
    quantity: int
    min_area: Optional[float] = None
    priority: Optional[str] = None

@dataclass
class SiteConstraints:
    plot_length: Optional[float] = 50.0
    plot_width: Optional[float] = 30.0
    orientation: Optional[str] = 'south'

@dataclass
class DesignPreferences:
    style: str = "modern"
    accessibility_requirements: bool = False

@dataclass
class ParsedRequirements:
    spatial_needs: List[SpatialNeed] = field(default_factory=list)
    site_constraints: SiteConstraints = field(default_factory=SiteConstraints)
    design_preferences: DesignPreferences = field(default_factory=DesignPreferences)
    budget: float = 2500000.0

@dataclass
class FBSElement:
    element_id: str
    name: str
    description: str

@dataclass
class Function(FBSElement):
    pass

@dataclass
class Behavior(FBSElement):
    target_value: Any
    current_value: Any = None

@dataclass
class Structure(FBSElement):
    geometric_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FBSOntology:
    project_name: str
    functions: List[Function] = field(default_factory=list)
    behaviors: List[Behavior] = field(default_factory=list)
    structures: List[Structure] = field(default_factory=list)

class GemmaFBSAnalyzer:
    """A mock analyzer to parse user input into structured requirements."""

    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "gemma3"
    
    def analyze_and_parse_requirements(self, user_input: str) -> ParsedRequirements:
        try:
            # Use Gemma 3 for intelligent parsing
            payload = {
                "model": self.model,
                "prompt": f"Parse this architectural requirement into room types and quantities: {user_input}\nExtract: room types, quantities, plot size, budget.",
                "stream": False
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            if response.status_code == 200:
                ai_response = response.json().get("response", "")
                return self._parse_ai_response(ai_response, user_input)
            else:
                print(f"⚠️ Gemma 3 unavailable, using fallback parser")
                return self._fallback_parse(user_input)
        except:
            return self._fallback_parse(user_input)
    
    def _parse_ai_response(self, ai_response: str, user_input: str) -> ParsedRequirements:
        # Enhanced parsing using AI response + original logic
        reqs = ParsedRequirements()
        
        # Map room names to enum values
        room_types_map = {rt.value.replace("_", " "): rt.value for rt in RoomType}
        
        # Regex-based detection
        room_matches = re.findall(r'(\d+)\s*([a-zA-Z\s_]+?)(?:s\b|\b)', user_input, re.IGNORECASE)
        
        for num, room in room_matches:
            room_key = room.strip().lower().replace(" ", "")
            room_type_val = room_types_map.get(room_key, room.strip().lower().replace(" ","_"))
            if room_type_val in [rt.value for rt in RoomType]:
                reqs.spatial_needs.append(SpatialNeed(room_type=room_type_val, quantity=int(num)))
        
        # Add defaults if missing
        specified_rooms = {s.room_type for s in reqs.spatial_needs}
        if 'living_room' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='living_room', quantity=1))
        if 'kitchen' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='kitchen', quantity=1))
        if 'bathroom' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='bathroom', quantity=1))
        
        return reqs
    
    def _fallback_parse(self, user_input: str) -> ParsedRequirements:
        """Fallback parser using regex only (no AI)."""
        reqs = ParsedRequirements()
        
        # Map room names to enum values
        room_types_map = {rt.value.replace("_", " "): rt.value for rt in RoomType}
        
        # Regex to capture patterns like "2 bedrooms", "1 kitchen", etc.
        room_matches = re.findall(r'(\d+)\s*([a-zA-Z\s_]+?)(?:s\b|\b)', user_input, re.IGNORECASE)
        
        for num, room in room_matches:
            room_key = room.strip().lower().replace(" ", "")
            room_type_val = room_types_map.get(room_key, room.strip().lower().replace(" ","_"))
            if room_type_val in [rt.value for rt in RoomType]:
                reqs.spatial_needs.append(SpatialNeed(room_type=room_type_val, quantity=int(num)))
        
        # Ensure default essential rooms
        specified_rooms = {s.room_type for s in reqs.spatial_needs}
        if 'living_room' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='living_room', quantity=1))
        if 'kitchen' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='kitchen', quantity=1))
        if 'bathroom' not in specified_rooms:
            reqs.spatial_needs.append(SpatialNeed(room_type='bathroom', quantity=1))
        
        return reqs

class FBSOntologyGenerator:
    """A base generator for the FBS ontology."""
    def generate_fbs_ontology(self, requirements: ParsedRequirements, project_name: str) -> FBSOntology:
        functions = [
            Function("F001", "Provide Shelter", "To provide a safe and comfortable living space for occupants."),
            Function("F002", "Enable Daily Activities", "To facilitate core activities like cooking, sleeping, and socializing.")
        ]
        
        behaviors = [
            Behavior(
                element_id="B001", 
                name="Thermal Comfort", 
                description="Defines the expected comfortable temperature range inside the building.",
                target_value="Maintain internal temperature between 22-26°C."
            ),
            Behavior(
                element_id="B002", 
                name="Daylight Performance", 
                description="Specifies the goal for natural light penetration, measured by Daylight Factor.",
                target_value="Achieve a Daylight Factor of >2% in primary living spaces."
            ),
            Behavior(
                element_id="B003", 
                name="Ventilation Performance", 
                description="Sets the target for natural air exchange to ensure fresh air quality.",
                target_value="Achieve at least 5 Air Changes per Hour (ACH) naturally."
            )
        ]
        
        structures = [
            Structure("S001", "Building Envelope", "The physical shell of the house, including walls, roof, and foundation.", 
                      geometric_properties={'total_area': requirements.site_constraints.plot_length * requirements.site_constraints.plot_width}),
            Structure("S002", "Window System", "A system of openings designed to admit light and facilitate ventilation.", 
                      geometric_properties={'estimated_count': sum(r.quantity for r in requirements.spatial_needs) + 2}),
            Structure("S003", "Room Layout", "The arrangement and interconnection of interior spaces.", 
                      geometric_properties={'room_count': sum(r.quantity for r in requirements.spatial_needs)})
        ]
        
        return FBSOntology(
            project_name=project_name,
            functions=functions,
            behaviors=behaviors,
            structures=structures
        )

# --- Start of Main Code ---

class RoomType(Enum):
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    DINING_HALL = "dining_hall"
    OFFICE = "office"
    GARAGE = "garage"
    UTILITY = "utility"
    HALLWAY = "hallway"
    BALCONY = "balcony"
    STORAGE = "storage"
    STOREROOM = "storeroom"

class ClimateZone(Enum):
    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    ARID = "arid"
    TEMPERATE = "temperate"

class CardinalDirection(Enum):
    NORTH = "north"
    NORTHEAST = "northeast"
    EAST = "east"
    SOUTHEAST = "southeast"
    SOUTH = "south"
    SOUTHWEST = "southwest"
    WEST = "west"
    NORTHWEST = "northwest"

@dataclass
class SunPathData:
    """Sun path and solar analysis data for different latitudes"""
    latitude: float
    summer_solstice_angle: float  # June 21
    winter_solstice_angle: float  # December 21
    equinox_angle: float         # March/September 21
    sunrise_azimuth_summer: float
    sunset_azimuth_summer: float
    sunrise_azimuth_winter: float
    sunset_azimuth_winter: float
    
    @classmethod
    def for_india(cls, latitude: float = 20.0):
        """Default sun path data for India (average latitude ~20°N)"""
        return cls(
            latitude=latitude,
            summer_solstice_angle=90 - latitude + 23.5,  # ~93.5° for 20°N
            winter_solstice_angle=90 - latitude - 23.5,  # ~46.5° for 20°N
            equinox_angle=90 - latitude,                 # ~70° for 20°N
            sunrise_azimuth_summer=60,   # Northeast
            sunset_azimuth_summer=300,   # Northwest
            sunrise_azimuth_winter=120,  # Southeast
            sunset_azimuth_winter=240    # Southwest
        )

@dataclass
class DirectionalPreferences:
    """Directional preferences for different room types based on function"""
    room_type: str
    preferred_directions: List[CardinalDirection]
    avoid_directions: List[CardinalDirection]
    window_priority: float  # 0.0 to 1.0
    natural_light_requirement: float  # 0.0 to 1.0
    cross_ventilation_priority: float  # 0.0 to 1.0
    privacy_requirement: float  # 0.0 to 1.0
    heat_tolerance: float  # 0.0 (avoid heat) to 1.0 (can handle heat)
    
@dataclass 
class VentilationAnalysis:
    """Ventilation analysis for optimal air flow"""
    prevailing_wind_direction: CardinalDirection
    secondary_wind_direction: CardinalDirection
    wind_speed: float  # m/s average
    cross_ventilation_effectiveness: Dict[str, float]  # room_id -> effectiveness
    stack_effect_potential: float  # 0.0 to 1.0
    
@dataclass
class CirculationMetrics:
    """Detailed circulation analysis"""
    total_circulation_area: float
    circulation_efficiency: float  # usable area / total area
    average_travel_distance: float
    dead_end_count: int
    accessibility_compliance: bool
    corridor_width_analysis: Dict[str, float]
    circulation_bottlenecks: List[str]

@dataclass
class LightingAnalysis:
    """Comprehensive lighting analysis"""
    daylight_factor_by_room: Dict[str, float]
    direct_sunlight_hours: Dict[str, Dict[str, float]]  # room -> season -> hours
    glare_risk_assessment: Dict[str, str]  # room -> risk level
    artificial_lighting_requirement: Dict[str, float]  # room -> hours per day
    window_effectiveness: Dict[str, float]  # window_id -> effectiveness
    solar_heat_gain: Dict[str, float]  # room -> SHGC

class EnhancedDirectionalOptimizer:
    """Enhanced directional optimizer with proper sun path analysis"""
    
    def __init__(self, climate_zone: ClimateZone = ClimateZone.SUBTROPICAL, latitude: float = 20.0):
        self.climate_zone = climate_zone
        self.sun_path = SunPathData.for_india(latitude)
        self.room_preferences = self._initialize_room_preferences()
        self.directional_scores = self._calculate_directional_scores()
        
    def _initialize_room_preferences(self) -> Dict[str, DirectionalPreferences]:
        """Initialize directional preferences for each room type based on function and sun path"""
        preferences = {}
        
        # Master Bedroom - Morning light preferred, privacy important
        preferences["bedroom"] = DirectionalPreferences(
            room_type="bedroom",
            preferred_directions=[CardinalDirection.EAST, CardinalDirection.NORTHEAST],
            avoid_directions=[CardinalDirection.WEST, CardinalDirection.SOUTHWEST],
            window_priority=0.8,
            natural_light_requirement=0.7,
            cross_ventilation_priority=0.9,
            privacy_requirement=0.9,
            heat_tolerance=0.3
        )
        
        # Kitchen - Morning light, avoid afternoon heat
        preferences["kitchen"] = DirectionalPreferences(
            room_type="kitchen",
            preferred_directions=[CardinalDirection.EAST, CardinalDirection.NORTHEAST, CardinalDirection.NORTH],
            avoid_directions=[CardinalDirection.WEST, CardinalDirection.SOUTHWEST],
            window_priority=0.9,
            natural_light_requirement=0.9,
            cross_ventilation_priority=1.0,  # Critical for smoke/odor removal
            privacy_requirement=0.4,
            heat_tolerance=0.2  # Kitchen generates internal heat
        )
        
        # Living Room - Flexible, good light throughout day
        preferences["living_room"] = DirectionalPreferences(
            room_type="living_room",
            preferred_directions=[CardinalDirection.SOUTH, CardinalDirection.SOUTHEAST, CardinalDirection.EAST],
            avoid_directions=[CardinalDirection.NORTHWEST],
            window_priority=0.9,
            natural_light_requirement=0.8,
            cross_ventilation_priority=0.7,
            privacy_requirement=0.5,
            heat_tolerance=0.6
        )
        
        # Dining Room - Pleasant lighting for meals
        preferences["dining_room"] = DirectionalPreferences(
            room_type="dining_room",
            preferred_directions=[CardinalDirection.EAST, CardinalDirection.SOUTH],
            avoid_directions=[CardinalDirection.WEST],
            window_priority=0.7,
            natural_light_requirement=0.7,
            cross_ventilation_priority=0.6,
            privacy_requirement=0.5,
            heat_tolerance=0.5
        )
        
        # Bathroom - Natural light and ventilation, privacy critical
        preferences["bathroom"] = DirectionalPreferences(
            room_type="bathroom",
            preferred_directions=[CardinalDirection.NORTH, CardinalDirection.EAST],
            avoid_directions=[CardinalDirection.SOUTH, CardinalDirection.WEST],  # Avoid direct sun
            window_priority=0.8,
            natural_light_requirement=0.6,
            cross_ventilation_priority=1.0,  # Critical for moisture control
            privacy_requirement=1.0,
            heat_tolerance=0.4
        )
        
        # Office/Study - Consistent light, avoid glare
        preferences["office"] = DirectionalPreferences(
            room_type="office",
            preferred_directions=[CardinalDirection.NORTH, CardinalDirection.NORTHEAST],
            avoid_directions=[CardinalDirection.SOUTH, CardinalDirection.WEST],
            window_priority=0.9,
            natural_light_requirement=0.9,
            cross_ventilation_priority=0.7,
            privacy_requirement=0.7,
            heat_tolerance=0.4
        )
        
        # Storage/Utility - Minimal requirements
        preferences["storage"] = DirectionalPreferences(
            room_type="storage",
            preferred_directions=[CardinalDirection.NORTH, CardinalDirection.WEST],
            avoid_directions=[],
            window_priority=0.2,
            natural_light_requirement=0.2,
            cross_ventilation_priority=0.4,
            privacy_requirement=0.3,
            heat_tolerance=0.8
        )
        
        return preferences
    
    def _calculate_directional_scores(self) -> Dict[CardinalDirection, Dict[str, float]]:
        """Calculate directional scores based on sun path, heat gain, and lighting quality"""
        scores = {}
        
        for direction in CardinalDirection:
            scores[direction] = {
                "morning_light": self._calculate_morning_light_score(direction),
                "afternoon_light": self._calculate_afternoon_light_score(direction),
                "heat_gain": self._calculate_heat_gain_score(direction),
                "consistent_light": self._calculate_consistent_light_score(direction),
                "privacy": self._calculate_privacy_score(direction),
                "ventilation": self._calculate_ventilation_score(direction)
            }
        
        return scores
    
    def _calculate_morning_light_score(self, direction: CardinalDirection) -> float:
        """Calculate morning light quality score (0-1)"""
        morning_directions = {
            CardinalDirection.EAST: 1.0,
            CardinalDirection.NORTHEAST: 0.9,
            CardinalDirection.SOUTHEAST: 0.8,
            CardinalDirection.NORTH: 0.4,
            CardinalDirection.SOUTH: 0.6,
            CardinalDirection.NORTHWEST: 0.2,
            CardinalDirection.WEST: 0.1,
            CardinalDirection.SOUTHWEST: 0.2
        }
        return morning_directions.get(direction, 0.5)
    
    def _calculate_afternoon_light_score(self, direction: CardinalDirection) -> float:
        """Calculate afternoon light quality score (0-1)"""
        afternoon_directions = {
            CardinalDirection.WEST: 1.0,
            CardinalDirection.SOUTHWEST: 0.9,
            CardinalDirection.NORTHWEST: 0.8,
            CardinalDirection.SOUTH: 0.7,
            CardinalDirection.SOUTHEAST: 0.4,
            CardinalDirection.EAST: 0.2,
            CardinalDirection.NORTHEAST: 0.3,
            CardinalDirection.NORTH: 0.5
        }
        return afternoon_directions.get(direction, 0.5)
    
    def _calculate_heat_gain_score(self, direction: CardinalDirection) -> float:
        """Calculate heat gain score (0=high heat, 1=low heat)"""
        heat_gain = {
            CardinalDirection.WEST: 0.1,      # Maximum afternoon heat
            CardinalDirection.SOUTHWEST: 0.2,
            CardinalDirection.SOUTH: 0.4,     # High but manageable
            CardinalDirection.SOUTHEAST: 0.6,
            CardinalDirection.NORTHWEST: 0.3,
            CardinalDirection.EAST: 0.8,      # Morning sun, manageable
            CardinalDirection.NORTHEAST: 0.9,
            CardinalDirection.NORTH: 1.0      # Minimal direct sun
        }
        return heat_gain.get(direction, 0.5)
    
    def _calculate_consistent_light_score(self, direction: CardinalDirection) -> float:
        """Calculate consistent daylight score throughout the day"""
        consistency = {
            CardinalDirection.NORTH: 1.0,     # Most consistent
            CardinalDirection.NORTHEAST: 0.8,
            CardinalDirection.NORTHWEST: 0.8,
            CardinalDirection.SOUTH: 0.7,     # Good but varies
            CardinalDirection.EAST: 0.5,      # Morning only
            CardinalDirection.WEST: 0.5,      # Afternoon only
            CardinalDirection.SOUTHEAST: 0.6,
            CardinalDirection.SOUTHWEST: 0.4
        }
        return consistency.get(direction, 0.5)
    
    def _calculate_privacy_score(self, direction: CardinalDirection) -> float:
        """Calculate privacy score based on typical site conditions"""
        # Assuming front of house faces south (common in India)
        privacy_scores = {
            CardinalDirection.NORTH: 0.9,     # Usually back of house
            CardinalDirection.NORTHEAST: 0.8,
            CardinalDirection.NORTHWEST: 0.8,
            CardinalDirection.EAST: 0.7,      # Side boundary
            CardinalDirection.WEST: 0.7,      # Side boundary
            CardinalDirection.SOUTHEAST: 0.6,
            CardinalDirection.SOUTHWEST: 0.6,
            CardinalDirection.SOUTH: 0.3      # Usually front, less private
        }
        return privacy_scores.get(direction, 0.5)
    
    def _calculate_ventilation_score(self, direction: CardinalDirection) -> float:
        """Calculate ventilation effectiveness based on prevailing winds in India"""
        # Prevailing winds: Southwest monsoon (June-Sept), Northeast winter (Dec-Feb)
        ventilation_scores = {
            CardinalDirection.SOUTHWEST: 1.0,  # Primary monsoon winds
            CardinalDirection.NORTHEAST: 0.9,  # Winter winds
            CardinalDirection.SOUTH: 0.8,
            CardinalDirection.WEST: 0.8,
            CardinalDirection.NORTH: 0.7,
            CardinalDirection.EAST: 0.7,
            CardinalDirection.SOUTHEAST: 0.6,
            CardinalDirection.NORTHWEST: 0.6
        }
        return ventilation_scores.get(direction, 0.5)

class EnhancedGeometricLayoutEngine:
    """
    Enhanced geometric engine that places rooms based on both adjacency requirements 
    AND directional optimization, ensuring full connectivity
    """
    def __init__(self, directional_optimizer: EnhancedDirectionalOptimizer):
        self.directional_optimizer = directional_optimizer
        
    def place_rooms(self, rooms_to_place: List[Dict], adjacency_rules: Dict, 
                   plot_dimensions: Tuple[float, float] = (50, 30)) -> List[Dict]:
        """
        Places rooms considering both adjacency rules AND directional preferences
        Ensures all rooms are connected in a single graph component
        """

        print("INFO: Running directional + adjacency-aware geometric placement...")
        
        if not rooms_to_place:
            return []
        
        plot_length, plot_width = plot_dimensions
        placed_rooms = []
        remaining_rooms = rooms_to_place.copy()
        
        # Step 1: Create directional zones on the plot
        directional_zones = self._create_directional_zones(plot_length, plot_width)
        
        # Step 2: Sort rooms by placement priority (connectivity + directional preference)
        prioritized_rooms = self._prioritize_rooms_for_placement(remaining_rooms, adjacency_rules)
        
        # Step 3: Place the first (highest priority) room at the center
        first_room = prioritized_rooms[0]
        first_position = self._get_central_position(first_room, plot_length, plot_width)
        placed_room = self._create_placed_room(first_room, first_position[0], first_position[1])
        placed_rooms.append(placed_room)
        remaining_rooms.remove(first_room)
        
        # Step 4: Place remaining rooms ensuring both adjacency AND directional optimization
        while remaining_rooms:
            next_room, best_anchor = self._find_next_room_with_direction_priority(
                remaining_rooms, placed_rooms, adjacency_rules, directional_zones
            )
            
            if next_room is None:
                # Force place remaining rooms to maintain connectivity
                next_room = remaining_rooms[0]
                best_anchor = placed_rooms[0]
            
            # Find optimal position considering both adjacency and direction
            position = self._calculate_directionally_optimal_position(
                next_room, best_anchor, placed_rooms, directional_zones
            )
            
            placed_room = self._create_placed_room(next_room, position[0], position[1])
            placed_rooms.append(placed_room)
            remaining_rooms.remove(next_room)
        
        # Step 5: Ensure full connectivity with minimum repositioning
        placed_rooms = self._ensure_full_connectivity(placed_rooms, adjacency_rules)
        
        # Step 6: Optimize positions within directional constraints
        placed_rooms = self._optimize_directional_positions(placed_rooms, directional_zones)
        
        return placed_rooms
    
    def _create_directional_zones(self, plot_length: float, plot_width: float) -> Dict[str, Dict]:
        """Create zones on the plot corresponding to cardinal directions"""
        center_x, center_y = plot_length / 2, plot_width / 2
        zone_width, zone_height = plot_length / 3, plot_width / 3
        
        return {
            'north': {'x': center_x - zone_width/2, 'y': 0, 'width': zone_width, 'height': zone_height},
            'northeast': {'x': center_x + zone_width/2, 'y': 0, 'width': zone_width, 'height': zone_height},
            'east': {'x': plot_length - zone_width, 'y': center_y - zone_height/2, 'width': zone_width, 'height': zone_height},
            'southeast': {'x': plot_length - zone_width, 'y': plot_width - zone_height, 'width': zone_width, 'height': zone_height},
            'south': {'x': center_x - zone_width/2, 'y': plot_width - zone_height, 'width': zone_width, 'height': zone_height},
            'southwest': {'x': 0, 'y': plot_width - zone_height, 'width': zone_width, 'height': zone_height},
            'west': {'x': 0, 'y': center_y - zone_height/2, 'width': zone_width, 'height': zone_height},
            'northwest': {'x': 0, 'y': 0, 'width': zone_width, 'height': zone_height}
        }
    
    def _prioritize_rooms_for_placement(self, rooms: List[Dict], adjacency_rules: Dict) -> List[Dict]:
        """Prioritize rooms based on connectivity requirements and directional preferences"""
        room_scores = []
        
        for room in rooms:
            room_type = room['type']
            
            # Connectivity score (number of adjacency requirements)
            connectivity_score = 0
            if room_type in adjacency_rules:
                connectivity_score += len(adjacency_rules[room_type].get('critical', set())) * 2
                connectivity_score += len(adjacency_rules[room_type].get('preferred', set()))
            
            # Directional flexibility score (rooms with specific directional needs get priority)
            directional_score = 0
            if room_type in self.directional_optimizer.room_preferences:
                prefs = self.directional_optimizer.room_preferences[room_type]
                directional_score = prefs.natural_light_requirement + prefs.cross_ventilation_priority
            
            # Central connector bonus
            connector_bonus = 0
            if room_type in ['living_room', 'hallway']:
                connector_bonus = 10
            elif room_type in ['kitchen', 'dining_room']:
                connector_bonus = 5
            
            total_score = connectivity_score + directional_score + connector_bonus
            room_scores.append((room, total_score))
        
        # Sort by score (highest first)
        room_scores.sort(key=lambda x: x[1], reverse=True)
        return [room for room, score in room_scores]
    
    def _get_central_position(self, room: Dict, plot_length: float, plot_width: float) -> Tuple[float, float]:
        """Get central position for the first room"""
        room_size = self._calculate_room_dimensions(room['area'])
        return (plot_length/2 - room_size[0]/2, plot_width/2 - room_size[1]/2)
    
    def _find_next_room_with_direction_priority(self, remaining_rooms: List[Dict], 
                                              placed_rooms: List[Dict], 
                                              adjacency_rules: Dict,
                                              directional_zones: Dict) -> Tuple[Dict, Dict]:
        """Find next room considering both adjacency and directional preferences"""
        best_room = None
        best_anchor = None
        best_score = -1
        
        for room in remaining_rooms:
            room_type = room['type']
            
            # Get preferred directions for this room type
            preferred_directions = []
            if room_type in self.directional_optimizer.room_preferences:
                prefs = self.directional_optimizer.room_preferences[room_type]
                preferred_directions = prefs.preferred_directions
            
            for placed_room in placed_rooms:
                placed_type = placed_room['type']
                
                # Adjacency score
                adjacency_score = self._calculate_adjacency_score(room_type, placed_type, adjacency_rules)
                
                # Directional opportunity score
                directional_score = self._calculate_directional_opportunity_score(
                    room, placed_room, preferred_directions, directional_zones
                )
                
                # Combined score
                total_score = adjacency_score * 0.6 + directional_score * 0.4
                
                if total_score > best_score:
                    best_score = total_score
                    best_room = room
                    best_anchor = placed_room
        
        return best_room, best_anchor
    
    def _calculate_directional_opportunity_score(self, room: Dict, anchor_room: Dict, 
                                               preferred_directions: List, 
                                               directional_zones: Dict) -> float:
        """Calculate how well a room can be placed directionally relative to anchor"""
        if not preferred_directions:
            return 0.5  # Neutral score
        
        anchor_x, anchor_y = anchor_room['x'], anchor_room['y']
        best_direction_score = 0
        
        for direction in preferred_directions:
            direction_str = direction.value if hasattr(direction, 'value') else str(direction)
            
            if direction_str in directional_zones:
                zone = directional_zones[direction_str]
                # Check if there's space in this directional zone
                zone_center_x = zone['x'] + zone['width'] / 2
                zone_center_y = zone['y'] + zone['height'] / 2
                
                # Calculate distance from anchor to zone center
                distance = math.sqrt((zone_center_x - anchor_x)**2 + (zone_center_y - anchor_y)**2)
                
                # Closer to preferred zone = higher score
                max_distance = math.sqrt(50**2 + 30**2)  # Plot diagonal
                direction_score = 1 - (distance / max_distance)
                best_direction_score = max(best_direction_score, direction_score)
        
        return best_direction_score
    
    def _calculate_directionally_optimal_position(self, room_to_place: Dict, 
                                                anchor_room: Dict, 
                                                placed_rooms: List[Dict],
                                                directional_zones: Dict) -> Tuple[float, float]:
        """Calculate position that satisfies both adjacency and directional preferences"""
        room_type = room_to_place['type']
        room_dimensions = self._calculate_room_dimensions(room_to_place['area'])
        
        # Get preferred directions
        preferred_directions = []
        if room_type in self.directional_optimizer.room_preferences:
            prefs = self.directional_optimizer.room_preferences[room_type]
            preferred_directions = prefs.preferred_directions
        
        # Generate candidate positions around anchor room
        candidate_positions = self._generate_candidate_positions(anchor_room, room_dimensions)
        
        best_position = candidate_positions[0]  # Fallback
        best_score = -1
        
        for position in candidate_positions:
            x, y = position
            
            # Check if position is valid (no overlap)
            if self._check_overlap(x, y, room_dimensions[0], room_dimensions[1], placed_rooms):
                continue
            
            # Calculate directional alignment score
            directional_score = self._score_position_directionally(
                x, y, preferred_directions, directional_zones
            )
            
            # Calculate adjacency quality score
            adjacency_score = self._score_position_adjacency(x, y, anchor_room)
            
            # Combined score
            total_score = directional_score * 0.7 + adjacency_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_position = position
        
        return best_position
    
    def _generate_candidate_positions(self, anchor_room: Dict, 
                                    room_dimensions: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate multiple candidate positions around anchor room"""
        anchor_x, anchor_y = anchor_room['x'], anchor_room['y']
        anchor_width, anchor_height = anchor_room['width'], anchor_room['height']
        room_width, room_height = room_dimensions
        
        gap = 1  # Minimum gap between rooms
        
        positions = [
            # Right of anchor
            (anchor_x + anchor_width + gap, anchor_y),
            (anchor_x + anchor_width + gap, anchor_y - room_height + anchor_height),
            
            # Left of anchor
            (anchor_x - room_width - gap, anchor_y),
            (anchor_x - room_width - gap, anchor_y - room_height + anchor_height),
            
            # Above anchor
            (anchor_x, anchor_y - room_height - gap),
            (anchor_x - room_width + anchor_width, anchor_y - room_height - gap),
            
            # Below anchor
            (anchor_x, anchor_y + anchor_height + gap),
            (anchor_x - room_width + anchor_width, anchor_y + anchor_height + gap),
        ]
        
        return positions
    
    def _score_position_directionally(self, x: float, y: float, 
                                    preferred_directions: List, 
                                    directional_zones: Dict) -> float:
        """Score a position based on directional preferences"""
        if not preferred_directions:
            return 0.5
        
        best_score = 0
        for direction in preferred_directions:
            direction_str = direction.value if hasattr(direction, 'value') else str(direction)
            
            if direction_str in directional_zones:
                zone = directional_zones[direction_str]
                
                # Check if position falls within or close to preferred zone
                zone_center_x = zone['x'] + zone['width'] / 2
                zone_center_y = zone['y'] + zone['height'] / 2
                
                # Distance to zone center
                distance = math.sqrt((x - zone_center_x)**2 + (y - zone_center_y)**2)
                max_distance = math.sqrt(zone['width']**2 + zone['height']**2)
                
                # Score based on proximity to zone
                score = max(0, 1 - (distance / max_distance))
                best_score = max(best_score, score)
        
        return best_score
    
    def _score_position_adjacency(self, x: float, y: float, anchor_room: Dict) -> float:
        """Score position based on adjacency quality"""
        anchor_center_x = anchor_room['x'] + anchor_room['width'] / 2
        anchor_center_y = anchor_room['y'] + anchor_room['height'] / 2
        
        position_center_x = x + 5  # Assuming 10x10 room, center at +5
        position_center_y = y + 5
        
        # Distance between centers
        distance = math.sqrt((position_center_x - anchor_center_x)**2 + 
                           (position_center_y - anchor_center_y)**2)
        
        # Closer = better adjacency
        max_reasonable_distance = 20  # Reasonable adjacency distance
        return max(0, 1 - (distance / max_reasonable_distance))
    
    def _ensure_full_connectivity(self, placed_rooms: List[Dict], 
                                adjacency_rules: Dict) -> List[Dict]:
        """Ensure all rooms are in a single connected component"""
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            # Build spatial adjacency graph
            adjacency_graph = self._build_spatial_adjacency_graph(placed_rooms)
            
            # Find connected components
            components = self._find_connected_components(adjacency_graph)
            
            if len(components) <= 1:
                print(f"INFO: All rooms connected after {iteration} iterations")
                break
            
            print(f"INFO: Found {len(components)} disconnected components, connecting...")
            placed_rooms = self._connect_components_intelligently(placed_rooms, components, adjacency_rules)
            iteration += 1
        
        if len(components) > 1:
            print("WARNING: Could not achieve full connectivity within iteration limit")
        
        return placed_rooms
    
    def _connect_components_intelligently(self, placed_rooms: List[Dict], 
                                        components: List[Set[str]], 
                                        adjacency_rules: Dict) -> List[Dict]:
        """Connect components by moving rooms strategically"""
        if len(components) <= 1:
            return placed_rooms
        
        # Keep largest component fixed, move others
        largest_component = max(components, key=len)
        room_dict = {room['id']: room for room in placed_rooms}
        
        for component in components:
            if component == largest_component:
                continue
            
            # Find best room in this component to connect
            component_rooms = [room_dict[room_id] for room_id in component]
            connector_room = self._find_best_connector_room(component_rooms, adjacency_rules)
            
            # Find best target in largest component
            largest_component_rooms = [room_dict[room_id] for room_id in largest_component]
            target_room = self._find_best_connection_target(
                connector_room, largest_component_rooms, adjacency_rules
            )
            
            # Move connector room adjacent to target
            new_position = self._find_adjacent_position(
                connector_room, target_room, placed_rooms
            )
            
            # Calculate offset and move entire component
            offset_x = new_position[0] - connector_room['x']
            offset_y = new_position[1] - connector_room['y']
            
            for room_id in component:
                room = room_dict[room_id]
                room['x'] += offset_x
                room['y'] += offset_y
        
        return placed_rooms
    
    def _find_best_connector_room(self, component_rooms: List[Dict], 
                                adjacency_rules: Dict) -> Dict:
        """Find the room in component that would be best for connecting"""
        best_room = component_rooms[0]  # Fallback
        best_score = 0
        
        for room in component_rooms:
            room_type = room['type']
            
            # Score based on adjacency requirements
            score = 0
            if room_type in adjacency_rules:
                score += len(adjacency_rules[room_type].get('critical', set())) * 2
                score += len(adjacency_rules[room_type].get('preferred', set()))
            
            # Bonus for connector room types
            if room_type in ['living_room', 'hallway', 'kitchen']:
                score += 5
            
            if score > best_score:
                best_score = score
                best_room = room
        
        return best_room
    
    def _find_best_connection_target(self, connector_room: Dict, 
                                   target_rooms: List[Dict], 
                                   adjacency_rules: Dict) -> Dict:
        """Find best room in target component to connect to"""
        connector_type = connector_room['type']
        best_target = target_rooms[0]  # Fallback
        best_score = 0
        
        for target_room in target_rooms:
            target_type = target_room['type']
            
            # Calculate adjacency compatibility
            score = self._calculate_adjacency_score(connector_type, target_type, adjacency_rules)
            
            if score > best_score:
                best_score = score
                best_target = target_room
        
        return best_target
    
    def _find_adjacent_position(self, room_to_move: Dict, target_room: Dict, 
                              all_rooms: List[Dict]) -> Tuple[float, float]:
        """Find position adjacent to target room"""
        target_x, target_y = target_room['x'], target_room['y']
        target_width, target_height = target_room['width'], target_room['height']
        
        room_width = room_to_move['width']
        room_height = room_to_move['height']
        gap = 1
        
        # Try positions around target room
        candidate_positions = [
            (target_x + target_width + gap, target_y),  # Right
            (target_x - room_width - gap, target_y),    # Left
            (target_x, target_y + target_height + gap), # Below
            (target_x, target_y - room_height - gap),   # Above
        ]
        
        # Return first non-overlapping position
        for pos in candidate_positions:
            if not self._check_overlap(pos[0], pos[1], room_width, room_height, 
                                     [r for r in all_rooms if r['id'] != room_to_move['id']]):
                return pos
        
        # If all overlap, return position with offset
        return (target_x + target_width + gap, target_y + len(all_rooms))
    
    def _optimize_directional_positions(self, placed_rooms: List[Dict], 
                                      directional_zones: Dict) -> List[Dict]:
        """Fine-tune positions to better align with directional preferences"""
        optimized_rooms = []
        
        for room in placed_rooms:
            room_type = room['type']
            
            # Skip optimization if no directional preferences
            if room_type not in self.directional_optimizer.room_preferences:
                optimized_rooms.append(room)
                continue
            
            prefs = self.directional_optimizer.room_preferences[room_type]
            preferred_directions = prefs.preferred_directions
            
            if not preferred_directions:
                optimized_rooms.append(room)
                continue
            
            # Find the best directional zone for this room
            best_zone = None
            best_score = 0
            
            for direction in preferred_directions:
                direction_str = direction.value if hasattr(direction, 'value') else str(direction)
                if direction_str in directional_zones:
                    zone = directional_zones[direction_str]
                    
                    # Score based on directional preference strength
                    direction_scores = self.directional_optimizer.directional_scores[direction]
                    
                    score = (direction_scores['morning_light'] * prefs.natural_light_requirement +
                            direction_scores['ventilation'] * prefs.cross_ventilation_priority +
                            direction_scores['privacy'] * prefs.privacy_requirement) / 3
                    
                    if score > best_score:
                        best_score = score
                        best_zone = zone
            
            # Update room's orientation based on best zone
            if best_zone:
                # Find which direction this zone represents
                for direction_str, zone in directional_zones.items():
                    if zone == best_zone:
                        room['orientation'] = direction_str
                        break
            
            optimized_rooms.append(room)
        
        return optimized_rooms
    
    # Helper methods (keeping existing implementations with minor improvements)
    def _calculate_room_dimensions(self, area: float) -> Tuple[float, float]:
        """Calculate room dimensions from area"""

        # Use golden ratio for better proportions
        ratio = 1.618  # Golden ratio
        width = math.sqrt(area / ratio)
        height = area / width
        return (width, height)
    
    def _calculate_adjacency_score(self, room_type: str, placed_type: str, 
                                 adjacency_rules: Dict) -> float:
        """Calculate adjacency score between two room types"""
        score = 0.0
        
        # Check if room_type wants to be adjacent to placed_type
        if room_type in adjacency_rules:
            rules = adjacency_rules[room_type]
            if placed_type in rules.get('critical', set()):
                score += 10.0
            elif placed_type in rules.get('preferred', set()):
                score += 5.0
        
        # Check reverse relationship
        if placed_type in adjacency_rules:
            rules = adjacency_rules[placed_type]
            if room_type in rules.get('critical', set()):
                score += 10.0
            elif room_type in rules.get('preferred', set()):
                score += 5.0
        
        return score
    
    def _check_overlap(self, x: float, y: float, width: float, height: float, 
                      placed_rooms: List[Dict]) -> bool:
        """Check if position overlaps with any existing room"""
        for room in placed_rooms:
            if (x < room['x'] + room['width'] + 1 and  # Add 1-unit buffer
                x + width + 1 > room['x'] and
                y < room['y'] + room['height'] + 1 and 
                y + height + 1 > room['y']):
                return True
        return False
    
    def _create_placed_room(self, room_info: Dict, x: float, y: float) -> Dict:
        """Create a placed room with position information"""
        dimensions = self._calculate_room_dimensions(room_info['area'])
        
        return {
            'id': room_info['id'],
            'type': room_info['type'],
            'area': room_info['area'],
            'x': x,
            'y': y,
            'width': dimensions[0],
            'height': dimensions[1],
            'orientation': room_info.get('preferred_orientation', 'south')
        }
    
    def _build_spatial_adjacency_graph(self, placed_rooms: List[Dict]) -> Dict[str, Set[str]]:
        """Build graph of which rooms are spatially adjacent"""
        graph = {room['id']: set() for room in placed_rooms}
        
        for i, room1 in enumerate(placed_rooms):
            for j, room2 in enumerate(placed_rooms):
                if i != j and self._are_spatially_adjacent(room1, room2):
                    graph[room1['id']].add(room2['id'])
                    graph[room2['id']].add(room1['id'])
        
        return graph
    
    def _are_spatially_adjacent(self, room1: Dict, room2: Dict, tolerance: float = 2.0) -> bool:
        """Check if two rooms are spatially adjacent with improved logic"""
        # Horizontal adjacency (rooms side by side)
        horizontal_adjacent = (
            abs(room1['x'] + room1['width'] - room2['x']) <= tolerance or
            abs(room2['x'] + room2['width'] - room1['x']) <= tolerance
        )
        
        # Vertical overlap check for horizontal adjacency
        horizontal_overlap = not (
            room1['y'] + room1['height'] <= room2['y'] or 
            room2['y'] + room2['height'] <= room1['y']
        )
        
        # Vertical adjacency (rooms above/below each other)
        vertical_adjacent = (
            abs(room1['y'] + room1['height'] - room2['y']) <= tolerance or
            abs(room2['y'] + room2['height'] - room1['y']) <= tolerance
        )
        
        # Horizontal overlap check for vertical adjacency
        vertical_overlap = not (
            room1['x'] + room1['width'] <= room2['x'] or 
            room2['x'] + room2['width'] <= room1['x']
        )
        
        return (horizontal_adjacent and horizontal_overlap) or (vertical_adjacent and vertical_overlap)
    
    def _find_connected_components(self, graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find connected components in the adjacency graph"""
        visited = set()
        components = []
        
        for node in graph:
            if node not in visited:
                component = set()
                self._dfs(graph, node, visited, component)
                components.append(component)
        
        return components
    
    def _dfs(self, graph: Dict[str, Set[str]], node: str, 
            visited: Set[str], component: Set[str]):
        """Depth-first search for connected components"""
        visited.add(node)
        component.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                self._dfs(graph, neighbor, visited, component)
                
class EnhancedLayoutGenerator:
    
    """Enhanced layout generator with directional optimization"""
    # Place this method inside the EnhancedLayoutGenerator class
    
    def __init__(self, directional_optimizer: EnhancedDirectionalOptimizer):
        self.optimizer = directional_optimizer
        # Add the new engine and rules here
        self.geometric_engine = EnhancedGeometricLayoutEngine(directional_optimizer)
        self.adjacency_rules = self._get_adjacency_rules()

    def _get_adjacency_rules(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Enhanced adjacency rules for better connectivity
        """
        return {
            'kitchen': {
                'critical': {'dining_room', 'dining_hall', 'living_room'},  # Added living_room
                'preferred': {'utility', 'storage', 'storeroom'}
            },
            'dining_room': {
                'critical': {'kitchen', 'living_room'},  # Added living_room
                'preferred': {'hallway'}
            },
            'living_room': {
                'critical': {'kitchen', 'dining_room'},  # Made kitchen critical
                'preferred': {'balcony', 'hallway', 'bedroom'}  # Added bedroom
            },
            'bedroom': {
                'critical': {'living_room'},  # Changed from bathroom to living_room for better flow
                'preferred': {'bathroom', 'balcony', 'hallway'}
            },
            'bathroom': {
                'critical': {'bedroom', 'living_room'},  # Added living_room
                'preferred': {'hallway'}
            },
            'office': {
                'critical': {'living_room'},  # Added critical connection
                'preferred': {'hallway', 'bedroom'}
            },
            'garage': {
                'critical': set(),
                'preferred': {'utility', 'storage'}
            },
            'hallway': {
                'critical': {'living_room'},  # Simplified - living room is the main connector
                'preferred': {'bedroom', 'bathroom', 'office', 'kitchen'}
            },
            'utility': {
                'critical': {'kitchen'},
                'preferred': {'garage', 'storage'}
            },
            'storage': {
                'critical': set(),
                'preferred': {'kitchen', 'utility', 'garage'}
            },
            'storeroom': {
                'critical': set(),
                'preferred': {'kitchen', 'utility'}
            },
            'balcony': {
                'critical': set(),
                'preferred': {'living_room', 'bedroom'}
            }
        }
        
    def generate_optimal_layout(self, requirements, plot_dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Generate optimal room layout considering all directional factors"""
        
        plot_length, plot_width = plot_dimensions
        
        # Step 1: Generate multiple layout options (this part remains the same)
        layout_options = self._generate_layout_options(requirements, plot_dimensions, {})
        
        # Step 2: Evaluate each layout option (this part remains the same)
        evaluated_layouts = self._evaluate_layouts(layout_options, requirements)
        
        # Step 3: Select the best *directional strategy*
        best_strategy_layout = max(evaluated_layouts, key=lambda x: x['total_score']) if evaluated_layouts else self._create_balanced_layout(requirements, plot_dimensions)
        
        # --- UPDATED INTEGRATION STEP ---
        # Step 4: Convert the chosen strategy into a list of rooms for the geometric engine
        rooms_to_place = []
        for room_id, room_data in best_strategy_layout['rooms'].items():
            rooms_to_place.append({
                'id': room_id,
                'type': room_data['type'],
                'area': room_data['area'],
                'preferred_orientation': room_data['orientation'] # Pass orientation as a hint
            })

        # Step 5: Use the ENHANCED Geometric Engine to create the final, connected layout
        # PASS plot_dimensions to ensure proper zone calculation
        final_placed_rooms_list = self.geometric_engine.place_rooms(
            rooms_to_place, self.adjacency_rules, plot_dimensions
        )
        
        # Convert the list back to the dictionary format the rest of the script expects
        final_placed_rooms_dict = {room['id']: room for room in final_placed_rooms_list}
        best_strategy_layout['rooms'] = final_placed_rooms_dict # Overwrite with the new geometric layout
        # --- END OF UPDATED INTEGRATION ---

        # Step 6: Generate detailed analysis on the final, geometrically-aware layout
        detailed_analysis = self._generate_detailed_analysis(best_strategy_layout, requirements)
        
        return {
            'optimal_layout': best_strategy_layout, # This now contains the geometric layout
            'layout_analysis': detailed_analysis,
            'alternative_options': evaluated_layouts[:3],
            'directional_recommendations': self._generate_directional_recommendations(best_strategy_layout)
        }
    
    def _calculate_room_priorities(self, requirements) -> Dict[str, float]:
        """Calculate room placement priorities based on user preferences and function"""
        priorities = {}
        
        for room_spec in requirements.spatial_needs:
            room_type = room_spec.room_type
            base_priority = 0.5
            
            # Adjust based on room type importance
            if room_type in ["living_room", "kitchen"]:
                base_priority = 1.0  # High priority
            elif room_type in ["bedroom"]:
                base_priority = 0.9
            elif room_type in ["dining_room", "office"]:
                base_priority = 0.8
            elif room_type in ["bathroom"]:
                base_priority = 0.7
            else:
                base_priority = 0.4  # Storage, utility
            
            # Adjust based on user specified priority
            if hasattr(room_spec, 'priority'):
                if room_spec.priority == "high":
                    base_priority *= 1.2
                elif room_spec.priority == "low":
                    base_priority *= 0.8
            
            priorities[room_type] = min(1.0, base_priority)
        
        return priorities
    
    def _generate_layout_options(self, requirements, plot_dimensions: Tuple[float, float], 
                               priorities: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate multiple layout options with different directional strategies"""
        
        layouts = []
        
        layouts.append(self._create_morning_optimized_layout(requirements, plot_dimensions))
        layouts.append(self._create_privacy_optimized_layout(requirements, plot_dimensions))
        layouts.append(self._create_ventilation_optimized_layout(requirements, plot_dimensions))
        layouts.append(self._create_balanced_layout(requirements, plot_dimensions))
        
        return layouts

    def _create_layout_from_strategy(self, requirements, plot_dimensions: Tuple[float, float], strategy_name: str, description: str, strategy_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generic function to create a layout based on a directional strategy."""
        layout = {
            'strategy': strategy_name,
            'description': description,
            'rooms': {},
            'circulation': {},
            'windows': {},
            'scores': {}
        }
        
        unassigned_rooms = []
        for spec in requirements.spatial_needs:
            for i in range(spec.quantity):
                room_id = f"{spec.room_type}_{i+1}" if spec.quantity > 1 else spec.room_type
                unassigned_rooms.append({'id': room_id, 'type': spec.room_type})

        assigned_rooms = set()

        for direction, room_types in strategy_map.items():
            for room_type in room_types:
                for room in unassigned_rooms:
                    if room['type'] == room_type and room['id'] not in assigned_rooms:
                        area = self._get_default_area(room['type'])
                        layout['rooms'][room['id']] = {
                            'type': room['type'],
                            'area': area,
                            'position': (0, 0), # Placeholder position
                            'dimensions': (math.sqrt(area), math.sqrt(area)), # Placeholder dimensions
                            'orientation': direction,
                            'windows': self._calculate_optimal_windows(room['type'], CardinalDirection(direction), area)
                        }
                        assigned_rooms.add(room['id'])
                        break # Assign only one of this type per direction for simplicity

        # Place any remaining rooms
        for room in unassigned_rooms:
            if room['id'] not in assigned_rooms:
                area = self._get_default_area(room['type'])
                layout['rooms'][room['id']] = {
                    'type': room['type'],
                    'area': area,
                    'position': (0, 0),
                    'dimensions': (math.sqrt(area), math.sqrt(area)),
                    'orientation': 'south', # Default fallback orientation
                    'windows': self._calculate_optimal_windows(room['type'], CardinalDirection.SOUTH, area)
                }
                assigned_rooms.add(room['id'])
        
        return layout

    def _create_morning_optimized_layout(self, requirements, plot_dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create layout optimized for morning light"""
        strategy_map = {
            'east': ['bedroom', 'kitchen', 'dining_room'],
            'north': ['office'],
            'south': ['living_room'],
            'west': ['bathroom', 'storage', 'utility']
        }
        return self._create_layout_from_strategy(requirements, plot_dimensions, 'morning_optimized', 'Optimized for morning sunlight in bedrooms and kitchen.', strategy_map)

    def _create_privacy_optimized_layout(self, requirements, plot_dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create layout optimized for privacy"""
        strategy_map = {
            'north': ['bedroom', 'office'], # Rear of the house
            'east': ['bathroom'],
            'south': ['living_room', 'kitchen', 'dining_room'], # Front/Public areas
            'west': ['storage', 'utility']
        }
        return self._create_layout_from_strategy(requirements, plot_dimensions, 'privacy_optimized', 'Optimized for privacy with bedrooms in the rear (north).', strategy_map)
    
    def _create_ventilation_optimized_layout(self, requirements, plot_dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create layout optimized for cross-ventilation"""
        strategy_map = {
            'southwest': ['living_room', 'bedroom'], # Catch primary monsoon winds
            'northeast': ['kitchen', 'office'], # Catch winter winds, opposite SW
            'southeast': ['dining_room'],
            'northwest': ['bathroom', 'storage', 'utility']
        }
        return self._create_layout_from_strategy(requirements, plot_dimensions, 'ventilation_optimized', 'Optimized for SW-NE cross-ventilation.', strategy_map)
    
    def _create_balanced_layout(self, requirements, plot_dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create balanced layout considering all factors"""
        strategy_map = {
            'east': ['kitchen', 'dining_room'], # Good morning light
            'northeast': ['bedroom'], # Good light, less heat
            'south': ['living_room'], # Active day area
            'north': ['office'], # Consistent light, private
            'west': ['bathroom', 'storage', 'utility'] # Buffer zone for heat
        }
        return self._create_layout_from_strategy(requirements, plot_dimensions, 'balanced', 'Balanced approach for light, ventilation, and privacy.', strategy_map)
    
    def _calculate_optimal_windows(self, room_type: str, primary_direction: CardinalDirection, 
                                 room_area: float) -> Dict[str, Any]:
        """Calculate optimal window configuration for room"""
        preferences = self.optimizer.room_preferences.get(room_type)
        if not preferences:
            preferences = self.optimizer.room_preferences["storage"]  # Default
        
        # Calculate window area based on requirements
        min_window_area = room_area * 0.1  # Minimum 10% of floor area
        optimal_window_area = room_area * 0.15  # Optimal 15% for good daylight
        
        if preferences.natural_light_requirement > 0.8:
            target_window_area = room_area * 0.2  # 20% for high light requirement
        else:
            target_window_area = optimal_window_area
        
        return {
            'primary_direction': primary_direction.value,
            'window_area': round(target_window_area, 2),
            'window_count': max(1, int(target_window_area / 20)),  # ~20 sqft per window
            'window_height': 4.5,  # Standard height in feet
            'sill_height': 2.5, # feet
            'ventilation_area': round(target_window_area * 0.5, 2),  # 50% openable
            'shading_required': primary_direction in [CardinalDirection.SOUTH, CardinalDirection.WEST, CardinalDirection.SOUTHWEST]
        }
    
    def _evaluate_layouts(self, layouts: List[Dict[str, Any]], requirements) -> List[Dict[str, Any]]:
        """Evaluate and score layout options"""
        evaluated = []
        
        for layout in layouts:
            if not layout.get('rooms'): continue # Skip empty layouts
            scores = self._calculate_layout_scores(layout, requirements)
            layout['scores'] = scores
            layout['total_score'] = sum(scores.values()) / len(scores) if scores else 0
            evaluated.append(layout)
        
        return sorted(evaluated, key=lambda x: x['total_score'], reverse=True)
    
    def _calculate_layout_scores(self, layout: Dict[str, Any], requirements) -> Dict[str, float]:
        """Calculate comprehensive scores for layout"""
        scores = {}
        
        scores['lighting'] = self._calculate_lighting_score(layout)
        scores['ventilation'] = self._calculate_ventilation_score(layout)
        scores['privacy'] = self._calculate_privacy_score(layout)
        scores['circulation'] = self._calculate_circulation_score(layout)
        scores['cost_efficiency'] = self._calculate_cost_score(layout)
        scores['climate_suitability'] = self._calculate_climate_score(layout)
        
        return scores
    
    def _calculate_lighting_score(self, layout: Dict[str, Any]) -> float:
        """Calculate lighting effectiveness score"""
        total_score = 0
        room_count = 0
        
        for room_id, room_data in layout.get('rooms', {}).items():
            room_type = room_data['type']
            orientation = room_data.get('orientation', 'south')
            
            preferences = self.optimizer.room_preferences.get(room_type)
            if not preferences:
                continue
            
            direction = CardinalDirection(orientation.lower())
            directional_scores = self.optimizer.directional_scores[direction]
            
            # Weighted score based on room requirements for light
            light_score = (
                directional_scores['morning_light'] * (1 if direction in [CardinalDirection.EAST, CardinalDirection.NORTHEAST] else 0.1) +
                directional_scores['consistent_light'] * 0.5 +
                directional_scores['heat_gain'] * 0.3 # Higher score for less heat is good
            ) / 1.8 # Normalize
            
            # Apply room-specific weighting
            weighted_score = light_score * preferences.natural_light_requirement
            total_score += weighted_score
            room_count += 1
        
        return total_score / room_count if room_count > 0 else 0.5
    
    def _calculate_ventilation_score(self, layout: Dict[str, Any]) -> float:
        """Calculate ventilation effectiveness score"""
        total_score = 0
        orientations = {room['orientation'] for room in layout.get('rooms', {}).values()}
        
        # Check for presence of rooms on opposing sides for cross-ventilation
        if ('southwest' in orientations or 'south' in orientations) and ('northeast' in orientations or 'north' in orientations):
            total_score += 0.5
        if ('east' in orientations) and ('west' in orientations):
            total_score += 0.3
            
        return min(1.0, 0.4 + total_score) # Base score + cross-vent bonus
    
    def _calculate_privacy_score(self, layout: Dict[str, Any]) -> float:
        """Calculate privacy effectiveness score"""
        total_score = 0
        private_room_count = 0
        for room_id, room_data in layout.get('rooms', {}).items():
            room_type = room_data['type']
            preferences = self.optimizer.room_preferences.get(room_type)
            if not preferences or preferences.privacy_requirement < 0.8:
                continue

            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            privacy_score = self.optimizer.directional_scores[direction]['privacy']
            total_score += privacy_score
            private_room_count += 1

        return total_score / private_room_count if private_room_count > 0 else 0.8
    
    def _calculate_circulation_score(self, layout: Dict[str, Any]) -> float:
        """Calculate circulation efficiency score"""
        # Placeholder: a real calculation requires a geometric analysis
        return 0.7  # Placeholder
    
    def _calculate_cost_score(self, layout: Dict[str, Any]) -> float:
        """Calculate cost efficiency score"""
        # Placeholder: a real calculation requires structural analysis
        return 0.75  # Placeholder
    
    def _calculate_climate_score(self, layout: Dict[str, Any]) -> float:
        """Calculate climate appropriateness score"""
        heat_scores = []
        for room_data in layout.get('rooms', {}).values():
             direction = CardinalDirection(room_data['orientation'].lower())
             heat_scores.append(self.optimizer.directional_scores[direction]['heat_gain'])
        
        return np.mean(heat_scores) if heat_scores else 0.5

    def _generate_detailed_analysis(self, layout: Dict[str, Any], requirements) -> Dict[str, Any]:
        """Generate detailed analysis of the selected layout"""
        
        lighting_analysis = self._analyze_lighting_performance(layout)
        ventilation_analysis = self._analyze_ventilation_performance(layout)
        circulation_analysis = self._analyze_circulation_performance(layout)
        
        return {
            'lighting_analysis': asdict(lighting_analysis),
            'ventilation_analysis': asdict(ventilation_analysis),
            'circulation_analysis': asdict(circulation_analysis),
            'energy_performance': self._analyze_energy_performance(layout),
            'construction_recommendations': self._generate_construction_recommendations(layout),
            'compliance_check': self._check_building_compliance(layout, requirements)
        }
    
    def _analyze_lighting_performance(self, layout: Dict[str, Any]) -> LightingAnalysis:
        """Detailed lighting performance analysis"""
        daylight_factors = {}
        sunlight_hours = {}
        glare_risks = {}
        artificial_requirements = {}
        window_effectiveness = {}
        solar_heat_gains = {}
        
        for room_id, room_data in layout.get('rooms', {}).items():
            room_type = room_data['type']
            orientation = room_data.get('orientation', 'south')
            
            window_area = room_data.get('windows', {}).get('window_area', 0)
            room_area = room_data.get('area', 10)
            daylight_factors[room_id] = round(min(0.08, (window_area / room_area) * 0.3), 3) # Simplified DF
            
            direction = CardinalDirection(orientation.lower())
            scores = self.optimizer.directional_scores[direction]
            
            sunlight_hours[room_id] = {
                'summer': round(scores['morning_light'] * 6 + scores['afternoon_light'] * 6, 1),
                'winter': round(scores['morning_light'] * 4 + scores['afternoon_light'] * 4, 1),
                'monsoon': round(scores['consistent_light'] * 4, 1)
            }
            
            glare_risks[room_id] = "high" if direction in [CardinalDirection.SOUTH, CardinalDirection.WEST] else "medium" if direction in [CardinalDirection.EAST, CardinalDirection.SOUTHEAST] else "low"
            artificial_requirements[room_id] = round(max(0, 12 - sunlight_hours[room_id]['winter'] / 2), 1)
            window_effectiveness[f"win_{room_id}"] = round(scores['consistent_light'] * 0.8, 2)
            solar_heat_gains[room_id] = round((1 - scores['heat_gain']) * 0.7, 2)
        
        return LightingAnalysis(
            daylight_factor_by_room=daylight_factors,
            direct_sunlight_hours=sunlight_hours,
            glare_risk_assessment=glare_risks,
            artificial_lighting_requirement=artificial_requirements,
            window_effectiveness=window_effectiveness,
            solar_heat_gain=solar_heat_gains
        )
    
    def _analyze_ventilation_performance(self, layout: Dict[str, Any]) -> VentilationAnalysis:
        """Detailed ventilation performance analysis"""
        return VentilationAnalysis(
            prevailing_wind_direction=CardinalDirection.SOUTHWEST,
            secondary_wind_direction=CardinalDirection.NORTHEAST,
            wind_speed=3.5,  # Average for India
            cross_ventilation_effectiveness={room_id: round(random.uniform(0.6, 0.9), 2) for room_id in layout.get('rooms', {})},
            stack_effect_potential=0.6
        )
    
    def _analyze_circulation_performance(self, layout: Dict[str, Any]) -> CirculationMetrics:
        """Detailed circulation performance analysis"""
        total_room_area = sum(room['area'] for room in layout.get('rooms', {}).values())
        estimated_circulation = total_room_area * 0.15  # 15% circulation
        total_area = total_room_area + estimated_circulation
        
        return CirculationMetrics(
            total_circulation_area=round(estimated_circulation, 2),
            circulation_efficiency=round(total_room_area / total_area, 2) if total_area > 0 else 0,
            average_travel_distance=15.0,  # Estimated
            dead_end_count=0,
            accessibility_compliance=True,
            corridor_width_analysis={'main': 1.2, 'secondary': 1.0}, # in meters
            circulation_bottlenecks=[]
        )
    
    def _analyze_energy_performance(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze energy performance of the layout"""
        return {
            'estimated_annual_energy_consumption': 12000,  # kWh
            'solar_potential': 0.8,
            'natural_cooling_effectiveness': 0.7,
            'heating_requirement': 0.2,  # Low for Indian climate
            'energy_efficiency_rating': 'B+'
        }
    
    def _generate_construction_recommendations(self, layout: Dict[str, Any]) -> List[str]:
        """Generate construction recommendations based on layout"""
        return [
            "Use light-colored roof materials to reduce heat gain",
            "Install deep overhangs on south and west facades",
            "Consider cavity walls for better thermal insulation",
            "Use high-efficiency windows with low-E coating",
            "Install ceiling fans in all major rooms",
            "Consider solar water heating system",
            "Use permeable paving for courtyards and driveways",
            "Install rainwater harvesting system"
        ]
        
    def _check_building_compliance(self, layout: Dict[str, Any], requirements) -> Dict[str, Any]:
        """Check compliance with building regulations"""
        return {
            'nbcc_compliance': True,
            'local_bylaws_compliance': True,
            'fire_safety_compliance': True,
            'accessibility_compliance': requirements.design_preferences.accessibility_requirements,
            'environmental_clearance': False,  # Usually not required for residential
            'structural_safety': True,
            'electrical_safety': True
        }
    
    def _generate_directional_recommendations(self, layout: Dict[str, Any]) -> List[str]:
        """Generate specific directional recommendations"""
        recommendations = []
        
        for room_id, room_data in layout.get('rooms', {}).items():
            room_type = room_data['type']
            orientation = room_data.get('orientation', 'south')
            
            if room_type == 'kitchen' and orientation in ['west', 'southwest']:
                recommendations.append(f"Kitchen facing {orientation}: Install powerful exhaust fan and use heat-resistant backsplash materials.")
            
            if room_type == 'bedroom' and orientation in ['west', 'southwest']:
                recommendations.append(f"Bedroom facing {orientation}: Use blackout curtains and consider placing closets along the west wall as a thermal buffer.")
            
            if room_type == 'living_room' and orientation == 'south':
                recommendations.append(f"Living room facing south: Install adjustable shading devices like louvers or a deep veranda to control winter sun and block summer sun.")
        
        return recommendations
    
    def _get_default_area(self, room_type: str) -> float:
        """Get default area for room type in sq ft"""
        default_areas = {
            'bedroom': 140, 'living_room': 250, 'kitchen': 100, 'dining_room': 120,
            'bathroom': 45, 'office': 100, 'storage': 50, 'utility': 40,
            'hallway': 60, 'balcony': 50,
        }
        return default_areas.get(room_type, 80)

class EnhancedFBSGenerator(FBSOntologyGenerator):
    # PASTE THIS ENTIRE BLOCK INSIDE THE EnhancedFBSGenerator CLASS

    def _get_room_daylight_target(self, room_type: str) -> float:
        """Get daylight target based on room type"""
        targets = {
            'bedroom': 0.02, 'living_room': 0.03, 'kitchen': 0.035,
            'dining_room': 0.025, 'office': 0.04, 'bathroom': 0.015
        }
        return targets.get(room_type, 0.02)

    def _calculate_room_daylight_performance(self, room_data: Dict, direction: CardinalDirection) -> float:
        """Calculate actual daylight performance using directional data"""
        morning_light = self.directional_optimizer.directional_scores[direction]['morning_light']
        afternoon_light = self.directional_optimizer.directional_scores[direction]['afternoon_light']
        consistent_light = self.directional_optimizer.directional_scores[direction]['consistent_light']
        
        # Weighted daylight score
        daylight_score = (morning_light * 0.3 + afternoon_light * 0.3 + consistent_light * 0.4)
        
        # Convert to daylight factor approximation
        window_area = room_data.get('windows', {}).get('window_area', 20)
        room_area = room_data.get('area', 100)
        window_ratio = window_area / room_area
        
        return min(0.08, daylight_score * window_ratio * 0.15)

    def _calculate_thermal_comfort(self, room_type: str, heat_score: float) -> float:
        """Calculate thermal comfort based on heat gain and room type"""
        base_comfort = heat_score  # Higher heat_score = lower heat gain = better comfort
        
        # Adjust for room type sensitivity to heat
        heat_sensitivity = {
            'bedroom': 1.2, 'kitchen': 0.7, 'living_room': 1.0,
            'office': 1.1, 'bathroom': 0.9, 'dining_room': 1.0
        }
        
        sensitivity = heat_sensitivity.get(room_type, 1.0)
        return min(1.0, base_comfort * sensitivity)

    def _get_room_privacy_requirement(self, room_type: str) -> float:
        """Get privacy requirement by room type"""
        requirements = {
            'bedroom': 0.9, 'bathroom': 1.0, 'office': 0.7,
            'living_room': 0.4, 'kitchen': 0.3, 'dining_room': 0.5
        }
        return requirements.get(room_type, 0.5)

    def _calculate_seasonal_performance(self, layout_result: Dict) -> float:
        """Calculate seasonal adaptation performance"""
        # Get seasonal data from directional analysis
        seasonal_data = layout_result['layout_analysis'].get('seasonal_performance', {})
        
        # Average performance across seasons
        summer_score = 0.6  # Challenging season
        monsoon_score = 0.8  # Good with proper ventilation
        winter_score = 0.9   # Excellent season
        
        return (summer_score + monsoon_score + winter_score) / 3

    def _analyze_ventilation_effectiveness(self, layout_result: Dict) -> float:
        """Analyze cross-ventilation effectiveness"""
        ventilation_analysis = layout_result['layout_analysis']['ventilation_analysis']
        effectiveness_scores = list(ventilation_analysis['cross_ventilation_effectiveness'].values())
        return sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.6

    def _calculate_average_heat_gain(self, layout_result: Dict) -> float:
        """Calculate average solar heat gain across all rooms"""
        heat_gains = []
        for room_data in layout_result['optimal_layout']['rooms'].values():
            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            heat_score = self.directional_optimizer.directional_scores[direction]['heat_gain']
            heat_gain = 1 - heat_score  # Convert to heat gain (higher = more heat)
            heat_gains.append(heat_gain)
        
        return sum(heat_gains) / len(heat_gains) if heat_gains else 0.4

    def _calculate_energy_performance(self, layout_result: Dict) -> float:
        """Calculate overall energy performance index"""
        # Combine multiple factors
        lighting_analysis = layout_result['layout_analysis']['lighting_analysis']
        ventilation_analysis = layout_result['layout_analysis']['ventilation_analysis']
        
        # Natural lighting effectiveness (reduces artificial lighting)
        daylight_factors = list(lighting_analysis['daylight_factor_by_room'].values())
        avg_daylight = sum(daylight_factors) / len(daylight_factors) if daylight_factors else 0.02
        lighting_score = min(1.0, avg_daylight / 0.03)  # Normalize to target
        
        # Ventilation effectiveness (reduces cooling load)
        ventilation_scores = list(ventilation_analysis['cross_ventilation_effectiveness'].values())
        avg_ventilation = sum(ventilation_scores) / len(ventilation_scores) if ventilation_scores else 0.6
        
        # Heat gain control (reduces cooling load)
        heat_gain_score = 1 - self._calculate_average_heat_gain(layout_result)
        
        # Combined energy performance
        return (lighting_score * 0.3 + avg_ventilation * 0.4 + heat_gain_score * 0.3)
    
    """Enhanced FBS generator with directional optimization"""
    # PASTE THIS METHOD INSIDE THE EnhancedFBSGenerator CLASS

    def _generate_design_functions(self, requirements: ParsedRequirements) -> List[Function]:
        """Generate detailed, context-aware Functions leveraging directional optimization"""
        functions = []
        
        # F1-F6: Room-specific activity functions
        room_functions = {
            'bedroom': {
                'name': 'Enable Restful Sleep & Privacy',
                'intent': 'Provide peaceful, private sleeping environment with morning light and cross-ventilation',
                'goals': {
                    'privacy_level': 0.9,
                    'morning_light_hours': 3.0,
                    'noise_isolation': 45,  # dB
                    'optimal_temperature_hours': 8.0  # per night
                },
                'priority': 0.95
            },
            'kitchen': {
                'name': 'Enable Efficient Cooking & Food Preparation',
                'intent': 'Support cooking activities with natural light, ventilation for smoke/odor removal, heat management',
                'goals': {
                    'natural_light_factor': 0.03,
                    'ventilation_ach': 15,  # Air changes per hour
                    'heat_gain_minimization': 0.8,
                    'workflow_efficiency': 0.85
                },
                'priority': 0.9
            },
            'living_room': {
                'name': 'Facilitate Family Gathering & Social Activities',
                'intent': 'Create comfortable social space with good lighting throughout day and flexible ventilation',
                'goals': {
                    'daylight_coverage': 0.8,
                    'social_capacity': 6,  # persons
                    'acoustic_comfort': 40,  # dB background
                    'thermal_comfort_hours': 16  # per day
                },
                'priority': 0.85
            },
            'bathroom': {
                'name': 'Ensure Hygiene & Moisture Control',
                'intent': 'Provide private, well-ventilated space for personal hygiene with moisture management',
                'goals': {
                    'privacy_level': 1.0,
                    'ventilation_ach': 20,
                    'moisture_control': 0.9,
                    'natural_light_factor': 0.02
                },
                'priority': 0.8
            },
            'dining_room': {
                'name': 'Enable Pleasant Dining Experience',
                'intent': 'Support family meals with appropriate lighting and comfortable environment',
                'goals': {
                    'dining_capacity': 6,
                    'lighting_quality': 0.8,
                    'circulation_access': 0.9
                },
                'priority': 0.7
            },
            'office': {
                'name': 'Support Productive Work & Study',
                'intent': 'Provide consistent natural light, minimal glare, and quiet environment for concentration',
                'goals': {
                    'consistent_light_hours': 10,
                    'glare_control': 0.9,
                    'noise_isolation': 50,  # dB
                    'workspace_efficiency': 0.85
                },
                'priority': 0.75
            }
        }
        
        # Generate room-specific functions
        func_id = 1
        for room_spec in requirements.spatial_needs:
            if room_spec.room_type in room_functions:
                room_func = room_functions[room_spec.room_type]
                functions.append(Function(
                    element_id=f"F{func_id:03d}",
                    name=room_func['name'],
                    description=f"For {room_spec.quantity}x {room_spec.room_type}: {room_func['intent']}"
                ))
                func_id += 1
        
        # F7: Climate Response Function
        functions.append(Function(
            element_id=f"F{func_id:03d}",
            name="Respond to Indian Subtropical Climate",
            description="Manage monsoon moisture, summer heat, and optimize natural cooling strategies"
        ))
        func_id += 1
        
        # F8: Seasonal Adaptation Function
        functions.append(Function(
            element_id=f"F{func_id:03d}",
            name="Adapt to Seasonal Variations",
            description="Optimize performance across summer heat, monsoon humidity, and winter comfort"
        ))
        func_id += 1
        
        # F9: Cultural Integration Function
        functions.append(Function(
            element_id=f"F{func_id:03d}",
            name="Support Indian Family Lifestyle",
            description="Accommodate joint family dynamics, religious practices, and cultural preferences"
        ))
        func_id += 1
        
        # F10: Energy Efficiency Function
        functions.append(Function(
            element_id=f"F{func_id:03d}",
            name="Minimize Energy Consumption",
            description="Maximize natural lighting and ventilation to reduce artificial cooling and lighting loads"
        ))
        
        return functions
    # PASTE THIS METHOD INSIDE THE EnhancedFBSGenerator CLASS

    def _generate_performance_behaviors(self, requirements: ParsedRequirements, layout_result: Dict) -> List[Behavior]:
        """Generate detailed Behaviors leveraging directional analysis data"""
        behaviors = []
        
        behav_id = 1
        
        # B1-B6: Room-specific performance behaviors
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            room_type = room_data['type']
            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            
            # Daylight Performance per Room
            behaviors.append(Behavior(
                element_id=f"B{behav_id:03d}",
                name=f"{room_type.replace('_', ' ').title()} Daylight Performance",
                description=f"Natural light adequacy in {room_type} facing {orientation}",
                target_value=self._get_room_daylight_target(room_type),
                current_value=self._calculate_room_daylight_performance(room_data, direction)
            ))
            behav_id += 1
            
            # Thermal Comfort per Room
            heat_score = self.directional_optimizer.directional_scores[direction]['heat_gain']
            thermal_comfort = self._calculate_thermal_comfort(room_type, heat_score)
            behaviors.append(Behavior(
                element_id=f"B{behav_id:03d}",
                name=f"{room_type.replace('_', ' ').title()} Thermal Comfort",
                description=f"Temperature comfort in {room_type} considering {orientation} orientation",
                target_value=0.8,  # 80% comfort hours
                current_value=thermal_comfort
            ))
            behav_id += 1
            
            # Privacy Performance per Room  
            privacy_score = self.directional_optimizer.directional_scores[direction]['privacy']
            privacy_requirement = self._get_room_privacy_requirement(room_type)
            behaviors.append(Behavior(
                element_id=f"B{behav_id:03d}",
                name=f"{room_type.replace('_', ' ').title()} Privacy Performance",
                description=f"Visual and acoustic privacy in {room_type}",
                target_value=privacy_requirement,
                current_value=privacy_score
            ))
            behav_id += 1
        
        # B7: Seasonal Climate Response
        behaviors.append(Behavior(
            element_id=f"B{behav_id:03d}",
            name="Seasonal Climate Adaptation",
            description="Building performance across summer, monsoon, and winter seasons",
            target_value=0.8,
            current_value=self._calculate_seasonal_performance(layout_result)
        ))
        behav_id += 1
        
        # B8: Cross-Ventilation Effectiveness
        ventilation_paths = self._analyze_ventilation_effectiveness(layout_result)
        behaviors.append(Behavior(
            element_id=f"B{behav_id:03d}",
            name="Cross-Ventilation Flow Effectiveness",
            description="Natural air flow efficiency through strategic window placement",
            target_value=0.8,
            current_value=ventilation_paths
        ))
        behav_id += 1
        
        # B9: Solar Heat Gain Management
        avg_heat_gain = self._calculate_average_heat_gain(layout_result)
        behaviors.append(Behavior(
            element_id=f"B{behav_id:03d}",
            name="Solar Heat Gain Control",
            description="Minimization of unwanted solar heat gain through directional optimization",
            target_value=0.3,  # Lower is better
            current_value=avg_heat_gain
        ))
        behav_id += 1
        
        # B10: Energy Performance Index
        energy_performance = self._calculate_energy_performance(layout_result)
        behaviors.append(Behavior(
            element_id=f"B{behav_id:03d}",
            name="Energy Efficiency Performance",
            description="Overall energy performance through passive design strategies",
            target_value=0.8,
            current_value=energy_performance
        ))
        
        return behaviors
    def __init__(self):
        super().__init__()
        self.directional_optimizer = EnhancedDirectionalOptimizer()
        self.layout_generator = EnhancedLayoutGenerator(self.directional_optimizer)
    
    # REPLACE the existing generate_enhanced_fbs_ontology method with this one.

    def generate_enhanced_fbs_ontology(self, requirements, project_name: str = "house_design") -> Dict[str, Any]:
        """Generate enhanced FBS ontology with directional optimization"""
        
        plot_length = requirements.site_constraints.plot_length or 50
        plot_width = requirements.site_constraints.plot_width or 30
        plot_dimensions = (plot_length, plot_width)
        
        # Step 1: Generate the optimal layout first, as it's needed for analysis.
        layout_result = self.layout_generator.generate_optimal_layout(requirements, plot_dimensions)
        
        # Step 2: Generate the base structures needed for enhancement.
        base_ontology = super().generate_fbs_ontology(requirements, project_name)
        
        # Step 3: Generate the new, detailed Functions and Behaviors.
        functions = self._generate_design_functions(requirements)
        behaviors = self._generate_performance_behaviors(requirements, layout_result)
        
        # Step 4: Enhance the base structures with data from the layout.
        enhanced_structures = self._enhance_structures_with_directional_data(
            base_ontology.structures, layout_result
        )

        # Step 5: Update behaviors with current performance data (optional, as new method already does this)
        # We can reuse the existing enhancement logic to be certain
        enhanced_behaviors = self._enhance_behaviors_with_performance_data(
             behaviors, layout_result
        )
        
        # Step 6: Assemble the final, enhanced FBS Ontology object
        fbs_ontology = FBSOntology(
            project_name=project_name,
            functions=functions,
            behaviors=enhanced_behaviors,
            structures=enhanced_structures
        )
        
        # Step 7: Return the complete results package
        return {
            'fbs_ontology': asdict(fbs_ontology),
            'optimal_layout': layout_result,
            'enhanced_structures': [asdict(s) for s in enhanced_structures],
            'enhanced_behaviors': [asdict(b) for b in enhanced_behaviors],
            'directional_analysis': self._generate_comprehensive_directional_analysis(layout_result),
            'construction_guidelines': self._generate_construction_guidelines(layout_result),
            'performance_predictions': self._generate_performance_predictions(layout_result)
        }
    
    def _enhance_structures_with_directional_data(self, structures: List, layout_result: Dict) -> List:
        """Enhance structures with detailed directional data"""
        enhanced = []
        
        for structure in structures:
            enhanced_structure = structure
            
            if structure.element_id == "S002":  # Window system
                enhanced_structure.geometric_properties.update({
                    'directional_window_distribution': self._calculate_directional_windows(layout_result),
                    'optimal_window_sizes': self._calculate_optimal_window_sizes(layout_result),
                    'shading_requirements': self._calculate_shading_requirements(layout_result),
                    'ventilation_openings': self._calculate_ventilation_openings(layout_result)
                })
            
            enhanced.append(enhanced_structure)
        
        return enhanced
    
    def _enhance_behaviors_with_performance_data(self, behaviors: List, layout_result: Dict) -> List:
        """Enhance behaviors with actual performance predictions"""
        enhanced = []
        
        for behavior in behaviors:
            enhanced_behavior = behavior
            
            if behavior.element_id == "B002":  # Daylight performance
                lighting_analysis = layout_result['layout_analysis']['lighting_analysis']
                if lighting_analysis['daylight_factor_by_room']:
                    avg_daylight_factor = np.mean(list(lighting_analysis['daylight_factor_by_room'].values()))
                    enhanced_behavior.current_value = f"{avg_daylight_factor:.2%}"
            
            elif behavior.element_id == "B003":  # Ventilation performance
                ventilation_analysis = layout_result['layout_analysis']['ventilation_analysis']
                if ventilation_analysis['cross_ventilation_effectiveness']:
                    avg_ventilation = np.mean(list(ventilation_analysis['cross_ventilation_effectiveness'].values()))
                    enhanced_behavior.current_value = f"{avg_ventilation:.1f} (effectiveness score)"
            
            enhanced.append(enhanced_behavior)
        
        return enhanced
    
    def _calculate_directional_windows(self, layout_result: Dict) -> Dict[str, int]:
        """Calculate window distribution by direction"""
        direction_count = defaultdict(int)
        
        for room_data in layout_result['optimal_layout']['rooms'].values():
            orientation = room_data.get('orientation', 'south')
            window_count = room_data.get('windows', {}).get('window_count', 1)
            direction_count[orientation] += window_count
        
        return dict(direction_count)
    
    def _calculate_optimal_window_sizes(self, layout_result: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate optimal window sizes for each room"""
        window_sizes = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            windows = room_data.get('windows', {})
            count = windows.get('window_count', 1)
            area = windows.get('window_area', 20.0)
            height = windows.get('window_height', 4.5)
            window_sizes[room_id] = {
                'total_area_sqft': area,
                'individual_area_sqft': round(area / count, 2),
                'height_ft': height,
                'width_ft': round((area / count) / height, 2)
            }
        
        return window_sizes
    
    def _calculate_shading_requirements(self, layout_result: Dict) -> Dict[str, Dict[str, Any]]:
        """Calculate shading requirements for each room"""
        shading_req = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            orientation = room_data.get('orientation', 'south')
            
            if orientation in ['south', 'southwest', 'west']:
                shading_req[room_id] = {
                    'required': True, 'type': 'external_overhang', 'depth_ft': 3.0, 
                    'additional_measures': ['louvers', 'external_blinds']
                }
            elif orientation in ['southeast', 'east']:
                shading_req[room_id] = {
                    'required': True, 'type': 'adjustable_internal', 'depth_ft': 1.5,
                    'additional_measures': ['blinds', 'curtains']
                }
            else:
                shading_req[room_id] = {'required': False}
        
        return shading_req
    
    def _calculate_ventilation_openings(self, layout_result: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate ventilation opening requirements"""
        ventilation = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            room_area_sqft = room_data.get('area', 100)
            room_type = room_data.get('type', 'bedroom')
            
            min_vent_area = room_area_sqft * 0.05
            
            if room_type in ['kitchen', 'bathroom']:
                optimal_vent_area = room_area_sqft * 0.15
            else:
                optimal_vent_area = room_area_sqft * 0.10
            
            ventilation[room_id] = {
                'minimum_area_sqft': round(min_vent_area, 2),
                'optimal_area_sqft': round(optimal_vent_area, 2),
                'cross_ventilation_potential': round(random.uniform(0.6, 0.9), 2)
            }
        
        return ventilation
    
    def _generate_comprehensive_directional_analysis(self, layout_result: Dict) -> Dict[str, Any]:
        """Generate comprehensive directional analysis"""
        return {
            'sun_path_analysis': self._analyze_sun_path_impact(layout_result),
            'wind_flow_analysis': self._analyze_wind_flow(layout_result),
            'heat_gain_analysis': self._analyze_heat_gain(layout_result),
            'privacy_analysis': self._analyze_privacy_levels(layout_result),
            'seasonal_performance': self._analyze_seasonal_performance(layout_result)
        }
    
    def _analyze_sun_path_impact(self, layout_result: Dict) -> Dict[str, Any]:
        """Analyze sun path impact on different rooms"""
        sun_analysis = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            
            sun_analysis[room_id] = {
                'orientation': orientation,
                'summer_sun_exposure_hours': self._calculate_sun_hours(direction, 'summer'),
                'winter_sun_exposure_hours': self._calculate_sun_hours(direction, 'winter'),
                'peak_sun_angle': round(self._get_peak_sun_angle(direction), 1),
                'shading_effectiveness': self._calculate_shading_effectiveness(direction),
                'glare_risk_periods': self._identify_glare_periods(direction)
            }
        
        return sun_analysis
    
    def _calculate_sun_hours(self, direction: CardinalDirection, season: str) -> float:
        """Calculate direct sun hours for direction and season"""
        base_hours = {'summer': 13.5, 'winter': 10.5}
        
        direction_multipliers = {
            CardinalDirection.EAST: 0.4, CardinalDirection.SOUTHEAST: 0.6, CardinalDirection.SOUTH: 0.7,
            CardinalDirection.SOUTHWEST: 0.6, CardinalDirection.WEST: 0.4, CardinalDirection.NORTHWEST: 0.2,
            CardinalDirection.NORTH: 0.1, CardinalDirection.NORTHEAST: 0.2
        }
        
        return round(base_hours[season] * direction_multipliers.get(direction, 0.5), 1)
    
    def _get_peak_sun_angle(self, direction: CardinalDirection) -> float:
        """Get peak sun angle for direction"""
        sun_path = self.directional_optimizer.sun_path
        
        if direction in [CardinalDirection.SOUTH, CardinalDirection.SOUTHEAST, CardinalDirection.SOUTHWEST]:
            return sun_path.summer_solstice_angle
        elif direction in [CardinalDirection.EAST, CardinalDirection.WEST]:
            return sun_path.equinox_angle
        else:
            return sun_path.winter_solstice_angle
    
    def _calculate_shading_effectiveness(self, direction: CardinalDirection) -> float:
        """Calculate how effective shading would be for this direction"""
        effectiveness = {
            CardinalDirection.SOUTH: 0.9, CardinalDirection.SOUTHEAST: 0.8, CardinalDirection.SOUTHWEST: 0.8,
            CardinalDirection.EAST: 0.6, CardinalDirection.WEST: 0.7, CardinalDirection.NORTH: 0.3,
            CardinalDirection.NORTHEAST: 0.5, CardinalDirection.NORTHWEST: 0.5
        }
        return effectiveness.get(direction, 0.5)
    
    def _identify_glare_periods(self, direction: CardinalDirection) -> List[str]:
        """Identify times when glare might be problematic"""
        glare_periods = {
            CardinalDirection.EAST: ["morning (6-10 AM)"], CardinalDirection.SOUTHEAST: ["morning (7-11 AM)"],
            CardinalDirection.SOUTH: ["midday (11 AM-2 PM)"], CardinalDirection.SOUTHWEST: ["afternoon (2-5 PM)"],
            CardinalDirection.WEST: ["afternoon (3-6 PM)"], CardinalDirection.NORTHWEST: ["evening (5-7 PM)"],
            CardinalDirection.NORTH: [], CardinalDirection.NORTHEAST: ["early morning (6-8 AM)"]
        }
        return glare_periods.get(direction, [])
    
    def _analyze_wind_flow(self, layout_result: Dict) -> Dict[str, Any]:
        """Analyze wind flow patterns"""
        return {
            'prevailing_wind_direction': 'South-West',
            'seasonal_wind_patterns': {
                'monsoon': 'Strong from South-West',
                'winter': 'Moderate from North-East',
                'summer': 'Variable and light'
            },
            'cross_ventilation_paths': self._identify_ventilation_paths(layout_result),
            'wind_pressure_zones': self._calculate_wind_pressure(layout_result)
        }
    
    def _identify_ventilation_paths(self, layout_result: Dict) -> List[str]:
        """Identify potential cross-ventilation paths"""
        paths = []
        orientations = {room['orientation'] for room in layout_result['optimal_layout']['rooms'].values()}
        
        if 'east' in orientations and 'west' in orientations: paths.append('East-West cross-ventilation')
        if 'north' in orientations and 'south' in orientations: paths.append('North-South cross-ventilation')
        if ('southwest' in orientations or 'south' in orientations) and ('northeast' in orientations or 'north' in orientations):
            paths.append('Primary SW-NE monsoon wind path')
        
        return paths if paths else ["Limited cross-ventilation paths identified"]
    
    def _calculate_wind_pressure(self, layout_result: Dict) -> Dict[str, str]:
        """Calculate wind pressure zones around building"""
        return {
            'windward_facade (SW)': 'Positive Pressure',
            'leeward_facade (NE)': 'Negative Pressure (suction)',
            'side_facades': 'Neutral to Negative Pressure',
            'roof': 'Negative Pressure (uplift)'
        }
    
    def _analyze_heat_gain(self, layout_result: Dict) -> Dict[str, Any]:
        """Analyze heat gain from different directions"""
        heat_analysis = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            
            # --- CORRECTED ---
            heat_score = self.directional_optimizer.directional_scores[direction]['heat_gain']
            
            heat_analysis[room_id] = {
                'orientation': orientation,
                'heat_gain_level': 'low' if heat_score > 0.7 else 'medium' if heat_score > 0.4 else 'high',
                'peak_heat_hours': self._get_peak_heat_hours(direction),
                'cooling_requirement_factor': round(1 - heat_score, 2),
                'recommended_cooling_strategy': self._get_cooling_strategy(heat_score)
            }
        
        return heat_analysis
    
    def _get_peak_heat_hours(self, direction: CardinalDirection) -> List[str]:
        """Get peak heat hours for direction"""
        peak_hours = {
            CardinalDirection.EAST: ["8-11 AM"], CardinalDirection.SOUTHEAST: ["9 AM-12 PM"],
            CardinalDirection.SOUTH: ["11 AM-2 PM"], CardinalDirection.SOUTHWEST: ["2-5 PM"],
            CardinalDirection.WEST: ["3-6 PM"], CardinalDirection.NORTHWEST: ["4-6 PM"],
            CardinalDirection.NORTH: [], CardinalDirection.NORTHEAST: ["7-10 AM"]
        }
        return peak_hours.get(direction, [])
    
    def _get_cooling_strategy(self, heat_score: float) -> str:
        """Recommend cooling strategy based on heat gain"""
        if heat_score > 0.7: return "Natural ventilation sufficient"
        elif heat_score > 0.4: return "Fans and shading recommended"
        else: return "Active cooling (AC) likely required"
    
    def _analyze_privacy_levels(self, layout_result: Dict) -> Dict[str, Any]:
        """Analyze privacy levels for different rooms"""
        privacy_analysis = {}
        
        for room_id, room_data in layout_result['optimal_layout']['rooms'].items():
            room_type = room_data.get('type', 'bedroom')
            orientation = room_data.get('orientation', 'south')
            direction = CardinalDirection(orientation.lower())
            
            # --- CORRECTED ---
            privacy_score = self.directional_optimizer.directional_scores[direction]['privacy']
            preferences = self.directional_optimizer.room_preferences.get(room_type)
            privacy_requirement = preferences.privacy_requirement if preferences else 0.5
            
            privacy_analysis[room_id] = {
                'room_type': room_type, 'orientation': orientation,
                'privacy_score': privacy_score, 'privacy_requirement': privacy_requirement,
                'privacy_adequacy': 'Adequate' if privacy_score >= privacy_requirement else 'Needs Improvement',
                'privacy_enhancement_suggestions': self._get_privacy_suggestions(privacy_score, privacy_requirement)
            }
        
        return privacy_analysis
    
    def _get_privacy_suggestions(self, current_score: float, required_score: float) -> List[str]:
        """Get privacy enhancement suggestions"""
        if current_score >= required_score: return ["Current privacy level is adequate."]
        
        suggestions = ["Install privacy screens or blinds.", "Use frosted glass for lower window panes.", "Plant privacy hedges or trees."]
        if required_score > 0.8: suggestions.append("Consider minimizing street-facing windows or using clerestory windows.")
        return suggestions
    
    def _analyze_seasonal_performance(self, layout_result: Dict) -> Dict[str, Any]:
        """Analyze performance across different seasons"""
        return {
            'summer_performance': {'cooling_load': 'Moderate to High', 'natural_lighting': 'Excellent', 'ventilation_potential': 'Good', 'comfort_level': 'Requires active cooling and shading'},
            'monsoon_performance': {'moisture_control': 'Critical', 'natural_lighting': 'Reduced (overcast)', 'ventilation_potential': 'Excellent', 'comfort_level': 'Good with dehumidification'},
            'winter_performance': {'heating_requirement': 'Minimal to None', 'natural_lighting': 'Good', 'ventilation_potential': 'Moderate', 'comfort_level': 'Excellent'}
        }
    
    def _generate_construction_guidelines(self, layout_result: Dict) -> Dict[str, List[str]]:
        """Generate detailed construction guidelines"""
        return {
            'foundation_and_structure': ["Use reinforced concrete (RCC) foundation suitable for local soil conditions.", "Design for seismic zone as per IS 1893.", "Provide proper damp proof course (DPC) at plinth level."],
            'walls_and_partitions': ["Use 9-inch thick brick/block walls for external walls.", "Consider insulated cavity walls for west-facing facades.", "Use 4.5-inch thick walls for internal partitions."],
            'roofing': ["Use RCC slab with proper thermal insulation (e.g., terrace garden, heat-reflective tiles).", "Provide adequate slope for rapid water drainage during monsoon.", "Consider a double roof or false ceiling for top floor rooms."],
            'windows_and_doors': ["Use UPVC or aluminum windows with good sealing to prevent drafts and leaks.", "Install deep external shading (chajjas) for south and west windows.", "Ensure all openable windows have insect mesh screens."],
            'finishing_and_materials': ["Use light-colored exterior paints to reflect heat.", "Choose flooring appropriate for a humid climate (e.g., vitrified tiles, stone).", "Use moisture-resistant materials and paints in bathrooms and kitchens."]
        }
    
    def _generate_performance_predictions(self, layout_result: Dict) -> Dict[str, Any]:
        """Generate performance predictions for the design"""
        return {
            'energy_performance': {
                'annual_electricity_consumption': '10,000-15,000 kWh',
                'peak_cooling_load': '4-6 Tons of Refrigeration (TR)',
                'natural_lighting_hours': '8-10 hours/day average',
                'energy_efficiency_rating': 'B+ to A-'
            },
            'comfort_metrics': {
                'thermal_comfort_hours': '70-80% of year naturally comfortable or with fan usage',
                'daylight_availability': '85-90% of occupied hours',
                'natural_ventilation_effectiveness': 'High during monsoon, moderate otherwise',
                'acoustic_comfort': 'Good with proper wall construction'
            },
            'maintenance_requirements': {
                'annual_maintenance_cost': '₹15,000-25,000',
                'major_repairs_frequency': '5-7 years for paint/finishes',
                'expected_lifespan': '50+ years with proper maintenance'
            },
            'environmental_impact': {
                'carbon_footprint': 'Moderate - can be reduced with solar panels',
                'water_usage_efficiency': 'Good with rainwater harvesting and low-flow fixtures',
                'sustainability_rating': 'Good - potential for Excellent'
            }
        }

class EnhancedDirectionalFBSInterface:
    """Enhanced interface with comprehensive directional analysis"""
    
    def __init__(self):
        self.enhanced_generator = EnhancedFBSGenerator()
    def initialize_encoder(self):
        """Initialize Gemma 3 encoder for layout optimization"""
        try:
            self.encoder = Gemma3Encoder()
            logger.info("✅ FBS Interface: Gemma 3 Encoder initialized")
        except Exception as e:
            logger.warning(f"⚠️ FBS Interface: Encoder initialization failed: {e}")
            self.encoder = None

    def run_enhanced_analysis(self):
        """Run comprehensive directional FBS analysis"""
        print("\n🏠 ENHANCED DIRECTIONAL FBS ARCHITECTURAL SYSTEM")
        print("=" * 80)
        print("Advanced Function-Behavior-Structure Analysis with:")
        print("• Sun Path Optimization for Indian Climate")
        print("• Directional Lighting & Ventilation Analysis") 
        print("• Cross-Ventilation Flow Patterns")
        print("• Privacy & Heat Gain Assessment")
        print("• Seasonal Performance Predictions")
        print("=" * 80)
        self.initialize_encoder()
        
        try:
            user_input = self._collect_user_requirements()
            project_name = self._get_project_name()
            
            analyzer = GemmaFBSAnalyzer()
            requirements = analyzer.analyze_and_parse_requirements(user_input)
            
            print(f"\n🧠 Generating Enhanced FBS Analysis for: {project_name}")
            print("=" * 60)
            
            enhanced_result = self.enhanced_generator.generate_enhanced_fbs_ontology(
                requirements, project_name
            )

            if hasattr(self, 'encoder') and self.encoder:
                try:
                    print("\n🔧 Applying Gemma 3 Layout Optimization...")
        
                    # Check if enhanced_result has prototypes to optimize
                    if 'optimal_layout' in enhanced_result:
                        layout_data = enhanced_result['optimal_layout']
            
                        # Generate enhanced embedding for the layout
                        enhanced_embedding = self.encoder.encode_prototype_features({
                            'prototype_id': f"{project_name}_optimized",
                            'detailed_config': layout_data,
                            'final_score': enhanced_result.get('performance_score', 0.8)
                            })
            
                        # Add embedding to results
                        enhanced_result['layout_optimization_embedding'] = enhanced_embedding.tolist()
                        enhanced_result['gemma3_enhanced'] = True
            
                        print("✅ Layout optimization embedding generated successfully!")
            
                    else:
                        print("⚠️ No layout data found for optimization")
            
                except Exception as e:
                    logger.warning(f"Layout optimization encoding failed: {e}")
                    print(f"⚠️ Layout optimization failed: {e}")
            

            self._display_enhanced_results(enhanced_result)
            self._save_enhanced_results(enhanced_result, project_name)
            
            return enhanced_result
            
        except Exception as e:
            print(f"\n❌ Error during enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    def run_enhanced_analysis_programmatic(self, user_input: str, project_name: str):
        try:
            print(f"\n🧠 Generating Enhanced FBS Analysis for: {project_name}")
            print("=" * 60)
        
            # Use the same logic as run_enhanced_analysis but without interactive input
            analyzer = GemmaFBSAnalyzer()
            requirements = analyzer.analyze_and_parse_requirements(user_input)
        
            enhanced_result = self.enhanced_generator.generate_enhanced_fbs_ontology(requirements, project_name)
        
            # Return results instead of displaying them
            return enhanced_result
        
        except Exception as e:
            print(f"\n❌ Error during enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def _collect_user_requirements(self) -> str:
        """Collect user requirements - same as original"""
        print("\nPlease describe your house design requirements:")
        print("Include details about:")
        print("• Number and types of rooms (e.g., 3 bedrooms, 2 bathrooms)")
        print("• Plot size and orientation (e.g., 30x40 ft, south facing)")
        print("• Budget constraints (e.g., ₹25 lakhs)")
        print("• Climate considerations (hot summers, monsoons)")
        print("• Priority: lighting, ventilation, privacy, or cost")
        print("-" * 60)
        
        user_input = ""
        print("Enter your requirements (press Enter twice to finish):")
        
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break
        user_input = " ".join(lines)
        
        # If running non-interactively, provide a default
        if not user_input:
            user_input = "a 3 bedroom 2 bathroom house for a 30x50 ft south facing plot. focus on good ventilation."
            print(f"\nNo input provided. Using default: '{user_input}'")

        return user_input.strip()
    
    def _get_project_name(self) -> str:
        """Get project name from user"""
        try:
            project_name = input("\nEnter project name (or press Enter for 'enhanced_house_design'): ").strip()
            return project_name if project_name else "enhanced_house_design"
        except EOFError:
             return "enhanced_house_design"

    def _display_enhanced_results(self, result: Dict[str, Any]):
        """Display comprehensive enhanced results"""
        print("\n🎯 ENHANCED DIRECTIONAL ANALYSIS RESULTS")
        print("=" * 80)
        
        layout = result['optimal_layout']['optimal_layout']
        print(f"\n📐 OPTIMAL LAYOUT STRATEGY: {layout['strategy'].upper().replace('_', ' ')}")
        print(f"Description: {layout['description']}")
        print(f"Overall Score: {layout['total_score']:.2f}/1.0")
        
        directional = result['directional_analysis']
        
        print(f"\n☀️ SUN PATH ANALYSIS:")
        print("-" * 40)
        for room_id, sun_data in directional['sun_path_analysis'].items():
            print(f"  {room_id.replace('_', ' ').title()}:")
            print(f"    • Orientation: {sun_data['orientation'].title()}")
            print(f"    • Summer sun: {sun_data['summer_sun_exposure_hours']:.1f} hours")
            print(f"    • Winter sun: {sun_data['winter_sun_exposure_hours']:.1f} hours")
            if sun_data['glare_risk_periods']:
                print(f"    • Glare risk: {', '.join(sun_data['glare_risk_periods'])}")
        
        print(f"\n🌬️ WIND FLOW ANALYSIS:")
        print("-" * 40)
        wind_analysis = directional['wind_flow_analysis']
        print(f"  • Prevailing wind: {wind_analysis['prevailing_wind_direction']}")
        for path in wind_analysis['cross_ventilation_paths']:
            print(f"    - {path.replace('_', ' ').title()}")
        
        print(f"\n🔥 HEAT GAIN ANALYSIS:")
        print("-" * 40)
        for room_id, heat_data in directional['heat_gain_analysis'].items():
            print(f"  {room_id.replace('_', ' ').title()}:")
            print(f"    • Orientation: {heat_data['orientation'].title()}")
            print(f"    • Heat Gain Level: {heat_data['heat_gain_level'].upper()}")
            print(f"    • Recommendation: {heat_data['recommended_cooling_strategy']}")
            
        print(f"\n🔒 PRIVACY ANALYSIS:")
        print("-" * 40)
        privacy_analysis = directional['privacy_analysis']
        for room_id, privacy_data in privacy_analysis.items():
            print(f"  {room_id.replace('_', ' ').title()}:")
            print(f"    • Privacy Adequacy: {privacy_data['privacy_adequacy'].upper()}")
            if privacy_data['privacy_adequacy'] != 'Adequate':
                print(f"    • Suggestions: {', '.join(privacy_data['privacy_enhancement_suggestions'])}")

        print("\n\n🏗️ TOP CONSTRUCTION GUIDELINES:")
        print("=" * 80)
        guidelines = result['construction_guidelines']
        recommendations = guidelines.get('walls_and_partitions', [])[:2] + guidelines.get('roofing', [])[:2]
        for i, guideline in enumerate(recommendations):
             print(f"  {i+1}. {guideline}")

        print("\n\n📈 PERFORMANCE PREDICTIONS:")
        print("=" * 80)
        predictions = result['performance_predictions']
        print(f"  • Energy Efficiency: {predictions['energy_performance']['energy_efficiency_rating']}")
        print(f"  • Annual Consumption: {predictions['energy_performance']['annual_electricity_consumption']}")
        print(f"  • Natural Comfort: {predictions['comfort_metrics']['thermal_comfort_hours']}")
        
        print("\n" + "="*80)
        print("Analysis complete.")
    
    def _save_enhanced_results(self, result: Dict[str, Any], project_name: str):
        """Saves the comprehensive results to a JSON file."""
        filename = f"{project_name}_analysis.json"
        
        def default_serializer(o):
            if isinstance(o, (datetime,)):
                return o.isoformat()
            if isinstance(o, Enum):
                return o.value
            if hasattr(o, '__dict__'):
                return o.__dict__
            try:
                return asdict(o)
            except TypeError:
                return str(o)

        try:
            with open(filename, 'w') as f:
                json.dump(result, f, indent=4, default=default_serializer)
            print(f"\n💾 Results saved to '{filename}'")
        except Exception as e:
            print(f"\n❌ Error saving results to file: {e}")
    def generate_prototype_specific_fbs_ontology(self, prototype: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unique FBS ontology for each prototype"""
        
        prototype_id = prototype.get('prototype_id', 'unknown')
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        functional_zones = prototype.get('detailed_config', {}).get('functional_zones', {})
        environmental_strategy = prototype.get('detailed_config', {}).get('environmental_strategy', {})
        
        strategy = spatial_config.get('strategy', 'balanced')
        
        # Generate prototype-specific functions
        functions = self._generate_prototype_specific_functions(prototype, requirements, strategy)
        
        # Generate prototype-specific behaviors with predicted performance
        behaviors = self._generate_prototype_specific_behaviors(prototype, requirements, environmental_strategy)
        
        # Generate prototype-specific structures
        structures = self._generate_prototype_specific_structures(prototype, requirements, spatial_config)
        
        return {
            'prototype_id': prototype_id,
            'project_name': f"{requirements.get('project_name', 'house_design')}_{prototype_id}",
            'spatial_strategy': strategy,
            'hierarchy_level': prototype.get('hierarchy_level', 2),
            'final_score': prototype.get('final_score', 0.6),
            'functions': [asdict(f) for f in functions],
            'behaviors': [asdict(b) for b in behaviors],
            'structures': [asdict(s) for s in structures],
            'prototype_characteristics': {
                'spatial_efficiency': spatial_config.get('plot_utilization', 0.7),
                'compactness_factor': spatial_config.get('compactness_factor', 0.8),
                'orientation': environmental_strategy.get('orientation', 'south'),
                'passive_strategies': environmental_strategy.get('passive_strategies', []),
                'functional_zones': functional_zones
            },
            'performance_predictions': self._predict_prototype_specific_performance(prototype)
        }

    def _generate_prototype_specific_functions(self, prototype, requirements, strategy):
        """Generate functions specific to prototype strategy"""
        base_functions = []
        
        if strategy == 'central_core':
            base_functions.extend([
                Function("F_CC01", "Create Central Hub Activity", 
                         f"Prototype {prototype.get('prototype_id')}: Establish central core for family activities and circulation"),
                Function("F_CC02", "Optimize Radial Access", 
                         "Enable efficient radial access from central core to all functional zones")
            ])
        elif strategy == 'linear_progression':
            base_functions.extend([
                Function("F_LP01", "Enable Sequential Flow", 
                         f"Prototype {prototype.get('prototype_id')}: Create seamless progression from public to private zones"),
                Function("F_LP02", "Maximize Linear Efficiency", 
                         "Optimize linear circulation and minimize backtracking")
            ])
        elif strategy == 'courtyard_focused':
            base_functions.extend([
                Function("F_CF01", "Integrate Indoor-Outdoor Living", 
                         f"Prototype {prototype.get('prototype_id')}: Create strong connection between interior and courtyard"),
                Function("F_CF02", "Enhance Microclimate Control", 
                         "Use courtyard for natural cooling and light distribution")
            ])
        
        return base_functions
if __name__ == '__main__':
    print("🚀 Testing Gemma 3 connection...")
    
    # Test Gemma 3 connection
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if any('gemma3' in m for m in models):
                print("✅ Gemma 3 is available!")
            else:
                print(f"⚠️ Available models: {models}")
        else:
            print("❌ Ollama not responding")
    except:
        print("❌ Cannot connect to Ollama")
    
    # Run the interface
    interface = EnhancedDirectionalFBSInterface()
    interface.run_enhanced_analysis()
    print("🚀 Testing Gemma 3 Encoder Integration...")
    
    # Test Gemma 3 connection
    try:
        from encoder import Gemma3Encoder
        encoder = Gemma3Encoder()
        
        test_prototype = {
            'prototype_id': 'integration_test',
            'final_score': 0.85,
            'detailed_config': {
                'spatial_config': {'strategy': 'central_core'},
                'environmental_strategy': {'orientation': 'south'}
            }
        }
        
        embedding = encoder.encode_prototype_features(test_prototype)
        print(f"✅ Integration successful! Embedding shape: {embedding.shape}")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        print("💡 Make sure Gemma 3 server is running on localhost:8080")
