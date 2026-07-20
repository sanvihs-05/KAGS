# core/fbsl_models.py
"""
Complete FBSL Data Models and Ontology Implementation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import uuid
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon, LineString


# ============================================================================
# ENUMERATIONS
# ============================================================================

class NodeType(Enum):
    PROBLEM = "problem"
    DESIGN_PROTOTYPE = "design_prototype"
    EVALUATION = "evaluation"
    RESEARCH = "research"
    REFINEMENT = "refinement"


class FunctionCategory(Enum):
    SPATIAL = "spatial"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    TECHNICAL = "technical"
    ECONOMIC = "economic"


class BehaviorType(Enum):
    EXPECTED = "expected"
    ACTUAL = "actual"
    MEASURED = "measured"
    SIMULATED = "simulated"


class BehaviorCategory(Enum):
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    LIGHTING = "lighting"
    SPATIAL = "spatial"
    STRUCTURAL = "structural"
    ENERGY = "energy"
    VENTILATION = "ventilation"


class StructureType(Enum):
    WALL = "wall"
    COLUMN = "column"
    BEAM = "beam"
    SLAB = "slab"
    ROOF = "roof"
    FOUNDATION = "foundation"
    PARTITION = "partition"
    MEP = "mep"


class TransformationType(Enum):
    FUNCTIONAL_DECOMPOSITION = "functional_decomposition"
    BEHAVIORAL_OPTIMIZATION = "behavioral_optimization"
    STRUCTURAL_VARIATION = "structural_variation"
    LAYOUT_PERMUTATION = "layout_permutation"
    AGGREGATION = "aggregation"
    REFINEMENT = "refinement"


# ============================================================================
# CORE FBSL COMPONENT CLASSES
# ============================================================================

@dataclass
class Function:
    """Represents a design function (teleological purpose)"""
    function_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: FunctionCategory = FunctionCategory.SPATIAL
    description: str = ""
    priority: float = 0.5  # 0 to 1
    
    # Target users and activities
    target_users: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    
    # Spatial requirements
    spatial_requirements: Dict[str, Any] = field(default_factory=dict)
    # {min_area, preferred_area, max_area, height, shape_preference}
    
    temporal_requirements: Dict[str, Any] = field(default_factory=dict)
    # {operating_hours, seasonal_variations, peak_times}
    
    # Relationships with other functions
    depends_on: List[str] = field(default_factory=list)  # function_ids
    conflicts_with: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)
    
    # Metadata
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function_id': self.function_id,
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'priority': self.priority,
            'target_users': self.target_users,
            'activities': self.activities,
            'spatial_requirements': self.spatial_requirements,
            'temporal_requirements': self.temporal_requirements,
            'depends_on': self.depends_on,
            'conflicts_with': self.conflicts_with,
            'enables': self.enables,
            'metadata': self.metadata
        }


@dataclass
class Behavior:
    """Represents a measurable performance attribute"""
    behavior_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    behavior_type: BehaviorType = BehaviorType.EXPECTED
    category: BehaviorCategory = BehaviorCategory.SPATIAL
    
    # Link to function
    derived_from_function: Optional[str] = None  # function_id
    
    # Metric definition
    metric_name: str = ""
    metric_unit: str = ""
    target_value: Optional[float] = None
    actual_value: Optional[float] = None
    tolerance: float = 0.1  # 10% default tolerance
    
    # Acceptable ranges
    min_acceptable: Optional[float] = None
    max_acceptable: Optional[float] = None
    optimal_range: Optional[Tuple[float, float]] = None
    
    # Performance data
    calculation_method: str = ""
    simulation_results: Dict[str, Any] = field(default_factory=dict)
    measured_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation
    is_satisfied: bool = False
    deviation_percentage: float = 0.0
    
    # Metadata
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_satisfaction(self) -> bool:
        """Check if behavior is satisfied"""
        if self.target_value is None or self.actual_value is None:
            return False
        
        deviation = abs(self.actual_value - self.target_value) / self.target_value
        self.deviation_percentage = deviation * 100
        self.is_satisfied = deviation <= self.tolerance
        return self.is_satisfied
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'behavior_id': self.behavior_id,
            'behavior_type': self.behavior_type.value,
            'category': self.category.value,
            'derived_from_function': self.derived_from_function,
            'metric_name': self.metric_name,
            'metric_unit': self.metric_unit,
            'target_value': self.target_value,
            'actual_value': self.actual_value,
            'tolerance': self.tolerance,
            'min_acceptable': self.min_acceptable,
            'max_acceptable': self.max_acceptable,
            'optimal_range': self.optimal_range,
            'is_satisfied': self.is_satisfied,
            'deviation_percentage': self.deviation_percentage,
            'metadata': self.metadata
        }


@dataclass
class Structure:
    """Represents a physical component or system"""
    structure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    structure_type: StructureType = StructureType.WALL
    category: str = "structural"  # structural, envelope, partition, services
    
    # Material properties
    material_type: str = ""
    material_properties: Dict[str, Any] = field(default_factory=dict)
    # {strength, thermal_conductivity, acoustic_rating, density, durability}
    
    # Geometric properties
    dimensions: Dict[str, float] = field(default_factory=dict)
    # {length, width, height, thickness, diameter}
    position: Dict[str, float] = field(default_factory=dict)  # {x, y, z, rotation}
    geometry: Optional[Any] = None  # Shapely geometry object
    
    # Performance characteristics
    load_bearing: bool = False
    span_capability: Optional[float] = None
    fire_rating: str = ""
    acoustic_rating: str = ""
    thermal_properties: Dict[str, float] = field(default_factory=dict)
    
    # Relationships
    connected_to: List[str] = field(default_factory=list)  # structure_ids
    supports: List[str] = field(default_factory=list)
    supported_by: List[str] = field(default_factory=list)
    
    # Cost and sustainability
    unit_cost: Optional[float] = None
    embodied_carbon: Optional[float] = None
    recyclability_score: float = 0.0
    
    # Metadata
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'structure_id': self.structure_id,
            'name': self.name,
            'structure_type': self.structure_type.value,
            'category': self.category,
            'material_type': self.material_type,
            'material_properties': self.material_properties,
            'dimensions': self.dimensions,
            'position': self.position,
            'load_bearing': self.load_bearing,
            'fire_rating': self.fire_rating,
            'acoustic_rating': self.acoustic_rating,
            'connected_to': self.connected_to,
            'metadata': self.metadata
        }


@dataclass
class Room:
    """Represents a spatial unit in the layout"""
    room_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    room_type: str = ""
    room_number: str = ""
    floor_level: int = 0
    
    # Link to function
    function_id: Optional[str] = None
    
    # Dimensions
    area: float = 0.0
    width: Optional[float] = None
    length: Optional[float] = None
    height: float = 3.0  # default ceiling height
    volume: Optional[float] = None
    
    # Position and geometry
    centroid: Optional[Point] = None
    polygon: Optional[Polygon] = None
    position_vector: Dict[str, float] = field(default_factory=dict)  # {x, y, z}
    
    # Adjacency requirements and actual
    required_adjacencies: List[str] = field(default_factory=list)  # room_ids
    preferred_adjacencies: List[str] = field(default_factory=list)
    avoid_adjacencies: List[str] = field(default_factory=list)
    actual_adjacencies: List[str] = field(default_factory=list)
    
    # Access points
    access_points: List[Dict[str, Any]] = field(default_factory=list)
    # [{door_id, position, type, width}]
    
    # Requirements
    natural_light_required: bool = True
    ventilation_type: str = "natural"  # natural, mechanical, mixed
    acoustic_requirement: str = "standard"  # standard, enhanced, critical
    privacy_level: str = "medium"  # low, medium, high, critical
    accessibility_compliant: bool = True
    
    # Visual representation
    color_code: str = "#f0f0f0"
    display_order: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_volume(self):
        """Calculate room volume"""
        if self.area and self.height:
            self.volume = self.area * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'room_id': self.room_id,
            'name': self.name,
            'room_type': self.room_type,
            'room_number': self.room_number,
            'floor_level': self.floor_level,
            'function_id': self.function_id,
            'area': self.area,
            'width': self.width,
            'length': self.length,
            'height': self.height,
            'volume': self.volume,
            'position_vector': self.position_vector,
            'required_adjacencies': self.required_adjacencies,
            'actual_adjacencies': self.actual_adjacencies,
            'natural_light_required': self.natural_light_required,
            'ventilation_type': self.ventilation_type,
            'privacy_level': self.privacy_level,
            'metadata': self.metadata
        }


@dataclass
class Layout:
    """Represents the complete spatial configuration"""
    layout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    configuration_name: str = ""
    
    # Rooms
    rooms: Dict[str, Room] = field(default_factory=dict)  # room_id -> Room
    
    # Area metrics
    total_area: float = 0.0
    used_area: float = 0.0
    circulation_area: float = 0.0
    service_area: float = 0.0
    
    # Boundary
    boundary: Optional[Polygon] = None
    bounding_box: Optional[Polygon] = None
    
    # Adjacency information
    adjacency_matrix: Optional[np.ndarray] = None  # Required adjacencies
    actual_adjacency_matrix: Optional[np.ndarray] = None  # Actual adjacencies
    adjacency_graph: Optional[nx.Graph] = None
    
    # Circulation
    circulation_paths: List[Dict[str, Any]] = field(default_factory=list)
    circulation_graph: Optional[nx.DiGraph] = None
    
    # Doors and openings
    openings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Efficiency metrics
    space_utilization_ratio: float = 0.0
    circulation_efficiency: float = 0.0
    adjacency_satisfaction_score: float = 0.0
    compactness_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self):
        """Calculate layout efficiency metrics"""
        # Space utilization
        if self.total_area > 0:
            self.space_utilization_ratio = self.used_area / self.total_area
        
        # Circulation efficiency
        if self.used_area > 0:
            circ_ratio = self.circulation_area / self.used_area
            # Optimal is 15-25%
            if 0.15 <= circ_ratio <= 0.25:
                self.circulation_efficiency = 1.0
            elif circ_ratio < 0.15:
                self.circulation_efficiency = circ_ratio / 0.15
            else:
                self.circulation_efficiency = max(0, 1 - (circ_ratio - 0.25) * 2)
        
        # Adjacency satisfaction. When the layout agent has measured
        # satisfaction against the BRIEF's stated requirements
        # (metadata['adjacency_measured']), keep that — the weighted
        # preference matrix below is a heuristic, not the brief.
        if not (self.metadata or {}).get('adjacency_measured'):
            if self.adjacency_matrix is not None and self.actual_adjacency_matrix is not None:
                required = np.sum(self.adjacency_matrix > 0.5)
                satisfied = np.sum((self.adjacency_matrix > 0.5) &
                                 (self.actual_adjacency_matrix > 0.5))
                self.adjacency_satisfaction_score = satisfied / max(1, required)
        
        # Compactness (using isoperimetric quotient)
        if self.boundary:
            perimeter = self.boundary.length
            area = self.boundary.area
            if perimeter > 0:
                self.compactness_score = (4 * np.pi * area) / (perimeter ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        # Room order for interpreting the adjacency matrices' rows/columns
        room_order = list(self.rooms.keys())
        adj = self.adjacency_matrix
        actual_adj = self.actual_adjacency_matrix
        return {
            'layout_id': self.layout_id,
            'configuration_name': self.configuration_name,
            'rooms': {k: v.to_dict() for k, v in self.rooms.items()},
            'room_order': room_order,
            'adjacency_matrix': adj.tolist() if adj is not None else None,
            'actual_adjacency_matrix': actual_adj.tolist() if actual_adj is not None else None,
            'total_area': self.total_area,
            'used_area': self.used_area,
            'circulation_area': self.circulation_area,
            'space_utilization_ratio': self.space_utilization_ratio,
            'circulation_efficiency': self.circulation_efficiency,
            'adjacency_satisfaction_score': self.adjacency_satisfaction_score,
            'compactness_score': self.compactness_score,
            'metadata': self.metadata
        }


# ============================================================================
# MAIN FBSL NODE CLASS
# ============================================================================

@dataclass
class FBSLLayoutNode:
    """Complete FBSL representation for spatial layout generation"""
    
    # Identifiers
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Node metadata
    node_type: NodeType = NodeType.PROBLEM
    generation_level: int = 0  # depth in thought graph
    iteration_number: int = 1
    
    # FBSL Components
    functions: Dict[str, Function] = field(default_factory=dict)  # function_id -> Function
    behaviors: Dict[str, Behavior] = field(default_factory=dict)  # behavior_id -> Behavior
    structures: Dict[str, Structure] = field(default_factory=dict)  # structure_id -> Structure
    layout: Optional[Layout] = None
    
    # Graph representations
    adjacency_graph: Optional[nx.Graph] = None
    circulation_graph: Optional[nx.DiGraph] = None
    visibility_graph: Optional[nx.Graph] = None
    
    # Scores
    functional_score: float = 0.0
    behavioral_score: float = 0.0
    structural_score: float = 0.0
    layout_score: float = 0.0
    sustainability_score: float = 0.0
    composite_score: float = 0.0
    
    # Constraints
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Embeddings
    function_embedding: Optional[np.ndarray] = None
    behavior_embedding: Optional[np.ndarray] = None
    structure_embedding: Optional[np.ndarray] = None
    layout_embedding: Optional[np.ndarray] = None
    composite_embedding: Optional[np.ndarray] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_function(self, function: Function):
        """Add a function to the node"""
        self.functions[function.function_id] = function
        self.updated_at = datetime.now()
    
    def add_behavior(self, behavior: Behavior):
        """Add a behavior to the node"""
        self.behaviors[behavior.behavior_id] = behavior
        self.updated_at = datetime.now()
    
    def add_structure(self, structure: Structure):
        """Add a structure to the node"""
        self.structures[structure.structure_id] = structure
        self.updated_at = datetime.now()
    
    def validate_fbsl_consistency(self) -> Tuple[bool, List[str]]:
        """Validate F->B->S->L consistency"""
        issues = []
        
        # Check if all functions have corresponding behaviors
        for func_id, func in self.functions.items():
            has_behavior = any(
                b.derived_from_function == func_id 
                for b in self.behaviors.values()
            )
            if not has_behavior:
                issues.append(f"Function {func.name} has no corresponding behaviors")
        
        # Check if behaviors are satisfied
        unsatisfied_behaviors = [
            b.metric_name for b in self.behaviors.values() 
            if not b.is_satisfied and b.behavior_type == BehaviorType.EXPECTED
        ]
        if unsatisfied_behaviors:
            issues.append(f"Unsatisfied behaviors: {', '.join(unsatisfied_behaviors)}")
        
        # Check if layout exists when required
        if self.node_type == NodeType.DESIGN_PROTOTYPE and self.layout is None:
            issues.append("Design prototype must have a layout")
        
        return len(issues) == 0, issues
    
    def calculate_composite_embedding(self):
        """
        Combine component embeddings into composite using CONCATENATION
        
        ✅ CRITICAL FIX: Implements theoretical formula e_fbsl = [e_f || e_b || e_s || e_l]
        Where || denotes concatenation, preserving component-specific information
        """
        embeddings = []
        embedding_names = []
        
        # Collect embeddings in order: F, B, S, L
        if self.function_embedding is not None:
            embeddings.append(self.function_embedding)
            embedding_names.append('F')
        if self.behavior_embedding is not None:
            embeddings.append(self.behavior_embedding)
            embedding_names.append('B')
        if self.structure_embedding is not None:
            embeddings.append(self.structure_embedding)
            embedding_names.append('S')
        if self.layout_embedding is not None:
            embeddings.append(self.layout_embedding)
            embedding_names.append('L')
        
        if embeddings:
            # ✅ THEORETICAL IMPLEMENTATION: Concatenate instead of averaging
            # e_fbsl = [e_f || e_b || e_s || e_l]
            # This preserves component-specific information for retrieval
            self.composite_embedding = np.concatenate(embeddings, axis=0)
            
            # Store metadata about which components are included
            self.metadata['embedding_components'] = embedding_names
            self.metadata['embedding_dimensions'] = {
                name: emb.shape[0] if len(emb.shape) > 0 else len(emb)
                for name, emb in zip(embedding_names, embeddings)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage"""
        return {
            'node_id': self.node_id,
            'project_id': self.project_id,
            'parent_node_id': self.parent_node_id,
            'children_ids': self.children_ids,
            'node_type': self.node_type.value,
            'generation_level': self.generation_level,
            'iteration_number': self.iteration_number,
            'functions': {k: v.to_dict() for k, v in self.functions.items()},
            'behaviors': {k: v.to_dict() for k, v in self.behaviors.items()},
            'structures': {k: v.to_dict() for k, v in self.structures.items()},
            'layout': self.layout.to_dict() if self.layout else None,
            'functional_score': self.functional_score,
            'behavioral_score': self.behavioral_score,
            'structural_score': self.structural_score,
            'layout_score': self.layout_score,
            'sustainability_score': self.sustainability_score,
            'composite_score': self.composite_score,
            'constraints_satisfied': self.constraints_satisfied,
            'violations': self.violations,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (f"FBSLLayoutNode(id={self.node_id[:8]}, "
                f"type={self.node_type.value}, "
                f"score={self.composite_score:.3f}, "
                f"funcs={len(self.functions)}, "
                f"behavs={len(self.behaviors)}, "
                f"structs={len(self.structures)})")


# ============================================================================
# HELPER CLASSES
# ============================================================================

@dataclass
class DesignConstraint:
    """Represents a design constraint"""
    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraint_type: str = ""  # spatial, performance, regulatory, budget
    description: str = ""
    is_hard_constraint: bool = True  # hard vs soft constraint
    
    # Constraint definition
    applies_to: str = ""  # function, behavior, structure, layout
    target_component_ids: List[str] = field(default_factory=list)
    
    # Validation function (as string for serialization)
    validation_rule: str = ""
    
    # Status
    is_satisfied: bool = False
    violation_severity: float = 0.0  # 0-1, 1 being critical
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationRule:
    """Represents a design transformation rule"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str = ""
    transformation_type: TransformationType = TransformationType.REFINEMENT
    
    # Applicability conditions
    preconditions: List[str] = field(default_factory=list)
    applicable_node_types: List[NodeType] = field(default_factory=list)
    
    # Transformation logic
    transforms: str = ""  # which FBSL component(s)
    operation: str = ""  # add, remove, modify, merge
    
    # Expected impact
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    # {functional_score: 0.1, layout_score: 0.05}
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Detailed evaluation results for a node"""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    
    # Detailed component scores
    functional_adequacy: Dict[str, Any] = field(default_factory=dict)
    behavioral_performance: Dict[str, Any] = field(default_factory=dict)
    structural_feasibility: Dict[str, Any] = field(default_factory=dict)
    layout_efficiency: Dict[str, Any] = field(default_factory=dict)
    sustainability: Dict[str, Any] = field(default_factory=dict)
    
    # Overall assessment
    composite_score: float = 0.0
    rank: int = 0
    
    # Qualitative assessment
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamp
    evaluated_at: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)