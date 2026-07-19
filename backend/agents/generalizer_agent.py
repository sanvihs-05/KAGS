"""
Generalizer Agent: Breaks down design problem into sub-problems
and explores design alternatives
"""

import logging
import copy
from typing import List, Dict, Any
import uuid

from ..core.fbsl_models import FBSLLayoutNode, Function, NodeType, TransformationType

logger = logging.getLogger(__name__)

class GeneralizerAgent:
    """
    Generalizer Agent: Decomposes design problems and generates alternatives
    
    GUARANTEES:
    - All variant nodes have proper layout deep-copies
    - Room counts are preserved across variants
    """
    
    def __init__(self):
        logger.info("✓ Generalizer Agent initialized")
    
    def decompose_problem(self, problem_node: FBSLLayoutNode, 
                         max_alternatives: int = 4) -> List[FBSLLayoutNode]:
        """
        Decompose problem into alternative design approaches
        
        Args:
            problem_node: Initial problem node
            max_alternatives: Maximum number of alternatives to generate
        
        Returns:
            List of alternative design nodes (each with complete layout)
        """
        logger.info(f"Decomposing problem node: {problem_node.node_id[:8]}...")
        logger.debug(f"  Input node has {len(problem_node.layout.rooms) if problem_node.layout else 0} rooms")
        
        alternatives = []
        
        # Strategy 1: Functional decomposition (split by room zones)
        if len(alternatives) < max_alternatives:
            zone_variants = self._decompose_by_zones(problem_node)
            alternatives.extend(zone_variants[:max_alternatives - len(alternatives)])
        
        # Strategy 2: Layout topology variations
        if len(alternatives) < max_alternatives:
            topology_variants = self._create_topology_variants(problem_node)
            alternatives.extend(topology_variants[:max_alternatives - len(alternatives)])
        
        # Strategy 3: Priority-based variations
        if len(alternatives) < max_alternatives:
            priority_variants = self._create_priority_variants(problem_node)
            alternatives.extend(priority_variants[:max_alternatives - len(alternatives)])
        
        # ✅ Verify all variants have layouts with rooms
        for i, alt in enumerate(alternatives):
            room_count = len(alt.layout.rooms) if alt.layout and alt.layout.rooms else 0
            logger.debug(f"  Variant {i+1}: {room_count} rooms")
            if room_count == 0:
                logger.warning(f"⚠️ Variant {i+1} has no rooms - this should not happen!")
        
        logger.info(f"✓ Generated {len(alternatives)} alternative nodes (all with layouts)")
        return alternatives
    
    def _decompose_by_zones(self, problem_node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Decompose by functional zones (private, social, service)"""
        
        # Categorize functions by zone
        zones = {
            'private': [],  # bedrooms, studies
            'social': [],   # living room, dining
            'service': [],  # kitchen, bathroom, utility
            'circulation': [] # hallways, entries
        }
        
        for func_id, func in problem_node.functions.items():
            activities = []
            if func.activities:
                if isinstance(func.activities, list):
                    activities = [str(a).lower() for a in func.activities]
                elif isinstance(func.activities, dict):
                    activities = [str(v).lower() for v in func.activities.values()]
                else:
                    activities = [str(func.activities).lower()]
            
            if any(act in ['sleeping', 'resting', 'privacy', 'work', 'study', 'bedroom', 'bed'] for act in activities):
                zones['private'].append((func_id, func))
            elif any(act in ['socializing', 'entertainment', 'relaxation', 'dining', 'living'] for act in activities):
                zones['social'].append((func_id, func))
            elif any(act in ['cooking', 'bathing', 'hygiene', 'cleaning', 'laundry', 'storage', 'kitchen', 'bathroom', 'bath'] for act in activities):
                zones['service'].append((func_id, func))
            elif any(act in ['circulation', 'entry', 'transition', 'hallway'] for act in activities):
                zones['circulation'].append((func_id, func))
            else:
                zones['social'].append((func_id, func))  # Default to social
        
        # Create variant: compact layout (zones clustered)
        compact_variant = self._create_variant(problem_node, 'compact_zonal', zones, 
                                               description='Compact clustered zones')
        
        # Create variant: linear layout (zones in sequence)
        linear_variant = self._create_variant(problem_node, 'linear_zonal', zones,
                                              description='Linear sequential zones')
        
        return [compact_variant, linear_variant]
    
    def _create_topology_variants(self, problem_node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Create variants with different layout topologies"""
        
        # Variant 1: Central circulation
        central_variant = self._create_deep_copy_variant(
            problem_node,
            'central_circulation',
            'Central hall with rooms around perimeter'
        )
        
        # Variant 2: Linear circulation
        linear_variant = self._create_deep_copy_variant(
            problem_node,
            'linear_circulation',
            'Linear corridor with rooms on sides'
        )
        
        return [central_variant, linear_variant]
    
    def _create_priority_variants(self, problem_node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Create variants emphasizing different priorities"""
        
        # Variant 1: Maximize natural light
        light_variant = self._create_deep_copy_variant(
            problem_node,
            'natural_light',
            'Maximize natural light access to all rooms'
        )
        
        # Variant 2: Maximize privacy
        privacy_variant = self._create_deep_copy_variant(
            problem_node,
            'privacy',
            'Maximize separation between private and public zones'
        )
        
        return [light_variant, privacy_variant]
    
    def _create_deep_copy_variant(self, 
                                  problem_node: FBSLLayoutNode,
                                  variant_type: str,
                                  description: str) -> FBSLLayoutNode:
        """
        Create a variant with PROPER deep copying of all components
        
        ✅ CRITICAL FIX: Deep copy layout including all rooms
        """
        variant = problem_node.__class__(
            parent_node_id=problem_node.node_id,
            node_type=NodeType.DESIGN_PROTOTYPE,
            generation_level=problem_node.generation_level + 1
        )
        
        # Deep copy functions
        for func_id, func in problem_node.functions.items():
            variant.functions[func_id] = copy.deepcopy(func)
        
        # Deep copy behaviors
        for behav_id, behav in problem_node.behaviors.items():
            variant.behaviors[behav_id] = copy.deepcopy(behav)
        
        # Deep copy structures
        for struct_id, struct in problem_node.structures.items():
            variant.structures[struct_id] = copy.deepcopy(struct)
        
        # ✅ CRITICAL: Deep copy layout with all rooms
        if problem_node.layout:
            variant.layout = copy.deepcopy(problem_node.layout)
            logger.debug(f"  → Copied layout with {len(variant.layout.rooms)} rooms to variant '{variant_type}'")
        else:
            logger.warning(f"⚠️ Problem node has no layout to copy for variant '{variant_type}'")
            # Create minimal layout as fallback
            from ..core.fbsl_models import Layout, Room
            variant.layout = Layout()
            variant.layout.configuration_name = f"Fallback Layout - {variant_type}"
            
            # Create at least one room
            room = Room(
                name="Living",
                room_type="living",
                room_number="1",
                area=20.0,
                height=3.0
            )
            room.calculate_volume()
            variant.layout.rooms[room.room_id] = room
            variant.layout.total_area = 20.0
            variant.layout.used_area = 20.0
            variant.layout.calculate_metrics()
        
        # Copy metadata
        variant.metadata = copy.deepcopy(problem_node.metadata)

        # Add variant-specific metadata
        variant.metadata['variant_type'] = variant_type
        variant.metadata['description'] = description

        self._apply_variant_physics(variant, variant_type)
        return variant

    # Real per-variant transformations: parameters the layout/physics actually
    # read, so each variant earns a different score instead of copying one.
    #   layout_aspect  → treemap footprint (compactness / circulation → S_l)
    #   window_scale   → glazing fraction (daylight → S_b)
    #   partition_thk  → partition thickness (acoustic STC → S_b, when scored)
    _VARIANT_PHYSICS = {
        'compact_zonal':       dict(layout_aspect=1.05, window_scale=1.00, partition_thk=None),
        'linear_zonal':        dict(layout_aspect=2.20, window_scale=1.00, partition_thk=None),
        'central_circulation': dict(layout_aspect=1.00, window_scale=1.10, partition_thk=None),
        'linear_circulation':  dict(layout_aspect=1.80, window_scale=0.95, partition_thk=None),
        'natural_light':       dict(layout_aspect=1.35, window_scale=1.35, partition_thk=0.08),
        'privacy':             dict(layout_aspect=1.15, window_scale=0.85, partition_thk=0.18),
    }

    def _apply_variant_physics(self, variant: FBSLLayoutNode, variant_type: str) -> None:
        """Apply the variant's real parameter changes (no-op for unknown types)."""
        cfg = self._VARIANT_PHYSICS.get(variant_type)
        if not cfg:
            return

        variant.metadata['layout_aspect'] = cfg['layout_aspect']

        for struct in variant.structures.values():
            dims = struct.dimensions or {}
            if cfg['window_scale'] != 1.0 and 'window_ratio' in dims:
                dims['window_ratio'] = round(
                    min(0.40, max(0.05, float(dims['window_ratio']) * cfg['window_scale'])), 3)
                struct.dimensions = dims
            if cfg['partition_thk'] is not None and 'partition' in (struct.name or '').lower():
                dims['thickness'] = cfg['partition_thk']
                struct.dimensions = dims

    def _create_variant(self, problem_node: FBSLLayoutNode,
                       variant_type: str, 
                       zones: Dict,
                       description: str = None) -> FBSLLayoutNode:
        """
        Create a variant node with zone information
        
        ✅ CRITICAL FIX: Deep copy layout properly
        """
        variant = problem_node.__class__(
            parent_node_id=problem_node.node_id,
            node_type=NodeType.DESIGN_PROTOTYPE,
            generation_level=problem_node.generation_level + 1
        )
        
        # Deep copy all components
        for func_id, func in problem_node.functions.items():
            variant.functions[func_id] = copy.deepcopy(func)
        
        for behav_id, behav in problem_node.behaviors.items():
            variant.behaviors[behav_id] = copy.deepcopy(behav)
        
        for struct_id, struct in problem_node.structures.items():
            variant.structures[struct_id] = copy.deepcopy(struct)
        
        # ✅ CRITICAL: Deep copy layout
        if problem_node.layout:
            variant.layout = copy.deepcopy(problem_node.layout)
            logger.debug(f"  → Copied layout with {len(variant.layout.rooms)} rooms to zone variant")
        else:
            logger.warning(f"⚠️ Problem node has no layout to copy for zone variant")
            from ..core.fbsl_models import Layout, Room
            variant.layout = Layout()
            variant.layout.configuration_name = f"Fallback Layout - {variant_type}"
            room = Room(name="Living", room_type="living", room_number="1", area=20.0, height=3.0)
            room.calculate_volume()
            variant.layout.rooms[room.room_id] = room
            variant.layout.total_area = 20.0
            variant.layout.used_area = 20.0
            variant.layout.calculate_metrics()
        
        # Copy metadata
        variant.metadata = copy.deepcopy(problem_node.metadata)
        
        # Add variant-specific metadata
        variant.metadata['variant_type'] = variant_type
        variant.metadata['zones'] = {k: [fid for fid, _ in v] for k, v in zones.items()}
        if description:
            variant.metadata['description'] = description

        self._apply_variant_physics(variant, variant_type)
        return variant


def test_generalizer_agent():
    """Test the generalizer agent"""
    from agents.encoder_agent import EncoderAgent
    from ..database.vector_store import VectorStoreManager
    
    print("🔀 Testing Generalizer Agent")
    print("=" * 60)
    
    # Create a problem node first
    vs = VectorStoreManager()
    encoder = EncoderAgent(vs)
    problem_node = encoder.encode_requirements(
        "2 bedroom apartment with kitchen, bathroom, and living room"
    )
    
    print(f"\n📋 Problem node created:")
    print(f"  • Functions: {len(problem_node.functions)}")
    print(f"  • Behaviors: {len(problem_node.behaviors)}")
    print(f"  • Rooms: {len(problem_node.layout.rooms) if problem_node.layout else 0}")
    
    # Initialize generalizer
    generalizer = GeneralizerAgent()
    
    # Decompose
    alternatives = generalizer.decompose_problem(problem_node, max_alternatives=4)
    
    print(f"\n✓ Generated {len(alternatives)} alternatives:")
    for i, alt in enumerate(alternatives, 1):
        room_count = len(alt.layout.rooms) if alt.layout and alt.layout.rooms else 0
        print(f"\n{i}. Alternative Node:")
        print(f"   • Node ID: {alt.node_id[:8]}...")
        print(f"   • Type: {alt.metadata.get('variant_type', 'N/A')}")
        print(f"   • Description: {alt.metadata.get('description', 'N/A')}")
        print(f"   • Functions: {len(alt.functions)}")
        print(f"   • Behaviors: {len(alt.behaviors)}")
        print(f"   • Rooms: {room_count}")
        
        if room_count == 0:
            print(f"   ⚠️ WARNING: This variant has no rooms!")
    
    print("\n✅ Generalizer Agent testing complete!")


if __name__ == "__main__":
    test_generalizer_agent()