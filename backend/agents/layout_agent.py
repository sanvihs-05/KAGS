# agents/layout_agent.py
"""
Layout Generation Agent - Enhanced with Phase 2 Spatial Algorithms
Integrates A* pathfinding, weighted adjacency, and force-directed optimization

✅ COMPLETE FIX: Robust handling of nodes with/without layouts
"""

import numpy as np
import logging
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point, box, LineString
from shapely import affinity, ops
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

from ..core.fbsl_models import (
    FBSLLayoutNode, Layout, Room, Function, Behavior, Structure,
    FunctionCategory, BehaviorCategory, StructureType
)
from ..core.spatial_algorithms import (
    AStarPathfinder,
    WeightedAdjacencyCalculator,
    ForceDirectedLayout,
    CirculationPath,
    calculate_circulation_efficiency,
    calculate_compactness_score
)
from ..utils.visualization import SVGFloorPlanGenerator, AdjacencyGraphVisualizer
from ..visualization.improved_layout_visualizer import ImprovedLayoutVisualizer
from ..visualization.enhanced_layout import EnhancedLayoutVisualizer

logger = logging.getLogger(__name__)


class LayoutGenerationAgent:
    """
    Generates spatial layouts from FBSL nodes using improved algorithms
    
    Key improvements:
    - Smart room placement with compact layouts
    - Strict adjacency detection
    - Clear, understandable visualizations
    - Proper connectivity analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize layout generation agent with improved visualizer"""
        self.config = config or {}
        
        # Algorithm parameters
        self.force_iterations = self.config.get('force_iterations', 100)
        self.cooling_rate = self.config.get('cooling_rate', 0.95)
        self.initial_temp = self.config.get('initial_temp', 100.0)
        self.min_spacing = self.config.get('min_spacing', 1.0)
        
        # Room dimension constraints
        self.min_room_dimension = self.config.get('min_room_dimension', 2.0)
        self.max_room_dimension = self.config.get('max_room_dimension', 15.0)
        self.min_corridor_width = self.config.get('min_corridor_width', 1.2)
        
        # Spatial algorithm components
        self.pathfinder = AStarPathfinder(grid_resolution=0.5)
        self.adjacency_calculator = WeightedAdjacencyCalculator(
            alpha=0.4,   # Functional dependency weight
            beta=0.35,   # Traffic flow weight
            gamma=0.25   # Privacy requirement weight
        )
        self.force_optimizer = ForceDirectedLayout(
            k_attraction=0.1,
            k_repulsion=100.0,
            learning_rate=0.01,
            max_iterations=200
        )
        
        # Initialize visualizers
        self.svg_generator = SVGFloorPlanGenerator()
        self.graph_visualizer = AdjacencyGraphVisualizer()
        self.enhanced_visualizer = EnhancedLayoutVisualizer()
        
        logger.info("✓ Layout Generation Agent initialized with spatial algorithms")
    
    async def generate_layout(
        self,
        node: FBSLLayoutNode,
        site_boundary: Optional[Polygon] = None
    ) -> Layout:
        """
        Main layout generation pipeline
        
        ✅ CRITICAL FIX: Robust handling of nodes without layouts
        
        Args:
            node: FBSL node with functions, behaviors, structures
            site_boundary: Optional site boundary polygon
        
        Returns:
            Complete Layout object with room positions and circulation
        """
        logger.info(f"Generating layout for node: {node.node_id[:8]}...")

        # ✅ IMPROVED: Check if node has layout with rooms
        # Only synthesize if truly missing (not just empty dict)
        has_rooms = False
        try:
            if node.layout and getattr(node.layout, 'rooms', None):
                if isinstance(node.layout.rooms, dict):
                    has_rooms = len(node.layout.rooms) > 0
                elif hasattr(node.layout.rooms, '__len__'):
                    has_rooms = len(node.layout.rooms) > 0
        except Exception:
            has_rooms = False

        if not has_rooms:
            # Only log warning if this is unexpected (not during GoT expansion where it's normal)
            is_got_expansion = node.metadata.get('transformation_type') is not None
            if not is_got_expansion:
                logger.debug("Node has no rooms defined — synthesizing layout from functions")
            
            synthesized = self._build_layout_from_functions(node)
            
            if synthesized and getattr(synthesized, 'rooms', None):
                room_count = len(synthesized.rooms) if isinstance(synthesized.rooms, dict) else (len(synthesized.rooms) if hasattr(synthesized.rooms, '__len__') else 0)
                if room_count > 0:
                    if not is_got_expansion:
                        logger.info(f"  ✓ Synthesized layout with {room_count} rooms from functions")
                    node.layout = synthesized
                else:
                    if not is_got_expansion:
                        logger.warning("  → Synthesis created no rooms, using fallback")
                    node.layout = self._create_absolute_fallback_layout(node)
            else:
                if not is_got_expansion:
                    logger.warning("  → Synthesis failed, creating minimal fallback layout")
                node.layout = self._create_absolute_fallback_layout(node)

        # Final verification - should rarely trigger now due to fixes in _create_child_node
        try:
            final_has_rooms = False
            if node.layout and getattr(node.layout, 'rooms', None):
                if isinstance(node.layout.rooms, dict):
                    final_has_rooms = len(node.layout.rooms) > 0
                elif hasattr(node.layout.rooms, '__len__'):
                    final_has_rooms = len(node.layout.rooms) > 0
            
            if not final_has_rooms:
                logger.error("❌ CRITICAL: Still no rooms after all attempts - creating emergency fallback")
                node.layout = self._create_absolute_fallback_layout(node)
        except Exception as e:
            logger.error(f"❌ Error verifying rooms: {e} - creating emergency fallback")
            node.layout = self._create_absolute_fallback_layout(node)

        logger.info(f"  → Starting layout generation with {len(node.layout.rooms)} rooms")

        # Step 1: Calculate room areas from functions
        room_specs = self._calculate_room_specifications(node)
        logger.info(f"  → Room specifications calculated: {len(room_specs)} rooms")

        # ✅ Populate Room.required_adjacencies from the brief BEFORE building
        # the weighted matrix below — _build_adjacency_matrix reads exactly
        # this field (via extract_adjacency_from_fbsl). Doing this after
        # placement (as before) left every room's required_adjacencies empty
        # at matrix-build time, so the matrix was always all-zero, the
        # `weight > 0.3` gate in circulation never fired, and NO circulation
        # paths were ever attempted — independent of the A* obstacle bug.
        required_pairs = self._brief_required_pairs(node)
        if required_pairs:
            type_rooms: Dict[str, List[str]] = {}
            for rid, room in node.layout.rooms.items():
                type_rooms.setdefault((room.room_type or '').lower(), []).append(rid)
            for t1, t2, kind in required_pairs:
                if kind != 'required':
                    continue
                for r1 in type_rooms.get(t1, []):
                    for r2 in type_rooms.get(t2, []):
                        room1 = node.layout.rooms[r1]
                        if r2 not in room1.required_adjacencies:
                            room1.required_adjacencies.append(r2)

        # Step 2: Build weighted adjacency matrix
        adjacency_matrix = self._build_adjacency_matrix(node.layout.rooms)
        logger.info(f"  → Adjacency matrix built: shape={adjacency_matrix.shape}, "
                    f"nonzero={int(np.count_nonzero(adjacency_matrix))}")
        
        # Step 3+4: Gap-free placement via zoned squarified treemap.
        # Replaces force-directed + SLSQP, which leave gaps by construction
        # (universal repulsion never lets rooms share a wall) — so their
        # adjacency/compactness metrics could not be trusted. The treemap tiles
        # the footprint exactly and preserves each room's target area.
        # Variant-controlled footprint aspect: variants set metadata['layout_aspect']
        # (compact ~1.05 … linear ~2.4), which changes real geometry and therefore
        # compactness/circulation — so layout variants earn different S_l scores.
        try:
            aspect = float(node.metadata.get('layout_aspect', 1.2) or 1.2)
        except (TypeError, ValueError):
            aspect = 1.2
        aspect = min(max(aspect, 0.4), 4.0)

        optimized_positions = self._squarified_treemap_placement(
            room_specs, node.layout.rooms, aspect=aspect,
            required_pairs=required_pairs
        )
        logger.info(f"  → Treemap placement complete (gap-free tiling, aspect={aspect:.2f})")
        
        # Step 5: Generate room polygons
        room_polygons = self._generate_room_polygons(optimized_positions, room_specs)
        logger.info(f"  → Room polygons generated: {len(room_polygons)} rooms")

        # ✅ Persist L onto the FBSL model itself: every Room carries its
        # coordinates, dimensions, and adjacency lists. Before this, the
        # geometry lived only in the SVG — the stored FBSL had no L.
        for rid, pos in optimized_positions.items():
            room = node.layout.rooms.get(rid)
            if room is None:
                continue
            room.position_vector = {'x': round(pos['x'], 3), 'y': round(pos['y'], 3), 'z': 0.0}
            room.width = round(pos['width'], 3)
            room.length = round(pos['length'], 3)
            room.actual_adjacencies = [
                other for other, opos in optimized_positions.items()
                if other != rid and self._rects_share_wall(pos, opos)
            ]
        
        # Step 6: Generate circulation paths using A* pathfinding
        circulation = self._generate_circulation(
            room_polygons, 
            adjacency_matrix,
            list(node.layout.rooms.keys())
        )
        logger.info(f"  → Circulation paths generated: {len(circulation)} paths")
        
        # Step 7: Create layout object
        layout = self._create_layout_object(
            room_polygons,
            circulation,
            site_boundary,
            adjacency_matrix,
            layout_rooms=node.layout.rooms if node.layout else {}
        )
        
        # Step 8: Calculate layout metrics
        layout.calculate_metrics()

        # ✅ MEASURED adjacency satisfaction against the BRIEF's requirements
        # (calculate_metrics uses the weighted preference matrix, which is a
        # different thing). 1.0 when the brief stated no adjacency needs.
        adj_score, adj_details = self._adjacency_satisfaction(
            optimized_positions, node.layout.rooms, required_pairs
        )
        layout.adjacency_satisfaction_score = adj_score
        layout.metadata['adjacency_requirements'] = adj_details
        layout.metadata['adjacency_measured'] = True
        if required_pairs:
            logger.info(
                f"  → Brief adjacency satisfaction: {adj_score:.2f} "
                f"({sum(1 for d in adj_details if d['satisfied'])}/{len(adj_details)} requirements)"
            )
        logger.info(f"✓ Layout generation complete: score={layout.compactness_score:.3f}")

        # Enhanced visuals (compass, strict adjacencies)
        try:
            project_name = node.metadata.get('project_name') if isinstance(node.metadata, dict) else None
            project_name = project_name or node.node_id
            enhanced_outputs = self.enhanced_visualizer.render(layout, project_name, node.node_id)
            if enhanced_outputs:
                node.metadata.setdefault('visualizations', {})
                node.metadata['visualizations'].update(enhanced_outputs)
                layout.metadata.setdefault('enhanced_visuals', enhanced_outputs)
                logger.info("  → Enhanced layout visuals generated")
        except Exception as exc:
            logger.warning(f"  → Enhanced visualization failed: {exc}")
        
        # Step 9: Generate visualizations
        try:
            # Generate floor plan SVG
            svg_floor_plan = self.svg_generator.generate_floor_plan(
                layout=layout,
                title=f"Floor Plan - {node.node_id[:8]}",
                show_dimensions=True,
                show_circulation=True,
                show_legend=True
            )
            layout.svg_floor_plan = svg_floor_plan
            logger.info("  → Floor plan SVG generated")
            
            # Generate adjacency graph SVG
            room_ids_list = list(node.layout.rooms.keys())
            
            adjacency_graph = self._build_networkx_adjacency_graph(
                room_polygons, 
                adjacency_matrix, 
                room_ids_list
            )
            adjacency_svg = self.graph_visualizer.generate_adjacency_svg(
                adjacency_graph=adjacency_graph,
                title=f"Adjacency Graph - {node.node_id[:8]}",
                show_weights=True,
                layout_type="spring"
            )
            layout.adjacency_svg = adjacency_svg
            logger.info("  → Adjacency graph SVG generated")
            
        except Exception as e:
            logger.warning(f"  → Visualization generation failed: {e}")
            layout.svg_floor_plan = None
            layout.adjacency_svg = None
        
        return layout

    def _build_layout_from_functions(self, node: FBSLLayoutNode) -> Optional[Layout]:
        """
        Synthesize a Layout from node.functions when no explicit layout exists.
        
        ✅ IMPROVED: Better error handling and logging
        
        This creates a Room for each spatial Function, using preferred areas
        from `spatial_requirements` where available.
        """
        from ..core.fbsl_models import Layout, Room

        if not node.functions:
            logger.debug("  No functions available to synthesize layout")
            return None

        layout = Layout()
        layout.configuration_name = "Synthesized Layout"

        default_area = 12.0
        created = 0

        for func_id, func in node.functions.items():
            try:
                # Only create spatial functions as rooms
                if hasattr(func, 'category') and func.category is not None:
                    # Handle both enum and string categories
                    cat_value = func.category.value if hasattr(func.category, 'value') else str(func.category)
                    if cat_value.lower() not in ['spatial', 'social', 'environmental']:
                        continue
                else:
                    # If no category, include it anyway
                    pass

                # Determine room type from function name or activities
                rt = None
                if func.activities:
                    if isinstance(func.activities, list) and len(func.activities) > 0:
                        rt = str(func.activities[0])
                    elif isinstance(func.activities, dict):
                        rt = next(iter(func.activities.values()), None)
                    else:
                        rt = str(func.activities)

                if not rt and hasattr(func, 'name') and func.name:
                    # extract part after 'provide_' if present
                    rt = func.name.replace('provide_', '').replace('_space', '').replace('_facilities', '')

                if not rt:
                    rt = 'room'

                rt_key = str(rt).replace(' ', '_').lower()

                # Area from spatial_requirements if provided
                preferred = None
                if hasattr(func, 'spatial_requirements') and func.spatial_requirements:
                    if isinstance(func.spatial_requirements, dict):
                        preferred = func.spatial_requirements.get('preferred_area') or func.spatial_requirements.get('area')

                area = float(preferred) if preferred else default_area

                # Height from spatial_requirements
                height = 3.0
                if hasattr(func, 'spatial_requirements') and isinstance(func.spatial_requirements, dict):
                    height = func.spatial_requirements.get('height', 3.0)

                room = Room(
                    name=rt_key.replace('_', ' ').title(),
                    room_type=rt_key,
                    room_number=str(created + 1),
                    function_id=func.function_id,
                    area=area,
                    height=height
                )
                room.calculate_volume()
                layout.rooms[room.room_id] = room
                created += 1
                logger.debug(f"    → Created room '{room.name}' ({area:.1f} m²) from function '{func.name}'")
                
            except Exception as e:
                logger.warning(f"  Failed to create room for function {func.name if hasattr(func, 'name') else func_id}: {e}")
                continue

        if created == 0:
            logger.warning("  Synthesis created 0 rooms from functions")
            return None

        layout.total_area = sum(r.area for r in layout.rooms.values())
        layout.used_area = layout.total_area
        layout.calculate_metrics()

        logger.info(f"  ✓ Synthesized layout: {created} rooms, {layout.total_area:.1f} m² total")
        return layout

    def _create_absolute_fallback_layout(self, node: FBSLLayoutNode) -> Layout:
        """
        Create absolute fallback layout when all else fails
        
        ✅ NEW: Emergency fallback that always succeeds
        """
        from ..core.fbsl_models import Layout, Room
        
        logger.info("  Creating absolute fallback layout...")
        
        fallback_layout = Layout()
        fallback_layout.configuration_name = "Emergency Fallback Layout"
        
        # Try to use first function if available
        if node.functions:
            func = list(node.functions.values())[0]
            room_name = func.name.replace('provide_', '').replace('_', ' ').title() if hasattr(func, 'name') else "Space"
            room_type = func.name.replace('provide_', '') if hasattr(func, 'name') else "space"
            
            # Get area from spatial requirements
            area = 12.0
            if hasattr(func, 'spatial_requirements') and isinstance(func.spatial_requirements, dict):
                area = func.spatial_requirements.get('preferred_area', 12.0)
            
            room = Room(
                name=room_name,
                room_type=room_type,
                room_number="1",
                function_id=func.function_id,
                area=area,
                height=3.0
            )
            logger.debug(f"    → Created fallback room from function: {room_name} ({area} m²)")
        else:
            # Absolute default
            room = Room(
                name="Living Space",
                room_type="living",
                room_number="1",
                area=20.0,
                height=3.0
            )
            logger.debug("    → Created default living space (no functions available)")
        
        room.calculate_volume()
        fallback_layout.rooms[room.room_id] = room
        fallback_layout.total_area = room.area
        fallback_layout.used_area = room.area
        fallback_layout.calculate_metrics()
        
        logger.info(f"  ✓ Emergency fallback created: 1 room ({room.area} m²)")
        return fallback_layout
    
    def _calculate_room_specifications(
        self, 
        node: FBSLLayoutNode
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate room dimensions from functions
        
        Args:
            node: FBSL node with room definitions
        
        Returns:
            Room specifications with area, dimensions, constraints
        """
        specs = {}
        
        if not node.layout or not node.layout.rooms:
            logger.warning("No layout or rooms to calculate specifications from")
            return specs
        
        for room_id, room in node.layout.rooms.items():
            # Get area from function requirements
            area = room.area if room.area > 0 else 16.0  # Default 16 sqm
            
            # Calculate dimensions (approximate square for initial placement)
            side = np.sqrt(area)
            
            specs[room_id] = {
                'area': area,
                'width': side,
                'length': side,
                'height': room.height if room.height > 0 else 3.0,
                'min_width': max(self.min_room_dimension, side * 0.7),
                'max_width': min(self.max_room_dimension, side * 1.3),
                'min_length': max(self.min_room_dimension, side * 0.7),
                'max_length': min(self.max_room_dimension, side * 1.3)
            }
        
        return specs
    
    def _build_adjacency_matrix(
        self, 
        rooms: Dict[str, Room]
    ) -> np.ndarray:
        """
        Build weighted adjacency matrix using Phase 2 algorithm
        
        Formula: w(i,j) = α×Functional_Dependency + β×Traffic_Flow + γ×Privacy_Requirement
        """
        rooms_dict = {room_id: room for room_id, room in rooms.items()}
        
        func_deps, traffic_flows, privacy_reqs = self.adjacency_calculator.extract_adjacency_from_fbsl(
            rooms_dict
        )
        
        adjacency_matrix = self.adjacency_calculator.calculate_adjacency_matrix(
            rooms_dict,
            functional_dependencies=func_deps,
            traffic_flows=traffic_flows,
            privacy_requirements=privacy_reqs
        )
        
        return adjacency_matrix
    
    def _force_directed_placement(
        self,
        room_specs: Dict[str, Dict],
        adjacency_matrix: np.ndarray,
        iterations: int = 200
    ) -> Dict[str, Dict[str, float]]:
        """Use force-directed algorithm for initial placement"""
        
        positions = self.force_optimizer.optimize_layout(
            room_specs=room_specs,
            adjacency_matrix=adjacency_matrix,
            initial_positions=None,
            bounds=(0, 0, 100, 100)
        )

        return positions

    # Room-type → zone lookup for treemap grouping (substring match).
    _ZONE_KEYWORDS = {
        'social': ['living', 'dining', 'kitchen', 'family', 'lounge', 'great'],
        'private': ['bedroom', 'bed', 'bath', 'ensuite', 'wc', 'toilet',
                    'study', 'office'],
        # everything else (laundry, mudroom, garage, storage, utility, hall) → service
    }

    def _zone_of(self, room_type: str) -> str:
        rt = (room_type or '').lower()
        for zone, keywords in self._ZONE_KEYWORDS.items():
            if any(k in rt for k in keywords):
                return zone
        return 'service'

    @staticmethod
    def _brief_required_pairs(node) -> List[tuple]:
        """Type-level adjacency requirements from the brief:
        [(type1, type2, 'required'|'avoid'), ...]"""
        pairs = []
        for a in ((getattr(node, 'metadata', None) or {}).get('required_adjacencies') or []):
            if not isinstance(a, dict):
                continue
            t1 = str(a.get('room1', '')).strip().lower()
            t2 = str(a.get('room2', '')).strip().lower()
            if t1 and t2 and t1 != t2:
                kind = 'avoid' if str(a.get('type', '')).lower() == 'avoid' else 'required'
                pairs.append((t1, t2, kind))
        return pairs

    @staticmethod
    def _rects_share_wall(p: Dict[str, float], q: Dict[str, float],
                          min_overlap: float = 0.3, eps: float = 1e-3) -> bool:
        """True if two placed tiles share a wall segment of >= min_overlap m."""
        px2, py2 = p['x'] + p['width'], p['y'] + p['length']
        qx2, qy2 = q['x'] + q['width'], q['y'] + q['length']
        if abs(px2 - q['x']) < eps or abs(qx2 - p['x']) < eps:
            if min(py2, qy2) - max(p['y'], q['y']) >= min_overlap:
                return True
        if abs(py2 - q['y']) < eps or abs(qy2 - p['y']) < eps:
            if min(px2, qx2) - max(p['x'], q['x']) >= min_overlap:
                return True
        return False

    def _adjacency_satisfaction(
        self,
        positions: Dict[str, Dict[str, float]],
        rooms: Dict[str, Room],
        pairs: List[tuple],
    ) -> tuple:
        """Measure brief adjacency satisfaction on placed tiles.

        A required (t1, t2) is satisfied when ANY room of t1 shares a wall
        with ANY room of t2; an avoid pair is satisfied when NO instances
        touch. Returns (score, details): score = satisfied/total, 1.0 when
        the brief stated no adjacency requirements (nothing to violate).
        """
        if not pairs:
            return 1.0, []
        by_type: Dict[str, List[str]] = {}
        for rid in positions:
            rtype = (rooms[rid].room_type if rid in rooms else '').lower()
            by_type.setdefault(rtype, []).append(rid)

        details = []
        satisfied = 0
        for t1, t2, kind in pairs:
            touching = any(
                self._rects_share_wall(positions[r1], positions[r2])
                for r1 in by_type.get(t1, [])
                for r2 in by_type.get(t2, [])
            )
            ok = touching if kind == 'required' else not touching
            satisfied += 1 if ok else 0
            details.append({'room1': t1, 'room2': t2, 'type': kind, 'satisfied': ok})
        return satisfied / len(pairs), details

    @staticmethod
    def _pair_aware_order(items: List[tuple], rooms: Dict[str, Room],
                          pairs: List[tuple]) -> List[tuple]:
        """Order [(rid, area)] area-desc but pull required partners adjacent,
        so the squarify shelf tiles them against each other."""
        type_of = {rid: (rooms[rid].room_type.lower() if rid in rooms else '')
                   for rid, _ in items}
        partners: Dict[str, set] = {}
        for t1, t2, kind in pairs:
            if kind == 'required':
                partners.setdefault(t1, set()).add(t2)
                partners.setdefault(t2, set()).add(t1)

        area_desc = sorted(items, key=lambda t: -t[1])
        ordered, used = [], set()
        for rid, a in area_desc:
            if rid in used:
                continue
            ordered.append((rid, a))
            used.add(rid)
            for want in partners.get(type_of.get(rid, ''), ()):  # pull partner next
                for rid2, a2 in area_desc:
                    if rid2 not in used and type_of.get(rid2) == want:
                        ordered.append((rid2, a2))
                        used.add(rid2)
                        break
        return ordered

    def _squarified_treemap_placement(
        self,
        room_specs: Dict[str, Dict],
        rooms: Dict[str, Room],
        aspect: float = 1.2,
        circulation_frac: float = 0.0,
        required_pairs: Optional[List[tuple]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Gap-free rectangular dissection of the footprint.

        Groups rooms into service | social | private zones, lays the zones out
        as left-to-right columns sized by area, and squarifies rooms within
        each zone. Returns {room_id: {x, y, width, length}} that tiles the
        footprint exactly, so every room's tile area == its target area and
        adjacency is real (rooms share walls). No gaps, no overlaps.

        circulation_frac defaults to 0.0 so each tile == its target area (keeps
        the drawn plan consistent with the areas the brief validator checks);
        set it > 0 to reserve corridor space by inflating the footprint.
        """
        room_ids = list(room_specs.keys())
        if not room_ids:
            return {}

        total_area = sum(room_specs[r]['area'] for r in room_ids)
        footprint = total_area / max(1e-9, (1.0 - circulation_frac))
        W = float(np.sqrt(footprint * max(aspect, 1e-6)))
        H = footprint / max(W, 1e-9)

        zones: Dict[str, List[str]] = {}
        for rid in room_ids:
            rtype = rooms[rid].room_type if rid in rooms else ''
            zones.setdefault(self._zone_of(rtype), []).append(rid)

        order = [z for z in ('service', 'social', 'private') if z in zones]

        def _tile(pair_aware: bool) -> Dict[str, Dict[str, float]]:
            positions: Dict[str, Dict[str, float]] = {}
            x_cursor = 0.0
            for z in order:
                zone_area = sum(room_specs[r]['area'] for r in zones[z]) / max(1e-9, (1.0 - circulation_frac))
                zw = W * zone_area / max(footprint, 1e-9)
                items = [(r, room_specs[r]['area']) for r in zones[z]]
                if pair_aware and required_pairs:
                    items = self._pair_aware_order(items, rooms, required_pairs)
                placed = self._squarify_rect(items, x_cursor, 0.0, zw, H,
                                             keep_order=pair_aware)
                for rid, (rx, ry, rw, rh) in placed.items():
                    positions[rid] = {'x': rx, 'y': ry, 'width': rw, 'length': rh}
                x_cursor += zw
            return positions

        default = _tile(pair_aware=False)
        if not required_pairs:
            return default

        # Adjacency-aware attempt: same zones, same areas, but required
        # partners tile consecutively. Keep whichever placement satisfies
        # more of the brief's adjacency requirements.
        paired = _tile(pair_aware=True)
        s_default, _ = self._adjacency_satisfaction(default, rooms, required_pairs)
        s_paired, _ = self._adjacency_satisfaction(paired, rooms, required_pairs)
        logger.info(
            f"  → Adjacency-aware tiling: default={s_default:.2f} "
            f"paired={s_paired:.2f} → using {'paired' if s_paired > s_default else 'default'}"
        )
        return paired if s_paired > s_default else default

    @staticmethod
    def _squarify_rect(items, x, y, w, h, keep_order: bool = False) -> Dict[str, tuple]:
        """
        Squarified treemap: tile rect (x, y, w, h) with items [(id, area)],
        preferring near-square tiles. Returns {id: (x, y, w, h)}.
        keep_order=True preserves the caller's item order (used for
        adjacency-aware placement where partners must tile consecutively).
        """
        if not keep_order:
            items = sorted(items, key=lambda t: -t[1])
        total = sum(a for _, a in items) or 1.0
        scale = (w * h) / total
        items = [(n, a * scale) for n, a in items]  # areas now sum to w*h
        out: Dict[str, tuple] = {}
        rx, ry, rw, rh = x, y, w, h

        def worst(row, length):
            s = max(sum(a for _, a in row), 1e-9)
            mx = max(a for _, a in row)
            mn = min(a for _, a in row)
            length = max(length, 1e-9)
            return max((length ** 2 * mx) / (s ** 2), (s ** 2) / (length ** 2 * mn))

        def layout_row(row, rx, ry, rw, rh):
            s = sum(a for _, a in row)
            if rw >= rh:  # horizontal shelf along the top edge
                rowh = s / max(rw, 1e-9)
                cx = rx
                for n, a in row:
                    cw = a / max(rowh, 1e-9)
                    out[n] = (cx, ry, cw, rowh)
                    cx += cw
                return rx, ry + rowh, rw, rh - rowh
            else:         # vertical shelf along the left edge
                roww = s / max(rh, 1e-9)
                cy = ry
                for n, a in row:
                    ch = a / max(roww, 1e-9)
                    out[n] = (rx, cy, roww, ch)
                    cy += ch
                return rx + roww, ry, rw - roww, rh

        row: list = []
        i = 0
        while i < len(items):
            length = rw if rw >= rh else rh
            if not row:
                row = [items[i]]
                i += 1
                continue
            if worst(row, length) >= worst(row + [items[i]], length):
                row.append(items[i])
                i += 1
            else:
                rx, ry, rw, rh = layout_row(row, rx, ry, rw, rh)
                row = []
        if row:
            layout_row(row, rx, ry, rw, rh)

        return out
    
    def _optimize_layout(
        self,
        initial_positions: Dict[str, Dict],
        room_specs: Dict[str, Dict],
        adjacency_matrix: np.ndarray,
        site_boundary: Optional[Polygon]
    ) -> Dict[str, Dict[str, float]]:
        """Optimize layout using scipy minimize with constraints"""
        
        room_ids = list(room_specs.keys())
        n = len(room_ids)

        if n == 0:
            logger.warning("_optimize_layout called with no room specs — returning initial positions")
            return initial_positions
        
        # Convert to flat array [x1, y1, w1, l1, x2, y2, w2, l2, ...]
        x0 = []
        for room_id in room_ids:
            pos = initial_positions[room_id]
            x0.extend([
                pos['x'],
                pos['y'],
                pos.get('width', room_specs[room_id]['width']),
                pos.get('height', room_specs[room_id]['length'])
            ])
        
        # Objective function
        def objective(x):
            score = 0.0
            
            positions = {}
            for i, room_id in enumerate(room_ids):
                idx = i * 4
                positions[room_id] = {
                    'x': x[idx],
                    'y': x[idx + 1],
                    'width': x[idx + 2],
                    'length': x[idx + 3]
                }
            
            # 1. Adjacency satisfaction
            for i in range(n):
                for j in range(i + 1, n):
                    room1_id = room_ids[i]
                    room2_id = room_ids[j]
                    
                    p1 = positions[room1_id]
                    p2 = positions[room2_id]
                    
                    center_dist = np.sqrt(
                        (p1['x'] + p1['width']/2 - p2['x'] - p2['width']/2)**2 +
                        (p1['y'] + p1['length']/2 - p2['y'] - p2['length']/2)**2
                    )
                    
                    min_separation = (p1['width'] + p2['width']) / 2
                    weight = adjacency_matrix[i, j]
                    
                    if weight > 0:
                        score += weight * max(0, center_dist - min_separation)**2
                    elif weight < 0:
                        score += abs(weight) * max(0, min_separation * 1.5 - center_dist)**2
            
            # 2. Compactness
            xs = [positions[rid]['x'] for rid in room_ids]
            ys = [positions[rid]['y'] for rid in room_ids]
            ws = [positions[rid]['width'] for rid in room_ids]
            ls = [positions[rid]['length'] for rid in room_ids]
            
            max_x = max(x + w for x, w in zip(xs, ws))
            max_y = max(y + l for y, l in zip(ys, ls))
            min_x = min(xs)
            min_y = min(ys)
            
            bbox_area = (max_x - min_x) * (max_y - min_y)
            total_room_area = sum(room_specs[rid]['area'] for rid in room_ids)
            
            score += (bbox_area - total_room_area) * 0.01
            
            # 3. Overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    room1_id = room_ids[i]
                    room2_id = room_ids[j]
                    
                    p1 = positions[room1_id]
                    p2 = positions[room2_id]
                    
                    overlap_x = max(0, min(p1['x'] + p1['width'], p2['x'] + p2['width']) - max(p1['x'], p2['x']))
                    overlap_y = max(0, min(p1['y'] + p1['length'], p2['y'] + p2['length']) - max(p1['y'], p2['y']))
                    
                    if overlap_x > 0 and overlap_y > 0:
                        overlap_area = overlap_x * overlap_y
                        score += overlap_area * 1000.0
            
            return score
        
        # Constraints
        constraints = []
        for i, room_id in enumerate(room_ids):
            idx = i * 4
            target_area = room_specs[room_id]['area']
            constraints.append({
                'type': 'eq',
                'fun': lambda x, i=idx, a=target_area: x[i+2] * x[i+3] - a
            })
        
        # Bounds
        bounds = []
        for room_id in room_ids:
            specs = room_specs[room_id]
            bounds.extend([
                (0, 100),
                (0, 100),
                (specs['min_width'], specs['max_width']),
                (specs['min_length'], specs['max_length'])
            ])
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            optimized = {}
            for i, room_id in enumerate(room_ids):
                idx = i * 4
                optimized[room_id] = {
                    'x': result.x[idx],
                    'y': result.x[idx + 1],
                    'width': result.x[idx + 2],
                    'length': result.x[idx + 3]
                }
            
            logger.info(f"  → Optimization converged: final_score={result.fun:.2f}")
            return optimized
            
        except Exception as e:
            logger.warning(f"  → Optimization failed: {e}, using force-directed positions")
            return initial_positions
    
    def _generate_room_polygons(
        self,
        positions: Dict[str, Dict[str, float]],
        room_specs: Dict[str, Dict]
    ) -> Dict[str, Polygon]:
        """Generate room polygons from positions"""
        
        polygons = {}
        
        for room_id, pos in positions.items():
            x = pos['x']
            y = pos['y']
            width = pos['width']
            length = pos['length']
            
            polygon = box(x, y, x + width, y + length)
            polygons[room_id] = polygon
        
        return polygons
    
    def _generate_circulation(
        self,
        room_polygons: Dict[str, Polygon],
        adjacency_matrix: np.ndarray,
        room_ids: List[str]
    ) -> List[CirculationPath]:
        """Generate circulation paths on the ROOM-CONNECTIVITY GRAPH.

        On a gap-free tiled plan, free-space A* cannot work: every room is an
        obstacle and both endpoints sit inside obstacles, so no path was ever
        found and circulation_efficiency read 0.0 for every design. Movement
        in a real house goes THROUGH rooms via doorways in shared walls — so
        circulation is the shortest path on the graph whose edges connect
        rooms sharing a wall long enough for a door (>= 0.7 m), weighted by
        centroid distance. Adjacent pairs route directly (efficiency 1.0);
        distant pairs pay for every detour, so compact vs linear plans now
        earn genuinely different circulation scores.
        """
        if not room_polygons:
            return []

        DOOR_MIN = 0.7  # minimum shared-wall length for a doorway (m)

        centroids: Dict[str, Tuple[float, float]] = {}
        for rid in room_ids:
            poly = room_polygons.get(rid)
            if poly is not None:
                centroids[rid] = (poly.centroid.x, poly.centroid.y)

        def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return float(np.hypot(b[0] - a[0], b[1] - a[1]))

        G = nx.Graph()
        G.add_nodes_from(centroids)
        ids = [r for r in room_ids if r in centroids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pi, pj = room_polygons[ids[i]], room_polygons[ids[j]]
                try:
                    if pi.distance(pj) < 0.01:
                        shared = pi.intersection(pj)
                        if getattr(shared, 'length', 0.0) >= DOOR_MIN:
                            G.add_edge(ids[i], ids[j],
                                       weight=_dist(centroids[ids[i]], centroids[ids[j]]))
                except Exception:
                    continue

        # ✅ All room pairs, not just weight > 0.3 preference pairs: the
        # weighted preference matrix reflects "should be adjacent" (function/
        # traffic/privacy scores), which is near-universally zero when the
        # brief states few explicit adjacencies — gating on it starved
        # circulation down to 0 paths regardless of the graph fix above.
        # Walkability of the WHOLE plan is what S_l's circulation term means.
        circulation_paths = []
        n = len(room_ids)
        for i in range(n):
            for j in range(i + 1, n):
                room1_id, room2_id = room_ids[i], room_ids[j]
                if room1_id not in centroids or room2_id not in centroids:
                    continue
                try:
                    node_path = nx.shortest_path(G, room1_id, room2_id, weight='weight')
                    pts = [centroids[r] for r in node_path]
                    length = sum(_dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
                    circulation_paths.append(CirculationPath(
                        start_room=room1_id,
                        end_room=room2_id,
                        path_points=pts,
                        length=length,
                        cost=length,
                        path_type='direct' if len(node_path) == 2 else 'corridor',
                    ))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                except Exception as e:
                        logger.debug(f"Circulation path {room1_id[:8]}→{room2_id[:8]} failed: {e}")
                        continue

        logger.info(f"  → Room-graph circulation: {len(circulation_paths)} paths "
                    f"({G.number_of_edges()} door connections)")
        return circulation_paths

    def _generate_circulation_astar_legacy(
        self,
        room_polygons: Dict[str, Polygon],
        adjacency_matrix: np.ndarray,
        room_ids: List[str]
    ) -> List[CirculationPath]:
        """Legacy free-space A* circulation (kept for reference; unusable on
        gap-free tiled plans because endpoints sit inside obstacles)."""

        if not room_polygons:
            return []

        circulation_paths = []
        n = len(room_ids)

        all_bounds = [poly.bounds for poly in room_polygons.values()]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)

        grid_bounds = (min_x - 2, min_y - 2, max_x + 2, max_y + 2)

        obstacles = []
        for room_id, polygon in room_polygons.items():
            bounds = polygon.bounds
            obstacles.append((bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]))

        for i in range(n):
            for j in range(i + 1, n):
                weight = adjacency_matrix[i, j]

                if weight > 0.3:
                    room1_id = room_ids[i]
                    room2_id = room_ids[j]

                    poly1 = room_polygons[room1_id]
                    poly2 = room_polygons[room2_id]

                    start = (poly1.centroid.x, poly1.centroid.y)
                    goal = (poly2.centroid.x, poly2.centroid.y)

                    try:
                        path = self.pathfinder.find_path(
                            start=start,
                            goal=goal,
                            obstacles=obstacles,
                            grid_bounds=grid_bounds
                        )
                        
                        if path:
                            path.start_room = room1_id
                            path.end_room = room2_id
                            circulation_paths.append(path)
                            logger.debug(f"    → Path found: {room1_id} → {room2_id} (length={path.length:.2f}m)")
                    
                    except Exception as e:
                        logger.warning(f"    → Path finding failed for {room1_id} → {room2_id}: {e}")
        
        return circulation_paths
    
    def _create_layout_object(
        self,
        room_polygons: Dict[str, Polygon],
        circulation: List[CirculationPath],
        site_boundary: Optional[Polygon],
        adjacency_matrix: Optional[np.ndarray] = None,
        layout_rooms: Optional[Dict[str, Room]] = None
    ) -> Layout:
        """Create Layout object from generated components"""
        
        # Determine site / bounding area. Prefer provided site_boundary area if available
        if site_boundary is not None:
            site_min_x, site_min_y, site_max_x, site_max_y = site_boundary.bounds
            site_area = site_boundary.area
        else:
            site_min_x = site_min_y = 0.0
            site_max_x = site_max_y = 0.0
            site_area = None

        if not room_polygons:
            all_bounds = [(0.0, 0.0, 10.0, 10.0)]
        else:
            all_bounds = [poly.bounds for poly in room_polygons.values()]

        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)

        bbox_area = (max_x - min_x) * (max_y - min_y)
        used_area = sum(poly.area for poly in room_polygons.values())

        # If a site boundary was provided, scale and translate room polygons to fit the site.
        if site_area is not None and bbox_area > 0:
            site_w = site_max_x - site_min_x
            site_h = site_max_y - site_min_y
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y

            # Compute uniform scale to fit within site while preserving aspect
            scale_x = site_w / bbox_w if bbox_w > 0 else 1.0
            scale_y = site_h / bbox_h if bbox_h > 0 else 1.0
            scale = min(scale_x, scale_y)

            # Apply scale and translate so min aligns with site min
            transformed_polygons = {}
            for rid, poly in room_polygons.items():
                # Translate to origin, scale, then translate to site origin
                p = affinity.translate(poly, xoff=-min_x, yoff=-min_y)
                p = affinity.scale(p, xfact=scale, yfact=scale, origin=(0, 0))
                p = affinity.translate(p, xoff=site_min_x, yoff=site_min_y)
                transformed_polygons[rid] = p

            room_polygons = transformed_polygons
            # Recompute bounds and areas
            all_bounds = [poly.bounds for poly in room_polygons.values()]
            min_x = min(b[0] for b in all_bounds)
            min_y = min(b[1] for b in all_bounds)
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
            bbox_area = (max_x - min_x) * (max_y - min_y)
            used_area = sum(poly.area for poly in room_polygons.values())

        # If site area available use it as total_area, otherwise use bbox area
        total_area = site_area if site_area is not None else bbox_area
        
        circulation_graph = nx.Graph()
        for path in circulation:
            circulation_graph.add_edge(
                path.start_room,
                path.end_room,
                length=path.length,
                cost=path.cost,
                path_type=path.path_type,
                path_points=path.path_points
            )
        
        layout = Layout(
            total_area=total_area,
            used_area=used_area
        )

        # Attach rooms from synthesized layout if available
        if layout_rooms:
            layout.rooms = layout_rooms

        layout.bounds = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y
        }
        
        layout.room_polygons = room_polygons
        layout.circulation_paths = circulation
        layout.circulation_graph = circulation_graph

        # Calculate circulation_area as union of buffered circulation paths
        try:
            buffered_paths = []
            for path in circulation:
                if not path.path_points:
                    continue
                line = LineString(path.path_points)
                buf = line.buffer(max(self.min_corridor_width, 1.0))
                buffered_paths.append(buf)

            if buffered_paths:
                union_buf = ops.unary_union(buffered_paths)
                circulation_area = union_buf.area
            else:
                circulation_area = 0.0
        except Exception:
            circulation_area = 0.0

        layout.circulation_area = circulation_area
        
        room_positions = {
            room_id: {
                'x': poly.centroid.x,
                'y': poly.centroid.y,
                'width': poly.bounds[2] - poly.bounds[0],
                'height': poly.bounds[3] - poly.bounds[1]
            }
            for room_id, poly in room_polygons.items()
        }
        
        # Attach adjacency matrices: required (if provided) and actual
        try:
            room_ids = list(room_polygons.keys())
            n = len(room_ids)
            actual_adj = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    poly_i = room_polygons[room_ids[i]]
                    poly_j = room_polygons[room_ids[j]]
                    # Touching or intersecting -> adjacency
                    if poly_i.touches(poly_j) or poly_i.intersects(poly_j):
                        actual_adj[i, j] = actual_adj[j, i] = 1.0
                    else:
                        # Near adjacency threshold (e.g., within 0.5m)
                        if poly_i.distance(poly_j) < 0.5:
                            actual_adj[i, j] = actual_adj[j, i] = 0.5

            layout.actual_adjacency_matrix = actual_adj
            if adjacency_matrix is not None:
                # If caller supplied required adjacency matrix, attach it
                layout.adjacency_matrix = adjacency_matrix
        except Exception:
            layout.actual_adjacency_matrix = None
            # leave required adjacency matrix as-is

        # Compute efficiency and compactness
        layout.circulation_efficiency = calculate_circulation_efficiency(
            circulation, room_positions
        )
        if circulation:
            # Mark as measured so calculate_metrics() keeps the path-ratio
            # value instead of overwriting it with the corridor-area proxy.
            layout.metadata['circulation_measured'] = True
        layout.compactness_score = calculate_compactness_score(room_positions)
        layout.space_utilization_ratio = used_area / max(total_area, 1.0)
        
        return layout
    
    def _build_networkx_adjacency_graph(
        self,
        room_polygons: Dict[str, Polygon],
        adjacency_matrix: np.ndarray,
        room_ids: List[str]
    ) -> nx.Graph:
        """Build NetworkX adjacency graph for visualization"""
        
        G = nx.Graph()
        
        for room_id, polygon in room_polygons.items():
            G.add_node(
                room_id,
                name=room_id,
                area=polygon.area,
                centroid=(polygon.centroid.x, polygon.centroid.y)
            )
        
        n = len(room_ids)
        for i in range(n):
            for j in range(i + 1, n):
                weight = adjacency_matrix[i, j]
                
                if weight > 0.1:
                    room1_id = room_ids[i]
                    room2_id = room_ids[j]
                    
                    G.add_edge(
                        room1_id,
                        room2_id,
                        weight=float(weight),
                        importance=float(weight)
                    )
        
        return G