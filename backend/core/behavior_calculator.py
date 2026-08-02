# core/behavior_calculator.py
"""
Behavior Calculator: Derives actual behaviors (Bs) from structures (S)
Implements: S → Bs transformation

Enhanced with GUARANTEED actual_value setting for all behaviors
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import re

from .fbsl_models import (
    FBSLLayoutNode, 
    Behavior, 
    Structure, 
    BehaviorType, 
    BehaviorCategory
)

logger = logging.getLogger(__name__)


class BehaviorCalculator:
    """
    Calculates actual behaviors (Bs) from structures
    Based on: Bs = f(S) where structures exhibit measurable performance
    
    ✅ GUARANTEES:
    - Every behavior gets an actual_value (never None)
    - Uses layout rooms when no structures available
    - Falls back to conservative estimates when needed
    """
    
    def __init__(self):
        """Initialize behavior calculator with material databases"""
        self.material_properties = self._initialize_material_properties()
        self.structural_rules = self._initialize_structural_rules()
        
        self.thermal_coefficients = {
            'internal_heat_gain': 5.0,
            'ventilation_rate': 0.5,
            'solar_heat_gain': 0.6
        }
        
        self.acoustic_coefficients = {
            'distance_attenuation': 6.0,
            'absorption_coefficient': 0.3
        }
        
        logger.info("✓ Behavior Calculator initialized with enhanced physics models")
    
    def calculate_actual_behaviors(self, node: FBSLLayoutNode) -> FBSLLayoutNode:
        """
        Calculate actual behaviors (Bs) from structures (S)
        
        ✅ CRITICAL FIX: Ensures ALL behaviors have actual_value set
        
        Priority order for calculating actual values:
        1. From structures (physics-based calculations)
        2. From layout rooms (area-based metrics)
        3. Conservative estimate (85% of target)
        4. Absolute fallback (1.0)
        """
        logger.info(f"Calculating actual behaviors for node: {node.node_id[:8]}...")
        
        calculated_count = 0
        satisfied_count = 0
        
        for behav_id, expected_behav in node.behaviors.items():
            if expected_behav.behavior_type != BehaviorType.EXPECTED:
                continue
            
            # ✅ GUARANTEE: actual_value will be set by end of this block
            actual_value = None
            
            # Try method 1: Calculate from structures (physics-based)
            if node.structures:
                actual_value = self._calculate_behavior_from_structures(
                    expected_behav,
                    node.structures,
                    node
                )
            
            # Try method 2: Extract from layout rooms (area-based)
            if actual_value is None and node.layout and node.layout.rooms:
                actual_value = self._calculate_from_layout_rooms(expected_behav, node)
            
            # Try method 3: Conservative estimate from target
            if actual_value is None and expected_behav.target_value:
                actual_value = expected_behav.target_value * 0.85
                logger.debug(f"  Using conservative estimate (85% of target) for {expected_behav.metric_name}")
            
            # Absolute fallback
            if actual_value is None:
                actual_value = 1.0
                logger.warning(f"⚠️ Using absolute fallback (1.0) for {expected_behav.metric_name}")
            
            # ✅ Set actual value and calculate satisfaction
            expected_behav.actual_value = float(actual_value)
            expected_behav.calculate_satisfaction()
            calculated_count += 1
            
            if expected_behav.is_satisfied:
                satisfied_count += 1
            
            logger.debug(
                f"  {expected_behav.metric_name}: "
                f"target={expected_behav.target_value:.2f}, "
                f"actual={actual_value:.2f}, "
                f"satisfied={expected_behav.is_satisfied}"
            )
        
        satisfaction_rate = (satisfied_count / max(calculated_count, 1)) * 100
        logger.info(
            f"  ✓ Calculated {calculated_count} behaviors "
            f"({satisfied_count}/{calculated_count} satisfied, {satisfaction_rate:.1f}%)"
        )
        
        return node
    
    def _calculate_from_layout_rooms(self, behavior: Behavior, node: FBSLLayoutNode) -> Optional[float]:
        """
        Calculate actual value from layout rooms
        
        ✅ NEW METHOD: Extracts actual values from layout when structures unavailable
        """
        metric_lower = behavior.metric_name.lower()
        
        # Area-based metrics. A per-room area behavior (e.g. 'bedroom_area',
        # target = one room's preferred area) must be compared to that room's
        # area — NOT the sum of the whole house. The old fallback summed EVERY
        # room when the function-id linkage failed (which it does after GoT
        # deep-copies remap ids), giving actual = total floor area (~175 m²)
        # against a ~14 m² target — a 12× ratio that pinned S_f/S_b at 1.0.
        if 'area' in metric_lower:
            rooms = list(node.layout.rooms.values())
            # 1) rooms linked to this behavior's function
            related = [r for r in rooms
                       if behavior.derived_from_function
                       and r.function_id == behavior.derived_from_function]
            # 2) fallback: match by room type parsed from the metric name
            #    ('bedroom_area' -> 'bedroom'), robust to broken id linkage
            if not related:
                rtype = metric_lower.replace('_area', '').replace('area', '').strip('_ ')
                related = [r for r in rooms if (r.room_type or '').lower() == rtype]
            if related:
                # per-room behavior: mean area of the matching room(s)
                mean_area = sum(r.area for r in related) / len(related)
                return mean_area
            # 3) last resort: mean room area — never the sum (which inflates
            #    a per-room ratio by the room count).
            if rooms:
                return sum(r.area for r in rooms) / len(rooms)
            return None
        
        # Volume-based metrics
        if 'volume' in metric_lower:
            total_volume = sum(r.volume for r in node.layout.rooms.values() if r.volume)
            if total_volume > 0:
                logger.debug(f"    Calculated volume from rooms: {total_volume:.2f} m³")
                return total_volume
        
        # Room count metrics
        if 'count' in metric_lower or 'number' in metric_lower:
            count = len(node.layout.rooms)
            logger.debug(f"    Calculated room count: {count}")
            return float(count)
        
        return None
    
    def _calculate_behavior_from_structures(
        self, 
        behavior: Behavior,
        structures: Dict[str, Structure],
        node: FBSLLayoutNode
    ) -> Optional[float]:
        """
        Calculate actual behavior value from structures
        
        Routes to appropriate calculation method based on behavior category
        """
        category = behavior.category
        
        # Route to category-specific calculation
        if category == BehaviorCategory.THERMAL:
            return self._calculate_thermal_behavior(behavior, structures, node)
        elif category == BehaviorCategory.ACOUSTIC:
            return self._calculate_acoustic_behavior(behavior, structures, node)
        elif category == BehaviorCategory.LIGHTING:
            return self._calculate_lighting_behavior(behavior, structures, node)
        elif category == BehaviorCategory.SPATIAL:
            return self._calculate_spatial_behavior(behavior, structures, node)
        elif category == BehaviorCategory.STRUCTURAL:
            return self._calculate_structural_behavior(behavior, structures, node)
        elif category == BehaviorCategory.VENTILATION:
            return self._calculate_ventilation_behavior(behavior, structures, node)
        else:
            # Default: 90% of target
            return behavior.target_value * 0.9 if behavior.target_value else 0.9
    
    def _calculate_thermal_behavior(
        self, 
        behavior: Behavior, 
        structures: Dict[str, Structure], 
        node: FBSLLayoutNode
    ) -> float:
        """Calculate thermal performance from structures"""
        
        envelope_structures = [
            s for s in structures.values() 
            if s.category == 'envelope' or any(
                keyword in s.name.lower() 
                for keyword in ['wall', 'roof', 'floor', 'foundation']
            )
        ]
        
        if not envelope_structures:
            logger.debug("    No envelope structures, using default thermal performance")
            return behavior.target_value * 0.85 if behavior.target_value else 0.85
        
        total_r_value = 0.0
        total_weight = 0.0
        
        for struct in envelope_structures:
            material = struct.material_type.lower() if struct.material_type else 'concrete'
            mat_props = self.material_properties.get(material, self.material_properties['concrete'])
            
            u_value = mat_props.get('u_value', 2.0)
            r_value = 1.0 / max(u_value, 0.1)
            
            area = struct.dimensions.get('area', 1.0) if struct.dimensions else 1.0
            total_r_value += r_value * area
            total_weight += area
        
        avg_r_value = total_r_value / max(total_weight, 1.0)
        target_r = 5.0
        # Uncapped (clamped to 2×): actual carries TRUE performance so a design
        # that EXCEEDS the target R-value scores higher than one that just meets
        # it, instead of both being flattened to the target. The scorer rewards
        # a bounded margin above target (see ScoringAgent._perf_score).
        performance_ratio = min(2.0, avg_r_value / target_r)

        actual_value = behavior.target_value * performance_ratio if behavior.target_value else performance_ratio
        
        logger.debug(f"    Thermal: R={avg_r_value:.2f}, ratio={performance_ratio:.2f}")
        return actual_value
    
    def _calculate_acoustic_behavior(
        self, 
        behavior: Behavior,
        structures: Dict[str, Structure], 
        node: FBSLLayoutNode
    ) -> float:
        """Calculate acoustic performance from structures"""
        
        acoustic_structures = [
            s for s in structures.values()
            if any(keyword in s.name.lower() for keyword in ['wall', 'partition', 'door'])
        ]
        
        if not acoustic_structures:
            logger.debug("    No acoustic structures, using default performance")
            return behavior.target_value * 0.75 if behavior.target_value else 0.75
        
        stc_ratings = []
        
        for struct in acoustic_structures:
            if struct.acoustic_rating:
                match = re.search(r'\d+', str(struct.acoustic_rating))
                if match:
                    stc_ratings.append(float(match.group()))
            else:
                material = struct.material_type.lower() if struct.material_type else 'gypsum_board'
                mat_props = self.material_properties.get(material, {})
                base_stc = mat_props.get('stc', 35.0)
                
                if struct.dimensions and 'thickness' in struct.dimensions:
                    thickness = struct.dimensions['thickness']
                    stc_adjustment = min(10.0, thickness * 50)
                    stc_ratings.append(base_stc + stc_adjustment)
                else:
                    stc_ratings.append(base_stc)
        
        if stc_ratings:
            avg_stc = np.mean(stc_ratings)
            target_stc = 45.0
            performance_ratio = min(2.0, avg_stc / target_stc)  # uncapped: reward exceeding target STC
            
            actual_value = behavior.target_value * performance_ratio if behavior.target_value else avg_stc
            logger.debug(f"    Acoustic: STC={avg_stc:.1f}, ratio={performance_ratio:.2f}")
            return actual_value
        
        return behavior.target_value * 0.80 if behavior.target_value else 0.80
    
    @staticmethod
    def _glazing_ratios_by_room_type(structures: Dict[str, Structure]) -> Dict[str, List[float]]:
        """Map room type → declared glazing ratios, from "<room_type>_window"
        structure names (the encoder's naming convention). Shared by the
        daylight and ventilation models, which both need per-room opening area
        rather than one building-wide average."""
        out: Dict[str, List[float]] = {}
        for s in structures.values():
            name = (s.name or '').lower()
            if not any(k in name for k in ('window', 'glazing', 'skylight', 'opening')):
                continue
            dims = s.dimensions or {}
            if 'window_ratio' not in dims:
                continue
            rtype = re.split(r'_(window|glazing|skylight|opening)', name)[0].strip()
            try:
                out.setdefault(rtype, []).append(float(dims['window_ratio']))
            except (TypeError, ValueError):
                continue
        return out

    # --- Daylight constants (BRE average daylight factor, Lynes) -------------
    _DF_TRANSMITTANCE = 0.70   # double glazing incl. frame and maintenance factors
    _DF_SKY_ANGLE = 75.0       # degrees of visible sky — low-rise/suburban default
    _DF_REFLECTANCE = 0.50     # area-weighted mean interior surface reflectance
    _DF_WINDOW_HEAD = 2.1      # m — window head height above floor
    _DF_TARGET = 3.0           # % — "well lit" habitable room

    def _calculate_lighting_behavior(
        self,
        behavior: Behavior,
        structures: Dict[str, Structure],
        node: FBSLLayoutNode
    ) -> float:
        """Average daylight factor per the BRE (Lynes) formula.

            DF = (T · A_w · θ) / (A_total · (1 − R²))

        T = glazing transmittance, A_w = net glazed area, θ = visible sky angle,
        A_total = total *interior surface* area of the room (floor + ceiling +
        walls), R = mean interior reflectance.

        This replaces `DF = window_ratio × 0.75 × 100`, which divided glazing by
        floor area alone. That both ignored room geometry — a tall or deep room
        needs more glazing for the same daylight — and was wrong by roughly 5×
        in absolute terms: an 18 % glazing ratio came out at DF 13.5 %, an
        atrium-like figure, where the BRE formula gives a realistic ~2.5 %.
        Dividing by total interior surface rather than floor area is what makes
        room proportion and ceiling height matter.

        A supplementary penalty applies the BRE limiting-depth rule
        (L/W + L/H_w ≤ 2/(1−R)): rooms deeper than the limit cannot daylight
        their back half, so DF is scaled by limit/actual. That rule is
        pass/fail in the standard; using it as a continuous factor is an
        approximation, made here so the ranking degrades smoothly rather than
        stepping. Windows are assumed to sit on the longer external wall (normal
        practice), so depth is the room's shorter dimension.

        Concept-stage estimate: no orientation, sun path, latitude, external
        obstruction survey or internal partitions — θ and R are fixed defaults.
        Comparative, not a compliance figure.
        """
        rooms = (node.layout.rooms if node.layout and node.layout.rooms else {}) or {}
        ratios_by_type = self._glazing_ratios_by_room_type(structures)
        has_windows = bool(ratios_by_type) or any(
            'window' in (s.name or '').lower() for s in structures.values()
        )

        if not rooms:
            if has_windows:
                return behavior.target_value * 0.85 if behavior.target_value else 0.85
            return behavior.target_value * 0.60 if behavior.target_value else 0.60

        depth_limit = 2.0 / max(1e-6, (1.0 - self._DF_REFLECTANCE))
        weighted, total_area = 0.0, 0.0

        for r in rooms.values():
            area = float(getattr(r, 'area', 0.0) or 0.0)
            if area <= 0:
                continue
            height = float(getattr(r, 'height', 3.0) or 3.0)
            rtype = (getattr(r, 'room_type', '') or '').lower()
            rlist = ratios_by_type.get(rtype, [])
            ratio = float(np.mean(rlist)) if rlist else 0.0

            # Plan dimensions; fall back to a square room when unknown.
            w = float(getattr(r, 'width', 0) or 0)
            l = float(getattr(r, 'length', 0) or 0)
            if w <= 0 or l <= 0:
                w = l = float(np.sqrt(area))
            long_side, short_side = max(w, l), min(w, l)

            a_glazed = ratio * area
            # floor + ceiling + walls
            a_total = 2.0 * area + 2.0 * (w + l) * height
            df = (
                self._DF_TRANSMITTANCE * a_glazed * self._DF_SKY_ANGLE
                / max(1e-9, a_total * (1.0 - self._DF_REFLECTANCE ** 2))
            )

            # Limiting depth: window on the longer wall → depth is the short side.
            depth_index = (short_side / max(long_side, 1e-9)
                           + short_side / self._DF_WINDOW_HEAD)
            if depth_index > depth_limit:
                df *= depth_limit / depth_index

            weighted += df * area
            total_area += area

        if total_area <= 0:
            return behavior.target_value * 0.60 if behavior.target_value else 0.60

        daylight_factor = weighted / total_area
        performance_ratio = min(2.0, daylight_factor / self._DF_TARGET)
        logger.debug(f"    Lighting: DF={daylight_factor:.2f}%, ratio={performance_ratio:.2f}")
        return behavior.target_value * performance_ratio if behavior.target_value else daylight_factor
    
    def _calculate_spatial_behavior(
        self, 
        behavior: Behavior,
        structures: Dict[str, Structure], 
        node: FBSLLayoutNode
    ) -> float:
        """Calculate spatial performance metrics"""
        
        metric = behavior.metric_name.lower()
        
        # Area-related metrics — a per-room area behavior (target = ONE room's
        # preferred area) must be compared to that room's area, not the whole
        # house. Summing every room gave actual ~= total floor area against a
        # ~14 m² target, a 12× ratio that flattened S_f/S_b to 1.0. Match the
        # behavior's function (or the room type in its metric name) and use the
        # mean matching-room area.
        if 'area' in metric:
            if node.layout and node.layout.rooms:
                rooms = list(node.layout.rooms.values())
                related = [r for r in rooms
                           if behavior.derived_from_function
                           and r.function_id == behavior.derived_from_function]
                if not related:
                    rtype = metric.replace('_area', '').replace('area', '').strip('_ ')
                    related = [r for r in rooms if (r.room_type or '').lower() == rtype]
                pool = related if related else rooms
                mean_area = sum(r.area for r in pool) / max(len(pool), 1)
                logger.debug(f"    Spatial area (per-room mean): {mean_area:.2f} m²")
                return mean_area
            elif node.functions:
                # per-room target: mean preferred area across functions
                areas = [func.spatial_requirements.get('preferred_area', 15.0)
                         for func in node.functions.values()
                         if func.spatial_requirements and isinstance(func.spatial_requirements, dict)]
                if areas:
                    return (sum(areas) / len(areas)) * 0.90
        
        # Privacy-related metrics
        if 'privacy' in metric:
            has_partitions = any('partition' in s.name.lower() for s in structures.values())
            has_doors = any('door' in s.name.lower() for s in structures.values())
            
            privacy_score = 0.5
            if has_partitions:
                privacy_score += 0.3
            if has_doors:
                privacy_score += 0.2
            
            return behavior.target_value * privacy_score if behavior.target_value else privacy_score
        
        # Circulation-related metrics
        if 'circulation' in metric or 'access' in metric:
            if node.layout and hasattr(node.layout, 'circulation_efficiency'):
                return node.layout.circulation_efficiency
            else:
                return behavior.target_value * 0.85 if behavior.target_value else 0.85
        
        return behavior.target_value if behavior.target_value else 1.0
    
    def _calculate_structural_behavior(
        self, 
        behavior: Behavior,
        structures: Dict[str, Structure], 
        node: FBSLLayoutNode
    ) -> float:
        """Calculate structural performance/feasibility"""
        
        load_bearing = [s for s in structures.values() if s.load_bearing]
        
        if not load_bearing:
            logger.debug("    No load-bearing structures defined")
            return behavior.target_value * 0.50 if behavior.target_value else 0.50
        
        has_foundation = any('foundation' in s.name.lower() for s in structures.values())
        has_columns = any('column' in s.name.lower() for s in structures.values())
        has_beams = any('beam' in s.name.lower() for s in structures.values())
        has_slabs = any('slab' in s.name.lower() or 'floor' in s.name.lower() for s in structures.values())
        has_walls = any('wall' in s.name.lower() for s in structures.values())
        
        components = [has_foundation, has_columns or has_walls, has_beams, has_slabs]
        completeness = sum(components) / len(components)
        
        logger.debug(f"    Structural completeness: {completeness:.2f}")
        return behavior.target_value * completeness if behavior.target_value else completeness
    
    # --- Ventilation constants (each an explicit, citable design assumption) ---
    _VENT_CD = 0.61            # sharp-edged orifice discharge coefficient
    _VENT_WIND_COEFF = 0.025   # BS 5925 single-sided single-opening empirical factor
    _VENT_WIND_SPEED = 3.0     # m/s — typical sheltered low-rise mean wind speed
    _VENT_OPENABLE_FRAC = 0.45 # openable share of a glazed area (casement/tilt-turn)
    _VENT_OPENING_H = 1.2      # m — window opening height, drives the stack term
    _VENT_DELTA_T = 3.0        # K indoor–outdoor difference, mild design condition
    _VENT_T_MEAN = 293.0       # K
    _VENT_MECH_RATE = 0.0005   # m³/s per m² floor (0.5 l/s·m², EU dwelling guidance)
    _VENT_MECH_BOOST = 3.0     # purge/boost multiple typical of an MVHR unit
    _VENT_PURGE_TARGET = 4.0   # ACH — rapid ("purge") ventilation criterion
    _VENT_INFILTRATION = 0.15  # ACH — background envelope leakage; no real
                               # envelope is airtight, so even a design with no
                               # openings and no plant is never exactly zero

    def _calculate_ventilation_behavior(
        self,
        behavior: Behavior,
        structures: Dict[str, Structure],
        node: FBSLLayoutNode
    ) -> float:
        """Air changes per hour computed from the actual opening geometry.

        This used to be a label lookup (HVAC → 1.0, windows → 0.75, else 0.40),
        which did no physics at all: it could not tell apart two naturally
        ventilated designs with very different glazing, and handed any design
        with an HVAC object a perfect score regardless of its capacity.

        Natural flow uses the BS 5925 / CIBSE AM10 concept-stage single-sided
        equations, taking the greater of the wind- and buoyancy-driven rates
        (they are not additive):

            Q_wind  = 0.025 · A_open · v
            Q_stack = (Cd/3) · A_open · √(g · H · ΔT / T̄)

        Mechanical systems contribute their design supply rate at boost. Note
        that because a room's glazed area scales with its floor area, the
        resulting ACH depends on the *glazing ratio and ceiling height* rather
        than room size — so it responds to the design decisions the GoT variants
        actually change. The building figure is the floor-area-weighted mean of
        the per-room rates, so windowless interior rooms (served by mechanical
        only) correctly drag it down. Scored against the purge criterion, which
        is the demanding case openable area is sized for.

        Concept-stage envelope flow only: no weather file, orientation, sun path
        or wind-pressure-coefficient set. Comparative, not a compliance figure.
        """
        rooms = (node.layout.rooms if node.layout and node.layout.rooms else {}) or {}
        if not rooms:
            # No geometry to reason about — fall back to presence heuristics.
            has_windows = any('window' in s.name.lower() for s in structures.values())
            score = 0.75 if has_windows else 0.40
            return behavior.target_value * score if behavior.target_value else score

        ratios_by_type = self._glazing_ratios_by_room_type(structures)

        has_hvac = any(
            getattr(s.structure_type, 'value', str(s.structure_type)) == 'mep'
            and any(k in (s.name or '').lower() for k in ('hvac', 'ventilation', 'air', 'duct'))
            for s in structures.values()
        )

        # m³/s delivered per m² of openable area (greater of wind and stack)
        q_per_m2 = max(
            self._VENT_WIND_COEFF * self._VENT_WIND_SPEED,
            (self._VENT_CD / 3.0) * float(np.sqrt(
                9.81 * self._VENT_OPENING_H * self._VENT_DELTA_T / self._VENT_T_MEAN
            )),
        )
        # A mechanical system supplies its design rate everywhere it serves.
        ach_mech = (
            self._VENT_MECH_RATE * 3600.0 * self._VENT_MECH_BOOST / 3.0
        ) if has_hvac else 0.0

        weighted, total_area = 0.0, 0.0
        for r in rooms.values():
            area = float(getattr(r, 'area', 0.0) or 0.0)
            height = float(getattr(r, 'height', 3.0) or 3.0)
            if area <= 0:
                continue
            rtype = (getattr(r, 'room_type', '') or '').lower()
            rlist = ratios_by_type.get(rtype, [])
            ratio = float(np.mean(rlist)) if rlist else 0.0
            # ACH from openings: area cancels, leaving glazing-ratio / height
            a_open = self._VENT_OPENABLE_FRAC * ratio * area
            ach_nat = (q_per_m2 * a_open) * 3600.0 / (area * height)
            # mechanical rate expressed against this room's own height
            ach_room = (self._VENT_INFILTRATION + ach_nat
                        + (ach_mech * 3.0 / height if has_hvac else 0.0))
            weighted += ach_room * area
            total_area += area

        if total_area <= 0:
            return behavior.target_value * 0.40 if behavior.target_value else 0.40

        ach = weighted / total_area
        performance_ratio = min(2.0, ach / self._VENT_PURGE_TARGET)
        logger.debug(f"    Ventilation: {ach:.2f} ACH, ratio={performance_ratio:.2f}")
        return behavior.target_value * performance_ratio if behavior.target_value else ach
    
    def _initialize_material_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize material thermal and acoustic properties"""
        return {
            'concrete': {'u_value': 2.0, 'stc': 50.0, 'density': 2400.0},
            'brick': {'u_value': 1.7, 'stc': 45.0, 'density': 1800.0},
            'wood': {'u_value': 1.3, 'stc': 35.0, 'density': 600.0},
            'steel': {'u_value': 5.0, 'stc': 40.0, 'density': 7800.0},
            'glass': {'u_value': 2.8, 'stc': 30.0, 'density': 2500.0},
            'gypsum_board': {'u_value': 0.8, 'stc': 35.0, 'density': 800.0},
            'insulation': {'u_value': 0.04, 'stc': 20.0, 'density': 30.0},
            'waterproof_membrane': {'u_value': 0.5, 'stc': 15.0, 'density': 1200.0},
        }
    
    def _initialize_structural_rules(self) -> Dict[str, Any]:
        """Initialize structural design rules and constraints"""
        return {
            'min_wall_thickness': 0.1,
            'max_span': 6.0,
            'min_foundation_depth': 0.6,
            'min_column_spacing': 3.0,
            'max_column_spacing': 8.0,
            'min_slab_thickness': 0.15,
            'min_beam_depth': 0.3,
        }