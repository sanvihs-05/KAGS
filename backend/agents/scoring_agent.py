# agents/scoring_agent.py
"""
Scoring Agent: Evaluates FBSL design nodes using multi-criteria analysis
Phase 1 Complete Implementation:
- Geometric mean for behavioral scores ✓
- Rho-parameter controlled composite aggregation ✓
- Coverage(f_i) measuring DEGREE of function satisfaction ✓
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple

from ..core.fbsl_models import FBSLLayoutNode, BehaviorType

logger = logging.getLogger(__name__)


class ScoringAgent:
    """
    Scoring Agent: Multi-criteria evaluation of FBSL nodes

    Phase 1 Improvements:
    - Behavioral score uses geometric mean: exp(mean(log(min(1, actual/target))))
    - Composite score uses (sum w_i * S_i^rho)^(1/rho) with configurable rho
      If rho == 0 -> weighted geometric mean: exp(sum w_i * log S_i)
    - Functional adequacy uses Coverage(f_i) = degree of satisfaction per function [0,1]
      NOT just binary count of satisfied behaviors
    """

    def __init__(self, weights: Dict[str, float] = None, rho: float = -1.0):
        """
        Initialize Scoring Agent
        
        Args:
            weights: Dictionary of criteria weights (will be normalized to sum to 1)
            rho: Compensation parameter for composite score
                 rho < 1: Anti-compensatory (penalizes low scores)
                 rho = 0: Geometric mean
                 rho > 1: Compensatory (allows trade-offs)
        """
        # Default weights
        self.weights = weights or {
            'functional_adequacy': 0.25,
            'behavioral_performance': 0.20,
            'structural_feasibility': 0.20,
            'layout_efficiency': 0.25,
            'sustainability': 0.10
        }
        
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values()) or 1.0
        for k in list(self.weights.keys()):
            self.weights[k] = float(self.weights[k]) / total_w

        # Rho parameter for composite aggregation
        self.rho = float(rho)

        logger.info(f"✓ Scoring Agent initialized (rho={self.rho}, weights normalized)")

    async def score_node(self, node: FBSLLayoutNode) -> Dict[str, Any]:
        """
        Comprehensive scoring of FBSL node
        
        Args:
            node: FBSL node to evaluate
        
        Returns:
            Dictionary with scores and details
        """
        logger.info(f"Scoring node: {node.node_id[:8]}...")
        
        # Calculate individual scores
        func_score, func_details = self._score_functions(node)
        behav_score, behav_details = self._score_behaviors(node)
        struct_score, struct_details = self._score_structures(node)
        layout_score, layout_details = self._score_layout(node)
        sust_score, sust_details = self._score_sustainability(node)
        
        # Build arrays for composite calculation (order must match weights)
        scores_array = np.array([
            func_score,
            behav_score,
            struct_score,
            layout_score,
            sust_score
        ], dtype=float)

        weights_array = np.array([
            self.weights['functional_adequacy'],
            self.weights['behavioral_performance'],
            self.weights['structural_feasibility'],
            self.weights['layout_efficiency'],
            self.weights['sustainability']
        ], dtype=float)

        # Defensive clamps and normalization
        scores_array = np.clip(scores_array, 0.0, 1.0)
        weights_array = np.clip(weights_array, 0.0, None)
        if weights_array.sum() == 0:
            weights_array = np.ones_like(weights_array) / len(weights_array)
        else:
            weights_array = weights_array / weights_array.sum()

        # Composite aggregation using rho-parameter
        # Formula: S_composite = (Σ(w_i × S_i^ρ))^(1/ρ)
        rho = self.rho
        try:
            if abs(rho) < 1e-9:
                # rho == 0: Weighted geometric mean
                # Formula: exp(Σ(w_i × ln(S_i)))
                safe_scores = np.clip(scores_array, 1e-9, 1.0)
                composite = float(np.exp(np.sum(weights_array * np.log(safe_scores))))
            else:
                # General form: (Σ(w_i × S_i^ρ))^(1/ρ)
                safe_scores = np.clip(scores_array, 1e-9, 1.0)
                powered = safe_scores ** rho
                weighted_sum = np.sum(weights_array * powered)
                weighted_sum = max(weighted_sum, 1e-12)  # Prevent division issues
                composite = float(weighted_sum ** (1.0 / rho))
        except Exception as e:
            logger.warning(f"Composite scoring numeric error: {e}, falling back to arithmetic mean")
            composite = float(np.mean(scores_array))

        # Final clamp
        composite = float(max(0.0, min(1.0, composite)))

        # Update node with scores
        node.functional_score = float(func_score)
        node.behavioral_score = float(behav_score)
        node.structural_score = float(struct_score)
        node.layout_score = float(layout_score)
        node.sustainability_score = float(sust_score)
        node.composite_score = float(composite)

        result = {
            'scores': {
                'functional_adequacy': node.functional_score,
                'behavioral_performance': node.behavioral_score,
                'structural_feasibility': node.structural_score,
                'layout_efficiency': node.layout_score,
                'sustainability': node.sustainability_score,
                'composite': node.composite_score
            },
            'details': {
                'functional': func_details,
                'behavioral': behav_details,
                'structural': struct_details,
                'layout': layout_details,
                'sustainability': sust_details
            }
        }

        logger.info(f"✓ Scoring complete: composite={node.composite_score:.3f}")
        return result

    def _score_functions(self, node: FBSLLayoutNode) -> Tuple[float, Dict]:
        """
        Score functional adequacy using Coverage(f_i) for each function
        
        Coverage(f_i) = weighted DEGREE of function satisfaction [0,1]
        NOT just binary "has behaviors satisfied" count
        
        For each function:
        1. Find all behaviors derived from that function
        2. Calculate performance ratio for each behavior (actual/target)
        3. Average these ratios to get coverage
        4. Weight by function priority
        """
        if not node.functions:
            return 0.0, {'message': 'No functions defined'}

        total_weighted = 0.0
        total_weight = 0.0
        details: Dict[str, Any] = {}

        for func_id, func in node.functions.items():
            # Gather behaviors derived from this function
            related_behaviors = [
                b for b in node.behaviors.values()
                if b.derived_from_function == func_id
            ]

            if related_behaviors:
                # Calculate DEGREE of satisfaction (not binary count)
                behavior_scores = []
                satisfied_count = 0
                
                for b in related_behaviors:
                    if b.target_value is not None and b.actual_value is not None:
                        # Ratio-based scoring: how close to target
                        ratio = b.actual_value / max(b.target_value, 1e-9)
                        score = float(min(1.0, ratio))  # Cap at 1.0 (met or exceeded)
                        behavior_scores.append(score)
                        if b.is_satisfied:
                            satisfied_count += 1
                    elif b.is_satisfied:
                        # Binary fallback: satisfied = 1.0
                        behavior_scores.append(1.0)
                        satisfied_count += 1
                    else:
                        # Not satisfied or not measured
                        behavior_scores.append(0.0)
                
                # Coverage = average performance across all behaviors
                coverage = float(np.mean(behavior_scores)) if behavior_scores else 0.0
            else:
                # Conservative fallback: function exists but no behaviors mapped
                coverage = 0.3
                satisfied_count = 0

            score = float(np.clip(coverage, 0.0, 1.0))
            weight = float(np.clip(func.priority, 0.0, 1.0))

            total_weighted += score * weight
            total_weight += weight

            details[func_id] = {
                'name': func.name,
                'coverage': score,  # DEGREE of satisfaction [0,1]
                'behaviors_mapped': len(related_behaviors),
                'satisfied_behaviors': satisfied_count,
                'avg_behavior_performance': coverage,
                'priority': weight
            }

        final_score = float(total_weighted / max(total_weight, 1e-9))
        final_score = float(np.clip(final_score, 0.0, 1.0))

        return final_score, details

    def _score_behaviors(self, node: FBSLLayoutNode) -> Tuple[float, Dict]:
        """
        Score behavioral performance using geometric mean
        
        Formula: S_B = exp(mean(log(min(1, b_actual_i / b_expected_i))))
        
        Geometric mean ensures that:
        - All behaviors must perform reasonably well
        - One poor behavior significantly impacts overall score
        - More conservative than arithmetic mean
        """
        if not node.behaviors:
            return 0.5, {'message': 'No behaviors defined'}

        per_behavior_scores = []
        details: Dict[str, Any] = {}

        for behav_id, behav in node.behaviors.items():
            # Calculate ratio: actual / target (capped at 1.0)
            if behav.target_value is not None and behav.actual_value is not None:
                ratio = behav.actual_value / max(behav.target_value, 1e-9)
                score = float(min(1.0, ratio))
            else:
                # Neutral default when not measured
                score = 0.5

            # Ensure numeric stability (avoid zeros for geometric mean)
            safe_score = float(np.clip(score, 1e-9, 1.0))
            per_behavior_scores.append(safe_score)

            details[behav_id] = {
                'metric': behav.metric_name,
                'target': behav.target_value,
                'actual': behav.actual_value,
                'ratio': score,
                'satisfied': bool(behav.is_satisfied)
            }

        # Geometric mean: exp(mean(log(scores)))
        if per_behavior_scores:
            log_scores = np.log(per_behavior_scores)
            geo_mean = float(np.exp(np.mean(log_scores)))
            final_score = float(np.clip(geo_mean, 0.0, 1.0))
        else:
            final_score = 0.5

        details['_summary'] = {
            'geometric_mean': final_score,
            'behavior_count': len(per_behavior_scores)
        }

        return final_score, details

    def _score_structures(self, node: FBSLLayoutNode) -> Tuple[float, Dict]:
        """Score structural feasibility"""
        if not node.structures:
            return 0.5, {'message': 'No structures defined'}

        score = 1.0
        details: Dict[str, Any] = {}

        # Check for essential structural features
        has_structural = any(s.load_bearing for s in node.structures.values())
        has_envelope = any(s.category == 'envelope' for s in node.structures.values())

        if not has_structural:
            score *= 0.7
        if not has_envelope:
            score *= 0.8

        details['has_structural_system'] = bool(has_structural)
        details['has_envelope'] = bool(has_envelope)
        details['total_structures'] = len(node.structures)

        score = float(np.clip(score, 0.0, 1.0))
        return score, details

    def _score_layout(self, node: FBSLLayoutNode) -> Tuple[float, Dict]:
        """Score layout efficiency"""
        if not node.layout:
            return 0.5, {'message': 'No layout defined'}

        layout = node.layout

        # Ensure metrics are calculated
        try:
            layout.calculate_metrics()
        except Exception:
            pass  # Use existing values

        # Extract metrics with reasonable defaults (not 0.0)
        # If metrics aren't calculated, assume reasonable defaults rather than 0.0
        space_util = float(getattr(layout, 'space_utilization_ratio', None) or 0.7)  # Default 70% utilization
        circ_eff = float(getattr(layout, 'circulation_efficiency', None) or 0.8)  # Default 80% efficiency
        # ✅ Adjacency: when the layout agent MEASURED satisfaction against the
        # brief's requirements, trust it — including a genuine 0.0. The 0.6
        # default only applies to layouts that never went through placement
        # (the old `or 0.6` silently turned every measured 0.0 into 0.6).
        _adj_raw = getattr(layout, 'adjacency_satisfaction_score', None)
        if (getattr(layout, 'metadata', None) or {}).get('adjacency_measured'):
            adj_sat = float(_adj_raw if _adj_raw is not None else 0.0)
        else:
            adj_sat = float(_adj_raw or 0.6)
        compact = float(getattr(layout, 'compactness_score', None) or 0.5)  # Default 0.5 compactness

        # Weighted combination (internal weights)
        score = (
            space_util * 0.3 +
            circ_eff * 0.25 +
            adj_sat * 0.3 +
            compact * 0.15
        )

        score = float(np.clip(score, 0.0, 1.0))

        details = {
            'space_utilization': space_util,
            'circulation_efficiency': circ_eff,
            'adjacency_satisfaction': adj_sat,
            'compactness': compact,
            'rooms_count': len(layout.rooms) if getattr(layout, 'rooms', None) else 0
        }

        return score, details

    def _score_sustainability(self, node: FBSLLayoutNode) -> Tuple[float, Dict]:
        """Score sustainability"""
        score = 0.5  # Default neutral baseline
        details: Dict[str, Any] = {'message': 'Basic sustainability assessment'}

        # Check metadata for sustainability hints
        if 'natural_light_access' in node.metadata:
            score += 0.05
            details['natural_light'] = True

        if 'energy_efficiency' in node.metadata:
            score += 0.05
            details['energy_efficient'] = True

        # Check structures for sustainable materials
        if node.structures:
            sustainable_materials = ['wood', 'bamboo', 'recycled']
            has_sustainable = any(
                any(mat in (s.material_type or "").lower() for mat in sustainable_materials)
                for s in node.structures.values()
            )
            if has_sustainable:
                score += 0.1
                details['sustainable_materials'] = True

        score = float(np.clip(score, 0.0, 1.0))
        return score, details