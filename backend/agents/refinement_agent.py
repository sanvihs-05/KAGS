"""
Refinement Agent: Implements iterative FBSL refinement loop
Based on Gero's reformulation types
"""

import logging
from typing import Dict, List, Any, Tuple
from ..core.fbsl_models import FBSLLayoutNode, Behavior, Structure, BehaviorType
from ..core.behavior_calculator import BehaviorCalculator

logger = logging.getLogger(__name__)

class RefinementAgent:
    """
    Refinement Agent: Iterative FBSL refinement
    Implements: Type 1 (Structure), Type 2 (Behavior), Type 3 (Function) reformulations
    """
    
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.01):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.behavior_calculator = BehaviorCalculator()
        logger.info("✓ Refinement Agent initialized")
    
    def refine_node(self, node: FBSLLayoutNode) -> Tuple[FBSLLayoutNode, Dict[str, Any]]:
        """
        Iteratively refine FBSL node until convergence or max iterations
        
        Args:
            node: FBSL node to refine
        
        Returns:
            (refined_node, refinement_history)
        """
        logger.info(f"Starting refinement for node: {node.node_id[:8]}...")
        
        # Ensure all functions have corresponding behaviors before refinement starts
        self._ensure_behaviors_exist(node)
        
        history = {
            'iterations': [],
            'converged': False,
            'final_score': 0.0
        }
        
        previous_score = 0.0
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n  Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Calculate actual behaviors from structures (S → Bs)
            node = self.behavior_calculator.calculate_actual_behaviors(node)
            
            # Step 2: Check if behaviors are satisfied
            unsatisfied = self._get_unsatisfied_behaviors(node)
            
            logger.info(f"    Unsatisfied behaviors: {len(unsatisfied)}/{len(node.behaviors)}")
            
            # Step 3: Calculate current score
            current_score = self._calculate_node_score(node)
            
            logger.info(f"    Score: {current_score:.3f} (prev: {previous_score:.3f})")
            
            # Step 4: Check convergence
            score_diff = abs(current_score - previous_score)
            
            iteration_data = {
                'iteration': iteration + 1,
                'score': current_score,
                'score_diff': score_diff,
                'unsatisfied_count': len(unsatisfied),
                'reformulation_type': None
            }
            
            if score_diff < self.convergence_threshold:
                logger.info(f"    ✓ Converged! (diff={score_diff:.4f})")
                history['converged'] = True
                history['iterations'].append(iteration_data)
                break
            
            # Step 5: Apply reformulation if needed
            if unsatisfied:
                node, reform_type = self._apply_reformulation(node, unsatisfied)
                iteration_data['reformulation_type'] = reform_type
                logger.info(f"    Applied: {reform_type}")
            
            history['iterations'].append(iteration_data)
            previous_score = current_score
        
        history['final_score'] = current_score
        
        logger.info(f"\n✓ Refinement complete: {len(history['iterations'])} iterations, "
                   f"converged={history['converged']}, final_score={current_score:.3f}")
        
        return node, history
    
    def _ensure_behaviors_exist(self, node: FBSLLayoutNode):
        """
        Ensure every function has at least one corresponding behavior.
        If missing, create an area-based behavior with a reasonable target.
        """
        from ..core.fbsl_models import Behavior, BehaviorCategory
        
        for func_id, func in node.functions.items():
            has_behavior = any(b.derived_from_function == func_id for b in node.behaviors.values())
            if not has_behavior:
                # Create default spatial behavior
                area = 12.0
                if func.spatial_requirements and isinstance(func.spatial_requirements, dict):
                    area = func.spatial_requirements.get('preferred_area', 12.0)
                
                behavior = Behavior(
                    category=BehaviorCategory.SPATIAL,
                    metric_name=f"{func.name}_area",
                    metric_unit="sqm",
                    target_value=area,
                    derived_from_function=func_id
                )
                node.add_behavior(behavior)
                logger.debug(f"Created missing behavior for function {func.name}")
    
    def _get_unsatisfied_behaviors(self, node: FBSLLayoutNode) -> List[Behavior]:
        """Get list of unsatisfied behaviors"""
        unsatisfied = []
        
        for behav in node.behaviors.values():
            if behav.behavior_type == BehaviorType.EXPECTED:
                if not behav.is_satisfied:
                    unsatisfied.append(behav)
        
        return unsatisfied
    
    def _calculate_node_score(self, node: FBSLLayoutNode) -> float:
        """Calculate simple node score based on behavior satisfaction"""
        if not node.behaviors:
            return 0.5
        
        satisfied_count = sum(
            1 for b in node.behaviors.values()
            if b.behavior_type == BehaviorType.EXPECTED and b.is_satisfied
        )
        
        total_count = sum(
            1 for b in node.behaviors.values()
            if b.behavior_type == BehaviorType.EXPECTED
        )
        
        return satisfied_count / max(total_count, 1)
    
    def _apply_reformulation(self, node: FBSLLayoutNode, 
                            unsatisfied: List[Behavior]) -> Tuple[FBSLLayoutNode, str]:
        """
        Apply Gero's reformulation strategies
        
        Type 1: Modify structures when Bs ≠ Be
        Type 2: Relax behaviors when no structure satisfies Be
        Type 3: Redefine functions when Be cannot be achieved
        """
        
        # Analyze unsatisfied behaviors
        avg_deviation = np.mean([
            abs(b.actual_value - b.target_value) / max(b.target_value, 0.001)
            for b in unsatisfied
            if b.actual_value and b.target_value
        ]) if unsatisfied else 0.0
        
        # Decision logic
        if avg_deviation < 0.3:
            # Type 1: Small deviation, modify structures
            return self._type1_reformulation(node, unsatisfied), "Type 1: Structure Modification"
        
        elif avg_deviation < 0.6:
            # Type 2: Moderate deviation, relax behaviors
            return self._type2_reformulation(node, unsatisfied), "Type 2: Behavior Relaxation"
        
        else:
            # Type 3: Large deviation, redefine functions
            return self._type3_reformulation(node, unsatisfied), "Type 3: Function Redefinition"
    
    def _type1_reformulation(self, node: FBSLLayoutNode, 
                            unsatisfied: List[Behavior]) -> FBSLLayoutNode:
        """
        Type 1: Modify structures (S' = S + ΔS)
        ✅ CRITICAL FIX: Optimize ΔS to minimize |Bs - Be|
        
        Theoretical formula: S' = S + ΔS where ΔS minimizes |Bₛ - Bₑ|
        """
        logger.info("      Applying Type 1: Optimizing structure modifications")
        
        # Calculate current deviation: |Bs - Be|
        current_deviations = {}
        for behav in unsatisfied:
            if behav.actual_value is not None and behav.target_value is not None:
                deviation = abs(behav.actual_value - behav.target_value)
                current_deviations[behav.behavior_id] = deviation
        
        # Try different structure modifications and select optimal ΔS
        best_node = node
        best_total_deviation = sum(current_deviations.values())
        
        # Strategy: Try adding structures incrementally and measure improvement
        for behav in unsatisfied:
            category = behav.category.value
            target_value = behav.target_value
            current_value = behav.actual_value or 0.0
            
            # Calculate required improvement
            required_improvement = target_value - current_value
            
            # Try structure modifications
            test_node = self._create_test_node(node)
            structure_added = False
            
            if category == 'thermal' and required_improvement > 0:
                self._add_thermal_structure(test_node)
                structure_added = True
            elif category == 'acoustic' and required_improvement > 0:
                self._add_acoustic_structure(test_node)
                structure_added = True
            elif category == 'lighting' and required_improvement > 0:
                self._add_lighting_structure(test_node)
                structure_added = True
            elif category == 'ventilation' and required_improvement > 0:
                # A natural-ventilation variant made a deliberate trade-off:
                # re-adding mechanical MEP here would silently turn it back
                # into the base design and collapse the variant space.
                if (getattr(node, 'metadata', None) or {}).get('ventilation_strategy') != 'natural':
                    self._add_ventilation_structure(test_node)
                    structure_added = True
            elif category == 'spatial' and required_improvement > 0:
                # For spatial behaviors, modify layout structures
                self._add_spatial_structure(test_node, behav)
                structure_added = True
            
            if structure_added:
                # Recalculate actual behaviors with new structure
                test_node = self.behavior_calculator.calculate_actual_behaviors(test_node)
                
                # Calculate new deviation for this behavior
                test_behav = test_node.behaviors.get(behav.behavior_id)
                if test_behav and test_behav.actual_value is not None:
                    new_deviation = abs(test_behav.actual_value - target_value)
                    
                    # If this reduces total deviation, accept it
                    new_total = sum(
                        abs(b.actual_value - b.target_value) 
                        if b.actual_value and b.target_value else current_deviations.get(b.behavior_id, 0)
                        for b in test_node.behaviors.values()
                        if b.behavior_type == BehaviorType.EXPECTED
                    )
                    
                    if new_total < best_total_deviation:
                        best_node = test_node
                        best_total_deviation = new_total
                        logger.info(
                            f"        Optimal ΔS found: {category} structure reduces "
                            f"deviation from {current_deviations.get(behav.behavior_id, 0):.3f} to {new_deviation:.3f}"
                        )
        
        return best_node
    
    def _create_test_node(self, node: FBSLLayoutNode) -> FBSLLayoutNode:
        """Create a copy of node for testing structure modifications"""
        import copy
        return copy.deepcopy(node)
    
    def _add_spatial_structure(self, node: FBSLLayoutNode, behav: Behavior):
        """Add spatial structure to improve area/volume behaviors"""
        from ..core.fbsl_models import Structure, StructureType
        
        # Add partition or modify room dimensions
        if 'area' in behav.metric_name.lower():
            # Add flexible partition that can be adjusted
            partition = Structure(
                name="adjustable_partition",
                structure_type=StructureType.PARTITION,
                material_type="movable_wall",
                dimensions={'thickness': 0.1, 'adjustable': True}
            )
            node.add_structure(partition)
    
    def _type2_reformulation(self, node: FBSLLayoutNode,
                            unsatisfied: List[Behavior]) -> FBSLLayoutNode:
        """
        Type 2: Relax behaviors (B'ₑ = Bₑ × (1 + tolerance))
        ✅ CRITICAL FIX: Modifies target value Bₑ, not just tolerance
        
        Theoretical formula: B'ₑ = Bₑ × (1 + tolerance)
        When no structure can satisfy strict requirements
        """
        logger.info("      Applying Type 2: Relaxing behavior target values")
        
        for behav in unsatisfied:
            if behav.target_value is not None:
                # ✅ THEORETICAL IMPLEMENTATION: Modify target value
                # B'ₑ = Bₑ × (1 + tolerance)
                original_target = behav.target_value
                relaxation_factor = 1.0 + behav.tolerance
                
                # For positive behaviors (more is better), increase target
                # For negative behaviors (less is better, e.g., cost), decrease target
                if behav.category.value in ['spatial', 'lighting', 'ventilation', 'thermal']:
                    # Positive behaviors: increase target
                    behav.target_value = original_target * relaxation_factor
                else:
                    # Negative behaviors (cost, energy): decrease target
                    behav.target_value = original_target / relaxation_factor
                
                # Also increase tolerance for future iterations
                behav.tolerance *= 1.2
                
                # Recalculate satisfaction with new target
                behav.calculate_satisfaction()
                
                logger.info(
                    f"        Relaxed {behav.metric_name}: "
                    f"target {original_target:.3f} → {behav.target_value:.3f} "
                    f"(factor: {relaxation_factor:.2f}, tolerance: {behav.tolerance:.2f})"
                )
        
        return node
    
    def _type3_reformulation(self, node: FBSLLayoutNode,
                            unsatisfied: List[Behavior]) -> FBSLLayoutNode:
        """
        Type 3: Redefine functions (F' = Transform(F))
        When behaviors cannot be achieved with current functions
        """
        logger.info("      Applying Type 3: Redefining functions")
        
        # Reduce priority of functions that lead to unsatisfiable behaviors
        for behav in unsatisfied:
            if behav.derived_from_function:
                func = node.functions.get(behav.derived_from_function)
                if func:
                    func.priority *= 0.8  # Reduce priority
                    logger.info(f"        Reduced priority of {func.name} to {func.priority:.2f}")
        
        return node
    
    # Helper methods to add structures
    def _add_thermal_structure(self, node: FBSLLayoutNode):
        """Add thermal insulation structure"""
        from ..core.fbsl_models import Structure, StructureType
        
        insulation = Structure(
            name="thermal_insulation",
            structure_type=StructureType.WALL,
            material_type="insulation",
            dimensions={'thickness': 0.15}
        )
        node.add_structure(insulation)
    
    def _add_acoustic_structure(self, node: FBSLLayoutNode):
        """Add acoustic partition"""
        from ..core.fbsl_models import Structure, StructureType
        
        partition = Structure(
            name="acoustic_partition",
            structure_type=StructureType.PARTITION,
            material_type="gypsum_board",
            acoustic_rating="STC50",
            dimensions={'thickness': 0.12}
        )
        node.add_structure(partition)
    
    def _add_lighting_structure(self, node: FBSLLayoutNode):
        """Add window opening"""
        from ..core.fbsl_models import Structure, StructureType
        
        window = Structure(
            name="window_opening",
            structure_type=StructureType.WALL,
            material_type="glass",
            dimensions={'width': 1.5, 'height': 2.0}
        )
        node.add_structure(window)
    
    def _add_ventilation_structure(self, node: FBSLLayoutNode):
        """Add ventilation system"""
        from ..core.fbsl_models import Structure, StructureType
        
        ventilation = Structure(
            name="ventilation_system",
            structure_type=StructureType.MEP,
            material_type="steel_duct",
            dimensions={'diameter': 0.2}
        )
        node.add_structure(ventilation)


import numpy as np