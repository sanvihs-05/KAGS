"""
Adaptive Complexity Calculator for FBSL-KAGS Framework

Calculates design complexity metrics from requirements and FBSL nodes
to adaptively determine:
- Number of prototypes to generate
- GoT exploration parameters (depth, breadth, max_nodes)
- Aggregation strategies
- Pruning thresholds
"""

import logging
from typing import Dict, Any, Optional
import re
import numpy as np
from .fbsl_models import FBSLLayoutNode

logger = logging.getLogger(__name__)


class ComplexityCalculator:
    """
    Calculates design complexity from requirements and FBSL nodes
    
    Complexity factors:
    1. Spatial complexity: number of rooms, area constraints
    2. Functional complexity: number of functions, interdependencies
    3. Behavioral complexity: number of behaviors, constraints
    4. Requirements complexity: text length, constraint keywords
    5. Adjacency complexity: number of required adjacencies
    """
    
    def __init__(self):
        """Initialize complexity calculator"""
        # Complexity thresholds
        self.low_complexity_threshold = 0.3
        self.medium_complexity_threshold = 0.6
        self.high_complexity_threshold = 0.8
        
        logger.info("✓ Complexity Calculator initialized")
    
    def calculate_requirements_complexity(self, requirements: str) -> Dict[str, float]:
        """
        Calculate complexity from requirements text
        
        Returns:
            {
                'text_complexity': float,  # Based on length, keywords
                'constraint_count': int,   # Number of constraints mentioned
                'room_count_estimate': int, # Estimated rooms from text
                'adjacency_count_estimate': int, # Estimated adjacencies
                'overall': float  # Overall complexity score [0, 1]
            }
        """
        req_lower = requirements.lower()
        
        # 1. Text complexity (normalized by length)
        text_length = len(requirements)
        text_complexity = min(1.0, text_length / 500.0)  # Normalize to 500 chars
        
        # 2. Count constraint keywords
        constraint_keywords = [
            'must', 'should', 'require', 'need', 'constraint', 'limit',
            'minimum', 'maximum', 'avoid', 'prevent', 'ensure',
            'ventilation', 'daylight', 'privacy', 'noise', 'heat',
            'orientation', 'adjacent', 'connected', 'grouped'
        ]
        constraint_count = sum(1 for keyword in constraint_keywords if keyword in req_lower)
        constraint_complexity = min(1.0, constraint_count / 15.0)  # Normalize to 15 constraints
        
        # 3. Estimate room count from text
        room_keywords = [
            'bedroom', 'bathroom', 'kitchen', 'living', 'dining', 'study',
            'office', 'utility', 'storage', 'balcony', 'room', 'space'
        ]
        room_count_estimate = sum(1 for keyword in room_keywords if keyword in req_lower)
        # Also count explicit numbers
        number_patterns = re.findall(r'(\d+)\s*(?:bedroom|bathroom|room)', req_lower)
        if number_patterns:
            room_count_estimate += sum(int(n) for n in number_patterns)
        room_complexity = min(1.0, room_count_estimate / 10.0)  # Normalize to 10 rooms
        
        # 4. Estimate adjacency requirements
        adjacency_keywords = [
            'adjacent', 'connected', 'next to', 'near', 'grouped',
            'flow', 'circulation', 'access', 'link'
        ]
        adjacency_count_estimate = sum(1 for keyword in adjacency_keywords if keyword in req_lower)
        adjacency_complexity = min(1.0, adjacency_count_estimate / 8.0)  # Normalize to 8 adjacencies
        
        # 5. Area specification complexity
        area_specs = re.findall(r'(\d+)\s*[-–]\s*(\d+)\s*(?:sqm|m²|sq\.?\s*m)', req_lower)
        area_complexity = min(1.0, len(area_specs) / 5.0)  # Normalize to 5 area specs
        
        # Overall complexity (weighted average)
        overall = (
            0.15 * text_complexity +
            0.25 * constraint_complexity +
            0.30 * room_complexity +
            0.15 * adjacency_complexity +
            0.15 * area_complexity
        )
        
        return {
            'text_complexity': text_complexity,
            'constraint_count': constraint_count,
            'room_count_estimate': room_count_estimate,
            'adjacency_count_estimate': adjacency_count_estimate,
            'area_spec_count': len(area_specs),
            'overall': overall
        }
    
    def calculate_fbsl_complexity(self, node: FBSLLayoutNode) -> Dict[str, float]:
        """
        Calculate complexity from FBSL node structure
        
        Returns:
            {
                'function_count': int,
                'behavior_count': int,
                'structure_count': int,
                'room_count': int,
                'function_interdependency': float,  # Based on depends_on, conflicts_with
                'behavior_diversity': float,  # Number of different behavior categories
                'overall': float  # Overall complexity score [0, 1]
            }
        """
        # Count components
        function_count = len(node.functions)
        behavior_count = len(node.behaviors)
        structure_count = len(node.structures)
        
        # Room count
        room_count = 0
        if node.layout and node.layout.rooms:
            room_count = len(node.layout.rooms) if isinstance(node.layout.rooms, dict) else len(node.layout.rooms)
        
        # Function interdependency (how many functions depend on or conflict with others)
        interdependency_count = 0
        for func in node.functions.values():
            interdependency_count += len(func.depends_on)
            interdependency_count += len(func.conflicts_with)
            interdependency_count += len(func.enables)
        
        max_possible_interdependencies = function_count * (function_count - 1) if function_count > 1 else 0
        function_interdependency = (
            min(1.0, interdependency_count / max(max_possible_interdependencies, 1))
            if max_possible_interdependencies > 0 else 0.0
        )
        
        # Behavior diversity (number of different categories)
        behavior_categories = set()
        for behav in node.behaviors.values():
            if hasattr(behav, 'category'):
                cat = behav.category.value if hasattr(behav.category, 'value') else str(behav.category)
                behavior_categories.add(cat)
        
        behavior_diversity = min(1.0, len(behavior_categories) / 6.0)  # 6 main categories
        
        # Normalize counts
        function_complexity = min(1.0, function_count / 15.0)
        behavior_complexity = min(1.0, behavior_count / 20.0)
        room_complexity = min(1.0, room_count / 12.0)
        
        # Overall complexity (weighted)
        overall = (
            0.25 * function_complexity +
            0.20 * behavior_complexity +
            0.25 * room_complexity +
            0.15 * function_interdependency +
            0.15 * behavior_diversity
        )
        
        return {
            'function_count': function_count,
            'behavior_count': behavior_count,
            'structure_count': structure_count,
            'room_count': room_count,
            'function_interdependency': function_interdependency,
            'behavior_diversity': behavior_diversity,
            'overall': overall
        }
    
    def calculate_combined_complexity(
        self,
        requirements: str,
        node: Optional[FBSLLayoutNode] = None
    ) -> Dict[str, Any]:
        """
        Calculate combined complexity from requirements and FBSL node
        
        Returns:
            Complete complexity metrics with adaptive parameters
        """
        req_complexity = self.calculate_requirements_complexity(requirements)
        
        if node:
            fbsl_complexity = self.calculate_fbsl_complexity(node)
            
            # Combined overall complexity
            combined_overall = 0.4 * req_complexity['overall'] + 0.6 * fbsl_complexity['overall']
            
            # Merge metrics
            result = {
                **req_complexity,
                **fbsl_complexity,
                'combined_overall': combined_overall,
                'complexity_level': self._classify_complexity(combined_overall)
            }
        else:
            result = {
                **req_complexity,
                'combined_overall': req_complexity['overall'],
                'complexity_level': self._classify_complexity(req_complexity['overall'])
            }
        
        # Add adaptive parameters
        result['adaptive_params'] = self._calculate_adaptive_parameters(result)
        
        return result
    
    def _classify_complexity(self, score: float) -> str:
        """Classify complexity level"""
        if score < self.low_complexity_threshold:
            return 'low'
        elif score < self.medium_complexity_threshold:
            return 'medium'
        elif score < self.high_complexity_threshold:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_adaptive_parameters(self, complexity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate adaptive parameters based on complexity
        
        Returns adaptive settings for:
        - GoT exploration (depth, breadth, max_nodes)
        - Aggregation (top_k)
        - Pruning (quality_threshold, diversity_threshold)
        - Final prototypes (target_count)
        """
        overall = complexity.get('combined_overall', complexity.get('overall', 0.5))
        room_count = complexity.get('room_count', complexity.get('room_count_estimate', 5))
        function_count = complexity.get('function_count', complexity.get('room_count_estimate', 5))
        
        # Base parameters (for medium complexity)
        base_depth = 2
        base_breadth = 3
        base_max_nodes = 50
        base_aggregation_top_k = 3
        base_target_prototypes = 5
        
        # Scale factors based on complexity
        if overall < self.low_complexity_threshold:
            # Low complexity: fewer prototypes, simpler exploration
            depth_scale = 0.7
            breadth_scale = 0.7
            nodes_scale = 0.6
            aggregation_scale = 0.7
            prototypes_scale = 0.6
        elif overall < self.medium_complexity_threshold:
            # Medium complexity: standard parameters
            depth_scale = 1.0
            breadth_scale = 1.0
            nodes_scale = 1.0
            aggregation_scale = 1.0
            prototypes_scale = 1.0
        elif overall < self.high_complexity_threshold:
            # High complexity: more exploration
            depth_scale = 1.3
            breadth_scale = 1.3
            nodes_scale = 1.5
            aggregation_scale = 1.2
            prototypes_scale = 1.3
        else:
            # Very high complexity: extensive exploration
            depth_scale = 1.5
            breadth_scale = 1.5
            nodes_scale = 2.0
            aggregation_scale = 1.5
            prototypes_scale = 1.5
        
        # Also scale by component count (more rooms/functions = more exploration needed)
        component_scale = min(1.5, 1.0 + (room_count + function_count) / 20.0)
        
        # Calculate adaptive parameters
        adaptive_depth = max(1, int(base_depth * depth_scale))
        adaptive_breadth = max(2, int(base_breadth * breadth_scale * component_scale))
        adaptive_max_nodes = max(20, int(base_max_nodes * nodes_scale * component_scale))
        adaptive_aggregation_top_k = max(2, int(base_aggregation_top_k * aggregation_scale))
        adaptive_target_prototypes = max(3, int(base_target_prototypes * prototypes_scale * component_scale))
        
        # Pruning thresholds (higher complexity = more lenient pruning)
        quality_threshold = max(0.3, 0.5 - (overall * 0.2))  # Lower threshold for high complexity
        diversity_threshold = max(0.1, 0.3 - (overall * 0.1))  # Lower threshold for high complexity
        
        return {
            'got_depth': adaptive_depth,
            'got_breadth': adaptive_breadth,
            'got_max_nodes': adaptive_max_nodes,
            'aggregation_top_k': adaptive_aggregation_top_k,
            'target_prototypes': adaptive_target_prototypes,
            'quality_threshold': quality_threshold,
            'diversity_threshold': diversity_threshold,
            'pruning_enabled': overall > self.low_complexity_threshold  # Only prune for medium+ complexity
        }

