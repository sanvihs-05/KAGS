import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import math
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringCriterion(Enum):
    """Multi-criteria scoring dimensions"""
    SPATIAL_EFFICIENCY = "spatial_efficiency"
    FUNCTIONAL_ORGANIZATION = "functional_organization"
    CIRCULATION_QUALITY = "circulation_quality"
    ENVIRONMENTAL_PERFORMANCE = "environmental_performance"
    COST_EFFICIENCY = "cost_efficiency"
    CONSTRUCTABILITY = "constructability"
    AESTHETIC_QUALITY = "aesthetic_quality"
    CODE_COMPLIANCE = "code_compliance"
    USER_PREFERENCE_ALIGNMENT = "user_preference_alignment"

@dataclass
class ScoringResult:
    """Individual scoring result for a criterion"""
    criterion: ScoringCriterion
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    sub_scores: Dict[str, float] = field(default_factory=dict)
    penalty_factors: Dict[str, float] = field(default_factory=dict)
    bonus_factors: Dict[str, float] = field(default_factory=dict)
    # Inside ScoringResult dataclass (after the fields)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'criterion': self.criterion.value,  # Convert Enum to str
            'score': float(self.score),  # Ensure float (in case of np.float)
            'confidence': float(self.confidence),
            'explanation': self.explanation,
            'sub_scores': {k: float(v) for k, v in self.sub_scores.items()},
            'penalty_factors': {k: float(v) for k, v in self.penalty_factors.items()},
            'bonus_factors': {k: float(v) for k, v in self.bonus_factors.items()}
            }

@dataclass
class ComprehensiveScore:
    """Complete scoring result for a prototype"""
    prototype_id: str
    individual_scores: Dict[ScoringCriterion, ScoringResult] = field(default_factory=dict)
    weighted_total: float = 0.0
    diversity_bonus: float = 0.0
    final_score: float = 0.0
    overall_confidence: float = 0.0
    ranking_factors: Dict[str, Any] = field(default_factory=dict)
    pareto_efficiency: Dict[str, float] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prototype_id': self.prototype_id,
            'individual_scores': {
                criterion.value: result.to_dict()  # Convert Enum key to str, recurse on ScoringResult
                for criterion, result in self.individual_scores.items()
                },
            'weighted_total': float(self.weighted_total),
            'diversity_bonus': float(self.diversity_bonus),
            'final_score': float(self.final_score),
            'overall_confidence': float(self.overall_confidence),
            'ranking_factors': {k: float(v) if isinstance(v, (float, np.float64)) else v for k, v in self.ranking_factors.items()},
            'pareto_efficiency': {k: float(v) for k, v in self.pareto_efficiency.items()}
            }

class ScoringWeightProfile(Enum):
    """Different weighting profiles for various user priorities"""
    BALANCED = "balanced"
    COST_FOCUSED = "cost_focused"
    SUSTAINABILITY_FOCUSED = "sustainability_focused"
    AESTHETIC_FOCUSED = "aesthetic_focused"
    FUNCTIONAL_FOCUSED = "functional_focused"
    EFFICIENCY_FOCUSED = "efficiency_focused"

class MultiCriteriaScoringAgent:
    """
    Enhanced Multi-Criteria Scoring Agent for GOT-RAG-FBS system
    Implements comprehensive prototype evaluation with dynamic weighting
    """
    
    def __init__(self, 
                 default_weight_profile: ScoringWeightProfile = ScoringWeightProfile.BALANCED,
                 diversity_weight: float = 0.1,
                 pareto_analysis: bool = True):
        
        self.default_weight_profile = default_weight_profile
        self.diversity_weight = diversity_weight
        self.pareto_analysis = pareto_analysis
        
        # Initialize scoring weights for different profiles
        self.weight_profiles = self._initialize_weight_profiles()
        
        # Scoring parameters
        self.scoring_parameters = self._initialize_scoring_parameters()
        
        # Building codes and standards (simplified)
        self.building_standards = self._initialize_building_standards()
        
        # Scoring statistics
        self.scoring_stats = {
            'total_prototypes_scored': 0,
            'avg_scoring_time': 0.0,
            'score_distributions': defaultdict(list),
            'pareto_optimal_count': 0
        }
        
        # Cache for expensive calculations
        self.calculation_cache = {}
    
    def _initialize_weight_profiles(self) -> Dict[ScoringWeightProfile, Dict[ScoringCriterion, float]]:
        """Initialize different weighting profiles for various user priorities"""
        
        profiles = {
            ScoringWeightProfile.BALANCED: {
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.15,
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.15,
                ScoringCriterion.CIRCULATION_QUALITY: 0.12,
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.12,
                ScoringCriterion.COST_EFFICIENCY: 0.12,
                ScoringCriterion.CONSTRUCTABILITY: 0.10,
                ScoringCriterion.AESTHETIC_QUALITY: 0.08,
                ScoringCriterion.CODE_COMPLIANCE: 0.10,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.06
            },
            
            ScoringWeightProfile.COST_FOCUSED: {
                ScoringCriterion.COST_EFFICIENCY: 0.25,
                ScoringCriterion.CONSTRUCTABILITY: 0.20,
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.15,
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.12,
                ScoringCriterion.CODE_COMPLIANCE: 0.12,
                ScoringCriterion.CIRCULATION_QUALITY: 0.08,
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.05,
                ScoringCriterion.AESTHETIC_QUALITY: 0.02,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.01
            },
            
            ScoringWeightProfile.SUSTAINABILITY_FOCUSED: {
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.25,
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.18,
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.15,
                ScoringCriterion.COST_EFFICIENCY: 0.12,
                ScoringCriterion.CODE_COMPLIANCE: 0.10,
                ScoringCriterion.CIRCULATION_QUALITY: 0.08,
                ScoringCriterion.CONSTRUCTABILITY: 0.07,
                ScoringCriterion.AESTHETIC_QUALITY: 0.03,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.02
            },
            
            ScoringWeightProfile.AESTHETIC_FOCUSED: {
                ScoringCriterion.AESTHETIC_QUALITY: 0.22,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.18,
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.15,
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.13,
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.12,
                ScoringCriterion.CODE_COMPLIANCE: 0.08,
                ScoringCriterion.CIRCULATION_QUALITY: 0.07,
                ScoringCriterion.COST_EFFICIENCY: 0.03,
                ScoringCriterion.CONSTRUCTABILITY: 0.02
            },
            
            ScoringWeightProfile.FUNCTIONAL_FOCUSED: {
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.22,
                ScoringCriterion.CIRCULATION_QUALITY: 0.18,
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.16,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.14,
                ScoringCriterion.CODE_COMPLIANCE: 0.12,
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.08,
                ScoringCriterion.COST_EFFICIENCY: 0.06,
                ScoringCriterion.CONSTRUCTABILITY: 0.02,
                ScoringCriterion.AESTHETIC_QUALITY: 0.02
            },
            
            ScoringWeightProfile.EFFICIENCY_FOCUSED: {
                ScoringCriterion.SPATIAL_EFFICIENCY: 0.25,
                ScoringCriterion.CIRCULATION_QUALITY: 0.20,
                ScoringCriterion.COST_EFFICIENCY: 0.18,
                ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.15,
                ScoringCriterion.CONSTRUCTABILITY: 0.10,
                ScoringCriterion.CODE_COMPLIANCE: 0.07,
                ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.03,
                ScoringCriterion.AESTHETIC_QUALITY: 0.01,
                ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.01
            }
        }
        
        # Normalize weights to ensure they sum to 1.0
        for profile_name, weights in profiles.items():
            total_weight = sum(weights.values())
            for criterion in weights:
                weights[criterion] = weights[criterion] / total_weight
        
        return profiles
    
    def _initialize_scoring_parameters(self) -> Dict[str, Any]:
        """Initialize parameters for scoring calculations"""
        
        return {
            # Spatial efficiency parameters
            'optimal_plot_utilization': 0.75,
            'max_plot_utilization': 0.85,
            'min_plot_utilization': 0.45,
            'optimal_circulation_ratio': 0.15,
            'max_circulation_ratio': 0.25,
            
            # Environmental parameters
            'min_natural_light_ratio': 0.10,
            'optimal_natural_light_ratio': 0.15,
            'cross_ventilation_bonus': 0.15,
            'passive_strategy_bonus_per_item': 0.05,
            'orientation_bonus': {
                'south': 0.10, 'southeast': 0.12, 'east': 0.08,
                'northeast': 0.06, 'southwest': 0.04, 'west': 0.02,
                'northwest': 0.01, 'north': 0.03
            },
            
            # Cost parameters
            'cost_per_sqm_base': 25000,  # INR per sqm for Bangalore
            'complexity_multipliers': {
                'central_core': 1.0,
                'linear_progression': 0.9,
                'courtyard_focused': 1.15,
                'functional_split': 1.05
            },
            
            # Constructability parameters
            'standard_room_shapes_bonus': 0.10,
            'regular_geometry_bonus': 0.08,
            'standard_spans_bonus': 0.06,
            
            # Code compliance thresholds
            'min_room_areas': {
                'bedroom': 100,  # sqft
                'living_room': 150,
                'kitchen': 80,
                'bathroom': 35,
                'dining_room': 100
            },
            'min_window_ratios': {
                'bedroom': 0.10,
                'living_room': 0.12,
                'kitchen': 0.10,
                'bathroom': 0.05,
                'dining_room': 0.10
            }
        }
    
    def _initialize_building_standards(self) -> Dict[str, Any]:
        """Initialize building codes and standards for compliance checking"""
        
        return {
            'nbc_2016_standards': {
                'min_ceiling_height': 2.4,  # meters
                'min_room_widths': {
                    'bedroom': 2.4,  # meters
                    'living_room': 3.0,
                    'kitchen': 1.8,
                    'bathroom': 1.2,
                    'dining_room': 2.4
                },
                'min_corridor_width': 1.0,  # meters
                'min_stair_width': 1.0,
                'fire_safety_requirements': {
                    'max_travel_distance': 22.5,  # meters
                    'min_exit_width': 1.0
                }
            },
            
            'karnataka_building_bylaws': {
                'max_ground_coverage': 0.60,
                'min_setbacks': {
                    'front': 3.0,  # meters
                    'side': 1.5,
                    'rear': 2.0
                },
                'min_open_space': 0.40  # 40% of plot
            },
            
            'accessibility_standards': {
                'min_door_width': 0.9,  # meters
                'max_threshold_height': 0.02,  # meters
                'ramp_requirements': True
            }
        }
    
    def score_prototypes(self, prototypes: List[Dict[str, Any]], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score prototypes with multi-criteria evaluation."""
        logger.info(f"Scoring {len(prototypes)} prototypes with multi-criteria evaluation")

        if not prototypes:
            return []

        # Get current weight profile
        weights = self.weight_profiles[self.default_weight_profile]

        # Create comprehensive scores list first
        comprehensive_scores = []

        for idx, proto in enumerate(prototypes):
            # Calculate individual criterion scores
            individual_scores = {}
            for criterion in ScoringCriterion:
                result = self._score_criterion(proto, requirements, criterion)
                individual_scores[criterion] = result

            # Calculate weighted total
            weighted_total = sum(
                weights[criterion] * result.score
                for criterion, result in individual_scores.items()
            )

            # Overall confidence
            overall_confidence = np.mean([result.confidence for result in individual_scores.values()])

            # Ranking factors
            ranking_factors = self._calculate_ranking_factors_simple(proto, requirements, individual_scores)

            # Create comprehensive score (without diversity bonus initially)
            score = ComprehensiveScore(
                prototype_id=proto['prototype_id'],
                individual_scores=individual_scores,
                weighted_total=weighted_total,
                diversity_bonus=0.0,  # Will be calculated later
                final_score=weighted_total,  # Will be updated with diversity
                overall_confidence=overall_confidence,
                ranking_factors=ranking_factors,
                pareto_efficiency={}  # Will be filled by Pareto analysis
            )

            comprehensive_scores.append(score)

        # Calculate diversity bonuses
        self._calculate_diversity_bonuses(comprehensive_scores, prototypes)

        # Update final scores with diversity bonuses
        for score in comprehensive_scores:
            score.final_score = min(max(score.weighted_total + score.diversity_bonus, 0.0), 1.0)

        # Perform Pareto analysis if enabled
        if self.pareto_analysis:
            self._perform_pareto_analysis_fixed(comprehensive_scores)

        # Embed scores back into prototypes
        scored_prototypes = []
        for idx, (proto, score) in enumerate(zip(prototypes, comprehensive_scores)):
            proto['comprehensive_score'] = score.to_dict()
            scored_prototypes.append(proto)

        # Update statistics
        self._update_scoring_statistics(scored_prototypes)

        logger.info(
            f"Completed scoring. Top score: "
            f"{max(p['comprehensive_score']['final_score'] for p in scored_prototypes):.3f}"
        )

        return scored_prototypes

    def _score_criterion(self, prototype: Dict[str, Any], requirements: Dict[str, Any], criterion: "ScoringCriterion") -> "ScoringResult":
        """Score individual criterion for a prototype - simplified version"""
        if criterion == ScoringCriterion.SPATIAL_EFFICIENCY:
            return self._score_spatial_efficiency(prototype, requirements)
        elif criterion == ScoringCriterion.CODE_COMPLIANCE:
            return self._score_code_compliance(prototype, requirements)
        else:
            return self._score_placeholder_criterion(criterion)

    def _score_placeholder_criterion(self, criterion: "ScoringCriterion") -> "ScoringResult":
        """Placeholder scoring for criteria not fully implemented"""
        base_scores = {
            ScoringCriterion.FUNCTIONAL_ORGANIZATION: 0.8,
            ScoringCriterion.CIRCULATION_QUALITY: 0.75,
            ScoringCriterion.ENVIRONMENTAL_PERFORMANCE: 0.7,
            ScoringCriterion.COST_EFFICIENCY: 0.85,
            ScoringCriterion.CONSTRUCTABILITY: 0.9,
            ScoringCriterion.AESTHETIC_QUALITY: 0.65,
            ScoringCriterion.USER_PREFERENCE_ALIGNMENT: 0.8
        }
        score = base_scores.get(criterion, 0.7)
        return ScoringResult(
            criterion=criterion,
            score=score,
            confidence=0.7,
            explanation=f"{criterion.value} placeholder scoring: {score:.2f}"
        )

    def _calculate_ranking_factors_simple(self, prototype: Dict[str, Any], requirements: Dict[str, Any], individual_scores: Dict) -> Dict[str, Any]:
        """Simplified ranking factors calculation"""
        ranking_factors = {}
        scores = [result.score for result in individual_scores.values()]
        confidences = [result.confidence for result in individual_scores.values()]

        ranking_factors['confidence_weighted_score'] = np.mean(scores) * np.mean(confidences)
        low_confidence_criteria = sum(1 for conf in confidences if conf < 0.6)
        ranking_factors['risk_level'] = low_confidence_criteria / len(confidences)
        ranking_factors['performance_balance'] = 1.0 - np.std(scores) if scores else 0.5

        budget = requirements.get('budget', 0)
        if budget > 0:
            ranking_factors['budget_fit'] = min(1.0, budget / 2500000)
        else:
            ranking_factors['budget_fit'] = 1.0

        return ranking_factors

    def _perform_pareto_analysis_fixed(self, comprehensive_scores: List["ComprehensiveScore"]):
        """Perform Pareto efficiency analysis - fixed version"""
        if len(comprehensive_scores) <= 1:
            return

        objectives = [
            ScoringCriterion.SPATIAL_EFFICIENCY,
            ScoringCriterion.COST_EFFICIENCY,
            ScoringCriterion.ENVIRONMENTAL_PERFORMANCE,
            ScoringCriterion.FUNCTIONAL_ORGANIZATION
        ]

        objective_matrix = []
        for score in comprehensive_scores:
            objective_scores = [
                score.individual_scores.get(obj, ScoringResult(obj, 0.5, 0.5, "")).score
                for obj in objectives
            ]
            objective_matrix.append(objective_scores)

        objective_matrix = np.array(objective_matrix)
        pareto_optimal = self._find_pareto_optimal(objective_matrix)

        for i, score in enumerate(comprehensive_scores):
            if i in pareto_optimal:
                score.pareto_efficiency['is_pareto_optimal'] = True
                score.pareto_efficiency['pareto_rank'] = 1
                self.scoring_stats['pareto_optimal_count'] += 1
            else:
                score.pareto_efficiency['is_pareto_optimal'] = False
                domination_count = 0
                for j in pareto_optimal:
                    if self._dominates(objective_matrix[j], objective_matrix[i]):
                        domination_count += 1
                score.pareto_efficiency['pareto_rank'] = domination_count + 1

            if not score.pareto_efficiency['is_pareto_optimal']:
                min_distance = float('inf')
                for j in pareto_optimal:
                    distance = np.linalg.norm(objective_matrix[i] - objective_matrix[j])
                    min_distance = min(min_distance, distance)
                score.pareto_efficiency['distance_to_pareto_front'] = min_distance
            else:
                score.pareto_efficiency['distance_to_pareto_front'] = 0.0

    def _update_scoring_statistics(self, scored_prototypes: List[Dict[str, Any]]):
        """Update scoring statistics"""
        self.scoring_stats['total_prototypes_scored'] += len(scored_prototypes)
        for proto in scored_prototypes:
            comp_score = proto['comprehensive_score']
            for criterion_name, criterion_data in comp_score['individual_scores'].items():
                self.scoring_stats['score_distributions'][criterion_name].append(criterion_data['score'])
    
    def _determine_weight_profile(self, requirements: Dict[str, Any]) -> ScoringWeightProfile:
        """Automatically determine appropriate weight profile based on requirements"""
    
         # Convert requirements to dict if it's a dataclass
        if hasattr(requirements, '__dict__'):
            requirements = asdict(requirements)
    
        # Handle design_preferences - convert to dict if needed
        design_prefs = requirements.get('design_preferences', {})
        if hasattr(design_prefs, '__dict__'):
            design_prefs = asdict(design_prefs)
    
        budget = requirements.get('budget', 0)
    
        # Budget-conscious users
        if budget > 0 and budget < 2000000:  # Less than 20 lakhs
            return ScoringWeightProfile.COST_FOCUSED
    
        # Sustainability focus
        if design_prefs.get('sustainability_focus', False) or design_prefs.get('green_building', False):
            return ScoringWeightProfile.SUSTAINABILITY_FOCUSED
        
        # Aesthetic focus
        if design_prefs.get('style_importance', 'medium') == 'high' or design_prefs.get('visual_appeal', False):
            return ScoringWeightProfile.AESTHETIC_FOCUSED
        
        # Functional focus
        if len(requirements.get('spatial_needs', [])) > 8 or design_prefs.get('functional_priority', False):
            return ScoringWeightProfile.FUNCTIONAL_FOCUSED
        
        # Efficiency focus
        if design_prefs.get('space_optimization', False) or design_prefs.get('compact_design', False):
            return ScoringWeightProfile.EFFICIENCY_FOCUSED
        
        return ScoringWeightProfile.BALANCED
    
    def _score_individual_criterion(self, 
                                  prototype: Dict[str, Any], 
                                  requirements: Dict[str, Any],
                                  criterion: ScoringCriterion,
                                  research_data: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score individual criterion for a prototype"""
        
        if criterion == ScoringCriterion.SPATIAL_EFFICIENCY:
            return self._score_spatial_efficiency(prototype, requirements, research_data)
        elif criterion == ScoringCriterion.FUNCTIONAL_ORGANIZATION:
            return self._score_functional_organization(prototype, requirements, research_data)
        elif criterion == ScoringCriterion.CIRCULATION_QUALITY:
            return self._score_circulation_quality(prototype, requirements)
        elif criterion == ScoringCriterion.ENVIRONMENTAL_PERFORMANCE:
            return self._score_environmental_performance(prototype, requirements, research_data)
        elif criterion == ScoringCriterion.COST_EFFICIENCY:
            return self._score_cost_efficiency(prototype, requirements)
        elif criterion == ScoringCriterion.CONSTRUCTABILITY:
            return self._score_constructability(prototype, requirements)
        elif criterion == ScoringCriterion.AESTHETIC_QUALITY:
            return self._score_aesthetic_quality(prototype, requirements)
        elif criterion == ScoringCriterion.CODE_COMPLIANCE:
            return self._score_code_compliance(prototype, requirements)
        elif criterion == ScoringCriterion.USER_PREFERENCE_ALIGNMENT:
            return self._score_user_preference_alignment(prototype, requirements)
        else:
            return ScoringResult(
                criterion=criterion,
                score=0.5,
                confidence=0.2,
                explanation="Criterion not implemented"
            )
    
    def _score_spatial_efficiency(self, 
                                prototype: Dict[str, Any], 
                                requirements: Dict[str, Any],
                                research_data: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score spatial efficiency of the prototype"""
        
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        site_constraints = requirements.get('site_constraints', {})
        
        # Plot utilization score
        plot_utilization = spatial_config.get('plot_utilization', 0.7)
        optimal_util = self.scoring_parameters['optimal_plot_utilization']
        
        if plot_utilization <= optimal_util:
            utilization_score = plot_utilization / optimal_util
        else:
            max_util = self.scoring_parameters['max_plot_utilization']
            penalty = (plot_utilization - optimal_util) / (max_util - optimal_util)
            utilization_score = 1.0 - (penalty * 0.3)  # 30% penalty for over-utilization
        
        utilization_score = max(0.0, min(1.0, utilization_score))
        
        # Compactness score
        compactness = spatial_config.get('compactness_factor', 0.7)
        compactness_score = compactness  # Direct mapping
        
        # Area efficiency from room layouts
        area_efficiency_score = 0.8  # Default assumption
        if 'room_layouts' in prototype.get('detailed_config', {}).get('structures', {}):
            room_layouts = prototype['detailed_config']['structures']['room_layouts']
            
            total_room_area = sum(room.get('area', 100) for room in room_layouts.values())
            spatial_needs = requirements.get('spatial_needs', [])
            required_area = sum(need.get('min_area', 100) for need in spatial_needs)
            
            if required_area > 0:
                area_efficiency_score = min(1.0, required_area / total_room_area)
        
        # Circulation efficiency
        circulation_config = prototype.get('detailed_config', {}).get('circulation_pattern', {})
        circulation_efficiency = circulation_config.get('efficiency_target', 0.85)
        
        # Aspect ratio efficiency (based on plot dimensions)
        plot_length = site_constraints.get('plot_length', 50)
        plot_width = site_constraints.get('plot_width', 30)
        plot_aspect_ratio = max(plot_length, plot_width) / min(plot_length, plot_width)
        
        # Optimal aspect ratio is between 1.2 and 2.0
        if 1.2 <= plot_aspect_ratio <= 2.0:
            aspect_score = 1.0
        elif plot_aspect_ratio < 1.2:
            aspect_score = plot_aspect_ratio / 1.2
        else:
            aspect_score = max(0.3, 2.0 / plot_aspect_ratio)
        
        # Combine sub-scores
        sub_scores = {
            'plot_utilization': utilization_score,
            'compactness': compactness_score,
            'area_efficiency': area_efficiency_score,
            'circulation_efficiency': circulation_efficiency,
            'aspect_ratio': aspect_score
        }
        
        # Weighted combination
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        final_score = sum(score * weight for score, weight in zip(sub_scores.values(), weights))
        
        # Research-based adjustments
        bonus_factors = {}
        if research_data and 'spatial_insights' in research_data.get('aggregated_findings', {}):
            spatial_insights = research_data['aggregated_findings']['spatial_insights']
            research_examples = spatial_insights.get('spatial_examples_analyzed', 0)
            if research_examples > 5:
                bonus_factors['research_validation'] = 0.05
                final_score += bonus_factors['research_validation']
        
        explanation = (
            f"Spatial efficiency: {final_score:.2f} "
            f"(utilization: {utilization_score:.2f}, compactness: {compactness_score:.2f}, "
            f"area eff: {area_efficiency_score:.2f})"
        )
        
        return ScoringResult(
            criterion=ScoringCriterion.SPATIAL_EFFICIENCY,
            score=min(1.0, final_score),
            confidence=0.85 if research_data else 0.75,
            explanation=explanation,
            sub_scores=sub_scores,
            bonus_factors=bonus_factors
        )
    
    def _score_code_compliance(self, 
                             prototype: Dict[str, Any], 
                             requirements: Dict[str, Any]) -> ScoringResult:
        """Score building code compliance"""
        min_height = self.building_standards['nbc_2016_standards']['min_ceiling_height']
        compliance_score = 0.0
        compliance_checks = 0
        violations = []
        
        # Room area compliance
        structures = prototype.get('detailed_config', {}).get('structures', {})
        if 'room_layouts' in structures:
            room_layouts = structures['room_layouts']
            
            for room_id, room in room_layouts.items():
                room_type = room.get('type', 'generic')
                room_area_sqft = room.get('area', 100)
                min_required = self.scoring_parameters['min_room_areas'].get(room_type, 80)
                
                compliance_checks += 1
                if room_area_sqft >= min_required:
                    compliance_score += 1
                else:
                    violations.append(f"{room_type} area {room_area_sqft} < {min_required} sqft")
        
        # Ceiling height compliance
        if 'architectural_elements' in structures:
            elements = structures['architectural_elements']
            structural_system = elements.get('structural_system', {})
            ceiling_height = structural_system.get('ceiling_height', 2.7)
            
            compliance_checks += 1
            min_height = self.building_standards['nbc_2016_standards']['min_ceiling_height']
            if ceiling_height >= min_height:
                compliance_score += 1
            else:
                violations.append(f"Ceiling height {ceiling_height}m < {min_height}m required")
        
        # Corridor width compliance
        circulation = prototype.get('detailed_config', {}).get('circulation_pattern', {})
        corridor_width = circulation.get('corridor_width', 1.2)
        
        compliance_checks += 1
        min_corridor = self.building_standards['nbc_2016_standards']['min_corridor_width']
        if corridor_width >= min_corridor:
            compliance_score += 1
        else:
            violations.append(f"Corridor width {corridor_width}m < {min_corridor}m required")
        
        # Plot coverage compliance
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        plot_utilization = spatial_config.get('plot_utilization', 0.7)
        
        compliance_checks += 1
        max_coverage = self.building_standards['karnataka_building_bylaws']['max_ground_coverage']
        if plot_utilization <= max_coverage:
            compliance_score += 1
        else:
            violations.append(f"Plot coverage {plot_utilization:.1%} > {max_coverage:.1%} allowed")
        
        # Window area compliance (simplified)
        window_compliance_score = self._check_window_compliance(prototype, requirements)
        compliance_score += window_compliance_score
        compliance_checks += 1
        
        # Calculate final compliance score
        final_score = compliance_score / max(1, compliance_checks) if compliance_checks > 0 else 0.5
        
        # Penalty for violations
        violation_penalty = len(violations) * 0.05
        final_score = max(0.0, final_score - violation_penalty)
        
        sub_scores = {
            'room_areas': (compliance_score - window_compliance_score) / max(1, compliance_checks - 1),
            'ceiling_height': 1.0 if structures.get('architectural_elements', {}).get('structural_system', {}).get('ceiling_height', 2.7) >= min_height else 0.0,
            'corridor_width': 1.0 if corridor_width >= min_corridor else 0.0,
            'plot_coverage': 1.0 if plot_utilization <= max_coverage else 0.0,
            'window_requirements': window_compliance_score
        }
        
        penalty_factors = {'violations': violation_penalty} if violations else {}
        
        explanation = (
            f"Code compliance: {final_score:.2f} "
            f"({len(violations)} violations, {compliance_checks} checks)"
        )
        
        return ScoringResult(
            criterion=ScoringCriterion.CODE_COMPLIANCE,
            score=final_score,
            confidence=0.90,  # High confidence for objective criteria
            explanation=explanation,
            sub_scores=sub_scores,
            penalty_factors=penalty_factors
        )
    
    def _check_window_compliance(self, prototype: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Check window area compliance for rooms"""
        
        structures = prototype.get('detailed_config', {}).get('structures', {})
        if 'room_layouts' not in structures:
            return 0.7  # Default assumption
        
        room_layouts = structures['room_layouts']
        compliance_count = 0
        total_checks = 0
        
        for room_id, room in room_layouts.items():
            room_type = room.get('type', 'generic')
            room_area = room.get('area', 100)
            
            min_window_ratio = self.scoring_parameters['min_window_ratios'].get(room_type, 0.08)
            
            # Get window information (if available)
            windows = room.get('windows', {})
            window_area_ratio = windows.get('area_ratio', min_window_ratio)
            
            total_checks += 1
            if window_area_ratio >= min_window_ratio:
                compliance_count += 1
        
        return compliance_count / max(1, total_checks)
    
    # Placeholder methods for other scoring criteria
    def _score_functional_organization(self, prototype, requirements, research_data=None):
        return ScoringResult(
            criterion=ScoringCriterion.FUNCTIONAL_ORGANIZATION,
            score=0.8,
            confidence=0.7,
            explanation="Functional organization scoring placeholder"
        )
    
    def _score_circulation_quality(self, prototype, requirements):
        return ScoringResult(
            criterion=ScoringCriterion.CIRCULATION_QUALITY,
            score=0.75,
            confidence=0.8,
            explanation="Circulation quality scoring placeholder"
        )
    
    def _score_environmental_performance(self, prototype, requirements, research_data=None):
        return ScoringResult(
            criterion=ScoringCriterion.ENVIRONMENTAL_PERFORMANCE,
            score=0.7,
            confidence=0.75,
            explanation="Environmental performance scoring placeholder"
        )
    
    def _score_cost_efficiency(self, prototype, requirements):
        return ScoringResult(
            criterion=ScoringCriterion.COST_EFFICIENCY,
            score=0.85,
            confidence=0.8,
            explanation="Cost efficiency scoring placeholder"
        )
    
    def _score_constructability(self, prototype, requirements):
        return ScoringResult(
            criterion=ScoringCriterion.CONSTRUCTABILITY,
            score=0.9,
            confidence=0.85,
            explanation="Constructability scoring placeholder"
        )
    
    def _score_aesthetic_quality(self, prototype, requirements):
        return ScoringResult(
            criterion=ScoringCriterion.AESTHETIC_QUALITY,
            score=0.65,
            confidence=0.6,
            explanation="Aesthetic quality scoring placeholder"
        )
    
    def _score_user_preference_alignment(self, prototype, requirements):
        return ScoringResult(
            criterion=ScoringCriterion.USER_PREFERENCE_ALIGNMENT,
            score=0.8,
            confidence=0.75,
            explanation="User preference alignment scoring placeholder"
        )
    
    def _calculate_diversity_bonuses(self, 
                                   comprehensive_scores: List[ComprehensiveScore], 
                                   prototypes: List[Dict[str, Any]]):
        """Calculate diversity bonuses for prototypes"""
        
        if len(comprehensive_scores) <= 1:
            return
        
        # Extract embeddings if available
        embeddings = []
        for prototype in prototypes:
            embedding = prototype.get('embedding')
            if embedding:
                embeddings.append(np.array(embedding))
            else:
                # Create simple embedding from key features
                embedding = self._create_simple_embedding(prototype)
                embeddings.append(embedding)
        
        if not embeddings:
            return
        
        embeddings = np.array(embeddings)
        
        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T) / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) * 
            np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
        )
        
        # Calculate diversity bonus for each prototype
        for i, score in enumerate(comprehensive_scores):
            # Diversity = inverse of maximum similarity to other prototypes
            other_similarities = similarities[i, :i].tolist() + similarities[i, i+1:].tolist()
            
            if other_similarities:
                max_similarity = max(other_similarities)
                diversity_bonus = 1.0 - max_similarity
                
                # Apply diminishing returns
                diversity_bonus = diversity_bonus ** 0.7
                
                score.diversity_bonus = diversity_bonus
            else:
                score.diversity_bonus = 1.0  # Only prototype gets full diversity bonus
    
    def _create_simple_embedding(self, prototype: Dict[str, Any]) -> np.ndarray:
        """Create simple embedding from prototype features"""
        
        features = []
        
        # Spatial strategy features
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        strategy = spatial_config.get('strategy', 'linear_progression')
        
        strategy_encoding = {
            'linear_progression': [1, 0, 0, 0],
            'central_core': [0, 1, 0, 0],
            'courtyard_focused': [0, 0, 1, 0],
            'functional_split': [0, 0, 0, 1]
        }
        features.extend(strategy_encoding.get(strategy, [0, 0, 0, 0]))
        
        # Functional zone ratios
        functional_zones = prototype.get('detailed_config', {}).get('functional_zones', {})
        public_ratio = functional_zones.get('public_zone', {}).get('ratio', 0.33)
        private_ratio = functional_zones.get('private_zone', {}).get('ratio', 0.33)
        service_ratio = functional_zones.get('service_zone', {}).get('ratio', 0.33)
        
        features.extend([public_ratio, private_ratio, service_ratio])
        
        # Circulation pattern
        circulation = prototype.get('detailed_config', {}).get('circulation_pattern', {})
        pattern_type = circulation.get('pattern_type', 'linear_spine')
        
        pattern_encoding = {
            'hub_and_spoke': [1, 0, 0, 0],
            'linear_spine': [0, 1, 0, 0],
            'perimeter_circulation': [0, 0, 1, 0],
            'dual_circulation': [0, 0, 0, 1]
        }
        features.extend(pattern_encoding.get(pattern_type, [0, 0, 0, 0]))
        
        # Environmental features
        env_strategy = prototype.get('detailed_config', {}).get('environmental_strategy', {})
        passive_strategies = env_strategy.get('passive_strategies', [])
        features.append(len(passive_strategies) / 5.0)  # Normalize
        
        # Plot utilization
        plot_utilization = spatial_config.get('plot_utilization', 0.7)
        features.append(plot_utilization)
        
        # Pad to fixed size
        target_size = 16
        while len(features) < target_size:
            features.append(0.0)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def _perform_pareto_analysis(self, comprehensive_scores: List[ComprehensiveScore]):
        """Perform Pareto efficiency analysis"""
        
        if len(comprehensive_scores) <= 1:
            return
        
        # Define key objectives for Pareto analysis
        objectives = [
            ScoringCriterion.SPATIAL_EFFICIENCY,
            ScoringCriterion.COST_EFFICIENCY,
            ScoringCriterion.ENVIRONMENTAL_PERFORMANCE,
            ScoringCriterion.FUNCTIONAL_ORGANIZATION
        ]
        
        # Extract objective scores
        objective_matrix = []
        for score in comprehensive_scores:
            objective_scores = [
                score.individual_scores.get(obj, ScoringResult(obj, 0.5, 0.5, "")).score 
                for obj in objectives
            ]
            objective_matrix.append(objective_scores)
        
        objective_matrix = np.array(objective_matrix)
        
        # Find Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal(objective_matrix)
        
        # Calculate Pareto efficiency metrics
        for i, score in enumerate(comprehensive_scores):
            if i in pareto_optimal:
                score.pareto_efficiency['is_pareto_optimal'] = True
                score.pareto_efficiency['pareto_rank'] = 1
                self.scoring_stats['pareto_optimal_count'] += 1
            else:
                score.pareto_efficiency['is_pareto_optimal'] = False
                # Calculate Pareto rank (dominated by how many solutions)
                domination_count = 0
                for j in pareto_optimal:
                    if self._dominates(objective_matrix[j], objective_matrix[i]):
                        domination_count += 1
                score.pareto_efficiency['pareto_rank'] = domination_count + 1
            
            # Calculate distance to Pareto front
            if not score.pareto_efficiency['is_pareto_optimal']:
                min_distance = float('inf')
                for j in pareto_optimal:
                    distance = np.linalg.norm(objective_matrix[i] - objective_matrix[j])
                    min_distance = min(min_distance, distance)
                score.pareto_efficiency['distance_to_pareto_front'] = min_distance
            else:
                score.pareto_efficiency['distance_to_pareto_front'] = 0.0
    
    def _find_pareto_optimal(self, objective_matrix: np.ndarray) -> List[int]:
        """Find Pareto optimal solutions"""
        
        n_solutions = objective_matrix.shape[0]
        pareto_optimal = []
        
        for i in range(n_solutions):
            is_dominated = False
            
            for j in range(n_solutions):
                if i != j and self._dominates(objective_matrix[j], objective_matrix[i]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(i)
        
        return pareto_optimal
    
    def _dominates(self, solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
        """Check if solution A dominates solution B"""
        
        better_or_equal = np.all(solution_a >= solution_b)
        strictly_better = np.any(solution_a > solution_b)
        
        return better_or_equal and strictly_better
    
    def _calculate_ranking_factors(self, 
                                 comprehensive_score: ComprehensiveScore, 
                                 requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional ranking factors"""
        
        ranking_factors = {}
        
        # Confidence-weighted score
        ranking_factors['confidence_weighted_score'] = (
            comprehensive_score.final_score * comprehensive_score.overall_confidence
        )
        
        # Risk assessment
        low_confidence_criteria = [
            criterion for criterion, result in comprehensive_score.individual_scores.items()
            if result.confidence < 0.6
        ]
        ranking_factors['risk_level'] = len(low_confidence_criteria) / len(ScoringCriterion)
        
        # Balanced performance
        criterion_scores = [result.score for result in comprehensive_score.individual_scores.values()]
        ranking_factors['performance_balance'] = 1.0 - np.std(criterion_scores)
        
        # Requirements satisfaction rate
        budget = requirements.get('budget', 0)
        if budget > 0:
            # Simplified cost estimation
            spatial_needs = requirements.get('spatial_needs', [])
            total_area = sum(need.get('min_area') or 100 for need in spatial_needs)
            estimated_cost = total_area * self.scoring_parameters['cost_per_sqm_base']
            
            ranking_factors['budget_fit'] = min(1.0, budget / max(estimated_cost, 1))
        else:
            ranking_factors['budget_fit'] = 1.0
        
        # Pareto efficiency consideration
        if comprehensive_score.pareto_efficiency.get('is_pareto_optimal', False):
            ranking_factors['pareto_bonus'] = 0.1
        else:
            ranking_factors['pareto_bonus'] = 0.0
        
        return ranking_factors
    
    def generate_scoring_report(self, 
                              comprehensive_scores: List[ComprehensiveScore],
                              requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive scoring report"""
        
        report = {
            'summary': {
                'total_prototypes': len(comprehensive_scores),
                'scoring_criteria': len(ScoringCriterion),
                'pareto_optimal_count': self.scoring_stats.get('pareto_optimal_count', 0),
                'avg_final_score': np.mean([score.final_score for score in comprehensive_scores]),
                'score_range': [
                    min(score.final_score for score in comprehensive_scores),
                    max(score.final_score for score in comprehensive_scores)
                ]
            },
            
            'top_performers': [],
            'criterion_analysis': {},
            'diversity_analysis': {},
            'pareto_analysis': {},
            'recommendations': []
        }
        
        # Top performers analysis
        for i, score in enumerate(comprehensive_scores[:5]):
            performer = {
                'rank': i + 1,
                'prototype_id': score.prototype_id,
                'final_score': score.final_score,
                'weighted_score': score.weighted_total,
                'diversity_bonus': score.diversity_bonus,
                'confidence': score.overall_confidence,
                'strengths': [],
                'weaknesses': []
            }
            
            # Identify strengths and weaknesses
            for criterion, result in score.individual_scores.items():
                if result.score >= 0.8:
                    performer['strengths'].append(criterion.value)
                elif result.score <= 0.5:
                    performer['weaknesses'].append(criterion.value)
            
            report['top_performers'].append(performer)
        
        # Criterion analysis
        for criterion in ScoringCriterion:
            criterion_scores = [
                score.individual_scores.get(criterion, ScoringResult(criterion, 0.5, 0.5, "")).score
                for score in comprehensive_scores
            ]
            
            report['criterion_analysis'][criterion.value] = {
                'avg_score': np.mean(criterion_scores),
                'std_score': np.std(criterion_scores),
                'min_score': min(criterion_scores),
                'max_score': max(criterion_scores),
                'top_performer': comprehensive_scores[np.argmax(criterion_scores)].prototype_id
            }
        
        return report
    
    def export_for_specializer(self, comprehensive_scores: List[ComprehensiveScore]) -> List[Dict[str, Any]]:
        """Export scored prototypes for Specializer processing"""
        
        exported_data = []
        
        for score in comprehensive_scores:
            prototype_data = {
                'prototype_id': score.prototype_id,
                'final_score': score.final_score,
                'weighted_total': score.weighted_total,
                'diversity_bonus': score.diversity_bonus,
                'overall_confidence': score.overall_confidence,
                
                # Individual criterion scores
                'criterion_scores': {
                    criterion.value: {
                        'score': result.score,
                        'confidence': result.confidence,
                        'explanation': result.explanation,
                        'sub_scores': result.sub_scores,
                        'bonus_factors': result.bonus_factors,
                        'penalty_factors': result.penalty_factors
                    }
                    for criterion, result in score.individual_scores.items()
                },
                
                # Ranking factors
                'ranking_factors': score.ranking_factors,
                
                # Pareto efficiency
                'pareto_efficiency': score.pareto_efficiency,
                
                # Aggregation metadata
                'aggregation_metadata': {
                    'ready_for_aggregation': True,
                    'scoring_timestamp': str(np.datetime64('now')),
                    'high_confidence_criteria': [
                        criterion.value for criterion, result in score.individual_scores.items()
                        if result.confidence >= 0.8
                    ],
                    'improvement_potential': [
                        criterion.value for criterion, result in score.individual_scores.items()
                        if result.score < 0.6
                    ]
                }
            }
            
            exported_data.append(prototype_data)
        
        return exported_data
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics for system monitoring"""
        
        return {
            'total_prototypes_scored': self.scoring_stats['total_prototypes_scored'],
            'avg_scoring_time': self.scoring_stats['avg_scoring_time'],
            'pareto_optimal_count': self.scoring_stats['pareto_optimal_count'],
            'score_distributions': {
                criterion: {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'count': len(scores)
                }
                for criterion, scores in self.scoring_stats['score_distributions'].items()
            },
            'weight_profiles_available': [profile.value for profile in self.weight_profiles.keys()],
            'scoring_criteria': [criterion.value for criterion in ScoringCriterion],
            'building_standards_loaded': len(self.building_standards),
            'cache_size': len(self.calculation_cache)
        }
    # Add to the MultiCriteriaScoringAgent class
    def check_performance_threshold(self, comprehensive_scores, threshold: float = 0.7) -> bool:
        """Check if scores meet performance threshold (for flowchart's N: Performance Threshold?)."""
        
        # Handle both ComprehensiveScore objects and dictionary formats
        final_scores = []
        
        for score in comprehensive_scores:
            if isinstance(score, ComprehensiveScore):
                # It's a ComprehensiveScore object
                final_scores.append(score.final_score)
            elif isinstance(score, dict):
                if 'comprehensive_score' in score:
                    # It's a scored prototype dictionary
                    final_scores.append(score['comprehensive_score']['final_score'])
                elif 'final_score' in score:
                    # It's already a comprehensive score dictionary
                    final_scores.append(score['final_score'])
                else:
                    # Fallback - use weighted_total or default
                    final_scores.append(score.get('weighted_total', 0.7))
            else:
                # Unknown format, use default
                final_scores.append(0.7)
        
        if not final_scores:
            return False
            
        avg_final_score = np.mean(final_scores)
        logger.info(f"Performance threshold check: avg_score={avg_final_score:.3f}, threshold={threshold}")
        return avg_final_score >= threshold


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Initializing Multi-Criteria Scoring Agent...")
    
    # Initialize scoring agent
    scoring_agent = MultiCriteriaScoringAgent(
        default_weight_profile=ScoringWeightProfile.BALANCED,
        diversity_weight=0.1,
        pareto_analysis=True
    )
    
    # Example prototypes from Generalizer + Research Agent
    sample_prototypes = [
        {
            'id': 'compact_central_zonal_0',
            'hierarchy_level': 2,
            'scores': {'total_score': 0.82},
            'detailed_config': {
                'spatial_config': {
                    'strategy': 'central_core',
                    'plot_utilization': 0.75,
                    'compactness_factor': 0.8
                },
                'functional_zones': {
                    'public_zone': {'ratio': 0.4, 'rooms': ['living_room', 'kitchen']},
                    'private_zone': {'ratio': 0.4, 'rooms': ['bedroom', 'bathroom']},
                    'service_zone': {'ratio': 0.2, 'rooms': ['utility']}
                },
                'circulation_pattern': {
                    'pattern_type': 'hub_and_spoke',
                    'efficiency_target': 0.85,
                    'corridor_width': 1.2,
                    'accessibility': False
                },
                'environmental_strategy': {
                    'orientation': 'south',
                    'passive_strategies': ['cross_ventilation', 'south_shading'],
                    'climate_zone': 'subtropical'
                },
                'structures': {
                    'room_layouts': {
                        'bedroom_1': {'type': 'bedroom', 'area': 120, 'aspect_ratio': 1.3},
                        'living_room': {'type': 'living_room', 'area': 200, 'aspect_ratio': 1.5},
                        'kitchen': {'type': 'kitchen', 'area': 100, 'aspect_ratio': 1.8}
                    },
                    'architectural_elements': {
                        'structural_system': {
                            'wall_type': 'masonry',
                            'ceiling_height': 2.7
                        },
                        'finishes': {
                            'quality_level': 'standard'
                        }
                    }
                }
            },
            'research_metadata': {
                'research_conducted': True,
                'avg_relevance': 0.65
            },
            'embedding': [0.1, 0.9, 0.2, 0.6, 0.3, 0.7, 0.8, 0.2, 0.0, 0.8, 0.3, 0.5, 0.7, 0.9, 0.2, 0.6]
        }
    ]
    
    # Example requirements
    sample_requirements = {
        'spatial_needs': [
            {'room_type': 'bedroom', 'quantity': 2, 'min_area': 120},
            {'room_type': 'bathroom', 'quantity': 2, 'min_area': 45},
            {'room_type': 'living_room', 'quantity': 1, 'min_area': 180},
            {'room_type': 'kitchen', 'quantity': 1, 'min_area': 90}
        ],
        'site_constraints': {
            'plot_length': 50,
            'plot_width': 30,
            'orientation': 'south'
        },
        'design_preferences': {
            'style': 'modern',
            'sustainability_focus': True,
            'space_optimization': True
        },
        'budget': 2800000
    }
    
    # Perform scoring
    print("🎯 Scoring prototypes...")
    comprehensive_scores = scoring_agent.score_prototypes(
        prototypes=sample_prototypes,
        requirements=sample_requirements
    )
    
    print(f"✅ Scoring complete! Top score: {comprehensive_scores[0].final_score:.3f}")
    print(f"📊 Generated report and export data ready for Specializer")