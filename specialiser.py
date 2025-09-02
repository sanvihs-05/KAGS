import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import math
from abc import ABC, abstractmethod
from itertools import combinations
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from encoder import Gemma3Encoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PruningStrategy(Enum):
    """Different pruning strategies"""
    SIMILARITY_BASED = "similarity_based"
    PARETO_DOMINANCE = "pareto_dominance"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    DIVERSITY_PRESERVING = "diversity_preserving"
    HYBRID_INTELLIGENT = "hybrid_intelligent"

class AggregationMethod(Enum):
    """Different aggregation methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    PARETO_OPTIMAL_FUSION = "pareto_optimal_fusion"
    HIERARCHICAL_MERGING = "hierarchical_merging"
    ADAPTIVE_CLUSTERING = "adaptive_clustering"
    FEATURE_INTERPOLATION = "feature_interpolation"

@dataclass
class PruningResult:
    """Result of pruning operation"""
    original_count: int
    pruned_count: int
    retained_prototypes: List[str]
    pruning_rationale: Dict[str, Any]
    diversity_preserved: float
    quality_maintained: float

@dataclass
class AggregationResult:
    """Result of aggregation operation"""
    source_prototypes: List[str]
    aggregated_prototype: Dict[str, Any]
    aggregation_confidence: float
    feature_sources: Dict[str, List[str]]
    performance_prediction: Dict[str, float]
    variant_potential: float

@dataclass
class SpecializationOutput:
    """Final output from Specializer"""
    final_prototypes: List[Dict[str, Any]]
    pruning_summary: PruningResult
    aggregation_summary: List[AggregationResult]
    specialization_metadata: Dict[str, Any]
    performance_guarantees: Dict[str, float]
    recommendation_ranking: List[Tuple[str, float, str]]

class PrototypeSpecializer:
    """
    Enhanced Prototype Specializer for GOT-RAG-FBS system
    Handles intelligent pruning and aggregation of scored prototypes
    """
    
    def __init__(self,
                 pruning_strategies: List[PruningStrategy] = None,
                 aggregation_methods: List[AggregationMethod] = None,
                 diversity_weight: float = 0.3,
                 quality_threshold: float = 0.6,
                 max_final_prototypes: int = 5):
        
        self.pruning_strategies = pruning_strategies or [
            PruningStrategy.HYBRID_INTELLIGENT,
            PruningStrategy.DIVERSITY_PRESERVING
        ]
        
        self.aggregation_methods = aggregation_methods or [
            AggregationMethod.ADAPTIVE_CLUSTERING,
            AggregationMethod.PARETO_OPTIMAL_FUSION
        ]
        
        self.diversity_weight = diversity_weight
        self.quality_threshold = quality_threshold
        self.max_final_prototypes = max_final_prototypes
        
        # Specialization parameters
        self.specialization_params = self._initialize_specialization_params()
        
        # Performance tracking
        self.specialization_stats = {
            'total_specializations': 0,
            'avg_pruning_ratio': 0.0,
            'avg_aggregation_success': 0.0,
            'diversity_preservation_rate': 0.0,
            'quality_improvement_rate': 0.0
        }
        
        # Caches for expensive operations
        self.similarity_cache = {}
        self.feature_cache = {}
        self.encoder = Gemma3Encoder()
        logger.info("✅ Initialized Gemma 3 Encoder for specialization")
    
    def _initialize_specialization_params(self) -> Dict[str, Any]:
        """Initialize specialization parameters"""
        
        return {
            # Pruning parameters
            'similarity_threshold': 0.85,
            'performance_percentile_cutoff': 0.3,  # Keep top 70%
            'diversity_cluster_min_distance': 0.4,
            'pareto_dominance_tolerance': 0.02,
            
            # Aggregation parameters
            'min_prototypes_for_aggregation': 2,
            'max_prototypes_per_aggregate': 4,
            'feature_weight_balance': 0.6,  # Balance between best features
            'interpolation_smoothing': 0.2,
            
            # Quality thresholds
            'min_acceptable_score': 0.5,
            'min_confidence_level': 0.6,
            'max_complexity_penalty': 0.15,
            
            # Clustering parameters
            'dbscan_eps': 0.3,
            'dbscan_min_samples': 2,
            'hierarchical_distance_threshold': 0.5,
            
            # Feature importance weights
            'spatial_weight': 0.25,
            'functional_weight': 0.25,
            'environmental_weight': 0.20,
            'cost_weight': 0.15,
            'aesthetic_weight': 0.10,
            'compliance_weight': 0.05
        }
    
    def specialize_prototypes(self,
                            scored_prototypes: List[Dict[str, Any]],
                            requirements: Dict[str, Any],
                            research_data: Optional[Dict[str, Any]] = None) -> SpecializationOutput:
        """
        Main specialization method - prune and aggregate prototypes
        
        Args:
            scored_prototypes: Prototypes with comprehensive scores from Scoring Agent
            requirements: Original user requirements
            research_data: Research findings from Research Agent
        
        Returns:
            SpecializationOutput with final optimized prototypes
        """
        
        logger.info(f"Specializing {len(scored_prototypes)} scored prototypes")
        
        if not scored_prototypes:
            return self._create_empty_output()
        
        # Step 1: Extract and prepare prototype features
        prototype_features = self._extract_prototype_features(scored_prototypes)
        
        # Step 2: Intelligent pruning
        pruning_result = self._prune_prototypes(
            scored_prototypes, prototype_features, requirements
        )
        
        retained_prototypes = [
            p for p in scored_prototypes 
            if p['prototype_id'] in pruning_result.retained_prototypes
        ]
        
        logger.info(f"Pruning: {len(scored_prototypes)} → {len(retained_prototypes)} prototypes")
        
        # Step 3: Smart aggregation
        aggregation_results = self._aggregate_prototypes(
            retained_prototypes, prototype_features, requirements, research_data
        )
        
        # Step 4: Generate final prototype set
        final_prototypes = self._generate_final_prototypes(
            retained_prototypes, aggregation_results, requirements
        )
        
        # Step 5: Create performance guarantees and rankings
        performance_guarantees = self._calculate_performance_guarantees(final_prototypes)
        recommendation_ranking = self._generate_recommendation_ranking(
            final_prototypes, requirements
        )
        
        # Update statistics
        self._update_specialization_stats(
            len(scored_prototypes), len(final_prototypes), 
            pruning_result, aggregation_results
        )
        
        # Create specialization metadata
        specialization_metadata = self._create_specialization_metadata(
            scored_prototypes, final_prototypes, pruning_result, aggregation_results
        )
        
        output = SpecializationOutput(
            final_prototypes=final_prototypes,
            pruning_summary=pruning_result,
            aggregation_summary=aggregation_results,
            specialization_metadata=specialization_metadata,
            performance_guarantees=performance_guarantees,
            recommendation_ranking=recommendation_ranking
        )
        
        logger.info(f"Specialization complete: {len(final_prototypes)} final prototypes")
        
        return output
    
    def _extract_prototype_features(self, scored_prototypes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract sophisticated features using Gemma 3 encoder"""
        feature_vectors = []
        prototype_ids = []

        for prototype in scored_prototypes:
            prototype_id = prototype['prototype_id']

            # Use cache if available
            if prototype_id in self.feature_cache:
                feature_vectors.append(self.feature_cache[prototype_id])
                prototype_ids.append(prototype_id)
                continue

            try:
                # Use Gemma 3 for sophisticated feature extraction
                embedding = self.encoder.encode_prototype_features(prototype)

                # Cache the sophisticated embedding
                self.feature_cache[prototype_id] = embedding
                feature_vectors.append(embedding)
                prototype_ids.append(prototype_id)

                logger.debug(f"Generated Gemma 3 features for {prototype_id}: {embedding.shape}")

            except Exception as e:
                logger.warning(f"Gemma 3 feature extraction failed for {prototype_id}: {e}")

                # Fallback to simple features
                simple_embedding = self._create_simple_embedding(prototype)
                self.feature_cache[prototype_id] = simple_embedding
                feature_vectors.append(simple_embedding)
                prototype_ids.append(prototype_id)

        # Convert to numpy array and normalize
        if feature_vectors:
            features_matrix = np.vstack(feature_vectors)
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_matrix)

            return {
                'features': features_normalized,
                'prototype_ids': prototype_ids,
                'scaler': scaler,
                'raw_features': features_matrix,
                'embedding_source': 'gemma3_enhanced'
            }

        return {'features': np.array([]), 'prototype_ids': []}
    
    def _prune_prototypes(self,
                         scored_prototypes: List[Dict[str, Any]],
                         prototype_features: Dict[str, Any],
                         requirements: Dict[str, Any]) -> PruningResult:
        """Intelligent pruning of prototypes"""
        
        original_count = len(scored_prototypes)
        retained_prototypes = set()
        pruning_rationale = defaultdict(list)
        
        # Get features
        features = prototype_features.get('features', np.array([]))
        prototype_ids = prototype_features.get('prototype_ids', [])
        
        if len(features) == 0:
            return PruningResult(
                original_count=original_count,
                pruned_count=0,
                retained_prototypes=[p['prototype_id'] for p in scored_prototypes],
                pruning_rationale={},
                diversity_preserved=1.0,
                quality_maintained=1.0
            )
        
        # Strategy 1: Performance threshold pruning
        performance_retained = self._prune_by_performance_threshold(
            scored_prototypes, requirements
        )
        
        for proto_id in performance_retained:
            retained_prototypes.add(proto_id)
            pruning_rationale['performance_threshold'].append(proto_id)
        
        # Strategy 2: Diversity-preserving pruning
        if len(performance_retained) > self.max_final_prototypes * 2:
            diversity_retained = self._prune_by_diversity_preservation(
                [p for p in scored_prototypes if p['prototype_id'] in performance_retained],
                features, prototype_ids
            )
            
            # Update retained set
            retained_prototypes = set(diversity_retained)
            pruning_rationale['diversity_preservation'] = diversity_retained
        
        # Strategy 3: Similarity-based pruning
        if len(retained_prototypes) > self.max_final_prototypes:
            similarity_retained = self._prune_by_similarity(
                [p for p in scored_prototypes if p['prototype_id'] in retained_prototypes],
                features, prototype_ids
            )
            
            retained_prototypes = set(similarity_retained)
            pruning_rationale['similarity_based'] = similarity_retained
        
        # Strategy 4: Pareto dominance pruning (preserve Pareto optimal)
        pareto_optimal = self._identify_pareto_optimal(
            [p for p in scored_prototypes if p['prototype_id'] in retained_prototypes]
        )
        
        # Ensure all Pareto optimal solutions are retained
        for proto_id in pareto_optimal:
            retained_prototypes.add(proto_id)
            pruning_rationale['pareto_preservation'].append(proto_id)
        
        # Final selection if still too many
        if len(retained_prototypes) > self.max_final_prototypes:
            final_retained = self._final_selection(
                [p for p in scored_prototypes if p['prototype_id'] in retained_prototypes],
                self.max_final_prototypes
            )
            retained_prototypes = set(final_retained)
            pruning_rationale['final_selection'] = final_retained
        
        # Calculate diversity and quality metrics
        diversity_preserved = self._calculate_diversity_preservation(
            scored_prototypes, retained_prototypes, features, prototype_ids
        )
        
        quality_maintained = self._calculate_quality_maintenance(
            scored_prototypes, retained_prototypes
        )
        
        return PruningResult(
            original_count=original_count,
            pruned_count=original_count - len(retained_prototypes),
            retained_prototypes=list(retained_prototypes),
            pruning_rationale=dict(pruning_rationale),
            diversity_preserved=diversity_preserved,
            quality_maintained=quality_maintained
        )
    
    def _prune_by_performance_threshold(self,
                                      scored_prototypes: List[Dict[str, Any]],
                                      requirements: Dict[str, Any]) -> List[str]:
        """Prune based on performance thresholds"""
        
        # Calculate dynamic threshold based on score distribution
        final_scores = [p['final_score'] for p in scored_prototypes]
        
        if not final_scores:
            return []
        
        # Use percentile-based threshold
        cutoff_percentile = self.specialization_params['performance_percentile_cutoff']
        threshold = np.percentile(final_scores, cutoff_percentile * 100)
        
        # Ensure minimum quality
        threshold = max(threshold, self.specialization_params['min_acceptable_score'])
        
        retained = []
        for prototype in scored_prototypes:
            final_score = prototype['final_score']
            overall_confidence = prototype['overall_confidence']
            
            # Multi-criteria retention check
            if (final_score >= threshold and 
                overall_confidence >= self.specialization_params['min_confidence_level']):
                retained.append(prototype['prototype_id'])
        
        # Ensure at least minimum prototypes are retained
        if len(retained) < 2 and len(scored_prototypes) >= 2:
            # Keep top 2 by final score
            sorted_prototypes = sorted(scored_prototypes, 
                                     key=lambda x: x['final_score'], reverse=True)
            retained = [p['prototype_id'] for p in sorted_prototypes[:2]]
        
        return retained
    
    def _prune_by_diversity_preservation(self,
                                       prototypes: List[Dict[str, Any]],
                                       features: np.ndarray,
                                       prototype_ids: List[str]) -> List[str]:
        """Preserve diversity while pruning"""
        
        if len(prototypes) <= self.max_final_prototypes:
            return [p['prototype_id'] for p in prototypes]
        
        # Create mapping from prototype_id to feature index
        id_to_idx = {proto_id: idx for idx, proto_id in enumerate(prototype_ids)}
        
        # Get features for current prototypes
        current_indices = [id_to_idx[p['prototype_id']] 
                          for p in prototypes if p['prototype_id'] in id_to_idx]
        
        if not current_indices:
            return [p['prototype_id'] for p in prototypes[:self.max_final_prototypes]]
        
        current_features = features[current_indices]
        
        # Use DBSCAN clustering to identify diverse groups
        dbscan = DBSCAN(
            eps=self.specialization_params['dbscan_eps'],
            min_samples=self.specialization_params['dbscan_min_samples']
        )
        
        clusters = dbscan.fit_predict(current_features)
        
        # Select representative from each cluster
        retained = []
        cluster_representatives = {}
        
        # Handle clustered points
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_prototypes = [prototypes[i] for i in cluster_indices]
            
            # Select best prototype from cluster
            best_in_cluster = max(cluster_prototypes, key=lambda x: x['final_score'])
            retained.append(best_in_cluster['prototype_id'])
            cluster_representatives[cluster_id] = best_in_cluster
        
        # Handle noise points (unclustered)
        noise_indices = [i for i, c in enumerate(clusters) if c == -1]
        noise_prototypes = [prototypes[i] for i in noise_indices]
        
        # Select top noise points by score
        noise_prototypes.sort(key=lambda x: x['final_score'], reverse=True)
        noise_to_keep = min(len(noise_prototypes), 
                          self.max_final_prototypes - len(retained))
        
        for prototype in noise_prototypes[:noise_to_keep]:
            retained.append(prototype['prototype_id'])
        
        # If still not enough, add more from best clusters
        if len(retained) < self.max_final_prototypes:
            remaining_slots = self.max_final_prototypes - len(retained)
            
            # Sort clusters by quality of their representatives
            cluster_quality = [(cluster_id, rep['final_score']) 
                             for cluster_id, rep in cluster_representatives.items()]
            cluster_quality.sort(key=lambda x: x[1], reverse=True)
            
            for cluster_id, _ in cluster_quality:
                if remaining_slots <= 0:
                    break
                
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                cluster_prototypes = [prototypes[i] for i in cluster_indices]
                
                # Add second best from this cluster if available
                cluster_prototypes.sort(key=lambda x: x['final_score'], reverse=True)
                
                for prototype in cluster_prototypes[1:]:  # Skip first (already added)
                    if remaining_slots <= 0:
                        break
                    if prototype['prototype_id'] not in retained:
                        retained.append(prototype['prototype_id'])
                        remaining_slots -= 1
        
        return retained[:self.max_final_prototypes]
    
    def _prune_by_similarity(self,
                           prototypes: List[Dict[str, Any]],
                           features: np.ndarray,
                           prototype_ids: List[str]) -> List[str]:
        """Remove highly similar prototypes"""
        
        if len(prototypes) <= 2:
            return [p['prototype_id'] for p in prototypes]
        
        # Create similarity matrix
        similarity_threshold = self.specialization_params['similarity_threshold']
        
        # Create mapping
        id_to_idx = {proto_id: idx for idx, proto_id in enumerate(prototype_ids)}
        current_indices = [id_to_idx[p['prototype_id']] 
                          for p in prototypes if p['prototype_id'] in id_to_idx]
        
        if not current_indices:
            return [p['prototype_id'] for p in prototypes]
        
        current_features = features[current_indices]
        
        # Calculate pairwise similarities
        similarities = np.dot(current_features, current_features.T)
        
        # Find pairs with high similarity
        retained_indices = list(range(len(prototypes)))
        
        for i in range(len(prototypes)):
            if i not in retained_indices:
                continue
                
            for j in range(i + 1, len(prototypes)):
                if j not in retained_indices:
                    continue
                
                similarity = similarities[i, j]
                
                if similarity > similarity_threshold:
                    # Keep the one with higher score
                    if prototypes[i]['final_score'] >= prototypes[j]['final_score']:
                        if j in retained_indices:
                            retained_indices.remove(j)
                    else:
                        if i in retained_indices:
                            retained_indices.remove(i)
                        break  # Move to next i
        
        return [prototypes[i]['prototype_id'] for i in retained_indices]
    
    def _identify_pareto_optimal(self, prototypes: List[Dict[str, Any]]) -> List[str]:
        """Identify Pareto optimal prototypes"""
        
        pareto_optimal = []
        
        for prototype in prototypes:
            pareto_efficiency = prototype.get('pareto_efficiency', {})
            if pareto_efficiency.get('is_pareto_optimal', False):
                pareto_optimal.append(prototype['prototype_id'])
        
        return pareto_optimal
    
    def _final_selection(self,
                        prototypes: List[Dict[str, Any]],
                        max_count: int) -> List[str]:
        """Final selection when all other strategies leave too many prototypes"""
        
        # Multi-criteria selection combining score, diversity, and confidence
        
        # Sort by composite score
        def composite_score(p):
            final_score = p['final_score']
            confidence = p['overall_confidence']
            diversity_bonus = p.get('diversity_bonus', 0.0)
            
            return (final_score * 0.6 + 
                   confidence * 0.2 + 
                   diversity_bonus * 0.2)
        
        prototypes.sort(key=composite_score, reverse=True)
        
        return [p['prototype_id'] for p in prototypes[:max_count]]
    
    def _aggregate_prototypes(self,
                            retained_prototypes: List[Dict[str, Any]],
                            prototype_features: Dict[str, Any],
                            requirements: Dict[str, Any],
                            research_data: Optional[Dict[str, Any]] = None) -> List[AggregationResult]:
        """Aggregate compatible prototypes"""
        
        aggregation_results = []
        
        if len(retained_prototypes) < self.specialization_params['min_prototypes_for_aggregation']:
            logger.info("Not enough prototypes for aggregation")
            return aggregation_results
        
        # Method 1: Adaptive clustering aggregation
        cluster_results = self._aggregate_by_clustering(
            retained_prototypes, prototype_features, requirements
        )
        aggregation_results.extend(cluster_results)
        
        # Method 2: Pareto optimal fusion
        pareto_results = self._aggregate_pareto_optimal(
            retained_prototypes, requirements
        )
        aggregation_results.extend(pareto_results)
        
        # Method 3: Feature interpolation for similar high-performers
        interpolation_results = self._aggregate_by_interpolation(
            retained_prototypes, requirements
        )
        aggregation_results.extend(interpolation_results)
        
        logger.info(f"Generated {len(aggregation_results)} aggregation candidates")
        
        return aggregation_results
    
    def _aggregate_by_clustering(self,
                               prototypes: List[Dict[str, Any]],
                               prototype_features: Dict[str, Any],
                               requirements: Dict[str, Any]) -> List[AggregationResult]:
        """Aggregate prototypes based on feature clustering"""
        
        results = []
        
        features = prototype_features.get('features', np.array([]))
        prototype_ids = prototype_features.get('prototype_ids', [])
        
        if len(features) == 0 or len(prototypes) < 3:
            return results
        
        # Create mapping
        id_to_idx = {proto_id: idx for idx, proto_id in enumerate(prototype_ids)}
        current_indices = [id_to_idx[p['prototype_id']] 
                          for p in prototypes if p['prototype_id'] in id_to_idx]
        
        if len(current_indices) < 3:
            return results
        
        current_features = features[current_indices]
        
        # Hierarchical clustering
        try:
            linkage_matrix = sch.linkage(current_features, method='ward')
            
            threshold = self.specialization_params['hierarchical_distance_threshold']
            cluster_labels = sch.fcluster(linkage_matrix, threshold, criterion='distance')
            
            # Group prototypes by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(prototypes[i])
            
            # Aggregate each cluster with multiple prototypes
            for cluster_id, cluster_prototypes in clusters.items():
                if (len(cluster_prototypes) >= self.specialization_params['min_prototypes_for_aggregation'] and
                    len(cluster_prototypes) <= self.specialization_params['max_prototypes_per_aggregate']):
                    
                    aggregated = self._create_aggregated_prototype(
                        cluster_prototypes, "hierarchical_cluster", requirements
                    )
                    
                    if aggregated:
                        results.append(aggregated)
        
        except Exception as e:
            logger.warning(f"Clustering aggregation failed: {e}")
        
        return results
    
    def _aggregate_pareto_optimal(self,
                                prototypes: List[Dict[str, Any]],
                                requirements: Dict[str, Any]) -> List[AggregationResult]:
        """Aggregate Pareto optimal prototypes"""
        
        results = []
        
        # Find Pareto optimal prototypes
        pareto_optimal = [p for p in prototypes 
                         if p.get('pareto_efficiency', {}).get('is_pareto_optimal', False)]
        
        if len(pareto_optimal) < 2:
            return results
        
        # Try to create a fusion of Pareto optimal solutions
        if len(pareto_optimal) <= self.specialization_params['max_prototypes_per_aggregate']:
            aggregated = self._create_aggregated_prototype(
                pareto_optimal, "pareto_fusion", requirements
            )
            
            if aggregated:
                results.append(aggregated)
        
        return results
    
    def _aggregate_by_interpolation(self,
                                  prototypes: List[Dict[str, Any]],
                                  requirements: Dict[str, Any]) -> List[AggregationResult]:
        """Aggregate by interpolating between similar high-performance prototypes"""
        
        results = []
        
        # Find high-performance prototypes
        high_performers = [p for p in prototypes if p['final_score'] > 0.75]
        
        if len(high_performers) < 2:
            return results
        
        # Find pairs with complementary strengths
        for i in range(len(high_performers)):
            for j in range(i + 1, len(high_performers)):
                proto1, proto2 = high_performers[i], high_performers[j]
                
                if self._are_complementary(proto1, proto2):
                    aggregated = self._create_interpolated_prototype(
                        [proto1, proto2], requirements
                    )
                    
                    if aggregated:
                        results.append(aggregated)
        
        return results[:2]  # Limit interpolation results
    
    def _are_complementary(self, proto1: Dict[str, Any], proto2: Dict[str, Any]) -> bool:
        """Check if two prototypes have complementary strengths"""
        
        scores1 = proto1.get('criterion_scores', {})
        scores2 = proto2.get('criterion_scores', {})
        
        complementarity_count = 0
        total_criteria = 0
        
        for criterion in scores1:
            if criterion in scores2:
                score1 = scores1[criterion].get('score', 0.5)
                score2 = scores2[criterion].get('score', 0.5)
                
                total_criteria += 1
                
                # Check if one is strong where the other is weak
                if (score1 > 0.7 and score2 < 0.6) or (score2 > 0.7 and score1 < 0.6):
                    complementarity_count += 1
        
        complementarity_ratio = complementarity_count / max(total_criteria, 1)
        return complementarity_ratio > 0.3  # At least 30% complementarity
    
    def _create_aggregated_prototype(self,
                                   source_prototypes: List[Dict[str, Any]],
                                   aggregation_type: str,
                                   requirements: Dict[str, Any]) -> Optional[AggregationResult]:
        """Create aggregated prototype from source prototypes"""
        
        if len(source_prototypes) < 2:
            return None
        
        source_ids = [p['prototype_id'] for p in source_prototypes]
        
        # Weighted averaging based on final scores
        weights = np.array([p['final_score'] for p in source_prototypes])
        weights = weights / np.sum(weights)  # Normalize
        
        # Aggregate criterion scores
        aggregated_criterion_scores = {}
        feature_sources = defaultdict(list)
        
        # Get all unique criteria from source prototypes
        all_criteria = set()
        for prototype in source_prototypes:
            all_criteria.update(prototype.get('criterion_scores', {}).keys())
        
        # Aggregate each criterion
        for criterion in all_criteria:
            criterion_scores = []
            criterion_confidences = []
            
            for i, prototype in enumerate(source_prototypes):
                criterion_data = prototype.get('criterion_scores', {}).get(criterion, {})
                score = criterion_data.get('score', 0.5)
                confidence = criterion_data.get('confidence', 0.5)
                
                criterion_scores.append(score)
                criterion_confidences.append(confidence)
                
                # Track which prototype contributed to this feature
                if score > 0.7:  # High-performing feature
                    feature_sources[criterion].append(prototype['prototype_id'])
            
            # Weighted aggregation
            if criterion_scores:
                aggregated_score = np.average(criterion_scores, weights=weights)
                aggregated_confidence = np.average(criterion_confidences, weights=weights)
                
                # Apply aggregation bonus for consistent high performance
                consistency_bonus = 1.0 - np.std(criterion_scores)
                aggregated_score += consistency_bonus * 0.05
                
                aggregated_criterion_scores[criterion] = {
                    'score': min(1.0, aggregated_score),
                    'confidence': aggregated_confidence,
                    'explanation': f"Aggregated from {len(source_prototypes)} prototypes using {aggregation_type}",
                    'source_prototypes': source_ids,
                    'aggregation_method': aggregation_type
                }
        
        # Aggregate overall scores
        final_scores = [p['final_score'] for p in source_prototypes]
        confidence_scores = [p['overall_confidence'] for p in source_prototypes]
        
        aggregated_final_score = np.average(final_scores, weights=weights)
        aggregated_confidence = np.average(confidence_scores, weights=weights)
        
        # Apply aggregation bonus
        aggregation_bonus = 0.05 * (1.0 - np.std(final_scores))
        aggregated_final_score += aggregation_bonus
        
        # Create aggregated prototype
        aggregated_prototype_id = f"aggregated_{aggregation_type}_{hash(tuple(source_ids)) % 10000}"
        
        aggregated_prototype = {
            'prototype_id': aggregated_prototype_id,
            'aggregation_metadata': {
                'is_aggregated': True,
                'aggregation_type': aggregation_type,
                'source_prototypes': source_ids,
                'source_count': len(source_prototypes),
                'aggregation_weights': weights.tolist(),
                'aggregation_timestamp': str(np.datetime64('now'))
            },
            'criterion_scores': aggregated_criterion_scores,
            'final_score': min(1.0, aggregated_final_score),
            'overall_confidence': aggregated_confidence,
            'weighted_total': np.average([p['weighted_total'] for p in source_prototypes], weights=weights),
            'diversity_bonus': np.average([p.get('diversity_bonus', 0.0) for p in source_prototypes], weights=weights),
            'pareto_efficiency': {
                'is_pareto_optimal': any(p.get('pareto_efficiency', {}).get('is_pareto_optimal', False) 
                                       for p in source_prototypes),
                'aggregated_pareto_rank': np.average([p.get('pareto_efficiency', {}).get('pareto_rank', 5.0) 
                                                    for p in source_prototypes], weights=weights)
            }
        }
        
        # Predict performance of aggregated prototype
        performance_prediction = self._predict_aggregated_performance(
            source_prototypes, aggregated_prototype, requirements
        )
        
        # Calculate variant potential
        variant_potential = self._calculate_variant_potential(
            source_prototypes, aggregated_prototype
        )
        
        return AggregationResult(
            source_prototypes=source_ids,
            aggregated_prototype=aggregated_prototype,
            aggregation_confidence=aggregated_confidence,
            feature_sources=dict(feature_sources),
            performance_prediction=performance_prediction,
            variant_potential=variant_potential
        )
    
    def _create_interpolated_prototype(self,
                                     source_prototypes: List[Dict[str, Any]],
                                     requirements: Dict[str, Any]) -> Optional[AggregationResult]:
        """Create prototype by interpolating between source prototypes"""
        
        if len(source_prototypes) != 2:
            return None
        
        proto1, proto2 = source_prototypes
        
        # Calculate interpolation weights based on performance balance
        score1 = proto1['final_score']
        score2 = proto2['final_score']
        
        # Use performance-based weighting with smoothing
        smoothing = self.specialization_params['interpolation_smoothing']
        weight1 = (score1 + smoothing) / (score1 + score2 + 2 * smoothing)
        weight2 = 1.0 - weight1
        
        weights = np.array([weight1, weight2])
        
        # Create interpolated prototype
        aggregated = self._create_aggregated_prototype(
            source_prototypes, "interpolation", requirements
        )
        
        if aggregated:
            # Modify for interpolation-specific properties
            aggregated.aggregated_prototype['aggregation_metadata']['interpolation_weights'] = {
                proto1['prototype_id']: weight1,
                proto2['prototype_id']: weight2
            }
            
            # Boost interpolation confidence for complementary prototypes
            if self._are_complementary(proto1, proto2):
                aggregated.aggregation_confidence = min(1.0, aggregated.aggregation_confidence + 0.1)
        
        return aggregated
    
    def _predict_aggregated_performance(self,
                                      source_prototypes: List[Dict[str, Any]],
                                      aggregated_prototype: Dict[str, Any],
                                      requirements: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance of aggregated prototype"""
        
        predictions = {}
        
        # Performance prediction based on source prototype analysis
        source_scores = [p['final_score'] for p in source_prototypes]
        source_confidences = [p['overall_confidence'] for p in source_prototypes]
        
        # Predicted final score (conservative estimate)
        predicted_score = np.mean(source_scores) + (np.std(source_scores) * 0.1)  # Slight optimism
        predictions['predicted_final_score'] = min(1.0, predicted_score)
        
        # Confidence prediction
        predicted_confidence = np.mean(source_confidences) * 0.95  # Slight penalty for aggregation uncertainty
        predictions['predicted_confidence'] = predicted_confidence
        
        # Risk assessment
        score_variance = np.var(source_scores)
        predictions['performance_risk'] = min(1.0, score_variance * 2.0)
        
        # Improvement potential over best source
        best_source_score = max(source_scores)
        improvement_potential = max(0.0, predicted_score - best_source_score)
        predictions['improvement_potential'] = improvement_potential
        
        # Requirements satisfaction prediction
        satisfaction_scores = []
        for prototype in source_prototypes:
            # Simplified requirements satisfaction estimation
            ranking_factors = prototype.get('ranking_factors', {})
            budget_fit = ranking_factors.get('budget_fit', 1.0)
            performance_balance = ranking_factors.get('performance_balance', 0.5)
            satisfaction_scores.append((budget_fit + performance_balance) / 2.0)
        
        predictions['requirements_satisfaction'] = np.mean(satisfaction_scores)
        
        return predictions
    
    def _calculate_variant_potential(self,
                                   source_prototypes: List[Dict[str, Any]],
                                   aggregated_prototype: Dict[str, Any]) -> float:
        """Calculate potential for generating variants from aggregated prototype"""
        
        # Factors influencing variant potential
        factors = []
        
        # 1. Diversity of source prototypes
        source_scores = [p['final_score'] for p in source_prototypes]
        diversity_factor = np.std(source_scores)  # Higher std = more diversity
        factors.append(diversity_factor)
        
        # 2. Complementarity of features
        complementarity_sum = 0
        comparison_count = 0
        
        for i in range(len(source_prototypes)):
            for j in range(i + 1, len(source_prototypes)):
                if self._are_complementary(source_prototypes[i], source_prototypes[j]):
                    complementarity_sum += 1
                comparison_count += 1
        
        complementarity_factor = complementarity_sum / max(comparison_count, 1)
        factors.append(complementarity_factor)
        
        # 3. Performance ceiling
        aggregated_score = aggregated_prototype['final_score']
        performance_headroom = 1.0 - aggregated_score  # Room for improvement
        factors.append(performance_headroom)
        
        # 4. Feature variability across criteria
        criterion_variabilities = []
        all_criteria = set()
        for prototype in source_prototypes:
            all_criteria.update(prototype.get('criterion_scores', {}).keys())
        
        for criterion in all_criteria:
            criterion_scores = []
            for prototype in source_prototypes:
                criterion_data = prototype.get('criterion_scores', {}).get(criterion, {})
                criterion_scores.append(criterion_data.get('score', 0.5))
            
            if len(criterion_scores) > 1:
                criterion_variabilities.append(np.std(criterion_scores))
        
        if criterion_variabilities:
            avg_criterion_variability = np.mean(criterion_variabilities)
            factors.append(avg_criterion_variability)
        
        # Combine factors with weights
        if factors:
            variant_potential = np.average(factors, weights=[0.3, 0.3, 0.2, 0.2][:len(factors)])
        else:
            variant_potential = 0.5
        
        return min(1.0, variant_potential)
    
    def _generate_final_prototypes(self,
                                 retained_prototypes: List[Dict[str, Any]],
                                 aggregation_results: List[AggregationResult],
                                 requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final set of prototypes combining retained and aggregated"""
        
        final_prototypes = []
        
        # Add retained prototypes
        for prototype in retained_prototypes:
            # Mark as individual (not aggregated)
            prototype['aggregation_metadata'] = {
                'is_aggregated': False,
                'selection_reason': 'retained_after_pruning'
            }
            final_prototypes.append(prototype)
        
        # Add high-quality aggregated prototypes
        for agg_result in aggregation_results:
            aggregated_proto = agg_result.aggregated_prototype
            
            # Quality check for aggregated prototypes
            if (aggregated_proto['final_score'] > self.quality_threshold and
                agg_result.aggregation_confidence > 0.6):
                
                final_prototypes.append(aggregated_proto)
        
        # Sort by final score
        final_prototypes.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply final limit
        final_prototypes = final_prototypes[:self.max_final_prototypes]
        
        # Add final ranking within the selected set
        for i, prototype in enumerate(final_prototypes):
            prototype['final_ranking'] = i + 1
            prototype['selection_metadata'] = {
                'final_selection_timestamp': str(np.datetime64('now')),
                'selection_rank': i + 1,
                'total_finalists': len(final_prototypes)
            }
        
        return final_prototypes
    
    def _calculate_performance_guarantees(self, final_prototypes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance guarantees for final prototype set"""
        
        if not final_prototypes:
            return {}
        
        guarantees = {}
        
        # Minimum guaranteed performance
        final_scores = [p['final_score'] for p in final_prototypes]
        guarantees['minimum_performance'] = min(final_scores)
        guarantees['average_performance'] = np.mean(final_scores)
        guarantees['maximum_performance'] = max(final_scores)
        
        # Confidence guarantees
        confidences = [p['overall_confidence'] for p in final_prototypes]
        guarantees['minimum_confidence'] = min(confidences)
        guarantees['average_confidence'] = np.mean(confidences)
        
        # Diversity guarantee
        if len(final_prototypes) > 1:
            diversity_scores = []
            for i, proto1 in enumerate(final_prototypes):
                for j, proto2 in enumerate(final_prototypes[i+1:], i+1):
                    # Simple diversity measure based on score differences
                    criteria_diffs = []
                    scores1 = proto1.get('criterion_scores', {})
                    scores2 = proto2.get('criterion_scores', {})
                    
                    for criterion in set(scores1.keys()).intersection(scores2.keys()):
                        score1 = scores1[criterion].get('score', 0.5)
                        score2 = scores2[criterion].get('score', 0.5)
                        criteria_diffs.append(abs(score1 - score2))
                    
                    if criteria_diffs:
                        diversity_scores.append(np.mean(criteria_diffs))
            
            if diversity_scores:
                guarantees['diversity_level'] = np.mean(diversity_scores)
            else:
                guarantees['diversity_level'] = 0.0
        else:
            guarantees['diversity_level'] = 1.0  # Single prototype is maximally "diverse"
        
        # Quality consistency guarantee
        score_std = np.std(final_scores)
        guarantees['quality_consistency'] = 1.0 - min(1.0, score_std)  # Lower std = higher consistency
        
        # Pareto optimality guarantee
        pareto_count = sum(1 for p in final_prototypes 
                          if p.get('pareto_efficiency', {}).get('is_pareto_optimal', False))
        guarantees['pareto_representation'] = pareto_count / len(final_prototypes)
        
        return guarantees
    
    def _generate_recommendation_ranking(self,
                                       final_prototypes: List[Dict[str, Any]],
                                       requirements: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """Generate recommendation ranking with explanations"""
        
        recommendations = []
        
        for i, prototype in enumerate(final_prototypes):
            prototype_id = prototype['prototype_id']
            final_score = prototype['final_score']
            
            # Generate recommendation reason
            reason = self._generate_recommendation_reason(prototype, i, requirements)
            
            recommendations.append((prototype_id, final_score, reason))
        
        return recommendations
    
    def _generate_recommendation_reason(self,
                                      prototype: Dict[str, Any],
                                      rank: int,
                                      requirements: Dict[str, Any]) -> str:
        """Generate explanation for recommendation"""
        
        # Identify top strengths
        criterion_scores = prototype.get('criterion_scores', {})
        top_criteria = sorted(criterion_scores.items(), 
                             key=lambda x: x[1].get('score', 0), reverse=True)
        
        top_strengths = [criteria for criteria, data in top_criteria[:2] 
                        if data.get('score', 0) > 0.75]
        
        # Check if aggregated
        is_aggregated = prototype.get('aggregation_metadata', {}).get('is_aggregated', False)
        
        # Check Pareto optimality
        is_pareto = prototype.get('pareto_efficiency', {}).get('is_pareto_optimal', False)
        
        # Build reason
        if rank == 0:
            reason = "Top recommendation: "
        elif rank == 1:
            reason = "Strong alternative: "
        else:
            reason = "Additional option: "
        
        if is_pareto:
            reason += "Pareto optimal solution. "
        
        if is_aggregated:
            source_count = prototype.get('aggregation_metadata', {}).get('source_count', 0)
            reason += f"Combines best features from {source_count} designs. "
        
        if top_strengths:
            strengths_text = ', '.join(top_strengths).replace('_', ' ')
            reason += f"Excels in {strengths_text}. "
        
        # Add performance summary
        final_score = prototype['final_score']
        confidence = prototype['overall_confidence']
        reason += f"Performance: {final_score:.1%} (confidence: {confidence:.1%})"
        
        return reason
    
    def _calculate_diversity_preservation(self,
                                        original_prototypes: List[Dict[str, Any]],
                                        retained_ids: Set[str],
                                        features: np.ndarray,
                                        prototype_ids: List[str]) -> float:
        """Calculate how well diversity was preserved during pruning"""
        
        if len(original_prototypes) <= 1 or len(retained_ids) <= 1:
            return 1.0
        
        # Calculate original diversity
        original_diversity = self._calculate_feature_diversity(features)
        
        # Calculate retained diversity
        id_to_idx = {proto_id: idx for idx, proto_id in enumerate(prototype_ids)}
        retained_indices = [id_to_idx[proto_id] for proto_id in retained_ids 
                           if proto_id in id_to_idx]
        
        if len(retained_indices) <= 1:
            return 0.0
        
        retained_features = features[retained_indices]
        retained_diversity = self._calculate_feature_diversity(retained_features)
        
        # Preservation ratio
        preservation_ratio = retained_diversity / max(original_diversity, 1e-6)
        
        return min(1.0, preservation_ratio)
    
    def _calculate_feature_diversity(self, features: np.ndarray) -> float:
        """Calculate diversity metric for a set of feature vectors"""
        
        if len(features) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.linalg.norm(features[i] - features[j])
                distances.append(distance)
        
        # Average distance as diversity measure
        return np.mean(distances) if distances else 0.0
    
    def _calculate_quality_maintenance(self,
                                     original_prototypes: List[Dict[str, Any]],
                                     retained_ids: Set[str]) -> float:
        """Calculate how well quality was maintained during pruning"""
        
        original_scores = [p['final_score'] for p in original_prototypes]
        retained_scores = [p['final_score'] for p in original_prototypes 
                          if p['prototype_id'] in retained_ids]
        
        if not original_scores or not retained_scores:
            return 0.0
        
        original_avg = np.mean(original_scores)
        retained_avg = np.mean(retained_scores)
        
        # Quality maintenance ratio
        maintenance_ratio = retained_avg / max(original_avg, 1e-6)
        
        return min(1.0, maintenance_ratio)
    
    def _create_empty_output(self) -> SpecializationOutput:
        """Create empty output for edge cases"""
        
        return SpecializationOutput(
            final_prototypes=[],
            pruning_summary=PruningResult(0, 0, [], {}, 0.0, 0.0),
            aggregation_summary=[],
            specialization_metadata={},
            performance_guarantees={},
            recommendation_ranking=[]
        )
    
    def _update_specialization_stats(self,
                                   original_count: int,
                                   final_count: int,
                                   pruning_result: PruningResult,
                                   aggregation_results: List[AggregationResult]):
        """Update specialization statistics"""
        
        self.specialization_stats['total_specializations'] += 1
        
        # Update pruning ratio
        if original_count > 0:
            pruning_ratio = (original_count - final_count) / original_count
            current_avg = self.specialization_stats['avg_pruning_ratio']
            total_specs = self.specialization_stats['total_specializations']
            
            new_avg = ((current_avg * (total_specs - 1)) + pruning_ratio) / total_specs
            self.specialization_stats['avg_pruning_ratio'] = new_avg
        
        # Update aggregation success rate
        successful_aggregations = len([r for r in aggregation_results 
                                     if r.aggregation_confidence > 0.6])
        total_attempts = len(aggregation_results)
        
        if total_attempts > 0:
            success_rate = successful_aggregations / total_attempts
            current_avg = self.specialization_stats['avg_aggregation_success']
            
            new_avg = ((current_avg * (self.specialization_stats['total_specializations'] - 1)) + success_rate) / self.specialization_stats['total_specializations']
            self.specialization_stats['avg_aggregation_success'] = new_avg
        
        # Update diversity preservation
        diversity_preserved = pruning_result.diversity_preserved
        current_avg = self.specialization_stats['diversity_preservation_rate']
        
        new_avg = ((current_avg * (self.specialization_stats['total_specializations'] - 1)) + diversity_preserved) / self.specialization_stats['total_specializations']
        self.specialization_stats['diversity_preservation_rate'] = new_avg
        
        # Update quality improvement (simplified)
        quality_maintained = pruning_result.quality_maintained
        current_avg = self.specialization_stats['quality_improvement_rate']
        
        new_avg = ((current_avg * (self.specialization_stats['total_specializations'] - 1)) + quality_maintained) / self.specialization_stats['total_specializations']
        self.specialization_stats['quality_improvement_rate'] = new_avg
    
    def _create_specialization_metadata(self,
                                      original_prototypes: List[Dict[str, Any]],
                                      final_prototypes: List[Dict[str, Any]],
                                      pruning_result: PruningResult,
                                      aggregation_results: List[AggregationResult]) -> Dict[str, Any]:
        """Create comprehensive specialization metadata"""
        
        metadata = {
            'specialization_summary': {
                'original_prototype_count': len(original_prototypes),
                'final_prototype_count': len(final_prototypes),
                'pruning_ratio': (len(original_prototypes) - len(final_prototypes)) / max(len(original_prototypes), 1),
                'aggregation_attempts': len(aggregation_results),
                'successful_aggregations': len([r for r in aggregation_results if r.aggregation_confidence > 0.6])
            },
            
            'pruning_analysis': {
                'strategies_used': list(pruning_result.pruning_rationale.keys()),
                'diversity_preserved': pruning_result.diversity_preserved,
                'quality_maintained': pruning_result.quality_maintained,
                'prototypes_pruned': pruning_result.pruned_count
            },
            
            'aggregation_analysis': {
                'methods_used': list(set(r.aggregated_prototype['aggregation_metadata']['aggregation_type'] 
                                       for r in aggregation_results)),
                'avg_aggregation_confidence': np.mean([r.aggregation_confidence for r in aggregation_results]) 
                                             if aggregation_results else 0.0,
                'total_source_prototypes': len(set().union(*[r.source_prototypes for r in aggregation_results])) 
                                          if aggregation_results else 0
            },
            
            'final_set_characteristics': {
                'score_range': [min(p['final_score'] for p in final_prototypes),
                               max(p['final_score'] for p in final_prototypes)] if final_prototypes else [0, 0],
                'avg_confidence': np.mean([p['overall_confidence'] for p in final_prototypes]) if final_prototypes else 0.0,
                'pareto_optimal_count': len([p for p in final_prototypes 
                                           if p.get('pareto_efficiency', {}).get('is_pareto_optimal', False)]),
                'aggregated_prototype_count': len([p for p in final_prototypes 
                                                 if p.get('aggregation_metadata', {}).get('is_aggregated', False)])
            },
            
            'specialization_parameters': {
                'strategies_used': [s.value for s in self.pruning_strategies],
                'aggregation_methods': [m.value for m in self.aggregation_methods],
                'diversity_weight': self.diversity_weight,
                'quality_threshold': self.quality_threshold,
                'max_final_prototypes': self.max_final_prototypes
            },
            
            'performance_analysis': {
                'specialization_timestamp': str(np.datetime64('now')),
                'processing_efficiency': {
                    'cache_hits': len(self.similarity_cache),
                    'feature_cache_size': len(self.feature_cache)
                }
            }
        }
        
        return metadata
    
    def export_for_fbs_generation(self, specialization_output: SpecializationOutput) -> Dict[str, Any]:
        """Export specialized prototypes for FBS Layout Generation"""
        
        export_data = {
            'specialized_prototypes': [],
            'specialization_metadata': specialization_output.specialization_metadata,
            'performance_guarantees': specialization_output.performance_guarantees,
            'recommendation_ranking': specialization_output.recommendation_ranking,
            'fbs_generation_hints': {}
        }
        
        # Process each final prototype for FBS generation
        for prototype in specialization_output.final_prototypes:
            fbs_ready_prototype = {
                'prototype_id': prototype['prototype_id'],
                'final_score': prototype['final_score'],
                'overall_confidence': prototype['overall_confidence'],
                'final_ranking': prototype.get('final_ranking', 0),
                
                # Aggregation information
                'is_aggregated': prototype.get('aggregation_metadata', {}).get('is_aggregated', False),
                'aggregation_type': prototype.get('aggregation_metadata', {}).get('aggregation_type', 'individual'),
                'source_prototypes': prototype.get('aggregation_metadata', {}).get('source_prototypes', []),
                
                # Performance characteristics
                'criterion_scores': prototype.get('criterion_scores', {}),
                'pareto_efficiency': prototype.get('pareto_efficiency', {}),
                
                # FBS generation priorities
                'generation_priorities': self._extract_generation_priorities(prototype),
                'constraint_sensitivities': self._extract_constraint_sensitivities(prototype),
                'optimization_targets': self._extract_optimization_targets(prototype)
            }
            
            export_data['specialized_prototypes'].append(fbs_ready_prototype)
        
        # Add FBS generation hints
        export_data['fbs_generation_hints'] = self._generate_fbs_hints(specialization_output)
        
        return export_data
    
    def _extract_generation_priorities(self, prototype: Dict[str, Any]) -> Dict[str, float]:
        """Extract generation priorities for FBS system"""
        
        priorities = {}
        criterion_scores = prototype.get('criterion_scores', {})
        
        # Map criterion scores to FBS priorities
        priority_mapping = {
            'spatial_efficiency': 'spatial_optimization',
            'functional_organization': 'functional_layout',
            'circulation_quality': 'circulation_design',
            'environmental_performance': 'environmental_optimization',
            'aesthetic_quality': 'visual_design',
            'constructability': 'structural_feasibility'
        }
        
        for criterion, fbs_priority in priority_mapping.items():
            if criterion in criterion_scores:
                score = criterion_scores[criterion].get('score', 0.5)
                confidence = criterion_scores[criterion].get('confidence', 0.5)
                
                # Priority = score * confidence (high-performing, confident areas get priority)
                priorities[fbs_priority] = score * confidence
        
        return priorities
    
    def _extract_constraint_sensitivities(self, prototype: Dict[str, Any]) -> Dict[str, float]:
        """Extract constraint sensitivities for FBS system"""
        
        sensitivities = {}
        criterion_scores = prototype.get('criterion_scores', {})
        
        # Areas with lower scores are more sensitive to constraints
        for criterion, data in criterion_scores.items():
            score = data.get('score', 0.5)
            confidence = data.get('confidence', 0.5)
            
            # Higher sensitivity for lower-performing areas with high confidence
            # (confident that it's not performing well = sensitive constraint)
            sensitivity = (1.0 - score) * confidence
            sensitivities[criterion] = sensitivity
        
        return sensitivities
    
    def _extract_optimization_targets(self, prototype: Dict[str, Any]) -> Dict[str, float]:
        """Extract optimization targets for FBS system"""
        
        targets = {}
        criterion_scores = prototype.get('criterion_scores', {})
        
        # Targets based on current performance and improvement potential
        for criterion, data in criterion_scores.items():
            current_score = data.get('score', 0.5)
            confidence = data.get('confidence', 0.5)
            
            # Target improvement based on confidence and current performance
            improvement_potential = (1.0 - current_score) * confidence
            target_score = min(1.0, current_score + (improvement_potential * 0.3))
            
            targets[criterion] = target_score
        
        return targets
    
    def _generate_fbs_hints(self, specialization_output: SpecializationOutput) -> Dict[str, Any]:
        """Generate hints for FBS Layout Generation system"""
        
        hints = {
            'generation_strategy': 'balanced',  # Default
            'focus_areas': [],
            'constraint_priorities': {},
            'variant_suggestions': []
        }
        
        final_prototypes = specialization_output.final_prototypes
        
        if not final_prototypes:
            return hints
        
        # Analyze common strengths and weaknesses
        all_criteria_scores = defaultdict(list)
        
        for prototype in final_prototypes:
            criterion_scores = prototype.get('criterion_scores', {})
            for criterion, data in criterion_scores.items():
                all_criteria_scores[criterion].append(data.get('score', 0.5))
        
        # Identify focus areas (consistently weak areas)
        weak_areas = []
        strong_areas = []
        
        for criterion, scores in all_criteria_scores.items():
            avg_score = np.mean(scores)
            score_consistency = 1.0 - np.std(scores)  # High consistency = low std
            
            if avg_score < 0.6 and score_consistency > 0.7:
                weak_areas.append(criterion)
            elif avg_score > 0.8 and score_consistency > 0.7:
                strong_areas.append(criterion)
        
        hints['focus_areas'] = weak_areas
        hints['strong_areas'] = strong_areas
        
        # Determine generation strategy
        if len(weak_areas) > 3:
            hints['generation_strategy'] = 'improvement_focused'
        elif len(strong_areas) > 3:
            hints['generation_strategy'] = 'optimization_focused'
        else:
            hints['generation_strategy'] = 'balanced'
        
        # Set constraint priorities based on performance guarantees
        performance_guarantees = specialization_output.performance_guarantees
        
        if performance_guarantees.get('minimum_performance', 0) < 0.7:
            hints['constraint_priorities']['quality_assurance'] = 0.9
        
        if performance_guarantees.get('diversity_level', 0) < 0.4:
            hints['constraint_priorities']['diversity_enhancement'] = 0.8
        
        if performance_guarantees.get('pareto_representation', 0) < 0.3:
            hints['constraint_priorities']['pareto_optimization'] = 0.7
        
        # Generate variant suggestions
        for i, prototype in enumerate(final_prototypes[:3]):  # Top 3 prototypes
            variant_suggestion = {
                'base_prototype': prototype['prototype_id'],
                'variant_type': 'optimization',
                'target_improvements': []
            }
            
            criterion_scores = prototype.get('criterion_scores', {})
            
            # Find areas with moderate scores that could be improved
            for criterion, data in criterion_scores.items():
                score = data.get('score', 0.5)
                confidence = data.get('confidence', 0.5)
                
                if 0.6 <= score <= 0.8 and confidence > 0.6:
                    variant_suggestion['target_improvements'].append({
                        'criterion': criterion,
                        'current_score': score,
                        'target_improvement': min(0.2, (1.0 - score) * 0.5),
                        'confidence': confidence
                    })
            
            if variant_suggestion['target_improvements']:
                hints['variant_suggestions'].append(variant_suggestion)
        
        # Add aggregation-specific hints
        aggregated_count = len([p for p in final_prototypes 
                               if p.get('aggregation_metadata', {}).get('is_aggregated', False)])
        
        if aggregated_count > 0:
            hints['aggregation_insights'] = {
                'aggregated_prototype_count': aggregated_count,
                'aggregation_success_indicators': self._analyze_aggregation_success(final_prototypes),
                'feature_combination_opportunities': self._identify_combination_opportunities(final_prototypes)
            }
        
        return hints
    
    def _analyze_aggregation_success(self, final_prototypes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success patterns of aggregated prototypes"""
        
        aggregated_prototypes = [p for p in final_prototypes 
                                if p.get('aggregation_metadata', {}).get('is_aggregated', False)]
        
        individual_prototypes = [p for p in final_prototypes 
                                if not p.get('aggregation_metadata', {}).get('is_aggregated', False)]
        
        if not aggregated_prototypes:
            return {}
        
        # Compare performance
        agg_scores = [p['final_score'] for p in aggregated_prototypes]
        ind_scores = [p['final_score'] for p in individual_prototypes] if individual_prototypes else [0.5]
        
        success_indicators = {
            'aggregated_avg_performance': np.mean(agg_scores),
            'individual_avg_performance': np.mean(ind_scores),
            'aggregation_advantage': np.mean(agg_scores) - np.mean(ind_scores),
            'aggregation_consistency': 1.0 - np.std(agg_scores) if len(agg_scores) > 1 else 1.0,
            'successful_aggregation_types': []
        }
        
        # Analyze successful aggregation types
        for prototype in aggregated_prototypes:
            if prototype['final_score'] > np.mean(ind_scores):
                agg_type = prototype.get('aggregation_metadata', {}).get('aggregation_type', 'unknown')
                success_indicators['successful_aggregation_types'].append(agg_type)
        
        return success_indicators
    
    def _identify_combination_opportunities(self, final_prototypes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for feature combination in future generations"""
        
        opportunities = []
        
        # Find complementary individual prototypes that weren't aggregated
        individual_prototypes = [p for p in final_prototypes 
                                if not p.get('aggregation_metadata', {}).get('is_aggregated', False)]
        
        if len(individual_prototypes) < 2:
            return opportunities
        
        # Check all pairs for complementarity
        for i in range(len(individual_prototypes)):
            for j in range(i + 1, len(individual_prototypes)):
                proto1, proto2 = individual_prototypes[i], individual_prototypes[j]
                
                if self._are_complementary(proto1, proto2):
                    opportunity = {
                        'prototype_1': proto1['prototype_id'],
                        'prototype_2': proto2['prototype_id'],
                        'complementarity_score': self._calculate_complementarity_score(proto1, proto2),
                        'potential_improvements': self._identify_potential_improvements(proto1, proto2)
                    }
                    opportunities.append(opportunity)
        
        # Sort by complementarity score
        opportunities.sort(key=lambda x: x['complementarity_score'], reverse=True)
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _calculate_complementarity_score(self, proto1: Dict[str, Any], proto2: Dict[str, Any]) -> float:
        """Calculate numerical complementarity score"""
        
        scores1 = proto1.get('criterion_scores', {})
        scores2 = proto2.get('criterion_scores', {})
        
        complementarity_values = []
        
        for criterion in set(scores1.keys()).intersection(scores2.keys()):
            score1 = scores1[criterion].get('score', 0.5)
            score2 = scores2[criterion].get('score', 0.5)
            
            # Complementarity is high when one is strong and the other is weak
            complementarity = abs(score1 - score2) * (max(score1, score2))
            complementarity_values.append(complementarity)
        
        return np.mean(complementarity_values) if complementarity_values else 0.0
    
    def _identify_potential_improvements(self, proto1: Dict[str, Any], proto2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvements possible through combination"""
        
        improvements = []
        scores1 = proto1.get('criterion_scores', {})
        scores2 = proto2.get('criterion_scores', {})
        
        for criterion in set(scores1.keys()).intersection(scores2.keys()):
            score1 = scores1[criterion].get('score', 0.5)
            score2 = scores2[criterion].get('score', 0.5)
            conf1 = scores1[criterion].get('confidence', 0.5)
            conf2 = scores2[criterion].get('confidence', 0.5)
            
            # Improvement potential: take best score, weighted by confidence
            best_score = max(score1, score2)
            avg_confidence = (conf1 + conf2) / 2.0
            current_combined = (score1 + score2) / 2.0
            
            improvement_potential = (best_score - current_combined) * avg_confidence
            
            if improvement_potential > 0.05:  # Significant improvement potential
                improvements.append({
                    'criterion': criterion,
                    'current_proto1_score': score1,
                    'current_proto2_score': score2,
                    'potential_combined_score': best_score,
                    'improvement_potential': improvement_potential,
                    'confidence': avg_confidence
                })
        
        # Sort by improvement potential
        improvements.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        return improvements[:5]  # Top 5 improvements
    
    def get_specialization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive specialization statistics"""
        
        return {
            'performance_metrics': {
                'total_specializations': self.specialization_stats['total_specializations'],
                'avg_pruning_ratio': self.specialization_stats['avg_pruning_ratio'],
                'avg_aggregation_success': self.specialization_stats['avg_aggregation_success'],
                'diversity_preservation_rate': self.specialization_stats['diversity_preservation_rate'],
                'quality_improvement_rate': self.specialization_stats['quality_improvement_rate']
            },
            
            'configuration': {
                'pruning_strategies': [s.value for s in self.pruning_strategies],
                'aggregation_methods': [m.value for m in self.aggregation_methods],
                'diversity_weight': self.diversity_weight,
                'quality_threshold': self.quality_threshold,
                'max_final_prototypes': self.max_final_prototypes
            },
            
            'cache_performance': {
                'similarity_cache_size': len(self.similarity_cache),
                'feature_cache_size': len(self.feature_cache),
                'cache_hit_efficiency': len(self.feature_cache) / max(self.specialization_stats['total_specializations'], 1)
            },
            
            'specialization_parameters': self.specialization_params
        }
    
    def clear_caches(self):
        """Clear internal caches"""
        self.similarity_cache.clear()
        self.feature_cache.clear()
        logger.info("Specializer caches cleared")
    
    def update_specialization_parameters(self, new_params: Dict[str, Any]):
        """Update specialization parameters"""
        self.specialization_params.update(new_params)
        logger.info(f"Updated specialization parameters: {list(new_params.keys())}")


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Initializing Prototype Specializer...")
    
    # Initialize specializer
    specializer = PrototypeSpecializer(
        pruning_strategies=[
            PruningStrategy.HYBRID_INTELLIGENT,
            PruningStrategy.DIVERSITY_PRESERVING
        ],
        aggregation_methods=[
            AggregationMethod.ADAPTIVE_CLUSTERING,
            AggregationMethod.PARETO_OPTIMAL_FUSION
        ],
        diversity_weight=0.3,
        quality_threshold=0.6,
        max_final_prototypes=5
    )
    
    # Example scored prototypes from Scoring Agent
    sample_scored_prototypes = [
        {
            'prototype_id': 'compact_central_zonal_0',
            'final_score': 0.85,
            'weighted_total': 0.82,
            'diversity_bonus': 0.03,
            'overall_confidence': 0.78,
            'criterion_scores': {
                'spatial_efficiency': {'score': 0.88, 'confidence': 0.85, 'explanation': 'High spatial efficiency'},
                'functional_organization': {'score': 0.82, 'confidence': 0.80, 'explanation': 'Good functional layout'},
                'environmental_performance': {'score': 0.75, 'confidence': 0.70, 'explanation': 'Decent environmental design'},
                'cost_efficiency': {'score': 0.90, 'confidence': 0.85, 'explanation': 'Very cost effective'},
                'code_compliance': {'score': 0.95, 'confidence': 0.95, 'explanation': 'Full code compliance'}
            },
            'pareto_efficiency': {
                'is_pareto_optimal': True,
                'pareto_rank': 1,
                'distance_to_pareto_front': 0.0
            },
            'ranking_factors': {
                'confidence_weighted_score': 0.82,
                'risk_level': 0.2,
                'performance_balance': 0.85,
                'budget_fit': 0.9
            }
        },
        {
            'prototype_id': 'linear_progression_1',
            'final_score': 0.78,
            'weighted_total': 0.76,
            'diversity_bonus': 0.02,
            'overall_confidence': 0.72,
            'criterion_scores': {
                'spatial_efficiency': {'score': 0.70, 'confidence': 0.75, 'explanation': 'Moderate spatial efficiency'},
                'functional_organization': {'score': 0.85, 'confidence': 0.82, 'explanation': 'Excellent functional layout'},
                'environmental_performance': {'score': 0.90, 'confidence': 0.85, 'explanation': 'Excellent environmental design'},
                'cost_efficiency': {'score': 0.65, 'confidence': 0.70, 'explanation': 'Moderate cost efficiency'},
                'code_compliance': {'score': 0.85, 'confidence': 0.90, 'explanation': 'Good code compliance'}
            },
            'pareto_efficiency': {
                'is_pareto_optimal': False,
                'pareto_rank': 2,
                'distance_to_pareto_front': 0.08
            },
            'ranking_factors': {
                'confidence_weighted_score': 0.75,
                'risk_level': 0.28,
                'performance_balance': 0.72,
                'budget_fit': 0.65
            }
        },
        {
            'prototype_id': 'courtyard_focused_2',
            'final_score': 0.73,
            'weighted_total': 0.72,
            'diversity_bonus': 0.01,
            'overall_confidence': 0.68,
            'criterion_scores': {
                'spatial_efficiency': {'score': 0.65, 'confidence': 0.70, 'explanation': 'Lower spatial efficiency'},
                'functional_organization': {'score': 0.75, 'confidence': 0.72, 'explanation': 'Good functional organization'},
                'environmental_performance': {'score': 0.95, 'confidence': 0.90, 'explanation': 'Outstanding environmental performance'},
                'cost_efficiency': {'score': 0.60, 'confidence': 0.65, 'explanation': 'Lower cost efficiency'},
                'code_compliance': {'score': 0.80, 'confidence': 0.85, 'explanation': 'Adequate code compliance'}
            },
            'pareto_efficiency': {
                'is_pareto_optimal': True,
                'pareto_rank': 1,
                'distance_to_pareto_front': 0.0
            },
            'ranking_factors': {
                'confidence_weighted_score': 0.68,
                'risk_level': 0.32,
                'performance_balance': 0.68,
                'budget_fit': 0.60
            }
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
        'design_preferences': {
            'sustainability_focus': True,
            'space_optimization': True
        },
        'budget': 2800000
    }
    
    # Perform specialization
    print("🔧 Performing prototype specialization...")
    specialization_result = specializer.specialize_prototypes(
        scored_prototypes=sample_scored_prototypes,
        requirements=sample_requirements
    )
    
    # Display results
    print(f"\n📊 Specialization Results:")
    print("=" * 60)
    
    print(f"Original prototypes: {specialization_result.pruning_summary.original_count}")
    print(f"Final prototypes: {len(specialization_result.final_prototypes)}")
    print(f"Prototypes pruned: {specialization_result.pruning_summary.pruned_count}")
    print(f"Aggregation attempts: {len(specialization_result.aggregation_summary)}")
    
    print(f"\n🎯 Final Prototype Rankings:")
    for rank, (proto_id, score, reason) in enumerate(specialization_result.recommendation_ranking, 1):
        print(f"  {rank}. {proto_id} (Score: {score:.3f})")
        print(f"     {reason}")
    
    print(f"\n📈 Performance Guarantees:")
    for guarantee, value in specialization_result.performance_guarantees.items():
        print(f"  {guarantee}: {value:.3f}")
    
    # Export for FBS generation
    fbs_export = specializer.export_for_fbs_generation(specialization_result)
    print(f"\n📤 FBS Export Ready: {len(fbs_export['specialized_prototypes'])} prototypes")
    print(f"Generation strategy: {fbs_export['fbs_generation_hints']['generation_strategy']}")
    
    # Show statistics
    stats = specializer.get_specialization_statistics()
    print(f"\n📊 Specializer Statistics:")
    print(f"  Total specializations: {stats['performance_metrics']['total_specializations']}")
    print(f"  Cache efficiency: {stats['cache_performance']['cache_hit_efficiency']:.1%}")
    
    print("\n✅ Specializer demonstration complete!")
    print("📋 Ready to pass specialized prototypes to FBS Layout Generation system...")