import numpy as np
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter, deque
import logging
from abc import ABC, abstractmethod
import itertools
import networkx as nx
import uuid
from datetime import datetime
from encoder import Gemma3Encoder 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Node in the Graph of Thoughts structure"""
    node_id: str
    node_type: str  # 'root', 'variant', 'synthesis'
    level: int
    generation: int
    
    # Graph topology
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    cross_connections: List[str] = field(default_factory=list)
    synthesis_sources: List[str] = field(default_factory=list)
    
    # Content
    prototype: Optional['DesignPrototype'] = None
    strategy_DNA: Dict[str, float] = field(default_factory=dict)
    
    # Research preparation
    research_queries: List[Dict] = field(default_factory=list)
    research_priority: float = 0.5
    
    # Expansion control
    branch_potential: float = 1.0
    explored: bool = False
    synthesis_candidate: bool = True
    
    # Metrics
    complexity_score: float = 0.0
    novelty_estimate: float = 0.0
    expansion_count: int = 0

@dataclass
class DesignPrototype:
    """Enhanced design prototype for GoT pipeline"""
    prototype_id: str
    hierarchy_level: int
    generation: int
    node_path: List[str] = field(default_factory=list)
    
    # Core design properties  
    spatial_config: Dict[str, Any] = field(default_factory=dict)
    functional_zones: Dict[str, Any] = field(default_factory=dict)
    circulation_pattern: Dict[str, Any] = field(default_factory=dict)
    environmental_strategy: Dict[str, Any] = field(default_factory=dict)
    
    # Graph-specific properties
    strategy_composition: Dict[str, float] = field(default_factory=dict)
    synthesis_rationale: str = ""
    cross_pollination_sources: List[str] = field(default_factory=list)
    
    # Pipeline preparation
    research_keywords: List[str] = field(default_factory=list)
    specialization_hints: Dict[str, Any] = field(default_factory=dict)
    scoring_features: Dict[str, float] = field(default_factory=dict)
    
    # Vector embedding
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

class GraphOfThoughtsGeneralizer:
    """
    True Graph of Thoughts Generalizer with extensive branching
    Designed to feed research agent -> specializer -> scoring agent pipeline
    """
    
    def __init__(self, 
                 max_generations: int = 8,
                 initial_strategies: int = 6,
                 branch_factor_base: float = 3.0,
                 synthesis_probability: float = 0.4,
                 cross_connection_rate: float = 0.3,
                 synthesis_threshold: float = 0.6):
        
        # Core parameters
        self.max_generations = max_generations
        self.initial_strategies = initial_strategies
        self.branch_factor_base = branch_factor_base
        self.synthesis_probability = synthesis_probability
        self.cross_connection_rate = cross_connection_rate
        self.synthesis_threshold = synthesis_threshold
        
        # Graph structures
        self.thought_graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.prototypes: Dict[str, DesignPrototype] = {}
        
        # Generation tracking
        self.current_generation = 0
        self.generation_active = True
        self.nodes_per_generation = defaultdict(int)
        self.total_prototypes = 0
        
        # Strategy management
        self.base_strategies = self._initialize_comprehensive_strategies()
        self.strategy_families = self._organize_strategy_families()
        self.hybrid_recipes = self._initialize_hybrid_recipes()
        
        # Research preparation
        self.research_queue = []
        self.precedent_categories = self._initialize_precedent_categories()

        self.encoder = Gemma3Encoder()
        logger.info("✅ Initialized Gemma 3 Encoder for GoT embeddings")
        
    def _initialize_comprehensive_strategies(self) -> Dict[str, Dict]:
        """Initialize comprehensive architectural strategies for extensive branching"""
        return {
            'central_courtyard': {
                'type': 'spatial_org',
                'characteristics': {
                    'core_type': 'open_courtyard',
                    'circulation': 'perimeter',
                    'privacy_model': 'inward_facing',
                    'climate_response': 'stack_ventilation'
                },
                'research_keywords': ['courtyard house', 'atrium design', 'climate responsive'],
                'branch_potential': 0.9,
                'synthesis_affinity': ['linear_spine', 'modular_grid', 'environmental_buffer']
            },
            
            'linear_spine': {
                'type': 'spatial_org',
                'characteristics': {
                    'organization': 'linear_sequence',
                    'circulation': 'central_corridor',
                    'zoning': 'progressive_privacy',
                    'flexibility': 'modular_expansion'
                },
                'research_keywords': ['linear planning', 'spine circulation', 'modular housing'],
                'branch_potential': 0.95,
                'synthesis_affinity': ['central_courtyard', 'split_level', 'flexible_grid']
            },
            
            'cluster_organization': {
                'type': 'spatial_org', 
                'characteristics': {
                    'arrangement': 'clustered_pavilions',
                    'connections': 'bridges_links',
                    'landscape': 'integrated_gardens',
                    'hierarchy': 'multiple_centers'
                },
                'research_keywords': ['cluster housing', 'pavilion architecture', 'landscape integration'],
                'branch_potential': 0.85,
                'synthesis_affinity': ['courtyard_spine', 'modular_courtyard']
            },
            
            'hub_spoke_circulation': {
                'type': 'circulation',
                'characteristics': {
                    'pattern': 'radial_from_center',
                    'efficiency': 'minimal_distance',
                    'flexibility': 'easy_reconfiguration',
                    'control': 'central_supervision'
                },
                'research_keywords': ['radial circulation', 'hub design', 'central distribution'],
                'branch_potential': 0.8,
                'synthesis_affinity': ['loop_circulation', 'layered_circulation']
            },
            
            'loop_circulation': {
                'type': 'circulation',
                'characteristics': {
                    'pattern': 'continuous_loop',
                    'redundancy': 'multiple_paths',
                    'privacy': 'graduated_access',
                    'efficiency': 'distributed_flow'
                },
                'research_keywords': ['loop circulation', 'ring corridor', 'continuous path'],
                'branch_potential': 0.85,
                'synthesis_affinity': ['hub_spoke_circulation', 'split_circulation']
            },
            
            'layered_circulation': {
                'type': 'circulation',
                'characteristics': {
                    'pattern': 'multiple_levels',
                    'separation': 'public_private_service',
                    'efficiency': 'dedicated_flows',
                    'integration': 'vertical_connections'
                },
                'research_keywords': ['multi-level circulation', 'separated flows', 'vertical organization'],
                'branch_potential': 0.9,
                'synthesis_affinity': ['hub_spoke_circulation', 'environmental_circulation']
            },
            
            'passive_solar_design': {
                'type': 'environmental',
                'characteristics': {
                    'orientation': 'solar_optimized',
                    'thermal_mass': 'strategic_placement',
                    'shading': 'dynamic_response',
                    'ventilation': 'stack_effect'
                },
                'research_keywords': ['passive solar', 'thermal design', 'energy efficient'],
                'branch_potential': 0.95,
                'synthesis_affinity': ['cross_ventilation', 'thermal_buffer', 'daylight_harvesting']
            },
            
            'cross_ventilation': {
                'type': 'environmental',
                'characteristics': {
                    'airflow': 'pressure_differential',
                    'openings': 'strategic_placement',
                    'cooling': 'natural_convection',
                    'comfort': 'air_movement'
                },
                'research_keywords': ['natural ventilation', 'cross breeze', 'passive cooling'],
                'branch_potential': 0.9,
                'synthesis_affinity': ['passive_solar_design', 'thermal_buffer']
            },
            
            'thermal_buffer_zones': {
                'type': 'environmental',
                'characteristics': {
                    'zoning': 'thermal_gradients',
                    'insulation': 'progressive_protection',
                    'transition': 'climatic_mediation',
                    'efficiency': 'reduced_loads'
                },
                'research_keywords': ['thermal buffers', 'transition zones', 'climate mediation'],
                'branch_potential': 0.85,
                'synthesis_affinity': ['passive_solar_design', 'cross_ventilation']
            },
            
            'vastu_principles': {
                'type': 'cultural',
                'characteristics': {
                    'orientation': 'cardinal_alignment',
                    'zoning': 'directional_functions',
                    'center': 'brahmasthan_void',
                    'energy': 'cosmic_harmony'
                },
                'research_keywords': ['vastu shastra', 'directional planning', 'traditional Indian'],
                'branch_potential': 0.8,
                'synthesis_affinity': ['feng_shui_principles', 'sacred_geometry']
            },
            
            'modular_flexibility': {
                'type': 'flexibility',
                'characteristics': {
                    'components': 'standardized_modules',
                    'assembly': 'plug_and_play',
                    'adaptation': 'reconfigurable_layouts',
                    'growth': 'incremental_expansion'
                },
                'research_keywords': ['modular design', 'flexible housing', 'adaptable architecture'],
                'branch_potential': 1.0,
                'synthesis_affinity': ['open_plan_flexibility', 'convertible_spaces']
            },
            
            'net_zero_energy': {
                'type': 'sustainability',
                'characteristics': {
                    'generation': 'renewable_sources',
                    'conservation': 'ultra_efficient',
                    'storage': 'battery_systems',
                    'monitoring': 'smart_controls'
                },
                'research_keywords': ['net zero', 'renewable energy', 'energy positive'],
                'branch_potential': 0.95,
                'synthesis_affinity': ['passive_solar_design', 'smart_home_integration']
            },
            
            'smart_home_integration': {
                'type': 'technology',
                'characteristics': {
                    'automation': 'intelligent_systems',
                    'connectivity': 'iot_devices',
                    'efficiency': 'optimized_performance',
                    'user_interface': 'intuitive_control'
                },
                'research_keywords': ['smart home', 'home automation', 'iot integration'],
                'branch_potential': 0.9,
                'synthesis_affinity': ['net_zero_energy', 'adaptive_systems']
            }
        }
    
    def _organize_strategy_families(self) -> Dict[str, List[str]]:
        """Organize strategies into families for targeted synthesis"""
        families = defaultdict(list)
        for strategy_id, strategy_data in self.base_strategies.items():
            family = strategy_data['type']
            families[family].append(strategy_id)
        return dict(families)
    
    def _initialize_hybrid_recipes(self) -> Dict[str, Dict]:
        """Define recipes for creating hybrid strategies"""
        return {
            'courtyard_spine': {
                'parents': ['central_courtyard', 'linear_spine'],
                'blend_ratio': [0.6, 0.4],
                'synthesis_logic': 'linear_organization_with_central_court',
                'expected_benefits': ['privacy', 'climate_control', 'clear_circulation']
            },
            'modular_courtyard': {
                'parents': ['modular_flexibility', 'central_courtyard'],
                'blend_ratio': [0.7, 0.3],
                'synthesis_logic': 'expandable_courtyard_modules',
                'expected_benefits': ['adaptability', 'climate_response', 'scalability']
            },
            'smart_passive': {
                'parents': ['smart_home_integration', 'passive_solar_design'],
                'blend_ratio': [0.5, 0.5],
                'synthesis_logic': 'intelligent_passive_systems',
                'expected_benefits': ['efficiency', 'automation', 'sustainability']
            }
        }
    
    def _initialize_precedent_categories(self) -> Dict[str, List[str]]:
        """Initialize precedent categories for research agent"""
        return {
            'tropical_modernism': ['Hassan Fathy', 'Geoffrey Bawa', 'Charles Correa'],
            'courtyard_houses': ['traditional Indian', 'Andalusian patios', 'Chinese siheyuan'],
            'passive_design': ['Hassan Fathy', 'Paul Rudolph', 'John Cummings'],
            'modular_housing': ['Moshe Safdie', 'Renzo Piano', 'Jean Prouvé'],
            'sustainable_design': ['William McDonough', 'Glenn Murcutt', 'Thomas Herzog'],
            'smart_integration': ['MIT House_n', 'BIQ House', 'Edge Amsterdam']
        }
    
    def generate_design_graph(self, 
                            requirements: Dict[str, Any],
                            expansion_control: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate the complete Graph of Thoughts for design exploration
        Returns comprehensive data for research agent -> specializer -> scoring pipeline
        """
        logger.info("🌟 Starting Graph of Thoughts Generation")
        
        # Initialize expansion control
        control = expansion_control or {
            'max_total_nodes': 500,
            'stop_on_convergence': False,
            'research_driven_expansion': True
        }
        
        try:
            # Phase 1: Generate root strategies
            root_nodes = self._generate_root_strategies(requirements)
            logger.info(f"✨ Generated {len(root_nodes)} root strategy nodes")
            
            # Phase 2: Extensive branching with graph connections
            self._extensive_branching(requirements, control)
            logger.info(f"🌳 Completed branching - Total nodes: {len(self.nodes)}")
            
            # Phase 3: Create synthesis nodes (hybrid strategies)
            self._generate_synthesis_nodes(requirements)
            logger.info(f"🔄 Generated synthesis nodes - Total: {len([n for n in self.nodes.values() if n.node_type == 'synthesis'])}")
            
            # Phase 4: Add cross-connections (lateral graph edges)
            self._add_cross_connections()
            logger.info(f"🕸️ Added cross-connections - Graph edges: {self.thought_graph.number_of_edges()}")
            
            # Phase 5: Prepare comprehensive output for pipeline
            pipeline_data = self._prepare_pipeline_output(requirements)
            logger.info(f"📦 Prepared {len(pipeline_data['prototypes'])} prototypes for pipeline")
            
            return pipeline_data
            
        except Exception as e:
            logger.error(f"Error in generate_design_graph: {e}")
            return self._create_minimal_fallback(requirements)
    
    def _generate_root_strategies(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate diverse root strategy nodes"""
        root_nodes = []
        
        # Extract requirement characteristics for strategy selection
        spatial_needs = requirements.get('spatial_needs', [])
        site_constraints = requirements.get('site_constraints', {})
        total_rooms = sum(need.get('quantity', 1) for need in spatial_needs)
        plot_area = site_constraints.get('plot_length', 40) * site_constraints.get('plot_width', 30)
        
        # Select initial strategies based on requirements
        selected_strategies = self._select_initial_strategies(requirements)
        
        generation_id = 0
        for i, strategy_id in enumerate(selected_strategies):
            node_id = f"root_{strategy_id}_{i}"
            
            # Create graph node
            node = GraphNode(
                node_id=node_id,
                node_type='root',
                level=0,
                generation=generation_id,
                branch_potential=self.base_strategies[strategy_id]['branch_potential'],
                research_priority=0.9  # High priority for root nodes
            )
            
            # Create design prototype
            prototype = self._create_strategy_prototype(
                node_id, strategy_id, requirements, generation_id
            )
            
            # Prepare research queries
            node.research_queries = self._prepare_research_queries(strategy_id, requirements)
            
            # Store in structures
            node.prototype = prototype
            self.nodes[node_id] = node
            self.prototypes[node_id] = prototype
            self.thought_graph.add_node(node_id, **asdict(node))
            
            root_nodes.append(node_id)
            self.total_prototypes += 1
        
        self.nodes_per_generation[0] = len(root_nodes)
        return root_nodes
    
    def _select_initial_strategies(self, requirements: Dict[str, Any]) -> List[str]:
        """Intelligently select initial strategies based on requirements"""
        spatial_needs = requirements.get('spatial_needs', [])
        site_constraints = requirements.get('site_constraints', {})
        
        total_rooms = sum(need.get('quantity', 1) for need in spatial_needs)
        plot_length = site_constraints.get('plot_length', 40)
        plot_width = site_constraints.get('plot_width', 30)
        aspect_ratio = plot_length / plot_width
        plot_area = plot_length * plot_width
        
        selected = []
        
        # Always include core spatial strategies
        selected.extend(['central_courtyard', 'linear_spine', 'cluster_organization'])
        
        # Add based on plot characteristics
        if aspect_ratio > 1.5:
            selected.append('linear_spine')  # Favor linear for elongated plots
        if plot_area > 1200:
            selected.append('central_courtyard')  # Large plots can handle courtyards
        if total_rooms >= 6:
            selected.append('layered_circulation')
        
        # Add environmental strategies
        selected.extend(['passive_solar_design', 'cross_ventilation'])
        
        # Add flexibility for uncertainty
        selected.append('modular_flexibility')
        
        # Add cultural context
        if requirements.get('cultural_preferences', {}).get('traditional_influence'):
            selected.append('vastu_principles')
        
        # Add technology if budget allows
        budget = requirements.get('budget', 0)
        if budget > 3000000:  # INR 30 lakhs
            selected.append('smart_home_integration')
        
        # Remove duplicates and limit to initial count
        unique_selected = list(dict.fromkeys(selected))[:self.initial_strategies]
        
        # Ensure minimum diversity
        if len(unique_selected) < 4:
            remaining = [s for s in self.base_strategies.keys() if s not in unique_selected]
            unique_selected.extend(remaining[:4-len(unique_selected)])
        
        return unique_selected
    
    def _extensive_branching(self, requirements: Dict[str, Any], control: Dict):
        """Perform extensive branching to create the graph structure"""
        
        max_total_nodes = control.get('max_total_nodes', 500)
        current_generation = 1
        
        # Get current frontier (nodes to expand)
        frontier = [node_id for node_id, node in self.nodes.items() 
                   if not node.explored and node.branch_potential > 0.3]
        
        while (frontier and 
               current_generation <= self.max_generations and 
               self.total_prototypes < max_total_nodes and
               self.generation_active):
            
            logger.info(f"🚀 Generation {current_generation}: Expanding {len(frontier)} nodes")
            
            new_nodes = []
            
            for parent_id in frontier:
                parent_node = self.nodes[parent_id]
                
                # Calculate branching factor for this node
                branch_factor = self._calculate_dynamic_branch_factor(
                    parent_node, current_generation, requirements
                )
                
                # Generate variants for this parent
                variants = self._generate_strategy_variants(
                    parent_id, branch_factor, requirements, current_generation
                )
                
                new_nodes.extend(variants)
                parent_node.explored = True
                parent_node.expansion_count = len(variants)
                
                # Stop if we've hit limits
                if self.total_prototypes >= max_total_nodes:
                    break
            
            # Update generation tracking
            self.nodes_per_generation[current_generation] = len(new_nodes)
            current_generation += 1
            
            # Update frontier for next generation
            frontier = [node_id for node_id in new_nodes 
                       if self.nodes[node_id].branch_potential > 0.2]
            
            logger.info(f"📈 Generation complete. New nodes: {len(new_nodes)}, Next frontier: {len(frontier)}")
        
        logger.info(f"🏁 Branching complete. Total nodes: {self.total_prototypes}")
    
    def _calculate_dynamic_branch_factor(self, 
                                       node: GraphNode, 
                                       generation: int, 
                                       requirements: Dict) -> int:
        """Calculate how many branches this node should generate"""
        
        # Base factor decreases with generation depth
        base_factor = max(1, int(self.branch_factor_base * (0.8 ** generation)))
        
        # Multiply by node's branch potential
        potential_factor = base_factor * node.branch_potential
        
        # Boost high-performing strategies
        if node.prototype and hasattr(node.prototype, 'scoring_features'):
            performance_boost = node.prototype.scoring_features.get('potential_score', 0.5)
            potential_factor *= (1 + performance_boost)
        
        # Reduce for later generations to prevent explosion
        generation_damping = max(0.3, 1.0 - (generation * 0.15))
        final_factor = potential_factor * generation_damping
        
        return max(1, min(8, int(final_factor)))  # Clamp between 1-8
    
    def _generate_strategy_variants(self, 
                                  parent_id: str, 
                                  variant_count: int,
                                  requirements: Dict[str, Any], 
                                  generation: int) -> List[str]:
        """Generate strategy variants for a parent node"""
        
        parent_node = self.nodes[parent_id]
        parent_prototype = parent_node.prototype
        
        variants = []
        
        for i in range(variant_count):
            variant_id = f"{parent_id}_var_{i}_{generation}"
            
            # Create variant with mutations/explorations
            variant_prototype = self._create_variant_prototype(
                variant_id, parent_prototype, requirements, generation, i
            )
            
            # Create graph node
            variant_node = GraphNode(
                node_id=variant_id,
                node_type='variant',
                level=generation,
                generation=generation,
                parent_ids=[parent_id],
                branch_potential=parent_node.branch_potential * 0.8,  # Decay over generations
                research_priority=0.6
            )
            
            # Prepare research queries for this variant
            variant_node.research_queries = self._prepare_variant_research_queries(
                variant_prototype, parent_prototype
            )
            
            # Store structures
            variant_node.prototype = variant_prototype
            self.nodes[variant_id] = variant_node
            self.prototypes[variant_id] = variant_prototype
            
            # Add to graph with edge
            self.thought_graph.add_node(variant_id, **asdict(variant_node))
            self.thought_graph.add_edge(parent_id, variant_id, relationship='variant_of')
            
            # Update parent's children
            parent_node.child_ids.append(variant_id)
            
            variants.append(variant_id)
            self.total_prototypes += 1
        
        return variants
    
    def _create_strategy_prototype(self, 
                                 node_id: str, 
                                 strategy_id: str, 
                                 requirements: Dict[str, Any],
                                 generation: int) -> DesignPrototype:
        """Create a design prototype for a strategy"""
        
        strategy_data = self.base_strategies[strategy_id]
        
        # Generate spatial configuration
        spatial_config = self._generate_spatial_config_from_strategy(
            strategy_id, strategy_data, requirements
        )
        
        # Generate circulation pattern
        circulation_pattern = self._generate_circulation_from_strategy(
            strategy_id, strategy_data, spatial_config
        )
        
        # Generate environmental strategy
        environmental_strategy = self._generate_environmental_from_strategy(
            strategy_id, strategy_data, requirements
        )
        
        # Create prototype
        prototype = DesignPrototype(
            prototype_id=node_id,
            hierarchy_level=0,  # Root level
            generation=generation,
            node_path=[node_id],
            spatial_config=spatial_config,
            circulation_pattern=circulation_pattern,
            environmental_strategy=environmental_strategy,
            strategy_composition={strategy_id: 1.0},  # Pure strategy
            research_keywords=strategy_data['research_keywords'].copy(),
            specialization_hints=self._generate_specialization_hints(strategy_id, strategy_data),
            scoring_features=self._generate_initial_scoring_features(strategy_data)
        )
        
        # Calculate embedding
        prototype.embedding = self._calculate_strategy_embedding(prototype)
        
        return prototype
    
    def _create_variant_prototype(self, 
                                variant_id: str,
                                parent_prototype: DesignPrototype, 
                                requirements: Dict[str, Any],
                                generation: int,
                                variant_index: int) -> DesignPrototype:
        """Create a variant prototype by modifying parent"""
        
        # Copy parent prototype
        variant = DesignPrototype(
            prototype_id=variant_id,
            hierarchy_level=generation,
            generation=generation,
            node_path=parent_prototype.node_path + [variant_id],
            spatial_config=parent_prototype.spatial_config.copy(),
            circulation_pattern=parent_prototype.circulation_pattern.copy(),
            environmental_strategy=parent_prototype.environmental_strategy.copy(),
            strategy_composition=parent_prototype.strategy_composition.copy(),
        )
        
        # Apply variant mutations
        self._apply_variant_mutations(variant, variant_index, requirements)
        
        # Update research keywords and hints
        variant.research_keywords = self._expand_research_keywords(
            parent_prototype.research_keywords, variant_index
        )
        variant.specialization_hints = self._update_specialization_hints(
            parent_prototype.specialization_hints, variant
        )
        variant.scoring_features = self._update_scoring_features(
            parent_prototype.scoring_features, variant
        )
        
        # Recalculate embedding
        variant.embedding = self._calculate_strategy_embedding(variant)
        
        return variant
    
    def _generate_synthesis_nodes(self, requirements: Dict[str, Any]):
        """Generate synthesis nodes that combine multiple strategies"""
        
        synthesis_count = 0
        max_synthesis = min(50, len(self.nodes) // 4)
        
        # Find synthesis candidates based on complementary strategies
        synthesis_pairs = self._identify_synthesis_candidates()
        
        for (node1_id, node2_id, synergy_score) in synthesis_pairs:
            if synthesis_count >= max_synthesis:
                break
            
            if synergy_score > self.synthesis_threshold:
                synthesis_id = f"synthesis_{node1_id}_{node2_id}_{synthesis_count}"
                
                # Create synthesis prototype
                synthesis_prototype = self._create_synthesis_prototype(
                    synthesis_id, 
                    self.prototypes[node1_id], 
                    self.prototypes[node2_id],
                    synergy_score,
                    requirements
                )
                
                # Create synthesis node
                synthesis_node = GraphNode(
                    node_id=synthesis_id,
                    node_type='synthesis',
                    level=max(self.nodes[node1_id].level, self.nodes[node2_id].level) + 1,
                    generation=self.current_generation + 1,
                    parent_ids=[node1_id, node2_id],
                    synthesis_sources=[node1_id, node2_id],
                    branch_potential=0.9,  # High potential for synthesis
                    research_priority=0.8
                )
                
                # Prepare synthesis-specific research queries
                synthesis_node.research_queries = self._prepare_synthesis_research_queries(
                    synthesis_prototype, node1_id, node2_id
                )
                
                # Store structures
                synthesis_node.prototype = synthesis_prototype
                self.nodes[synthesis_id] = synthesis_node
                self.prototypes[synthesis_id] = synthesis_prototype
                
                # Add to graph with multiple parent edges
                self.thought_graph.add_node(synthesis_id, **asdict(synthesis_node))
                self.thought_graph.add_edge(node1_id, synthesis_id, relationship='synthesis_input')
                self.thought_graph.add_edge(node2_id, synthesis_id, relationship='synthesis_input')
                
                # Update parent nodes
                self.nodes[node1_id].child_ids.append(synthesis_id)
                self.nodes[node2_id].child_ids.append(synthesis_id)
                
                synthesis_count += 1
                self.total_prototypes += 1
        
        logger.info(f"🔄 Created {synthesis_count} synthesis nodes")
    
    def _identify_synthesis_candidates(self) -> List[Tuple[str, str, float]]:
        """Identify pairs of nodes that would create good synthesis"""
        
        candidates = []
        
        # Get nodes that are synthesis candidates
        synthesis_candidates = [
            node_id for node_id, node in self.nodes.items() 
            if node.synthesis_candidate and node.node_type in ['root', 'variant']
        ]
        
        # Calculate pairwise synergy scores
        for i, node1_id in enumerate(synthesis_candidates):
            for node2_id in synthesis_candidates[i+1:]:
                synergy_score = self._calculate_synergy_score(node1_id, node2_id)
                if synergy_score > 0.5:  # Minimum threshold
                    candidates.append((node1_id, node2_id, synergy_score))
        
        # Sort by synergy score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates
    
    def _calculate_synergy_score(self, node1_id: str, node2_id: str) -> float:
        """Calculate synergy potential between two nodes"""
        
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        proto1 = node1.prototype
        proto2 = node2.prototype
        
        # Base synergy from predefined affinities
        synergy = 0.0
        
        # Check if strategies have explicit affinity
        for strategy1, weight1 in proto1.strategy_composition.items():
            for strategy2, weight2 in proto2.strategy_composition.items():
                if strategy1 in self.base_strategies:
                    affinities = self.base_strategies[strategy1].get('synthesis_affinity', [])
                    if strategy2 in affinities:
                        synergy += weight1 * weight2 * 0.8
        
        # Complementary characteristics
        if proto1.spatial_config.get('type') != proto2.spatial_config.get('type'):
            synergy += 0.3  # Different types complement
        
        # Environmental compatibility
        env1_strategies = proto1.environmental_strategy.get('passive_strategies', [])
        env2_strategies = proto2.environmental_strategy.get('passive_strategies', [])
        common_env = len(set(env1_strategies) & set(env2_strategies))
        if common_env > 0:
            synergy += common_env * 0.2
        
        # Embedding similarity (not too similar, not too different)
        if proto1.embedding is not None and proto2.embedding is not None:
            similarity = np.dot(proto1.embedding, proto2.embedding) / (
                np.linalg.norm(proto1.embedding) * np.linalg.norm(proto2.embedding)
            )
            # Sweet spot around 0.3-0.7 similarity
            if 0.3 <= similarity <= 0.7:
                synergy += 0.4
        
        return min(1.0, synergy)
    
    def _create_synthesis_prototype(self, 
                                  synthesis_id: str,
                                  proto1: DesignPrototype,
                                  proto2: DesignPrototype,
                                  synergy_score: float,
                                  requirements: Dict[str, Any]) -> DesignPrototype:
        """Create a synthesis prototype combining two parent prototypes"""
        
        # Determine blend ratios based on synergy and performance
        blend_ratio1 = 0.6 if synergy_score > 0.8 else 0.5
        blend_ratio2 = 1.0 - blend_ratio1
        
        # Combine strategy compositions
        combined_composition = {}
        for strategy, weight in proto1.strategy_composition.items():
            combined_composition[strategy] = weight * blend_ratio1
        for strategy, weight in proto2.strategy_composition.items():
            if strategy in combined_composition:
                combined_composition[strategy] += weight * blend_ratio2
            else:
                combined_composition[strategy] = weight * blend_ratio2
        
        # Synthesize spatial configuration
        spatial_config = self._synthesize_spatial_config(
            proto1.spatial_config, proto2.spatial_config, blend_ratio1
        )
        
        # Synthesize circulation pattern  
        circulation_pattern = self._synthesize_circulation_pattern(
            proto1.circulation_pattern, proto2.circulation_pattern, blend_ratio1
        )
        
        # Synthesize environmental strategy
        environmental_strategy = self._synthesize_environmental_strategy(
            proto1.environmental_strategy, proto2.environmental_strategy
        )
        
        # Create synthesis prototype
        synthesis = DesignPrototype(
            prototype_id=synthesis_id,
            hierarchy_level=max(proto1.hierarchy_level, proto2.hierarchy_level) + 1,
            generation=max(proto1.generation, proto2.generation) + 1,
            node_path=proto1.node_path + proto2.node_path + [synthesis_id],
            spatial_config=spatial_config,
            circulation_pattern=circulation_pattern,
            environmental_strategy=environmental_strategy,
            strategy_composition=combined_composition,
            synthesis_rationale=f"Synthesis of {list(proto1.strategy_composition.keys())} and {list(proto2.strategy_composition.keys())}",
            cross_pollination_sources=[proto1.prototype_id, proto2.prototype_id],
            research_keywords=list(set(proto1.research_keywords + proto2.research_keywords)),
            specialization_hints=self._merge_specialization_hints(
                proto1.specialization_hints, proto2.specialization_hints
            ),
            scoring_features=self._synthesize_scoring_features(
                proto1.scoring_features, proto2.scoring_features, synergy_score
            )
        )
        
        # Calculate new embedding
        synthesis.embedding = self._calculate_synthesis_embedding(proto1, proto2, blend_ratio1)
        
        return synthesis
    
    def _add_cross_connections(self):
        """Add lateral cross-connections between nodes to create true graph structure"""
        
        cross_connections = 0
        max_connections = int(len(self.nodes) * self.cross_connection_rate)
        
        # Group nodes by generation for lateral connections
        nodes_by_generation = defaultdict(list)
        for node_id, node in self.nodes.items():
            nodes_by_generation[node.generation].append(node_id)
        
        # Add lateral connections within each generation
        for generation, node_ids in nodes_by_generation.items():
            if len(node_ids) < 2:
                continue
            
            # Calculate cross-connection potential between pairs
            for i, node1_id in enumerate(node_ids):
                for node2_id in node_ids[i+1:]:
                    if cross_connections >= max_connections:
                        break
                    
                    connection_strength = self._calculate_cross_connection_strength(
                        node1_id, node2_id
                    )
                    
                    if connection_strength > 0.6:  # Threshold for connection
                        # Add bidirectional cross-connection
                        self.nodes[node1_id].cross_connections.append(node2_id)
                        self.nodes[node2_id].cross_connections.append(node1_id)
                        
                        # Add to graph
                        self.thought_graph.add_edge(
                            node1_id, node2_id, 
                            relationship='cross_connection',
                            strength=connection_strength
                        )
                        self.thought_graph.add_edge(
                            node2_id, node1_id,
                            relationship='cross_connection', 
                            strength=connection_strength
                        )
                        
                        cross_connections += 2
        
        # Add skip-level connections (connect across generations)
        skip_connections = self._add_skip_level_connections()
        cross_connections += skip_connections
        
        logger.info(f"🕸️ Added {cross_connections} cross-connections")
    
    def _calculate_cross_connection_strength(self, node1_id: str, node2_id: str) -> float:
        """Calculate strength of potential cross-connection"""
        
        proto1 = self.prototypes[node1_id]
        proto2 = self.prototypes[node2_id]
        
        strength = 0.0
        
        # Embedding similarity (moderate similarity preferred)
        if proto1.embedding is not None and proto2.embedding is not None:
            similarity = np.dot(proto1.embedding, proto2.embedding) / (
                np.linalg.norm(proto1.embedding) * np.linalg.norm(proto2.embedding)
            )
            if 0.4 <= similarity <= 0.8:  # Sweet spot for cross-connections
                strength += 0.4
        
        # Complementary research keywords
        common_keywords = set(proto1.research_keywords) & set(proto2.research_keywords)
        if common_keywords:
            strength += len(common_keywords) * 0.1
        
        # Different primary strategies but compatible
        proto1_primary = max(proto1.strategy_composition, key=proto1.strategy_composition.get)
        proto2_primary = max(proto2.strategy_composition, key=proto2.strategy_composition.get)
        
        if proto1_primary != proto2_primary:
            if proto1_primary in self.base_strategies:
                affinities = self.base_strategies[proto1_primary].get('synthesis_affinity', [])
                if proto2_primary in affinities:
                    strength += 0.5
        
        return min(1.0, strength)
    
    def _add_skip_level_connections(self) -> int:
        """Add connections that skip hierarchical levels"""
        
        skip_connections = 0
        max_skip = min(20, len(self.nodes) // 10)
        
        # Find promising deep nodes that could connect to earlier generations
        deep_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.generation >= 3 and node.branch_potential > 0.7
        ]
        
        early_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.generation <= 1 and node.node_type in ['root', 'variant']
        ]
        
        for deep_id in deep_nodes[:max_skip//2]:
            for early_id in early_nodes:
                connection_potential = self._calculate_skip_connection_potential(deep_id, early_id)
                
                if connection_potential > 0.7:
                    # Add skip connection
                    self.nodes[deep_id].cross_connections.append(early_id)
                    self.thought_graph.add_edge(
                        early_id, deep_id,
                        relationship='skip_connection',
                        strength=connection_potential
                    )
                    skip_connections += 1
                    break  # One skip connection per deep node
        
        return skip_connections
    
    def _calculate_skip_connection_potential(self, deep_id: str, early_id: str) -> float:
        """Calculate potential for skip-level connection"""
        
        deep_proto = self.prototypes[deep_id]
        early_proto = self.prototypes[early_id]
        
        # Check if deep node has evolved from early node's strategy family
        early_strategies = set(early_proto.strategy_composition.keys())
        deep_strategies = set(deep_proto.strategy_composition.keys())
        
        overlap = early_strategies & deep_strategies
        if overlap:
            return 0.8  # Strong connection if strategy overlap
        
        # Check for complementary evolution
        if deep_proto.embedding is not None and early_proto.embedding is not None:
            similarity = np.dot(deep_proto.embedding, early_proto.embedding) / (
                np.linalg.norm(deep_proto.embedding) * np.linalg.norm(early_proto.embedding)
            )
            if 0.3 <= similarity <= 0.6:  # Moderate similarity suggests evolution
                return 0.7
        
        return 0.0
    
    def _prepare_pipeline_output(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive output for the research -> specializer -> scoring pipeline"""
        
        # Collect all prototypes
        all_prototypes = []
        for prototype in self.prototypes.values():
            all_prototypes.append(prototype.to_dict())
        
        # Prepare graph structure data
        graph_data = {
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'edges': [(u, v, d) for u, v, d in self.thought_graph.edges(data=True)],
            'node_count': len(self.nodes),
            'edge_count': self.thought_graph.number_of_edges(),
            'generations': dict(self.nodes_per_generation)
        }
        
        # Prepare research queries for research agent
        research_queries = []
        for node in self.nodes.values():
            for query in node.research_queries:
                research_queries.append({
                    'node_id': node.node_id,
                    'prototype_id': node.prototype.prototype_id if node.prototype else None,
                    'priority': node.research_priority,
                    'query_data': query
                })
        
        # Prepare specialization hints for specializer
        specialization_data = {}
        for prototype in self.prototypes.values():
            specialization_data[prototype.prototype_id] = {
                'hints': prototype.specialization_hints,
                'strategy_composition': prototype.strategy_composition,
                'synthesis_rationale': prototype.synthesis_rationale,
                'cross_pollination_sources': prototype.cross_pollination_sources
            }
        
        # Prepare scoring features for scoring agent
        scoring_preparation = {}
        for prototype in self.prototypes.values():
            scoring_preparation[prototype.prototype_id] = {
                'initial_features': prototype.scoring_features,
                'complexity_indicators': {
                    'strategy_count': len(prototype.strategy_composition),
                    'synthesis_depth': len(prototype.cross_pollination_sources),
                    'generation_depth': prototype.generation
                },
                'diversity_markers': {
                    'research_keyword_count': len(prototype.research_keywords),
                    'unique_characteristics': self._extract_unique_characteristics(prototype)
                }
            }
        
        return {
            'prototypes': all_prototypes,
            'graph_structure': graph_data,
            'research_queries': sorted(research_queries, key=lambda x: x['priority'], reverse=True),
            'specialization_data': specialization_data,
            'scoring_preparation': scoring_preparation,
            'generation_statistics': {
                'total_prototypes': self.total_prototypes,
                'generations_created': max(self.nodes_per_generation.keys()) if self.nodes_per_generation else 0,
                'synthesis_nodes': len([n for n in self.nodes.values() if n.node_type == 'synthesis']),
                'cross_connections': sum(len(n.cross_connections) for n in self.nodes.values()) // 2,
                'strategy_diversity': len(set().union(*[p.strategy_composition.keys() for p in self.prototypes.values()]))
            },
            'requirements_context': requirements
        }
    
    def _create_minimal_fallback(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal fallback if generation fails"""
        
        fallback_prototype = DesignPrototype(
            prototype_id="fallback_minimal",
            hierarchy_level=0,
            generation=0,
            spatial_config={'strategy': 'linear_spine', 'plot_utilization': 0.7},
            circulation_pattern={'pattern_type': 'linear_spine', 'efficiency_target': 0.8},
            environmental_strategy={'orientation': 'south', 'passive_strategies': ['natural_ventilation']},
            strategy_composition={'linear_spine': 1.0},
            research_keywords=['linear planning', 'simple layout'],
            specialization_hints={'focus': 'basic_functionality'},
            scoring_features={'complexity': 0.3, 'feasibility': 0.9}
        )
        
        return {
            'prototypes': [fallback_prototype.to_dict()],
            'graph_structure': {'nodes': {}, 'edges': [], 'node_count': 1, 'edge_count': 0},
            'research_queries': [],
            'specialization_data': {},
            'scoring_preparation': {},
            'generation_statistics': {'total_prototypes': 1, 'fallback_mode': True}
        }
    
    # Helper methods for prototype creation and synthesis
    def _generate_spatial_config_from_strategy(self, strategy_id: str, strategy_data: Dict, requirements: Dict) -> Dict[str, Any]:
        """Generate spatial configuration from strategy definition"""
        
        characteristics = strategy_data['characteristics']
        spatial_needs = requirements.get('spatial_needs', [])
        site_constraints = requirements.get('site_constraints', {})
        
        config = {
            'strategy': strategy_id,
            'strategy_type': strategy_data['type'],
            'plot_utilization': 0.75,
            'primary_characteristics': list(characteristics.keys())
        }
        
        # Strategy-specific configurations
        if strategy_id == 'central_courtyard':
            config.update({
                'core_type': characteristics.get('core_type', 'open_courtyard'),
                'courtyard_ratio': 0.25,
                'perimeter_organization': True,
                'privacy_model': 'inward_facing'
            })
        
        elif strategy_id == 'linear_spine':
            config.update({
                'organization': 'sequential_zones',
                'spine_position': 'central',
                'zone_progression': ['public', 'semi_private', 'private'],
                'expansion_capability': True
            })
        
        elif strategy_id == 'modular_flexibility':
            config.update({
                'module_size': (3.0, 3.0),
                'grid_system': True,
                'reconfiguration_capability': True,
                'standardized_components': True
            })
        
        return config
    
    def _generate_circulation_from_strategy(self, strategy_id: str, strategy_data: Dict, spatial_config: Dict) -> Dict[str, Any]:
        """Generate circulation pattern from strategy"""
        
        circulation = {
            'strategy_source': strategy_id,
            'efficiency_target': 0.85,
            'accessibility_compliant': True
        }
        
        # Map strategies to circulation patterns
        if strategy_id == 'central_courtyard':
            circulation.update({
                'pattern_type': 'perimeter_circulation',
                'courtyard_integration': True,
                'outdoor_connections': 'direct'
            })
        
        elif strategy_id == 'linear_spine':
            circulation.update({
                'pattern_type': 'spine_circulation',
                'spine_width': 1.5,
                'branch_points': 3,
                'privacy_control': 'graduated'
            })
        
        elif strategy_id == 'hub_spoke_circulation':
            circulation.update({
                'pattern_type': 'radial_circulation',
                'central_hub_size': 2.5,
                'spoke_count': 4,
                'hub_function': 'multi_purpose'
            })
        
        return circulation
    
    def _generate_environmental_from_strategy(self, strategy_id: str, strategy_data: Dict, requirements: Dict) -> Dict[str, Any]:
        """Generate environmental strategy from base strategy"""
        
        site_constraints = requirements.get('site_constraints', {})
        orientation = site_constraints.get('orientation', 'south')
        
        env_strategy = {
            'strategy_source': strategy_id,
            'climate_zone': 'subtropical',
            'orientation': orientation,
            'passive_strategies': []
        }
        
        # Strategy-specific environmental approaches
        if strategy_id == 'passive_solar_design':
            env_strategy['passive_strategies'].extend([
                'solar_orientation', 'thermal_mass', 'natural_daylighting', 'stack_ventilation'
            ])
        
        elif strategy_id == 'cross_ventilation':
            env_strategy['passive_strategies'].extend([
                'cross_breeze', 'pressure_differentials', 'natural_cooling'
            ])
        
        elif strategy_id == 'central_courtyard':
            env_strategy['passive_strategies'].extend([
                'courtyard_cooling', 'stack_effect', 'microclimate_creation'
            ])
        
        # Add default strategies if none specified
        if not env_strategy['passive_strategies']:
            env_strategy['passive_strategies'] = ['natural_ventilation', 'daylighting']
        
        return env_strategy

    # Implement the missing methods from the original code
    def _apply_variant_mutations(self, variant: DesignPrototype, variant_index: int, requirements: Dict[str, Any]):
        """Apply mutations to create variant from parent"""
        
        # Mutation strategies based on variant index
        mutation_types = ['parameter_adjustment', 'feature_addition', 'constraint_relaxation', 'hybrid_injection']
        mutation_type = mutation_types[variant_index % len(mutation_types)]
        
        if mutation_type == 'parameter_adjustment':
            # Adjust numerical parameters
            if 'plot_utilization' in variant.spatial_config:
                adjustment = (variant_index - 2) * 0.05  # -0.1 to +0.1
                variant.spatial_config['plot_utilization'] = max(0.5, min(0.9, 
                    variant.spatial_config['plot_utilization'] + adjustment))
            
        elif mutation_type == 'feature_addition':
            # Add new features
            new_features = ['smart_systems', 'renewable_energy', 'universal_design', 'outdoor_integration']
            if variant_index < len(new_features):
                feature = new_features[variant_index]
                variant.environmental_strategy['additional_features'] = [feature]
                variant.research_keywords.append(feature)
        
        elif mutation_type == 'constraint_relaxation':
            # Relax certain constraints
            variant.spatial_config['flexibility_factor'] = 1.0 + (variant_index * 0.1)
            
        elif mutation_type == 'hybrid_injection':
            # Inject elements from other strategies
            other_strategies = [s for s in self.base_strategies.keys() 
                             if s not in variant.strategy_composition]
            if other_strategies and variant_index < len(other_strategies):
                inject_strategy = other_strategies[variant_index]
                variant.strategy_composition[inject_strategy] = 0.2
                variant.research_keywords.extend(self.base_strategies[inject_strategy]['research_keywords'][:2])
    
    def _synthesize_spatial_config(self, config1: Dict, config2: Dict, blend_ratio: float) -> Dict[str, Any]:
        """Synthesize spatial configurations from two prototypes"""
        
        synthesized = {}
        
        # Combine strategy identifiers
        synthesized['strategies'] = [config1.get('strategy', 'unknown'), config2.get('strategy', 'unknown')]
        synthesized['synthesis_ratio'] = [blend_ratio, 1.0 - blend_ratio]
        
        # Blend numerical parameters
        for key in ['plot_utilization', 'courtyard_ratio']:
            if key in config1 and key in config2:
                synthesized[key] = config1[key] * blend_ratio + config2[key] * (1.0 - blend_ratio)
            elif key in config1:
                synthesized[key] = config1[key]
            elif key in config2:
                synthesized[key] = config2[key]
        
        # Combine characteristics
        chars1 = config1.get('primary_characteristics', [])
        chars2 = config2.get('primary_characteristics', [])
        synthesized['primary_characteristics'] = list(set(chars1 + chars2))
        
        # Synthesis-specific additions
        synthesized['synthesis_features'] = {
            'hybrid_organization': True,
            'multi_strategy_integration': True,
            'enhanced_flexibility': True
        }
        
        return synthesized
    
    def _synthesize_circulation_pattern(self, circ1: Dict, circ2: Dict, blend_ratio: float) -> Dict[str, Any]:
        """Synthesize circulation patterns"""
        
        synthesized = {
            'pattern_types': [circ1.get('pattern_type', 'unknown'), circ2.get('pattern_type', 'unknown')],
            'synthesis_ratio': [blend_ratio, 1.0 - blend_ratio],
            'hybrid_circulation': True
        }
        
        # Blend efficiency targets
        eff1 = circ1.get('efficiency_target', 0.85)
        eff2 = circ2.get('efficiency_target', 0.85)
        synthesized['efficiency_target'] = eff1 * blend_ratio + eff2 * (1.0 - blend_ratio)
        
        # Combine features
        features = []
        if 'courtyard_integration' in circ1:
            features.append('courtyard_integration')
        if 'privacy_control' in circ2:
            features.append('privacy_control')
        
        synthesized['combined_features'] = features
        
        return synthesized
    
    def _synthesize_environmental_strategy(self, env1: Dict, env2: Dict) -> Dict[str, Any]:
        """Synthesize environmental strategies"""
        
        # Combine passive strategies
        strategies1 = env1.get('passive_strategies', [])
        strategies2 = env2.get('passive_strategies', [])
        combined_strategies = list(set(strategies1 + strategies2))
        
        synthesized = {
            'synthesis_sources': [env1.get('strategy_source', 'unknown'), env2.get('strategy_source', 'unknown')],
            'passive_strategies': combined_strategies,
            'enhanced_performance': True,
            'multi_modal_approach': True
        }
        
        # Take the better orientation if different
        if env1.get('orientation') == env2.get('orientation'):
            synthesized['orientation'] = env1.get('orientation')
        else:
            synthesized['orientation'] = 'optimized_hybrid'
        
        return synthesized
    
    def _prepare_research_queries(self, strategy_id: str, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare research queries for a strategy"""
        
        strategy_data = self.base_strategies[strategy_id]
        
        queries = []
        
        # Primary strategy research
        queries.append({
            'query_type': 'strategy_precedents',
            'keywords': strategy_data['research_keywords'],
            'focus': 'contemporary_examples',
            'priority': 0.9,
            'expected_insights': ['spatial_organization', 'circulation_efficiency', 'environmental_performance']
        })
        
        # Climate-specific research
        queries.append({
            'query_type': 'climate_adaptation',
            'keywords': strategy_data['research_keywords'] + ['tropical', 'subtropical', 'monsoon'],
            'focus': 'climate_responsive_design',
            'priority': 0.8,
            'expected_insights': ['passive_cooling', 'monsoon_protection', 'thermal_comfort']
        })
        
        # Technology integration research
        if requirements.get('budget', 0) > 3000000:
            queries.append({
                'query_type': 'technology_integration',
                'keywords': strategy_data['research_keywords'] + ['smart_home', 'automation'],
                'focus': 'intelligent_systems',
                'priority': 0.6,
                'expected_insights': ['automation_opportunities', 'smart_integration', 'future_proofing']
            })
        
        return queries
    
    def _prepare_variant_research_queries(self, variant: DesignPrototype, parent: DesignPrototype) -> List[Dict[str, Any]]:
        """Prepare research queries for variant prototypes"""
        
        queries = []
        
        # Research the specific mutations/variations
        variant_keywords = [kw for kw in variant.research_keywords if kw not in parent.research_keywords]
        
        if variant_keywords:
            queries.append({
                'query_type': 'variant_exploration',
                'keywords': variant_keywords,
                'focus': 'innovation_opportunities',
                'priority': 0.7,
                'expected_insights': ['novel_approaches', 'performance_improvements', 'design_innovation']
            })
        
        # Research performance optimization
        queries.append({
            'query_type': 'performance_optimization',
            'keywords': parent.research_keywords[:3],  # Top 3 keywords from parent
            'focus': 'efficiency_improvements',
            'priority': 0.6,
            'expected_insights': ['space_efficiency', 'cost_optimization', 'construction_methods']
        })
        
        return queries
    
    def _prepare_synthesis_research_queries(self, synthesis: DesignPrototype, node1_id: str, node2_id: str) -> List[Dict[str, Any]]:
        """Prepare research queries for synthesis prototypes"""
        
        queries = []
        
        # Research hybrid precedents
        parent1_strategies = list(self.prototypes[node1_id].strategy_composition.keys())
        parent2_strategies = list(self.prototypes[node2_id].strategy_composition.keys())
        
        queries.append({
            'query_type': 'hybrid_precedents',
            'keywords': synthesis.research_keywords[:5],  # Top 5 combined keywords
            'focus': 'successful_combinations',
            'priority': 0.85,
            'expected_insights': ['integration_strategies', 'synergy_benefits', 'implementation_challenges'],
            'synthesis_context': {
                'parent_strategies': parent1_strategies + parent2_strategies,
                'combination_rationale': synthesis.synthesis_rationale
            }
        })
        
        # Research integration challenges
        queries.append({
            'query_type': 'integration_challenges',
            'keywords': ['hybrid design', 'strategy integration', 'multi-modal architecture'],
            'focus': 'implementation_feasibility',
            'priority': 0.7,
            'expected_insights': ['technical_challenges', 'cost_implications', 'performance_trade-offs']
        })
        
        return queries
    
    def _generate_specialization_hints(self, strategy_id: str, strategy_data: Dict) -> Dict[str, Any]:
        """Generate hints for the specializer agent"""
        
        return {
            'primary_focus': strategy_data['type'],
            'key_characteristics': list(strategy_data['characteristics'].keys()),
            'optimization_targets': self._get_optimization_targets(strategy_id),
            'critical_parameters': self._get_critical_parameters(strategy_id),
            'performance_metrics': self._get_performance_metrics(strategy_id),
            'specialization_priority': strategy_data['branch_potential']
        }
    
    def _get_optimization_targets(self, strategy_id: str) -> List[str]:
        """Get optimization targets for specializer"""
        
        targets_map = {
            'central_courtyard': ['courtyard_sizing', 'perimeter_efficiency', 'climate_response'],
            'linear_spine': ['circulation_efficiency', 'zone_transitions', 'expansion_flexibility'],
            'passive_solar_design': ['solar_gain_optimization', 'thermal_mass_placement', 'shading_efficiency'],
            'modular_flexibility': ['module_sizing', 'connection_details', 'reconfiguration_ease'],
            'smart_home_integration': ['system_integration', 'user_interface', 'automation_logic']
        }
        
        return targets_map.get(strategy_id, ['spatial_efficiency', 'functional_organization', 'cost_optimization'])
    
    def _get_critical_parameters(self, strategy_id: str) -> List[str]:
        """Get critical parameters that need careful optimization"""
        
        params_map = {
            'central_courtyard': ['courtyard_ratio', 'perimeter_width', 'opening_ratios'],
            'linear_spine': ['spine_width', 'zone_proportions', 'transition_distances'],
            'passive_solar_design': ['window_wall_ratio', 'overhang_dimensions', 'thermal_mass_ratio'],
            'modular_flexibility': ['module_dimensions', 'connection_tolerances', 'structural_grid'],
            'cross_ventilation': ['opening_sizes', 'pressure_differentials', 'airflow_paths']
        }
        
        return params_map.get(strategy_id, ['plot_utilization', 'circulation_ratio', 'room_proportions'])
    
    def _get_performance_metrics(self, strategy_id: str) -> List[str]:
        """Get key performance metrics for scoring"""
        
        metrics_map = {
            'central_courtyard': ['climate_performance', 'privacy_score', 'outdoor_integration'],
            'linear_spine': ['circulation_efficiency', 'privacy_gradient', 'expansion_potential'],
            'passive_solar_design': ['energy_performance', 'thermal_comfort', 'daylighting_quality'],
            'modular_flexibility': ['adaptability_score', 'construction_efficiency', 'lifecycle_flexibility'],
            'smart_home_integration': ['automation_coverage', 'user_experience', 'technology_integration']
        }
        
        return metrics_map.get(strategy_id, ['spatial_efficiency', 'functional_score', 'cost_effectiveness'])
    
    def _generate_initial_scoring_features(self, strategy_data: Dict) -> Dict[str, float]:
        """Generate initial scoring features for the scoring agent"""
        
        return {
            'branch_potential': strategy_data['branch_potential'],
            'complexity_estimate': self._estimate_complexity(strategy_data),
            'innovation_potential': self._estimate_innovation_potential(strategy_data),
            'implementation_feasibility': self._estimate_feasibility(strategy_data),
            'market_relevance': 0.7,  # Default assumption
            'sustainability_score': self._estimate_sustainability(strategy_data)
        }
    
    def _estimate_complexity(self, strategy_data: Dict) -> float:
        """Estimate design complexity"""
        
        complexity_factors = {
            'spatial_org': 0.6,
            'circulation': 0.5,
            'environmental': 0.7,
            'cultural': 0.8,
            'flexibility': 0.6,
            'sustainability': 0.8,
            'technology': 0.9
        }
        
        return complexity_factors.get(strategy_data['type'], 0.6)
    
    def _estimate_innovation_potential(self, strategy_data: Dict) -> float:
        """Estimate innovation potential"""
        
        # More characteristics = higher innovation potential
        char_count = len(strategy_data['characteristics'])
        base_innovation = min(1.0, char_count / 5.0)
        
        # Certain types have higher innovation potential
        innovation_multipliers = {
            'technology': 1.2,
            'sustainability': 1.1,
            'flexibility': 1.1,
            'environmental': 1.05
        }
        
        multiplier = innovation_multipliers.get(strategy_data['type'], 1.0)
        return min(1.0, base_innovation * multiplier)
    
    def _estimate_feasibility(self, strategy_data: Dict) -> float:
        """Estimate implementation feasibility"""
        
        feasibility_scores = {
            'spatial_org': 0.9,
            'circulation': 0.95,
            'environmental': 0.8,
            'cultural': 0.85,
            'flexibility': 0.7,
            'sustainability': 0.6,
            'technology': 0.5
        }
        
        return feasibility_scores.get(strategy_data['type'], 0.7)
    
    def _estimate_sustainability(self, strategy_data: Dict) -> float:
        """Estimate sustainability score"""
        
        sustainability_scores = {
            'environmental': 0.95,
            'sustainability': 1.0,
            'cultural': 0.8,
            'spatial_org': 0.6,
            'circulation': 0.7,
            'flexibility': 0.8,
            'technology': 0.7
        }
        
        return sustainability_scores.get(strategy_data['type'], 0.6)
    
    def _update_specialization_hints(self, parent_hints: Dict, variant: DesignPrototype) -> Dict[str, Any]:
        """Update specialization hints for variant"""
        
        updated_hints = parent_hints.copy()
        
        # Add variant-specific hints
        updated_hints['variant_focus'] = 'parameter_optimization'
        updated_hints['inheritance_source'] = parent_hints.get('primary_focus', 'unknown')
        updated_hints['exploration_areas'] = list(variant.strategy_composition.keys())
        
        return updated_hints
    
    def _update_scoring_features(self, parent_features: Dict, variant: DesignPrototype) -> Dict[str, float]:
        """Update scoring features for variant"""
        
        updated_features = parent_features.copy()
        
        # Modify based on variant characteristics
        updated_features['novelty_bonus'] = 0.1  # Variants get novelty bonus
        updated_features['exploration_depth'] = variant.generation / 10.0  # Depth bonus
        
        # Adjust complexity based on strategy composition
        strategy_count = len(variant.strategy_composition)
        updated_features['complexity_estimate'] = min(1.0, parent_features.get('complexity_estimate', 0.5) + (strategy_count - 1) * 0.1)
        
        return updated_features
    
    def _merge_specialization_hints(self, hints1: Dict, hints2: Dict) -> Dict[str, Any]:
        """Merge specialization hints from two parents"""
        
        merged = {
            'synthesis_mode': True,
            'parent_focuses': [hints1.get('primary_focus', 'unknown'), hints2.get('primary_focus', 'unknown')],
            'combined_characteristics': list(set(
                hints1.get('key_characteristics', []) + hints2.get('key_characteristics', [])
            )),
            'optimization_targets': list(set(
                hints1.get('optimization_targets', []) + hints2.get('optimization_targets', [])
            )),
            'critical_parameters': list(set(
                hints1.get('critical_parameters', []) + hints2.get('critical_parameters', [])
            )),
            'performance_metrics': list(set(
                hints1.get('performance_metrics', []) + hints2.get('performance_metrics', [])
            )),
            'specialization_priority': (hints1.get('specialization_priority', 0.5) + hints2.get('specialization_priority', 0.5)) / 2
        }
        
        return merged
    
    def _synthesize_scoring_features(self, features1: Dict, features2: Dict, synergy_score: float) -> Dict[str, float]:
        """Synthesize scoring features from two parents"""
        
        synthesized = {}
        
        # Average numerical features
        for key in features1.keys() | features2.keys():
            val1 = features1.get(key, 0.5)
            val2 = features2.get(key, 0.5)
            synthesized[key] = (val1 + val2) / 2
        
        # Add synthesis-specific features
        synthesized['synthesis_bonus'] = synergy_score * 0.2
        synthesized['hybrid_complexity'] = min(1.0, synthesized.get('complexity_estimate', 0.5) + 0.2)
        synthesized['integration_challenge'] = 1.0 - synergy_score  # Higher synergy = lower integration challenge
        
        return synthesized
    
    def _calculate_strategy_embedding(self, prototype: "DesignPrototype") -> np.ndarray:
        """Calculate sophisticated embedding using Gemma 3"""
        try:
            # Convert prototype to dictionary format for encoder
            prototype_dict = prototype.to_dict()
            
            # Use Gemma 3 encoder for sophisticated embedding
            embedding = self.encoder.encode_prototype_features(prototype_dict)
            
            logger.debug(f"Generated Gemma 3 embedding: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.warning(f"Gemma 3 encoding failed, using fallback: {e}")
            # Fallback to simple embedding
            return self._simple_fallback_embedding(prototype)

    def _simple_fallback_embedding(self, prototype: "DesignPrototype") -> np.ndarray:
        """Fallback embedding if Gemma 3 fails"""
        features = []
        for strategy_id in self.base_strategies.keys():
            weight = prototype.strategy_composition.get(strategy_id, 0.0)
            features.append(weight)
        
        # Ensure fixed dimensionality (truncate/pad to 32)
        return np.array(features[:32], dtype=np.float32)
    
    def _calculate_synthesis_embedding(self, proto1: DesignPrototype, proto2: DesignPrototype, blend_ratio: float) -> np.ndarray:
        """Calculate embedding for synthesis prototype"""
        
        if proto1.embedding is not None and proto2.embedding is not None:
            # Weighted average of parent embeddings
            synthesis_embedding = (proto1.embedding * blend_ratio + 
                                 proto2.embedding * (1.0 - blend_ratio))
            
            # Add synthesis signature (small random component)
            synthesis_signature = np.random.normal(0, 0.05, synthesis_embedding.shape)
            return synthesis_embedding + synthesis_signature
        
        # Fallback to random if embeddings not available
        return np.random.rand(32).astype(np.float32)
    
    def _expand_research_keywords(self, parent_keywords: List[str], variant_index: int) -> List[str]:
        """Expand research keywords for variants"""
        
        expanded = parent_keywords.copy()
        
        # Add variant-specific keywords
        variant_keywords = [
            ['optimization', 'efficiency'],
            ['innovation', 'novel_approach'],
            ['sustainability', 'green_design'],
            ['technology', 'smart_systems'],
            ['flexibility', 'adaptability'],
            ['comfort', 'user_experience']
        ]
        
        if variant_index < len(variant_keywords):
            expanded.extend(variant_keywords[variant_index])
        
        return list(set(expanded))  # Remove duplicates
    
    def _extract_unique_characteristics(self, prototype: DesignPrototype) -> List[str]:
        """Extract unique characteristics for diversity scoring"""
        
        characteristics = []
        
        # From strategy composition
        characteristics.extend(list(prototype.strategy_composition.keys()))
        
        # From spatial config
        spatial = prototype.spatial_config
        if spatial.get('courtyard_ratio', 0) > 0:
            characteristics.append('courtyard_design')
        if spatial.get('grid_system', False):
            characteristics.append('modular_grid')
        if spatial.get('synthesis_features', {}).get('hybrid_organization', False):
            characteristics.append('hybrid_organization')
        
        # From environmental strategy
        env = prototype.environmental_strategy
        if len(env.get('passive_strategies', [])) > 3:
            characteristics.append('multi_modal_environmental')
        if 'smart' in str(env).lower():
            characteristics.append('technology_integrated')
        
        # From synthesis
        if prototype.synthesis_rationale:
            characteristics.append('synthesis_derived')
        if prototype.cross_pollination_sources:
            characteristics.append('cross_pollinated')
        
        return list(set(characteristics))
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        return {
            'graph_metrics': {
                'total_nodes': len(self.nodes),
                'total_edges': self.thought_graph.number_of_edges(),
                'graph_density': nx.density(self.thought_graph) if len(self.nodes) > 1 else 0,
                'connected_components': nx.number_weakly_connected_components(self.thought_graph),
                'average_degree': np.mean([d for _, d in self.thought_graph.degree()]) if len(self.nodes) > 0 else 0
            },
            'node_type_distribution': {
                node_type: len([n for n in self.nodes.values() if n.node_type == node_type])
                for node_type in ['root', 'variant', 'synthesis']
            },
            'generation_distribution': dict(self.nodes_per_generation),
            'strategy_diversity': {
                'unique_strategies': len(set().union(*[p.strategy_composition.keys() for p in self.prototypes.values()])),
                'pure_strategies': len([p for p in self.prototypes.values() if len(p.strategy_composition) == 1]),
                'hybrid_strategies': len([p for p in self.prototypes.values() if len(p.strategy_composition) > 1])

            },
            'research_preparation': {
                'total_research_queries': sum(len(n.research_queries) for n in self.nodes.values()),
                'high_priority_queries': len([q for n in self.nodes.values() for q in n.research_queries if n.research_priority > 0.7])
            },
            'branching_metrics': {
                'max_generation': max((n.generation for n in self.nodes.values()), default=0),
                'average_branch_factor': np.mean([n.expansion_count for n in self.nodes.values() if n.expansion_count > 0]) if any(n.expansion_count > 0 for n in self.nodes.values()) else 0,
                'cross_connections': sum(len(n.cross_connections) for n in self.nodes.values()) // 2
            }
        }


# Example usage for testing the Graph of Thoughts system
if __name__ == "__main__":
    print("🌟 Graph of Thoughts Generalizer - Pipeline Integration Test")
    print("=" * 70)
    
    # Sample requirements
    sample_requirements = {
        'spatial_needs': [
            {'room_type': 'bedroom', 'quantity': 3, 'min_area': 140},
            {'room_type': 'bathroom', 'quantity': 2, 'min_area': 50},
            {'room_type': 'living_room', 'quantity': 1, 'min_area': 220},
            {'room_type': 'kitchen', 'quantity': 1, 'min_area': 120},
            {'room_type': 'dining_room', 'quantity': 1, 'min_area': 150},
            {'room_type': 'office', 'quantity': 1, 'min_area': 100}
        ],
        'site_constraints': {
            'plot_length': 45,
            'plot_width': 35,
            'orientation': 'southeast'
        },
        'design_preferences': {
            'style': 'contemporary',
            'accessibility_requirements': False,
            'sustainability_priority': 'high'
        },
        'budget': 4000000,
        'cultural_preferences': {
            'traditional_influence': True
        }
    }
    
    # Initialize the Graph of Thoughts Generalizer
    print("🚀 Initializing Graph of Thoughts Generalizer...")
    got_generalizer = GraphOfThoughtsGeneralizer(
        max_generations=6,
        initial_strategies=6,
        branch_factor_base=3.0,
        synthesis_probability=0.4,
        cross_connection_rate=0.3
    )
    
    # Generate the design graph
    print("🌳 Generating comprehensive design graph...")
    pipeline_data = got_generalizer.generate_design_graph(
        requirements=sample_requirements,
        expansion_control={
            'max_total_nodes': 200,
            'stop_on_convergence': False,
            'research_driven_expansion': True
        }
    )
    
    # Display results
    print(f"\n📊 GRAPH GENERATION COMPLETE")
    print("=" * 50)
    
    stats = pipeline_data['generation_statistics']
    print(f"Total Prototypes Generated: {stats['total_prototypes']}")
    print(f"Generations Created: {stats['generations_created']}")
    print(f"Synthesis Nodes: {stats['synthesis_nodes']}")
    print(f"Cross-connections: {stats['cross_connections']}")
    print(f"Strategy Diversity: {stats['strategy_diversity']} unique strategies")
    
    print(f"\n🔍 RESEARCH QUERIES PREPARED")
    research_queries = pipeline_data['research_queries']
    print(f"Total Research Queries: {len(research_queries)}")
    print(f"High Priority Queries: {len([q for q in research_queries if q['priority'] > 0.8])}")
    
    # Sample some research queries
    print(f"\n📋 Sample Research Queries:")
    for i, query in enumerate(research_queries[:3]):
        print(f"{i+1}. {query['query_data']['query_type']} (Priority: {query['priority']:.2f})")
        print(f"   Keywords: {', '.join(query['query_data']['keywords'][:3])}")
        print(f"   Focus: {query['query_data']['focus']}")
    
    print(f"\n🎯 SPECIALIZATION DATA PREPARED")
    spec_data = pipeline_data['specialization_data']
    print(f"Prototypes with Specialization Hints: {len(spec_data)}")
    
    print(f"\n📈 SCORING PREPARATION COMPLETE")
    scoring_data = pipeline_data['scoring_preparation']
    print(f"Prototypes Ready for Scoring: {len(scoring_data)}")
    
    # Show graph structure
    graph_info = pipeline_data['graph_structure']
    print(f"\n🕸️ GRAPH STRUCTURE")
    print(f"Graph Nodes: {graph_info['node_count']}")
    print(f"Graph Edges: {graph_info['edge_count']}")
    print(f"Graph Density: {graph_info['edge_count'] / max(1, graph_info['node_count'] * (graph_info['node_count'] - 1)):.3f}")
    
    # Show sample prototypes
    print(f"\n🏗️ SAMPLE PROTOTYPES")
    for i, prototype in enumerate(pipeline_data['prototypes'][:5]):
        print(f"\nPrototype {i+1}: {prototype['prototype_id']}")
        print(f"  Strategies: {list(prototype['strategy_composition'].keys())}")
        print(f"  Generation: {prototype['generation']}")
        print(f"  Research Keywords: {len(prototype['research_keywords'])} keywords")
        if prototype.get('synthesis_rationale'):
            print(f"  Synthesis: {prototype['synthesis_rationale'][:50]}...")
    
    print(f"\n✅ READY FOR PIPELINE")
    print("📤 Data prepared for: Research Agent → Specializer → Scoring Agent")
    print("🎯 Graph contains extensive branching with cross-connections and synthesis")
    print("🔍 Research queries prioritized and categorized")
    print("🛠️ Specialization hints and scoring features prepared")
    print("\n🏁 Graph of Thoughts Generation Complete!")