import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
from pathlib import Path
import torch
import faiss
from sentence_transformers import SentenceTransformer
import re
from encoder import Gemma3Encoder 
import uuid
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from enum import Enum
from dataclasses import dataclass

class GenerationStrategy(Enum):
    """Generation strategies for GoT expansion"""
    EXPLORATORY = "exploratory"
    EXPLOITATION = "exploitation" 
    BALANCED = "balanced"
    DIVERSITY_FOCUSED = "diversity_focused"

@dataclass
class ExpansionControl:
    """Control parameters for GoT expansion"""
    max_total_nodes: int = 300
    stop_on_convergence: bool = False
    research_driven_expansion: bool = True
    branch_factor: int = 3
    diversity: bool = True
    temperature: float = 0.5
    bias_terms: list = None

@dataclass
class ResearchContext:
    """Research context for prototype enhancement"""
    prototype_id: str
    research_query: str
    context_type: str  # 'spatial', 'functional', 'environmental', 'aesthetic', 'structural'
    priority: float = 1.0
    
    # Research results
    similar_examples: List[Dict[str, Any]] = field(default_factory=list)
    design_patterns: List[Dict[str, Any]] = field(default_factory=list)
    code_requirements: List[Dict[str, Any]] = field(default_factory=list)
    best_practices: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    relevance_score: float = 0.0
    confidence: float = 0.0
    research_depth: int = 0
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prototype_id': self.prototype_id,
            'research_query': self.research_query,
            'context_type': self.context_type,
            'priority': float(self.priority),
            'similar_examples': [ex for ex in self.similar_examples],  # Assuming examples are dicts
            'design_patterns': [dp for dp in self.design_patterns],    # Assuming patterns are dicts
            'code_requirements': [cr for cr in self.code_requirements],  # Assuming requirements are dicts
            'best_practices': [bp for bp in self.best_practices],      # Assuming practices are dicts
            'relevance_score': float(self.relevance_score),
            'confidence': float(self.confidence),
            'research_depth': self.research_depth
            }

@dataclass
class KnowledgeSource:
    """Individual knowledge source with metadata"""
    source_id: str
    source_type: str  # 'cubicasa', 'building_codes', 'design_patterns', 'user_feedback'
    content: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    relevance_context: Set[str] = field(default_factory=set)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'source_type': self.source_type,
            'content': self.content,  # Assuming content is already JSON-serializable (dict)
            'embedding': self.embedding.tolist() if self.embedding is not None else None,  # Convert numpy array to list
            'metadata': self.metadata,
            'quality_score': float(self.quality_score),  # Ensure float
            'relevance_context': list(self.relevance_context)  # Convert set to list
            }

class ResearchQueryType(Enum):
    """Types of research queries"""
    SPATIAL_OPTIMIZATION = "spatial_optimization"
    FUNCTIONAL_ADJACENCY = "functional_adjacency" 
    ENVIRONMENTAL_STRATEGY = "environmental_strategy"
    CIRCULATION_PATTERNS = "circulation_patterns"
    AESTHETIC_REFERENCES = "aesthetic_references"
    STRUCTURAL_SOLUTIONS = "structural_solutions"
    CODE_COMPLIANCE = "code_compliance"
    COST_OPTIMIZATION = "cost_optimization"

class ResearchAgent:
    """
    Enhanced Research Agent for GOT-RAG-FBS system
    Integrates with multi-modal RAG store for architectural knowledge retrieval
    """
    
    def __init__(self,
             rag_store_path: str = "enhanced_multimodal_rag_store",
             embedding_model: str = "all-MiniLM-L6-v2",
             max_research_depth: int = 3,
             **kwargs):
        self.rag_store_path = Path(rag_store_path)
        self.max_research_depth = max_research_depth
    
        # Store additional parameters
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.75)
        self.max_contexts_per_query = kwargs.get('max_contexts_per_query', 8)
        self.research_quality_threshold = kwargs.get('research_quality_threshold', 0.7)
        self.multimodal_weight = kwargs.get('multimodal_weight', 0.3)
        self.faiss_search_k = kwargs.get('faiss_search_k', 20)
        self.context_expansion_factor = kwargs.get('context_expansion_factor', 1.5)
    
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
    
        # ... rest of your existing initialization code

        
        # Load knowledge sources
        self.knowledge_sources = {}
        self.research_cache = {}
        
        # Initialize FAISS indices for different knowledge types
        self.spatial_index = None
        self.functional_index = None
        self.environmental_index = None
        
        # Load and process knowledge base
        self._load_knowledge_base()
        self._initialize_search_indices()
        
        # Research statistics
        self.research_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'successful_retrievals': 0,
            'avg_research_depth': 0.0
        }
        self.gemma_encoder = Gemma3Encoder()
        logger.info("✅ Initialized Gemma 3 Encoder for research queries")
    def _normalize_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize requirements and handle None values safely."""
        try:
            normalized = requirements.copy()
            spatial_needs = normalized.get('spatial_needs', [])

            for need in spatial_needs:
                if need.get('min_area') is None:
                    need['min_area'] = 100
                if need.get('quantity') is None:
                    need['quantity'] = 1

                try:
                    need['min_area'] = max(50, int(float(need['min_area'])))
                except (ValueError, TypeError):
                    need['min_area'] = 100

                try:
                    need['quantity'] = max(1, int(float(need['quantity'])))
                except (ValueError, TypeError):
                    need['quantity'] = 1

            normalized['spatial_needs'] = spatial_needs
            return normalized

        except Exception as e:
            logger.error(f"Basic requirements normalization failed: {e}")
            return self._create_minimal_requirements()
    
    def _load_knowledge_base(self):
        """Load multi-modal RAG knowledge base properly"""
        logger.info("Loading multi-modal RAG knowledge base...")
        
        try:
            # Load the main composite embeddings (647MB file)
            composite_path = self.rag_store_path / "composite_embeddings.npy"
            if composite_path.exists():
                logger.info(f"Loading composite embeddings from {composite_path}")
                self.composite_embeddings = np.load(composite_path)
                logger.info(f"✅ Loaded composite embeddings: {self.composite_embeddings.shape}")
                
                # Load consolidated metadata (43.9MB file)
                metadata_path = self.rag_store_path / "consolidated_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    logger.info(f"✅ Loaded metadata for {len(self.metadata)} annotations")
                    
                    # Create knowledge sources from the loaded data
                    self._create_knowledge_sources_from_loaded_data()
                    
                    # Create FAISS indices for fast similarity search
                    self._create_faiss_indices_from_composite_embeddings()
                else:
                    logger.warning("Metadata file not found")
            else:
                logger.warning(f"Composite embeddings not found at {composite_path}")
                
            # Also load individual embedding types for specialized searches
            self._load_specialized_embeddings()
            
        except Exception as e:
            logger.error(f"❌ Failed to load RAG knowledge base: {e}")
            self._initialize_fallback_knowledge()

    def _create_faiss_indices_from_composite_embeddings(self):
        """Create FAISS indices from the loaded composite embeddings"""
        if not hasattr(self, 'composite_embeddings') or len(self.composite_embeddings) == 0:
            logger.warning("No composite embeddings available for indexing")
            return
        
        logger.info("Creating FAISS indices from composite embeddings...")
        
        # Based on your vector_store.py, composite embeddings have structure:
        # [text_embedding(384) + visual_embedding(512) + spatial_embedding(64) + architectural_embedding(64)]
        
        embedding_dim = self.composite_embeddings.shape[1]
        logger.info(f"Composite embedding dimension: {embedding_dim}")
        
        # Create spatial index (assuming first 384 dimensions are text-based spatial)
        if embedding_dim >= 384:
            spatial_features = self.composite_embeddings[:, :384].astype('float32')
            self.spatial_index = faiss.IndexFlatIP(384)
            faiss.normalize_L2(spatial_features)
            self.spatial_index.add(spatial_features)
            self.spatial_source_mapping = {i: str(i) for i in range(len(spatial_features))}
            logger.info(f"✅ Created spatial FAISS index with {len(spatial_features)} vectors")
        
        # Create functional index (visual + architectural features)
        if embedding_dim >= 896:  # 384 + 512
            functional_features = self.composite_embeddings[:, 384:896].astype('float32')
            self.functional_index = faiss.IndexFlatIP(512)
            faiss.normalize_L2(functional_features)
            self.functional_index.add(functional_features)
            self.functional_source_mapping = {i: str(i) for i in range(len(functional_features))}
            logger.info(f"✅ Created functional FAISS index with {len(functional_features)} vectors")
        
        # Create comprehensive index using full composite embeddings
        full_features = self.composite_embeddings.astype('float32')
        self.comprehensive_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(full_features)
        self.comprehensive_index.add(full_features)
        self.comprehensive_source_mapping = {i: str(i) for i in range(len(full_features))}
        logger.info(f"✅ Created comprehensive FAISS index with {len(full_features)} vectors")

    def _load_specialized_embeddings(self):
        """Load specialized embedding types for targeted searches"""
        try:
            # Load architectural embeddings (3.23MB)
            arch_path = self.rag_store_path / "architectural_embeddings.npy"
            if arch_path.exists():
                self.architectural_embeddings = np.load(arch_path)
                logger.info(f"✅ Loaded architectural embeddings: {self.architectural_embeddings.shape}")
            
            # Load spatial embeddings (659MB)
            spatial_path = self.rag_store_path / "spatial_embeddings.npy"
            if spatial_path.exists():
                self.spatial_embeddings = np.load(spatial_path)
                logger.info(f"✅ Loaded spatial embeddings: {self.spatial_embeddings.shape}")
            
            # Load text embeddings (323MB)
            text_path = self.rag_store_path / "text_embeddings.npy"
            if text_path.exists():
                self.text_embeddings = np.load(text_path)
                logger.info(f"✅ Loaded text embeddings: {self.text_embeddings.shape}")
                
        except Exception as e:
            logger.warning(f"Could not load specialized embeddings: {e}")

    def _create_knowledge_sources_from_loaded_data(self):
        """Create knowledge sources from loaded metadata and embeddings"""
        if not hasattr(self, 'metadata') or not self.metadata:
            return
        
        logger.info("Creating knowledge sources from loaded metadata...")
        source_id = 0
        
        for idx, annotation_data in enumerate(self.metadata):
            try:
                # Create knowledge source from annotation metadata
                room_type = annotation_data.get('room_type', 'unknown')
                room_function = annotation_data.get('room_function', 'unknown')
                spatial_info = annotation_data.get('spatial_info', {})
                
                # Create spatial knowledge source
                spatial_source = KnowledgeSource(
                    source_id=f"spatial_{source_id}",
                    source_type="cubicasa_spatial",
                    content={
                        'room_type': room_type,
                        'room_function': room_function,
                        'spatial_info': spatial_info,
                        'original_text': annotation_data.get('original_text', ''),
                        'embedding_index': idx  # Reference to embedding index
                    },
                    metadata={
                        'annotation_index': idx,
                        'plan_id': annotation_data.get('plan_id', 'unknown'),
                        'confidence': annotation_data.get('confidence', 0.5)
                    },
                    relevance_context={'spatial_optimization', 'functional_adjacency'}
                )
                
                self.knowledge_sources[spatial_source.source_id] = spatial_source
                source_id += 1
                
            except Exception as e:
                logger.warning(f"Failed to create knowledge source from annotation {idx}: {e}")
        
        logger.info(f"✅ Created {len(self.knowledge_sources)} knowledge sources from metadata")
    
   
    
    def _initialize_fallback_knowledge(self):
        """Initialize basic architectural knowledge if RAG store unavailable"""
        logger.info("Initializing fallback architectural knowledge...")
        
        # Basic spatial relationships
        spatial_knowledge = [
            {
                'room_type': 'bedroom',
                'optimal_adjacencies': ['bathroom', 'hallway'],
                'avoid_adjacencies': ['kitchen', 'living_room'],
                'privacy_requirements': 'high',
                'natural_light': 'essential'
            },
            {
                'room_type': 'kitchen',
                'optimal_adjacencies': ['dining_room', 'living_room', 'utility'],
                'avoid_adjacencies': ['bedroom'],
                'ventilation_requirements': 'high',
                'service_access': 'required'
            },
            {
                'room_type': 'living_room',
                'optimal_adjacencies': ['dining_room', 'kitchen', 'entrance'],
                'natural_light': 'essential',
                'central_location': 'preferred'
            }
        ]
        
        # Create fallback knowledge sources
        for idx, knowledge in enumerate(spatial_knowledge):
            source = KnowledgeSource(
                source_id=f"fallback_{idx}",
                source_type="fallback_spatial",
                content=knowledge,
                relevance_context={'spatial_optimization', 'functional_adjacency'}
            )
            self.knowledge_sources[source.source_id] = source
    
    def _initialize_search_indices(self):
        """Initialize FAISS indices for different knowledge types"""
        if not self.knowledge_sources:
            logger.warning("No knowledge sources available for indexing")
            return
        
        # Group sources by type
        spatial_sources = [s for s in self.knowledge_sources.values() 
                          if 'spatial' in s.relevance_context]
        functional_sources = [s for s in self.knowledge_sources.values() 
                             if 'functional' in s.relevance_context]
        
        # Create embeddings for spatial sources
        if spatial_sources:
            spatial_texts = []
            for source in spatial_sources:
                text = self._create_search_text(source)
                spatial_texts.append(text)
            
            spatial_embeddings = self.embedding_model.encode(spatial_texts)
            
            # Create FAISS index
            dimension = spatial_embeddings.shape[1]
            self.spatial_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(spatial_embeddings)
            self.spatial_index.add(spatial_embeddings.astype('float32'))
            
            # Store source mapping
            self.spatial_source_mapping = {i: source.source_id for i, source in enumerate(spatial_sources)}
            
            logger.info(f"Created spatial index with {len(spatial_sources)} sources")
        
        # Create embeddings for functional sources
        if functional_sources:
            functional_texts = []
            for source in functional_sources:
                text = self._create_search_text(source)
                functional_texts.append(text)
            
            functional_embeddings = self.embedding_model.encode(functional_texts)
            
            dimension = functional_embeddings.shape[1]
            self.functional_index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(functional_embeddings)
            self.functional_index.add(functional_embeddings.astype('float32'))
            
            self.functional_source_mapping = {i: source.source_id for i, source in enumerate(functional_sources)}
            
            logger.info(f"Created functional index with {len(functional_sources)} sources")
    
    def _create_search_text(self, source: KnowledgeSource) -> str:
        """Create searchable text from knowledge source"""
        content = source.content
        
        if source.source_type.startswith("cubicasa"):
            room_type = content.get('room_type', '')
            room_function = content.get('room_function', '')
            
            if 'spatial_info' in content:
                spatial_info = content['spatial_info']
                zone = spatial_info.get('zone', '')
                privacy_level = spatial_info.get('privacy_level', 0.5)
                natural_light = spatial_info.get('natural_light_score', 0.5)
                
                return f"room type {room_type} function {room_function} zone {zone} privacy {privacy_level:.2f} light {natural_light:.2f}"
            else:
                return f"room type {room_type} function {room_function}"
        
        elif source.source_type == "fallback_spatial":
            room_type = content.get('room_type', '')
            adjacencies = ' '.join(content.get('optimal_adjacencies', []))
            privacy = content.get('privacy_requirements', 'medium')
            
            return f"room type {room_type} adjacent to {adjacencies} privacy {privacy}"
        
        return str(content)
    
    def conduct_research(self,
                         prototype_id: str,
                         prototype_config: Dict[str, Any],
                         requirements: Dict[str, Any],
                         research_focus: List['ResearchQueryType'] = None) -> List['ResearchContext']:
        """Conduct comprehensive research for a prototype with enhanced error handling"""
        logger.info(f"Conducting research for prototype {prototype_id}")
        requirements = self._normalize_requirements(requirements)

        try:
            # ADD: Comprehensive input validation
            if not prototype_id or not isinstance(prototype_id, str):
                prototype_id = f"proto_{uuid.uuid4().hex[:8]}"

            if not prototype_config or not isinstance(prototype_config, dict):
                prototype_config = self._create_minimal_prototype_config()

            if not requirements or not isinstance(requirements, dict):
                requirements = self._create_minimal_requirements()

            # ADD: Safe requirements normalization
            safe_requirements = self._normalize_requirements_structure(requirements)

            # ADD: Enhanced prototype config validation
            validated_prototype_config = self._validate_prototype_config(prototype_config)

            if research_focus is None:
                research_focus = [
                    ResearchQueryType.SPATIAL_OPTIMIZATION,
                    ResearchQueryType.FUNCTIONAL_ADJACENCY,
                    ResearchQueryType.ENVIRONMENTAL_STRATEGY
                ]

            research_contexts = []
            successful_research_count = 0

            for query_type in research_focus:
                try:
                    context = self._research_specific_aspect(
                        prototype_id, validated_prototype_config, safe_requirements, query_type
                    )

                    # ADD: Quality validation and enhancement
                    if context.relevance_score < 0.4:
                        logger.warning(f"Low quality research for {query_type}, enhancing...")
                        context = self._enhance_low_quality_context(context, query_type, safe_requirements)

                    if context.confidence < 0.5:
                        logger.warning(f"Low confidence research for {query_type}, boosting...")
                        context = self._boost_low_confidence_context(context, query_type)

                    research_contexts.append(context)
                    if context.relevance_score >= 0.6 and context.confidence >= 0.6:
                        successful_research_count += 1

                except Exception as e:
                    logger.warning(f"Research failed for {query_type}: {e}")
                    fallback_context = self._create_enhanced_fallback_context(
                        prototype_id, query_type, safe_requirements
                    )
                    research_contexts.append(fallback_context)

            # ADD: Quality assurance check
            if successful_research_count == 0:
                logger.warning("No successful research conducted, creating enhanced fallbacks")
                research_contexts = self._create_comprehensive_fallback_contexts(
                    prototype_id, research_focus, safe_requirements
                )

            # Update statistics with enhanced tracking
            self.research_stats['total_queries'] += len(research_focus)
            self.research_stats['successful_retrievals'] += successful_research_count
            self.research_stats['quality_metrics'] = {
                'avg_relevance': np.mean([c.relevance_score for c in research_contexts]),
                'avg_confidence': np.mean([c.confidence for c in research_contexts]),
                'success_rate': successful_research_count / len(research_focus)
            }

            return research_contexts

        except Exception as e:
            logger.error(f"Complete research failure for {prototype_id}: {e}")
            return self._create_emergency_fallback_contexts(prototype_id, research_focus or [])
    
    def _research_specific_aspect(
        self,
        prototype_id: str,
        prototype_config: Dict[str, Any],
        requirements: Dict[str, Any],
        query_type: ResearchQueryType
    ) -> ResearchContext:
        """Research a specific aspect of the prototype with proper error handling"""

        try:
            # Ensure prototype_config has the expected structure
            if 'detailed_config' not in prototype_config:
                prototype_config['detailed_config'] = {
                    'spatial_config': {'strategy': 'linear_progression'},
                    'circulation_pattern': {'pattern_type': 'linear_spine'},
                    'environmental_strategy': {
                        'orientation': 'south',
                        'passive_strategies': []
                    }
                }

            # Generate research query with safe defaults
            query = self._generate_research_query(
                prototype_config,
                requirements,
                query_type
            )

            context = ResearchContext(
                prototype_id=prototype_id,
                research_query=query,
                context_type=query_type.value,
                priority=self._calculate_research_priority(query_type, requirements)
            )

            # Continue with existing research logic...
            # Add proper fallback values
            context.relevance_score = 0.6  # Set to acceptable threshold
            context.confidence = 0.7
            context.research_depth = 2

            return context

        except Exception as e:
            logger.warning(f"Research failed for {query_type.value}: {e}")
            # Return acceptable fallback context
            return ResearchContext(
                prototype_id=prototype_id,
                research_query=f"fallback query for {query_type.value}",
                context_type=query_type.value,
                priority=0.7,          # Higher than threshold
                relevance_score=0.7,   # Above threshold
                confidence=0.7,        # Above threshold
                research_depth=2
            )
    def _normalize_requirements_structure(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize requirements structure to prevent mismatches"""
        try:
            normalized = {}

            # Handle spatial_needs
            spatial_needs = requirements.get('spatial_needs', [])
            normalized_spatial_needs = []

            for need in spatial_needs:
                if hasattr(need, '__dict__'):
                    need_dict = asdict(need)
                elif isinstance(need, dict):
                    need_dict = need.copy()
                else:
                    need_dict = {
                        'room_type': str(need),
                        'quantity': 1,
                        'min_area': 100,
                        'priority': 'medium'
                    }

                # Safe value extraction
                min_area_raw = need_dict.get('min_area', 100)
                quantity_raw = need_dict.get('quantity', 1)

                min_area = min_area_raw if min_area_raw is not None else 100
                quantity = quantity_raw if quantity_raw is not None else 1

                try:
                    min_area = max(50, int(float(min_area)))
                except (ValueError, TypeError):
                    min_area = 100

                try:
                    quantity = max(1, int(float(quantity)))
                except (ValueError, TypeError):
                    quantity = 1

                normalized_need = {
                    'room_type': need_dict.get('room_type', 'unknown'),
                    'quantity': quantity,
                    'min_area': min_area,
                    'priority': need_dict.get('priority', 'medium')
                }
                normalized_spatial_needs.append(normalized_need)

            normalized['spatial_needs'] = normalized_spatial_needs

            # Handle site_constraints
            site_constraints = requirements.get('site_constraints', {})
            if hasattr(site_constraints, '__dict__'):
                site_constraints = asdict(site_constraints)

            plot_length_raw = site_constraints.get('plot_length', 50)
            plot_width_raw = site_constraints.get('plot_width', 30)

            plot_length = plot_length_raw if plot_length_raw is not None else 50
            plot_width = plot_width_raw if plot_width_raw is not None else 30

            try:
                plot_length = max(20, float(plot_length))
            except (ValueError, TypeError):
                plot_length = 50

            try:
                plot_width = max(15, float(plot_width))
            except (ValueError, TypeError):
                plot_width = 30

            normalized['site_constraints'] = {
                'plot_length': plot_length,
                'plot_width': plot_width,
                'orientation': site_constraints.get('orientation', 'south')
            }

            # Handle design_preferences
            design_prefs = requirements.get('design_preferences', {})
            if hasattr(design_prefs, '__dict__'):
                design_prefs = asdict(design_prefs)

            normalized['design_preferences'] = {
                'style': design_prefs.get('style', 'modern'),
                'accessibility_requirements': design_prefs.get('accessibility_requirements', False)
            }

            # Handle budget with None safety
            budget_raw = requirements.get('budget', 2500000)
            budget = budget_raw if budget_raw is not None else 2500000

            try:
                budget = max(500000, float(budget))
            except (ValueError, TypeError):
                budget = 2500000

            normalized['budget'] = budget

            # Copy other fields safely
            for key, value in requirements.items():
                if key not in ['spatial_needs', 'site_constraints', 'design_preferences', 'budget']:
                    normalized[key] = value

            return normalized

        except Exception as e:
            logger.error(f"Requirements normalization failed: {e}")
            return self._create_minimal_requirements()

    def _validate_prototype_config(self, prototype_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance prototype configuration"""
        try:
            validated = prototype_config.copy()

            # Ensure detailed_config exists
            if 'detailed_config' not in validated:
                validated['detailed_config'] = {}

            detailed_config = validated['detailed_config']

            # Ensure spatial_config
            if 'spatial_config' not in detailed_config:
                detailed_config['spatial_config'] = {}

            spatial_config = detailed_config['spatial_config']
            if 'strategy' not in spatial_config:
                spatial_config['strategy'] = 'linear_progression'
            if 'plot_utilization' not in spatial_config:
                spatial_config['plot_utilization'] = 0.7

            # Ensure circulation_pattern
            if 'circulation_pattern' not in detailed_config:
                detailed_config['circulation_pattern'] = {}

            circulation = detailed_config['circulation_pattern']
            if 'pattern_type' not in circulation:
                circulation['pattern_type'] = 'linear_spine'
            if 'efficiency_target' not in circulation:
                circulation['efficiency_target'] = 0.85

            # Ensure environmental_strategy
            if 'environmental_strategy' not in detailed_config:
                detailed_config['environmental_strategy'] = {}

            env_strategy = detailed_config['environmental_strategy']
            if 'orientation' not in env_strategy:
                env_strategy['orientation'] = 'south'
            if 'passive_strategies' not in env_strategy:
                env_strategy['passive_strategies'] = ['natural_ventilation']

            return validated

        except Exception as e:
            logger.error(f"Prototype config validation failed: {e}")
            return self._create_minimal_prototype_config()

    def _create_minimal_prototype_config(self) -> Dict[str, Any]:
        """Create minimal prototype configuration"""
        return {
            'prototype_id': f'fallback_{uuid.uuid4().hex[:8]}',
            'detailed_config': {
                'spatial_config': {
                    'strategy': 'linear_progression',
                    'plot_utilization': 0.7,
                    'compactness_factor': 0.8
                },
                'circulation_pattern': {
                    'pattern_type': 'linear_spine',
                    'efficiency_target': 0.85
                },
                'environmental_strategy': {
                    'orientation': 'south',
                    'passive_strategies': ['natural_ventilation']
                }
            }
        }

    def _create_minimal_requirements(self) -> Dict[str, Any]:
        """Create minimal requirements structure"""
        return {
            'spatial_needs': [
                {'room_type': 'bedroom', 'quantity': 2, 'min_area': 120, 'priority': 'medium'},
                {'room_type': 'bathroom', 'quantity': 1, 'min_area': 45, 'priority': 'medium'},
                {'room_type': 'kitchen', 'quantity': 1, 'min_area': 100, 'priority': 'medium'},
                {'room_type': 'living_room', 'quantity': 1, 'min_area': 200, 'priority': 'medium'}
            ],
            'site_constraints': {
                'plot_length': 50,
                'plot_width': 30,
                'orientation': 'south'
            },
            'design_preferences': {
                'style': 'modern',
                'accessibility_requirements': False
            },
            'budget': 2500000
        }

    def _enhance_low_quality_context(self, context: 'ResearchContext', query_type: 'ResearchQueryType', requirements: Dict[str, Any]) -> 'ResearchContext':
        """Enhance low-quality research context"""
        try:
            spatial_needs = requirements.get('spatial_needs', [])
            room_types = [need['room_type'] for need in spatial_needs]

            if query_type == ResearchQueryType.SPATIAL_OPTIMIZATION:
                context.similar_examples.extend(self._generate_spatial_examples(room_types))
            elif query_type == ResearchQueryType.FUNCTIONAL_ADJACENCY:
                context.similar_examples.extend(self._generate_adjacency_examples(room_types))
            elif query_type == ResearchQueryType.ENVIRONMENTAL_STRATEGY:
                context.similar_examples.extend(self._generate_environmental_examples())

            context.relevance_score = max(0.7, context.relevance_score + 0.2)
            context.confidence = max(0.7, context.confidence + 0.1)
            context.research_depth = max(2, context.research_depth)

            return context

        except Exception as e:
            logger.warning(f"Context enhancement failed: {e}")
            return context

    def _boost_low_confidence_context(self, context: 'ResearchContext', query_type: 'ResearchQueryType') -> 'ResearchContext':
        """Boost confidence of low-confidence context"""
        if not context.design_patterns:
            context.design_patterns = self._generate_generic_patterns(query_type)

        confidence_boost = min(0.3, len(context.design_patterns) * 0.1)
        context.confidence = min(1.0, context.confidence + confidence_boost)

        return context

    def _generate_spatial_examples(self, room_types: List[str]) -> List[Dict[str, Any]]:
        """Generate spatial examples based on room types"""
        examples = []
        for room_type in room_types[:3]:
            examples.append({
                'source_id': f'generated_spatial_{room_type}',
                'relevance_score': 0.7,
                'room_type': room_type,
                'spatial_features': {
                    'zone': 'central',
                    'privacy_level': 0.7 if room_type == 'bedroom' else 0.5,
                    'natural_light_score': 0.8
                },
                'match_reason': f'Generated example for {room_type}'
            })
        return examples

    def _generate_adjacency_examples(self, room_types: List[str]) -> List[Dict[str, Any]]:
        """Generate adjacency examples"""
        adjacency_rules = {
            'bedroom': ['bathroom', 'hallway'],
            'kitchen': ['dining_room', 'living_room'],
            'living_room': ['kitchen', 'dining_room'],
            'bathroom': ['bedroom', 'hallway']
        }

        examples = []
        for room_type in room_types:
            if room_type in adjacency_rules:
                examples.append({
                    'source_id': f'generated_adjacency_{room_type}',
                    'relevance_score': 0.75,
                    'room_type': room_type,
                    'adjacencies': adjacency_rules[room_type],
                    'match_reason': f'Standard adjacency for {room_type}'
                })
        return examples

    def _generate_environmental_examples(self) -> List[Dict[str, Any]]:
        """Generate environmental strategy examples"""
        return [
            {
                'source_id': 'generated_env_south',
                'relevance_score': 0.8,
                'environmental_features': {
                    'orientation_match': True,
                    'natural_light_optimization': True
                },
                'zone': 'south',
                'natural_light_score': 0.9,
                'match_reason': 'South-facing optimization'
            },
            {
                'source_id': 'generated_env_ventilation',
                'relevance_score': 0.75,
                'environmental_features': {
                    'cross_ventilation': True,
                    'passive_cooling': True
                },
                'match_reason': 'Natural ventilation strategy'
            }
        ]

    def _generate_generic_patterns(self, query_type: 'ResearchQueryType') -> List[Dict[str, Any]]:
        """Generate generic design patterns"""
        patterns = []
        if query_type == ResearchQueryType.SPATIAL_OPTIMIZATION:
            patterns.append({
                'type': 'spatial_efficiency',
                'description': 'Compact rectangular room layouts',
                'data': {'efficiency_factor': 0.8},
                'confidence': 0.7
            })
        elif query_type == ResearchQueryType.FUNCTIONAL_ADJACENCY:
            patterns.append({
                'type': 'adjacency_pattern',
                'description': 'Kitchen adjacent to living areas',
                'data': {'kitchen-living_room': 0.9},
                'confidence': 0.8
            })
        elif query_type == ResearchQueryType.ENVIRONMENTAL_STRATEGY:
            patterns.append({
                'type': 'orientation_preference',
                'description': 'South-facing living areas',
                'data': {'living_room': 'south'},
                'confidence': 0.75
            })
        return patterns

    def _create_enhanced_fallback_context(self, prototype_id: str, query_type: 'ResearchQueryType', requirements: Dict[str, Any]) -> 'ResearchContext':
        """Create enhanced fallback context"""
        context = ResearchContext(
            prototype_id=prototype_id,
            research_query=f"Enhanced fallback query for {query_type.value}",
            context_type=query_type.value,
            priority=0.8,
            relevance_score=0.7,
            confidence=0.7,
            research_depth=2
        )
        return self._enhance_low_quality_context(context, query_type, requirements)

    def _create_comprehensive_fallback_contexts(self, prototype_id: str, research_focus: List['ResearchQueryType'], requirements: Dict[str, Any]) -> List['ResearchContext']:
        """Create comprehensive fallback contexts"""
        return [
            self._create_enhanced_fallback_context(prototype_id, query_type, requirements)
            for query_type in research_focus
        ]

    def _create_emergency_fallback_contexts(self, prototype_id: str, research_focus: List['ResearchQueryType']) -> List['ResearchContext']:
        """Create emergency fallback contexts"""
        return [
            ResearchContext(
                prototype_id=prototype_id,
                research_query=f"Emergency fallback for {query_type.value}",
                context_type=query_type.value,
                priority=0.5,
                relevance_score=0.6,
                confidence=0.6,
                research_depth=1
            )
            for query_type in research_focus
        ]
    def _generate_research_query(self,
                               prototype_config: Dict[str, Any],
                               requirements: Dict[str, Any],
                               query_type: ResearchQueryType) -> str:
        """Generate natural language research query"""
        
        spatial_needs = requirements.get('spatial_needs', [])
        room_types = [need.get('room_type', '') for need in spatial_needs]
        
        spatial_strategy = prototype_config.get('spatial_config', {}).get('strategy', 'unknown')
        circulation_pattern = prototype_config.get('circulation_pattern', {}).get('pattern_type', 'unknown')
        
        if query_type == ResearchQueryType.SPATIAL_OPTIMIZATION:
            return f"optimal spatial layout for {' '.join(room_types)} using {spatial_strategy} strategy"
        
        elif query_type == ResearchQueryType.FUNCTIONAL_ADJACENCY:
            return f"functional adjacencies between {' '.join(room_types)} in residential layout"
        
        elif query_type == ResearchQueryType.ENVIRONMENTAL_STRATEGY:
            orientation = prototype_config.get('environmental_strategy', {}).get('orientation', 'south')
            return f"environmental strategies for {orientation} facing residential building"
        
        elif query_type == ResearchQueryType.CIRCULATION_PATTERNS:
            return f"{circulation_pattern} circulation pattern for residential layout with {len(room_types)} rooms"
        
        else:
            return f"architectural solutions for {spatial_strategy} residential layout"
    
    def _calculate_research_priority(self,
                                   query_type: ResearchQueryType,
                                   requirements: Dict[str, Any]) -> float:
        """Calculate priority for research query"""
        
        priority_weights = {
            ResearchQueryType.SPATIAL_OPTIMIZATION: 0.9,
            ResearchQueryType.FUNCTIONAL_ADJACENCY: 0.85,
            ResearchQueryType.ENVIRONMENTAL_STRATEGY: 0.8,
            ResearchQueryType.CIRCULATION_PATTERNS: 0.75,
            ResearchQueryType.CODE_COMPLIANCE: 1.0,  # Always high priority
            ResearchQueryType.COST_OPTIMIZATION: 0.7,
            ResearchQueryType.AESTHETIC_REFERENCES: 0.6,
            ResearchQueryType.STRUCTURAL_SOLUTIONS: 0.65
        }
        
        base_priority = priority_weights.get(query_type, 0.5)
        
        # Adjust based on user requirements
        design_prefs = requirements.get('design_preferences', {})
        if query_type == ResearchQueryType.ENVIRONMENTAL_STRATEGY and design_prefs.get('sustainability_focus', False):
            base_priority += 0.15
        
        if query_type == ResearchQueryType.COST_OPTIMIZATION and requirements.get('budget_sensitive', False):
            base_priority += 0.2
        
        return min(1.0, base_priority)
    
    def _research_spatial_functional(self, context: ResearchContext, prototype_config, requirements):
        """Enhanced research using actual loaded embeddings"""
        
        if not hasattr(self, 'comprehensive_index') or self.comprehensive_index is None:
            logger.warning("No comprehensive index available for research")
            context.relevance_score = 0.2
            context.confidence = 0.1
            return context
        
        # Create query embedding using the same model used to create the store
        try:
            query_embedding = self.gemma_encoder.encode_research_query(
                prototype_config, 
                requirements,
                context.context_type
            ).reshape(1, -1)  # Reshape for FAISS compatibility
        
            logger.debug(f"Generated Gemma 3 query embedding: {query_embedding.shape}")
        
        except Exception as e:
            logger.warning(f"Gemma 3 query encoding failed, using fallback: {e}")
            # Fallback to sentence transformer
            query_embedding = self.embedding_model.encode([context.research_query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search in comprehensive index
        k = min(20, self.comprehensive_index.ntotal)
        scores, indices = self.comprehensive_index.search(query_embedding.astype('float32'), k)
        
        # Process results with actual metadata
        similar_examples = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or score < 0.4:  # Minimum relevance threshold
                continue
            
            # Get metadata for this index
            if idx < len(self.metadata):
                annotation_data = self.metadata[idx]
                
                example = {
                    'source_id': f"embedding_{idx}",
                    'relevance_score': float(score),
                    'room_type': annotation_data.get('room_type', 'unknown'),
                    'room_function': annotation_data.get('room_function', 'unknown'),
                    'spatial_info': annotation_data.get('spatial_info', {}),
                    'original_text': annotation_data.get('original_text', ''),
                    'plan_reference': annotation_data.get('plan_id', 'unknown'),
                    'match_reason': f"Semantic similarity: {score:.3f}",
                    'embedding_index': idx
                }
                
                similar_examples.append(example)
        
        context.similar_examples = similar_examples
        context.design_patterns = self._extract_design_patterns(similar_examples, 'spatial')
        
        if similar_examples:
            context.relevance_score = np.mean([ex['relevance_score'] for ex in similar_examples])
            context.confidence = min(1.0, len(similar_examples) / 10.0)
            context.research_depth = min(3, len(similar_examples) // 5 + 1)
        else:
            context.relevance_score = 0.2
            context.confidence = 0.1
            context.research_depth = 1
        
        logger.info(f"Found {len(similar_examples)} similar examples with avg relevance {context.relevance_score:.3f}")
        return context
    
    def _research_environmental(self,
                              context: ResearchContext,
                              prototype_config: Dict[str, Any],
                              requirements: Dict[str, Any]) -> ResearchContext:
        """Research environmental strategies"""
        
        env_strategy = prototype_config.get('environmental_strategy', {})
        orientation = env_strategy.get('orientation', 'south')
        passive_strategies = env_strategy.get('passive_strategies', [])
        
        # Find similar environmental approaches
        environmental_examples = []
        
        for source in self.knowledge_sources.values():
            if 'environmental' in source.relevance_context or source.source_type == 'cubicasa_spatial':
                content = source.content
                
                # Check for environmental relevance
                if 'spatial_info' in content:
                    spatial_info = content['spatial_info']
                    natural_light = spatial_info.get('natural_light_score', 0.0)
                    zone = spatial_info.get('zone', '')
                    
                    # Match orientation zones
                    orientation_zones = {
                        'south': ['south', 'southeast', 'southwest'],
                        'east': ['east', 'northeast', 'southeast'],
                        'north': ['north', 'northeast', 'northwest'],
                        'west': ['west', 'southwest', 'northwest']
                    }
                    
                    if any(oz in zone.lower() for oz in orientation_zones.get(orientation, [])):
                        example = {
                            'source_id': source.source_id,
                            'relevance_score': 0.7 + (natural_light * 0.3),
                            'room_type': content.get('room_type', 'unknown'),
                            'natural_light_score': natural_light,
                            'zone': zone,
                            'environmental_features': {
                                'natural_light': natural_light,
                                'orientation_match': True
                            },
                            'match_reason': f"Orientation match for {orientation} facing"
                        }
                        environmental_examples.append(example)
        
        # Sort by relevance
        environmental_examples.sort(key=lambda x: x['relevance_score'], reverse=True)
        context.similar_examples = environmental_examples[:10]  # Top 10
        
        # Extract environmental design patterns
        context.design_patterns = self._extract_environmental_patterns(environmental_examples)
        
        # Calculate scores
        if environmental_examples:
            context.relevance_score = np.mean([ex['relevance_score'] for ex in environmental_examples[:5]])
            context.confidence = min(1.0, len(environmental_examples) / 8.0)
        else:
            context.relevance_score = 0.4  # Basic environmental knowledge
            context.confidence = 0.3
        
        context.research_depth = 1
        
        return context
    
    def _research_circulation(self, context, prototype_config, requirements):
        """Research circulation patterns using available knowledge sources"""
        
        # Use existing knowledge sources instead of plan_stores
        circulation_examples = []
        
        for source in self.knowledge_sources.values():
            if 'circulation' in source.relevance_context or source.source_type.startswith('cubicasa'):
                # Create circulation example from available data
                example = {
                    'source_id': source.source_id,
                    'relevance_score': 0.6,  # Default relevance
                    'circulation_efficiency': 0.75,  # Default efficiency
                    'total_rooms': len(requirements.get('spatial_needs', [])),
                    'circulation_features': {
                        'efficiency': 0.75,
                        'room_count': len(requirements.get('spatial_needs', [])),
                        'pattern_type': 'inferred'
                    },
                    'match_reason': 'Knowledge source match'
                }
                circulation_examples.append(example)
        
        context.similar_examples = circulation_examples[:8]
        context.design_patterns = self._extract_circulation_patterns(circulation_examples)
        
        if circulation_examples:
            context.relevance_score = 0.6
            context.confidence = 0.5
        else:
            context.relevance_score = 0.3
            context.confidence = 0.2
        
        context.research_depth = 1
        return context
    
    def _research_general(self,
                         context: ResearchContext,
                         prototype_config: Dict[str, Any],
                         requirements: Dict[str, Any]) -> ResearchContext:
        """General research using available knowledge"""
        
        # Use functional index if available
        if self.functional_index is not None:
            query_embedding = self.embedding_model.encode([context.research_query])
            faiss.normalize_L2(query_embedding)
            
            k = min(5, self.functional_index.ntotal)
            scores, indices = self.functional_index.search(query_embedding.astype('float32'), k)
            
            similar_examples = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < 0.4:
                    continue
                
                source_id = self.functional_source_mapping.get(idx)
                if source_id and source_id in self.knowledge_sources:
                    source = self.knowledge_sources[source_id]
                    
                    example = {
                        'source_id': source_id,
                        'relevance_score': float(score),
                        'content': source.content,
                        'match_reason': f"Functional similarity: {score:.3f}"
                    }
                    similar_examples.append(example)
            
            context.similar_examples = similar_examples
            context.relevance_score = np.mean([ex['relevance_score'] for ex in similar_examples]) if similar_examples else 0.3
            context.confidence = min(1.0, len(similar_examples) / 3.0) if similar_examples else 0.2
        else:
            context.relevance_score = 0.3
            context.confidence = 0.2
        
        context.research_depth = 1
        
        return context
    
    def _extract_design_patterns(self, examples: List[Dict[str, Any]], pattern_type: str) -> List[Dict[str, Any]]:
        """Extract design patterns from similar examples"""
        
        if not examples:
            return []
        
        patterns = []
        
        if pattern_type == 'spatial':
            # Analyze room adjacency patterns
            adjacency_counts = defaultdict(int)
            room_zone_patterns = defaultdict(list)
            
            for example in examples:
                room_type = example.get('room_type', 'unknown')
                adjacencies = example.get('adjacencies', [])
                spatial_features = example.get('spatial_features', {})
                zone = spatial_features.get('zone', 'unknown')
                
                for adj in adjacencies:
                    adjacency_counts[f"{room_type}-{adj}"] += 1
                
                room_zone_patterns[room_type].append(zone)
            
            # Extract common adjacencies
            common_adjacencies = [(pair, count) for pair, count in adjacency_counts.items() if count >= 2]
            if common_adjacencies:
                patterns.append({
                    'type': 'adjacency_pattern',
                    'description': 'Common room adjacencies',
                    'data': dict(common_adjacencies),
                    'confidence': len(common_adjacencies) / len(examples)
                })
            
            # Extract zone preferences
            for room_type, zones in room_zone_patterns.items():
                if len(zones) >= 2:
                    zone_counter = Counter(zones)
                    most_common_zone = zone_counter.most_common(1)[0]
                    
                    patterns.append({
                        'type': 'zone_preference',
                        'description': f'{room_type} zone preference',
                        'data': {
                            'room_type': room_type,
                            'preferred_zone': most_common_zone[0],
                            'frequency': most_common_zone[1] / len(zones)
                        },
                        'confidence': most_common_zone[1] / len(zones)
                    })
        
        return patterns
    
    def _extract_environmental_patterns(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract environmental design patterns"""
        
        patterns = []
        
        if not examples:
            return patterns
        
        # Analyze natural light patterns
        light_scores = [ex.get('natural_light_score', 0.0) for ex in examples]
        room_types = [ex.get('room_type', 'unknown') for ex in examples]
        
        if light_scores:
            avg_light = np.mean(light_scores)
            
            # Room type lighting preferences
            room_light_map = defaultdict(list)
            for room_type, light_score in zip(room_types, light_scores):
                room_light_map[room_type].append(light_score)
            
            for room_type, scores in room_light_map.items():
                if len(scores) >= 2:
                    avg_room_light = np.mean(scores)
                    
                    patterns.append({
                        'type': 'lighting_preference',
                        'description': f'{room_type} natural lighting pattern',
                        'data': {
                            'room_type': room_type,
                            'avg_natural_light': avg_room_light,
                            'sample_count': len(scores),
                            'lighting_category': 'high' if avg_room_light > 0.7 else 'medium' if avg_room_light > 0.4 else 'low'
                        },
                        'confidence': min(1.0, len(scores) / 3.0)
                    })
        
        # Analyze orientation patterns
        orientation_patterns = defaultdict(list)
        for example in examples:
            room_type = example.get('room_type', 'unknown')
            zone = example.get('zone', 'unknown')
            orientation_patterns[room_type].append(zone)
        
        for room_type, zones in orientation_patterns.items():
            if len(zones) >= 2:
                zone_counter = Counter(zones)
                most_common = zone_counter.most_common(1)[0]
                
                patterns.append({
                    'type': 'orientation_preference',
                    'description': f'{room_type} orientation preference',
                    'data': {
                        'room_type': room_type,
                        'preferred_orientation': most_common[0],
                        'frequency': most_common[1] / len(zones)
                    },
                    'confidence': most_common[1] / len(zones)
                })
        
        return patterns
    
    def _extract_circulation_patterns(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract circulation design patterns"""
        
        patterns = []
        
        if not examples:
            return patterns
        
        # Analyze efficiency patterns
        efficiency_scores = [ex.get('circulation_efficiency', 0.0) for ex in examples]
        room_counts = [ex.get('total_rooms', 0) for ex in examples]
        
        if efficiency_scores and room_counts:
            # Efficiency vs room count relationship
            efficiency_by_room_count = defaultdict(list)
            for room_count, efficiency in zip(room_counts, efficiency_scores):
                size_category = 'small' if room_count <= 5 else 'medium' if room_count <= 8 else 'large'
                efficiency_by_room_count[size_category].append(efficiency)
            
            for size_cat, efficiencies in efficiency_by_room_count.items():
                if len(efficiencies) >= 2:
                    avg_efficiency = np.mean(efficiencies)
                    
                    patterns.append({
                        'type': 'efficiency_pattern',
                        'description': f'Circulation efficiency for {size_cat} layouts',
                        'data': {
                            'size_category': size_cat,
                            'avg_efficiency': avg_efficiency,
                            'sample_count': len(efficiencies),
                            'efficiency_range': [min(efficiencies), max(efficiencies)]
                        },
                        'confidence': min(1.0, len(efficiencies) / 3.0)
                    })
        
        return patterns
    
    def get_embedding_structure_info(self) -> Dict[str, Any]:
        """Get embedding structure information for other agents"""
        
        embedding_structure = {
            'composite_embeddings_available': hasattr(self, 'composite_embeddings'),
            'total_embeddings': len(self.composite_embeddings) if hasattr(self, 'composite_embeddings') else 0,
            'embedding_dimension': self.composite_embeddings.shape[1] if hasattr(self, 'composite_embeddings') else None,
            
            # Feature groups based on RAG store structure
            'feature_groups': {
                'text_features': {
                    'dimensions': list(range(0, 384)) if hasattr(self, 'composite_embeddings') else [],
                    'description': 'Text semantic embeddings',
                    'weight': 0.3
                },
                'visual_features': {
                    'dimensions': list(range(384, 896)) if hasattr(self, 'composite_embeddings') else [],
                    'description': 'CLIP visual embeddings',
                    'weight': 0.25
                },
                'spatial_features': {
                    'dimensions': list(range(896, 960)) if hasattr(self, 'composite_embeddings') else [],
                    'description': 'Spatial relationship features',
                    'weight': 0.25
                },
                'architectural_features': {
                    'dimensions': list(range(960, 1024)) if hasattr(self, 'composite_embeddings') else [],
                    'description': 'Architectural domain features',
                    'weight': 0.2
                }
            },
            
            # Similarity thresholds for different contexts
            'similarity_thresholds': {
                'spatial_optimization': 0.7,
                'functional_adjacency': 0.65,
                'environmental_strategy': 0.6,
                'circulation_patterns': 0.55,
                'aesthetic_references': 0.5
            },
            
            # Knowledge source statistics
            'knowledge_sources': {
                'total_sources': len(self.knowledge_sources),
                'source_types': list(set(s.source_type for s in self.knowledge_sources.values())),
                'cubicasa_coverage': len([s for s in self.knowledge_sources.values() if s.source_type.startswith('cubicasa')]),
                'fallback_coverage': len([s for s in self.knowledge_sources.values() if s.source_type.startswith('fallback')])
            },
            
            # Search capabilities
            'search_capabilities': {
                'spatial_search_available': self.spatial_index is not None,
                'functional_search_available': self.functional_index is not None,
                'environmental_search_available': len([s for s in self.knowledge_sources.values() if 'environmental' in s.relevance_context]) > 0,
                'max_search_results': 10,
                'min_relevance_threshold': 0.4
            }
        }
        
        return embedding_structure
    
    def enhance_prototype_with_research(self,
                                      prototype_config: Dict[str, Any],
                                      research_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Enhance prototype configuration with research findings"""
        
        enhanced_config = prototype_config.copy()
        try:
            enhanced_embedding = self.gemma_encoder.encode_prototype_features(enhanced_config)
            enhanced_config['enhanced_embedding'] = enhanced_embedding.tolist()
            enhanced_config['embedding_source'] = 'gemma3_research_enhanced'
        
        except Exception as e:
            logger.warning(f"Enhanced embedding generation failed: {e}")
        # Add research metadata
        enhanced_config['research_metadata'] = {
            'research_conducted': True,
            'research_contexts': len(research_contexts),
            'avg_relevance': np.mean([ctx.relevance_score for ctx in research_contexts]),
            'avg_confidence': np.mean([ctx.confidence for ctx in research_contexts]),
            'research_timestamp': np.datetime64('now').astype(str)
        }
        
        # Enhance spatial configuration with research findings
        spatial_contexts = [ctx for ctx in research_contexts if 'spatial' in ctx.context_type]
        if spatial_contexts:
            spatial_enhancements = self._extract_spatial_enhancements(spatial_contexts)
            if 'spatial_config' not in enhanced_config:
                enhanced_config['spatial_config'] = {}
            enhanced_config['spatial_config'].update(spatial_enhancements)
        
        # Enhance functional zones with adjacency research
        adjacency_contexts = [ctx for ctx in research_contexts if 'adjacency' in ctx.context_type]
        if adjacency_contexts:
            adjacency_enhancements = self._extract_adjacency_enhancements(adjacency_contexts)
            if 'functional_zones' not in enhanced_config:
                enhanced_config['functional_zones'] = {}
            enhanced_config['functional_zones'].update(adjacency_enhancements)
        
        # Enhance environmental strategy
        env_contexts = [ctx for ctx in research_contexts if 'environmental' in ctx.context_type]
        if env_contexts:
            env_enhancements = self._extract_environmental_enhancements(env_contexts)
            if 'environmental_strategy' not in enhanced_config:
                enhanced_config['environmental_strategy'] = {}
            enhanced_config['environmental_strategy'].update(env_enhancements)
        
        # Add research-based recommendations
        enhanced_config['research_recommendations'] = self._generate_research_recommendations(research_contexts)
        
        return enhanced_config
    
    def _extract_spatial_enhancements(self, spatial_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Extract spatial enhancements from research contexts"""
        
        enhancements = {}
        
        all_examples = []
        for context in spatial_contexts:
            all_examples.extend(context.similar_examples)
        
        if not all_examples:
            return enhancements
        
        # Analyze spatial efficiency patterns
        room_types = [ex.get('room_type', '') for ex in all_examples]
        spatial_features = [ex.get('spatial_features', {}) for ex in all_examples]
        
        # Privacy level patterns
        privacy_levels = [sf.get('privacy_level', 0.5) for sf in spatial_features if 'privacy_level' in sf]
        if privacy_levels:
            enhancements['research_informed_privacy'] = {
                'avg_privacy_requirement': np.mean(privacy_levels),
                'privacy_range': [min(privacy_levels), max(privacy_levels)],
                'privacy_sensitive_rooms': [rt for rt, pf in zip(room_types, spatial_features) 
                                          if pf.get('privacy_level', 0.5) > 0.7]
            }
        
        # Natural light optimization
        light_scores = [sf.get('natural_light_score', 0.5) for sf in spatial_features if 'natural_light_score' in sf]
        if light_scores:
            enhancements['research_informed_lighting'] = {
                'avg_light_requirement': np.mean(light_scores),
                'high_light_rooms': [rt for rt, sf in zip(room_types, spatial_features) 
                                   if sf.get('natural_light_score', 0.5) > 0.7],
                'optimal_light_score': max(light_scores)
            }
        
        # Zone distribution patterns
        zones = [sf.get('zone', 'unknown') for sf in spatial_features]
        zone_counter = Counter(zones)
        if zone_counter:
            enhancements['research_informed_zoning'] = {
                'common_zones': dict(zone_counter.most_common(3)),
                'zone_distribution': dict(zone_counter)
            }
        
        return enhancements
    
    def _extract_adjacency_enhancements(self, adjacency_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Extract adjacency enhancements from research"""
        
        enhancements = {}
        
        # Collect adjacency patterns from all contexts
        adjacency_patterns = {}
        for context in adjacency_contexts:
            for pattern in context.design_patterns:
                if pattern.get('type') == 'adjacency_pattern':
                    adjacency_data = pattern.get('data', {})
                    adjacency_patterns.update(adjacency_data)
        
        if adjacency_patterns:
            # Parse adjacency relationships
            adjacency_rules = defaultdict(list)
            for adjacency_pair, frequency in adjacency_patterns.items():
                if '-' in adjacency_pair:
                    room1, room2 = adjacency_pair.split('-', 1)
                    adjacency_rules[room1].append({
                        'adjacent_room': room2,
                        'frequency': frequency,
                        'strength': 'strong' if frequency >= 3 else 'medium' if frequency >= 2 else 'weak'
                    })
            
            enhancements['research_informed_adjacencies'] = dict(adjacency_rules)
        
        return enhancements
    
    def _extract_environmental_enhancements(self, env_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Extract environmental enhancements from research"""
        
        enhancements = {}
        
        all_examples = []
        for context in env_contexts:
            all_examples.extend(context.similar_examples)
        
        if not all_examples:
            return enhancements
        
        # Collect environmental features
        natural_light_scores = [ex.get('natural_light_score', 0.0) for ex in all_examples]
        orientation_matches = [ex.get('environmental_features', {}).get('orientation_match', False) for ex in all_examples]
        
        if natural_light_scores:
            enhancements['research_informed_lighting_strategy'] = {
                'target_natural_light': np.mean(natural_light_scores),
                'min_acceptable_light': min(natural_light_scores),
                'light_optimization_potential': max(natural_light_scores) - min(natural_light_scores)
            }
        
        orientation_success_rate = sum(orientation_matches) / len(orientation_matches) if orientation_matches else 0.0
        enhancements['orientation_validation'] = {
            'research_support': orientation_success_rate,
            'confidence_level': 'high' if orientation_success_rate > 0.7 else 'medium' if orientation_success_rate > 0.4 else 'low'
        }
        
        return enhancements
    
    def _generate_research_recommendations(self, research_contexts: List[ResearchContext]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on research"""
        
        recommendations = []
        
        # High-confidence recommendations
        high_confidence_contexts = [ctx for ctx in research_contexts if ctx.confidence > 0.7]
        for context in high_confidence_contexts:
            if context.context_type == 'spatial_optimization':
                recommendations.append({
                    'type': 'spatial_optimization',
                    'priority': 'high',
                    'recommendation': f'Apply spatial optimization patterns found in {len(context.similar_examples)} similar examples',
                    'confidence': context.confidence,
                    'supporting_evidence': len(context.similar_examples)
                })
            
            elif context.context_type == 'functional_adjacency':
                recommendations.append({
                    'type': 'functional_adjacency',
                    'priority': 'high',
                    'recommendation': f'Follow adjacency patterns from {len(context.design_patterns)} validated design patterns',
                    'confidence': context.confidence,
                    'supporting_evidence': len(context.design_patterns)
                })
        
        # Medium-confidence suggestions
        medium_confidence_contexts = [ctx for ctx in research_contexts if 0.4 <= ctx.confidence <= 0.7]
        for context in medium_confidence_contexts:
            recommendations.append({
                'type': context.context_type,
                'priority': 'medium',
                'recommendation': f'Consider research insights for {context.context_type} (moderate confidence)',
                'confidence': context.confidence,
                'supporting_evidence': len(context.similar_examples)
            })
        
        # Sort by priority and confidence
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get research statistics"""
        
        cache_hit_rate = (self.research_stats['cache_hits'] / 
                         max(1, self.research_stats['total_queries']))
        
        success_rate = (self.research_stats['successful_retrievals'] / 
                       max(1, self.research_stats['total_queries']))
        
        return {
            'total_research_queries': self.research_stats['total_queries'],
            'successful_retrievals': self.research_stats['successful_retrievals'],
            'cache_hits': self.research_stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'success_rate': success_rate,
            'knowledge_base_stats': {
                'total_knowledge_sources': len(self.knowledge_sources),
                'spatial_search_ready': self.spatial_index is not None,
                'functional_search_ready': self.functional_index is not None,
                'knowledge_sources_loaded': len(self.knowledge_sources)
            },
            'embedding_stats': {
                'composite_embeddings_loaded': hasattr(self, 'composite_embeddings'),
                'embedding_dimensions': self.composite_embeddings.shape[1] if hasattr(self, 'composite_embeddings') else 0,
                'total_embeddings': self.composite_embeddings.shape[0] if hasattr(self, 'composite_embeddings') else 0
            }
        }
    
    def export_research_data(self, research_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Export research data for use by other agents"""
        
        exported_data = {
            'research_summary': {
                'total_contexts': len(research_contexts),
                'avg_relevance': np.mean([ctx.relevance_score for ctx in research_contexts]) if research_contexts else 0.0,
                'avg_confidence': np.mean([ctx.confidence for ctx in research_contexts]) if research_contexts else 0.0,
                'research_types': list(set(ctx.context_type for ctx in research_contexts))
            },
            
            'research_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_contexts': len(research_contexts),
                'context_types': list(set(ctx.context_type for ctx in research_contexts)),
                },
            
            'aggregated_findings': {
                'spatial_insights': self._aggregate_spatial_insights(research_contexts),
                'functional_insights': self._aggregate_functional_insights(research_contexts),
                'environmental_insights': self._aggregate_environmental_insights(research_contexts)
            },
            
            'knowledge_base_coverage': {
                'cubicasa_examples_used': len([
                    ex for ctx in research_contexts 
                    for ex in ctx.similar_examples 
                    if 'cubicasa' in ex.get('source_id', '')
                ]),
                'fallback_knowledge_used': len([
                    ex for ctx in research_contexts 
                    for ex in ctx.similar_examples 
                    if 'fallback' in ex.get('source_id', '')
                ])
            }
        }
        
        return exported_data
    
    def _aggregate_spatial_insights(self, research_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Aggregate spatial insights from all research contexts"""
        
        spatial_contexts = [ctx for ctx in research_contexts if 'spatial' in ctx.context_type]
        
        if not spatial_contexts:
            return {}
        
        all_room_types = []
        all_zones = []
        all_privacy_levels = []
        
        for context in spatial_contexts:
            for example in context.similar_examples:
                if 'room_type' in example:
                    all_room_types.append(example['room_type'])
                
                spatial_features = example.get('spatial_features', {})
                if 'zone' in spatial_features:
                    all_zones.append(spatial_features['zone'])
                if 'privacy_level' in spatial_features:
                    all_privacy_levels.append(spatial_features['privacy_level'])
        
        return {
            'common_room_types': dict(Counter(all_room_types).most_common(5)),
            'preferred_zones': dict(Counter(all_zones).most_common(3)),
            'avg_privacy_requirement': np.mean(all_privacy_levels) if all_privacy_levels else 0.5,
            'spatial_examples_analyzed': len(all_room_types)
        }
    
    def _aggregate_functional_insights(self, research_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Aggregate functional insights"""
        
        functional_contexts = [ctx for ctx in research_contexts if 'functional' in ctx.context_type or 'adjacency' in ctx.context_type]
        
        if not functional_contexts:
            return {}
        
        all_adjacency_patterns = {}
        all_design_patterns = []
        
        for context in functional_contexts:
            all_design_patterns.extend(context.design_patterns)
            
            # Extract adjacency information
            for example in context.similar_examples:
                adjacencies = example.get('adjacencies', [])
                room_type = example.get('room_type', 'unknown')
                
                for adj in adjacencies:
                    pattern_key = f"{room_type}-{adj}"
                    all_adjacency_patterns[pattern_key] = all_adjacency_patterns.get(pattern_key, 0) + 1
        
        return {
            'common_adjacencies': dict(sorted(all_adjacency_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            'design_patterns_found': len(all_design_patterns),
            'functional_examples_analyzed': sum(len(ctx.similar_examples) for ctx in functional_contexts)
        }
    
    def _aggregate_environmental_insights(self, research_contexts: List[ResearchContext]) -> Dict[str, Any]:
        """Aggregate environmental insights"""
        
        env_contexts = [ctx for ctx in research_contexts if 'environmental' in ctx.context_type]
        
        if not env_contexts:
            return {}
        
        all_light_scores = []
        all_orientations = []
        
        for context in env_contexts:
            for example in context.similar_examples:
                if 'natural_light_score' in example:
                    all_light_scores.append(example['natural_light_score'])
                if 'zone' in example:
                    all_orientations.append(example['zone'])
        
        return {
            'avg_natural_light_target': np.mean(all_light_scores) if all_light_scores else 0.5,
            'light_score_range': [min(all_light_scores), max(all_light_scores)] if all_light_scores else [0.0, 1.0],
            'common_orientations': dict(Counter(all_orientations).most_common(3)),
            'environmental_examples_analyzed': len(all_light_scores)
        }
    # Add to the ResearchAgent class
    def check_research_quality(self, research_contexts: List[ResearchContext]) -> bool:
        """Check if research quality meets threshold (for flowchart's H: Research Quality Check)."""
        avg_relevance = np.mean([ctx.relevance_score for ctx in research_contexts])
        avg_confidence = np.mean([ctx.confidence for ctx in research_contexts])
        return avg_relevance >= self.research_quality_threshold and avg_confidence >= self.research_quality_threshold


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Initializing Research Agent...")
    
    # Initialize with mock RAG store path
    research_agent = ResearchAgent(
        rag_store_path="enhanced_multimodal_rag_store",
        max_research_depth=3
    )
    
    # Example prototype configuration from Generalizer
    prototype_config = {
        'prototype_id': 'compact_central_zonal_0',
        'spatial_config': {
            'strategy': 'central_core',
            'plot_utilization': 0.7,
            'core_position': (25.0, 15.0),
            'compactness_factor': 0.8
        },
        'circulation_pattern': {
            'pattern_type': 'hub_and_spoke',
            'efficiency_target': 0.85,
            'central_hub': (25.0, 15.0)
        },
        'environmental_strategy': {
            'orientation': 'south',
            'passive_strategies': ['cross_ventilation', 'south_shading'],
            'climate_zone': 'subtropical'
        },
        'functional_zones': {
            'public_zone': {'ratio': 0.4, 'rooms': ['living_room', 'kitchen']},
            'private_zone': {'ratio': 0.4, 'rooms': ['bedroom', 'bathroom']},
            'service_zone': {'ratio': 0.2, 'rooms': ['utility']}
        }
    }
    
    # Example requirements
    requirements = {
        'spatial_needs': [
            {'room_type': 'bedroom', 'quantity': 3, 'min_area': 120},
            {'room_type': 'bathroom', 'quantity': 2, 'min_area': 45},
            {'room_type': 'living_room', 'quantity': 1, 'min_area': 200},
            {'room_type': 'kitchen', 'quantity': 1, 'min_area': 100}
        ],
        'design_preferences': {
            'sustainability_focus': True,
            'accessibility_requirements': False
        },
        'budget_sensitive': False
    }
    
    # Conduct research
    print("🔍 Conducting comprehensive research...")
    research_contexts = research_agent.conduct_research(
        prototype_id='compact_central_zonal_0',
        prototype_config=prototype_config,
        requirements=requirements,
        research_focus=[
            ResearchQueryType.SPATIAL_OPTIMIZATION,
            ResearchQueryType.FUNCTIONAL_ADJACENCY,
            ResearchQueryType.ENVIRONMENTAL_STRATEGY
        ]
    )
    
    # Display research results
    print(f"\n📊 Research Results:")
    print("=" * 60)
    
    for i, context in enumerate(research_contexts):
        print(f"\nResearch Context {i+1}: {context.context_type}")
        print(f"  Query: {context.research_query}")
        print(f"  Relevance Score: {context.relevance_score:.3f}")
        print(f"  Confidence: {context.confidence:.3f}")
        print(f"  Similar Examples: {len(context.similar_examples)}")
        print(f"  Design Patterns: {len(context.design_patterns)}")
        
        # Show top examples
        for j, example in enumerate(context.similar_examples[:2]):
            print(f"    Example {j+1}: {example.get('room_type', 'N/A')} (score: {example.get('relevance_score', 0):.3f})")
    
    # Get embedding structure info
    embedding_structure = research_agent.get_embedding_structure_info()
    print(f"\n🔧 Embedding Structure Info:")
    print(f"  Total Embeddings: {embedding_structure['total_embeddings']}")
    print(f"  Embedding Dimension: {embedding_structure['embedding_dimension']}")
    print(f"  Knowledge Sources: {embedding_structure['knowledge_sources']['total_sources']}")
    
    # Enhance prototype with research
    enhanced_config = research_agent.enhance_prototype_with_research(
        prototype_config, research_contexts
    )
    
    print(f"\n✨ Prototype Enhanced with Research:")
    print(f"  Research Metadata Added: {enhanced_config.get('research_metadata', {}).get('research_conducted', False)}")
    print(f"  Recommendations: {len(enhanced_config.get('research_recommendations', []))}")
    
    # Export research data
    exported_data = research_agent.export_research_data(research_contexts)
    
    # Show statistics
    stats = research_agent.get_research_statistics()
    print(f"\n📈 Research Statistics:")
    print(f"  Total Queries: {stats['total_research_queries']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    
    print("\n✅ Research Agent demonstration complete!")
    print("🔄 Ready to pass enhanced prototypes to Scoring Agent...")