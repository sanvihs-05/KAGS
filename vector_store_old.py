"""
RAG-Enhanced Multi-Language Annotation Processor for CubiCasa5K Dataset
FIXED VERSION - Handles SVG coordinate parsing issues
Creates highly informational vector embeddings with architectural context, spatial relationships,
and semantic information for advanced RAG applications
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Union
import logging
from dataclasses import dataclass, asdict
import re
from collections import Counter, defaultdict
import math

# Core libraries for language processing
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    print("Installing langdetect...")
    import subprocess
    subprocess.check_call(["pip", "install", "langdetect"])
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Installing deep-translator...")
    import subprocess
    subprocess.check_call(["pip", "install", "deep-translator"])
    from deep_translator import GoogleTranslator

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

logger = logging.getLogger(__name__)

@dataclass
class SpatialInfo:
    """Comprehensive spatial information for annotations"""
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None
    area: Optional[float] = None
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    relative_position: Optional[str] = None  # "top-left", "center", etc.
    zone: Optional[str] = None  # "north", "south", "east", "west"

@dataclass
class ArchitecturalContext:
    """Rich architectural context for embeddings"""
    room_function: str  # primary function
    room_privacy_level: str  # public, semi-private, private
    circulation_type: str  # main, secondary, service
    natural_light_access: bool
    ventilation_requirements: str  # high, medium, low
    plumbing_required: bool
    electrical_requirements: str  # standard, high, specialized
    acoustic_considerations: str  # quiet, moderate, noisy
    typical_adjacencies: List[str]  # commonly adjacent room types
    size_category: str  # small, medium, large
    accessibility_importance: str  # critical, important, standard

@dataclass
class EnrichedAnnotation:
    """Highly informational annotation for RAG applications"""
    # Basic text information
    original_text: str
    detected_language: str
    confidence: float
    normalized_text: str
    translated_text: str
    
    # Architectural classification
    room_type: Optional[str] = None
    room_function: Optional[str] = None
    semantic_category: Optional[str] = None
    architectural_context: Optional[ArchitecturalContext] = None
    
    # Spatial information
    spatial_info: Optional[SpatialInfo] = None
    adjacent_spaces: Optional[List[str]] = None
    accessibility_paths: Optional[List[str]] = None
    
    # Content classification
    is_dimension: bool = False
    is_abbreviation: bool = False
    is_room_label: bool = False
    is_furniture: bool = False
    is_fixture: bool = False
    
    # Embedding information
    text_embedding: Optional[np.ndarray] = None
    spatial_embedding: Optional[np.ndarray] = None
    contextual_embedding: Optional[np.ndarray] = None
    composite_embedding: Optional[np.ndarray] = None  # Combined for RAG
    
    # RAG-specific metadata
    rag_context: Optional[str] = None  # Rich context for RAG retrieval
    search_keywords: Optional[List[str]] = None
    semantic_tags: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize empty lists for None fields"""
        if self.adjacent_spaces is None:
            self.adjacent_spaces = []
        if self.accessibility_paths is None:
            self.accessibility_paths = []
        if self.search_keywords is None:
            self.search_keywords = []
        if self.semantic_tags is None:
            self.semantic_tags = []

@dataclass
class RAGVectorStore:
    """RAG-optimized vector store with rich metadata"""
    plan_id: str
    annotations: List[EnrichedAnnotation]
    
    # Spatial relationships
    adjacency_matrix: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    visibility_graph: Optional[Dict[str, List[str]]] = None
    circulation_paths: Optional[List[List[str]]] = None
    room_graph: Optional[Dict[str, List[str]]] = None
    
    # Layout analysis
    layout_type: Optional[str] = None  # open, compartmentalized, mixed
    layout_features: Optional[Dict[str, Any]] = None
    circulation_efficiency: Optional[float] = None
    space_utilization: Optional[float] = None
    privacy_zones: Optional[Dict[str, List[str]]] = None
    
    # RAG metadata
    embedding_matrix: Optional[np.ndarray] = None
    metadata_for_search: Optional[List[Dict[str, Any]]] = None
    search_index: Optional[Dict[str, List[int]]] = None  # keyword -> annotation indices
    
    # Multilingual support
    language_distribution: Optional[Dict[str, int]] = None
    dominant_language: str = 'fi'
    room_type_mapping: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Initialize empty dicts/lists for None fields"""
        if self.room_graph is None:
            self.room_graph = {}
        if self.layout_features is None:
            self.layout_features = {}
        if self.metadata_for_search is None:
            self.metadata_for_search = []
        if self.search_index is None:
            self.search_index = {}
        if self.language_distribution is None:
            self.language_distribution = {}
        if self.room_type_mapping is None:
            self.room_type_mapping = {}

class RAGEnhancedAnnotationProcessor:
    """RAG-focused processor with highly informational embeddings"""
    
    def __init__(self, 
                 target_language: str = 'en',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 cache_translations: bool = True):
        
        self.target_language = target_language
        self.cache_translations = cache_translations
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize translator
        self.translator = GoogleTranslator(source='auto', target=self.target_language)
        
        # Caches
        self.translation_cache = {}
        self.embedding_cache = {}
        
        # Enhanced Finnish architectural knowledge base
        self.finnish_room_types = self._initialize_finnish_room_types()
        self.architectural_contexts = self._initialize_architectural_contexts()
        self.spatial_relationships = self._initialize_spatial_relationships()
        self.dimension_patterns = self._initialize_dimension_patterns()
        
        # Statistics
        self.processing_stats = {
            'total_annotations': 0,
            'language_distribution': Counter(),
            'room_type_distribution': Counter(),
            'architectural_contexts': Counter(),
            'failed_detections': 0,
            'enhanced_embeddings_created': 0
        }
    
    def parse_svg_coordinate(self, coord_str: str) -> float:
        """Parse SVG coordinate string handling various units"""
        if not coord_str:
            return 0.0
        
        # Remove common SVG units and parse
        coord_str = str(coord_str).strip()
        
        # Handle units: px, pt, em, ex, %, in, cm, mm
        unit_conversions = {
            'px': 1.0,
            'pt': 1.33,  # 1pt = 1.33px
            'em': 16.0,  # Approximate: 1em = 16px
            'ex': 8.0,   # Approximate: 1ex = 8px
            'in': 96.0,  # 1in = 96px
            'cm': 37.8,  # 1cm = 37.8px
            'mm': 3.78,  # 1mm = 3.78px
            '%': 1.0     # Will handle percentage separately
        }
        
        # Try to extract number and unit
        coord_match = re.match(r'^([+-]?\d*\.?\d+)([a-z%]*)$', coord_str.lower())
        
        if coord_match:
            number_str, unit = coord_match.groups()
            try:
                number = float(number_str)
                
                if unit in unit_conversions:
                    return number * unit_conversions[unit]
                elif unit == '%':
                    # For percentage, return as-is (caller should handle context)
                    return number
                else:
                    # Unknown unit, return raw number
                    return number
            except ValueError:
                return 0.0
        
        # Fallback: try to parse as float directly
        try:
            return float(coord_str)
        except ValueError:
            logger.warning(f"Could not parse coordinate: {coord_str}")
            return 0.0
    
    def _initialize_finnish_room_types(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive Finnish room type knowledge base"""
        return {
            # Living spaces
            'oh': {
                'full_name': 'olohuone',
                'english': 'living_room',
                'function': 'social_relaxation',
                'privacy': 'semi-private',
                'typical_size': 'large',
                'adjacencies': ['keittiö', 'ruokailutila', 'eteinen'],
                'keywords': ['living', 'social', 'relaxation', 'family', 'entertainment'],
                'description': 'Main living and social space for family activities and relaxation'
            },
            'mh': {
                'full_name': 'makuuhuone',
                'english': 'bedroom',
                'function': 'rest_sleep',
                'privacy': 'private',
                'typical_size': 'medium',
                'adjacencies': ['kylpyhuone', 'käytävä', 'vaatehuone'],
                'keywords': ['bedroom', 'sleep', 'rest', 'private', 'personal'],
                'description': 'Private sleeping and resting space with storage for personal belongings'
            },
            'kh': {
                'full_name': 'kylpyhuone',
                'english': 'bathroom',
                'function': 'hygiene_bathing',
                'privacy': 'private',
                'typical_size': 'small',
                'adjacencies': ['makuuhuone', 'käytävä', 'sauna'],
                'keywords': ['bathroom', 'hygiene', 'bathing', 'washing', 'private'],
                'description': 'Private space for bathing, personal hygiene, and grooming activities'
            },
            'vh': {
                'full_name': 'vesihuone',
                'english': 'utility_room',
                'function': 'utility_cleaning',
                'privacy': 'service',
                'typical_size': 'small',
                'adjacencies': ['keittiö', 'eteinen', 'tekninen_tila'],
                'keywords': ['utility', 'laundry', 'cleaning', 'water', 'service'],
                'description': 'Service space for laundry, cleaning equipment, and utility functions'
            },
            'keittiö': {
                'full_name': 'keittiö',
                'english': 'kitchen',
                'function': 'food_preparation',
                'privacy': 'semi-private',
                'typical_size': 'medium',
                'adjacencies': ['ruokailutila', 'olohuone', 'eteinen'],
                'keywords': ['kitchen', 'cooking', 'food', 'preparation', 'dining'],
                'description': 'Food preparation and cooking space, often connected to dining areas'
            },
            'wc': {
                'full_name': 'wc',
                'english': 'toilet',
                'function': 'sanitation',
                'privacy': 'private',
                'typical_size': 'small',
                'adjacencies': ['käytävä', 'eteinen'],
                'keywords': ['toilet', 'wc', 'sanitation', 'restroom', 'private'],
                'description': 'Small private space for sanitation and personal needs'
            },
            'sauna': {
                'full_name': 'sauna',
                'english': 'sauna',
                'function': 'wellness_bathing',
                'privacy': 'private',
                'typical_size': 'small',
                'adjacencies': ['kylpyhuone', 'peseytymistila', 'ulko-ovi'],
                'keywords': ['sauna', 'wellness', 'relaxation', 'steam', 'bathing'],
                'description': 'Traditional Finnish steam room for relaxation and wellness'
            },
            'eteinen': {
                'full_name': 'eteinen',
                'english': 'entrance_hall',
                'function': 'circulation_entry',
                'privacy': 'public',
                'typical_size': 'small',
                'adjacencies': ['olohuone', 'keittiö', 'käytävä'],
                'keywords': ['entrance', 'entry', 'foyer', 'circulation', 'access'],
                'description': 'Entry space providing access to other rooms and storage for outerwear'
            },
            'varasto': {
                'full_name': 'varasto',
                'english': 'storage',
                'function': 'storage',
                'privacy': 'service',
                'typical_size': 'small',
                'adjacencies': ['käytävä', 'tekninen_tila'],
                'keywords': ['storage', 'closet', 'pantry', 'organization'],
                'description': 'Storage space for household items, seasonal goods, or equipment'
            },
            'tekn': {
                'full_name': 'tekninen_tila',
                'english': 'technical_room',
                'function': 'mechanical_systems',
                'privacy': 'service',
                'typical_size': 'small',
                'adjacencies': ['varasto', 'ulko-ovi'],
                'keywords': ['technical', 'mechanical', 'hvac', 'utilities', 'systems'],
                'description': 'Technical space housing mechanical systems, HVAC, or utilities'
            },
            'parveke': {
                'full_name': 'parveke',
                'english': 'balcony',
                'function': 'outdoor_relaxation',
                'privacy': 'semi-private',
                'typical_size': 'small',
                'adjacencies': ['olohuone', 'makuuhuone', 'keittiö'],
                'keywords': ['balcony', 'outdoor', 'terrace', 'fresh_air', 'views'],
                'description': 'Outdoor space connected to interior rooms for fresh air and views'
            },
            'ulkotila': {
                'full_name': 'ulkotila',
                'english': 'outdoor_space',
                'function': 'outdoor_activities',
                'privacy': 'public',
                'typical_size': 'variable',
                'adjacencies': ['parveke', 'eteinen', 'terassi'],
                'keywords': ['outdoor', 'exterior', 'garden', 'yard', 'landscape'],
                'description': 'External outdoor area for various outdoor activities and landscaping'
            }
        }
    
    def _initialize_architectural_contexts(self) -> Dict[str, ArchitecturalContext]:
        """Initialize rich architectural contexts for different room types"""
        contexts = {}
        
        room_configs = {
            'living_room': {
                'room_function': 'social_entertainment',
                'room_privacy_level': 'semi-private',
                'circulation_type': 'main',
                'natural_light_access': True,
                'ventilation_requirements': 'medium',
                'plumbing_required': False,
                'electrical_requirements': 'high',
                'acoustic_considerations': 'moderate',
                'typical_adjacencies': ['kitchen', 'dining_room', 'entrance'],
                'size_category': 'large',
                'accessibility_importance': 'critical'
            },
            'bedroom': {
                'room_function': 'rest_sleep',
                'room_privacy_level': 'private',
                'circulation_type': 'secondary',
                'natural_light_access': True,
                'ventilation_requirements': 'medium',
                'plumbing_required': False,
                'electrical_requirements': 'standard',
                'acoustic_considerations': 'quiet',
                'typical_adjacencies': ['bathroom', 'hallway', 'closet'],
                'size_category': 'medium',
                'accessibility_importance': 'important'
            },
            'kitchen': {
                'room_function': 'food_preparation',
                'room_privacy_level': 'semi-private',
                'circulation_type': 'main',
                'natural_light_access': True,
                'ventilation_requirements': 'high',
                'plumbing_required': True,
                'electrical_requirements': 'high',
                'acoustic_considerations': 'moderate',
                'typical_adjacencies': ['dining_room', 'living_room', 'pantry'],
                'size_category': 'medium',
                'accessibility_importance': 'critical'
            },
            'bathroom': {
                'room_function': 'hygiene_sanitation',
                'room_privacy_level': 'private',
                'circulation_type': 'secondary',
                'natural_light_access': False,
                'ventilation_requirements': 'high',
                'plumbing_required': True,
                'electrical_requirements': 'standard',
                'acoustic_considerations': 'moderate',
                'typical_adjacencies': ['bedroom', 'hallway'],
                'size_category': 'small',
                'accessibility_importance': 'critical'
            },
            'utility_room': {
                'room_function': 'cleaning_maintenance',
                'room_privacy_level': 'service',
                'circulation_type': 'service',
                'natural_light_access': False,
                'ventilation_requirements': 'high',
                'plumbing_required': True,
                'electrical_requirements': 'standard',
                'acoustic_considerations': 'noisy',
                'typical_adjacencies': ['kitchen', 'entrance', 'storage'],
                'size_category': 'small',
                'accessibility_importance': 'standard'
            }
        }
        
        for room_type, config in room_configs.items():
            contexts[room_type] = ArchitecturalContext(**config)
        
        return contexts
    
    def _initialize_spatial_relationships(self) -> Dict[str, List[str]]:
        """Define typical spatial relationships between room types"""
        return {
            'living_room': ['kitchen', 'dining_room', 'entrance', 'hallway', 'balcony'],
            'kitchen': ['living_room', 'dining_room', 'utility_room', 'pantry', 'balcony'],
            'bedroom': ['bathroom', 'hallway', 'closet', 'balcony'],
            'bathroom': ['bedroom', 'hallway', 'utility_room'],
            'entrance': ['living_room', 'kitchen', 'hallway', 'storage'],
            'hallway': ['bedroom', 'bathroom', 'storage', 'entrance'],
            'utility_room': ['kitchen', 'bathroom', 'entrance', 'storage'],
            'storage': ['hallway', 'entrance', 'utility_room'],
            'sauna': ['bathroom', 'utility_room', 'balcony'],
            'balcony': ['living_room', 'bedroom', 'kitchen']
        }
    
    def _initialize_dimension_patterns(self) -> List[re.Pattern]:
        """Initialize patterns for detecting dimensions"""
        patterns = [
            re.compile(r"\d+['\"]?\s*x\s*\d+['\"]?"),  # 10' x 12' or 10 x 12
            re.compile(r"\d+[.,]\d+\s*m\s*x\s*\d+[.,]\d+\s*m"),  # 3.5 m x 4.2 m
            re.compile(r"\d+[.,]\d+\s*x\s*\d+[.,]\d+"),  # 3.5 x 4.2
            re.compile(r"\d+\s*m²"),  # 25 m²
            re.compile(r"\d+\s*sq\s*ft"),  # 250 sq ft
        ]
        return patterns
    
    def detect_language_enhanced(self, text: str) -> Tuple[str, float]:
        """Enhanced language detection with Finnish architectural context"""
        
        if not text or len(text.strip()) < 2:
            return 'unknown', 0.0
        
        text_lower = text.lower().strip()
        
        # Check for Finnish architectural abbreviations first
        finnish_indicators = ['mh', 'oh', 'kh', 'vh', 'wc', 'sauna', 'eteinen', 'varasto', 
                            'keittiö', 'parveke', 'ulkotila', 'tekn', 'khh']
        
        if any(abbr in text_lower for abbr in finnish_indicators):
            return 'fi', 0.95
        
        # Check for dimension patterns (language neutral)
        for pattern in self.dimension_patterns:
            if pattern.search(text):
                return 'neutral', 0.9  # Dimensions are language neutral
        
        # Use langdetect for other cases
        try:
            cleaned_text = self._clean_text_for_detection(text)
            if len(cleaned_text) < 2:
                return 'unknown', 0.0
            
            lang_probs = detect_langs(cleaned_text)
            if lang_probs:
                best_lang = lang_probs[0]
                # Boost Finnish confidence if architectural terms present
                if best_lang.lang == 'fi' or any(term in text_lower for term in ['huone', 'tila', 'keittiö']):
                    return 'fi', min(0.95, best_lang.prob + 0.2)
                return best_lang.lang, best_lang.prob
            else:
                return 'unknown', 0.0
                
        except LangDetectException:
            self.processing_stats['failed_detections'] += 1
            return 'unknown', 0.0
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        # Remove dimensions and numbers first
        cleaned = re.sub(r'\d+[.,]?\d*\s*[mx²\'"]+', '', text)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def normalize_room_type_enhanced(self, text: str, language: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced room type normalization with rich context"""
        
        text_lower = text.lower().strip()
        
        # Check Finnish room types first
        if text_lower in self.finnish_room_types:
            room_info = self.finnish_room_types[text_lower]
            return room_info['english'], room_info
        
        # Check for partial matches
        for abbr, info in self.finnish_room_types.items():
            if abbr in text_lower or text_lower in abbr:
                return info['english'], info
            if info['full_name'] in text_lower or text_lower in info['full_name']:
                return info['english'], info
        
        # Check dimensions
        for pattern in self.dimension_patterns:
            if pattern.search(text):
                return 'dimension', {
                    'english': 'dimension',
                    'function': 'measurement',
                    'keywords': ['dimension', 'measurement', 'size'],
                    'description': f'Dimensional measurement: {text}'
                }
        
        # Default fallback
        return text_lower.replace(' ', '_'), {
            'english': text_lower.replace(' ', '_'),
            'function': 'unknown',
            'keywords': [text_lower],
            'description': f'Unknown room type: {text}'
        }
    
    def create_rag_context(self, annotation: EnrichedAnnotation) -> str:
        """Create rich contextual description for RAG retrieval"""
        
        context_parts = []
        
        # Basic information
        context_parts.append(f"Room: {annotation.translated_text}")
        
        if annotation.room_type and annotation.room_type != annotation.translated_text:
            context_parts.append(f"Type: {annotation.room_type}")
        
        # Architectural context
        if annotation.architectural_context:
            ctx = annotation.architectural_context
            context_parts.append(f"Function: {ctx.room_function}")
            context_parts.append(f"Privacy level: {ctx.room_privacy_level}")
            context_parts.append(f"Size category: {ctx.size_category}")
            
            if ctx.plumbing_required:
                context_parts.append("Requires plumbing")
            
            if ctx.natural_light_access:
                context_parts.append("Has natural light access")
            
            context_parts.append(f"Ventilation needs: {ctx.ventilation_requirements}")
            context_parts.append(f"Electrical requirements: {ctx.electrical_requirements}")
            
            if ctx.typical_adjacencies:
                context_parts.append(f"Typically adjacent to: {', '.join(ctx.typical_adjacencies)}")
        
        # Spatial information
        if annotation.spatial_info:
            spatial = annotation.spatial_info
            if spatial.area:
                context_parts.append(f"Area: {spatial.area:.1f} sq units")
            if spatial.relative_position:
                context_parts.append(f"Position: {spatial.relative_position}")
            if spatial.zone:
                context_parts.append(f"Zone: {spatial.zone}")
        
        # Adjacent spaces
        if annotation.adjacent_spaces:
            context_parts.append(f"Adjacent to: {', '.join(annotation.adjacent_spaces)}")
        
        # Content classification
        classifications = []
        if annotation.is_dimension:
            classifications.append("dimensional measurement")
        if annotation.is_room_label:
            classifications.append("room label")
        if annotation.is_furniture:
            classifications.append("furniture")
        if annotation.is_fixture:
            classifications.append("fixture")
        
        if classifications:
            context_parts.append(f"Content type: {', '.join(classifications)}")
        
        return " | ".join(context_parts)
    
    def create_enhanced_embedding(self, annotation: EnrichedAnnotation) -> np.ndarray:
        """Create highly informational composite embedding for RAG"""
        
        embeddings_to_combine = []
        
        # 1. Text embedding (base)
        text_for_embedding = annotation.translated_text
        if annotation.room_function:
            text_for_embedding += f" {annotation.room_function}"
        
        text_embedding = self.embedding_model.encode(text_for_embedding)
        embeddings_to_combine.append(text_embedding)
        
        # 2. Contextual embedding (architectural context)
        if annotation.rag_context:
            context_embedding = self.embedding_model.encode(annotation.rag_context)
            embeddings_to_combine.append(context_embedding * 0.7)  # Weight contextual info
        
        # 3. Semantic tags embedding
        if annotation.semantic_tags:
            tags_text = " ".join(annotation.semantic_tags)
            tags_embedding = self.embedding_model.encode(tags_text)
            embeddings_to_combine.append(tags_embedding * 0.5)  # Weight semantic tags
        
        # 4. Spatial embedding (encode spatial relationships)
        if annotation.adjacent_spaces:
            spatial_text = f"adjacent to {' '.join(annotation.adjacent_spaces)}"
            spatial_embedding = self.embedding_model.encode(spatial_text)
            embeddings_to_combine.append(spatial_embedding * 0.3)  # Weight spatial info
        
        # Combine embeddings
        if len(embeddings_to_combine) > 1:
            # Weighted average of embeddings
            weights = [1.0] + [0.7, 0.5, 0.3][:len(embeddings_to_combine)-1]
            composite_embedding = np.average(embeddings_to_combine, axis=0, weights=weights)
        else:
            composite_embedding = embeddings_to_combine[0]
        
        # Normalize the final embedding
        composite_embedding = composite_embedding / np.linalg.norm(composite_embedding)
        
        return composite_embedding
    
    def process_annotation_enhanced(self, text: str, x: float = 0, y: float = 0) -> EnrichedAnnotation:
        """Process annotation with enhanced information extraction"""
        
        # Basic language detection
        detected_lang, confidence = self.detect_language_enhanced(text)
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Translate if needed
        if detected_lang == 'fi' or detected_lang == 'unknown':
            translated_text = self.translate_finnish_text(normalized_text)
        else:
            translated_text = normalized_text
        
        # Enhanced room type detection
        room_type, room_info = self.normalize_room_type_enhanced(normalized_text, detected_lang)
        
        # Content classification
        is_dimension = any(pattern.search(text) for pattern in self.dimension_patterns)
        is_abbreviation = len(text.strip()) <= 3 and text.isupper()
        is_room_label = room_type in self.finnish_room_types or room_type in [info['english'] for info in self.finnish_room_types.values()]
        
        # Create spatial info
        spatial_info = SpatialInfo(
            x=x, y=y,
            center_x=x, center_y=y,
            relative_position=self._determine_relative_position(x, y),
            zone=self._determine_zone(x, y)
        )
        
        # Get architectural context
        architectural_context = None
        if room_type in self.architectural_contexts:
            architectural_context = self.architectural_contexts[room_type]
        
        # Create semantic tags
        semantic_tags = self._generate_semantic_tags(room_info, is_dimension, is_room_label)
        
        # Create annotation
        annotation = EnrichedAnnotation(
            original_text=text,
            detected_language=detected_lang,
            confidence=confidence,
            normalized_text=normalized_text,
            translated_text=translated_text,
            room_type=room_type,
            room_function=room_info.get('function', 'unknown'),
            semantic_category=self._determine_semantic_category(room_info),
            architectural_context=architectural_context,
            spatial_info=spatial_info,
            is_dimension=is_dimension,
            is_abbreviation=is_abbreviation,
            is_room_label=is_room_label,
            semantic_tags=semantic_tags,
            search_keywords=self._generate_search_keywords(room_info, text)
        )
        
        # Create RAG context
        annotation.rag_context = self.create_rag_context(annotation)
        
        # Create enhanced embedding
        annotation.composite_embedding = self.create_enhanced_embedding(annotation)
        
        # Update statistics
        self.processing_stats['total_annotations'] += 1
        self.processing_stats['language_distribution'][detected_lang] += 1
        self.processing_stats['room_type_distribution'][room_type] += 1
        self.processing_stats['enhanced_embeddings_created'] += 1
        
        return annotation
    
    def translate_finnish_text(self, text: str) -> str:
        """Enhanced Finnish text translation with architectural terms"""
        
        text_lower = text.lower().strip()
        
        # Direct Finnish architectural translation
        direct_translations = {
            'makuuhuone': 'bedroom',
            'olohuone': 'living room',
            'keittiö': 'kitchen',
            'kylpyhuone': 'bathroom',
            'vesihuone': 'utility room',
            'eteinen': 'entrance hall',
            'käytävä': 'hallway',
            'varasto': 'storage room',
            'sauna': 'sauna',
            'parveke': 'balcony',
            'ulkotila': 'outdoor space',
            'tekninen tila': 'technical room',
            'wc': 'toilet',
            'khh': 'utility room',
            'kodinhoitotila': 'utility room'
        }
        
        if text_lower in direct_translations:
            return direct_translations[text_lower]
        
        # Check abbreviations
        if text_lower in self.finnish_room_types:
            return self.finnish_room_types[text_lower]['english'].replace('_', ' ')
        
        # Use Google Translate for other cases
        try:
            if self.cache_translations and text in self.translation_cache:
                return self.translation_cache[text]
            
            translated = self.translator.translate(text)
            
            if self.cache_translations:
                self.translation_cache[text] = translated
            
            return translated
        except:
            return text  # Return original if translation fails
    
    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common Finnish abbreviations
        abbreviation_expansions = {
            'mh': 'makuuhuone',
            'oh': 'olohuone', 
            'kh': 'kylpyhuone',
            'vh': 'vesihuone',
            'khh': 'kodinhoitotila',
            'et': 'eteinen',
            'var': 'varasto',
            'tekn': 'tekninen tila'
        }
        
        normalized_lower = normalized.lower()
        for abbr, full in abbreviation_expansions.items():
            if normalized_lower == abbr:
                return full
            elif normalized_lower.startswith(abbr + ' '):
                return normalized.replace(abbr, full, 1)
        
        return normalized
    
    def _determine_relative_position(self, x: float, y: float, width: float = 1000, height: float = 800) -> str:
        """Determine relative position within floor plan"""
        x_pos = "left" if x < width * 0.33 else "center" if x < width * 0.67 else "right"
        y_pos = "top" if y < height * 0.33 else "middle" if y < height * 0.67 else "bottom"
        return f"{y_pos}-{x_pos}"
    
    def _determine_zone(self, x: float, y: float, width: float = 1000, height: float = 800) -> str:
        """Determine architectural zone"""
        center_x, center_y = width / 2, height / 2
        
        if x < center_x and y < center_y:
            return "northwest"
        elif x >= center_x and y < center_y:
            return "northeast"
        elif x < center_x and y >= center_y:
            return "southwest"
        else:
            return "southeast"
    
    def _generate_semantic_tags(self, room_info: Dict[str, Any], is_dimension: bool, is_room_label: bool) -> List[str]:
        """Generate semantic tags for better RAG retrieval"""
        tags = []
        
        if is_dimension:
            tags.extend(['dimension', 'measurement', 'size', 'area'])
        
        if is_room_label:
            tags.append('room_label')
        
        if 'keywords' in room_info:
            tags.extend(room_info['keywords'])
        
        if 'function' in room_info:
            tags.append(room_info['function'])
        
        # Add contextual tags based on function
        function = room_info.get('function', '')
        if 'hygiene' in function or 'bathroom' in function:
            tags.extend(['water', 'plumbing', 'privacy', 'sanitation'])
        elif 'food' in function or 'kitchen' in function:
            tags.extend(['cooking', 'dining', 'appliances', 'ventilation'])
        elif 'sleep' in function or 'rest' in function:
            tags.extend(['quiet', 'private', 'comfort', 'personal'])
        elif 'social' in function:
            tags.extend(['gathering', 'entertainment', 'family', 'open'])
        
        return list(set(tags))  # Remove duplicates
    
    def _determine_semantic_category(self, room_info: Dict[str, Any]) -> str:
        """Determine high-level semantic category"""
        function = room_info.get('function', 'unknown')
        
        if any(term in function for term in ['social', 'entertainment', 'living']):
            return 'social_space'
        elif any(term in function for term in ['sleep', 'rest', 'private']):
            return 'private_space'
        elif any(term in function for term in ['hygiene', 'bathroom', 'sanitation']):
            return 'hygiene_space'
        elif any(term in function for term in ['food', 'cooking', 'kitchen']):
            return 'service_space'
        elif any(term in function for term in ['storage', 'utility', 'technical']):
            return 'utility_space'
        elif any(term in function for term in ['circulation', 'entry', 'hallway']):
            return 'circulation_space'
        elif any(term in function for term in ['outdoor', 'balcony', 'terrace']):
            return 'outdoor_space'
        else:
            return 'general_space'
    
    def _generate_search_keywords(self, room_info: Dict[str, Any], original_text: str) -> List[str]:
        """Generate comprehensive search keywords for RAG retrieval"""
        keywords = [original_text.lower()]
        
        if 'keywords' in room_info:
            keywords.extend(room_info['keywords'])
        
        if 'english' in room_info:
            keywords.append(room_info['english'])
        
        if 'full_name' in room_info:
            keywords.append(room_info['full_name'])
        
        # Add functional keywords
        function = room_info.get('function', '')
        if function != 'unknown':
            keywords.extend(function.split('_'))
        
        return list(set(keywords))
    
    def extract_svg_elements_enhanced(self, svg_file: Path) -> List[Tuple[str, float, float]]:
        """FIXED: Extract text elements with spatial coordinates handling various units"""
        
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            
            text_elements = []
            
            # SVG namespace handling
            namespaces = {
                'svg': 'http://www.w3.org/2000/svg'
            }
            
            # Extract text elements with coordinates
            for text_elem in root.findall('.//svg:text', namespaces):
                text_content = ''.join(text_elem.itertext()).strip()
                if text_content and len(text_content) > 0:
                    # Get coordinates with enhanced parsing
                    x_str = text_elem.get('x', '0')
                    y_str = text_elem.get('y', '0')
                    
                    x = self.parse_svg_coordinate(x_str)
                    y = self.parse_svg_coordinate(y_str)
                    
                    text_elements.append((text_content, x, y))
            
            # Also check for text without namespace
            for text_elem in root.findall('.//text'):
                text_content = ''.join(text_elem.itertext()).strip()
                if text_content and len(text_content) > 0:
                    # Get coordinates with enhanced parsing
                    x_str = text_elem.get('x', '0')
                    y_str = text_elem.get('y', '0')
                    
                    x = self.parse_svg_coordinate(x_str)
                    y = self.parse_svg_coordinate(y_str)
                    
                    text_elements.append((text_content, x, y))
            
            # Handle tspan elements (text spans)
            for tspan_elem in root.findall('.//svg:tspan', namespaces):
                text_content = ''.join(tspan_elem.itertext()).strip()
                if text_content and len(text_content) > 0:
                    x_str = tspan_elem.get('x', '0')
                    y_str = tspan_elem.get('y', '0')
                    
                    x = self.parse_svg_coordinate(x_str)
                    y = self.parse_svg_coordinate(y_str)
                    
                    text_elements.append((text_content, x, y))
            
            for tspan_elem in root.findall('.//tspan'):
                text_content = ''.join(tspan_elem.itertext()).strip()
                if text_content and len(text_content) > 0:
                    x_str = tspan_elem.get('x', '0')
                    y_str = tspan_elem.get('y', '0')
                    
                    x = self.parse_svg_coordinate(x_str)
                    y = self.parse_svg_coordinate(y_str)
                    
                    text_elements.append((text_content, x, y))
            
            print(f"   Extracted {len(text_elements)} text elements from {svg_file.name}")
            
            return text_elements
            
        except Exception as e:
            logger.error(f"Failed to extract SVG elements from {svg_file}: {e}")
            return []
    
    def calculate_spatial_relationships(self, annotations: List[EnrichedAnnotation]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        """Calculate spatial adjacency matrix and room graph"""
        
        n = len(annotations)
        adjacency_matrix = np.zeros((n, n))
        room_graph = defaultdict(list)
        
        # Define proximity threshold (adjust based on your floor plan scale)
        proximity_threshold = 100.0  # pixels or units
        
        for i, ann1 in enumerate(annotations):
            if not ann1.spatial_info or not ann1.is_room_label:
                continue
                
            for j, ann2 in enumerate(annotations):
                if i == j or not ann2.spatial_info or not ann2.is_room_label:
                    continue
                
                # Calculate distance
                dx = ann1.spatial_info.x - ann2.spatial_info.x
                dy = ann1.spatial_info.y - ann2.spatial_info.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # If rooms are close, consider them adjacent
                if distance < proximity_threshold:
                    adjacency_matrix[i, j] = 1
                    room_graph[ann1.room_type].append(ann2.room_type)
        
        return adjacency_matrix, dict(room_graph)
    
    def analyze_layout_features(self, annotations: List[EnrichedAnnotation], room_graph: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze layout features for RAG context"""
        
        features = {}
        
        # Count room types
        room_counts = Counter(ann.room_type for ann in annotations if ann.is_room_label)
        features['room_distribution'] = dict(room_counts)
        
        # Analyze connectivity
        total_connections = sum(len(adjacencies) for adjacencies in room_graph.values())
        features['connectivity_score'] = total_connections / max(len(room_graph), 1)
        
        # Identify layout patterns
        if 'living_room' in room_graph and 'kitchen' in room_graph:
            if 'kitchen' in room_graph.get('living_room', []):
                features['layout_type'] = 'open_plan'
            else:
                features['layout_type'] = 'separated_plan'
        else:
            features['layout_type'] = 'unknown'
        
        # Privacy analysis
        private_rooms = ['bedroom', 'bathroom']
        public_rooms = ['living_room', 'kitchen', 'entrance_hall']
        
        private_count = sum(1 for room in room_counts if room in private_rooms)
        public_count = sum(1 for room in room_counts if room in public_rooms)
        
        features['privacy_ratio'] = private_count / max(public_count, 1)
        
        # Spatial efficiency
        room_labels = [ann for ann in annotations if ann.is_room_label]
        if len(room_labels) > 0:
            avg_x = np.mean([ann.spatial_info.x for ann in room_labels])
            avg_y = np.mean([ann.spatial_info.y for ann in room_labels])
            spread = np.mean([abs(ann.spatial_info.x - avg_x) + abs(ann.spatial_info.y - avg_y) for ann in room_labels])
            features['spatial_compactness'] = 1000 / max(spread, 1)  # Inverse of spread
        
        return features
    
    def process_svg_for_rag(self, svg_file: Path) -> Optional[RAGVectorStore]:
        """Process SVG file and create RAG-optimized vector store"""
        
        # Extract text elements with coordinates
        text_elements = self.extract_svg_elements_enhanced(svg_file)
        
        if not text_elements:
            logger.debug(f"No text elements found in {svg_file}")
            return None
        
        # Process each annotation
        annotations = []
        for text, x, y in text_elements:
            try:
                annotation = self.process_annotation_enhanced(text, x, y)
                annotations.append(annotation)
            except Exception as e:
                logger.warning(f"Failed to process annotation '{text}': {e}")
                continue
        
        if not annotations:
            return None
        
        # Calculate spatial relationships
        adjacency_matrix, room_graph = self.calculate_spatial_relationships(annotations)
        
        # Update adjacency information in annotations
        for i, annotation in enumerate(annotations):
            adjacent_indices = np.where(adjacency_matrix[i] == 1)[0]
            adjacent_rooms = [annotations[j].room_type for j in adjacent_indices if annotations[j].is_room_label and annotations[j].room_type]
            annotation.adjacent_spaces = adjacent_rooms if adjacent_rooms else []
        
        # Recalculate embeddings with adjacency information
        for annotation in annotations:
            annotation.composite_embedding = self.create_enhanced_embedding(annotation)
        
        # Analyze layout features
        layout_features = self.analyze_layout_features(annotations, room_graph)
        
        # Create embedding matrix
        embeddings = [ann.composite_embedding for ann in annotations if ann.composite_embedding is not None]
        embedding_matrix = np.vstack(embeddings) if embeddings else None
        
        # Create metadata for search
        metadata_for_search = []
        search_index = defaultdict(list)
        
        for i, annotation in enumerate(annotations):
            metadata = {
                'text': annotation.original_text,
                'translated': annotation.translated_text,
                'room_type': annotation.room_type,
                'function': annotation.room_function,
                'semantic_category': annotation.semantic_category,
                'rag_context': annotation.rag_context,
                'keywords': annotation.search_keywords,
                'semantic_tags': annotation.semantic_tags,
                'spatial_zone': annotation.spatial_info.zone if annotation.spatial_info else None,
                'adjacent_rooms': annotation.adjacent_spaces,
                'is_room_label': annotation.is_room_label,
                'is_dimension': annotation.is_dimension
            }
            metadata_for_search.append(metadata)
            
            # Build search index
            if annotation.search_keywords:
                for keyword in annotation.search_keywords:
                    search_index[keyword.lower()].append(i)
            
            if annotation.semantic_tags:
                for tag in annotation.semantic_tags:
                    search_index[tag.lower()].append(i)
        
        # Language analysis
        language_dist = Counter(ann.detected_language for ann in annotations)
        dominant_lang = language_dist.most_common(1)[0][0] if language_dist else 'unknown'
        
        # Room type mapping
        room_type_mapping = {ann.original_text: ann.room_type for ann in annotations}
        
        return RAGVectorStore(
            plan_id=svg_file.stem,
            annotations=annotations,
            adjacency_matrix=adjacency_matrix,
            room_graph=dict(room_graph),
            layout_features=layout_features,
            embedding_matrix=embedding_matrix,
            metadata_for_search=metadata_for_search,
            search_index=dict(search_index),
            language_distribution=dict(language_dist),
            dominant_language=dominant_lang,
            room_type_mapping=room_type_mapping
        )
    
    def create_comprehensive_rag_store(self, 
                                     dataset_path: str,
                                     output_path: str = "rag_vector_store",
                                     max_files: Optional[int] = None) -> Dict[str, Any]:
        """Create comprehensive RAG vector store from CubiCasa5K dataset"""
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🏗️ Creating comprehensive RAG vector store...")
        
        # Find SVG files
        svg_files = []
        
        # Look for high_quality_architectural directory
        target_dir = dataset_path / "high_quality_architectural" if (dataset_path / "high_quality_architectural").exists() else dataset_path
        
        if target_dir.exists():
            for num_dir in target_dir.iterdir():
                if num_dir.is_dir() and num_dir.name.isdigit():
                    svg_files_in_dir = list(num_dir.glob("*.svg"))
                    if svg_files_in_dir:
                        # Select best representative file
                        best_file = self.select_best_svg_file(num_dir)
                        if best_file:
                            svg_files.append(best_file)
        else:
            # Fallback: search entire dataset
            svg_files = list(dataset_path.rglob("*.svg"))
        
        if max_files:
            svg_files = svg_files[:max_files]
        
        print(f"📊 Processing {len(svg_files)} SVG files...")
        
        # Process all files
        processed_stores = []
        failed_files = []
        
        for i, svg_file in enumerate(svg_files):
            if i % 10 == 0 and i > 0:
                print(f"   Processed {i}/{len(svg_files)} files...")
            
            try:
                rag_store = self.process_svg_for_rag(svg_file)
                if rag_store and len(rag_store.annotations) > 0:
                    processed_stores.append(rag_store)
                    print(f"   ✅ {svg_file.name}: {len(rag_store.annotations)} annotations")
                else:
                    failed_files.append(str(svg_file))
                    print(f"   ❌ {svg_file.name}: No annotations found")
            except Exception as e:
                logger.error(f"Failed to process {svg_file}: {e}")
                failed_files.append(str(svg_file))
                print(f"   ❌ {svg_file.name}: Error - {e}")
        
        print(f"\n✅ Successfully processed {len(processed_stores)} floor plans")
        print(f"❌ Failed to process {len(failed_files)} files")
        
        if not processed_stores:
            return {'error': 'No files processed successfully'}
        
        # Consolidate all embeddings and metadata
        all_embeddings = []
        all_metadata = []
        global_search_index = defaultdict(list)
        plan_index = {}  # Maps global indices to plan_id
        
        global_idx = 0
        for store in processed_stores:
            for i, annotation in enumerate(store.annotations):
                if annotation.composite_embedding is not None:
                    all_embeddings.append(annotation.composite_embedding)
                    
                    # Enhanced metadata with plan context
                    metadata = store.metadata_for_search[i].copy()
                    metadata['plan_id'] = store.plan_id
                    metadata['global_index'] = global_idx
                    metadata['layout_type'] = store.layout_features.get('layout_type', 'unknown')
                    metadata['connectivity_score'] = store.layout_features.get('connectivity_score', 0)
                    metadata['room_distribution'] = store.layout_features.get('room_distribution', {})
                    
                    all_metadata.append(metadata)
                    plan_index[global_idx] = store.plan_id
                    
                    # Update global search index
                    if annotation.search_keywords:
                        for keyword in annotation.search_keywords:
                            global_search_index[keyword.lower()].append(global_idx)
                    
                    if annotation.semantic_tags:
                        for tag in annotation.semantic_tags:
                            global_search_index[tag.lower()].append(global_idx)
                    
                    global_idx += 1
        
        # Create final embedding matrix
        if all_embeddings:
            final_embedding_matrix = np.vstack(all_embeddings)
        else:
            return {'error': 'No embeddings created'}
        
        # Save comprehensive RAG store
        self._save_rag_vector_store(
            embedding_matrix=final_embedding_matrix,
            metadata=all_metadata,
            search_index=dict(global_search_index),
            plan_index=plan_index,
            processed_stores=processed_stores,
            output_path=output_path
        )
        
        # Generate comprehensive statistics
        stats = self._generate_rag_statistics(processed_stores, all_metadata, failed_files)
        
        print(f"\n🎯 RAG Vector Store Created Successfully!")
        print(f"   Total vectors: {len(all_embeddings)}")
        print(f"   Total floor plans: {len(processed_stores)}")
        print(f"   Total search keywords: {len(global_search_index)}")
        print(f"   Average annotations per plan: {len(all_embeddings) / len(processed_stores):.1f}")
        
        return stats
    
    def select_best_svg_file(self, directory: Path) -> Optional[Path]:
        """Select the best representative SVG file from a directory"""
        svg_files = list(directory.glob("*.svg"))
        if not svg_files:
            return None
        
        # Priority order
        priorities = ["model.svg", "F1_original.svg", "scaled.svg", "F1_scaled.svg"]
        
        for priority_file in priorities:
            candidate = directory / priority_file
            if candidate.exists():
                return candidate
        
        # Fallback: largest file
        return max(svg_files, key=lambda f: f.stat().st_size)
    
    def _save_rag_vector_store(self, 
                              embedding_matrix: np.ndarray,
                              metadata: List[Dict[str, Any]],
                              search_index: Dict[str, List[int]],
                              plan_index: Dict[int, str],
                              processed_stores: List[RAGVectorStore],
                              output_path: Path):
        """Save comprehensive RAG vector store"""
        
        # Save embeddings
        np.save(output_path / "embeddings.npy", embedding_matrix)
        
        # Save metadata
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # Save search index
        with open(output_path / "search_index.json", 'w', encoding='utf-8') as f:
            json.dump(search_index, f, indent=2, ensure_ascii=False)
        
        # Save plan index
        with open(output_path / "plan_index.json", 'w', encoding='utf-8') as f:
            json.dump(plan_index, f, indent=2, ensure_ascii=False)
        
        # Save individual plan stores
        plans_dir = output_path / "individual_plans"
        plans_dir.mkdir(exist_ok=True)
        
        for store in processed_stores:
            plan_data = {
                'plan_id': store.plan_id,
                'annotations_count': len(store.annotations),
                'room_graph': store.room_graph,
                'layout_features': store.layout_features,
                'language_distribution': store.language_distribution,
                'dominant_language': store.dominant_language,
                'room_type_mapping': store.room_type_mapping
            }
            
            with open(plans_dir / f"{store.plan_id}.json", 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)
    
    def _generate_rag_statistics(self, 
                               processed_stores: List[RAGVectorStore],
                               all_metadata: List[Dict[str, Any]],
                               failed_files: List[str]) -> Dict[str, Any]:
        """Generate comprehensive RAG statistics"""
        
        # Aggregate statistics
        total_annotations = len(all_metadata)
        room_type_dist = Counter()
        function_dist = Counter()
        semantic_category_dist = Counter()
        layout_type_dist = Counter()
        
        for metadata in all_metadata:
            if metadata.get('room_type'):
                room_type_dist[metadata['room_type']] += 1
            if metadata.get('function'):
                function_dist[metadata['function']] += 1
            if metadata.get('semantic_category'):
                semantic_category_dist[metadata['semantic_category']] += 1
            if metadata.get('layout_type'):
                layout_type_dist[metadata['layout_type']] += 1
        
        # Language statistics
        language_dist = Counter()
        for store in processed_stores:
            for lang, count in store.language_distribution.items():
                language_dist[lang] += count
        
        # Connectivity analysis
        connectivity_scores = [store.layout_features.get('connectivity_score', 0) for store in processed_stores]
        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0
        
        stats = {
            'processing_timestamp': str(np.datetime64('now')),
            'total_floor_plans': len(processed_stores),
            'total_annotations': total_annotations,
            'total_embeddings': total_annotations,
            'failed_files_count': len(failed_files),
            'embedding_dimension': 384,  # all-MiniLM-L6-v2 dimension
            
            # Distribution statistics
            'room_type_distribution': dict(room_type_dist.most_common(20)),
            'function_distribution': dict(function_dist.most_common(15)),
            'semantic_category_distribution': dict(semantic_category_dist),
            'layout_type_distribution': dict(layout_type_dist),
            'language_distribution': dict(language_dist),
            
            # Quality metrics
            'average_annotations_per_plan': total_annotations / len(processed_stores) if processed_stores else 0,
            'average_connectivity_score': avg_connectivity,
            'multilingual_plans': sum(1 for store in processed_stores if len(store.language_distribution) > 1),
            
            # Processing statistics
            'processing_stats': dict(self.processing_stats),
            'finnish_detection_accuracy': self.processing_stats['language_distribution']['fi'] / max(self.processing_stats['total_annotations'], 1),
            
            # RAG-specific metrics
            'average_keywords_per_annotation': np.mean([len(m.get('keywords', [])) for m in all_metadata]),
            'average_semantic_tags_per_annotation': np.mean([len(m.get('semantic_tags', [])) for m in all_metadata]),
            'room_label_percentage': sum(1 for m in all_metadata if m.get('is_room_label', False)) / max(total_annotations, 1) * 100
        }
        
        return stats


def main():
    """Main execution with RAG-enhanced processing"""
    
    print("🌍 RAG-Enhanced CubiCasa5K Annotation Processor - FIXED VERSION")
    print("=" * 65)
    
    # Initialize processor
    processor = RAGEnhancedAnnotationProcessor(
        target_language='en',
        embedding_model='all-MiniLM-L6-v2',
        cache_translations=True
    )
    
    # Example usage
    print("\n📁 Processing sample data...")
    
    # Create sample SVG with Finnish annotations
    sample_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
        <text x="100" y="50">MH</text>
        <text x="200" y="50">14'10" x 9'4"</text>
        <text x="400" y="100">OH</text>
        <text x="500" y="100">16'3" x 12'11"</text>
        <text x="100" y="200">KH</text>
        <text x="200" y="200">5'1" x 8'7"</text>
        <text x="400" y="300">VH</text>
        <text x="600" y="400">ULKOTILA</text>
        <text x="700" y="500">sauna</text>
    </svg>'''
    
    # Save and process sample
    sample_dir = Path("sample_rag")
    sample_dir.mkdir(exist_ok=True)
    
    sample_svg = sample_dir / "sample_plan.svg"
    with open(sample_svg, 'w', encoding='utf-8') as f:
        f.write(sample_svg_content)
    
    # Process sample
    result = processor.process_svg_for_rag(sample_svg)
    
    if result:
        print(f"\n✅ Sample RAG processing results:")
        print(f"   Plan ID: {result.plan_id}")
        print(f"   Total annotations: {len(result.annotations)}")
        print(f"   Room labels: {sum(1 for ann in result.annotations if ann.is_room_label)}")
        print(f"   Dimensions: {sum(1 for ann in result.annotations if ann.is_dimension)}")
        print(f"   Layout type: {result.layout_features.get('layout_type', 'unknown')}")
        print(f"   Connectivity score: {result.layout_features.get('connectivity_score', 0):.2f}")
        
        print(f"\n📝 Enhanced annotations sample:")
        for i, ann in enumerate(result.annotations[:5]):  # Show first 5
            print(f"   {i+1}. '{ann.original_text}' -> {ann.room_type}")
            print(f"      RAG Context: {ann.rag_context[:100]}...")
            print(f"      Keywords: {ann.search_keywords[:5]}")
            print()
    
    # Process full dataset
    dataset_path = input("\nEnter CubiCasa5K dataset path (or press Enter to skip): ").strip()
    
    if dataset_path and Path(dataset_path).exists():
        max_files = input("Max files to process (default: 100): ").strip()
        max_files = int(max_files) if max_files.isdigit() else 100
        
        print(f"\n🔄 Creating RAG vector store from {max_files} files...")
        
        stats = processor.create_comprehensive_rag_store(
            dataset_path=dataset_path,
            output_path="comprehensive_rag_store",
            max_files=max_files
        )
        
        if 'error' not in stats:
            print(f"\n📊 RAG Vector Store Statistics:")
            print(f"   Total embeddings: {stats['total_embeddings']}")
            print(f"   Average annotations per plan: {stats['average_annotations_per_plan']:.1f}")
            print(f"   Finnish detection accuracy: {stats['finnish_detection_accuracy']:.1%}")
            print(f"   Room label percentage: {stats['room_label_percentage']:.1f}%")
            print(f"   Top room types: {list(stats['room_type_distribution'].keys())[:5]}")
            print(f"   Semantic categories: {list(stats['semantic_category_distribution'].keys())}")
        else:
            print(f"❌ Error: {stats['error']}")
    else:
        print("Dataset path not provided. Sample processing complete.")


if __name__ == "__main__":
    main()