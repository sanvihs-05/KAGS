"""
Enhanced RAG-Powered Multi-Modal Annotation Processor for CubiCasa5K Dataset
Includes visual features, domain-specific embeddings, and architectural fine-tuning
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
import pickle
from PIL import Image, ImageDraw
import io
import base64
import torch

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
    from sentence_transformers.util import cos_sim
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

# Visual processing
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    print("CLIP not available. Visual features will be disabled.")
    CLIP_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Advanced image processing disabled.")
    OPENCV_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn"])
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
# Add GPU availability check
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
else:
    print("⚠️  CUDA not available, using CPU")

logger = logging.getLogger(__name__)

@dataclass
class VisualFeatures:
    """Visual features extracted from floor plan images"""
    image_embedding: Optional[np.ndarray] = None  # CLIP image embedding
    spatial_layout_vector: Optional[np.ndarray] = None  # Geometric layout features
    room_shape_descriptors: Optional[List[float]] = None  # Shape analysis
    color_histogram: Optional[np.ndarray] = None  # Color distribution
    edge_density: Optional[float] = None  # Edge complexity
    symmetry_score: Optional[float] = None  # Layout symmetry
    circulation_flow: Optional[np.ndarray] = None  # Flow patterns
    visual_complexity: Optional[float] = None  # Overall complexity score
    room_boundary_features: Optional[List[Dict]] = None  # Individual room boundaries

@dataclass
class DomainSpecificFeatures:
    """Domain-specific architectural features"""
    architectural_style: Optional[str] = None  # Modern, traditional, etc.
    building_type: Optional[str] = None  # Apartment, house, etc.
    accessibility_score: Optional[float] = None  # Accessibility rating
    energy_efficiency_indicators: Optional[Dict[str, float]] = None
    structural_elements: Optional[List[str]] = None  # Walls, columns, etc.
    circulation_efficiency: Optional[float] = None  # Movement efficiency
    privacy_gradient: Optional[List[float]] = None  # Privacy zones
    natural_light_analysis: Optional[Dict[str, float]] = None  # Light access
    ventilation_paths: Optional[List[List[Tuple[float, float]]]] = None

@dataclass
class EnhancedSpatialInfo:
    """Enhanced spatial information with visual context"""
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None
    area: Optional[float] = None
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    relative_position: Optional[str] = None
    zone: Optional[str] = None
    # Enhanced features
    visual_prominence: Optional[float] = None  # Visual importance in layout
    accessibility_path_length: Optional[float] = None  # Path to entrance
    natural_light_score: Optional[float] = None  # Light access rating
    privacy_level: Optional[float] = None  # Privacy score 0-1
    circulation_node: Optional[bool] = None  # Is this a circulation hub?

@dataclass
class MultiModalAnnotation:
    """Multi-modal annotation with text, spatial, and visual features"""
    # Text features (from original)
    original_text: str
    detected_language: str
    confidence: float
    normalized_text: str
    translated_text: str
    
    # Enhanced architectural classification
    room_type: Optional[str] = None
    room_function: Optional[str] = None
    semantic_category: Optional[str] = None
    architectural_context: Optional[Dict[str, Any]] = None
    domain_features: Optional[DomainSpecificFeatures] = None
    
    # Enhanced spatial information
    spatial_info: Optional[EnhancedSpatialInfo] = None
    adjacent_spaces: Optional[List[str]] = None
    accessibility_paths: Optional[List[str]] = None
    
    # Visual features
    visual_features: Optional[VisualFeatures] = None
    local_image_patch: Optional[np.ndarray] = None  # Local image around annotation
    
    # Content classification (enhanced)
    is_dimension: bool = False
    is_abbreviation: bool = False
    is_room_label: bool = False
    is_furniture: bool = False
    is_fixture: bool = False
    is_structural_element: bool = False
    is_circulation_node: bool = False
    
    # Multi-modal embeddings
    text_embedding: Optional[np.ndarray] = None
    spatial_embedding: Optional[np.ndarray] = None
    visual_embedding: Optional[np.ndarray] = None
    architectural_embedding: Optional[np.ndarray] = None  # Domain-specific
    composite_embedding: Optional[np.ndarray] = None  # Final multi-modal embedding
    
    # Enhanced RAG metadata
    rag_context: Optional[str] = None
    search_keywords: Optional[List[str]] = None
    semantic_tags: Optional[List[str]] = None
    architectural_tags: Optional[List[str]] = None
    visual_descriptors: Optional[List[str]] = None
@dataclass
class ArchitecturalContext:
    """Architectural context information"""
    room_function: str
    room_privacy_level: str
    size_category: str
    plumbing_required: bool = False
    natural_light_access: bool = True
    accessibility_importance: str = 'normal'

class VisualFeatureExtractor:
    """Extract visual features from floor plan images"""
    
    def __init__(self):
        # Add device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
        self.clip_model = None
        self.clip_preprocess = None
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
                # Move CLIP model to GPU
                self.clip_model = self.clip_model.to(self.device)
                print("✅ CLIP model loaded for visual features")
            except Exception as e:
                print(f"❌ Failed to load CLIP: {e}")

                
    
    def svg_to_image(self, svg_path: Path, size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """Convert SVG to image array for processing"""
        try:
            if not OPENCV_AVAILABLE:
                # Fallback: create blank image with text positions
                img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
                return img
            
            # Use cairosvg if available, otherwise opencv
            try:
                import cairosvg
                png_data = cairosvg.svg2png(url=str(svg_path), 
                                          output_width=size[0], 
                                          output_height=size[1])
                img = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                # Fallback: parse SVG manually and create approximate image
                return self._create_approximate_image_from_svg(svg_path, size)
                
        except Exception as e:
            logger.warning(f"Failed to convert SVG to image: {e}")
            return None
    
    def _create_approximate_image_from_svg(self, svg_path: Path, size: Tuple[int, int]) -> np.ndarray:
        """Create approximate image representation from SVG elements"""
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Draw rectangles for rooms (approximate)
            for rect in root.findall('.//{http://www.w3.org/2000/svg}rect'):
                x = float(rect.get('x', 0))
                y = float(rect.get('y', 0))
                width = float(rect.get('width', 50))
                height = float(rect.get('height', 50))
                
                # Scale to image size
                x_scaled = x * size[0] / 1000
                y_scaled = y * size[1] / 1000
                w_scaled = width * size[0] / 1000
                h_scaled = height * size[1] / 1000
                
                draw.rectangle([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled], 
                             outline='black', width=2)
            
            # Mark text positions
            for text in root.findall('.//{http://www.w3.org/2000/svg}text'):
                x = float(text.get('x', 0))
                y = float(text.get('y', 0))
                
                x_scaled = x * size[0] / 1000
                y_scaled = y * size[1] / 1000
                
                draw.circle([x_scaled, y_scaled], 3, fill='red')
                
        except Exception as e:
            logger.warning(f"Error creating approximate image: {e}")
        
        return np.array(img)
    
    def extract_clip_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract CLIP image embeddings"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return None
            
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Preprocess for CLIP
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.warning(f"Failed to extract CLIP features: {e}")
            return None
    
    def extract_spatial_layout_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial layout features"""
        try:
            if OPENCV_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Edge detection
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Contour analysis
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_contours = len(contours)
                avg_contour_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
                
                # Symmetry analysis
                h, w = gray.shape
                left_half = gray[:, :w//2]
                right_half = cv2.flip(gray[:, w//2:], 1)
                min_width = min(left_half.shape[1], right_half.shape[1])
                symmetry_score = np.corrcoef(left_half[:, :min_width].flatten(), 
                                           right_half[:, :min_width].flatten())[0, 1]
                symmetry_score = symmetry_score if not np.isnan(symmetry_score) else 0
                
                # Spatial distribution
                moments = cv2.moments(edges)
                if moments['m00'] != 0:
                    centroid_x = moments['m10'] / moments['m00'] / w
                    centroid_y = moments['m01'] / moments['m00'] / h
                else:
                    centroid_x = centroid_y = 0.5
                
                return np.array([
                    edge_density,
                    num_contours / 100,  # Normalized
                    avg_contour_area / (w * h),  # Normalized
                    symmetry_score,
                    centroid_x,
                    centroid_y,
                    np.std(gray.flatten()) / 255,  # Contrast measure
                ])
            else:
                # Fallback: simple spatial features
                return np.array([0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3])
                
        except Exception as e:
            logger.warning(f"Failed to extract spatial features: {e}")
            return np.zeros(7)
    
    def extract_local_features(self, image: np.ndarray, x: float, y: float, 
                             patch_size: int = 64) -> Dict[str, Any]:
        """Extract local features around annotation position"""
        try:
            h, w = image.shape[:2]
            
            # Extract local patch
            x_int, y_int = int(x), int(y)
            x_start = max(0, x_int - patch_size // 2)
            y_start = max(0, y_int - patch_size // 2)
            x_end = min(w, x_start + patch_size)
            y_end = min(h, y_start + patch_size)
            
            local_patch = image[y_start:y_end, x_start:x_end]
            
            # Calculate local features
            local_features = {
                'patch': local_patch,
                'local_contrast': np.std(local_patch) if local_patch.size > 0 else 0,
                'local_brightness': np.mean(local_patch) if local_patch.size > 0 else 0,
                'edge_proximity': self._calculate_edge_proximity(image, x_int, y_int),
                'color_variance': np.var(local_patch, axis=(0, 1)) if local_patch.size > 0 else np.zeros(3)
            }
            
            return local_features
            
        except Exception as e:
            logger.warning(f"Failed to extract local features: {e}")
            return {'patch': np.zeros((patch_size, patch_size, 3))}
    
    def _calculate_edge_proximity(self, image: np.ndarray, x: int, y: int, radius: int = 20) -> float:
        """Calculate proximity to edges (walls, boundaries)"""
        try:
            if not OPENCV_AVAILABLE:
                return 0.5
                
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            h, w = edges.shape
            y_start = max(0, y - radius)
            y_end = min(h, y + radius)
            x_start = max(0, x - radius)
            x_end = min(w, x + radius)
            
            local_edges = edges[y_start:y_end, x_start:x_end]
            edge_density = np.sum(local_edges > 0) / max(local_edges.size, 1)
            
            return edge_density
            
        except Exception:
            return 0.5

class DomainSpecificEmbedder:
    """Domain-specific embedder for architectural content"""
    
    def __init__(self, base_model: str = 'all-MiniLM-L6-v2'):
        # Add device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        # Initialize with device
        self.base_model = SentenceTransformer(base_model, device=self.device)
    
        # Architectural vocabulary for domain adaptation
        self.architectural_vocabulary = self._build_architectural_vocabulary()
        self.room_function_hierarchy = self._build_room_hierarchy()
        self.spatial_relationship_patterns = self._build_spatial_patterns()
    
        # Cache for domain embeddings
        self.domain_embedding_cache = {}

    
    def _build_architectural_vocabulary(self) -> Dict[str, List[str]]:
        """Build comprehensive architectural vocabulary"""
        return {
            'room_types': [
                'living room', 'bedroom', 'kitchen', 'bathroom', 'dining room',
                'entrance hall', 'hallway', 'corridor', 'closet', 'storage',
                'utility room', 'laundry room', 'pantry', 'office', 'study',
                'balcony', 'terrace', 'patio', 'garage', 'basement',
                'attic', 'loft', 'family room', 'guest room', 'master bedroom',
                'powder room', 'mudroom', 'foyer', 'vestibule'
            ],
            'functions': [
                'sleeping', 'resting', 'cooking', 'dining', 'bathing',
                'cleaning', 'storage', 'circulation', 'entry', 'entertainment',
                'work', 'study', 'relaxation', 'socializing', 'privacy',
                'hygiene', 'food preparation', 'gathering', 'transition'
            ],
            'features': [
                'plumbing', 'electricity', 'ventilation', 'natural light',
                'artificial lighting', 'heating', 'cooling', 'insulation',
                'soundproofing', 'accessibility', 'emergency exit',
                'fire safety', 'security', 'privacy', 'openness'
            ],
            'relationships': [
                'adjacent to', 'connected to', 'accessible from',
                'overlooks', 'separated from', 'integrated with',
                'flows into', 'leads to', 'opens to', 'enclosed by'
            ],
            'qualities': [
                'spacious', 'compact', 'bright', 'dark', 'open',
                'enclosed', 'public', 'private', 'quiet', 'active',
                'formal', 'informal', 'functional', 'decorative'
            ]
        }
    
    def _build_room_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchical relationships between room types"""
        return {
            'social_spaces': ['living_room', 'family_room', 'dining_room', 'entertainment_room'],
            'private_spaces': ['bedroom', 'master_bedroom', 'guest_room', 'study', 'office'],
            'service_spaces': ['kitchen', 'bathroom', 'utility_room', 'laundry_room', 'pantry'],
            'circulation_spaces': ['hallway', 'corridor', 'entrance_hall', 'foyer', 'stairs'],
            'storage_spaces': ['closet', 'storage_room', 'pantry', 'basement', 'attic'],
            'transition_spaces': ['mudroom', 'vestibule', 'airlock', 'breezeway'],
            'outdoor_spaces': ['balcony', 'terrace', 'patio', 'deck', 'garden']
        }
    
    def _build_spatial_patterns(self) -> Dict[str, np.ndarray]:
        """Build spatial relationship patterns"""
        patterns = {}
        
        # Common adjacency patterns (learned from architectural principles)
        patterns['kitchen_adjacencies'] = np.array([0.8, 0.6, 0.3, 0.9, 0.4])  # dining, living, bedroom, utility, outdoor
        patterns['bedroom_adjacencies'] = np.array([0.2, 0.9, 0.1, 0.7, 0.3])  # living, bathroom, kitchen, hallway, closet
        patterns['bathroom_adjacencies'] = np.array([0.1, 0.9, 0.2, 0.8, 0.1])  # living, bedroom, kitchen, hallway, utility
        
        return patterns
    
    def create_domain_specific_embedding(self, text: str, room_type: str, 
                                       function: str, spatial_context: List[str]) -> np.ndarray:
        """Create domain-specific architectural embedding"""
        
        cache_key = f"{text}_{room_type}_{function}_{','.join(sorted(spatial_context))}"
        if cache_key in self.domain_embedding_cache:
            return self.domain_embedding_cache[cache_key]
        
        # Base text embedding
        base_embedding = self.base_model.encode(text)
        
        # Function embedding
        function_text = f"{room_type} for {function}"
        function_embedding = self.base_model.encode(function_text)
        
        # Spatial context embedding
        if spatial_context:
            spatial_text = f"space adjacent to {', '.join(spatial_context)}"
            spatial_embedding = self.base_model.encode(spatial_text)
        else:
            spatial_embedding = np.zeros_like(base_embedding)
        
        # Architectural context embedding
        arch_descriptors = self._generate_architectural_descriptors(room_type, function)
        arch_text = " ".join(arch_descriptors)
        arch_embedding = self.base_model.encode(arch_text) if arch_text else np.zeros_like(base_embedding)
        
        # Hierarchical embedding (room category)
        hierarchy_context = self._get_hierarchical_context(room_type)
        hierarchy_embedding = self.base_model.encode(hierarchy_context) if hierarchy_context else np.zeros_like(base_embedding)
        
        # Weighted combination
        composite_embedding = (
            base_embedding * 0.4 +
            function_embedding * 0.25 +
            spatial_embedding * 0.15 +
            arch_embedding * 0.15 +
            hierarchy_embedding * 0.05
        )
        
        # Normalize
        composite_embedding = composite_embedding / np.linalg.norm(composite_embedding)
        
        # Cache result
        self.domain_embedding_cache[cache_key] = composite_embedding
        
        return composite_embedding
    
    def _generate_architectural_descriptors(self, room_type: str, function: str) -> List[str]:
        """Generate architectural descriptors for a room"""
        descriptors = []
        
        # Function-based descriptors
        if 'sleep' in function or 'rest' in function:
            descriptors.extend(['quiet', 'private', 'comfortable', 'personal'])
        elif 'cook' in function or 'food' in function:
            descriptors.extend(['ventilated', 'functional', 'equipped', 'accessible'])
        elif 'hygiene' in function or 'bath' in function:
            descriptors.extend(['private', 'ventilated', 'water_access', 'tiled'])
        elif 'social' in function or 'entertain' in function:
            descriptors.extend(['open', 'spacious', 'central', 'welcoming'])
        
        # Room-type specific descriptors
        if room_type in ['bedroom', 'master_bedroom']:
            descriptors.extend(['carpeted', 'climate_controlled', 'sound_isolated'])
        elif room_type == 'kitchen':
            descriptors.extend(['hard_surfaces', 'well_lit', 'multi_functional'])
        elif room_type == 'bathroom':
            descriptors.extend(['moisture_resistant', 'easy_to_clean', 'well_ventilated'])
        
        return descriptors
    
    def _get_hierarchical_context(self, room_type: str) -> str:
        """Get hierarchical context for room type"""
        for category, rooms in self.room_function_hierarchy.items():
            if room_type in rooms:
                return f"{category.replace('_', ' ')} type room"
        return "general purpose room"

class EnhancedRAGProcessor:
    """Enhanced RAG processor with multi-modal capabilities"""
    
    def __init__(self, 
                 target_language: str = 'en',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 enable_visual_features: bool = True,
                 cache_translations: bool = True):
        self.target_language = target_language
        self.enable_visual_features = enable_visual_features
        self.cache_translations = cache_translations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"RAG Processor using device: {self.device}")
        self.base_embedder = SentenceTransformer(embedding_model, device=self.device)
        self.domain_embedder = DomainSpecificEmbedder(embedding_model)
        
        if enable_visual_features:
            self.visual_extractor = VisualFeatureExtractor()
        else:
            self.visual_extractor = None
        
        # Initialize translator
        self.translator = GoogleTranslator(source='auto', target=self.target_language)
        
        # Caches
        self.translation_cache = {}
        self.embedding_cache = {}
        
        # Enhanced knowledge bases (inherit from original)
        self.finnish_room_types = self._initialize_finnish_room_types()
        self.architectural_contexts = self._initialize_architectural_contexts()
        self.spatial_relationships = self._initialize_spatial_relationships()
        self.dimension_patterns = self._initialize_dimension_patterns()
        
        # Statistics
        self.processing_stats = {
            'total_annotations': 0,
            'visual_features_extracted': 0,
            'domain_embeddings_created': 0,
            'multi_modal_embeddings': 0,
            'language_distribution': Counter(),
            'room_type_distribution': Counter(),
            'failed_visual_extractions': 0
        }
    def _infer_building_type(self, room_type: str, room_info: Dict) -> str:
        """Infer building type from room characteristics"""
        if any(room in room_type.lower() for room in ['apartment', 'flat']):
            return 'apartment'
        elif any(room in room_type.lower() for room in ['house', 'villa']):
            return 'house'
        else:
            return 'residential'

    def _calculate_accessibility_score(self, room_type: str, spatial_info) -> float:
        """Calculate accessibility score"""
        base_score = 0.7
        if room_type in ['bathroom', 'kitchen']:
            base_score = 0.9  # Higher accessibility importance
        return base_score

    def _calculate_energy_indicators(self, room_type: str, room_info: Dict) -> Dict[str, float]:
        """Calculate energy efficiency indicators"""
        return {
            'heating_efficiency': 0.8,
            'natural_lighting': 0.7,
            'ventilation_rating': 0.6
        }

    def _detect_structural_elements(self, room_type: str) -> List[str]:
        """Detect structural elements"""
        elements = ['walls']
        if room_type in ['kitchen', 'bathroom']:
            elements.append('plumbing')
        return elements

    def _calculate_privacy_gradient(self, room_type: str, spatial_info) -> List[float]:
        """Calculate privacy gradient"""
        if room_type in ['bedroom', 'bathroom']:
            return [0.9, 0.8, 0.7]  # High privacy
        elif room_type in ['living_room', 'kitchen']:
            return [0.3, 0.4, 0.5]  # Lower privacy
        else:
            return [0.5, 0.5, 0.5]  # Medium privacy

    def _get_adaptive_proximity_threshold(self, room_type1: str, room_type2: str, base_threshold: float) -> float:
        """Get adaptive proximity threshold"""
        # Kitchen-dining rooms are typically closer
        if ('kitchen' in room_type1 and 'dining' in room_type2) or ('kitchen' in room_type2 and 'dining' in room_type1):
            return base_threshold * 0.8
        # Bedroom-bathroom connections
        elif ('bedroom' in room_type1 and 'bathroom' in room_type2) or ('bedroom' in room_type2 and 'bathroom' in room_type1):
            return base_threshold * 0.9
        return base_threshold

    def _analyze_enhanced_layout_features(self, annotations, room_graph, floor_plan_image) -> Dict[str, Any]:
        """Analyze enhanced layout features"""
        return {
            'total_rooms': len([ann for ann in annotations if ann.is_room_label]),
            'circulation_efficiency': 0.75,
            'natural_light_distribution': 0.8,
            'privacy_zones': ['private', 'semi-private', 'public']
        }

    def _create_multimodal_rag_store(self, plan_id, annotations, adjacency_matrix, room_graph, layout_features, floor_plan_image):
        """Create multi-modal RAG store"""
        # Create embedding matrices
        text_embeddings = []
        composite_embeddings = []

        for ann in annotations:
            if ann.text_embedding is not None:
                text_embeddings.append(ann.text_embedding)
            if ann.composite_embedding is not None:
                composite_embeddings.append(ann.composite_embedding)

        text_embedding_matrix = np.vstack(text_embeddings) if text_embeddings else None
        composite_embedding_matrix = np.vstack(composite_embeddings) if composite_embeddings else None

        return MultiModalRAGStore(
            plan_id=plan_id,
            annotations=annotations,
            adjacency_matrix=adjacency_matrix,
            room_graph=room_graph,
            layout_features=layout_features,
            floor_plan_image=floor_plan_image,
            text_embedding_matrix=text_embedding_matrix,
            composite_embedding_matrix=composite_embedding_matrix,
            language_distribution=Counter([ann.detected_language for ann in annotations])
        )
    def process_multimodal_annotation(self, text: str, x: float, y: float, 
                                    floor_plan_image: Optional[np.ndarray] = None) -> MultiModalAnnotation:
        """Process annotation with multi-modal features"""
        
        # Start with text processing (inherit from original logic)
        detected_lang, confidence = self._detect_language_enhanced(text)
        normalized_text = self._normalize_text(text)
        translated_text = self._translate_finnish_text(normalized_text) if detected_lang in ['fi', 'unknown'] else normalized_text
        
        room_type, room_info = self._normalize_room_type_enhanced(normalized_text, detected_lang)
        
        # Enhanced spatial information
        spatial_info = EnhancedSpatialInfo(
            x=x, y=y,
            center_x=x, center_y=y,
            relative_position=self._determine_relative_position(x, y),
            zone=self._determine_zone(x, y)
        )
        
        # Content classification
        is_dimension = any(pattern.search(text) for pattern in self.dimension_patterns)
        is_room_label = room_type in self.finnish_room_types or room_type in [info.get('english', '') for info in self.finnish_room_types.values()]
        
        # Visual features extraction
        visual_features = None
        local_patch = None
        
        if self.enable_visual_features and floor_plan_image is not None and self.visual_extractor:
            try:
                # Extract CLIP features for entire image
                clip_embedding = self.visual_extractor.extract_clip_features(floor_plan_image)
                
                # Extract spatial layout features
                spatial_layout = self.visual_extractor.extract_spatial_layout_features(floor_plan_image)
                
                # Extract local features around annotation
                local_features = self.visual_extractor.extract_local_features(floor_plan_image, x, y)
                local_patch = local_features.get('patch')
                
                # Create visual features object
                visual_features = VisualFeatures(
                    image_embedding=clip_embedding,
                    spatial_layout_vector=spatial_layout,
                    edge_density=local_features.get('edge_proximity', 0.5),
                    visual_complexity=np.std(spatial_layout) if spatial_layout is not None else 0.5
                )
                
                # Enhanced spatial information with visual context
                spatial_info.visual_prominence = local_features.get('local_contrast', 0.5)
                spatial_info.natural_light_score = self._estimate_natural_light_access(floor_plan_image, x, y)
                
                self.processing_stats['visual_features_extracted'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to extract visual features: {e}")
                self.processing_stats['failed_visual_extractions'] += 1
        
        # Domain-specific features
        domain_features = self._extract_domain_features(room_type, room_info, spatial_info)
        
        # Create annotation
        annotation = MultiModalAnnotation(
            original_text=text,
            detected_language=detected_lang,
            confidence=confidence,
            normalized_text=normalized_text,
            translated_text=translated_text,
            room_type=room_type,
            room_function=room_info.get('function', 'unknown'),
            semantic_category=self._determine_semantic_category(room_info),
            spatial_info=spatial_info,
            visual_features=visual_features,
            local_image_patch=local_patch,
            domain_features=domain_features,
            is_dimension=is_dimension,
            is_room_label=is_room_label,
            is_structural_element=self._is_structural_element(text, room_type),
            is_circulation_node=self._is_circulation_node(room_type)
        )
        
        # Generate enhanced tags and keywords
        annotation.semantic_tags = self._generate_semantic_tags(room_info, is_dimension, is_room_label)
        annotation.architectural_tags = self._generate_architectural_tags(room_type, room_info)
        annotation.visual_descriptors = self._generate_visual_descriptors(visual_features, spatial_info)
        annotation.search_keywords = self._generate_search_keywords(room_info, text, annotation.architectural_tags)
        
        # Create multi-modal embeddings
        self._create_multimodal_embeddings(annotation)
        
        # Create RAG context
        annotation.rag_context = self._create_enhanced_rag_context(annotation)
        
        # Update statistics
        self.processing_stats['total_annotations'] += 1
        self.processing_stats['language_distribution'][detected_lang] += 1
        self.processing_stats['room_type_distribution'][room_type] += 1
        if visual_features:
            self.processing_stats['multi_modal_embeddings'] += 1
        
        return annotation
    
    def _create_multimodal_embeddings(self, annotation: MultiModalAnnotation):
        """Create comprehensive multi-modal embeddings"""
        
        # Text embedding (base)
        text_embedding = self.base_embedder.encode(annotation.translated_text)
        annotation.text_embedding = text_embedding
        
        # Domain-specific architectural embedding
        adjacent_spaces = annotation.adjacent_spaces or []
        architectural_embedding = self.domain_embedder.create_domain_specific_embedding(
            annotation.translated_text,
            annotation.room_type or 'unknown',
            annotation.room_function or 'unknown',
            adjacent_spaces
        )
        annotation.architectural_embedding = architectural_embedding
        
        # Spatial embedding (encode spatial relationships and features)
        spatial_features = self._create_spatial_feature_vector(annotation.spatial_info)
        spatial_embedding = self.base_embedder.encode(f"spatial location {annotation.spatial_info.relative_position} {annotation.spatial_info.zone}")
        
        # Combine with numerical spatial features
        spatial_embedding = np.concatenate([spatial_embedding, spatial_features])
        annotation.spatial_embedding = spatial_embedding
        
        # Visual embedding
        visual_embedding = None
        if annotation.visual_features and annotation.visual_features.image_embedding is not None:
            visual_embedding = annotation.visual_features.image_embedding
            
            # Enhance with spatial layout features
            if annotation.visual_features.spatial_layout_vector is not None:
                visual_embedding = np.concatenate([
                    visual_embedding, 
                    annotation.visual_features.spatial_layout_vector
                ])
        annotation.visual_embedding = visual_embedding
        
        # Create composite multi-modal embedding
        embeddings_to_combine = [text_embedding, architectural_embedding]
        weights = [0.3, 0.4]  # Text, architectural
        
        if spatial_embedding is not None:
            # Pad or truncate spatial embedding to match text embedding size
            if len(spatial_embedding) > len(text_embedding):
                spatial_embedding = spatial_embedding[:len(text_embedding)]
            elif len(spatial_embedding) < len(text_embedding):
                spatial_embedding = np.pad(spatial_embedding, (0, len(text_embedding) - len(spatial_embedding)))
            
            embeddings_to_combine.append(spatial_embedding)
            weights.append(0.2)
        
        if visual_embedding is not None:
            # Resize visual embedding to match text embedding
            if len(visual_embedding) != len(text_embedding):
                from sklearn.decomposition import PCA
                pca = PCA(n_components=len(text_embedding))
                visual_embedding_resized = pca.fit_transform(visual_embedding.reshape(1, -1)).flatten()
            else:
                visual_embedding_resized = visual_embedding
            
            embeddings_to_combine.append(visual_embedding_resized)
            weights.append(0.1)
        
        # Weighted combination
        composite_embedding = np.average(embeddings_to_combine, axis=0, weights=weights)
        composite_embedding = composite_embedding / np.linalg.norm(composite_embedding)
        
        annotation.composite_embedding = composite_embedding
        self.processing_stats['domain_embeddings_created'] += 1
    
    def _create_spatial_feature_vector(self, spatial_info: EnhancedSpatialInfo) -> np.ndarray:
        """Create numerical spatial feature vector"""
        features = [
            spatial_info.x / 1000.0,  # Normalized coordinates
            spatial_info.y / 1000.0,
            spatial_info.visual_prominence or 0.5,
            spatial_info.natural_light_score or 0.5,
            spatial_info.privacy_level or 0.5,
            1.0 if spatial_info.circulation_node else 0.0,
            spatial_info.accessibility_path_length or 0.5
        ]
        return np.array(features)
    
    def _extract_domain_features(self, room_type: str, room_info: Dict, 
                               spatial_info: EnhancedSpatialInfo) -> DomainSpecificFeatures:
        """Extract domain-specific architectural features"""
        
        # Determine building type based on room distribution
        building_type = self._infer_building_type(room_type, room_info)
        
        # Calculate accessibility score
        accessibility_score = self._calculate_accessibility_score(room_type, spatial_info)
        
        # Energy efficiency indicators
        energy_indicators = self._calculate_energy_indicators(room_type, room_info)
        
        # Structural elements detection
        structural_elements = self._detect_structural_elements(room_type)
        
        # Privacy gradient analysis
        privacy_gradient = self._calculate_privacy_gradient(room_type, spatial_info)
        
        return DomainSpecificFeatures(
            building_type=building_type,
            accessibility_score=accessibility_score,
            energy_efficiency_indicators=energy_indicators,
            structural_elements=structural_elements,
            privacy_gradient=privacy_gradient
        )
    
    def _create_enhanced_rag_context(self, annotation: MultiModalAnnotation) -> str:
        """Create enhanced contextual description for RAG retrieval"""
        
        context_parts = []
        
        # Basic information
        context_parts.append(f"Room: {annotation.translated_text}")
        if annotation.room_type and annotation.room_type != annotation.translated_text:
            context_parts.append(f"Type: {annotation.room_type}")
        
        # Architectural context
        if annotation.architectural_context:
            ctx = annotation.architectural_context
            context_parts.append(f"Function: {ctx.room_function}")
            context_parts.append(f"Privacy: {ctx.room_privacy_level}")
            context_parts.append(f"Size: {ctx.size_category}")
            
            if ctx.plumbing_required:
                context_parts.append("Plumbing required")
            if ctx.natural_light_access:
                context_parts.append("Natural light access")
        
        # Enhanced spatial context
        if annotation.spatial_info:
            spatial = annotation.spatial_info
            context_parts.append(f"Location: {spatial.relative_position} {spatial.zone}")
            
            if spatial.visual_prominence and spatial.visual_prominence > 0.7:
                context_parts.append("Visually prominent")
            if spatial.natural_light_score and spatial.natural_light_score > 0.7:
                context_parts.append("Good natural lighting")
            if spatial.privacy_level and spatial.privacy_level > 0.7:
                context_parts.append("High privacy")
            if spatial.circulation_node:
                context_parts.append("Circulation hub")
        
        # Visual context
        if annotation.visual_features:
            vf = annotation.visual_features
            if vf.visual_complexity and vf.visual_complexity > 0.6:
                context_parts.append("Complex layout area")
            if vf.edge_density and vf.edge_density > 0.5:
                context_parts.append("Near structural elements")
        
        # Domain features
        if annotation.domain_features:
            df = annotation.domain_features
            if df.building_type:
                context_parts.append(f"Building type: {df.building_type}")
            if df.accessibility_score and df.accessibility_score > 0.8:
                context_parts.append("Highly accessible")
        
        # Adjacent spaces
        if annotation.adjacent_spaces:
            context_parts.append(f"Adjacent to: {', '.join(annotation.adjacent_spaces)}")
        
        # Architectural and visual tags
        if annotation.architectural_tags:
            context_parts.append(f"Architectural features: {', '.join(annotation.architectural_tags[:3])}")
        
        if annotation.visual_descriptors:
            context_parts.append(f"Visual characteristics: {', '.join(annotation.visual_descriptors[:2])}")
        
        return " | ".join(context_parts)
    
    def process_svg_with_multimodal_features(self, svg_file: Path) -> Optional['MultiModalRAGStore']:
        """Process SVG file with enhanced multi-modal features"""
        
        # Convert SVG to image for visual processing
        floor_plan_image = None
        if self.enable_visual_features and self.visual_extractor:
            floor_plan_image = self.visual_extractor.svg_to_image(svg_file)
        
        # Extract text elements (inherit from original)
        text_elements = self._extract_svg_elements_enhanced(svg_file)
        
        if not text_elements:
            logger.debug(f"No text elements found in {svg_file}")
            return None
        
        # Process each annotation with multi-modal features
        annotations = []
        for text, x, y in text_elements:
            try:
                annotation = self.process_multimodal_annotation(text, x, y, floor_plan_image)
                annotations.append(annotation)
            except Exception as e:
                logger.warning(f"Failed to process annotation '{text}': {e}")
                continue
        
        if not annotations:
            return None
        
        # Enhanced spatial relationship analysis
        adjacency_matrix, room_graph = self._calculate_enhanced_spatial_relationships(
            annotations, floor_plan_image
        )
        
        # Update adjacency information in annotations
        for i, annotation in enumerate(annotations):
            adjacent_indices = np.where(adjacency_matrix[i] == 1)[0]
            adjacent_rooms = [
                annotations[j].room_type for j in adjacent_indices 
                if annotations[j].is_room_label and annotations[j].room_type
            ]
            annotation.adjacent_spaces = adjacent_rooms if adjacent_rooms else []
        
        # Recalculate embeddings with updated adjacency
        for annotation in annotations:
            self._create_multimodal_embeddings(annotation)
        
        # Enhanced layout analysis
        layout_features = self._analyze_enhanced_layout_features(
            annotations, room_graph, floor_plan_image
        )
        
        # Create multi-modal RAG store
        return self._create_multimodal_rag_store(
            svg_file.stem, annotations, adjacency_matrix, room_graph, 
            layout_features, floor_plan_image
        )
    
    def _calculate_enhanced_spatial_relationships(self, annotations: List[MultiModalAnnotation], 
                                                floor_plan_image: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        """Enhanced spatial relationship calculation with visual cues"""
        
        n = len(annotations)
        adjacency_matrix = np.zeros((n, n))
        room_graph = defaultdict(list)
        
        # Base proximity threshold
        base_threshold = 100.0
        
        for i, ann1 in enumerate(annotations):
            if not ann1.spatial_info or not ann1.is_room_label:
                continue
                
            for j, ann2 in enumerate(annotations):
                if i == j or not ann2.spatial_info or not ann2.is_room_label:
                    continue
                
                # Calculate Euclidean distance
                dx = ann1.spatial_info.x - ann2.spatial_info.x
                dy = ann1.spatial_info.y - ann2.spatial_info.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Adaptive threshold based on room types
                threshold = self._get_adaptive_proximity_threshold(
                    ann1.room_type, ann2.room_type, base_threshold
                )
                
                # Visual proximity enhancement
                if floor_plan_image is not None and self.enable_visual_features:
                    visual_connection_score = self._calculate_visual_connection(
                        floor_plan_image, ann1.spatial_info, ann2.spatial_info
                    )
                    # Adjust threshold based on visual connection
                    threshold *= (1 + visual_connection_score)
                
                if distance < threshold:
                    adjacency_matrix[i, j] = 1
                    room_graph[ann1.room_type].append(ann2.room_type)
        
        return adjacency_matrix, dict(room_graph)
    
    def _calculate_visual_connection(self, image: np.ndarray, 
                                   spatial1: EnhancedSpatialInfo, 
                                   spatial2: EnhancedSpatialInfo) -> float:
        """Calculate visual connection between two spatial points"""
        try:
            if not OPENCV_AVAILABLE:
                return 0.0
                
            # Create line between points
            x1, y1 = int(spatial1.x), int(spatial1.y)
            x2, y2 = int(spatial2.x), int(spatial2.y)
            
            # Sample points along the line
            num_samples = 20
            line_points = []
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    line_points.append((x, y))
            
            # Analyze visual obstacles (walls, barriers)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge intersections along the line
            edge_intersections = 0
            for x, y in line_points:
                if edges[y, x] > 0:
                    edge_intersections += 1
            
            # Calculate connection score (fewer intersections = better connection)
            connection_score = 1.0 - (edge_intersections / max(len(line_points), 1))
            return max(0.0, connection_score)
            
        except Exception:
            return 0.0
    
    # Additional helper methods (inheriting and extending from original)
    
    def _detect_language_enhanced(self, text: str) -> Tuple[str, float]:
        """Enhanced language detection (inherited from original with minor updates)"""
        if not text or len(text.strip()) < 2:
            return 'unknown', 0.0
        
        text_lower = text.lower().strip()
        
        # Finnish architectural abbreviations
        finnish_indicators = ['mh', 'oh', 'kh', 'vh', 'wc', 'sauna', 'eteinen', 'varasto', 
                            'keittiö', 'parveke', 'ulkotila', 'tekn', 'khh']
        
        if any(abbr in text_lower for abbr in finnish_indicators):
            return 'fi', 0.95
        
        # Check for dimension patterns
        for pattern in self.dimension_patterns:
            if pattern.search(text):
                return 'neutral', 0.9
        
        # Use langdetect
        try:
            cleaned_text = self._clean_text_for_detection(text)
            if len(cleaned_text) < 2:
                return 'unknown', 0.0
            
            lang_probs = detect_langs(cleaned_text)
            if lang_probs:
                best_lang = lang_probs[0]
                if best_lang.lang == 'fi' or any(term in text_lower for term in ['huone', 'tila', 'keittiö']):
                    return 'fi', min(0.95, best_lang.prob + 0.2)
                return best_lang.lang, best_lang.prob
            else:
                return 'unknown', 0.0
                
        except LangDetectException:
            return 'unknown', 0.0
    
    def _estimate_natural_light_access(self, image: np.ndarray, x: float, y: float) -> float:
        """Estimate natural light access based on position and visual cues"""
        try:
            h, w = image.shape[:2]
            
            # Distance from edges (windows typically on perimeter)
            edge_distances = [x, y, w - x, h - y]
            min_edge_distance = min(edge_distances) / min(w, h)
            
            # Closer to edges = higher light access potential
            edge_score = 1.0 - min_edge_distance
            
            # Local brightness analysis
            patch_size = 50
            x_int, y_int = int(x), int(y)
            x_start = max(0, x_int - patch_size // 2)
            y_start = max(0, y_int - patch_size // 2)
            x_end = min(w, x_start + patch_size)
            y_end = min(h, y_start + patch_size)
            
            local_patch = image[y_start:y_end, x_start:x_end]
            local_brightness = np.mean(local_patch) / 255.0 if local_patch.size > 0 else 0.5
            
            # Combine scores
            light_score = (edge_score * 0.6 + local_brightness * 0.4)
            return np.clip(light_score, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _is_structural_element(self, text: str, room_type: str) -> bool:
        """Detect if annotation represents structural element"""
        structural_keywords = ['wall', 'column', 'beam', 'support', 'pillar', 'structure']
        return any(keyword in text.lower() for keyword in structural_keywords)
    
    def _is_circulation_node(self, room_type: str) -> bool:
        """Detect if room is a circulation node"""
        circulation_rooms = ['hallway', 'corridor', 'entrance_hall', 'foyer', 'stairs']
        return room_type in circulation_rooms
    
    def _generate_architectural_tags(self, room_type: str, room_info: Dict) -> List[str]:
        """Generate architectural-specific tags"""
        tags = []
        
        # Add functional tags
        function = room_info.get('function', '')
        if 'sleep' in function:
            tags.extend(['bedroom_type', 'private_zone', 'quiet_space'])
        elif 'cook' in function:
            tags.extend(['kitchen_type', 'service_zone', 'utility_space'])
        elif 'hygiene' in function:
            tags.extend(['bathroom_type', 'wet_area', 'private_facilities'])
        elif 'social' in function:
            tags.extend(['social_zone', 'gathering_space', 'entertainment_area'])
        
        # Add size-based tags
        size = room_info.get('typical_size', 'medium')
        tags.append(f'{size}_space')
        
        # Add accessibility tags
        if room_info.get('accessibility_importance') == 'critical':
            tags.extend(['accessible_required', 'ada_compliant'])
        
        return tags
    
    def _generate_visual_descriptors(self, visual_features: Optional[VisualFeatures], 
                                   spatial_info: EnhancedSpatialInfo) -> List[str]:
        """Generate visual descriptors from visual features"""
        descriptors = []
        
        if visual_features:
            if visual_features.edge_density and visual_features.edge_density > 0.7:
                descriptors.append('high_detail_area')
            if visual_features.visual_complexity and visual_features.visual_complexity > 0.6:
                descriptors.append('complex_layout')
            if visual_features.symmetry_score and visual_features.symmetry_score > 0.8:
                descriptors.append('symmetric_design')
        
        if spatial_info.visual_prominence and spatial_info.visual_prominence > 0.7:
            descriptors.append('visually_prominent')
        
        if spatial_info.natural_light_score and spatial_info.natural_light_score > 0.8:
            descriptors.append('well_lit')
        
        return descriptors
    
    # Include necessary methods from original implementation
    def _initialize_finnish_room_types(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive Finnish room type knowledge base (inherited from original)"""
        return {
            'oh': {
                'full_name': 'olohuone', 'english': 'living_room', 'function': 'social_relaxation',
                'privacy': 'semi-private', 'typical_size': 'large',
                'adjacencies': ['keittiö', 'ruokailutila', 'eteinen'],
                'keywords': ['living', 'social', 'relaxation', 'family', 'entertainment'],
                'description': 'Main living and social space for family activities and relaxation'
            },
            'mh': {
                'full_name': 'makuuhuone', 'english': 'bedroom', 'function': 'rest_sleep',
                'privacy': 'private', 'typical_size': 'medium',
                'adjacencies': ['kylpyhuone', 'käytävä', 'vaatehuone'],
                'keywords': ['bedroom', 'sleep', 'rest', 'private', 'personal'],
                'description': 'Private sleeping and resting space with storage for personal belongings'
            },
            'kh': {
                'full_name': 'kylpyhuone', 'english': 'bathroom', 'function': 'hygiene_bathing',
                'privacy': 'private', 'typical_size': 'small',
                'adjacencies': ['makuuhuone', 'käytävä', 'sauna'],
                'keywords': ['bathroom', 'hygiene', 'bathing', 'washing', 'private'],
                'description': 'Private space for bathing, personal hygiene, and grooming activities'
            },
            'vh': {
                'full_name': 'vesihuone', 'english': 'utility_room', 'function': 'utility_cleaning',
                'privacy': 'service', 'typical_size': 'small',
                'adjacencies': ['keittiö', 'eteinen', 'tekninen_tila'],
                'keywords': ['utility', 'laundry', 'cleaning', 'water', 'service'],
                'description': 'Service space for laundry, cleaning equipment, and utility functions'
            },
            'keittiö': {
                'full_name': 'keittiö', 'english': 'kitchen', 'function': 'food_preparation',
                'privacy': 'semi-private', 'typical_size': 'medium',
                'adjacencies': ['ruokailutila', 'olohuone', 'eteinen'],
                'keywords': ['kitchen', 'cooking', 'food', 'preparation', 'dining'],
                'description': 'Food preparation and cooking space, often connected to dining areas'
            },
            'wc': {
                'full_name': 'wc', 'english': 'toilet', 'function': 'sanitation',
                'privacy': 'private', 'typical_size': 'small',
                'adjacencies': ['käytävä', 'eteinen'],
                'keywords': ['toilet', 'wc', 'sanitation', 'restroom', 'private'],
                'description': 'Small private space for sanitation and personal needs'
            }
        }
    
    # Additional helper methods needed for functionality
    def _initialize_architectural_contexts(self):
        """Initialize architectural contexts (placeholder - implement as needed)"""
        return {}
    
    def _initialize_spatial_relationships(self):
        """Initialize spatial relationships (placeholder - implement as needed)"""
        return {}
    
    def _initialize_dimension_patterns(self):
        """Initialize dimension patterns (placeholder - implement as needed)"""
        return [
            re.compile(r"\d+['\"]?\s*x\s*\d+['\"]?"),
            re.compile(r"\d+[.,]\d+\s*m\s*x\s*\d+[.,]\d+\s*m"),
            re.compile(r"\d+[.,]\d+\s*x\s*\d+[.,]\d+"),
            re.compile(r"\d+\s*m²"),
            re.compile(r"\d+\s*sq\s*ft"),
        ]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (placeholder - implement as needed)"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _translate_finnish_text(self, text: str) -> str:
        """Translate Finnish text (placeholder - implement as needed)"""
        try:
            if self.cache_translations and text in self.translation_cache:
                return self.translation_cache[text]
            
            translated = self.translator.translate(text)
            
            if self.cache_translations:
                self.translation_cache[text] = translated
            
            return translated
        except:
            return text
    
    def _normalize_room_type_enhanced(self, text: str, language: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced room type normalization (placeholder - implement as needed)"""
        text_lower = text.lower().strip()
        
        if text_lower in self.finnish_room_types:
            room_info = self.finnish_room_types[text_lower]
            return room_info['english'], room_info
        
        return text_lower.replace(' ', '_'), {
            'english': text_lower.replace(' ', '_'),
            'function': 'unknown',
            'keywords': [text_lower],
            'description': f'Unknown room type: {text}'
        }
    
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
        else:
            return 'general_space'
    
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
        
        return list(set(tags))
    
    def _generate_search_keywords(self, room_info: Dict[str, Any], original_text: str, architectural_tags: List[str]) -> List[str]:
        """Generate comprehensive search keywords"""
        keywords = [original_text.lower()]
        
        if 'keywords' in room_info:
            keywords.extend(room_info['keywords'])
        
        if architectural_tags:
            keywords.extend(architectural_tags)
        
        return list(set(keywords))
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        cleaned = re.sub(r'\d+[.,]?\d*\s*[mx²\'"]+', '', text)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    def _parse_coordinate(self, coord_str: Optional[str]) -> float:
        """
        Parse SVG coordinate strings, stripping common units like px, em, %, etc.
        """
        if not coord_str:
            return 0.0
        try:
            # Ensure input is a string and remove leading/trailing whitespace
            coord_str = str(coord_str).strip()
            # Remove any letters (px, em, pt, etc.) and the percent sign
            numeric_part = re.sub(r'[a-zA-Z%]+', '', coord_str)
            return float(numeric_part)
        except (ValueError, TypeError) as e:
            # Log a warning if parsing fails and return a default value
            logger.warning(f"Could not parse coordinate '{coord_str}': {e}. Defaulting to 0.0.")
            return 0.0
    def _extract_svg_elements_enhanced(self, svg_file: Path) -> List[Tuple[str, float, float]]:
        """
        Extracts text elements with their coordinates from an SVG file.
        This method uses element IDs for de-duplication and is robust
        against missing namespaces.
        """
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()

            text_elements = []
            processed_ids = set()  # Track processed elements by their memory ID

            # Define the standard SVG namespace to use in findall
            namespaces = {'svg': 'http://www.w3.org/2000/svg'}

            # Create a combined list of elements found with and without the namespace
            elements_to_check = root.findall('.//svg:text', namespaces) + root.findall('.//text')

            for text_elem in elements_to_check:
                elem_id = id(text_elem)
                if elem_id in processed_ids:
                    continue
                processed_ids.add(elem_id)

                text_content = ''.join(text_elem.itertext()).strip()
                if text_content:
                    x = self._parse_coordinate(text_elem.get('x', '0'))
                    y = self._parse_coordinate(text_elem.get('y', '0'))
                    text_elements.append((text_content, x, y))

            return text_elements

        except ET.ParseError as e:
            logger.error(f"XML Parse Error in file {svg_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to extract SVG elements from {svg_file}: {e}")
            return []




# Additional classes and methods would continue here...
# This includes the full implementation of MultiModalRAGStore, enhanced analysis methods, etc.

@dataclass
class MultiModalRAGStore:
    """Enhanced RAG store with multi-modal capabilities"""
    plan_id: str
    annotations: List[MultiModalAnnotation]
    
    # Enhanced spatial relationships
    adjacency_matrix: Optional[np.ndarray] = None
    visual_adjacency_matrix: Optional[np.ndarray] = None  # Based on visual connections
    distance_matrix: Optional[np.ndarray] = None
    room_graph: Optional[Dict[str, List[str]]] = None
    circulation_paths: Optional[List[List[str]]] = None
    
    # Multi-modal features
    floor_plan_image: Optional[np.ndarray] = None
    global_visual_features: Optional[VisualFeatures] = None
    layout_clustering: Optional[Dict[str, List[int]]] = None  # Clustered annotations
    
    # Enhanced layout analysis
    layout_type: Optional[str] = None
    architectural_style: Optional[str] = None
    layout_features: Optional[Dict[str, Any]] = None
    accessibility_analysis: Optional[Dict[str, float]] = None
    energy_efficiency_score: Optional[float] = None
    
    # Multi-modal embeddings
    text_embedding_matrix: Optional[np.ndarray] = None
    visual_embedding_matrix: Optional[np.ndarray] = None
    spatial_embedding_matrix: Optional[np.ndarray] = None
    architectural_embedding_matrix: Optional[np.ndarray] = None
    composite_embedding_matrix: Optional[np.ndarray] = None
    
    # Enhanced search capabilities
    multimodal_search_index: Optional[Dict[str, List[int]]] = None
    similarity_clusters: Optional[Dict[str, List[int]]] = None
    visual_similarity_matrix: Optional[np.ndarray] = None
    
    # Metadata
    language_distribution: Optional[Dict[str, int]] = None
    dominant_language: str = 'fi'
    room_type_mapping: Optional[Dict[str, str]] = None
    processing_metadata: Optional[Dict[str, Any]] = None


class AdvancedRAGQueryEngine:
    """Advanced query engine for multi-modal RAG retrieval"""
    
    def __init__(self, rag_store: MultiModalRAGStore):
        self.rag_store = rag_store
    
        # Add device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    
        # Initialize search indices
        self.text_index = self._build_text_search_index()
        self.visual_index = self._build_visual_search_index()
        self.spatial_index = self._build_spatial_search_index()
    
        # Precompute similarity matrices
        self._precompute_similarity_matrices()

    
    def _build_text_search_index(self) -> Dict[str, List[int]]:
        """Build text-based search index"""
        index = defaultdict(list)
        
        for i, ann in enumerate(self.rag_store.annotations):
            # Index by keywords
            for keyword in ann.search_keywords or []:
                index[keyword.lower()].append(i)
            
            # Index by semantic tags
            for tag in ann.semantic_tags or []:
                index[tag.lower()].append(i)
            
            # Index by architectural tags
            for tag in ann.architectural_tags or []:
                index[tag.lower()].append(i)
            
            # Index by room type
            if ann.room_type:
                index[ann.room_type.lower()].append(i)
            
            # Index by function
            if ann.room_function:
                index[ann.room_function.lower()].append(i)
        
        return dict(index)
    
    def _build_visual_search_index(self) -> Dict[str, List[int]]:
        """Build visual feature-based search index"""
        index = defaultdict(list)
        
        for i, ann in enumerate(self.rag_store.annotations):
            if ann.visual_descriptors:
                for descriptor in ann.visual_descriptors:
                    index[descriptor.lower()].append(i)
            
            # Index by visual prominence
            if ann.spatial_info and ann.spatial_info.visual_prominence:
                if ann.spatial_info.visual_prominence > 0.7:
                    index['prominent'].append(i)
                elif ann.spatial_info.visual_prominence < 0.3:
                    index['subtle'].append(i)
            
            # Index by natural light
            if ann.spatial_info and ann.spatial_info.natural_light_score:
                if ann.spatial_info.natural_light_score > 0.7:
                    index['well_lit'].append(i)
                elif ann.spatial_info.natural_light_score < 0.3:
                    index['low_light'].append(i)
        
        return dict(index)
    
    def _build_spatial_search_index(self) -> Dict[str, List[int]]:
        """Build spatial relationship-based search index"""
        index = defaultdict(list)
        
        for i, ann in enumerate(self.rag_store.annotations):
            # Index by spatial zone
            if ann.spatial_info and ann.spatial_info.zone:
                index[ann.spatial_info.zone].append(i)
            
            # Index by relative position
            if ann.spatial_info and ann.spatial_info.relative_position:
                index[ann.spatial_info.relative_position].append(i)
            
            # Index by adjacencies
            for adjacent_room in ann.adjacent_spaces or []:
                index[f"near_{adjacent_room}"].append(i)
            
            # Index by circulation nodes
            if ann.is_circulation_node:
                index['circulation_hub'].append(i)
            
            # Index by privacy level
            if ann.spatial_info and ann.spatial_info.privacy_level:
                if ann.spatial_info.privacy_level > 0.7:
                    index['high_privacy'].append(i)
                elif ann.spatial_info.privacy_level < 0.3:
                    index['low_privacy'].append(i)
        
        return dict(index)
    
    def _precompute_similarity_matrices(self):
        """Precompute similarity matrices for different modalities"""
        if not self.rag_store.annotations:
            return
        
        # Text similarity
        if self.rag_store.text_embedding_matrix is not None:
            self.text_similarity_matrix = cosine_similarity(self.rag_store.text_embedding_matrix)
        
        # Visual similarity
        if self.rag_store.visual_embedding_matrix is not None:
            self.visual_similarity_matrix = cosine_similarity(self.rag_store.visual_embedding_matrix)
        
        # Spatial similarity
        if self.rag_store.spatial_embedding_matrix is not None:
            self.spatial_similarity_matrix = cosine_similarity(self.rag_store.spatial_embedding_matrix)
        
        # Composite similarity
        if self.rag_store.composite_embedding_matrix is not None:
            self.composite_similarity_matrix = cosine_similarity(self.rag_store.composite_embedding_matrix)
    
    def multimodal_search(self, 
                         query: str,
                         modalities: List[str] = ['text', 'spatial', 'visual'],
                         top_k: int = 10,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Advanced multi-modal search with filters"""
        
        results = []
        
        # Text-based search
        if 'text' in modalities:
            text_results = self._text_semantic_search(query, top_k * 2)
            results.extend(text_results)
        
        # Spatial search
        if 'spatial' in modalities:
            spatial_results = self._spatial_context_search(query, top_k * 2)
            results.extend(spatial_results)
        
        # Visual search
        if 'visual' in modalities:
            visual_results = self._visual_context_search(query, top_k * 2)
            results.extend(visual_results)
        
        # Combine and rank results
        combined_results = self._combine_and_rank_results(results, query, modalities)
        
        # Apply filters
        if filters:
            combined_results = self._apply_filters(combined_results, filters)
        
        # Return top results
        return combined_results[:top_k]
    
    def _text_semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic text search using embeddings"""
        query_embedding = self.embedder.encode(query)
        
        if self.rag_store.composite_embedding_matrix is None:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.rag_store.composite_embedding_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                results.append({
                    'annotation_idx': idx,
                    'annotation': self.rag_store.annotations[idx],
                    'similarity_score': float(similarities[idx]),
                    'search_type': 'semantic_text',
                    'match_reason': 'Text semantic similarity'
                })
        
        return results
    
    def _spatial_context_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search based on spatial context and relationships"""
        query_lower = query.lower()
        results = []
        
        # Keyword-based spatial search
        spatial_keywords = ['adjacent', 'near', 'connected', 'next to', 'close to', 'far from']
        room_keywords = ['bedroom', 'kitchen', 'bathroom', 'living room', 'hallway']
        
        matched_indices = set()
        
        # Direct spatial keyword matching
        for keyword, indices in self.spatial_index.items():
            if any(spatial_word in query_lower for spatial_word in spatial_keywords):
                if any(room_word in keyword for room_word in room_keywords):
                    matched_indices.update(indices)
        
        # Zone-based search
        zones = ['north', 'south', 'east', 'west', 'center', 'corner']
        positions = ['top', 'bottom', 'left', 'right', 'middle']
        
        for zone in zones + positions:
            if zone in query_lower:
                matched_indices.update(self.spatial_index.get(zone, []))
        
        # Convert to results
        for idx in list(matched_indices)[:top_k]:
            results.append({
                'annotation_idx': idx,
                'annotation': self.rag_store.annotations[idx],
                'similarity_score': 0.8,  # Fixed score for keyword matches
                'search_type': 'spatial_context',
                'match_reason': 'Spatial relationship match'
            })
        
        return results
    
    def _visual_context_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search based on visual characteristics"""
        query_lower = query.lower()
        results = []
        
        # Visual characteristic keywords
        visual_keywords = {
            'bright': ['well_lit', 'prominent'],
            'dark': ['low_light', 'subtle'],
            'complex': ['complex_layout', 'high_detail_area'],
            'simple': ['simple_layout'],
            'prominent': ['prominent', 'visually_prominent'],
            'hidden': ['subtle', 'low_prominence']
        }
        
        matched_indices = set()
        
        for query_word, visual_descriptors in visual_keywords.items():
            if query_word in query_lower:
                for descriptor in visual_descriptors:
                    matched_indices.update(self.visual_index.get(descriptor, []))
        
        # Convert to results
        for idx in list(matched_indices)[:top_k]:
            results.append({
                'annotation_idx': idx,
                'annotation': self.rag_store.annotations[idx],
                'similarity_score': 0.7,  # Fixed score for visual matches
                'search_type': 'visual_context',
                'match_reason': 'Visual characteristic match'
            })
        
        return results
    
    def _combine_and_rank_results(self, results: List[Dict], query: str, modalities: List[str]) -> List[Dict]:
        """Combine results from different modalities and rank them"""
        
        # Group results by annotation index
        grouped_results = defaultdict(list)
        for result in results:
            grouped_results[result['annotation_idx']].append(result)
        
        # Combine scores and create final results
        final_results = []
        for ann_idx, ann_results in grouped_results.items():
            # Calculate combined score
            scores = [r['similarity_score'] for r in ann_results]
            search_types = [r['search_type'] for r in ann_results]
            
            # Weighted combination based on modalities
            weights = {
                'semantic_text': 0.4,
                'spatial_context': 0.35,
                'visual_context': 0.25
            }
            
            combined_score = sum(score * weights.get(search_type, 0.3) 
                               for score, search_type in zip(scores, search_types))
            
            # Boost score for multi-modal matches
            if len(set(search_types)) > 1:
                combined_score *= 1.2
            
            final_results.append({
                'annotation_idx': ann_idx,
                'annotation': self.rag_store.annotations[ann_idx],
                'combined_score': combined_score,
                'individual_scores': {st: sc for st, sc in zip(search_types, scores)},
                'match_types': list(set(search_types)),
                'match_reasons': [r['match_reason'] for r in ann_results]
            })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            annotation = result['annotation']
            include = True
            
            # Room type filter
            if 'room_type' in filters:
                if annotation.room_type not in filters['room_type']:
                    include = False
            
            # Function filter
            if 'function' in filters:
                if annotation.room_function not in filters['function']:
                    include = False
            
            # Privacy level filter
            if 'privacy_level' in filters and annotation.spatial_info:
                min_privacy = filters['privacy_level'].get('min', 0.0)
                max_privacy = filters['privacy_level'].get('max', 1.0)
                privacy = annotation.spatial_info.privacy_level or 0.5
                if not (min_privacy <= privacy <= max_privacy):
                    include = False
            
            # Spatial zone filter
            if 'zone' in filters and annotation.spatial_info:
                if annotation.spatial_info.zone not in filters['zone']:
                    include = False
            
            # Natural light filter
            if 'natural_light' in filters and annotation.spatial_info:
                min_light = filters['natural_light'].get('min', 0.0)
                light_score = annotation.spatial_info.natural_light_score or 0.5
                if light_score < min_light:
                    include = False
            
            # Language filter
            if 'language' in filters:
                if annotation.detected_language not in filters['language']:
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def find_similar_spaces(self, annotation_idx: int, similarity_type: str = 'composite', top_k: int = 5) -> List[Dict]:
        """Find spaces similar to a given annotation"""
        
        if annotation_idx >= len(self.rag_store.annotations):
            return []
        
        similarity_matrix = None
        if similarity_type == 'text' and hasattr(self, 'text_similarity_matrix'):
            similarity_matrix = self.text_similarity_matrix
        elif similarity_type == 'visual' and hasattr(self, 'visual_similarity_matrix'):
            similarity_matrix = self.visual_similarity_matrix
        elif similarity_type == 'spatial' and hasattr(self, 'spatial_similarity_matrix'):
            similarity_matrix = self.spatial_similarity_matrix
        elif similarity_type == 'composite' and hasattr(self, 'composite_similarity_matrix'):
            similarity_matrix = self.composite_similarity_matrix
        
        if similarity_matrix is None:
            return []
        
        # Get similarities for the target annotation
        similarities = similarity_matrix[annotation_idx]
        
        # Get top similar indices (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        target_annotation = self.rag_store.annotations[annotation_idx]
        
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score > 0.1:  # Minimum threshold
                results.append({
                    'annotation_idx': idx,
                    'annotation': self.rag_store.annotations[idx],
                    'similarity_score': float(similarity_score),
                    'similarity_type': similarity_type,
                    'target_room': target_annotation.room_type,
                    'similar_room': self.rag_store.annotations[idx].room_type
                })
        
        return results


class EnhancedRAGPipeline:
    """Complete enhanced RAG pipeline with training and inference capabilities"""
    
    def __init__(self):
        self.processor = None
        self.query_engine = None
        self.stores = {}  # Dictionary of plan_id -> MultiModalRAGStore
        
        # Training data for domain adaptation
        self.training_samples = []
        self.fine_tuned_embedder = None
    
    def initialize_processor(self, 
                           embedding_model: str = 'all-MiniLM-L6-v2',
                           enable_visual_features: bool = True):
        """Initialize the RAG processor"""
        self.processor = EnhancedRAGProcessor(
            embedding_model=embedding_model,
            enable_visual_features=enable_visual_features
        )
        print(f"✅ Initialized Enhanced RAG Processor with {embedding_model}")
    
    def process_dataset(self, 
                       dataset_path: str,
                       output_path: str = "enhanced_rag_store",
                       max_files: Optional[int] = None) -> Dict[str, Any]:
        """Process entire dataset with enhanced multi-modal features"""
        
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize_processor() first.")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🏗️ Processing dataset with enhanced multi-modal features...")
        
        # Find SVG files
        svg_files = self._find_svg_files(dataset_path)
        
        if max_files:
            svg_files = svg_files[:max_files]
        
        print(f"📊 Processing {len(svg_files)} floor plans...")
        
        # Process files
        processed_stores = []
        processing_stats = {
            'total_files': len(svg_files),
            'successful': 0,
            'failed': 0,
            'total_annotations': 0,
            'total_visual_features': 0,
            'total_multimodal_embeddings': 0
        }
        
        for i, svg_file in enumerate(svg_files):
            if i % 10 == 0 and i > 0:
                print(f"   Progress: {i}/{len(svg_files)} files processed...")
            
            try:
                rag_store = self.processor.process_svg_with_multimodal_features(svg_file)
                
                if rag_store and len(rag_store.annotations) > 0:
                    processed_stores.append(rag_store)
                    self.stores[rag_store.plan_id] = rag_store
                    
                    processing_stats['successful'] += 1
                    processing_stats['total_annotations'] += len(rag_store.annotations)
                    
                    # Count multi-modal features
                    visual_count = sum(1 for ann in rag_store.annotations if ann.visual_features)
                    multimodal_count = sum(1 for ann in rag_store.annotations if ann.composite_embedding is not None)
                    
                    processing_stats['total_visual_features'] += visual_count
                    processing_stats['total_multimodal_embeddings'] += multimodal_count
                    
                    print(f"   ✅ {svg_file.name}: {len(rag_store.annotations)} annotations, {visual_count} visual features")
                else:
                    processing_stats['failed'] += 1
                    print(f"   ❌ {svg_file.name}: No annotations found")
                    
            except Exception as e:
                processing_stats['failed'] += 1
                logger.error(f"Failed to process {svg_file}: {e}")
                print(f"   ❌ {svg_file.name}: Error - {str(e)[:50]}...")
        
        # Save processed stores
        self._save_enhanced_stores(processed_stores, output_path)
        
        # Generate comprehensive statistics
        final_stats = self._generate_enhanced_statistics(processed_stores, processing_stats)
        
        print(f"\n🎯 Enhanced RAG Processing Complete!")
        print(f"   Successfully processed: {processing_stats['successful']}/{processing_stats['total_files']}")
        print(f"   Total annotations: {processing_stats['total_annotations']}")
        print(f"   Multi-modal embeddings: {processing_stats['total_multimodal_embeddings']}")
        print(f"   Visual features extracted: {processing_stats['total_visual_features']}")
        
        return final_stats
    
    def create_query_engine(self, plan_id: str) -> AdvancedRAGQueryEngine:
        """Create advanced query engine for a specific floor plan"""
        if plan_id not in self.stores:
            raise ValueError(f"Plan {plan_id} not found in processed stores")
        
        query_engine = AdvancedRAGQueryEngine(self.stores[plan_id])
        return query_engine
    
    def query_plan(self, 
                   plan_id: str,
                   query: str,
                   modalities: List[str] = ['text', 'spatial', 'visual'],
                   top_k: int = 5,
                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query a specific floor plan"""
        
        query_engine = self.create_query_engine(plan_id)
        results = query_engine.multimodal_search(
            query=query,
            modalities=modalities,
            top_k=top_k,
            filters=filters
        )
        
        return results
    
    def cross_plan_search(self,
                         query: str,
                         modalities: List[str] = ['text', 'spatial', 'visual'],
                         top_k: int = 10,
                         plan_filter: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Search across multiple floor plans"""
        
        results = {}
        
        plans_to_search = plan_filter if plan_filter else list(self.stores.keys())
        
        for plan_id in plans_to_search:
            try:
                plan_results = self.query_plan(
                    plan_id=plan_id,
                    query=query,
                    modalities=modalities,
                    top_k=top_k
                )
                
                if plan_results:
                    results[plan_id] = plan_results
                    
            except Exception as e:
                logger.warning(f"Failed to search plan {plan_id}: {e}")
        
        return results
    
    def _find_svg_files(self, dataset_path: Path) -> List[Path]:
        """Find SVG files in the dataset"""
        svg_files = []
        
        # Look for high_quality_architectural directory
        target_dir = dataset_path / "high_quality_architectural" if (dataset_path / "high_quality_architectural").exists() else dataset_path
        
        if target_dir.exists():
            for num_dir in target_dir.iterdir():
                if num_dir.is_dir() and num_dir.name.isdigit():
                    svg_files_in_dir = list(num_dir.glob("*.svg"))
                    if svg_files_in_dir:
                        # Select best representative file
                        best_file = self._select_best_svg_file(num_dir)
                        if best_file:
                            svg_files.append(best_file)
        else:
            # Fallback: search entire dataset
            svg_files = list(dataset_path.rglob("*.svg"))
        
        return svg_files
    
    def _select_best_svg_file(self, directory: Path) -> Optional[Path]:
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
    
    def _save_enhanced_stores(self, stores: List[MultiModalRAGStore], output_path: Path):
        """Save enhanced RAG stores with multi-modal features"""
        
        # Save individual stores
        stores_dir = output_path / "stores"
        stores_dir.mkdir(exist_ok=True)
        
        for store in stores:
            store_file = stores_dir / f"{store.plan_id}.pkl"
            with open(store_file, 'wb') as f:
                pickle.dump(store, f)
        
        # Save consolidated embeddings
        if stores:
            self._save_consolidated_embeddings(stores, output_path)
        
        print(f"💾 Saved {len(stores)} enhanced RAG stores to {output_path}")
    
    def _save_consolidated_embeddings(self, stores: List[MultiModalRAGStore], output_path: Path):
        """Save consolidated multi-modal embeddings"""
        
        # Collect all embeddings
        all_composite_embeddings = []
        all_text_embeddings = []
        all_spatial_embeddings = []
        all_visual_embeddings = []
        all_architectural_embeddings = []
        
        metadata = []
        plan_mapping = {}
        
        global_idx = 0
        for store in stores:
            for ann in store.annotations:
                if ann.composite_embedding is not None:
                    all_composite_embeddings.append(ann.composite_embedding)
                    
                    if ann.text_embedding is not None:
                        all_text_embeddings.append(ann.text_embedding)
                    
                    if ann.spatial_embedding is not None:
                        all_spatial_embeddings.append(ann.spatial_embedding)
                    
                    if ann.visual_embedding is not None:
                        all_visual_embeddings.append(ann.visual_embedding)
                    
                    if ann.architectural_embedding is not None:
                        all_architectural_embeddings.append(ann.architectural_embedding)
                    
                    # Metadata
                    meta = {
                        'plan_id': store.plan_id,
                        'text': ann.original_text,
                        'translated': ann.translated_text,
                        'room_type': ann.room_type,
                        'function': ann.room_function,
                        'has_visual_features': ann.visual_features is not None,
                        'global_idx': global_idx
                    }
                    metadata.append(meta)
                    plan_mapping[global_idx] = store.plan_id
                    global_idx += 1
        
        # Save embeddings
        if all_composite_embeddings:
            np.save(output_path / "composite_embeddings.npy", np.vstack(all_composite_embeddings))
        
        if all_text_embeddings:
            np.save(output_path / "text_embeddings.npy", np.vstack(all_text_embeddings))
        
        if all_spatial_embeddings:
            # Handle variable length spatial embeddings
            max_len = max(len(emb) for emb in all_spatial_embeddings)
            padded_spatial = [np.pad(emb, (0, max_len - len(emb))) for emb in all_spatial_embeddings]
            np.save(output_path / "spatial_embeddings.npy", np.vstack(padded_spatial))
        
        if all_visual_embeddings:
            # Handle variable length visual embeddings
            max_len = max(len(emb) for emb in all_visual_embeddings)
            padded_visual = [np.pad(emb, (0, max_len - len(emb))) for emb in all_visual_embeddings]
            np.save(output_path / "visual_embeddings.npy", np.vstack(padded_visual))
        
        if all_architectural_embeddings:
            np.save(output_path / "architectural_embeddings.npy", np.vstack(all_architectural_embeddings))
        
        # Save metadata
        with open(output_path / "consolidated_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        with open(output_path / "plan_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(plan_mapping, f, indent=2)
    
    def _generate_enhanced_statistics(self, stores: List[MultiModalRAGStore], processing_stats: Dict) -> Dict[str, Any]:
        """Generate comprehensive enhanced statistics"""
        
        stats = {
            'processing_summary': processing_stats,
            'timestamp': str(np.datetime64('now')),
            'total_plans': len(stores),
            'multimodal_capabilities': {
                'visual_processing': CLIP_AVAILABLE,
                'opencv_available': OPENCV_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            }
        }
        
        if stores:
            # Aggregate room type distribution
            room_type_dist = Counter()
            function_dist = Counter()
            language_dist = Counter()
            
            total_visual_features = 0
            total_multimodal_embeddings = 0
            architectural_styles = Counter()
            layout_types = Counter()
            
            for store in stores:
                # Language and room type distributions
                if store.language_distribution:
                    for lang, count in store.language_distribution.items():
                        language_dist[lang] += count
                
                # Architectural analysis
                if store.architectural_style:
                    architectural_styles[store.architectural_style] += 1
                
                if store.layout_type:
                    layout_types[store.layout_type] += 1
                
                # Annotation analysis
                for ann in store.annotations:
                    if ann.room_type:
                        room_type_dist[ann.room_type] += 1
                    if ann.room_function:
                        function_dist[ann.room_function] += 1
                    if ann.visual_features:
                        total_visual_features += 1
                    if ann.composite_embedding is not None:
                        total_multimodal_embeddings += 1
            
            # Calculate quality metrics
            avg_annotations_per_plan = sum(len(s.annotations) for s in stores) / len(stores)
            multimodal_coverage = total_multimodal_embeddings / max(sum(len(s.annotations) for s in stores), 1)
            visual_coverage = total_visual_features / max(sum(len(s.annotations) for s in stores), 1)
            
            stats.update({
                'content_analysis': {
                    'room_type_distribution': dict(room_type_dist.most_common(15)),
                    'function_distribution': dict(function_dist.most_common(10)),
                    'language_distribution': dict(language_dist),
                    'architectural_styles': dict(architectural_styles),
                    'layout_types': dict(layout_types)
                },
                'quality_metrics': {
                    'average_annotations_per_plan': avg_annotations_per_plan,
                    'multimodal_embedding_coverage': multimodal_coverage,
                    'visual_feature_coverage': visual_coverage,
                    'total_multimodal_embeddings': total_multimodal_embeddings,
                    'total_visual_features': total_visual_features
                }
            })
        
        return stats
    def load_existing_stores(self, store_path: str):
        """Load existing RAG stores from path"""
        try:
            store_path = Path(store_path)
            if not store_path.exists():
                logger.warning(f"Store path {store_path} does not exist")
                return
            
            # Load consolidated embeddings if they exist
            composite_embeddings_file = store_path / "composite_embeddings.npy"
            metadata_file = store_path / "consolidated_metadata.json"
        
            if composite_embeddings_file.exists() and metadata_file.exists():
                logger.info(f"Loading composite embeddings: {composite_embeddings_file}")
                logger.info(f"Loading metadata: {metadata_file}")
            
            stores_dir = store_path / "stores"
            if stores_dir.exists():
                store_files = list(stores_dir.glob("*.pkl"))
                logger.info(f"Found {len(store_files)} individual store files")
            
        except Exception as e:
            logger.warning(f"Could not load existing stores: {e}")


def demonstrate_enhanced_rag():
    """Demonstrate the enhanced RAG capabilities"""
    
    print("🚀 Enhanced Multi-Modal RAG for CubiCasa5K Dataset")
    print("=" * 60)
    
    # Create sample data for demonstration
    sample_dir = Path("enhanced_rag_demo")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample SVG with comprehensive annotations
    sample_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800" viewBox="0 0 1000 800">
        <!-- Room labels -->
        <text x="150" y="100">MH</text>
        <text x="250" y="100">14'10" x 9'4"</text>
        <text x="500" y="150">OH</text>
        <text x="600" y="150">16'3" x 12'11"</text>
        <text x="150" y="300">KH</text>
        <text x="250" y="300">5'1" x 8'7"</text>
        <text x="150" y="500">VH</text>
        <text x="700" y="200">keittiö</text>
        <text x="800" y="200">12'x10'</text>
        <text x="450" y="600">eteinen</text>
        <text x="850" y="450">ULKOTILA</text>
        <text x="100" y="700">sauna</text>
        <text x="300" y="650">WC</text>
        <text x="750" y="650">parveke</text>
        
        <!-- Simple room boundaries for visual processing -->
        <rect x="100" y="50" width="200" height="150" fill="none" stroke="black" stroke-width="2"/>
        <rect x="400" y="100" width="250" height="200" fill="none" stroke="black" stroke-width="2"/>
        <rect x="100" y="250" width="150" height="120" fill="none" stroke="black" stroke-width="2"/>
        <rect x="100" y="450" width="120" height="100" fill="none" stroke="black" stroke-width="2"/>
        <rect x="650" y="150" width="200" height="150" fill="none" stroke="black" stroke-width="2"/>
        <rect x="400" y="550" width="150" height="80" fill="none" stroke="black" stroke-width="2"/>
        <rect x="750" y="400" width="200" height="150" fill="none" stroke="black" stroke-width="2"/>
        <rect x="50" y="650" width="100" height="100" fill="none" stroke="black" stroke-width="2"/>
        <rect x="250" y="600" width="80" height="80" fill="none" stroke="black" stroke-width="2"/>
        <rect x="700" y="600" width="120" height="80" fill="none" stroke="black" stroke-width="2"/>
    </svg>'''
    
    sample_svg_file = sample_dir / "sample_enhanced_plan.svg"
    with open(sample_svg_file, 'w', encoding='utf-8') as f:
        f.write(sample_svg_content)
    
    # Initialize the enhanced RAG pipeline
    pipeline = EnhancedRAGPipeline()
    pipeline.initialize_processor(
        embedding_model='all-MiniLM-L6-v2',
        enable_visual_features=True
    )
    
    print("\n📊 Processing sample floor plan with enhanced features...")
    
    # Process the sample file directly
    try:
        rag_store = pipeline.processor.process_svg_with_multimodal_features(sample_svg_file)
        
        if rag_store:
            pipeline.stores[rag_store.plan_id] = rag_store
            
            print(f"✅ Successfully processed: {rag_store.plan_id}")
            print(f"   Total annotations: {len(rag_store.annotations)}")
            print(f"   Room labels: {sum(1 for ann in rag_store.annotations if ann.is_room_label)}")
            print(f"   Dimensions: {sum(1 for ann in rag_store.annotations if ann.is_dimension)}")
            print(f"   Visual features: {sum(1 for ann in rag_store.annotations if ann.visual_features)}")
            print(f"   Multi-modal embeddings: {sum(1 for ann in rag_store.annotations if ann.composite_embedding is not None)}")
            
            # Demonstrate enhanced RAG queries
            print(f"\n🔍 Demonstrating Enhanced RAG Queries:")
            print("-" * 40)
            
            # Create query engine
            query_engine = pipeline.create_query_engine(rag_store.plan_id)
            
            # Sample queries
            demo_queries = [
                {
                    'query': 'bedroom with natural light',
                    'modalities': ['text', 'visual', 'spatial'],
                    'description': 'Multi-modal search for bedroom with lighting'
                },
                {
                    'query': 'rooms adjacent to kitchen',
                    'modalities': ['text', 'spatial'],
                    'description': 'Spatial relationship search'
                },
                {
                    'query': 'private spaces in the north area',
                    'modalities': ['text', 'spatial'],
                    'description': 'Privacy and location-based search'
                },
                {
                    'query': 'visually prominent spaces',
                    'modalities': ['visual', 'spatial'],
                    'description': 'Visual characteristic search'
                },
                {
                    'query': 'service areas with plumbing',
                    'modalities': ['text'],
                    'description': 'Functional requirement search'
                }
            ]
            
            for i, demo_query in enumerate(demo_queries, 1):
                print(f"\n{i}. {demo_query['description']}")
                print(f"   Query: '{demo_query['query']}'")
                print(f"   Modalities: {demo_query['modalities']}")
                
                try:
                    results = query_engine.multimodal_search(
                        query=demo_query['query'],
                        modalities=demo_query['modalities'],
                        top_k=3
                    )
                    
                    if results:
                        print(f"   Results ({len(results)} found):")
                        for j, result in enumerate(results):
                            ann = result['annotation']
                            score = result['combined_score']
                            match_types = result['match_types']
                            
                            print(f"     {j+1}. '{ann.original_text}' -> {ann.room_type}")
                            print(f"        Score: {score:.3f} | Match types: {match_types}")
                            if ann.spatial_info:
                                print(f"        Location: {ann.spatial_info.relative_position} {ann.spatial_info.zone}")
                            if ann.room_function:
                                print(f"        Function: {ann.room_function}")
                    else:
                        print("   No results found")
                        
                except Exception as e:
                    print(f"   Error: {e}")
            
            # Demonstrate similarity search
            print(f"\n🔗 Demonstrating Similarity Search:")
            print("-" * 40)
            
            # Find a bedroom annotation
            bedroom_idx = None
            for i, ann in enumerate(rag_store.annotations):
                if ann.room_type == 'bedroom' or 'mh' in ann.original_text.lower():
                    bedroom_idx = i
                    break
            
            if bedroom_idx is not None:
                print(f"Finding spaces similar to: '{rag_store.annotations[bedroom_idx].original_text}'")
                
                similar_spaces = query_engine.find_similar_spaces(
                    annotation_idx=bedroom_idx,
                    similarity_type='composite',
                    top_k=3
                )
                
                if similar_spaces:
                    print("Similar spaces:")
                    for result in similar_spaces:
                        ann = result['annotation']
                        score = result['similarity_score']
                        print(f"  • '{ann.original_text}' -> {ann.room_type} (similarity: {score:.3f})")
                else:
                    print("No similar spaces found")
            
            # Display enhanced annotation details
            print(f"\n📋 Enhanced Annotation Examples:")
            print("-" * 40)
            
            for i, ann in enumerate(rag_store.annotations[:3]):  # Show first 3
                print(f"\n{i+1}. Original: '{ann.original_text}'")
                print(f"   Translated: '{ann.translated_text}'")
                print(f"   Room Type: {ann.room_type}")
                print(f"   Function: {ann.room_function}")
                print(f"   Semantic Category: {ann.semantic_category}")
                
                if ann.spatial_info:
                    print(f"   Location: {ann.spatial_info.relative_position} ({ann.spatial_info.zone})")
                    if ann.spatial_info.visual_prominence:
                        print(f"   Visual Prominence: {ann.spatial_info.visual_prominence:.2f}")
                    if ann.spatial_info.natural_light_score:
                        print(f"   Natural Light: {ann.spatial_info.natural_light_score:.2f}")
                
                if ann.adjacent_spaces:
                    print(f"   Adjacent to: {', '.join(ann.adjacent_spaces)}")
                
                if ann.architectural_tags:
                    print(f"   Architectural Tags: {ann.architectural_tags[:3]}")
                
                if ann.visual_descriptors:
                    print(f"   Visual Descriptors: {ann.visual_descriptors}")
                
                if ann.composite_embedding is not None:
                    print(f"   Embedding Dimension: {len(ann.composite_embedding)}")
                
                print(f"   RAG Context: {ann.rag_context[:100]}...")
        
        else:
            print("❌ Failed to process sample file")
            
    except Exception as e:
        print(f"❌ Error processing sample: {e}")
        import traceback
        traceback.print_exc()
    
    # Additional demonstration features
    print(f"\n🎯 Enhanced Features Summary:")
    print("-" * 40)
    print("✅ Multi-modal embeddings (text + visual + spatial + architectural)")
    print("✅ CLIP-based visual feature extraction")
    print("✅ Domain-specific architectural embeddings")
    print("✅ Enhanced spatial relationship analysis")
    print("✅ Advanced multi-modal query engine")
    print("✅ Visual similarity search")
    print("✅ Architectural context understanding")
    print("✅ Finnish language processing with architectural terms")
    print("✅ Natural language to architectural concept mapping")
    print("✅ Privacy, lighting, and accessibility analysis")
    
    print(f"\n💡 Next Steps for Production Use:")
    print("-" * 40)
    print("1. Fine-tune on larger architectural datasets")
    print("2. Train custom visual encoders on floor plan images")
    print("3. Implement graph neural networks for spatial relationships")
    print("4. Add temporal analysis for renovation/design evolution")
    print("5. Integrate with CAD software APIs")
    print("6. Deploy as microservice with REST API")
    print("7. Add real-time collaborative features")
    print("8. Implement automated floor plan generation")
    
    return pipeline


# Additional utility functions for enhanced RAG

def fine_tune_architectural_embedder(training_data: List[Dict], base_model: str = 'all-MiniLM-L6-v2'):
    """
    Fine-tune embeddings on architectural data
    
    Args:
        training_data: List of dicts with 'text', 'room_type', 'function', 'context'
        base_model: Base sentence transformer model
    
    Returns:
        Fine-tuned sentence transformer model
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import TripletLoss
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
    
    print("🎯 Fine-tuning embeddings on architectural data...")
    
    # Load base model
    model = SentenceTransformer(base_model)
    
    # Create training triplets (anchor, positive, negative)
    training_examples = []
    
    # Group by room type for positive/negative sampling
    room_groups = defaultdict(list)
    for item in training_data:
        room_groups[item['room_type']].append(item)
    
    # Generate triplets
    for room_type, items in room_groups.items():
        other_room_types = [rt for rt in room_groups.keys() if rt != room_type]
        
        for item in items:
            anchor = item['text']
            
            # Positive: same room type
            if len(items) > 1:
                positive_candidates = [i for i in items if i != item]
                positive = np.random.choice(positive_candidates)['text']
            else:
                positive = anchor  # Self if only one example
            
            # Negative: different room type
            if other_room_types:
                negative_room = np.random.choice(other_room_types)
                negative = np.random.choice(room_groups[negative_room])['text']
                
                training_examples.append([anchor, positive, negative])
    
    # Create data loader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)
    
    # Define loss
    train_loss = TripletLoss(model=model)
    
    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  # Start with 1 epoch for demo
        warmup_steps=100,
        show_progress_bar=True
    )
    
    print("✅ Fine-tuning completed")
    return model


def create_architectural_knowledge_graph(rag_stores: List[MultiModalRAGStore]):
    """
    Create knowledge graph from processed RAG stores
    
    Args:
        rag_stores: List of processed MultiModalRAGStore objects
    
    Returns:
        NetworkX graph with architectural relationships
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not available. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "networkx"])
        import networkx as nx
    
    print("🕸️ Creating architectural knowledge graph...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and relationships
    for store in rag_stores:
        plan_node = f"plan_{store.plan_id}"
        G.add_node(plan_node, type='floor_plan', layout_type=store.layout_type)
        
        # Add room nodes and relationships
        for ann in store.annotations:
            if ann.is_room_label and ann.room_type:
                room_node = f"{store.plan_id}_{ann.room_type}_{ann.spatial_info.x}_{ann.spatial_info.y}"
                
                # Add room node
                G.add_node(room_node, 
                          type='room',
                          room_type=ann.room_type,
                          function=ann.room_function,
                          text=ann.translated_text,
                          privacy_level=ann.spatial_info.privacy_level if ann.spatial_info else None,
                          natural_light=ann.spatial_info.natural_light_score if ann.spatial_info else None)
                
                # Connect to floor plan
                G.add_edge(plan_node, room_node, relationship='contains')
                
                # Add adjacency relationships
                for adjacent_room in ann.adjacent_spaces or []:
                    # Find adjacent room nodes
                    adjacent_nodes = [n for n, d in G.nodes(data=True) 
                                    if d.get('room_type') == adjacent_room and 
                                       n.startswith(f"{store.plan_id}_")]
                    
                    for adj_node in adjacent_nodes:
                        G.add_edge(room_node, adj_node, relationship='adjacent_to')
                
                # Add functional relationships
                if ann.room_function:
                    function_node = f"function_{ann.room_function}"
                    if not G.has_node(function_node):
                        G.add_node(function_node, type='function')
                    G.add_edge(room_node, function_node, relationship='serves_function')
    
    print(f"✅ Created knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def main():
    """Main demonstration function"""
    
    print("🌟 Enhanced Multi-Modal RAG System for CubiCasa5K")
    print("=" * 60)
    
    # Run demonstration
    pipeline = demonstrate_enhanced_rag()
    
    # Optional: Process full dataset if path provided
    dataset_path = input("\n🔍 Enter CubiCasa5K dataset path for full processing (or press Enter to skip): ").strip()
    
    if dataset_path and Path(dataset_path).exists():
        max_files = input("Max files to process (default: 50): ").strip()
        max_files = int(max_files) if max_files.isdigit() else 50
        
        print(f"\n🚀 Processing full dataset with {max_files} files...")
        
        try:
            stats = pipeline.process_dataset(
                dataset_path=dataset_path,
                output_path="enhanced_multimodal_rag_store",
                max_files=max_files
            )
            
            if stats and 'error' not in stats:
                print(f"\n📊 Final Statistics:")
                print(f"   Total plans processed: {stats['total_plans']}")
                print(f"   Multi-modal embedding coverage: {stats['quality_metrics']['multimodal_embedding_coverage']:.1%}")
                print(f"   Visual feature coverage: {stats['quality_metrics']['visual_feature_coverage']:.1%}")
                print(f"   Top room types: {list(stats['content_analysis']['room_type_distribution'].keys())[:5]}")
                
                # Create knowledge graph
                if len(pipeline.stores) > 0:
                    kg = create_architectural_knowledge_graph(list(pipeline.stores.values()))
                    print(f"   Knowledge graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
            else:
                print(f"❌ Processing failed: {stats.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during full dataset processing: {e}")
    
    print(f"\n✨ Enhanced RAG demonstration completed!")
    print("The system now supports:")
    print("• Multi-modal embeddings with text, visual, spatial, and architectural features")
    print("• Advanced spatial relationship analysis with visual cues")
    print("• Domain-specific architectural knowledge integration")
    print("• Cross-modal similarity search and retrieval")
    print("• Finnish architectural term processing")
    print("• Natural language architectural queries")


if __name__ == "__main__":
    main()