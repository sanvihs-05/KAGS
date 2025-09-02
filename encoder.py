import numpy as np
import logging
import json
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
import hashlib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EncodingConfig:
    """Configuration for Ollama embedding"""
    model_name: str = "nomic-embed-text"  # Embedding-capable model
    api_endpoint: str = "http://localhost:11434/api/embed"  # Correct Ollama endpoint
    max_sequence_length: int = 8192
    batch_size: int = 32
    cache_embeddings: bool = True
    normalize_embeddings: bool = True
    embedding_dimension: int = 768  # nomic-embed-text dimension

class Gemma3Encoder:
    """
    Advanced Embedding Encoder for architectural design prototypes
    Supports both text and multimodal encoding capabilities via Ollama
    """
    
    def __init__(self, config: EncodingConfig = None):
        self.config = config or EncodingConfig()
        self.session = requests.Session()
        self.cache = {} if self.config.cache_embeddings else None
        
        # Test connection to Ollama
        self._test_connection()
        
        logger.info(f"✅ Initialized Embedding Encoder: {self.config.model_name}")
    
    def _test_connection(self):
        """Test connection to Ollama API"""
        try:
            test_payload = {
                "model": self.config.model_name,
                "input": "test connection"
            }

            response = self.session.post(
                self.config.api_endpoint,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Ollama API connection successful")
            else:
                logger.warning(f"⚠️ Ollama API returned status: {response.status_code}")
                logger.info(f"Response: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama API: {e}")
            logger.info("💡 Make sure Ollama is running with embedding model")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts using Ollama embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.config.embedding_dimension)
        
        # Check cache first
        if self.cache is not None:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache:
                    cached_embeddings.append((i, self.cache[text_hash]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if not uncached_texts:
                # All texts are cached
                result = np.zeros((len(texts), self.config.embedding_dimension))
                for i, embedding in cached_embeddings:
                    result[i] = embedding
                return result
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_embeddings = []
        
        # Encode uncached texts in batches
        new_embeddings = []
        
        for i in range(0, len(uncached_texts), self.config.batch_size):
            batch_texts = uncached_texts[i:i + self.config.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            new_embeddings.extend(batch_embeddings)
        
        # Cache new embeddings
        if self.cache is not None:
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._get_text_hash(text)
                self.cache[text_hash] = embedding
        
        # Combine cached and new embeddings
        result = np.zeros((len(texts), self.config.embedding_dimension))
        
        # Add cached embeddings
        for i, embedding in cached_embeddings:
            result[i] = embedding
        
        # Add new embeddings
        for idx, embedding in zip(uncached_indices, new_embeddings):
            result[idx] = embedding
        
        return result
    
    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts using Ollama API"""
        try:
            embeddings = []
            
            # Ollama /api/embed processes one text at a time
            for text in texts:
                payload = {
                    "model": self.config.model_name,
                    "input": text
                }
                
                # Make API request to Ollama
                response = self.session.post(
                    self.config.api_endpoint,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Ollama returns embeddings array
                    if data.get("embeddings") and len(data["embeddings"]) > 0:
                        embedding = np.array(data["embeddings"][0], dtype=np.float32)
                        if self.config.normalize_embeddings:
                            embedding = self._normalize_embedding(embedding)
                        embeddings.append(embedding)
                    else:
                        logger.warning("⚠️ Ollama returned null embedding, using fallback")
                        embeddings.append(self._random_embedding())
                else:
                    logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                    embeddings.append(self._random_embedding())
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Encoding batch failed: {e}")
            # Fallback to random embeddings
            return [self._random_embedding() for _ in texts]
    
    def encode_prototype_features(self, prototype: Dict[str, Any]) -> np.ndarray:
        """
        Encode architectural prototype features using embeddings
        
        Args:
            prototype: Prototype configuration dictionary
            
        Returns:
            Comprehensive embedding vector
        """
        try:
            # Extract comprehensive features for encoding
            feature_texts = self._extract_prototype_texts(prototype)
            
            # Encode all features
            if feature_texts:
                text_embeddings = self.encode_texts(feature_texts)
                # Average pool the embeddings
                prototype_embedding = np.mean(text_embeddings, axis=0)
            else:
                prototype_embedding = self._random_embedding()
            
            # Add numerical features
            numerical_features = self._extract_numerical_features(prototype)
            
            # Combine text and numerical features
            combined_embedding = np.concatenate([
                prototype_embedding,
                numerical_features
            ])
            
            return combined_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Failed to encode prototype features: {e}")
            # Return default embedding
            return self._random_embedding(size=self.config.embedding_dimension + 50)
    
    def _extract_prototype_texts(self, prototype: Dict[str, Any]) -> List[str]:
        """Extract textual descriptions from prototype for encoding"""
        texts = []
        
        # Spatial configuration
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        if spatial_config:
            strategy = spatial_config.get('strategy', '')
            if strategy:
                texts.append(f"Architectural spatial strategy: {strategy}")
            
            compactness = spatial_config.get('compactness_factor', 0)
            if compactness:
                texts.append(f"Building compactness factor {compactness:.2f} for efficient space utilization")
        
        # Environmental strategy
        env_strategy = prototype.get('detailed_config', {}).get('environmental_strategy', {})
        if env_strategy:
            orientation = env_strategy.get('orientation', '')
            passive_strategies = env_strategy.get('passive_strategies', [])
            
            if orientation:
                texts.append(f"Building orientation facing {orientation} for optimal solar exposure")
            
            for strategy in passive_strategies:
                texts.append(f"Passive environmental strategy: {strategy} for sustainable design")
        
        # Functional zones
        functional_zones = prototype.get('detailed_config', {}).get('functional_zones', {})
        for zone_name, zone_data in functional_zones.items():
            if isinstance(zone_data, dict):
                rooms = zone_data.get('rooms', [])
                ratio = zone_data.get('ratio', 0)
                if rooms:
                    texts.append(f"{zone_name} contains {', '.join(rooms)} with {ratio:.1%} area allocation")
        
        # Circulation pattern
        circulation = prototype.get('detailed_config', {}).get('circulation_pattern', {})
        if circulation:
            pattern_type = circulation.get('pattern_type', '')
            efficiency = circulation.get('efficiency_target', 0)
            if pattern_type:
                texts.append(f"Circulation pattern: {pattern_type} with {efficiency:.1%} efficiency target")
        
        # Performance scores (convert to descriptive text)
        criterion_scores = prototype.get('criterion_scores', {})
        for criterion, data in criterion_scores.items():
            score = data.get('score', 0)
            if score > 0.7:
                texts.append(f"High performance in {criterion.replace('_', ' ')}: {score:.1%}")
            elif score < 0.5:
                texts.append(f"Improvement needed in {criterion.replace('_', ' ')}: {score:.1%}")
        
        return texts
    
    def _extract_numerical_features(self, prototype: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from prototype"""
        features = []
        
        # Performance scores
        features.append(prototype.get('final_score', 0.5))
        features.append(prototype.get('overall_confidence', 0.5))
        features.append(prototype.get('weighted_total', 0.5))
        features.append(prototype.get('diversity_bonus', 0.0))
        
        # Spatial features
        spatial_config = prototype.get('detailed_config', {}).get('spatial_config', {})
        features.append(spatial_config.get('plot_utilization', 0.7))
        features.append(spatial_config.get('compactness_factor', 0.7))
        
        # Environmental features
        env_strategy = prototype.get('detailed_config', {}).get('environmental_strategy', {})
        passive_count = len(env_strategy.get('passive_strategies', []))
        features.append(passive_count / 5.0)  # Normalize
        
        # Circulation features
        circulation = prototype.get('detailed_config', {}).get('circulation_pattern', {})
        features.append(circulation.get('efficiency_target', 0.85))
        
        # Functional zone ratios
        functional_zones = prototype.get('detailed_config', {}).get('functional_zones', {})
        for zone_type in ['public_zone', 'private_zone', 'service_zone']:
            zone_data = functional_zones.get(zone_type, {})
            features.append(zone_data.get('ratio', 0.33))
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            return embedding / norm
        return embedding
    
    def _random_embedding(self, size: int = None) -> np.ndarray:
        """Generate random embedding as fallback"""
        size = size or self.config.embedding_dimension
        embedding = np.random.normal(0, 0.02, size).astype(np.float32)
        if self.config.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
        return embedding
    
    def encode_research_query(self, 
                            prototype_config: Dict[str, Any],
                            requirements: Dict[str, Any],
                            query_type: str) -> np.ndarray:
        """
        Encode research query for RAG retrieval
        
        Args:
            prototype_config: Prototype configuration
            requirements: User requirements
            query_type: Type of research query
            
        Returns:
            Query embedding for similarity search
        """
        # Generate comprehensive research query text
        query_text = self._generate_research_query_text(
            prototype_config, requirements, query_type
        )
        
        # Encode the query
        query_embedding = self.encode_texts([query_text])[0]
        
        return query_embedding
    
    def _generate_research_query_text(self, 
                                    prototype_config: Dict[str, Any],
                                    requirements: Dict[str, Any],
                                    query_type: str) -> str:
        """Generate comprehensive research query text"""
        
        # Extract key information
        spatial_needs = requirements.get('spatial_needs', [])
        room_types = [need.get('room_type', '') for need in spatial_needs]
        
        spatial_strategy = prototype_config.get('detailed_config', {}).get('spatial_config', {}).get('strategy', 'unknown')
        
        env_strategy = prototype_config.get('detailed_config', {}).get('environmental_strategy', {})
        orientation = env_strategy.get('orientation', 'south')
        passive_strategies = env_strategy.get('passive_strategies', [])
        
        # Generate query based on type
        if query_type == "spatial_optimization":
            query = f"Optimize spatial layout for {', '.join(room_types)} using {spatial_strategy} architectural strategy with efficient space utilization and functional zoning"
        
        elif query_type == "functional_adjacency":
            query = f"Functional room adjacencies and relationships for {', '.join(room_types)} in residential architecture with optimal circulation and privacy"
        
        elif query_type == "environmental_strategy":
            query = f"Environmental design strategies for {orientation} facing building with {', '.join(passive_strategies)} sustainable architecture climate response"
        
        elif query_type == "circulation_patterns":
            circulation_type = prototype_config.get('detailed_config', {}).get('circulation_pattern', {}).get('pattern_type', 'linear')
            query = f"{circulation_type} circulation patterns for residential layout with {len(room_types)} rooms efficient movement and accessibility"
        
        else:
            query = f"Architectural design solutions for {spatial_strategy} residential layout with {', '.join(room_types)} sustainable and functional design"
        
        return query
    
    def save_cache(self, cache_path: str = "gemma3_encoder_cache.pkl"):
        """Save embedding cache to disk"""
        if self.cache:
            try:
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.cache, f)
                logger.info(f"✅ Saved {len(self.cache)} embeddings to cache")
            except Exception as e:
                logger.error(f"❌ Failed to save cache: {e}")
    
    def load_cache(self, cache_path: str = "gemma3_encoder_cache.pkl"):
        """Load embedding cache from disk"""
        try:
            import pickle
            if Path(cache_path).exists():
                with open(cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")
            self.cache = {}

# Usage example and integration with existing system
class EnhancedResearchAgent:
    """Enhanced Research Agent using Ollama Encoder"""
    
    def __init__(self, rag_store_path: str = "enhanced_multimodal_rag_store"):
        self.encoder = Gemma3Encoder()
        self.rag_store_path = Path(rag_store_path)
        
        # Load cache if available
        self.encoder.load_cache()
        
        # Load RAG store
        self._load_rag_store()
    
    def _load_rag_store(self):
        """Load RAG store with embeddings"""
        try:
            # Load metadata
            metadata_path = self.rag_store_path / "consolidated_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded {len(self.metadata)} metadata entries")
            
            # Load or create embeddings
            embeddings_path = self.rag_store_path / "ollama_embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                logger.info(f"✅ Loaded embeddings: {self.embeddings.shape}")
            else:
                logger.info("🔄 Creating new embeddings with Ollama...")
                self._create_embeddings()
                
        except Exception as e:
            logger.error(f"❌ Failed to load RAG store: {e}")
    
    def _create_embeddings(self):
        """Create embeddings for RAG store using Ollama"""
        if not hasattr(self, 'metadata'):
            logger.error("No metadata available for embedding creation")
            return
        
        # Extract texts for embedding
        texts = []
        for item in self.metadata:
            text = item.get('original_text', '')
            if not text:
                # Create text from metadata
                room_type = item.get('room_type', 'unknown')
                room_function = item.get('room_function', 'unknown')
                spatial_info = item.get('spatial_info', {})
                text = f"Room type {room_type} with function {room_function} and spatial characteristics {spatial_info}"
            texts.append(text)
        
        # Encode with Ollama
        logger.info(f"🔄 Encoding {len(texts)} texts with Ollama...")
        self.embeddings = self.encoder.encode_texts(texts)
        
        # Save embeddings
        embeddings_path = self.rag_store_path / "ollama_embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        logger.info(f"✅ Saved embeddings: {self.embeddings.shape}")
        
        # Save encoder cache
        self.encoder.save_cache()

# Export the enhanced encoder
if __name__ == "__main__":
    print("🚀 Testing Ollama Encoder...")
    
    # Initialize encoder
    config = EncodingConfig()
    encoder = Gemma3Encoder(config)
    
    # Test encoding
    test_texts = [
        "Modern residential architecture with open plan design",
        "Sustainable building with passive solar heating and natural ventilation", 
        "Compact urban housing with efficient space utilization"
    ]
    
    print("🔄 Encoding test texts...")
    embeddings = encoder.encode_texts(test_texts)
    print(f"✅ Generated embeddings shape: {embeddings.shape}")
    
    # Test prototype encoding
    sample_prototype = {
        'prototype_id': 'test_proto',
        'final_score': 0.85,
        'detailed_config': {
            'spatial_config': {
                'strategy': 'central_core',
                'compactness_factor': 0.8
            },
            'environmental_strategy': {
                'orientation': 'south',
                'passive_strategies': ['cross_ventilation', 'solar_shading']
            }
        }
    }
    
    print("🔄 Encoding prototype features...")
    proto_embedding = encoder.encode_prototype_features(sample_prototype)
    print(f"✅ Generated prototype embedding shape: {proto_embedding.shape}")
    
    print("✅ Ollama Encoder ready for integration!")
