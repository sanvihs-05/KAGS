import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FinnishFloorPlanEmbeddingLoader:
    """
    Loader specifically for Finnish floor plan embeddings from CubiCasa5K
    with multi-modal features (CLIP visual + text + spatial + architectural)
    """
    
    def __init__(self, embeddings_path: str):
        self.embeddings_path = Path(embeddings_path)
        
        # Initialize embedding model for encoding new queries
        logger.info("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ SentenceTransformer model loaded")
        
        # Finnish room type mapping
        self.finnish_room_types = {
            'oh': {'full_name': 'olohuone', 'english': 'living_room', 'function': 'social_relaxation'},
            'mh': {'full_name': 'makuuhuone', 'english': 'bedroom', 'function': 'rest_sleep'},
            'kh': {'full_name': 'kylpyhuone', 'english': 'bathroom', 'function': 'hygiene_bathing'},
            'vh': {'full_name': 'vesihuone', 'english': 'utility_room', 'function': 'utility_cleaning'},
            'wc': {'full_name': 'wc', 'english': 'toilet', 'function': 'sanitation'},
            'keittiö': {'full_name': 'keittiö', 'english': 'kitchen', 'function': 'food_preparation'},
            'k': {'english': 'kitchen', 'finnish': 'keittiö', 'type': 'service'},
            'kk': {'english': 'kitchenette', 'finnish': 'keittokomero', 'type': 'service'},
            's': {'english': 'sauna', 'finnish': 'sauna', 'type': 'wellness'},
            'sauna': {'english': 'sauna', 'finnish': 'sauna', 'type': 'wellness'},
            'et': {'english': 'hallway', 'finnish': 'eteinen', 'type': 'circulation'},
            'eteinen': {'english': 'hallway', 'finnish': 'eteinen', 'type': 'circulation'},
            'var': {'english': 'storage', 'finnish': 'varasto', 'type': 'storage'},
            'varasto': {'english': 'storage', 'finnish': 'varasto', 'type': 'storage'},
            'p': {'english': 'balcony', 'finnish': 'parveke', 'type': 'outdoor'},
            'parveke': {'english': 'balcony', 'finnish': 'parveke', 'type': 'outdoor'},
            'cl': {'english': 'closet', 'finnish': 'kaappi', 'type': 'storage'},
            'cb': {'english': 'cabinet', 'finnish': 'kaappi', 'type': 'storage'},
            'cwh': {'english': 'clothes_room', 'finnish': 'vaatehuone', 'type': 'storage'},
            'sink': {'english': 'sink', 'finnish': 'pesuallas', 'type': 'fixture'},
            'undefined': {'english': 'undefined', 'finnish': 'määrittelemätön', 'type': 'unknown'},
        }
        
        # Load all data
        self.plan_mapping = None
        self.metadata = None
        self.embeddings = {}
        self.stores = {}
        
        self._load_all()
    
    def _load_all(self):
        """Load all embedding files and metadata"""
        # Load plan mapping
        plan_mapping_file = self.embeddings_path / 'plan_mapping.json'
        if plan_mapping_file.exists():
            try:
                with open(plan_mapping_file, 'r') as f:
                    self.plan_mapping = json.load(f)
                logger.info(f"✓ Loaded plan mapping: {len(self.plan_mapping)} entries")
            except Exception as e:
                logger.warning(f"Failed to load plan mapping {plan_mapping_file}: {e}")

        # Load consolidated metadata
        metadata_file = self.embeddings_path / 'consolidated_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Loaded metadata: {len(self.metadata)} annotations")
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")

        # Load embedding files
        embedding_types = ['architectural', 'composite', 'spatial', 'text', 'visual']
        for emb_type in embedding_types:
            emb_file = self.embeddings_path / f'{emb_type}_embeddings.npy'
            if emb_file.exists():
                try:
                    # Use memory mapping for large files, and always cast to float32 to save RAM
                    arr = np.load(emb_file, mmap_mode='r')
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    self.embeddings[emb_type] = arr
                    logger.info(f"✓ Loaded {emb_type} embeddings: {self.embeddings[emb_type].shape} (dtype={arr.dtype})")
                except Exception as e:
                    logger.warning(f"Failed to load embedding {emb_file}: {e}")

        # Try to load individual RAG stores (optional)
        stores_dir = self.embeddings_path / 'stores'
        if stores_dir.exists():
            import pickle

            class CompatUnpickler(pickle.Unpickler):
                """Compatibility Unpickler that maps classes pickled under `__main__`
                (e.g. `MultiModalRAGStore`, `MultiModalAnnotation`) to simple
                shim classes created at unpickling time. This allows older
                pickles (created in different module contexts) to be loaded
                without failing the import.
                """

                def find_class(self, module, name):
                    if module == '__main__':
                        # Create a lightweight shim class with the original name so
                        # pickle can set attributes on it. The resulting object
                        # will behave like a simple container.
                        return type(name, (object,), {})
                    return super().find_class(module, name)

            for store_file in stores_dir.glob('*.pkl'):
                with open(store_file, 'rb') as f:
                    try:
                        unpickler = CompatUnpickler(f)
                        store = unpickler.load()
                    except (AttributeError, pickle.UnpicklingError) as ue:
                        logger.warning(
                            f"Could not unpickle {store_file}: {ue} — skipping pickled store, will use .npy data instead."
                        )
                        continue
                    except Exception as e:
                        logger.warning(f"Could not load store {store_file}: {e}")
                        continue

                    # store loaded successfully; ensure it has plan_id
                    pid = getattr(store, 'plan_id', None)
                    if pid:
                        self.stores[pid] = store
                    else:
                        logger.warning(f"Loaded pickle {store_file} but missing 'plan_id' attribute; skipping store")

            if self.stores:
                logger.info(f"✓ Loaded {len(self.stores)} individual RAG stores")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a text query to embedding vector"""
        return self.embedding_model.encode(query)
    
    def get_embedding_by_index(self, index: int, embedding_type: str = 'composite') -> Optional[np.ndarray]:
        """Get embedding by global index"""
        if embedding_type not in self.embeddings:
            return None
        
        if index < 0 or index >= len(self.embeddings[embedding_type]):
            return None
        
        return self.embeddings[embedding_type][index]
    
    def get_metadata_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata by global index"""
        if not self.metadata or index >= len(self.metadata):
            return None
        return self.metadata[index]
    
    def search_by_room_type(self, room_type: str, 
                           embedding_type: str = 'composite',
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar rooms by type"""
        results = []
        
        # Normalize room type
        room_type_lower = room_type.lower()
        
        # Find all annotations of this room type
        matching_indices = []
        for i, meta in enumerate(self.metadata):
            meta_room_type = meta.get('room_type', '').lower()
            if room_type_lower in meta_room_type or meta_room_type in room_type_lower:
                matching_indices.append(i)
        
        # Get embeddings for matches
        for idx in matching_indices[:top_k]:
            embedding = self.get_embedding_by_index(idx, embedding_type)
            metadata = self.get_metadata_by_index(idx)
            
            if embedding is not None and metadata:
                finnish_type = metadata.get('room_type', '')
                english_type = self.finnish_room_types.get(finnish_type, {}).get('english', finnish_type)
                
                results.append({
                    'index': idx,
                    'plan_id': metadata.get('plan_id'),
                    'room_type': english_type,
                    'finnish_type': finnish_type,
                    'original_text': metadata.get('text'),
                    'translated_text': metadata.get('translated'),
                    'function': metadata.get('function'),
                    'embedding': embedding,
                    'metadata': metadata
                })
        
        return results
    
    def search_similar_by_embedding(self, query_embedding: np.ndarray,
                                    embedding_type: str = 'composite',
                                    top_k: int = 5,
                                    filter_room_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar spaces using embedding similarity"""
        if embedding_type not in self.embeddings:
            logger.error(f"Embedding type {embedding_type} not available")
            return []
        
        embeddings = self.embeddings[embedding_type]
        
        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(emb_norms, query_norm)
        
        # Apply room type filter if specified
        valid_indices = list(range(len(similarities)))
        if filter_room_type:
            filter_lower = filter_room_type.lower()
            valid_indices = [
                i for i in valid_indices
                if i < len(self.metadata) and 
                filter_lower in self.metadata[i].get('room_type', '').lower()
            ]
        
        # Get top-k from valid indices
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in valid_similarities[:top_k]]
        
        # Build results
        results = []
        for idx in top_indices:
            metadata = self.get_metadata_by_index(idx)
            if metadata:
                finnish_type = metadata.get('room_type', '')
                english_type = self.finnish_room_types.get(finnish_type, {}).get('english', finnish_type)
                
                results.append({
                    'index': idx,
                    'similarity': float(similarities[idx]),
                    'plan_id': metadata.get('plan_id'),
                    'room_type': english_type,
                    'finnish_type': finnish_type,
                    'original_text': metadata.get('text'),
                    'translated_text': metadata.get('translated'),
                    'function': metadata.get('function'),
                    'has_visual_features': metadata.get('has_visual_features', False),
                    'embedding': self.embeddings[embedding_type][idx],
                    'metadata': metadata
                })
        
        return results
    
    def get_room_adjacencies(self, plan_id: str) -> Optional[Dict[str, List[str]]]:
        """Get room adjacency information for a specific plan"""
        if plan_id not in self.stores:
            return None
        
        store = self.stores[plan_id]
        adjacencies = {}
        
        for ann in store.annotations:
            if ann.room_type and ann.adjacent_spaces:
                adjacencies[ann.room_type] = ann.adjacent_spaces
        
        return adjacencies
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded embeddings"""
        stats = {
            'total_annotations': len(self.metadata) if self.metadata else 0,
            'total_plans': len(set(m['plan_id'] for m in self.metadata)) if self.metadata else 0,
            'embedding_types': list(self.embeddings.keys()),
            'embedding_dimensions': {k: v.shape for k, v in self.embeddings.items()},
            'rag_stores_loaded': len(self.stores)
        }
        
        # Room type distribution
        if self.metadata:
            from collections import Counter
            room_types = Counter(m.get('room_type', 'unknown') for m in self.metadata)
            stats['room_type_distribution'] = dict(room_types.most_common(20))
        
        return stats
    
    def translate_finnish_room_type(self, finnish_type: str) -> str:
        """Translate Finnish room type to English"""
        return self.finnish_room_types.get(finnish_type.lower(), {}).get('english', finnish_type)