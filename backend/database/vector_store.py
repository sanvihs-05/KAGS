import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.embedding_loader import FinnishFloorPlanEmbeddingLoader
from utils.faiss_client import FaissClient

class VectorStoreManager:
    def __init__(self, config_path: str = None):
        file_path = Path(__file__).resolve()
        if config_path is None:
            # Prefer package-local config (backend/config), fall back to top-level config/
            candidates = [
                file_path.parents[1] / "config" / "config.yaml",
                file_path.parents[2] / "config" / "config.yaml"
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Config file not found. Tried: {', '.join(str(p) for p in candidates)}"
                )
            config_path = found
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        
        vs_config = config['vector_store']
        
        # Initialize ChromaDB for new embeddings
        self.client = chromadb.Client(Settings(
            persist_directory=vs_config['persist_directory'],
            anonymized_telemetry=False
        ))
        
        # Initialize embedding model for new embeddings
        self.embedding_model = SentenceTransformer(
            vs_config['embedding_model']
        )
        
        # Load existing Finnish floor plan embeddings and optionally a FAISS index
        self.finnish_embeddings = None
        self.faiss_client = None
        if vs_config.get('existing_embeddings', {}).get('enabled', False):
            emb_config = vs_config['existing_embeddings']
            try:
                emb_path = emb_config['path']
                # Create a FaissClient that will lazily load/build the index
                self.faiss_client = FaissClient(embeddings_dir=emb_path,
                                                index_dir=emb_config.get('faiss_index_dir', 'data/faiss_index'),
                                                embedding_type=emb_config.get('embedding_type', 'composite'))

                # Try to load stats via the loader (FaissClient will initialize loader lazily)
                # Accessing loader to print statistics
                _loader = FinnishFloorPlanEmbeddingLoader(emb_path)
                stats = _loader.get_plan_statistics()
                self.finnish_embeddings = _loader
                print(f"[OK] Loaded Finnish floor plan embeddings:")
                print(f"  - {stats['total_annotations']} annotations")
                print(f"  - {stats['total_plans']} floor plans")
                print(f"  - {len(stats['embedding_types'])} embedding types")
                print(f"  - Top room types: {list(stats.get('room_type_distribution', {}).keys())[:5]}")

                # Ensure FAISS index exists (build if needed)
                try:
                    self.faiss_client.ensure_index(build_if_missing=True)
                except Exception as e:
                    print(f"[WARNING] FAISS index not available or failed to build: {e}")
                    # keep going; fallback searches will use the loader's in-memory search
            except Exception as e:
                print(f"[WARNING] Could not load Finnish embeddings: {e}")
        
        print("[OK] Vector store initialized")
    
    def search_finnish_rooms(self, room_type: str, 
                            embedding_type: str = 'composite',
                            top_k: int = 10) -> List[Dict[str, Any]]:
        """Search Finnish floor plans by room type"""
        if self.faiss_client is not None:
            # FAISS doesn't directly filter by room_type metadata; use the loader's search_by_room_type
            # for type-specific queries because it's fast enough for filtered queries.
            if self.finnish_embeddings is None:
                print("[WARNING] Finnish embeddings not loaded")
                return []
            return self.finnish_embeddings.search_by_room_type(
                room_type,
                embedding_type=embedding_type,
                top_k=top_k
            )

        if self.finnish_embeddings is None:
            print("[WARNING] Finnish embeddings not loaded")
            return []
        return self.finnish_embeddings.search_by_room_type(
            room_type,
            embedding_type=embedding_type,
            top_k=top_k
        )
    
    def search_similar_finnish_spaces(self, query_embedding: np.ndarray,
                                     embedding_type: str = 'composite',
                                     top_k: int = 5,
                                     filter_room_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar spaces in Finnish floor plans"""
        # Prefer FAISS search if available
        if self.faiss_client is not None:
            # If the caller provided a text query, FaissClient will encode it; if a vector,
            # FaissClient will accept the numpy array directly.
            try:
                results = self.faiss_client.search(query_embedding, top_k=top_k)
            except Exception:
                results = []

            # Post-filter by room type if requested and map to expected result schema
            filtered = []
            for r in results:
                meta = r.get('metadata') or {}
                if filter_room_type:
                    if meta and filter_room_type.lower() in (meta.get('room_type') or '').lower():
                        filtered.append({
                            'index': r['index'],
                            'similarity': r['score'],
                            'plan_id': meta.get('plan_id'),
                            'room_type': meta.get('room_type'),
                            'original_text': meta.get('text'),
                            'translated_text': meta.get('translated'),
                            'function': meta.get('function')
                        })
                else:
                    filtered.append({
                        'index': r['index'],
                        'similarity': r['score'],
                        'plan_id': meta.get('plan_id'),
                        'room_type': meta.get('room_type'),
                        'original_text': meta.get('text'),
                        'translated_text': meta.get('translated'),
                        'function': meta.get('function')
                    })

            return filtered[:top_k]

        # Fallback: use the in-memory loader search which accepts embeddings
        if self.finnish_embeddings is None:
            return []
        return self.finnish_embeddings.search_similar_by_embedding(
            query_embedding,
            embedding_type=embedding_type,
            top_k=top_k,
            filter_room_type=filter_room_type
        )
    
    def get_finnish_room_adjacencies(self, plan_id: str) -> Optional[Dict[str, List[str]]]:
        """Get room adjacencies for a Finnish floor plan"""
        if self.finnish_embeddings is None:
            return None
        
        return self.finnish_embeddings.get_room_adjacencies(plan_id)
    
    def translate_room_type(self, finnish_type: str) -> str:
        """Translate Finnish room type to English"""
        if self.finnish_embeddings is None:
            return finnish_type
        
        return self.finnish_embeddings.translate_finnish_room_type(finnish_type)
    
    # Keep existing methods...
    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: List[Dict], ids: List[str]):
        collection = self.get_or_create_collection(collection_name)
        embeddings = self.embedding_model.encode(documents).tolist()
        
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, collection_name: str, query_text: str, n_results: int = 5):
        collection = self.get_or_create_collection(collection_name)
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        return collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )