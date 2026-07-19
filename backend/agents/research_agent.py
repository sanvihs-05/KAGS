"""
Research Agent: Retrieves relevant knowledge from Finnish floor plans
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.fbsl_models import FBSLLayoutNode
from ..database.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class ResearchAgent:
    """Research Agent: Knowledge retrieval and enhancement"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        logger.info("✓ Research Agent initialized")
    
    def research_node(self, node: FBSLLayoutNode, depth: int = 3) -> Dict[str, Any]:
        """
        Conduct research to enhance FBSL node
        
        Args:
            node: FBSL node to research
            depth: Number of similar cases to retrieve
        
        Returns:
            Research findings and recommendations
        """
        logger.info(f"Researching node: {node.node_id[:8]}...")
        
        findings = {
            'similar_spaces': [],
            'room_precedents': {},
            'adjacency_patterns': {},
            'recommendations': []
        }
        
        # Only research if Finnish embeddings available
        if not self.vector_store.finnish_embeddings:
            logger.warning("Finnish embeddings not available for research")
            return findings
        
        # Research each function
        for func_id, func in node.functions.items():
            # Search for similar spaces using function embedding
            if func.embedding is not None:
                # Prefer FAISS client for fast inner-product (dot) search when available
                similar = []
                try:
                    faiss_client = getattr(self.vector_store, 'faiss_client', None)
                except Exception:
                    faiss_client = None

                if faiss_client is not None:
                    try:
                        # FaissClient.search accepts numpy vectors and returns {index, score, metadata}
                        raw_results = faiss_client.search(func.embedding, top_k=depth)
                        for r in raw_results:
                            meta = r.get('metadata') or {}
                            similar.append({
                                'index': r.get('index'),
                                'similarity': float(r.get('score') or 0.0),
                                'plan_id': meta.get('plan_id'),
                                'room_type': meta.get('room_type'),
                                'original_text': meta.get('text'),
                                'translated_text': meta.get('translated'),
                                'function': meta.get('function')
                            })
                    except Exception:
                        similar = []

                # Fallback to existing vector_store method which may use loader search
                if not similar:
                    similar = self.vector_store.search_similar_finnish_spaces(
                        func.embedding,
                        embedding_type='composite',
                        top_k=depth
                    )

                findings['similar_spaces'].extend(similar)
                findings['room_precedents'][func.name] = similar
        
        # Generate recommendations
        findings['recommendations'] = self._generate_recommendations(findings, node)
        
        logger.info(f"✓ Research complete: {len(findings['similar_spaces'])} precedents found")
        return findings
    
    def _generate_recommendations(self, findings: Dict, node: FBSLLayoutNode) -> List[Dict]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        # Recommend based on precedents
        for func_name, precedents in findings['room_precedents'].items():
            if precedents:
                avg_similarity = np.mean([p['similarity'] for p in precedents])
                if avg_similarity > 0.7:
                    recommendations.append({
                        'type': 'precedent',
                        'function': func_name,
                        'suggestion': f"High similarity to {len(precedents)} Finnish floor plans",
                        'priority': 'high',
                        'similarity': float(avg_similarity)
                    })
                elif avg_similarity > 0.5:
                    recommendations.append({
                        'type': 'precedent',
                        'function': func_name,
                        'suggestion': f"Moderate similarity to existing designs",
                        'priority': 'medium',
                        'similarity': float(avg_similarity)
                    })
        
        return recommendations
    
    def enhance_node_with_research(self, node: FBSLLayoutNode, 
                                   findings: Dict) -> FBSLLayoutNode:
        """Apply research findings to enhance node"""
        
        # Add research metadata
        node.metadata['research_findings'] = findings
        node.metadata['precedents_count'] = len(findings['similar_spaces'])
        
        # Store recommendations
        if 'recommended_adjacencies' not in node.metadata:
            node.metadata['recommended_adjacencies'] = []
        
        for rec in findings['recommendations']:
            if rec['priority'] == 'high':
                node.metadata['recommended_adjacencies'].append(rec)
        
        return node