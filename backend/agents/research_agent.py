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
    
    def reconcile_areas_with_precedents(
        self,
        node: FBSLLayoutNode,
        findings: Dict,
        lam: float = 0.6,
    ) -> int:
        """
        Deterministic reconciliation: ground each room's area in the retrieved
        precedents so generation is actually CONDITIONED on retrieval (RAG),
        not merely annotated by it.

        For each function whose precedents carry an 'area', blend the stated
        area with a similarity-weighted precedent estimate:

            â_precedent = Σ_i sim_i · area_i / Σ_i sim_i
            a* = lam · a_stated + (1 - lam) · â_precedent

        The result is CLAMPED to the brief's [min_area, max_area] band so
        reconciliation can nudge a room but never push the design outside the
        brief (the validator stays satisfied). No-op for rooms whose precedents
        expose no area. Returns the number of rooms adjusted.
        """
        room_precedents = (findings or {}).get('room_precedents', {})
        if not room_precedents or not node.layout or not node.layout.rooms:
            return 0

        # function_id -> its room  (rooms link back via function_id)
        room_by_func = {
            r.function_id: r for r in node.layout.rooms.values()
            if getattr(r, 'function_id', None)
        }

        adjusted = 0
        for func_id, func in node.functions.items():
            precedents = room_precedents.get(func.name, [])
            # similarity-weighted precedent area over precedents that expose one
            num, den = 0.0, 0.0
            for p in precedents:
                area = p.get('area')
                sim = float(p.get('similarity', 0.0) or 0.0)
                if area is not None and sim > 0:
                    num += sim * float(area)
                    den += sim
            if den <= 0:
                continue

            est = num / den
            room = room_by_func.get(func_id)
            if room is None:
                continue

            stated = float(room.area) if room.area else est
            blended = lam * stated + (1.0 - lam) * est

            # clamp to the brief band so the validator can never be broken
            sr = getattr(func, 'spatial_requirements', None) or {}
            lo = float(sr.get('min_area', blended * 0.5))
            hi = float(sr.get('max_area', blended * 1.5))
            new_area = float(min(max(blended, lo), hi))

            if abs(new_area - room.area) > 1e-6:
                room.area = new_area
                if isinstance(sr, dict):
                    sr['preferred_area'] = new_area
                adjusted += 1

        if adjusted:
            logger.info(f"✓ RAG reconciliation adjusted {adjusted} room area(s) from precedents")
        return adjusted

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

        # ✅ Ground room areas in precedents so generation depends on retrieval
        # (safe no-op when precedents expose no area; result clamped to brief band).
        try:
            n_adj = self.reconcile_areas_with_precedents(node, findings)
            node.metadata['rag_areas_reconciled'] = n_adj
        except Exception as e:
            logger.warning(f"RAG area reconciliation skipped: {e}")

        return node