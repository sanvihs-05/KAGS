"""
Graph of Thoughts Engine for FBSL design space exploration.
"""

import uuid
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import networkx as nx

from .fbsl_models import (
    FBSLLayoutNode, Layout, Room, Function, Behavior, Structure,
    FunctionCategory, BehaviorCategory, StructureType, TransformationType
)

logger = logging.getLogger(__name__)


@dataclass
class ThoughtEdge:
    """Edge in the thought graph representing a transformation"""
    edge_id: str
    from_node_id: str
    to_node_id: str
    transformation_type: TransformationType
    cost_weight: float = 1.0
    quality_weight: float = 0.5


@dataclass
class GraphPath:
    """Path through the thought graph"""
    nodes: List[str]  # Node IDs
    edges: List[str]  # Edge IDs
    total_score: float
    avg_quality: float
    total_cost: float

class GraphOfThoughtsEngine:
    """
    Graph of Thought Engine for FBSL design space exploration
    
    Implements:
    - Node generation: f_node: P × C → N
    - Thought expansion: f_expand(n) = {n1, n2, ..., nk}
    - Path scoring: Score(path) = Σ w(ei) × q(ni)
    - Node aggregation: Aggregate(N1, N2, ..., Nk)
    """
    
    def __init__(self, max_depth: int = 3, breadth: int = 4, encoder: Optional[Any] = None):
        self.graph = nx.DiGraph()  # Directed graph for causality
        self.node_registry: Dict[str, FBSLLayoutNode] = {}  # Store actual nodes
        self.edge_registry: Dict[str, ThoughtEdge] = {}
        
        self.max_depth = max_depth
        self.breadth = breadth
        # Optional encoder agent (injected by orchestrator) used to
        # re-run layout extraction when a child node lacks `layout`.
        self.encoder = encoder
        
        # Scoring parameters
        self.alpha_compatibility = 0.4  # Weight for compatibility
        self.alpha_quality = 0.6  # Weight for quality
        
        logger.info(f"✓ GoT Engine initialized (depth={max_depth}, breadth={breadth})")
    
    async def generate_thought_graph(self, 
                                     problem_node: FBSLLayoutNode,
                                     expansion_strategies: List[str] = None,
                                     epsilon: float = 1e-3,
                                     delta: Optional[float] = None,
                                     patience: int = 2,
                                     max_nodes: Optional[int] = None) -> nx.DiGraph:
        """
        Generate complete thought graph from problem node
        
        Args:
            problem_node: Root problem node
            expansion_strategies: List of strategies to use
                ['functional', 'behavioral', 'structural', 'layout']
        
        Returns:
            NetworkX DiGraph with FBSL nodes and transformation edges
        """
        logger.info(f"Generating thought graph from problem: {problem_node.node_id[:8]}...")
        
        if expansion_strategies is None:
            expansion_strategies = ['functional', 'behavioral', 'structural', 'layout']
        
        # Add root node
        self._add_node_to_graph(problem_node)

        # BFS expansion with adaptive stopping
        queue = [(problem_node, 0)]  # (node, depth)
        nodes_generated = 0

        prev_best = float('-inf')
        stagnation = 0

        # Determine stopping delta (δ). If delta is provided, use it; otherwise use epsilon for backward compatibility.
        stop_delta = delta if delta is not None else epsilon

        while queue:
            current_node, depth = queue.pop(0)

            # Early stopping by depth
            if depth >= self.max_depth:
                logger.info(f"  Reached max depth {self.max_depth}")
                continue

            # Early stopping by node budget
            if max_nodes is not None and len(self.node_registry) >= max_nodes:
                logger.info(f"  Reached max_nodes={max_nodes}, stopping expansion")
                break

            logger.info(f"  Expanding node at depth {depth}: {current_node.node_id[:8]}...")

            # Expand node using strategies
            children = await self.expand_node(current_node, expansion_strategies)

            # Add top breadth children
            for child in children[:self.breadth]:
                # Add child node
                self._add_node_to_graph(child)
                
                # Add edge with transformation metadata
                edge = ThoughtEdge(
                    edge_id=str(uuid.uuid4()),
                    from_node_id=current_node.node_id,
                    to_node_id=child.node_id,
                    transformation_type=child.metadata.get('transformation_type', TransformationType.REFINEMENT),
                    cost_weight=1.0,
                    quality_weight=child.composite_score if child.composite_score > 0 else 0.5
                )
                self._add_edge_to_graph(edge)
                
                # Add to queue for further expansion
                queue.append((child, depth + 1))
                nodes_generated += 1

            # ✅ SCORE-BASED STOPPING: Check best composite score improvement
            # If high scores are coming and there's no improvement, stop exploring
            try:
                # Get all nodes with scores
                scored_nodes = [
                    n for n in self.node_registry.values() 
                    if hasattr(n, 'composite_score') and n.composite_score is not None and n.composite_score > 0
                ]
                
                if scored_nodes:
                    current_best = max(n.composite_score for n in scored_nodes)
                    # Also check if we have multiple high-scoring nodes (indicates good exploration)
                    high_scoring_count = sum(1 for n in scored_nodes if n.composite_score >= current_best * 0.9)
                else:
                    # No scores yet, use quality estimate
                    current_best = max(
                        (self._calculate_node_quality(n) for n in self.node_registry.values()),
                        default=0.0
                    )
                    high_scoring_count = 0
            except Exception:
                current_best = 0.0
                high_scoring_count = 0

            improvement = current_best - prev_best if prev_best != float('-inf') else float('inf')
            
            # ✅ IMPROVED: Stop if:
            # 1. No improvement AND we have high scores (scores plateaued)
            # 2. OR we have multiple high-scoring alternatives (good diversity achieved)
            should_stop = False
            
            if prev_best != float('-inf'):
                if improvement < stop_delta:
                    stagnation += 1
                    # If we have high scores and no improvement, that's a good stopping point
                    if current_best > 0.7:  # High score threshold
                        logger.info(
                            f"  High scores achieved ({current_best:.3f}) with no improvement "
                            f"(improvement={improvement:.6f}), count={stagnation}/{patience}"
                        )
                        # Stop faster if we have high scores
                        if stagnation >= max(1, patience - 1):
                            should_stop = True
                    else:
                        logger.info(f"  Stagnation detected (improvement={improvement:.6f}), count={stagnation}/{patience}")
                else:
                    stagnation = 0
                    logger.debug(f"  Score improvement: {improvement:.6f} (current best: {current_best:.3f})")
            
            # Also stop if we have multiple high-scoring alternatives (good exploration)
            if high_scoring_count >= 3 and current_best > 0.6:
                logger.info(
                    f"  Multiple high-scoring alternatives found ({high_scoring_count} nodes ≥ {current_best * 0.9:.3f}), "
                    f"stopping exploration"
                )
                should_stop = True

            prev_best = max(prev_best, current_best)

            if should_stop or stagnation >= patience:
                reason = "high scores with no improvement" if should_stop else f"no improvement < {stop_delta} for {patience} expansions"
                logger.info(f"  ✅ Score-based stopping triggered: {reason}")
                break
        
        logger.info(f"✓ Graph generation complete: {nodes_generated} nodes generated")
        logger.info(f"  Total nodes: {len(self.node_registry)}")
        logger.info(f"  Total edges: {len(self.edge_registry)}")
        
        return self.graph
    
    async def expand_node(self, 
                         node: FBSLLayoutNode,
                         strategies: List[str]) -> List[FBSLLayoutNode]:
        """
        Expand node using transformation strategies
        
        Implements: f_expand(n) = {n1, n2, ..., nk}
        """
        children = []
        
        for strategy in strategies:
            if strategy == 'functional':
                variants = await self._functional_decomposition(node)
                children.extend(variants)
            
            elif strategy == 'behavioral':
                variants = await self._behavioral_optimization(node)
                children.extend(variants)
            
            elif strategy == 'structural':
                variants = await self._structural_variation(node)
                children.extend(variants)
            
            elif strategy == 'layout':
                variants = await self._layout_permutation(node)
                children.extend(variants)
        
        logger.info(f"    Generated {len(children)} child nodes")
        return children
    
    async def _functional_decomposition(self, node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Decompose functions into sub-functions"""
        variants = []
        
        # Strategy: Split functions by priority
        high_priority_funcs = [f for f in node.functions.values() if f.priority > 0.7]
        
        if len(high_priority_funcs) >= 2:
            # Create variant focusing on subset of functions
            variant = self._create_child_node(node, TransformationType.FUNCTIONAL_DECOMPOSITION)

            # Keep only high priority functions
            variant.functions = {
                fid: f for fid, f in node.functions.items() 
                if f.priority > 0.7
            }

            # If filtering removed all functions, fall back to including the top-1 function
            # to avoid creating a node with no functions which later causes layout synthesis failure
            if not variant.functions and node.functions:
                # Pick top 1 by priority
                sorted_funcs = sorted(node.functions.items(), key=lambda kv: kv[1].priority, reverse=True)
                fid, f = sorted_funcs[0]
                variant.functions = {fid: f}
                logger.debug(f"  → Functional decomposition would be empty; falling back to top function: {f.name}")
            
            # Update behaviors accordingly
            variant.behaviors = {
                bid: b for bid, b in node.behaviors.items()
                if b.derived_from_function in variant.functions
            }
            # If no behaviors matched (possible when functions were restored above), ensure at least one
            if not variant.behaviors and variant.functions:
                # Create simple area behaviors for each function using a conservative default
                from .fbsl_models import Behavior, BehaviorCategory
                for fid, func in variant.functions.items():
                    if not any(b.derived_from_function == fid for b in variant.behaviors.values()):
                        target = 10.0
                        if getattr(func, 'spatial_requirements', None) and isinstance(func.spatial_requirements, dict):
                            target = func.spatial_requirements.get('preferred_area', target)
                        beh = Behavior(
                            category=BehaviorCategory.SPATIAL,
                            metric_name=f"{func.name}_area",
                            metric_unit="sqm",
                            target_value=float(target),
                            derived_from_function=fid,
                            tolerance=0.2
                        )
                        variant.add_behavior(beh)
            
            variant.metadata['description'] = 'Focus on high-priority functions'
            variants.append(variant)
        
        return variants
    
    async def _behavioral_optimization(self, node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Optimize behavior targets"""
        variants = []
        
        # Strategy: Relax some behaviors to improve overall satisfaction
        if node.behaviors:
            variant = self._create_child_node(node, TransformationType.BEHAVIORAL_OPTIMIZATION)
            
            # Relax behaviors with low satisfaction
            for behav in variant.behaviors.values():
                if behav.actual_value and behav.target_value:
                    if not behav.is_satisfied:
                        behav.tolerance *= 1.3  # Increase tolerance
            
            variant.metadata['description'] = 'Relaxed behavior tolerances'
            variants.append(variant)
        
        return variants
    
    async def _structural_variation(self, node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Generate structural variations"""
        variants = []
        
        # Strategy: Use alternative materials
        if node.structures:
            variant = self._create_child_node(node, TransformationType.STRUCTURAL_VARIATION)
            
            # Modify materials (simplified)
            for struct in variant.structures.values():
                if struct.material_type == 'gypsum_board':
                    struct.material_type = 'concrete'
                elif struct.material_type == 'concrete':
                    struct.material_type = 'brick'
            
            variant.metadata['description'] = 'Alternative material system'
            variants.append(variant)
        
        return variants
    
    async def _layout_permutation(self, node: FBSLLayoutNode) -> List[FBSLLayoutNode]:
        """Generate layout permutations"""
        variants = []
        
        # Strategy: Different spatial arrangements
        variant = self._create_child_node(node, TransformationType.LAYOUT_PERMUTATION)
        variant.metadata['description'] = 'Alternative spatial arrangement'
        variant.metadata['layout_strategy'] = 'compact' if len(variants) % 2 == 0 else 'linear'
        
        variants.append(variant)
        
        return variants
    
    def _create_child_node(self, parent: FBSLLayoutNode, 
                          transformation: TransformationType) -> FBSLLayoutNode:
        """
        Create child node as copy of parent
        
        ✅ CRITICAL FIX: Ensures child nodes always have valid layouts with rooms
        """
        import copy
        
        child = copy.deepcopy(parent)
        
        # ✅ IMPROVED: Check if child has valid layout with rooms
        def has_valid_layout(node):
            """Check if node has a valid layout with at least one room"""
            try:
                if not node.layout:
                    return False
                rooms = getattr(node.layout, 'rooms', None)
                if rooms is None:
                    return False
                # Check if rooms is a dict with at least one item
                if isinstance(rooms, dict):
                    return len(rooms) > 0
                # If it's a list or other iterable
                if hasattr(rooms, '__len__'):
                    return len(rooms) > 0
                return False
            except Exception:
                return False
        
        child_has_valid_layout = has_valid_layout(child)
        parent_has_valid_layout = has_valid_layout(parent)

        # Step 1: Try to preserve parent's layout if child lost it
        if not child_has_valid_layout and parent_has_valid_layout:
            try:
                child.layout = copy.deepcopy(parent.layout)
                # Verify the copy worked
                if has_valid_layout(child):
                    logger.debug(f"  → Preserved parent layout in child node")
                else:
                    # Deep copy failed, try shallow copy
                    child.layout = parent.layout
                    if has_valid_layout(child):
                        logger.debug(f"  → Preserved parent layout (shallow copy)")
            except Exception as e:
                logger.warning(f"  → Failed to copy parent layout: {e}")
                # Best-effort: shallow copy fallback
                try:
                    child.layout = parent.layout
                except Exception:
                    pass

        # Step 2: Re-check after copy attempt
        child_has_valid_layout = has_valid_layout(child)

        # Step 3: If still no layout, try to synthesize from functions immediately
        if not child_has_valid_layout:
            # Try to synthesize layout from functions using layout agent
            try:
                from ..agents.layout_agent import LayoutGenerationAgent
                layout_agent = LayoutGenerationAgent()
                
                # Build layout from functions synchronously (we need it now)
                synthesized_layout = layout_agent._build_layout_from_functions(child)
                
                # Check if synthesized layout has rooms
                if synthesized_layout:
                    try:
                        rooms = getattr(synthesized_layout, 'rooms', None)
                        if rooms and ((isinstance(rooms, dict) and len(rooms) > 0) or (hasattr(rooms, '__len__') and len(rooms) > 0)):
                            child.layout = synthesized_layout
                            room_count = len(rooms) if isinstance(rooms, dict) else len(rooms)
                            logger.debug(f"  → Synthesized layout from functions for child node: {room_count} rooms")
                            child_has_valid_layout = True
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"  → Layout synthesis from functions failed: {e}")

        # Step 4: If still no layout, try re-running encoder (last resort)
        if not child_has_valid_layout and getattr(self, 'encoder', None) is not None:
            req_text = None
            try:
                req_text = child.metadata.get('original_requirements')
            except Exception:
                pass

            if not req_text:
                try:
                    req_text = parent.metadata.get('original_requirements')
                except Exception:
                    pass

            if req_text:
                try:
                    produced_node = self.encoder.encode_requirements(req_text)
                    if produced_node and has_valid_layout(produced_node):
                        try:
                            child.layout = copy.deepcopy(produced_node.layout)
                            logger.debug(f"  → Re-ran encoder to get layout for child node")
                            child_has_valid_layout = True
                        except Exception:
                            child.layout = produced_node.layout
                            child_has_valid_layout = has_valid_layout(child)
                except Exception as e:
                    logger.debug(f"  → Encoder re-run failed: {e}")

        # Step 5: Final fallback - create minimal layout from first function
        if not child_has_valid_layout:
            try:
                from ..core.fbsl_models import Layout, Room
                if not child.layout:
                    child.layout = Layout()
                if not child.layout.rooms:
                    child.layout.rooms = {}
                
                # Create at least one room from first available function
                if child.functions:
                    func = list(child.functions.values())[0]
                    area = 12.0
                    if hasattr(func, 'spatial_requirements') and isinstance(func.spatial_requirements, dict):
                        area = func.spatial_requirements.get('preferred_area', 12.0)
                    
                    room = Room(
                        name=func.name.replace('provide_', '').replace('_', ' ').title() if hasattr(func, 'name') else "Space",
                        room_type=func.name.replace('provide_', '') if hasattr(func, 'name') else "space",
                        room_number="1",
                        function_id=func.function_id,
                        area=area,
                        height=3.0
                    )
                    room.calculate_volume()
                    child.layout.rooms[room.room_id] = room
                    child.layout.total_area = area
                    child.layout.used_area = area
                    child.layout.calculate_metrics()
                    logger.debug(f"  → Created minimal fallback layout: 1 room ({area} m²)")
            except Exception as e:
                logger.warning(f"  → Final fallback layout creation failed: {e}")

        child.node_id = str(uuid.uuid4())
        child.parent_node_id = parent.node_id
        child.generation_level = parent.generation_level + 1
        child.metadata['transformation_type'] = transformation
        
        # ✅ FINAL VALIDATION: Log warning if child still has no valid layout
        if not has_valid_layout(child):
            logger.warning(f"  ⚠ Child node {child.node_id[:8]} created without valid layout!")
        else:
            room_count = len(child.layout.rooms) if child.layout and child.layout.rooms else 0
            logger.debug(f"  ✓ Child node {child.node_id[:8]} created with {room_count} rooms")
        
        return child
    
    def find_best_paths(self, 
                       from_node_id: str,
                       to_node_id: Optional[str] = None,
                       top_k: int = 5) -> List[GraphPath]:
        """
        Find best paths through the graph
        
        Implements: Score(path) = Σ w(ei) × q(ni)
        
        Args:
            from_node_id: Starting node
            to_node_id: Target node (None = all leaf nodes)
            top_k: Number of best paths to return
        
        Returns:
            List of best paths sorted by score
        """
        logger.info(f"Finding best paths from {from_node_id[:8]}...")
        
        if to_node_id:
            # Single target
            targets = [to_node_id]
        else:
            # All leaf nodes
            targets = [
                nid for nid in self.node_registry.keys()
                if self.graph.out_degree(nid) == 0
            ]
        
        all_paths = []
        
        for target in targets:
            if not nx.has_path(self.graph, from_node_id, target):
                continue
            
            # Get all simple paths
            paths = nx.all_simple_paths(self.graph, from_node_id, target, cutoff=self.max_depth)
            
            for path_nodes in paths:
                # Calculate path score
                path_score = self._calculate_path_score(path_nodes)
                all_paths.append(path_score)
        
        # Sort by total score
        all_paths.sort(key=lambda p: p.total_score, reverse=True)
        
        logger.info(f"✓ Found {len(all_paths)} paths, returning top {top_k}")
        return all_paths[:top_k]
    
    def _calculate_path_score(self, node_ids: List[str]) -> GraphPath:
        """
        Calculate score for a path
        
        Score(path) = Σ w(ei) × q(ni)
        """
        total_score = 0.0
        total_cost = 0.0
        qualities = []
        edges = []
        
        for i in range(len(node_ids) - 1):
            from_id = node_ids[i]
            to_id = node_ids[i + 1]
            
            # Find edge
            edge_data = self.graph.get_edge_data(from_id, to_id)
            if edge_data:
                edge_id = edge_data.get('edge_id')
                if edge_id and edge_id in self.edge_registry:
                    edge = self.edge_registry[edge_id]
                    edges.append(edge_id)
                    
                    # Get node quality
                    node = self.node_registry.get(to_id)
                    node_quality = node.composite_score if node else 0.5
                    qualities.append(node_quality)
                    
                    # Calculate contribution: w(e) × q(n)
                    contribution = edge.cost_weight * node_quality
                    total_score += contribution
                    total_cost += edge.cost_weight
        
        avg_quality = np.mean(qualities) if qualities else 0.0
        
        return GraphPath(
            nodes=node_ids,
            edges=edges,
            total_score=total_score,
            avg_quality=avg_quality,
            total_cost=total_cost
        )
    
    def aggregate_nodes(self, node_ids: List[str], top_k: int = 3, score_threshold: Optional[float] = None,
                        compatibility_threshold: float = 0.5, selection_metric: str = 'composite') -> FBSLLayoutNode:
        """
        Aggregate multiple nodes into one by combining several high-quality candidates.

        Strategy:
        - Compute pairwise compatibility and node qualities.
        - Select a set of high-scoring nodes (top_k or those above score_threshold).
        - Use the node with highest aggregate score as base, and merge compatible
          components from other high-scoring nodes when compatibility > compatibility_threshold.

        Args:
            node_ids: List of node IDs to consider
            top_k: Number of top-quality nodes to attempt to combine
            score_threshold: Optional absolute threshold to select high-quality nodes
            compatibility_threshold: Minimum compatibility to merge components

        Returns:
            Aggregated FBSL node
        """
        logger.info(f"Aggregating {len(node_ids)} nodes with multi-merge strategy (top_k={top_k})...")

        nodes = [self.node_registry[nid] for nid in node_ids if nid in self.node_registry]
        if not nodes:
            raise ValueError("No valid nodes to aggregate")

        n = len(nodes)

        # Pairwise compatibility
        compatibility_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    compatibility_matrix[i, j] = self._calculate_compatibility_complete(nodes[i], nodes[j])

        # Quality per node
        qualities = [self._calculate_node_quality(node) for node in nodes]

        # Aggregate score per node (same formula as before) - used to pick base
        aggregate_scores = []
        for i in range(n):
            score = 0.0
            for j in range(n):
                if i != j:
                    lambda_i = 1.0 / max(1, (n - 1))
                    score += lambda_i * compatibility_matrix[i, j] * qualities[i]
            aggregate_scores.append(score)

        # Choose selection scores according to requested metric
        if selection_metric == 'composite':
            # prefer node.composite_score, fall back to quality if composite is not set
            selection_scores = np.array([getattr(node, 'composite_score', 0.0) or qualities[i] for i, node in enumerate(nodes)])
        else:
            # selection_metric == 'quality' or any other -> use computed quality
            selection_scores = np.array(qualities)

        # Identify high-scoring indices based on selection_scores
        sorted_idx = np.argsort(selection_scores)[::-1]
        high_indices = list(sorted_idx[:min(top_k, n)])
        if score_threshold is not None:
            above = [i for i, s in enumerate(selection_scores) if s >= score_threshold]
            high_indices = list(dict.fromkeys(high_indices + above))

        # Choose base node as argmax aggregate score
        best_idx = int(np.argmax(aggregate_scores))
        if best_idx not in high_indices:
            high_indices.insert(0, best_idx)

        best_node = nodes[best_idx]
        logger.info(f"  Selected base node index {best_idx} (quality={qualities[best_idx]:.3f}, aggregate={aggregate_scores[best_idx]:.3f})")

        # Create aggregated node from base
        aggregated = self._create_child_node(best_node, TransformationType.AGGREGATION)

        merged_from = [nodes[idx].node_id for idx in high_indices]

        # Merge compatible elements from other high-scoring nodes
        for i in high_indices:
            if i == best_idx:
                continue
            node = nodes[i]
            compat = compatibility_matrix[best_idx, i]
            if compat > compatibility_threshold:
                # Merge functions
                for func_id, func in node.functions.items():
                    if func_id not in aggregated.functions:
                        if self._is_function_compatible(func, aggregated):
                            aggregated.functions[func_id] = func
                            logger.info(f"    Merged function: {func.name} (compat={compat:.2f})")

                # Merge behaviors
                for behav_id, behav in node.behaviors.items():
                    if behav_id not in aggregated.behaviors:
                        aggregated.behaviors[behav_id] = behav

                # Merge structures
                for struct_id, struct in node.structures.items():
                    if struct_id not in aggregated.structures:
                        aggregated.structures[struct_id] = struct

        # Metadata updates
        aggregated.metadata['aggregated_from'] = merged_from
        aggregated.metadata['aggregate_scores'] = aggregate_scores
        aggregated.metadata['compatibility_matrix'] = compatibility_matrix.tolist()
        aggregated.metadata['base_quality'] = float(qualities[best_idx])
        aggregated.metadata['base_composite_score'] = float(getattr(best_node, 'composite_score', 0.0) or 0.0)
        aggregated.metadata['selection_metric'] = selection_metric
        aggregated.metadata['description'] = (
            f"Aggregated from {len(merged_from)} high-quality nodes (selection_metric={selection_metric}) "
            f"(base_quality={qualities[best_idx]:.3f}, base_composite={aggregated.metadata['base_composite_score']:.3f})"
        )

        return aggregated

    def _calculate_compatibility_complete(self, node1: FBSLLayoutNode, 
                                          node2: FBSLLayoutNode) -> float:
        """
        Calculate complete compatibility between two nodes
        
        Implements: Compatibility(Nᵢ, Nⱼ) = 1 - (|Conflict(Lᵢ, Lⱼ)| / |Lᵢ ∪ Lⱼ|)
        
        Extended to consider:
        - Function conflicts
        - Behavior conflicts
        - Structure conflicts
        - Layout conflicts
        """
        total_conflicts = 0
        total_elements = 0
        
        # 1. Function conflicts
        func1_ids = set(node1.functions.keys())
        func2_ids = set(node2.functions.keys())
        union_funcs = func1_ids | func2_ids
        
        if union_funcs:
            total_elements += len(union_funcs)
            
            # Check explicit conflicts
            for func1 in node1.functions.values():
                for func2 in node2.functions.values():
                    if func2.function_id in func1.conflicts_with:
                        total_conflicts += 1
                    # Check if activities conflict (e.g., "privacy" vs "social")
                    if self._activities_conflict(func1.activities, func2.activities):
                        total_conflicts += 0.5  # Soft conflict
        
        # 2. Behavior conflicts (incompatible targets)
        behav1_ids = set(node1.behaviors.keys())
        behav2_ids = set(node2.behaviors.keys())
        common_behavs = behav1_ids & behav2_ids
        
        for behav_id in common_behavs:
            b1 = node1.behaviors[behav_id]
            b2 = node2.behaviors[behav_id]
            
            total_elements += 1
            
            # Check if target values are incompatible
            if b1.target_value and b2.target_value:
                deviation = abs(b1.target_value - b2.target_value) / max(b1.target_value, 0.001)
                if deviation > 0.5:  # More than 50% difference
                    total_conflicts += 1
        
        # 3. Structure conflicts (incompatible materials/systems)
        if node1.structures and node2.structures:
            total_elements += 1
            
            # Check for conflicting structural systems
            materials1 = set(s.material_type for s in node1.structures.values())
            materials2 = set(s.material_type for s in node2.structures.values())
            
            # Example: concrete and wood are conflicting primary materials
            conflicting_pairs = [('concrete', 'wood'), ('steel', 'timber')]
            for mat1 in materials1:
                for mat2 in materials2:
                    for conflict in conflicting_pairs:
                        if (mat1 in conflict[0] and mat2 in conflict[1]) or \
                           (mat1 in conflict[1] and mat2 in conflict[0]):
                            total_conflicts += 0.3  # Partial conflict
        
        # 4. Layout conflicts (spatial incompatibility)
        if node1.layout and node2.layout:
            L1 = node1.layout
            L2 = node2.layout
            
            total_elements += 1
            
            # Check area compatibility
            if L1.total_area > 0 and L2.total_area > 0:
                area_diff = abs(L1.total_area - L2.total_area)
                avg_area = (L1.total_area + L2.total_area) / 2
                area_deviation = area_diff / avg_area
                
                if area_deviation > 0.3:  # More than 30% difference
                    total_conflicts += area_deviation
            
            # Check room count compatibility
            if L1.rooms and L2.rooms:
                count_diff = abs(len(L1.rooms) - len(L2.rooms))
                if count_diff > 2:
                    total_conflicts += 0.5
        
        # Calculate final compatibility
        if total_elements == 0:
            return 1.0  # No elements to conflict
        
        compatibility = 1.0 - (total_conflicts / total_elements)
        return max(0.0, min(1.0, compatibility))  # Clamp to [0, 1]

    def _activities_conflict(self, activities1: List[str], activities2: List[str]) -> bool:
        """Check if activities conflict"""
        conflicting_pairs = [
            ('privacy', 'social'),
            ('quiet', 'entertainment'),
            ('rest', 'work'),
            ('storage', 'circulation')
        ]
        
        for act1 in activities1:
            for act2 in activities2:
                for conflict in conflicting_pairs:
                    if (act1 in conflict[0] and act2 in conflict[1]) or \
                       (act1 in conflict[1] and act2 in conflict[0]):
                        return True
        return False

    def _calculate_node_quality(self, node: FBSLLayoutNode) -> float:
        """
        Calculate comprehensive node quality
        
        Considers:
        - FBSL completeness
        - Behavior satisfaction rate
        - Composite score
        """
        quality_components = []
        
        # 1. FBSL completeness (does it have all components?)
        completeness = 0.0
        if node.functions:
            completeness += 0.25
        if node.behaviors:
            completeness += 0.25
        if node.structures:
            completeness += 0.25
        if node.layout:
            completeness += 0.25
        quality_components.append(completeness)
        
        # 2. Behavior satisfaction rate
        if node.behaviors:
            satisfied = sum(1 for b in node.behaviors.values() 
                            if b.is_satisfied and b.behavior_type.value == 'expected')
            total = sum(1 for b in node.behaviors.values() 
                        if b.behavior_type.value == 'expected')
            satisfaction_rate = satisfied / max(total, 1)
            quality_components.append(satisfaction_rate)
        else:
            quality_components.append(0.5)
        
        # 3. Existing composite score
        if node.composite_score > 0:
            quality_components.append(node.composite_score)
        else:
            quality_components.append(0.5)
        
        # 4. Function-behavior consistency
        consistency = 0.0
        if node.functions and node.behaviors:
            funcs_with_behaviors = sum(
                1 for f in node.functions.values()
                if any(b.derived_from_function == f.function_id for b in node.behaviors.values())
            )
            consistency = funcs_with_behaviors / len(node.functions)
        quality_components.append(consistency)
        
        # Weighted average
        weights = [0.2, 0.3, 0.3, 0.2]
        quality = sum(w * q for w, q in zip(weights, quality_components))
        
        return quality
    
    def _calculate_compatibility(self, node1: FBSLLayoutNode, node2: FBSLLayoutNode) -> float:
        """
        Calculate compatibility between two nodes
        
        Implements: Compatibility(Ni, Nj) = 1 - (|Conflict(Li, Lj)| / |Li ∪ Lj|)
        
        Simplified: Check function and behavior conflicts
        """
        conflicts = 0
        total_elements = 0
        
        # Check function conflicts
        for func1 in node1.functions.values():
            total_elements += 1
            for func2 in node2.functions.values():
                # Check if functions conflict
                if func2.function_id in func1.conflicts_with:
                    conflicts += 1
        
        # Check layout conflicts (if layouts exist)
        if node1.layout and node2.layout:
            # Check if total areas are compatible
            area_diff = abs(node1.layout.total_area - node2.layout.total_area)
            avg_area = (node1.layout.total_area + node2.layout.total_area) / 2
            
            if avg_area > 0:
                area_conflict = area_diff / avg_area
                if area_conflict > 0.3:  # More than 30% difference
                    conflicts += 1
            
            total_elements += 1
        
        if total_elements == 0:
            return 1.0
        
        compatibility = 1.0 - (conflicts / total_elements)
        return max(0.0, compatibility)
    
    def _is_function_compatible(self, func, node: FBSLLayoutNode) -> bool:
        """Check if function is compatible with node"""
        # Check conflicts
        for existing_func in node.functions.values():
            if existing_func.function_id in func.conflicts_with:
                return False
            if func.function_id in existing_func.conflicts_with:
                return False
        
        return True
    
    def _add_node_to_graph(self, node: FBSLLayoutNode):
        """Add node to graph and registry"""
        self.graph.add_node(node.node_id, node_type=node.node_type.value)
        self.node_registry[node.node_id] = node
        # Log layout presence for debugging empty-room issues
        try:
            room_count = len(node.layout.rooms) if (node.layout and getattr(node.layout, 'rooms', None)) else 0
        except Exception:
            room_count = 0
        logger.debug(f"  → Node added to graph: {node.node_id[:8]} (rooms={room_count}, funcs={len(node.functions)})")
    
    def _add_edge_to_graph(self, edge: ThoughtEdge):
        """Add edge to graph and registry"""
        self.graph.add_edge(
            edge.from_node_id,
            edge.to_node_id,
            edge_id=edge.edge_id,
            transformation=edge.transformation_type.value,
            weight=edge.cost_weight
        )
        self.edge_registry[edge.edge_id] = edge
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the thought graph"""
        stats = {
            'total_nodes': len(self.node_registry),
            'total_edges': len(self.edge_registry),
            'max_depth': self.max_depth,
            'graph_depth': 0,
            'branching_factor': 0,
            'leaf_nodes': 0,
            'transformation_types': {}
        }
        
        if self.graph.number_of_nodes() > 0:
            # Calculate actual depth
            root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
            if root_nodes:
                root = root_nodes[0]
                paths = [nx.shortest_path_length(self.graph, root, n) 
                        for n in self.graph.nodes() if nx.has_path(self.graph, root, n)]
                stats['graph_depth'] = max(paths) if paths else 0
            
            # Calculate branching factor
            out_degrees = [self.graph.out_degree(n) for n in self.graph.nodes()]
            stats['branching_factor'] = np.mean([d for d in out_degrees if d > 0]) if out_degrees else 0
            
            # Count leaf nodes
            stats['leaf_nodes'] = sum(1 for n in self.graph.nodes() if self.graph.out_degree(n) == 0)
        
        return stats
    
    def visualize_graph(self, output_path: str = "thought_graph.png"):
        """Visualize the thought graph using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Use hierarchical layout
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # Color nodes by generation level
            colors = []
            for node_id in self.graph.nodes():
                node = self.node_registry.get(node_id)
                if node:
                    colors.append(node.generation_level)
                else:
                    colors.append(0)
            
            # Draw
            nx.draw(self.graph, pos, 
                   node_color=colors,
                   node_size=500,
                   cmap=plt.cm.viridis,
                   with_labels=False,
                   arrows=True,
                   edge_color='gray',
                   alpha=0.7)
            
            # Add labels with node IDs
            labels = {nid: nid[:8] for nid in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            plt.title("FBSL Thought Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"✓ Graph visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for visualization")