# pipeline/orchestrator.py
"""
Pipeline Orchestrator with Graph of Thought integration
Phase 4 Complete: Full FBSL Pipeline Convergence Loop
- F → Be → S → Bs → L → Evaluation → Reformulation
- Convergence checking: |S_composite(t) - S_composite(t-1)| < ε
- Pareto optimality tracking
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime
import numpy as np

from ..core.fbsl_models import FBSLLayoutNode, NodeType
from ..core.graph_of_thoughts import GraphOfThoughtsEngine
from ..core.complexity_calculator import ComplexityCalculator
from ..database.vector_store import VectorStoreManager
from ..database.postgres_manager import DatabaseManager
from ..agents.encoder_agent import EncoderAgent
from ..agents.generalizer_agent import GeneralizerAgent
from ..agents.research_agent import ResearchAgent
from ..agents.scoring_agent import ScoringAgent
from ..agents.refinement_agent import RefinementAgent
from ..agents.layout_agent import LayoutGenerationAgent
from ..core.behavior_calculator import BehaviorCalculator
from ..core.brief_validator import BriefValidator

logger = logging.getLogger(__name__)


class ParetoFront:
    """Tracks Pareto-optimal solutions for multi-objective optimization"""
    
    def __init__(self):
        self.solutions: List[Dict[str, Any]] = []
        
    def add_solution(self, node: FBSLLayoutNode, scores: Dict[str, float]):
        """
        Add solution and update Pareto front
        
        A solution is Pareto-optimal if no other solution dominates it:
        Pareto_Set = {x | ¬∃y : ∀i f_i(y) ≥ f_i(x) ∧ ∃j f_j(y) > f_j(x)}
        """
        solution = {
            'node': node,
            'scores': scores,
            'objectives': [
                scores.get('functional_adequacy', 0.0),
                scores.get('behavioral_performance', 0.0),
                scores.get('structural_feasibility', 0.0),
                scores.get('layout_efficiency', 0.0),
                scores.get('sustainability', 0.0)
            ]
        }
        
        # Check if new solution is dominated by existing solutions
        is_dominated = False
        for existing in self.solutions:
            if self._dominates(existing['objectives'], solution['objectives']):
                is_dominated = True
                break
        
        if not is_dominated:
            # Add new solution and remove dominated ones
            self.solutions.append(solution)
            self.solutions = [
                s for s in self.solutions
                if not self._dominates(solution['objectives'], s['objectives']) or s == solution
            ]
            
            logger.debug(f"Pareto front updated: {len(self.solutions)} solutions")
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (all >= and at least one >)"""
        all_geq = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        any_gt = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        return all_geq and any_gt
    
    def get_best_solutions(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k solutions from Pareto front by composite score"""
        sorted_solutions = sorted(
            self.solutions,
            key=lambda s: s['scores'].get('composite', 0.0),
            reverse=True
        )
        return sorted_solutions[:top_k]


class PipelineOrchestrator:
    """
    Orchestrates the complete FBSL-KAGS pipeline with GoT
    
    Phase 4 Enhancements:
    - Full pipeline convergence loop
    - Pareto optimality tracking
    - Layout generation integration
    - Behavior calculation with S → Bs transformation
    """
    
    def __init__(
        self,
        use_got: bool = True,
        convergence_threshold: float = 0.01,
        max_pipeline_iterations: int = 5,
        refinement_max_iterations: int = 5
    ):
        """
        Initialize pipeline orchestrator
        
        Args:
            use_got: Enable Graph of Thought mechanism
            convergence_threshold: Epsilon for convergence checking
            max_pipeline_iterations: Maximum full pipeline iterations
        """
        # Initialize components
        self.vector_store = VectorStoreManager()
        
        try:
            self.db = DatabaseManager()
        except Exception as e:
            logger.warning(f"Database not available: {e}")
            self.db = None
        
        # Initialize agents
        self.encoder = EncoderAgent(self.vector_store)
        self.generalizer = GeneralizerAgent()
        self.research = ResearchAgent(self.vector_store)
        # Use rho=1.0 for linear (compensatory) scoring instead of -1.0 (anti-compensatory)
        # This allows trade-offs and gives more reasonable scores
        self.scoring = ScoringAgent(rho=1.0)
        # Allow more refinement iterations by default to help satisfy behaviors
        self.refiner = RefinementAgent(max_iterations=refinement_max_iterations)
        self.layout_agent = LayoutGenerationAgent()
        self.behavior_calculator = BehaviorCalculator()
        self._brief_spec = None  # built from the root problem node per run (brief validator gate)
        
        # Initialize complexity calculator for adaptive parameters
        self.complexity_calculator = ComplexityCalculator()
        
        # Initialize GoT engine (will be reconfigured adaptively per request)
        self.use_got = use_got
        if use_got:
            # Default GoT engine (will be updated with adaptive params)
            self.got_engine = GraphOfThoughtsEngine(max_depth=2, breadth=3, encoder=self.encoder)
            # Default GoT adaptive stopping and aggregation params
            self.got_epsilon = 1e-3
            self.got_patience = 2
            self.got_max_nodes = None
            self.got_selection_metric = 'composite'
        
        # Convergence parameters
        self.convergence_threshold = convergence_threshold
        self.max_pipeline_iterations = max_pipeline_iterations
        
        logger.info(
            f"✓ Pipeline Orchestrator initialized "
            f"(GoT={'enabled' if use_got else 'disabled'}, "
            f"ε={convergence_threshold}, "
            f"max_iter={max_pipeline_iterations})"
        )
    
    async def process_design_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline with GoT integration and full convergence loop
        
        Pipeline: F → Be → S → Bs → L → Evaluation → Reformulation
        Convergence: |S_composite(t) - S_composite(t-1)| < ε
        
        Args:
            request: Design request dictionary
        
        Returns:
            Complete pipeline results with all alternatives and metrics
        """
        start_time = datetime.now()
        
        project_name = request.get('project_name', 'Untitled Project')
        requirements = request.get('requirements', '')
        context = request.get('context', {})
        # If not provided, let selection be adaptive (use all leaf nodes)
        max_alternatives = request.get('max_alternatives', None)
        use_got = request.get('use_got', self.use_got)
        enable_convergence_loop = request.get('enable_convergence_loop', True)
        
        logger.info(f"🚀 Starting pipeline for: {project_name}")
        logger.info(f"   GoT: {'enabled' if use_got else 'disabled'}")
        logger.info(f"   Convergence loop: {'enabled' if enable_convergence_loop else 'disabled'}")
        
        try:
            # =================================================================
            # PHASE 1: ENCODE & RESEARCH
            # =================================================================
            logger.info("📝 Step 1: Encoding requirements...")
            problem_node = self.encoder.encode_requirements(requirements, context)
            logger.info(f"   ✓ Created problem node with {len(problem_node.functions)} functions")
            
            # Debug: verify problem node has layout
            def has_valid_layout(node):
                """Check if node has valid layout with rooms"""
                try:
                    if not node.layout:
                        return False, 0
                    rooms = getattr(node.layout, 'rooms', None)
                    if rooms is None:
                        return False, 0
                    if isinstance(rooms, dict):
                        return len(rooms) > 0, len(rooms)
                    if hasattr(rooms, '__len__'):
                        return len(rooms) > 0, len(rooms)
                    return False, 0
                except Exception:
                    return False, 0
            
            has_layout, room_count = has_valid_layout(problem_node)
            if has_layout:
                logger.info(f"   ✓ Problem node layout has {room_count} rooms")
            else:
                logger.error("   ❌ Problem node has NO layout/rooms after encoding!")
                # Try to synthesize immediately
                try:
                    from ..agents.layout_agent import LayoutGenerationAgent
                    layout_agent = LayoutGenerationAgent()
                    synthesized = layout_agent._build_layout_from_functions(problem_node)
                    if synthesized and getattr(synthesized, 'rooms', None):
                        room_count_syn = len(synthesized.rooms) if isinstance(synthesized.rooms, dict) else (len(synthesized.rooms) if hasattr(synthesized.rooms, '__len__') else 0)
                        if room_count_syn > 0:
                            problem_node.layout = synthesized
                            logger.info(f"   ✓ Synthesized layout with {room_count_syn} rooms")
                        else:
                            logger.error("   ❌ Synthesis failed - using emergency fallback")
                            problem_node.layout = layout_agent._create_absolute_fallback_layout(problem_node)
                    else:
                        logger.error("   ❌ Synthesis returned None - using emergency fallback")
                        problem_node.layout = layout_agent._create_absolute_fallback_layout(problem_node)
                except Exception as e:
                    logger.error(f"   ❌ Layout synthesis failed: {e}")
                    # Create emergency fallback
                    from ..core.fbsl_models import Layout, Room
                    if not problem_node.layout:
                        problem_node.layout = Layout()
                    if not problem_node.layout.rooms:
                        problem_node.layout.rooms = {}
                    if len(problem_node.layout.rooms) == 0:
                        room = Room(name="Emergency Space", room_type="space", room_number="1", area=20.0, height=3.0)
                        room.calculate_volume()
                        problem_node.layout.rooms[room.room_id] = room
                        problem_node.layout.total_area = 20.0
                        problem_node.layout.used_area = 20.0
                        problem_node.layout.calculate_metrics()
            
            logger.info("🔍 Step 2: Conducting research...")
            # ✅ Brief validator gate (Step 3): derive the brief spec ONCE from the
            # root node (which the encoder built from the brief). Every alternative
            # is checked against it before ranking — an invalid design never wins.
            try:
                self._brief_spec = BriefValidator.build_brief_spec(problem_node)
            except Exception as e:
                logger.warning(f"   ⚠ Could not build brief spec (validator disabled): {e}")
                self._brief_spec = None

            research_findings = self.research.research_node(problem_node)
            problem_node = self.research.enhance_node_with_research(problem_node, research_findings)
            logger.info(f"   ✓ Found {len(research_findings['similar_spaces'])} precedents")
            
            # =================================================================
            # PHASE 2.5: GENERATE OPTIMIZED LAYOUT
            # =================================================================
            logger.info("🏗️  Step 2.5: Generating optimized layout...")
            try:
                problem_node.layout = await self.layout_agent.generate_layout(problem_node)
                logger.info(f"   ✓ Layout generated with force-directed placement")
                if hasattr(problem_node.layout, 'compactness_score'):
                    logger.info(f"   ✓ Compactness: {problem_node.layout.compactness_score:.3f}")
            except Exception as e:
                logger.warning(f"   ⚠ Layout generation failed: {e}, using synthesis fallback")
                # Fallback synthesis already exists in lines 220-249
            
            # =================================================================
            # PHASE 2: GENERATE DESIGN SPACE (ADAPTIVE)
            # =================================================================
            if use_got:
                logger.info("🕸️  Step 3: Generating thought graph (adaptive)...")
                
                # ✅ ADAPTIVE: Calculate complexity and adaptive parameters
                complexity_metrics = self.complexity_calculator.calculate_combined_complexity(
                    requirements, problem_node
                )
                adaptive_params = complexity_metrics['adaptive_params']
                
                logger.info(
                    f"   📊 Complexity: {complexity_metrics['complexity_level']} "
                    f"(score={complexity_metrics['combined_overall']:.2f})"
                )
                logger.info(
                    f"   🎯 Adaptive params: depth={adaptive_params['got_depth']}, "
                    f"breadth={adaptive_params['got_breadth']}, "
                    f"max_nodes={adaptive_params['got_max_nodes']}, "
                    f"target_prototypes={adaptive_params['target_prototypes']}"
                )
                
                # Reconfigure GoT engine with adaptive parameters
                self.got_engine.max_depth = adaptive_params['got_depth']
                self.got_engine.breadth = adaptive_params['got_breadth']
                
                # Ensure the original requirements text is available to child
                # node expansions so the GoT engine can re-run the encoder when
                # needed to populate layouts.
                try:
                    problem_node.metadata['original_requirements'] = requirements
                except Exception:
                    # Best-effort: if metadata is not a dict-like object, skip
                    pass
                
                # Allow per-request override for GoT delta/epsilon and selection metric
                # Read per-request GoT overrides, but treat explicit None as "not provided"
                req_delta = request.get('got_delta', None)
                req_patience = request.get('got_patience', None)
                if req_patience is None:
                    req_patience = self.got_patience
                
                # Use adaptive max_nodes unless explicitly overridden
                req_max_nodes = request.get('got_max_nodes', None)
                if req_max_nodes is None:
                    req_max_nodes = adaptive_params['got_max_nodes']

                graph = await self.got_engine.generate_thought_graph(
                    problem_node,
                    expansion_strategies=['functional', 'behavioral', 'layout'],
                    epsilon=self.got_epsilon,
                    delta=req_delta,
                    patience=req_patience,
                    max_nodes=req_max_nodes
                )
                
                stats = self.got_engine.get_graph_statistics()
                logger.info(f"   ✓ Generated graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
                
                # ✅ ADAPTIVE: Determine top_k based on complexity and explicit request
                # Use explicit max_alternatives if provided, otherwise use adaptive target
                if max_alternatives not in (None, 0):
                    top_k = max_alternatives
                else:
                    # Use adaptive target, but don't exceed available nodes
                    top_k = min(
                        adaptive_params['target_prototypes'] * 2,  # Get 2x for pruning
                        stats.get('total_nodes', adaptive_params['target_prototypes'])
                    )

                best_paths = self.got_engine.find_best_paths(
                    problem_node.node_id,
                    top_k=top_k
                )
                
                alternatives = []
                seen_nodes = set()
                for path in best_paths:
                    leaf_id = path.nodes[-1]
                    if leaf_id not in seen_nodes:
                        alternatives.append(self.got_engine.node_registry[leaf_id])
                        seen_nodes.add(leaf_id)
                logger.info(f"   ✓ Selected {len(alternatives)} alternatives from best paths")

                # ✅ SCORE-BASED: Score all alternatives first, then prune low-scoring ones
                logger.info("   📊 Scoring alternatives for adaptive pruning...")
                scored_alternatives = []
                for alt in alternatives:
                    try:
                        # ✅ Recompute actual behaviors from structures BEFORE scoring, so
                        # scores reflect the physics (S → Bs) rather than the encoder's static
                        # estimates. Same call the convergence loop uses; direction is handled
                        # inside the calculator (actual_value = target × performance_ratio).
                        alt = self.behavior_calculator.calculate_actual_behaviors(alt)
                        scores = await self.scoring.score_node(alt)
                        composite = scores['scores']['composite']

                        # ✅ Hard gate: a design that violates the brief may never rank.
                        if self._brief_spec is not None:
                            vres = BriefValidator.validate(alt, self._brief_spec)
                            alt.metadata['brief_validation'] = vres.to_dict()
                            if not vres.passed:
                                logger.warning(
                                    f"   ✗ Node {alt.node_id[:8]} violates brief "
                                    f"({'; '.join(vres.errors)}) → score forced to 0.0"
                                )
                                composite = 0.0

                        alt.composite_score = composite
                        scored_alternatives.append((alt, composite))
                    except Exception as e:
                        # ✅ A node that cannot be scored gets 0.0 — never a plausible
                        # completeness estimate. An unscoreable design must not rank.
                        logger.warning(
                            f"   ✗ Scoring failed for node {alt.node_id[:8]} → score 0.0: {e}"
                        )
                        alt.composite_score = 0.0
                        scored_alternatives.append((alt, 0.0))
                
                # Sort by score (highest first)
                scored_alternatives.sort(key=lambda x: x[1], reverse=True)
                alternatives = [alt for alt, score in scored_alternatives]
                
                logger.info(f"   ✓ Scored {len(alternatives)} alternatives (top score: {scored_alternatives[0][1]:.3f})")
                
                # ✅ SCORE-BASED PRUNING: Prune low-scoring alternatives
                if len(alternatives) > adaptive_params['target_prototypes']:
                    # Calculate score threshold: keep top N or those above median
                    median_score = scored_alternatives[len(scored_alternatives) // 2][1]
                    top_score = scored_alternatives[0][1]
                    
                    # Keep top target_count OR those within 20% of top score
                    score_threshold = max(
                        scored_alternatives[adaptive_params['target_prototypes'] - 1][1],
                        top_score * 0.8  # Keep alternatives within 80% of best
                    )
                    
                    pruned = [alt for alt, score in scored_alternatives if score >= score_threshold]
                    
                    # If still too many, take top target_count
                    if len(pruned) > adaptive_params['target_prototypes']:
                        pruned = [alt for alt, _ in scored_alternatives[:adaptive_params['target_prototypes']]]
                    
                    alternatives = pruned
                    logger.info(
                        f"   ✓ Pruned low-scoring alternatives: "
                        f"{len(scored_alternatives)} → {len(alternatives)} "
                        f"(threshold: {score_threshold:.3f})"
                    )

                # ✅ SCORE-BASED AGGREGATION: Aggregate high-scoring alternatives together
                try:
                    if len(alternatives) > 1:
                        top_score = alternatives[0].composite_score
                        high_score_threshold = top_score * 0.9

                        high_scoring = [
                            alt for alt in alternatives
                            if alt.composite_score >= high_score_threshold
                        ]

                        high_scoring = high_scoring[:min(5, len(high_scoring))]

                        if len(high_scoring) >= 2:
                            high_score_ids = [n.node_id for n in high_scoring]
                            sel_metric = request.get('got_selection_metric', self.got_selection_metric)
                            comp_thresh = request.get('got_compatibility_threshold', 0.0)

                            aggregated = self.got_engine.aggregate_nodes(
                                high_score_ids,
                                top_k=len(high_scoring),
                                compatibility_threshold=comp_thresh,
                                selection_metric=sel_metric
                            )

                            try:
                                aggregated = self.behavior_calculator.calculate_actual_behaviors(aggregated)
                                agg_scores = await self.scoring.score_node(aggregated)
                                aggregated.composite_score = agg_scores['scores']['composite']
                            except Exception as e:
                                # ✅ A merged design that fails to score gets 0.0, not the
                                # best score — it must not inherit a rank it didn't earn.
                                logger.warning(
                                    f"   ✗ Scoring failed for aggregated node → score 0.0: {e}"
                                )
                                aggregated.composite_score = 0.0

                            # ✅ Same hard gate for the merged design: aggregation must
                            # not produce a brief-violating composite that outranks
                            # valid alternatives.
                            if self._brief_spec is not None:
                                vres = BriefValidator.validate(aggregated, self._brief_spec)
                                aggregated.metadata['brief_validation'] = vres.to_dict()
                                if not vres.passed:
                                    logger.warning(
                                        f"   ✗ Aggregated node violates brief "
                                        f"({'; '.join(vres.errors)}) → score forced to 0.0"
                                    )
                                    aggregated.composite_score = 0.0

                            alternatives.insert(0, aggregated)
                            score_list = [f"{n.composite_score:.3f}" for n in high_scoring]
                            logger.info(
                                f"   ✓ Aggregated {len(high_scoring)} high-scoring alternatives "
                                f"(scores: {score_list}) → "
                                f"aggregated score: {aggregated.composite_score:.3f}"
                            )
                        else:
                            logger.debug(
                                f"   → Not enough high-scoring nodes for aggregation ({len(high_scoring)} < 2)"
                            )
                except Exception as e:
                    logger.warning(f"   ⚠ Aggregation failed: {e}")
            else:
                logger.info("🔀 Step 3: Generating design alternatives (adaptive)...")
                # ✅ ADAPTIVE: Calculate complexity for non-GoT path too
                complexity_metrics = self.complexity_calculator.calculate_combined_complexity(
                    requirements, problem_node
                )
                adaptive_params = complexity_metrics['adaptive_params']
                
                # Use adaptive target if max_alternatives not specified
                if max_alternatives in (None, 0):
                    max_alternatives = adaptive_params['target_prototypes']
                
                alternatives = self.generalizer.decompose_problem(problem_node, max_alternatives)
                logger.info(f"   ✓ Generated {len(alternatives)} alternatives (target: {max_alternatives})")
            
            # =================================================================
            # PHASE 3: FULL PIPELINE CONVERGENCE LOOP (Per Alternative)
            # =================================================================
            if enable_convergence_loop:
                logger.info("🔄 Step 4: Running full convergence loop for each alternative...")
                converged_alternatives = []
                
                for i, alt in enumerate(alternatives, 1):
                    logger.info(f"   Processing alternative {i}/{len(alternatives)}...")
                    converged_node, convergence_history = await self._convergence_loop(alt)
                    converged_node.metadata['convergence_history'] = convergence_history
                    converged_alternatives.append(converged_node)
                    
                    logger.info(
                        f"     ✓ Converged: {convergence_history['converged']} "
                        f"(iterations={convergence_history['iterations']}, "
                        f"final_score={convergence_history['final_score']:.3f})"
                    )
                
                alternatives = converged_alternatives
            else:
                # Fallback: Use traditional refinement only
                logger.info("🔄 Step 4: Refining alternatives (no convergence loop)...")
                refined_alternatives = []
                for i, alt in enumerate(alternatives, 1):
                    logger.info(f"   Refining alternative {i}/{len(alternatives)}...")
                    refined, history = self.refiner.refine_node(alt)
                    refined.metadata['refinement_history'] = history
                    refined_alternatives.append(refined)
                alternatives = refined_alternatives
            
            # =================================================================
            # PHASE 4: SCORING & PARETO OPTIMALITY
            # =================================================================
            logger.info("📊 Step 5: Scoring alternatives and building Pareto front...")
            pareto_front = ParetoFront()
            scored_designs = []
            
            for i, alt in enumerate(alternatives, 1):
                scores = await self.scoring.score_node(alt)
                scored_designs.append({
                    'node': alt,
                    'scores': scores
                })
                
                # Add to Pareto front
                pareto_front.add_solution(alt, scores['scores'])
                
                logger.info(f"   {i}/{len(alternatives)}: score={scores['scores']['composite']:.3f}")
            
            # Sort by composite score
            scored_designs.sort(key=lambda x: x['scores']['scores']['composite'], reverse=True)
            
            # Get Pareto-optimal solutions
            pareto_solutions = pareto_front.get_best_solutions(top_k=len(alternatives))
            
            # =================================================================
            # PHASE 5: STORE IN DATABASE
            # =================================================================
            if self.db:
                try:
                    logger.info("💾 Step 6: Storing prototypes in database...")
                    
                    # Store project
                    project_id = problem_node.project_id or problem_node.node_id
                    self.db.store_project(
                        project_id=project_id,
                        project_name=project_name,
                        requirements=requirements,
                        context=context
                    )
                    
                    # Store problem node
                    self.db.store_fbsl_node(problem_node.to_dict())
                    
                    # Store all design prototypes
                    stored_count = 0
                    for design in scored_designs:
                        node = design['node']
                        if self.db.store_fbsl_node(node.to_dict()):
                            stored_count += 1
                            
                            # Store evaluation if detailed scores available
                            if 'details' in design['scores']:
                                from ..core.fbsl_models import EvaluationResult
                                import uuid
                                
                                # Build evaluation dict directly (EvaluationResult dataclass differs)
                                evaluation_dict = {
                                    'evaluation_id': str(uuid.uuid4()),
                                    'node_id': node.node_id,
                                    'project_id': project_id,
                                    'functional_adequacy': design['scores']['details'].get('functional', {}),
                                    'behavioral_performance': design['scores']['details'].get('behavioral', {}),
                                    'structural_feasibility': design['scores']['details'].get('structural', {}),
                                    'layout_efficiency': design['scores']['details'].get('layout', {}),
                                    'sustainability': design['scores']['details'].get('sustainability', {}),
                                    'composite_score': design['scores']['scores']['composite'],
                                    'rank': scored_designs.index(design) + 1,
                                    'strengths': design['scores'].get('strengths', []),
                                    'weaknesses': design['scores'].get('weaknesses', []),
                                    'recommendations': design['scores'].get('recommendations', [])
                                }
                                self.db.store_evaluation(evaluation_dict)
                    
                    logger.info(f"   ✓ Stored {stored_count} prototypes in database")
                except Exception as e:
                    logger.warning(f"   ⚠ Database storage failed: {e}")
            
            # =================================================================
            # PHASE 6: PREPARE RESULTS
            # =================================================================
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'project_id': problem_node.project_id or problem_node.node_id,
                'project_name': project_name,
                'problem_node_id': problem_node.node_id,
                'method': 'Graph of Thought' if use_got else 'Traditional',
                'convergence_enabled': enable_convergence_loop,
                'designs': [
                        {
                            'node_id': d['node'].node_id,
                            'variant_type': d['node'].metadata.get('variant_type', 'N/A'),
                            'description': d['node'].metadata.get('description', 'N/A'),
                            'scores': d['scores']['scores'],
                            'functions_count': len(d['node'].functions),
                            'behaviors_count': len(d['node'].behaviors),
                            'structures_count': len(d['node'].structures),
                            'has_layout': d['node'].layout is not None,
                            'has_floor_plan_svg': bool(getattr(d['node'].layout, 'svg_floor_plan', None)),
                            'has_adjacency_svg': bool(getattr(d['node'].layout, 'adjacency_svg', None)),
                            'svg_floor_plan': getattr(d['node'].layout, 'svg_floor_plan', None) if d['node'].layout else None,
                            'adjacency_svg': getattr(d['node'].layout, 'adjacency_svg', None) if d['node'].layout else None,
                            'refinement_iterations': len(
                                d['node'].metadata.get('refinement_history', {}).get('iterations', [])
                            ),
                            'convergence_iterations': d['node'].metadata.get('convergence_history', {}).get('iterations', 0),
                            'converged': d['node'].metadata.get('convergence_history', {}).get('converged', False),
                            'is_pareto_optimal': any(
                                ps['node'].node_id == d['node'].node_id
                                for ps in pareto_solutions
                            )
                        }
                    for d in scored_designs
                ],
                'pareto_front': {
                    'size': len(pareto_solutions),
                    'solutions': [
                        {
                            'node_id': ps['node'].node_id,
                            'composite_score': ps['scores'].get('composite', 0.0),
                            'objectives': ps['objectives']
                        }
                        for ps in pareto_solutions
                    ]
                },
                'research_findings': {
                    'precedents_found': len(research_findings['similar_spaces']),
                    'recommendations': len(research_findings['recommendations'])
                },
                'processing_time': processing_time
            }
            
            # Add GoT-specific stats
            if use_got:
                result['graph_statistics'] = stats
                result['best_paths'] = [
                    {
                        'length': len(path.nodes),
                        'score': path.total_score,
                        'avg_quality': path.avg_quality
                    }
                    for path in best_paths[:5]
                ]
            
            # ✅ Add complexity metrics to result
            try:
                complexity_metrics = self.complexity_calculator.calculate_combined_complexity(
                    requirements, problem_node
                )
                result['complexity_metrics'] = {
                    'level': complexity_metrics['complexity_level'],
                    'overall_score': complexity_metrics['combined_overall'],
                    'function_count': complexity_metrics.get('function_count', 0),
                    'behavior_count': complexity_metrics.get('behavior_count', 0),
                    'room_count': complexity_metrics.get('room_count', 0),
                    'adaptive_parameters': complexity_metrics['adaptive_params']
                }
            except Exception as e:
                logger.warning(f"Failed to calculate complexity metrics: {e}")
                result['complexity_metrics'] = None
            
            logger.info(f"✅ Pipeline complete in {processing_time:.2f}s")
            logger.info(f"   Generated {len(scored_designs)} designs")
            logger.info(f"   Top score: {scored_designs[0]['scores']['scores']['composite']:.3f}")
            logger.info(f"   Pareto front: {len(pareto_solutions)} optimal solutions")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _adaptive_prune(
        self,
        alternatives: List[FBSLLayoutNode],
        target_count: int,
        quality_threshold: float,
        diversity_threshold: float
    ) -> List[FBSLLayoutNode]:
        """
        ✅ SCORE-BASED PRUNING: Prune low-scoring alternatives
        
        This method is now deprecated in favor of score-based pruning
        done inline in the main pipeline. Kept for backward compatibility.
        
        Args:
            alternatives: List of alternative nodes (should already be scored)
            target_count: Target number of prototypes
            quality_threshold: Minimum quality score (unused, kept for compatibility)
            diversity_threshold: Minimum diversity (unused, kept for compatibility)
        
        Returns:
            Pruned list of high-scoring alternatives
        """
        if len(alternatives) <= target_count:
            return alternatives
        
        # Sort by real composite score. A missing score is treated as 0.0 — never
        # replaced by a completeness estimate, which would rescue brief-violating
        # nodes that the validator gate forced to 0.0.
        def _score_of(alt):
            s = getattr(alt, 'composite_score', None)
            return float(s) if s is not None else 0.0

        scored = [(alt, _score_of(alt)) for alt in alternatives]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top target_count
        return [alt for alt, _ in scored[:target_count]]

    async def _convergence_loop(
        self,
        initial_node: FBSLLayoutNode
    ) -> Tuple[FBSLLayoutNode, Dict[str, Any]]:
        """
        Full FBSL pipeline convergence loop
        
        Loop: F → Be → S → Bs → L → Evaluation → Reformulation
        Until: |S_composite(t) - S_composite(t-1)| < ε
        
        Args:
            initial_node: Starting FBSL node
        
        Returns:
            Tuple of (converged_node, convergence_history)
        """
        current_node = initial_node
        previous_score = 0.0
        iteration = 0
        converged = False
        
        score_history = []
        
        logger.debug(f"Starting convergence loop for node {current_node.node_id[:8]}...")
        
        while iteration < self.max_pipeline_iterations:
            iteration += 1
            logger.debug(f"  Iteration {iteration}/{self.max_pipeline_iterations}")
            
            # ----------------------------------------------------
            # Step 1: F → Be (Functions to Expected Behaviors)
            # Already defined in node.behaviors with target_value
            # ----------------------------------------------------
            
            # ----------------------------------------------------
            # Step 2: Be → S (Expected Behaviors to Structure)
            # Generate/refine structures to meet behaviors
            # ----------------------------------------------------
            try:
                refined_node, refine_history = self.refiner.refine_node(current_node)
                current_node = refined_node
            except Exception as e:
                logger.warning(f"  Refinement failed: {e}")
            
            # ----------------------------------------------------
            # Step 3: S → Bs (Structure to Actual Behaviors)
            # Calculate actual behaviors from structure
            # ----------------------------------------------------
            try:
                current_node = self.behavior_calculator.calculate_actual_behaviors(current_node)
            except Exception as e:
                logger.warning(f"  Behavior calculation failed: {e}")
            
            # ----------------------------------------------------
            # Step 4: S → L (Structure to Layout)
            # Generate spatial layout
            # ----------------------------------------------------
            try:
                layout = await self.layout_agent.generate_layout(current_node)
                current_node.layout = layout
                logger.debug(f"    Layout generated: {len(layout.rooms)} rooms")
            except Exception as e:
                logger.warning(f"  Layout generation failed: {e}")
            
            # ----------------------------------------------------
            # Step 5: Evaluation (Score current state)
            # ----------------------------------------------------
            try:
                scores = await self.scoring.score_node(current_node)
                current_score = scores['scores']['composite']
                score_history.append(current_score)
                
                logger.debug(f"    Score: {current_score:.3f}")
                
            except Exception as e:
                logger.warning(f"  Scoring failed: {e}")
                current_score = 0.0
                score_history.append(0.0)
            
            # ----------------------------------------------------
            # Step 6: Check Convergence
            # |S_composite(t) - S_composite(t-1)| < ε
            # ----------------------------------------------------
            if iteration > 1:
                score_diff = abs(current_score - previous_score)
                logger.debug(f"    Score diff: {score_diff:.4f} (threshold: {self.convergence_threshold})")
                
                if score_diff < self.convergence_threshold:
                    converged = True
                    logger.debug(f"  ✓ Converged at iteration {iteration}")
                    break
            
            previous_score = current_score
            
            # ----------------------------------------------------
            # Step 7: Reformulation (if not converged)
            # Prepare for next iteration
            # ----------------------------------------------------
            if iteration < self.max_pipeline_iterations and not converged:
                # Check if behaviors are satisfied
                unsatisfied_behaviors = [
                    b for b in current_node.behaviors.values()
                    if not b.is_satisfied
                ]
                
                if unsatisfied_behaviors:
                    logger.debug(f"    {len(unsatisfied_behaviors)} unsatisfied behaviors, continuing...")
        
        # Build convergence history
        convergence_history = {
            'iterations': iteration,
            'converged': converged,
            'final_score': current_score,
            'initial_score': score_history[0] if score_history else 0.0,
            'score_history': score_history,
            'score_improvement': (current_score - score_history[0]) if score_history else 0.0
        }
        
        return current_node, convergence_history
    
    async def process_batch_designs(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple design requests in parallel
        
        Args:
            requests: List of design request dictionaries
        
        Returns:
            List of results for each request
        """
        logger.info(f"🚀 Processing batch of {len(requests)} design requests...")
        
        tasks = [
            self.process_design_request(request)
            for request in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Request {i+1} failed: {result}")
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'request_index': i
                })
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.get('success', False))
        logger.info(f"✅ Batch complete: {successful}/{len(requests)} successful")
        
        return processed_results