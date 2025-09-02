# refinement_agent.py
from typing import Dict, List
import numpy as np
from scoring_agent import MultiCriteriaScoringAgent  # Import for performance checks
from research_agent import ResearchAgent  # Import for quality checks

class RefinementAgent:
    def __init__(self, max_iterations: int = 5, performance_threshold: float = 0.7):
        self.max_iterations = max_iterations
        self.performance_threshold = performance_threshold
        self.scoring_agent = MultiCriteriaScoringAgent()
        self.research_agent = ResearchAgent()

    def refine_prototypes(self, prototypes: List[Dict], requirements: Dict, research_data: Dict) -> List[Dict]:
        """Recursive refinement loop (for flowchart's O→P→P1→C)."""
        iteration = 0
        while iteration < self.max_iterations:
            scores = self.scoring_agent.score_prototypes(prototypes, requirements)
            if self.scoring_agent.check_performance_threshold(scores, self.performance_threshold):
                return prototypes  # Terminate if threshold met
            
            # FBS-guided feedback: Analyze low scores and refine
            low_performers = [s for s in scores if s.final_score < self.performance_threshold]
            for lp in low_performers:
                # Example refinement: Boost environmental strategy
                prototypes[0]['detailed_config']['environmental_strategy']['passive_strategies'].append('new_strategy')
            
            # Research quality recursion if needed
            research_contexts = self.research_agent.conduct_research('id', prototypes[0], requirements)
            if not self.research_agent.check_research_quality(research_contexts):
                # Generate enhanced queries (flowchart I)
                enhanced_queries = self._generate_enhanced_queries(research_contexts)
                research_data.update({'enhanced_queries': enhanced_queries})
            
            iteration += 1
        return prototypes  # Terminate after max iterations

    def _generate_enhanced_queries(self, contexts: List) -> List[str]:
        """Generate better queries for research recursion (flowchart I)."""
        return [f"Enhanced: {ctx.research_query}" for ctx in contexts]