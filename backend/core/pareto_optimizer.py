"""
Pareto Optimality Module for FBSL-KAGS Framework

Implements Pareto dominance checking, frontier identification, and trade-off analysis
for multi-objective optimization across the five FBSL evaluation dimensions.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ParetoOptimizer:
    """
    Pareto optimality analyzer for FBSL design solutions
    
    Provides methods for:
    - Dominance checking
    - Pareto frontier identification
    - Trade-off characterization
    - Solution selection with diversity preservation
    """
    
    def __init__(self):
        self.dimensions = ['functional_score', 'behavioral_score', 'structural_score', 
                          'layout_score', 'sustainability_score']
        self.dimension_names = ['functional', 'behavioral', 'structural', 'layout', 'sustainability']
    
    def check_dominance(self, sol1, sol2) -> bool:
        """
        Check if sol1 Pareto-dominates sol2
        
        A solution dominates another if it is better or equal in all dimensions
        and strictly better in at least one dimension.
        
        Args:
            sol1: First solution (FBSLLayoutNode)
            sol2: Second solution (FBSLLayoutNode)
            
        Returns:
            True if sol1 dominates sol2, False otherwise
        """
        all_better_or_equal = True
        at_least_one_better = False
        
        for dim in self.dimensions:
            score1 = getattr(sol1, dim, 0.0) or 0.0
            score2 = getattr(sol2, dim, 0.0) or 0.0
            
            if score1 < score2:
                all_better_or_equal = False
                break
            
            if score1 > score2:
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def identify_pareto_frontier(self, solutions: List) -> List:
        """
        Identify non-dominated solutions (Pareto frontier)
        
        The Pareto frontier consists of all solutions where no other solution
        dominates them across all evaluation dimensions.
        
        Args:
            solutions: List of FBSL nodes with scores
            
        Returns:
            List of non-dominated solutions
        """
        if not solutions:
            return []
        
        pareto_set = []
        
        for candidate in solutions:
            is_dominated = False
            
            # Check if any other solution dominates this candidate
            for other in solutions:
                if candidate.node_id == other.node_id:
                    continue
                
                if self.check_dominance(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_set.append(candidate)
        
        logger.info(f"  Pareto frontier: {len(pareto_set)}/{len(solutions)} non-dominated solutions")
        return pareto_set
    
    def characterize_trade_offs(self, solution, pareto_solutions: List) -> Dict[str, Any]:
        """
        Characterize trade-offs for a Pareto solution
        
        Identifies which dimensions the solution champions (is best at)
        and what trade-offs it makes in other dimensions.
        
        Args:
            solution: The solution to characterize
            pareto_solutions: All Pareto-optimal solutions for comparison
            
        Returns:
            Dictionary with 'champions' and 'trade_offs' keys
        """
        champions = []
        trade_offs = []
        
        for i, dim_attr in enumerate(self.dimensions):
            dim_name = self.dimension_names[i]
            sol_score = getattr(solution, dim_attr, 0.0) or 0.0
            best_score = max(getattr(s, dim_attr, 0.0) or 0.0 for s in pareto_solutions)
            
            # Champion if within 1% of best
            if abs(sol_score - best_score) < 0.01:
                champions.append(dim_name)
            else:
                cost = best_score - sol_score
                if cost > 0.01:  # Only include significant trade-offs
                    trade_offs.append((dim_name, cost))
        
        # Sort trade-offs by cost (largest first)
        trade_offs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'champions': champions,
            'trade_offs': trade_offs,
            'composite_score': getattr(solution, 'composite_score', 0.0) or 0.0
        }
    
    def generate_trade_off_report(self, pareto_solutions: List) -> str:
        """
        Generate human-readable trade-off analysis report
        
        Args:
            pareto_solutions: List of Pareto-optimal solutions
            
        Returns:
            Formatted report string
        """
        if not pareto_solutions:
            return "No Pareto solutions to analyze"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PARETO FRONTIER TRADE-OFF ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Sort by composite score
        sorted_solutions = sorted(pareto_solutions, 
                                 key=lambda s: getattr(s, 'composite_score', 0.0) or 0.0, 
                                 reverse=True)
        
        for i, solution in enumerate(sorted_solutions, 1):
            analysis = self.characterize_trade_offs(solution, pareto_solutions)
            
            report_lines.append(f"Solution {i}: {solution.node_id[:8]}")
            report_lines.append(f"  Composite Score: {analysis['composite_score']:.3f}")
            
            if analysis['champions']:
                champ_str = ", ".join(analysis['champions'])
                report_lines.append(f"  ✓ Champions: {champ_str}")
            
            if analysis['trade_offs']:
                report_lines.append(f"  ⚖ Trade-offs:")
                for dim, cost in analysis['trade_offs']:
                    report_lines.append(f"    - {dim}: -{cost:.3f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        return "\n".join(report_lines)
    
    def pareto_preserving_selection(self, solutions: List, 
                                   target_count: int = 5,
                                   min_pareto_solutions: int = 3) -> List:
        """
        Select solutions while preserving Pareto frontier diversity
        
        Args:
            solutions: All candidate solutions
            target_count: Desired number of solutions
            min_pareto_solutions: Minimum Pareto solutions to retain
            
        Returns:
            Selected solutions (Pareto-optimal + high-scoring if needed)
        """
        if not solutions:
            return []
        
        # Identify Pareto frontier
        pareto_set = self.identify_pareto_frontier(solutions)
        
        # If Pareto set is sufficient, return top-k by composite score
        if len(pareto_set) >= target_count:
            sorted_pareto = sorted(pareto_set, 
                                  key=lambda s: getattr(s, 'composite_score', 0.0) or 0.0, 
                                  reverse=True)
            return sorted_pareto[:target_count]
        
        # If Pareto set is too small, add high-scoring dominated solutions
        if len(pareto_set) < min_pareto_solutions:
            dominated = [s for s in solutions if s not in pareto_set]
            sorted_dominated = sorted(dominated, 
                                     key=lambda s: getattr(s, 'composite_score', 0.0) or 0.0, 
                                     reverse=True)
            
            needed = min_pareto_solutions - len(pareto_set)
            additional = sorted_dominated[:needed]
            
            logger.info(f"  Added {len(additional)} high-scoring dominated solutions")
            return pareto_set + additional
        
        return pareto_set
    
    def get_pareto_statistics(self, pareto_solutions: List) -> Dict[str, Any]:
        """
        Get statistics about the Pareto frontier
        
        Args:
            pareto_solutions: List of Pareto-optimal solutions
            
        Returns:
            Dictionary with Pareto frontier statistics
        """
        if not pareto_solutions:
            return {
                'size': 0,
                'champion_counts': {},
                'avg_trade_off_cost': 0.0
            }
        
        champion_counts = {dim: 0 for dim in self.dimension_names}
        total_trade_off_cost = 0.0
        total_trade_offs = 0
        
        for solution in pareto_solutions:
            analysis = self.characterize_trade_offs(solution, pareto_solutions)
            
            for champion in analysis['champions']:
                champion_counts[champion] += 1
            
            for _, cost in analysis['trade_offs']:
                total_trade_off_cost += cost
                total_trade_offs += 1
        
        return {
            'size': len(pareto_solutions),
            'champion_counts': champion_counts,
            'avg_trade_off_cost': total_trade_off_cost / max(total_trade_offs, 1)
        }
