# layout_optimizer.py
from layout_generator import CompactRoomPlacer, generate_all_visualizations_fixed
from fbs import EnhancedDirectionalFBSInterface
import numpy as np
import json
from typing import List, Dict

class LayoutOptimizer:
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.fbs_interface = EnhancedDirectionalFBSInterface()

    def optimize_layouts(self, prototypes: List[Dict], requirements: Dict) -> List[Dict]:
        """Iterative layout optimization loop (for flowchart's S→W→W1→S)."""
        best_layouts = []
        prev_performance = 0.0
        iteration = 0
        
        while iteration < self.max_iterations:
            # Generate variants
            variants = CompactRoomPlacer.generate_layout_variants(prototypes[0].get('rooms', []))
            
            # Cross-layout comparison (flowchart U)
            performances = self._compare_layouts(variants, prototypes[0])
            current_performance = np.mean(performances)
            
            # Check convergence (flowchart W1)
            if abs(current_performance - prev_performance) < self.convergence_threshold:
                break
            
            # Select best (flowchart V)
            best_idx = np.argmax(performances)
            best_layout = variants[best_idx]
            best_layouts.append(best_layout)
            
            # Generate visualizations and FBS
            json_data = {'rooms': best_layout, 'prototype': prototypes[0]}
            with open('temp_layout.json', 'w') as f:
                json.dump(json_data, f)
            generate_all_visualizations_fixed('temp_layout.json', 'optimized_layout')
            
            prev_performance = current_performance
            iteration += 1
        
        return best_layouts

    def _compare_layouts(self, variants: List[List[Dict]], prototype: Dict) -> List[float]:
        """Cross-layout FBS comparison (flowchart U)."""
        performances = []
        for variant in variants:
            ontology = self.fbs_interface.map_fbs_to_layout(variant, prototype)
            # Dummy performance score based on FBS (e.g., count structures)
            score = len(ontology.structures) / 10.0  # Replace with real metric
            performances.append(score)
        return performances