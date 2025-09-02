# main.py
import argparse
from generalizer import GraphOfThoughtsGeneralizer
from research_agent import ResearchAgent
from scoring_agent import MultiCriteriaScoringAgent
from specialiser import PrototypeSpecializer
from refinement_agent import RefinementAgent
from layout_optimizer import LayoutOptimizer
from fbs_synthesizer import FBSSynthesizer
from fbs import GemmaFBSAnalyzer, EnhancedDirectionalFBSInterface
from config import *
from utils import load_json, save_json
from dataclasses import asdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Architectural Design Pipeline")
    parser.add_argument('--user_input', type=str, default="Design a house with 3 bedrooms, 2 bathrooms, 1 kitchen, and 1 living room.", help="User requirements")
    parser.add_argument('--project_name', type=str, default="house_design", help="Project name")
    return parser.parse_args()

def phase1_initial_analysis(args):
    """Flowchart: A→B (User Requirements → Initial FBS Analysis)"""
    parser = GemmaFBSAnalyzer()
    requirements = parser.analyze_and_parse_requirements(args.user_input)
    fbs_interface = EnhancedDirectionalFBSInterface()
    initial_ontology = fbs_interface.generate_fbs_ontology(requirements, args.project_name)
    save_json(asdict(initial_ontology), f'outputs/{args.project_name}_initial_ontology.json')
    return requirements, initial_ontology

def phase2_got_generation(requirements):
    """Flowchart: B→C→D→E→F (GoT Generalizer with Decomposition)"""
    generalizer = GraphOfThoughtsGeneralizer()
    if len(requirements.spatial_needs) > 10:  # Example complexity check
        sub_strategies = generalizer.decompose_strategy_space(requirements)
        # Recursive GoT (implement as a loop or recursive call if needed)
        prototypes = []
        for sub in sub_strategies:
            prototypes.extend(generalizer.generate_design_graph(requirements)['prototypes'])
    else:
        prototypes = generalizer.generate_design_graph(requirements)['prototypes']
    save_json(prototypes, 'prototypes.json')
    return prototypes

def phase3_research_and_scoring(prototypes, requirements):
    """Flowchart: G→H→I→J (Research → Quality Check → Scoring)"""
    research_agent = ResearchAgent()
    research_data = {}
    for proto in prototypes:
        contexts = research_agent.conduct_research(proto['prototype_id'], proto, requirements)
        if not research_agent.check_research_quality(contexts):
            # Recursive enhancement (loop)
            for _ in range(MAX_REFINEMENT_ITERATIONS):
                enhanced_queries = []  # Generate from refinement_agent if needed
                contexts = research_agent.conduct_research(proto['prototype_id'], proto, requirements)  # Re-run with enhanced
                if research_agent.check_research_quality(contexts):
                    break
        research_data[proto['prototype_id']] = contexts
    
    scoring_agent = MultiCriteriaScoringAgent()
    scored_prototypes = scoring_agent.score_prototypes(prototypes, requirements)
    save_json(scored_prototypes, 'scored_prototypes.json')
    return scored_prototypes, research_data

def phase4_refinement(scored_prototypes, requirements, research_data):
    """Flowchart: J→K→L→M→N→O→P (FBS Ontology → Refinement Loop)"""
    fbs_interface = EnhancedDirectionalFBSInterface()
    ontologies = [fbs_interface.generate_prototype_specific_fbs_ontology(proto, requirements) for proto in scored_prototypes]
    
    refinement_agent = RefinementAgent()
    refined_prototypes = refinement_agent.refine_prototypes(scored_prototypes, requirements, research_data)
    save_json(refined_prototypes, 'refined_prototypes.json')
    return refined_prototypes

def phase5_specialization(refined_prototypes, requirements, research_data):
    """Flowchart: N→Q (Specializer)"""
    specializer = PrototypeSpecializer()
    specialization_output = specializer.specialize_prototypes(refined_prototypes, requirements, research_data)
    specialized_prototypes = specialization_output.final_prototypes
    save_json(specialized_prototypes, 'specialized_prototypes.json')
    return specialized_prototypes

def phase6_layout_optimization(specialized_prototypes, requirements):
    """Flowchart: Q→R→S→T→U→V→W (Layout Generation → Optimization Loop)"""
    optimizer = LayoutOptimizer()
    optimized_layouts = optimizer.optimize_layouts(specialized_prototypes, requirements)
    save_json(optimized_layouts, 'optimized_layouts.json')
    return optimized_layouts

def phase7_final_synthesis(optimized_layouts, specialized_prototypes):
    """Flowchart: X→Y→Z (Final Synthesis → Documentation)"""
    fbs_interface = EnhancedDirectionalFBSInterface()
    ontologies = [fbs_interface.map_fbs_to_layout(layout, proto) for layout, proto in zip(optimized_layouts, specialized_prototypes)]
    
    synthesizer = FBSSynthesizer()
    final_fbs = synthesizer.synthesize_final_fbs(ontologies)
    doc_path = synthesizer.generate_documentation(final_fbs)
    print(f"Final documentation: {doc_path}")
    return final_fbs

if __name__ == '__main__':
    args = parse_arguments()
    
    # Run phases sequentially
    requirements, initial_ontology = phase1_initial_analysis(args)
    prototypes = phase2_got_generation(requirements)
    scored_prototypes, research_data = phase3_research_and_scoring(prototypes, requirements)
    refined_prototypes = phase4_refinement(scored_prototypes, requirements, research_data)
    specialized_prototypes = phase5_specialization(refined_prototypes, requirements, research_data)
    optimized_layouts = phase6_layout_optimization(specialized_prototypes, requirements)
    final_fbs = phase7_final_synthesis(optimized_layouts, specialized_prototypes)