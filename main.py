# main.py - Fixed version
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
import os


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
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    save_json(asdict(initial_ontology), f'outputs/{args.project_name}_initial_ontology.json')
    return requirements, initial_ontology

def phase2_got_generation(requirements):
    """Flowchart: B→C→D→E→F (GoT Generalizer with Decomposition)"""
    generalizer = GraphOfThoughtsGeneralizer()
    if len(requirements['spatial_needs']) > 10:  # Example complexity check
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
    
    # Fix: scored_prototypes is already a list of dicts, no need to call .to_dict()
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
    
    # Flatten the score structure for Specializer compatibility
    flattened_prototypes = []
    
    for proto in refined_prototypes:
        flattened_proto = proto.copy()
        
        # Extract comprehensive_score data to top level
        if 'comprehensive_score' in proto:
            comp_score = proto['comprehensive_score']
            
            # Move key scores to top level
            flattened_proto['final_score'] = comp_score.get('final_score', 0.7)
            flattened_proto['weighted_total'] = comp_score.get('weighted_total', 0.7)
            flattened_proto['diversity_bonus'] = comp_score.get('diversity_bonus', 0.0)
            flattened_proto['overall_confidence'] = comp_score.get('overall_confidence', 0.7)
            
            # Move individual scores
            flattened_proto['criterion_scores'] = comp_score.get('individual_scores', {})
            
            # Move other important data
            flattened_proto['pareto_efficiency'] = comp_score.get('pareto_efficiency', {})
            flattened_proto['ranking_factors'] = comp_score.get('ranking_factors', {})
            
        else:
            # Fallback values if comprehensive_score is missing
            flattened_proto['final_score'] = 0.7
            flattened_proto['weighted_total'] = 0.7
            flattened_proto['diversity_bonus'] = 0.0
            flattened_proto['overall_confidence'] = 0.7
            flattened_proto['criterion_scores'] = {}
            flattened_proto['pareto_efficiency'] = {}
            flattened_proto['ranking_factors'] = {}
        
        flattened_prototypes.append(flattened_proto)
    
    # Now run specialization with flattened data
    specializer = PrototypeSpecializer()
    specialization_output = specializer.specialize_prototypes(
        flattened_prototypes, requirements, research_data
    )
    
    specialized_prototypes = specialization_output.final_prototypes
    save_json(specialized_prototypes, 'specialized_prototypes.json')
    
    return specialized_prototypes

def phase6_layout_optimization(specialized_prototypes, requirements):
    """Flowchart: Q→R→S→T→U→V→W (Layout Generation → Optimization Loop)"""
    optimizer = LayoutOptimizer()
    optimized_layouts = optimizer.optimize_layouts(specialized_prototypes, requirements)
    save_json(optimized_layouts, 'optimized_layouts.json')
    return optimized_layouts

def phase7_final_synthesis(optimized_layouts, specialized_prototypes, requirements):
    """Flowchart: X→Y→Z (Final Synthesis → Documentation) - FIXED VERSION"""
    fbs_interface = EnhancedDirectionalFBSInterface()
    
    # Create proper FBS ontologies first, then map them to layouts
    ontologies = []
    
    # Convert requirements dict back to object if needed
    from dataclasses import dataclass
    from typing import List, Dict, Any
    
    @dataclass
    class Requirements:
        spatial_needs: List[Dict[str, Any]]
        functional_requirements: List[str]
        design_constraints: Dict[str, Any]
        user_preferences: Dict[str, Any]
        site_context: Dict[str, Any]
    
    # Create requirements object from dict if needed
    if isinstance(requirements, dict):
        req_obj = Requirements(
            spatial_needs=requirements.get('spatial_needs', []),
            functional_requirements=requirements.get('functional_requirements', []),
            design_constraints=requirements.get('design_constraints', {}),
            user_preferences=requirements.get('user_preferences', {}),
            site_context=requirements.get('site_context', {})
        )
    else:
        req_obj = requirements
    
    for i, (layout, proto) in enumerate(zip(optimized_layouts, specialized_prototypes)):
        try:
            # First, create a proper FBS ontology from the prototype
            print(f"Creating FBS ontology for prototype {i+1}...")
            
            # Generate FBS ontology for this prototype
            ontology = fbs_interface.generate_prototype_specific_fbs_ontology(proto, req_obj)
            
            # Now map the layout to this ontology
            print(f"Mapping layout to FBS ontology for prototype {i+1}...")
            mapped_ontology = fbs_interface.map_fbs_to_layout(layout, ontology)
            
            ontologies.append(mapped_ontology)
            
        except Exception as e:
            print(f"Error processing prototype {i+1}: {str(e)}")
            # Create a basic fallback ontology
            try:
                basic_ontology = fbs_interface.generate_fbs_ontology(req_obj, f"prototype_{i+1}")
                ontologies.append(basic_ontology)
            except Exception as e2:
                print(f"Failed to create fallback ontology: {str(e2)}")
                continue
    
    if not ontologies:
        print("Warning: No ontologies created, generating basic ontology...")
        # Create at least one basic ontology
        try:
            basic_ontology = fbs_interface.generate_fbs_ontology(req_obj, "basic_design")
            ontologies.append(basic_ontology)
        except Exception as e:
            print(f"Failed to create basic ontology: {str(e)}")
            return None
    
    # Synthesize final FBS
    print("Synthesizing final FBS...")
    synthesizer = FBSSynthesizer()
    final_fbs = synthesizer.synthesize_final_fbs(ontologies)
    
    # Convert final_fbs to a JSON-serializable format
    def make_serializable(obj):
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            # Handle dataclass or custom objects
            result = {}
            for key, value in obj.__dict__.items():
                if callable(value):
                    # Skip function objects
                    continue
                elif hasattr(value, '__dict__'):
                    result[key] = make_serializable(value)
                elif isinstance(value, list):
                    result[key] = [make_serializable(item) for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: make_serializable(v) for k, v in value.items() if not callable(v)}
                else:
                    result[key] = value
            return result
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() if not callable(v)}
        elif callable(obj):
            # Return string representation for callable objects
            return str(obj)
        else:
            return obj
    
    # Create JSON-serializable version
    print("Converting final FBS to serializable format...")
    serializable_fbs = make_serializable(final_fbs)
    
    # Generate documentation with serializable data
    print("Generating final documentation...")
    try:
        doc_path = synthesizer.generate_documentation(serializable_fbs)
        print(f"Final documentation: {doc_path}")
    except Exception as e:
        print(f"Error generating documentation: {str(e)}")
        # Create a simple fallback documentation
        doc_path = "outputs/final_design_summary.txt"
        os.makedirs('outputs', exist_ok=True)
        with open(doc_path, 'w') as f:
            f.write("Final Design Summary\n")
            f.write("===================\n\n")
            f.write(f"Number of layouts processed: {len(optimized_layouts)}\n")
            f.write(f"Number of prototypes: {len(specialized_prototypes)}\n")
            f.write(f"FBS ontologies created: {len(ontologies)}\n")
            f.write("\nDesign process completed successfully.\n")
        print(f"Fallback documentation created: {doc_path}")
    
    return final_fbs

if __name__ == '__main__':
    args = parse_arguments()
    
    # Run phases sequentially
    print("Phase 1: Initial Analysis...")
    requirements, initial_ontology = phase1_initial_analysis(args)
    requirements_dict = asdict(requirements)
    
    print("Phase 2: GoT Generation...")
    prototypes = phase2_got_generation(requirements_dict)
    
    print("Phase 3: Research and Scoring...")
    scored_prototypes, research_data = phase3_research_and_scoring(prototypes, requirements_dict)
    
    print("Phase 4: Refinement...")
    refined_prototypes = phase4_refinement(scored_prototypes, requirements_dict, research_data)
    
    print("Phase 5: Specialization...")
    specialized_prototypes = phase5_specialization(refined_prototypes, requirements_dict, research_data)
    
    print("Phase 6: Layout Optimization...")
    optimized_layouts = phase6_layout_optimization(specialized_prototypes, requirements_dict)
    
    print("Phase 7: Final Synthesis...")
    # Pass the requirements_dict to phase7_final_synthesis
    final_fbs = phase7_final_synthesis(optimized_layouts, specialized_prototypes, requirements_dict)
    
    if final_fbs:
        print("Pipeline completed successfully!")
        # Save final FBS
        save_json(asdict(final_fbs), f'outputs/{args.project_name}_final_fbs.json')
    else:
        print("Pipeline completed with errors in final synthesis phase.")