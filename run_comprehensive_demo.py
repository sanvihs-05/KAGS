"""
Comprehensive FBSL-KAGS Framework Demonstration
Test Case: Sustainable 4-Bedroom Family Home with Home Office

This script runs the complete pipeline and generates detailed reports
showing the entire process including scoring, pruning, and aggregation.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Test case configuration
TEST_CASE = {
    "project_name": "Sustainable_Family_Home_Demo",
    "requirements": """
    Design a sustainable 4-bedroom family home for a family of 5 (2 adults, 3 children aged 8, 12, 15).
    
    FUNCTIONAL REQUIREMENTS:
    - 4 bedrooms (1 master suite, 3 children's bedrooms)
    - Home office/study space (remote work capability)
    - Open-plan living/dining/kitchen area
    - 2.5 bathrooms (master ensuite, family bathroom, powder room)
    - Laundry/utility room
    - Outdoor connection (patio/deck)
    - Storage solutions throughout
    
    SUSTAINABILITY REQUIREMENTS:
    - Passive solar design principles
    - Natural ventilation and daylighting
    - Energy-efficient envelope (U-value < 0.25 W/m²K)
    - Renewable materials preference
    - Water conservation features
    - Target: Net-zero energy ready
    
    SPATIAL REQUIREMENTS:
    - Total area: 180-220 sqm
    - Master bedroom: 18-22 sqm with ensuite
    - Children's bedrooms: 12-15 sqm each
    - Home office: 10-12 sqm (quiet location)
    - Living/dining/kitchen: 45-55 sqm (open plan)
    - Outdoor living: 20-25 sqm
    
    SITE CONTEXT:
    - Suburban location
    - North-facing preferred orientation
    - Moderate climate (hot summers, mild winters)
    - Privacy from neighbors required
    - Street noise considerations
    
    DESIGN PRIORITIES:
    1. Family interaction and flexibility
    2. Environmental performance
    3. Privacy and acoustic separation
    4. Natural light and ventilation
    5. Future adaptability
    """,
    
    "pipeline_config": {
        "max_alternatives": 5,
        "got_delta": 0.05,
        "got_patience": 3,
        "got_max_nodes": 50,
        "top_k": 5  # Generate reports for top 5 prototypes
    }
}

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def save_test_case():
    """Save test case configuration"""
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    config_file = output_dir / "test_case_config.json"
    with open(config_file, 'w') as f:
        json.dump(TEST_CASE, f, indent=2)
    
    print(f"✓ Test case configuration saved: {config_file}")
    return config_file

def main():
    """Run comprehensive demonstration"""
    
    print_header("FBSL-KAGS FRAMEWORK COMPREHENSIVE DEMONSTRATION")
    
    print("📋 TEST CASE: Sustainable 4-Bedroom Family Home")
    print()
    print("This demonstration will show:")
    print("  1. Complete FBSL-KAGS pipeline execution")
    print("  2. Graph of Thoughts (GoT) exploration process")
    print("  3. Scoring mechanism for design prototypes")
    print("  4. Pruning decisions and criteria")
    print("  5. Aggregation of selected prototypes")
    print("  6. Visual outputs for all final prototypes")
    print()
    
    # Save test case
    config_file = save_test_case()
    
    print_header("NEXT STEPS")
    print("To run the complete pipeline:")
    print()
    print("  .\\venv\\Scripts\\python.exe scripts\\run_full_pipeline.py \\")
    print(f'    --requirements "{TEST_CASE["requirements"][:100]}..." \\')
    print(f'    --project_name {TEST_CASE["project_name"]} \\')
    print(f'    --top_k {TEST_CASE["pipeline_config"]["top_k"]} \\')
    print(f'    --max_alternatives {TEST_CASE["pipeline_config"]["max_alternatives"]}')
    print()
    print("This will generate:")
    print("  • Complete FBSL reports for each prototype")
    print("  • Visual outputs (SVG floor plans + adjacency graphs)")
    print("  • Project summary with scoring details")
    print("  • Detailed walkthrough documentation")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())
