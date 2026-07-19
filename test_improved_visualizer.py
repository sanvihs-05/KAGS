"""
Test script for improved layout visualizer
Demonstrates the fixed visualization with a sample 2BR apartment
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.visualization.improved_layout_visualizer import ImprovedLayoutVisualizer
from backend.models.fbsl_models import Layout, Room
import math


def create_sample_2br_apartment():
    """Create a sample 2-bedroom apartment layout"""
    
    layout = Layout()
    layout.configuration_name = "Sample 2BR Apartment"
    
    # Define rooms with areas
    rooms_data = [
        ("living_room", "Living Room", 250),      # 250 sq ft
        ("kitchen", "Kitchen", 120),              # 120 sq ft
        ("dining_room", "Dining Room", 100),      # 100 sq ft
        ("bedroom", "Master Bedroom", 180),       # 180 sq ft
        ("bedroom", "Bedroom 2", 140),            # 140 sq ft
        ("bathroom", "Master Bath", 50),          # 50 sq ft
        ("bathroom", "Guest Bath", 40),           # 40 sq ft
        ("hallway", "Hallway", 60),               # 60 sq ft
        ("balcony", "Balcony", 80),               # 80 sq ft
    ]
    
    for i, (room_type, room_name, area) in enumerate(rooms_data, 1):
        # Calculate approximate square dimensions
        side = math.sqrt(area)
        
        room = Room(
            name=room_name,
            room_type=room_type,
            room_number=str(i),
            area=area,
            height=9.0,  # 9 feet ceiling height
        )
        
        # Set adjacency preferences
        if room_type == "kitchen":
            room.required_adjacencies = ["dining_room", "living_room"]
        elif room_type == "dining_room":
            room.required_adjacencies = ["kitchen", "living_room"]
        elif room_type == "living_room":
            room.required_adjacencies = ["kitchen", "dining_room", "hallway", "balcony"]
        elif room_type == "bedroom" and "Master" in room_name:
            room.required_adjacencies = ["bathroom", "hallway"]
        elif room_type == "bedroom":
            room.required_adjacencies = ["bathroom", "hallway"]
        elif room_type == "bathroom":
            room.required_adjacencies = ["bedroom", "hallway"]
        elif room_type == "hallway":
            room.required_adjacencies = ["living_room", "bedroom", "bathroom"]
        elif room_type == "balcony":
            room.required_adjacencies = ["living_room"]
        
        room.calculate_volume()
        layout.rooms[room.room_id] = room
    
    layout.total_area = sum(r.area for r in layout.rooms.values())
    layout.used_area = layout.total_area
    
    return layout


def main():
    """Run the test"""
    
    print("=" * 80)
    print("TESTING IMPROVED LAYOUT VISUALIZER")
    print("=" * 80)
    print()
    
    # Create sample layout
    print("📐 Creating sample 2BR apartment layout...")
    layout = create_sample_2br_apartment()
    print(f"   ✓ Created layout with {len(layout.rooms)} rooms")
    print(f"   ✓ Total area: {layout.total_area:.0f} sq ft")
    print()
    
    # List rooms
    print("🏠 Rooms:")
    for room in layout.rooms.values():
        adjacencies = ", ".join(room.required_adjacencies) if room.required_adjacencies else "none"
        print(f"   • {room.name} ({room.area:.0f} sq ft) - wants adjacency to: {adjacencies}")
    print()
    
    # Initialize visualizer
    print("🎨 Initializing improved visualizer...")
    visualizer = ImprovedLayoutVisualizer(output_dir="visual_outputs")
    print("   ✓ Visualizer ready")
    print()
    
    # Generate visualizations
    print("🖼️  Generating visualizations...")
    try:
        svg_path, adjacency_path = visualizer.render(
            layout=layout,
            project_name="sample_2br_apartment",
            node_id="test_demo_001"
        )
        
        print("   ✅ SUCCESS!")
        print()
        print("📁 Generated files:")
        print(f"   • SVG Floor Plan: {svg_path}")
        print(f"   • Adjacency Graph: {adjacency_path}")
        print()
        print("=" * 80)
        print("✅ TEST COMPLETE - Check the visual_outputs/ directory!")
        print("=" * 80)
        print()
        print("The adjacency graph shows:")
        print("  • Panel 1: Spatial layout with actual room positions")
        print("  • Panel 2: Graph topology with edge types")
        print("  • Panel 3: Connectivity analysis with satisfaction scores")
        print("  • Panel 4: Performance metrics")
        print()
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
