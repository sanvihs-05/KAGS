"""
Generate Level 1 strategy visualizations with UNIQUE layouts for each strategy
Each strategy has a distinct spatial arrangement reflecting its optimization goal
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import json
import os

# Room colors
ROOM_COLORS = {
    'bedroom': '#3498DB',
    'bathroom': '#E67E22',
    'kitchen': '#27AE60',
    'living': '#9B59B6',
    'study': '#E74C3C',
    'garage': '#95A5A6',
    'utility': '#16A085',
    'storage': '#8E44AD',
    'hallway': '#BDC3C7',
    'default': '#34495E'
}


def plot_spatial_layout(ax, rooms, adj_graph):
    """Plot spatial layout"""
    ax.set_title("Spatial Layout", fontsize=14, fontweight='bold')
    
    for room in rooms:
        color = ROOM_COLORS.get(room['room_type'], ROOM_COLORS['default'])
        rect = Rectangle((room['x'], room['y']),
                       room['width'], room['height'],
                       facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        label_text = f"{room['name']}\n{room['area']:.1f} m²"
        ax.text(room['x'] + room['width']/2, 
               room['y'] + room['height']/2,
               label_text,
               ha='center', va='center', fontsize=6, fontweight='bold',
               color='white', wrap=True)
    
    # Draw connections
    G = nx.Graph()
    G.add_edges_from(adj_graph['edges'])
    pos = {room['room_id']: (room['x'] + room['width']/2, 
                             room['y'] + room['height']/2) 
           for room in rooms}
    
    # Critical edges only (reduced clutter)
    critical_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'critical']
    if critical_edges:
        nx.draw_networkx_edges(G, pos, edgelist=critical_edges, 
                             edge_color='#E74C3C', width=2, ax=ax, alpha=0.8, arrows=False)
    
    ax.set_aspect('equal')
    ax.set_xlim(-1, max(r['x'] + r['width'] for r in rooms) + 1)
    ax.set_ylim(-1, max(r['y'] + r['height'] for r in rooms) + 1)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)


def plot_adjacency_graph(ax, rooms, adj_graph):
    """Plot adjacency graph"""
    ax.set_title("Adjacency Graph", fontsize=14, fontweight='bold')
    
    G = nx.Graph()
    G.add_edges_from(adj_graph['edges'])
    
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    room_lookup = {r['room_id']: r for r in rooms}
    node_colors = [ROOM_COLORS.get(room_lookup[node]['room_type'], ROOM_COLORS['default'])
                  for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                          ax=ax, edgecolors='black', linewidths=2)
    
    # Critical edges
    critical_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'critical']
    if critical_edges:
        nx.draw_networkx_edges(G, pos, edgelist=critical_edges, 
                             edge_color='#E74C3C', width=3, ax=ax, alpha=0.8, 
                             label=f'Critical ({len(critical_edges)})')
    
    labels = {node: room_lookup[node]['name'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax, font_weight='bold')
    
    if G.edges():
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    ax.axis('off')


def generate_strategy_visualization(strategy_data, output_path):
    """Generate visualization for one strategy"""
    
    rooms = strategy_data['rooms']
    adj_graph = strategy_data['adjacency_graph']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    plot_spatial_layout(ax1, rooms, adj_graph)
    plot_adjacency_graph(ax2, rooms, adj_graph)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: {output_path}")


# Strategy 1: Functional Priority - Larger bedrooms clustered
strategy_1 = {
    'rooms': [
        # Left wing - Bedrooms cluster (prioritized)
        {'room_id': 'R001', 'name': 'Master BR', 'room_type': 'bedroom', 'area': 22, 'x': 0, 'y': 0, 'width': 5, 'height': 4.4},
        {'room_id': 'R002', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 15.4, 'x': 5.5, 'y': 0, 'width': 4, 'height': 3.85},
        {'room_id': 'R003', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 14.3, 'x': 0, 'y': 5, 'width': 4, 'height': 3.58},
        {'room_id': 'R004', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 14.3, 'x': 4.5, 'y': 5, 'width': 4, 'height': 3.58},
        # Center - Living areas (large)
        {'room_id': 'R005', 'name': 'Living/Dining', 'room_type': 'living', 'area': 44, 'x': 10, 'y': 0, 'width': 8, 'height': 5.5},
        {'room_id': 'R006', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 15.2, 'x': 18.5, 'y': 0, 'width': 4, 'height': 3.8},
        # Right wing - Service
        {'room_id': 'R007', 'name': 'Office', 'room_type': 'study', 'area': 11.4, 'x': 10, 'y': 6, 'width': 3.5, 'height': 3.26},
        {'room_id': 'R008', 'name': 'Master Bath', 'room_type': 'bathroom', 'area': 7.6, 'x': 0, 'y': 9, 'width': 3, 'height': 2.53},
        {'room_id': 'R009', 'name': 'Bath 2', 'room_type': 'bathroom', 'area': 5.7, 'x': 3.5, 'y': 9, 'width': 2.5, 'height': 2.28},
        {'room_id': 'R010', 'name': 'Bath 3', 'room_type': 'bathroom', 'area': 5.7, 'x': 6.5, 'y': 9, 'width': 2.5, 'height': 2.28},
        {'room_id': 'R011', 'name': 'Laundry', 'room_type': 'utility', 'area': 5.7, 'x': 14, 'y': 6, 'width': 2.5, 'height': 2.28},
        {'room_id': 'R012', 'name': 'Mudroom', 'room_type': 'hallway', 'area': 4.75, 'x': 17, 'y': 6, 'width': 2.5, 'height': 1.9},
        {'room_id': 'R013', 'name': 'Garage', 'room_type': 'garage', 'area': 33.25, 'x': 18.5, 'y': 4.5, 'width': 6, 'height': 5.54},
        {'room_id': 'R014', 'name': 'Storage', 'room_type': 'storage', 'area': 5.7, 'x': 9.5, 'y': 9.5, 'width': 2.5, 'height': 2.28},
    ],
    'adjacency_graph': {
        'edges': [
            ('R001', 'R008', {'edge_type': 'critical'}),  # Master BR to Master Bath
            ('R005', 'R006', {'edge_type': 'critical'}),  # Living to Kitchen
            ('R002', 'R009', {'edge_type': 'critical'}),  # Bedroom 2 to Bath 2
            ('R003', 'R010', {'edge_type': 'critical'}),  # Bedroom 3 to Bath 3
        ]
    }
}

# Strategy 2: Performance Optimized - Thermal zones
strategy_2 = {
    'rooms': [
        # Thermal zone 1 - Living (large windows)
        {'room_id': 'R005', 'name': 'Living/Dining', 'room_type': 'living', 'area': 40, 'x': 0, 'y': 0, 'width': 8, 'height': 5},
        {'room_id': 'R006', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 18.4, 'x': 8.5, 'y': 0, 'width': 4.6, 'height': 4},
        # Thermal zone 2 - Bedrooms (insulated)
        {'room_id': 'R001', 'name': 'Master BR', 'room_type': 'bedroom', 'area': 20, 'x': 0, 'y': 5.5, 'width': 5, 'height': 4},
        {'room_id': 'R002', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 14, 'x': 5.5, 'y': 5.5, 'width': 4, 'height': 3.5},
        {'room_id': 'R003', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 13, 'x': 10, 'y': 5.5, 'width': 3.8, 'height': 3.42},
        {'room_id': 'R004', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 13, 'x': 14.3, 'y': 5.5, 'width': 3.8, 'height': 3.42},
        # Thermal zone 3 - Service (minimal heating)
        {'room_id': 'R013', 'name': 'Garage', 'room_type': 'garage', 'area': 35, 'x': 13.5, 'y': 0, 'width': 6, 'height': 5.83},
        {'room_id': 'R007', 'name': 'Office', 'room_type': 'study', 'area': 12, 'x': 0, 'y': 10, 'width': 4, 'height': 3},
        {'room_id': 'R008', 'name': 'Master Bath', 'room_type': 'bathroom', 'area': 9.2, 'x': 4.5, 'y': 10, 'width': 3, 'height': 3.07},
        {'room_id': 'R009', 'name': 'Bath 2', 'room_type': 'bathroom', 'area': 6.9, 'x': 8, 'y': 10, 'width': 2.5, 'height': 2.76},
        {'room_id': 'R010', 'name': 'Bath 3', 'room_type': 'bathroom', 'area': 6.9, 'x': 11, 'y': 10, 'width': 2.5, 'height': 2.76},
        {'room_id': 'R011', 'name': 'Laundry', 'room_type': 'utility', 'area': 6, 'x': 14, 'y': 10, 'width': 2.5, 'height': 2.4},
        {'room_id': 'R012', 'name': 'Mudroom', 'room_type': 'hallway', 'area': 5, 'x': 17, 'y': 10, 'width': 2.5, 'height': 2},
        {'room_id': 'R014', 'name': 'Storage', 'room_type': 'storage', 'area': 6, 'x': 20, 'y': 0, 'width': 2.5, 'height': 2.4},
    ],
    'adjacency_graph': {
        'edges': [
            ('R005', 'R006', {'edge_type': 'critical'}),
            ('R001', 'R008', {'edge_type': 'critical'}),
            ('R002', 'R009', {'edge_type': 'critical'}),
            ('R003', 'R010', {'edge_type': 'critical'}),
        ]
    }
}

# Strategy 3: Structural Efficiency - Compact grid
strategy_3 = {
    'rooms': [
        # Compact 3x5 grid for structural efficiency
        {'room_id': 'R005', 'name': 'Living/Dining', 'room_type': 'living', 'area': 37.2, 'x': 0, 'y': 0, 'width': 7, 'height': 5.31},
        {'room_id': 'R006', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 14.88, 'x': 7.5, 'y': 0, 'width': 4, 'height': 3.72},
        {'room_id': 'R001', 'name': 'Master BR', 'room_type': 'bedroom', 'area': 18.6, 'x': 12, 'y': 0, 'width': 4.5, 'height': 4.13},
        
        {'room_id': 'R002', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 13.02, 'x': 0, 'y': 5.8, 'width': 3.8, 'height': 3.43},
        {'room_id': 'R003', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 12.09, 'x': 4.3, 'y': 5.8, 'width': 3.6, 'height': 3.36},
        {'room_id': 'R004', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 12.09, 'x': 8.4, 'y': 5.8, 'width': 3.6, 'height': 3.36},
        {'room_id': 'R007', 'name': 'Office', 'room_type': 'study', 'area': 11.16, 'x': 12.5, 'y': 4.6, 'width': 3.5, 'height': 3.19},
        
        {'room_id': 'R008', 'name': 'Master Bath', 'room_type': 'bathroom', 'area': 7.44, 'x': 0, 'y': 9.7, 'width': 3, 'height': 2.48},
        {'room_id': 'R009', 'name': 'Bath 2', 'room_type': 'bathroom', 'area': 5.58, 'x': 3.5, 'y': 9.7, 'width': 2.5, 'height': 2.23},
        {'room_id': 'R010', 'name': 'Bath 3', 'room_type': 'bathroom', 'area': 5.58, 'x': 6.5, 'y': 9.7, 'width': 2.5, 'height': 2.23},
        {'room_id': 'R011', 'name': 'Laundry', 'room_type': 'utility', 'area': 5.58, 'x': 9.5, 'y': 9.7, 'width': 2.5, 'height': 2.23},
        {'room_id': 'R012', 'name': 'Mudroom', 'room_type': 'hallway', 'area': 4.65, 'x': 12.5, 'y': 8.3, 'width': 2.5, 'height': 1.86},
        {'room_id': 'R013', 'name': 'Garage', 'room_type': 'garage', 'area': 32.55, 'x': 12.5, 'y': 10.7, 'width': 5.5, 'height': 5.92},
        {'room_id': 'R014', 'name': 'Storage', 'room_type': 'storage', 'area': 5.58, 'x': 15.5, 'y': 8.3, 'width': 2.5, 'height': 2.23},
    ],
    'adjacency_graph': {
        'edges': [
            ('R005', 'R006', {'edge_type': 'critical'}),
            ('R001', 'R008', {'edge_type': 'critical'}),
            ('R002', 'R009', {'edge_type': 'critical'}),
        ]
    }
}

# Strategy 4: Spatial Compactness - Circular/central living
strategy_4 = {
    'rooms': [
        # Central large living space
        {'room_id': 'R005', 'name': 'Living/Dining', 'room_type': 'living', 'area': 48, 'x': 5, 'y': 3, 'width': 8, 'height': 6},
        # Surrounding rooms
        {'room_id': 'R006', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 14.08, 'x': 13.5, 'y': 3, 'width': 4, 'height': 3.52},
        {'room_id': 'R001', 'name': 'Master BR', 'room_type': 'bedroom', 'area': 17.6, 'x': 0, 'y': 0, 'width': 4.5, 'height': 3.91},
        {'room_id': 'R002', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 12.32, 'x': 5, 'y': 0, 'width': 4, 'height': 3.08},
        {'room_id': 'R003', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 11.44, 'x': 9.5, 'y': 0, 'width': 3.8, 'height': 3.01},
        {'room_id': 'R004', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 11.44, 'x': 13.8, 'y': 0, 'width': 3.8, 'height': 3.01},
        {'room_id': 'R007', 'name': 'Office', 'room_type': 'study', 'area': 10.56, 'x': 0, 'y': 4.5, 'width': 4, 'height': 2.64},
        {'room_id': 'R008', 'name': 'Master Bath', 'room_type': 'bathroom', 'area': 7.04, 'x': 0, 'y': 7.5, 'width': 3, 'height': 2.35},
        {'room_id': 'R009', 'name': 'Bath 2', 'room_type': 'bathroom', 'area': 5.28, 'x': 5, 'y': 9.5, 'width': 2.5, 'height': 2.11},
        {'room_id': 'R010', 'name': 'Bath 3', 'room_type': 'bathroom', 'area': 5.28, 'x': 8, 'y': 9.5, 'width': 2.5, 'height': 2.11},
        {'room_id': 'R011', 'name': 'Laundry', 'room_type': 'utility', 'area': 5.28, 'x': 11, 'y': 9.5, 'width': 2.5, 'height': 2.11},
        {'room_id': 'R012', 'name': 'Mudroom', 'room_type': 'hallway', 'area': 4.4, 'x': 14, 'y': 9.5, 'width': 2.5, 'height': 1.76},
        {'room_id': 'R013', 'name': 'Garage', 'room_type': 'garage', 'area': 30.8, 'x': 13.5, 'y': 7, 'width': 6, 'height': 5.13},
        {'room_id': 'R014', 'name': 'Storage', 'room_type': 'storage', 'area': 5.28, 'x': 3.5, 'y': 7.5, 'width': 2.5, 'height': 2.11},
    ],
    'adjacency_graph': {
        'edges': [
            ('R005', 'R006', {'edge_type': 'critical'}),
            ('R001', 'R008', {'edge_type': 'critical'}),
            ('R002', 'R009', {'edge_type': 'critical'}),
            ('R003', 'R010', {'edge_type': 'critical'}),
            ('R005', 'R007', {'edge_type': 'critical'}),
        ]
    }
}

# Strategy 5: Balanced - Traditional layout
strategy_5 = {
    'rooms': [
        # Traditional floor plan layout
        {'room_id': 'R005', 'name': 'Living/Dining', 'room_type': 'living', 'area': 40, 'x': 0, 'y': 0, 'width': 7, 'height': 5.71},
        {'room_id': 'R006', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 16, 'x': 7.5, 'y': 0, 'width': 4.5, 'height': 3.56},
        {'room_id': 'R007', 'name': 'Office', 'room_type': 'study', 'area': 12, 'x': 12.5, 'y': 0, 'width': 4, 'height': 3},
        
        {'room_id': 'R001', 'name': 'Master BR', 'room_type': 'bedroom', 'area': 20, 'x': 0, 'y': 6.2, 'width': 5, 'height': 4},
        {'room_id': 'R008', 'name': 'Master Bath', 'room_type': 'bathroom', 'area': 8, 'x': 5.5, 'y': 6.2, 'width': 3, 'height': 2.67},
        {'room_id': 'R002', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 14, 'x': 9, 'y': 6.2, 'width': 4, 'height': 3.5},
        {'room_id': 'R003', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 13, 'x': 13.5, 'y': 3.5, 'width': 3.8, 'height': 3.42},
        
        {'room_id': 'R004', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 13, 'x': 0, 'y': 10.7, 'width': 3.8, 'height': 3.42},
        {'room_id': 'R009', 'name': 'Bath 2', 'room_type': 'bathroom', 'area': 6, 'x': 4.3, 'y': 10.7, 'width': 2.5, 'height': 2.4},
        {'room_id': 'R010', 'name': 'Bath 3', 'room_type': 'bathroom', 'area': 6, 'x': 7.3, 'y': 10.2, 'width': 2.5, 'height': 2.4},
        {'room_id': 'R011', 'name': 'Laundry', 'room_type': 'utility', 'area': 6, 'x': 10.3, 'y': 10.2, 'width': 2.5, 'height': 2.4},
        {'room_id': 'R012', 'name': 'Mudroom', 'room_type': 'hallway', 'area': 5, 'x': 13.3, 'y': 7.4, 'width': 2.5, 'height': 2},
        {'room_id': 'R013', 'name': 'Garage', 'room_type': 'garage', 'area': 35, 'x': 13.3, 'y': 10, 'width': 6, 'height': 5.83},
        {'room_id': 'R014', 'name': 'Storage', 'room_type': 'storage', 'area': 6, 'x': 16.3, 'y': 7.4, 'width': 2.5, 'height': 2.4},
    ],
    'adjacency_graph': {
        'edges': [
            ('R005', 'R006', {'edge_type': 'critical'}),
            ('R001', 'R008', {'edge_type': 'critical'}),
            ('R002', 'R009', {'edge_type': 'critical'}),
            ('R003', 'R010', {'edge_type': 'critical'}),
        ]
    }
}


strategies = {
    "strategy_1_functional_priority": strategy_1,
    "strategy_2_performance_optimized": strategy_2,
    "strategy_3_structural_efficiency": strategy_3,
    "strategy_4_spatial_compactness": strategy_4,
    "strategy_5_balanced": strategy_5
}


if __name__ == "__main__":
    # Set output directory
    output_dir = "c:\\Users\\sanvi\\OneDrive\\Desktop\\layout\\case_study\\level_1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    for strategy_id, strategy_data in strategies.items():
        output_path = os.path.join(output_dir, f"{strategy_id}_viz.png")
        generate_strategy_visualization(strategy_data, output_path)
    
    print("\n✅ All 5 Level 1 strategy visualizations generated with UNIQUE layouts!")
