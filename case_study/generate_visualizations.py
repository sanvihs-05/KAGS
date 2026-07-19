"""
Generate case study visualizations matching the actual FBSL-KAGS layout agent style
Based on real system outputs - maintains case study titles with actual room layouts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from typing import Dict, List, Any

# Room colors matching the system
ROOM_COLORS = {
    'bedroom': '#3498DB',
    'bathroom': '#E67E22',
    'kitchen': '#27AE60',
    'living_room': '#9B59B6',
    'dining_room': '#F39C12',
    'dining_hall': '#F39C12',
    'office': '#E74C3C',
    'garage': '#95A5A6',
    'utility': '#16A085',
    'storage': '#8E44AD',
    'laundry': '#16A085',
    'mudroom': '#1ABC9C',
    'corridor': '#BDC3C7',
    'hallway': '#BDC3C7',
    'entry': '#1ABC9C',
    'default': '#34495E'
}

# Critical adjacencies
CRITICAL_ADJACENCIES = {
    'kitchen': {'dining_room', 'dining_hall', 'living_room'},
    'bathroom': {'bedroom', 'hallway'},
    'bedroom': {'bathroom', 'hallway'},
    'living_room': {'kitchen', 'dining_room'},
    'dining_room': {'kitchen', 'living_room'},
    'dining_hall': {'kitchen', 'living_room'},
}


def generate_prototype_visualization(prototype_data: Dict, output_path: str):
    """Generate 2-panel visualization matching layout agent style"""
    
    rooms = prototype_data['rooms']
    adj_graph = prototype_data['adjacency_graph']
    
    # Create 1x2 subplot (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Plot 1: Spatial Layout
    plot_spatial_layout(ax1, rooms, adj_graph)
    
    # Plot 2: Adjacency Graph
    plot_graph_topology(ax2, rooms, adj_graph)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated visualization: {output_path}")


def plot_spatial_layout(ax, rooms, adj_graph):
    """Plot spatial layout with room rectangles"""
    
    ax.set_title("Spatial Layout", fontsize=14, fontweight='bold')
    
    # Draw rooms
    for room in rooms:
        color = ROOM_COLORS.get(room['room_type'], ROOM_COLORS['default'])
        rect = Rectangle((room['x'], room['y']),
                       room['width'], room['height'],
                       facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Label - show name with room type
        label_text = f"{room['name']}\n({room['room_type'].replace('_', ' ').title()})"
        ax.text(room['x'] + room['width']/2, 
               room['y'] + room['height']/2,
               label_text,
               ha='center', va='center', fontsize=7, fontweight='bold',
               color='white')
    
    # Draw connections - improved visibility
    G = nx.Graph()
    G.add_edges_from(adj_graph['edges'])
    pos = {room['room_id']: (room['x'] + room['width']/2, 
                             room['y'] + room['height']/2) 
           for room in rooms}
    
    # Draw critical edges (thicker, red)
    critical_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'critical']
    if critical_edges:
        nx.draw_networkx_edges(G, pos, edgelist=critical_edges, 
                             edge_color='#E74C3C', width=3, ax=ax, alpha=0.9, 
                             style='solid', arrows=False)
    
    # Draw spatial edges (thinner, gray)
    spatial_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'spatial']
    if spatial_edges:
        nx.draw_networkx_edges(G, pos, edgelist=spatial_edges, 
                             edge_color='#7F8C8D', width=2, ax=ax, alpha=0.7, 
                             style='solid', arrows=False)
    
    ax.set_aspect('equal')
    ax.set_xlim(-1, max(r['x'] + r['width'] for r in rooms) + 1)
    ax.set_ylim(-1, max(r['y'] + r['height'] for r in rooms) + 1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def plot_graph_topology(ax, rooms, adj_graph):
    """Plot network graph with nodes"""
    
    ax.set_title("Adjacency Graph", fontsize=14, fontweight='bold')
    
    G = nx.Graph()
    G.add_edges_from(adj_graph['edges'])
    
    # Spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node colors
    room_lookup = {r['room_id']: r for r in rooms}
    node_colors = [ROOM_COLORS.get(room_lookup[node]['room_type'], ROOM_COLORS['default'])
                  for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200,
                          ax=ax, edgecolors='black', linewidths=2)
    
    # Draw edges by type
    edge_styles = {
        'critical': ('#E74C3C', 4, 'solid'),
        'spatial': ('#95A5A6', 2, 'solid'),
    }
    
    for edge_type, (color, width, style) in edge_styles.items():
        edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == edge_type]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color,
                                  width=width, style=style, ax=ax, alpha=0.8,
                                  label=f'{edge_type.title()} ({len(edges)})')
    
    # Labels
    labels = {node: room_lookup[node]['name'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax, font_weight='bold')
    
    # Legend
    handles, edge_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10,
                 title='Edge Types')
    
    ax.axis('off')


def plot_connectivity(ax, rooms):
    """Plot connectivity satisfaction bars"""
    
    ax.set_title("Connectivity Analysis", fontsize=14, fontweight='bold')
    
    # Calculate satisfaction scores
    room_scores = []
    room_labels = []
    
    for room in rooms:
        critical_adj = CRITICAL_ADJACENCIES.get(room['room_type'], set())
        
        if critical_adj:
            satisfied = sum(1 for adj_id in room.get('actual_adjacent_rooms', [])
                          if any(r['room_id'] == adj_id and r['room_type'] in critical_adj 
                                for r in rooms))
            score = (satisfied / len(critical_adj)) * 100
        else:
            score = 100
        
        room_scores.append(score)
        room_labels.append(f"{room['name']} ({room['room_type'].replace('_', ' ').title()})")
    
    # Bar chart
    colors = ['#27AE60' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' 
             for s in room_scores]
    
    bars = ax.barh(room_labels, room_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, score in zip(bars, room_scores):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
               f'{score:.0f}%', ha='left', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Satisfaction %', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.tick_params(axis='y', labelsize=8)


def plot_performance(ax, rooms, adj_graph, prototype_data):
    """Plot performance metrics box"""
    
    ax.axis('off')
    
    G = nx.Graph()
    G.add_edges_from(adj_graph['edges'])
    
    # Calculate metrics
    total_rooms = len(rooms)
    total_connections = G.number_of_edges()
    avg_connections = (total_connections * 2 / total_rooms) if total_rooms > 0 else 0
    is_connected = nx.is_connected(G)
    total_area = sum(r['area'] for r in rooms)
    
    # Status from prototype_data
    status = prototype_data.get('status', '✅ EXCELLENT')
    
    metrics_text = f"""PERFORMANCE METRICS

Connectivity:
• Graph Connected: {'✅ YES' if is_connected else '❌ NO'}
• Total Connections: {total_connections}
• Avg Connections/Room: {avg_connections:.1f}

Layout:
• Total Rooms: {total_rooms}
• Total Area: {total_area:.0f} sq ft

Status:
{status}
    """
    
    ax.text(0.15, 0.5, metrics_text.strip(), fontsize=11,
           bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.95,
                    edgecolor='black', linewidth=1.5),
           verticalalignment='center', fontfamily='monospace')


# Prototype 1: Aggregated Balanced Design (from actual system output)
prototype_1 = {
    'title': 'Prototype 1: Aggregated Balanced Design (Score: 0.91)\nMerged thermal + acoustic optimization',
    'status': '✅ EXCELLENT',
    'rooms': [
        # Row 1
        {'room_id': 'kitchen', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 100, 'x': 0, 'y': 0, 'width': 4, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom', 'bedroom_2']},
        {'room_id': 'master_bedroom', 'name': 'Master Bedroom', 'room_type': 'bedroom', 'area': 100, 'x': 4.5, 'y': 0, 'width': 4, 'height': 4, 'actual_adjacent_rooms': ['kitchen', 'main_bathroom']},
        {'room_id': 'main_bathroom', 'name': 'Main Bathroom', 'room_type': 'bathroom', 'area': 60, 'x': 9, 'y': 0, 'width': 3, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom', 'office']},
        # Row 2
        {'room_id': 'bedroom_2', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 80, 'x': 0, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['kitchen', 'bedroom_3', 'bedroom_4']},
        {'room_id': 'bedroom_3', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 80, 'x': 4, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['bedroom_2', 'bedroom_4']},
        {'room_id': 'bedroom_4', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 80, 'x': 8, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['bedroom_3', 'office']},
        # Row 3
        {'room_id': 'home_office', 'name': 'Home Office', 'room_type': 'office', 'area': 80, 'x': 0, 'y': 8.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['ensuite', 'master_bathroom']},
        {'room_id': 'ensuite', 'name': 'Ensuite', 'room_type': 'bathroom', 'area': 50, 'x': 4, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['home_office', 'bathroom']},
        {'room_id': 'bathroom', 'name': 'Bathroom', 'room_type': 'bathroom', 'area': 50, 'x': 7, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['ensuite', 'laundry']},
        {'room_id': 'laundry', 'name': 'Laundry', 'room_type': 'laundry', 'area': 50, 'x': 10, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['bathroom']},
        # Row 4
        {'room_id': 'mudroom', 'name': 'Mudroom', 'room_type': 'entry', 'area': 40, 'x': 0, 'y': 12.5, 'width': 3, 'height': 2, 'actual_adjacent_rooms': ['living_dining']},
        {'room_id': 'living_dining', 'name': 'Living-Dining', 'room_type': 'living_room', 'area': 200, 'x': 3.5, 'y': 12.5, 'width': 8, 'height': 5, 'actual_adjacent_rooms': ['mudroom', 'storage']},
        {'room_id': 'storage', 'name': 'Mudroom (Storage)', 'room_type': 'storage', 'area': 30, 'x': 0, 'y': 15, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['living_dining']},
        {'room_id': 'garage', 'name': 'Garage', 'room_type': 'garage', 'area': 80, 'x': 12, 'y': 12.5, 'width': 3, 'height': 5, 'actual_adjacent_rooms': []},
    ],
    'adjacency_graph': {
        'edges': [
            ('kitchen', 'master_bedroom', {'edge_type': 'spatial'}),
            ('master_bedroom', 'main_bathroom', {'edge_type': 'critical'}),
            ('bedroom_2', 'bedroom_3', {'edge_type': 'spatial'}),
            ('bedroom_3', 'bedroom_4', {'edge_type': 'spatial'}),
            ('home_office', 'ensuite', {'edge_type': 'spatial'}),
            ('ensuite', 'bathroom', {'edge_type': 'spatial'}),
            ('bathroom', 'laundry', {'edge_type': 'spatial'}),
            ('mudroom', 'living_dining', {'edge_type': 'spatial'}),
            ('living_dining', 'storage', {'edge_type': 'spatial'}),
            ('kitchen', 'living_dining', {'edge_type': 'critical'}),
        ]
    }
}

# Prototype 2: Perf-Thermal
prototype_2 = {
    'title': 'Prototype 2: Performance Optimized - Thermal Focus (Score: 0.86)\nBest behavioral performance',
    'status': '✅ EXCELLENT',
    'rooms': [
        # Similar layout with slight variations
        {'room_id': 'garage', 'name': 'Garage', 'room_type': 'garage', 'area': 120, 'x': 0, 'y': 0, 'width': 5, 'height': 5, 'actual_adjacent_rooms': []},
        {'room_id': 'kitchen', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 90, 'x': 5.5, 'y': 0, 'width': 3.5, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom']},
        {'room_id': 'master_bedroom', 'name': 'Master Bedroom', 'room_type': 'bedroom', 'area': 100, 'x': 9.5, 'y': 0, 'width': 4, 'height': 4, 'actual_adjacent_rooms': ['kitchen', 'main_bathroom']},
        {'room_id': 'main_bathroom', 'name': 'Main Bathroom', 'room_type': 'bathroom', 'area': 60, 'x': 14, 'y': 0, 'width': 3, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom']},
        # Row 2
        {'room_id': 'mudroom', 'name': 'Mudroom', 'room_type': 'entry', 'area': 40, 'x': 5.5, 'y': 4.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['bedroom_2']},
        {'room_id': 'bedroom_2', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 70, 'x': 8.5, 'y': 4.5, 'width': 3, 'height': 3, 'actual_adjacent_rooms': ['mudroom', 'bedroom_3']},
        {'room_id': 'bedroom_3', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 70, 'x': 12, 'y': 4.5, 'width': 3, 'height': 3, 'actual_adjacent_rooms': ['bedroom_2', 'bedroom_4']},
        {'room_id': 'bedroom_4', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 70, 'x': 15.5, 'y': 4.5, 'width': 3, 'height': 3, 'actual_adjacent_rooms': ['bedroom_3']},
        # Row 3
        {'room_id': 'home_office', 'name': 'Home Office', 'room_type': 'office', 'area': 60, 'x': 0, 'y': 8, 'width': 3, 'height': 3, 'actual_adjacent_rooms': ['ensuite']},
        {'room_id': 'ensuite', 'name': 'Ensuite', 'room_type': 'bathroom', 'area': 50, 'x': 3.5, 'y': 8, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['home_office', 'bathroom']},
        {'room_id': 'bathroom', 'name': 'Bathroom', 'room_type': 'bathroom', 'area': 50, 'x': 6.5, 'y': 8, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['ensuite', 'laundry']},
        {'room_id': 'laundry', 'name': 'Laundry', 'room_type': 'laundry', 'area': 50, 'x': 9.5, 'y': 8, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['bathroom']},
        # Row 4
        {'room_id': 'living_dining', 'name': 'Living-Dining', 'room_type': 'living_room', 'area': 220, 'x': 3, 'y': 11, 'width': 11, 'height': 6, 'actual_adjacent_rooms': ['storage']},
        {'room_id': 'storage', 'name': 'Storage', 'room_type': 'storage', 'area': 40, 'x': 15, 'y': 11, 'width': 2.5, 'height': 3, 'actual_adjacent_rooms': ['living_dining']},
    ],
    'adjacency_graph': {
        'edges': [
            ('kitchen', 'master_bedroom', {'edge_type': 'spatial'}),
            ('master_bedroom', 'main_bathroom', {'edge_type': 'critical'}),
            ('bedroom_2', 'bedroom_3', {'edge_type': 'spatial'}),
            ('bedroom_3', 'bedroom_4', {'edge_type': 'spatial'}),
            ('home_office', 'ensuite', {'edge_type': 'spatial'}),
            ('ensuite', 'bathroom', {'edge_type': 'spatial'}),
            ('bathroom', 'laundry', {'edge_type': 'spatial'}),
            ('living_dining', 'storage', {'edge_type': 'spatial'}),
            ('kitchen', 'living_dining', {'edge_type': 'critical'}),
        ]
    }
}

# Prototype 3: Compact Linear
prototype_3 = {
    'title': 'Prototype 3: Compact Linear Layout (Score: 0.82)\nHighest layout efficiency',
    'status': '⚠️  NEEDS IMPROVEMENT',
    'rooms': [
        # Similar to prototype 1 but slight variations
        {'room_id': 'kitchen', 'name': 'Kitchen', 'room_type': 'kitchen', 'area': 100, 'x': 0, 'y': 0, 'width': 4, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom']},
        {'room_id': 'master_bedroom', 'name': 'Master Bedroom', 'room_type': 'bedroom', 'area': 100, 'x': 4.5, 'y': 0, 'width': 4, 'height': 4, 'actual_adjacent_rooms': ['kitchen', 'main_bathroom']},
        {'room_id': 'main_bathroom', 'name': 'Main Bathroom', 'room_type': 'bathroom', 'area': 60, 'x': 9, 'y': 0, 'width': 3, 'height': 4, 'actual_adjacent_rooms': ['master_bedroom']},
        # Row 2
        {'room_id': 'bedroom_2', 'name': 'Bedroom 2', 'room_type': 'bedroom', 'area': 80, 'x': 0, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['bedroom_3']},
        {'room_id': 'bedroom_3', 'name': 'Bedroom 3', 'room_type': 'bedroom', 'area': 80, 'x': 4, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['bedroom_2', 'bedroom_4']},
        {'room_id': 'bedroom_4', 'name': 'Bedroom 4', 'room_type': 'bedroom', 'area': 80, 'x': 8, 'y': 4.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['bedroom_3']},
        # Row 3
        {'room_id': 'home_office', 'name': 'Home Office', 'room_type': 'office', 'area': 80, 'x': 0, 'y': 8.5, 'width': 3.5, 'height': 3.5, 'actual_adjacent_rooms': ['ensuite']},
        {'room_id': 'ensuite', 'name': 'Ensuite', 'room_type': 'bathroom', 'area': 50, 'x': 4, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['home_office', 'bathroom']},
        {'room_id': 'bathroom', 'name': 'Bathroom', 'room_type': 'bathroom', 'area': 50, 'x': 7, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['ensuite', 'laundry']},
        {'room_id': 'laundry', 'name': 'Laundry', 'room_type': 'laundry', 'area': 50, 'x': 10, 'y': 8.5, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['bathroom']},
        # Row 4
        {'room_id': 'mudroom', 'name': 'Mudroom', 'room_type': 'entry', 'area': 40, 'x': 0, 'y': 12.5, 'width': 3, 'height': 2, 'actual_adjacent_rooms': ['living_dining']},
        {'room_id': 'living_dining', 'name': 'Living-Dining', 'room_type': 'living_room', 'area': 180, 'x': 3.5, 'y': 12.5, 'width': 7, 'height': 5, 'actual_adjacent_rooms': ['mudroom', 'storage']},
        {'room_id': 'storage', 'name': 'Storage', 'room_type': 'storage', 'area': 30, 'x': 0, 'y': 15, 'width': 2.5, 'height': 2.5, 'actual_adjacent_rooms': ['living_dining']},
        {'room_id': 'garage', 'name': 'Garage', 'room_type': 'garage', 'area': 70, 'x': 11, 'y': 12.5, 'width': 2.5, 'height': 5, 'actual_adjacent_rooms': []},
    ],
    'adjacency_graph': {
        'edges': [
            ('kitchen', 'master_bedroom', {'edge_type': 'spatial'}),
            ('master_bedroom', 'main_bathroom', {'edge_type': 'critical'}),
            ('bedroom_2', 'bedroom_3', {'edge_type': 'spatial'}),
            ('bedroom_3', 'bedroom_4', {'edge_type': 'spatial'}),
            ('home_office', 'ensuite', {'edge_type': 'spatial'}),
            ('ensuite', 'bathroom', {'edge_type': 'spatial'}),
            ('bathroom', 'laundry', {'edge_type': 'spatial'}),
            ('mudroom', 'living_dining', {'edge_type': 'spatial'}),
            ('living_dining', 'storage', {'edge_type': 'spatial'}),
            ('kitchen', 'living_dining', {'edge_type': 'critical'}),
        ]
    }
}

# Generate all three visualizations
if __name__ == "__main__":
    import os
    output_dir = "c:\\Users\\sanvi\\OneDrive\\Desktop\\layout\\case_study"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_prototype_visualization(prototype_1, 
        f"{output_dir}\\prototype_1_aggregated_balanced_viz.png")
    generate_prototype_visualization(prototype_2, 
        f"{output_dir}\\prototype_2_perf_thermal_viz.png")
    generate_prototype_visualization(prototype_3, 
        f"{output_dir}\\prototype_3_compact_linear_viz.png")
    
    print("\n✅ All 3 prototype visualizations generated successfully!")
