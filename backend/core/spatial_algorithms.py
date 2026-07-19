# core/spatial_algorithms.py
"""
Spatial Algorithms for FBSL Layout Generation
Phase 2 Complete Implementation:
- A* pathfinding for circulation ✓
- Weighted adjacency matrix computation ✓
- Force-directed layout optimization ✓
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from heapq import heappush, heappop
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """Represents a cell in the spatial grid"""
    x: int
    y: int
    walkable: bool = True
    cost: float = 1.0
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


@dataclass
class CirculationPath:
    """Represents a circulation path between two rooms"""
    start_room: str
    end_room: str
    path_points: List[Tuple[float, float]]
    length: float
    cost: float
    path_type: str  # 'corridor', 'open', 'direct'


class AStarPathfinder:
    """
    A* Algorithm for circulation path generation
    
    Uses cost function: f(n) = g(n) + h(n)
    where:
        g(n) = actual distance from start
        h(n) = heuristic estimate to goal (Manhattan distance)
    """
    
    def __init__(self, grid_resolution: float = 0.5):
        """
        Initialize A* pathfinder
        
        Args:
            grid_resolution: Grid cell size in meters
        """
        self.grid_resolution = grid_resolution
        
    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float]],  # (x, y, width, height)
        grid_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    ) -> Optional[CirculationPath]:
        """
        Find optimal circulation path using A* algorithm
        
        Args:
            start: Starting point (x, y)
            goal: Goal point (x, y)
            obstacles: List of rectangular obstacles
            grid_bounds: Boundaries of the search space
        
        Returns:
            CirculationPath or None if no path found
        """
        logger.info(f"A* pathfinding from {start} to {goal}")
        
        # Convert to grid coordinates
        grid = self._create_grid(obstacles, grid_bounds)
        start_cell = self._to_grid_coord(start[0], start[1], grid_bounds)
        goal_cell = self._to_grid_coord(goal[0], goal[1], grid_bounds)
        
        # A* search
        path_cells = self._astar_search(grid, start_cell, goal_cell)
        
        if not path_cells:
            logger.warning("No path found!")
            return None
        
        # Convert back to world coordinates
        path_points = [
            self._to_world_coord(cell.x, cell.y, grid_bounds)
            for cell in path_cells
        ]
        
        # Smooth path
        smoothed_path = self._smooth_path(path_points, obstacles)
        
        # Calculate metrics
        length = self._calculate_path_length(smoothed_path)
        cost = self._calculate_path_cost(smoothed_path, obstacles)
        
        return CirculationPath(
            start_room="",  # Will be set by caller
            end_room="",
            path_points=smoothed_path,
            length=length,
            cost=cost,
            path_type='corridor'
        )
    
    def _create_grid(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        bounds: Tuple[float, float, float, float]
    ) -> Dict[Tuple[int, int], GridCell]:
        """Create walkable grid with obstacles marked"""
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate grid dimensions
        width = int((max_x - min_x) / self.grid_resolution) + 1
        height = int((max_y - min_y) / self.grid_resolution) + 1
        
        # Initialize all cells as walkable
        grid = {}
        for x in range(width):
            for y in range(height):
                grid[(x, y)] = GridCell(x, y, walkable=True)
        
        # Mark obstacles
        for obs_x, obs_y, obs_w, obs_h in obstacles:
            # Convert obstacle bounds to grid coordinates
            obs_min_x = int((obs_x - min_x) / self.grid_resolution)
            obs_min_y = int((obs_y - min_y) / self.grid_resolution)
            obs_max_x = int((obs_x + obs_w - min_x) / self.grid_resolution)
            obs_max_y = int((obs_y + obs_h - min_y) / self.grid_resolution)
            
            # Mark cells as unwalkable
            for x in range(max(0, obs_min_x), min(width, obs_max_x + 1)):
                for y in range(max(0, obs_min_y), min(height, obs_max_y + 1)):
                    if (x, y) in grid:
                        grid[(x, y)].walkable = False
        
        return grid
    
    def _astar_search(
        self,
        grid: Dict[Tuple[int, int], GridCell],
        start: GridCell,
        goal: GridCell
    ) -> Optional[List[GridCell]]:
        """Core A* search algorithm"""
        
        # Priority queue: (f_score, counter, cell)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        # Tracking dictionaries
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        # Closed set
        closed_set = set()
        
        while open_set:
            # Get cell with lowest f_score
            _, _, current = heappop(open_set)
            
            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check neighbors (8-directional)
            for neighbor in self._get_neighbors(current, grid):
                if not neighbor.walkable or neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        # No path found
        return None
    
    def _heuristic(self, cell1: GridCell, cell2: GridCell) -> float:
        """Manhattan distance heuristic"""
        return abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)
    
    def _distance(self, cell1: GridCell, cell2: GridCell) -> float:
        """Euclidean distance between cells"""
        dx = cell1.x - cell2.x
        dy = cell1.y - cell2.y
        return np.sqrt(dx**2 + dy**2)
    
    def _get_neighbors(
        self,
        cell: GridCell,
        grid: Dict[Tuple[int, int], GridCell]
    ) -> List[GridCell]:
        """Get 8-directional neighbors"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = cell.x + dx, cell.y + dy
            if (nx, ny) in grid:
                neighbors.append(grid[(nx, ny)])
        
        return neighbors
    
    def _reconstruct_path(
        self,
        came_from: Dict[GridCell, GridCell],
        current: GridCell
    ) -> List[GridCell]:
        """Reconstruct path from goal to start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
    
    def _to_grid_coord(
        self,
        x: float,
        y: float,
        bounds: Tuple[float, float, float, float]
    ) -> GridCell:
        """Convert world coordinates to grid cell"""
        min_x, min_y, _, _ = bounds
        gx = int((x - min_x) / self.grid_resolution)
        gy = int((y - min_y) / self.grid_resolution)
        return GridCell(gx, gy)
    
    def _to_world_coord(
        self,
        gx: int,
        gy: int,
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        min_x, min_y, _, _ = bounds
        x = min_x + gx * self.grid_resolution
        y = min_y + gy * self.grid_resolution
        return (x, y)
    
    def _smooth_path(
        self,
        path: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float]]:
        """Smooth path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to connect to furthest visible point
            for j in range(len(path) - 1, i, -1):
                if self._line_of_sight(path[i], path[j], obstacles):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                # No line of sight, move to next point
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def _line_of_sight(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float]]
    ) -> bool:
        """Check if there's a clear line of sight between two points"""
        for obs_x, obs_y, obs_w, obs_h in obstacles:
            if self._line_intersects_rect(p1, p2, obs_x, obs_y, obs_w, obs_h):
                return False
        return True
    
    def _line_intersects_rect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        rx: float,
        ry: float,
        rw: float,
        rh: float
    ) -> bool:
        """Check if line segment intersects rectangle"""
        # Simple bounding box check first
        if max(p1[0], p2[0]) < rx or min(p1[0], p2[0]) > rx + rw:
            return False
        if max(p1[1], p2[1]) < ry or min(p1[1], p2[1]) > ry + rh:
            return False
        
        # More detailed check would go here
        # For now, conservative check
        return True
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length"""
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def _calculate_path_cost(
        self,
        path: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float, float]]
    ) -> float:
        """Calculate path cost (considering turns, length, etc.)"""
        if len(path) < 2:
            return 0.0
        
        cost = self._calculate_path_length(path)
        
        # Add penalty for turns
        if len(path) > 2:
            turn_penalty = 0.5 * (len(path) - 2)
            cost += turn_penalty
        
        return cost


class WeightedAdjacencyCalculator:
    """
    Calculate weighted adjacency matrix for spatial relationships
    
    Formula: w(i,j) = α×Functional_Dependency + β×Traffic_Flow + γ×Privacy_Requirement
    """
    
    def __init__(
        self,
        alpha: float = 0.4,  # Functional dependency weight
        beta: float = 0.35,  # Traffic flow weight
        gamma: float = 0.25  # Privacy requirement weight
    ):
        """
        Initialize adjacency calculator
        
        Args:
            alpha: Weight for functional dependency
            beta: Weight for traffic flow
            gamma: Weight for privacy requirement
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        logger.info(f"Adjacency calculator initialized (α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f})")
    
    def calculate_adjacency_matrix(
        self,
        rooms: Dict[str, Dict],
        functional_dependencies: Optional[Dict[Tuple[str, str], float]] = None,
        traffic_flows: Optional[Dict[Tuple[str, str], float]] = None,
        privacy_requirements: Optional[Dict[Tuple[str, str], float]] = None
    ) -> np.ndarray:
        """
        Calculate complete weighted adjacency matrix
        
        Args:
            rooms: Dictionary of room specifications
            functional_dependencies: Functional relationships between rooms [0,1]
            traffic_flows: Expected traffic between rooms (trips/day normalized to [0,1])
            privacy_requirements: Privacy needs between rooms [0,1] (higher = more separation needed)
        
        Returns:
            Weighted adjacency matrix where w[i,j] ∈ [-1, 1]
            Positive values indicate desired adjacency
            Negative values indicate desired separation
        """
        room_ids = list(rooms.keys())
        n = len(room_ids)
        
        # Initialize component matrices
        func_matrix = np.zeros((n, n))
        traffic_matrix = np.zeros((n, n))
        privacy_matrix = np.zeros((n, n))
        
        # Build component matrices
        for i, room1_id in enumerate(room_ids):
            for j, room2_id in enumerate(room_ids):
                if i >= j:
                    continue
                
                # Functional dependency
                if functional_dependencies:
                    key = (room1_id, room2_id)
                    reverse_key = (room2_id, room1_id)
                    func_matrix[i, j] = func_matrix[j, i] = functional_dependencies.get(
                        key, functional_dependencies.get(reverse_key, 0.0)
                    )
                
                # Traffic flow
                if traffic_flows:
                    key = (room1_id, room2_id)
                    reverse_key = (room2_id, room1_id)
                    traffic_matrix[i, j] = traffic_matrix[j, i] = traffic_flows.get(
                        key, traffic_flows.get(reverse_key, 0.0)
                    )
                
                # Privacy (inverted - higher privacy need = negative adjacency weight)
                if privacy_requirements:
                    key = (room1_id, room2_id)
                    reverse_key = (room2_id, room1_id)
                    privacy_val = privacy_requirements.get(
                        key, privacy_requirements.get(reverse_key, 0.0)
                    )
                    # Invert: high privacy need = want separation
                    privacy_matrix[i, j] = privacy_matrix[j, i] = -privacy_val
        
        # Compute weighted adjacency matrix
        adjacency_matrix = (
            self.alpha * func_matrix +
            self.beta * traffic_matrix +
            self.gamma * privacy_matrix
        )
        
        # Normalize to [-1, 1] range
        max_val = np.abs(adjacency_matrix).max()
        if max_val > 0:
            adjacency_matrix = adjacency_matrix / max_val
        
        logger.info(f"Adjacency matrix calculated: shape={adjacency_matrix.shape}, range=[{adjacency_matrix.min():.2f}, {adjacency_matrix.max():.2f}]")
        
        return adjacency_matrix
    
    def extract_adjacency_from_fbsl(
        self,
        rooms: Dict[str, Dict]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Extract adjacency relationships from FBSL room specifications
        
        Args:
            rooms: Dictionary of rooms with adjacency lists
        
        Returns:
            Tuple of (functional_deps, traffic_flows, privacy_reqs)
        """
        functional_deps = {}
        traffic_flows = {}
        privacy_reqs = {}
        
        room_ids = list(rooms.keys())
        
        for room1_id in room_ids:
            room1 = rooms[room1_id]
            
            # Required adjacencies = high functional dependency + high traffic
            if hasattr(room1, 'required_adjacencies'):
                for room2_id in room1.required_adjacencies:
                    key = tuple(sorted([room1_id, room2_id]))
                    functional_deps[key] = 1.0
                    traffic_flows[key] = 0.8
            
            # Preferred adjacencies = moderate functional dependency
            if hasattr(room1, 'preferred_adjacencies'):
                for room2_id in room1.preferred_adjacencies:
                    key = tuple(sorted([room1_id, room2_id]))
                    functional_deps[key] = 0.6
                    traffic_flows[key] = 0.5
            
            # Avoid adjacencies = high privacy requirement
            if hasattr(room1, 'avoid_adjacencies'):
                for room2_id in room1.avoid_adjacencies:
                    key = tuple(sorted([room1_id, room2_id]))
                    privacy_reqs[key] = 1.0
        
        return functional_deps, traffic_flows, privacy_reqs


class ForceDirectedLayout:
    """
    Force-directed layout optimization for room placement
    
    Uses physics simulation:
        Force(r_i, r_j) = k_attraction × A[i,j] × d(r_i, r_j) - k_repulsion / d²(r_i, r_j)
    
    Position update:
        p_i(t+1) = p_i(t) + η × Σ_j Force(r_i, r_j)
    """
    
    def __init__(
        self,
        k_attraction: float = 0.1,
        k_repulsion: float = 100.0,
        learning_rate: float = 0.01,
        max_iterations: int = 200
    ):
        """
        Initialize force-directed layout optimizer
        
        Args:
            k_attraction: Attraction force constant
            k_repulsion: Repulsion force constant
            learning_rate: Position update step size
            max_iterations: Maximum simulation iterations
        """
        self.k_attraction = k_attraction
        self.k_repulsion = k_repulsion
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        logger.info(f"Force-directed layout initialized (k_a={k_attraction}, k_r={k_repulsion})")
    
    def optimize_layout(
        self,
        room_specs: Dict[str, Dict],
        adjacency_matrix: np.ndarray,
        initial_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize room positions using force-directed algorithm
        
        Args:
            room_specs: Room specifications with area/dimensions
            adjacency_matrix: Weighted adjacency matrix
            initial_positions: Initial room positions (random if None)
            bounds: (min_x, min_y, max_x, max_y) boundaries
        
        Returns:
            Optimized positions dictionary {room_id: {x, y, width, height}}
        """
        room_ids = list(room_specs.keys())
        n = len(room_ids)
        
        # Initialize positions
        if initial_positions is None:
            positions = {
                room_id: {
                    'x': np.random.rand() * 50,
                    'y': np.random.rand() * 50
                }
                for room_id in room_ids
            }
        else:
            positions = {rid: {'x': pos[0], 'y': pos[1]} for rid, pos in initial_positions.items()}
        
        # Force-directed iterations
        for iteration in range(self.max_iterations):
            forces = {room_id: {'x': 0.0, 'y': 0.0} for room_id in room_ids}
            
            # Calculate forces
            for i, room1_id in enumerate(room_ids):
                for j, room2_id in enumerate(room_ids):
                    if i == j:
                        continue
                    
                    # Vector from room1 to room2
                    dx = positions[room2_id]['x'] - positions[room1_id]['x']
                    dy = positions[room2_id]['y'] - positions[room1_id]['y']
                    distance = np.sqrt(dx**2 + dy**2) + 0.01  # Avoid division by zero
                    
                    # Unit direction vector
                    ux = dx / distance
                    uy = dy / distance
                    
                    # Attraction force (proportional to adjacency weight and distance)
                    attraction_weight = adjacency_matrix[i, j]
                    if attraction_weight > 0:
                        f_attract = self.k_attraction * attraction_weight * distance
                        forces[room1_id]['x'] += f_attract * ux
                        forces[room1_id]['y'] += f_attract * uy
                    elif attraction_weight < 0:
                        # Negative weight = push away
                        f_repel = self.k_attraction * abs(attraction_weight) * distance
                        forces[room1_id]['x'] -= f_repel * ux
                        forces[room1_id]['y'] -= f_repel * uy
                    
                    # Universal repulsion force (prevents overlap)
                    f_repulse = self.k_repulsion / (distance ** 2)
                    forces[room1_id]['x'] -= f_repulse * ux
                    forces[room1_id]['y'] -= f_repulse * uy
            
            # Update positions
            max_displacement = 0.0
            for room_id in room_ids:
                dx = forces[room_id]['x'] * self.learning_rate
                dy = forces[room_id]['y'] * self.learning_rate
                
                positions[room_id]['x'] += dx
                positions[room_id]['y'] += dy
                
                # Apply bounds if specified
                if bounds:
                    min_x, min_y, max_x, max_y = bounds
                    positions[room_id]['x'] = np.clip(positions[room_id]['x'], min_x, max_x)
                    positions[room_id]['y'] = np.clip(positions[room_id]['y'], min_y, max_y)
                
                max_displacement = max(max_displacement, np.sqrt(dx**2 + dy**2))
            
            # Check convergence
            if max_displacement < 0.01:
                logger.info(f"Force-directed layout converged at iteration {iteration}")
                break
        
        # Add dimensions to positions
        for room_id in room_ids:
            positions[room_id]['width'] = room_specs[room_id].get('width', 5.0)
            positions[room_id]['height'] = room_specs[room_id].get('length', 5.0)
        
        logger.info(f"Force-directed layout complete: {len(positions)} rooms positioned")
        return positions


# Utility functions
def calculate_circulation_efficiency(
    paths: List[CirculationPath],
    room_positions: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate circulation efficiency score
    
    Formula: efficiency = Σ(optimal_length / actual_length) / n_paths
    """
    if not paths:
        return 0.0
    
    efficiencies = []
    for path in paths:
        # Calculate direct distance
        start_pos = room_positions.get(path.start_room, {'x': 0, 'y': 0})
        end_pos = room_positions.get(path.end_room, {'x': 0, 'y': 0})
        
        direct_dist = np.sqrt(
            (end_pos['x'] - start_pos['x'])**2 +
            (end_pos['y'] - start_pos['y'])**2
        )
        
        if direct_dist > 0:
            efficiency = direct_dist / max(path.length, direct_dist)
            efficiencies.append(efficiency)
    
    return np.mean(efficiencies) if efficiencies else 0.0


def calculate_compactness_score(
    room_positions: Dict[str, Dict[str, float]]
) -> float:
    """
    Calculate layout compactness as footprint squareness: min(W, H) / max(W, H)
    of the plan's bounding box. 1.0 = square plan; -> 0 = long/thin corridor.

    Replaces total_room_area / bounding_box_area, which for a gap-free tiled
    plan is always ~1.0 and cannot distinguish a square from a corridor.

    Positions carry the room CENTROID as (x, y) plus bbox width/height (or
    'length'), so each room's true extent is [x - w/2, x + w/2] x [y - h/2,
    y + h/2].
    """
    if not room_positions:
        return 0.0

    xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []
    for pos in room_positions.values():
        w = pos.get('width', 5.0)
        h = pos.get('height', pos.get('length', 5.0))
        cx, cy = pos['x'], pos['y']
        xs_lo.append(cx - w / 2.0); xs_hi.append(cx + w / 2.0)
        ys_lo.append(cy - h / 2.0); ys_hi.append(cy + h / 2.0)

    W = max(xs_hi) - min(xs_lo)
    H = max(ys_hi) - min(ys_lo)
    if W <= 0 or H <= 0:
        return 0.0

    return min(W, H) / max(W, H)