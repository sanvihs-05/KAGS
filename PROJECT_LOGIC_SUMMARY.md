# FBSL-KAGS Framework: Complete Project Logic Summary

## Table of Contents
1. [Core Ontology & Data Models](#core-ontology--data-models)
2. [Main Transformation Pipeline](#main-transformation-pipeline)
3. [Complexity Calculation & Adaptation](#complexity-calculation--adaptation)
4. [Graph of Thoughts (GoT) Mechanism](#graph-of-thoughts-got-mechanism)
5. [Multi-Criteria Scoring System](#multi-criteria-scoring-system)
6. [Spatial Layout Algorithms](#spatial-layout-algorithms)
7. [Node Aggregation](#node-aggregation)
8. [Iterative Refinement Process](#iterative-refinement-process)
9. [Behavior Calculation](#behavior-calculation)
10. [Knowledge Retrieval & Embeddings](#knowledge-retrieval--embeddings)
11. [Agent Implementations](#agent-implementations)
12. [Utilities & Support Systems](#utilities--support-systems)
13. [Database Schema](#database-schema)
14. [Complete Pipeline Workflow](#complete-pipeline-workflow)
15. [Parameter Reference](#parameter-reference)

---

## CORE ONTOLOGY & DATA MODELS

### FBSL Framework (Function-Behavior-Structure-Layout)

The project extends Gero's FBS (Function-Behavior-Structure) ontology by adding a **Layout** dimension, creating a complete spatial design representation:

#### **1. Function (F)**
- **Definition**: Teleological purpose of a design element
- **Components**:
  - `name`: Function identifier (e.g., "living room")
  - `category`: SPATIAL, ENVIRONMENTAL, SOCIAL, TECHNICAL, ECONOMIC
  - `priority`: Numerical weight [0, 1] indicating importance
  - `activities`: List of user activities (e.g., "dining", "watching TV")
  - `spatial_requirements`: Dict with min_area, preferred_area, max_area, height, shape_preference
  - `temporal_requirements`: Operating hours, seasonal variations, peak times
  - `dependencies`: depends_on, conflicts_with, enables relationships with other functions
  
- **Example**:
  ```
  Function(name="Living Room", priority=0.8, 
           spatial_requirements={preferred_area: 25, min_area: 15})
  ```

#### **2. Expected Behavior (Bₑ)**
- **Definition**: Performance characteristics derived from functions
- **Components**:
  - `metric_name`: Performance measure (e.g., "temperature", "daylight_factor")
  - `target_value`: Desired performance level (e.g., 21°C)
  - `category`: THERMAL, ACOUSTIC, LIGHTING, SPATIAL, STRUCTURAL, ENERGY, VENTILATION
  - `tolerance`: Acceptable deviation from target (default 10%)
  - `min_acceptable`, `max_acceptable`: Performance bounds
  - `derived_from_function`: Links behavior to its source function

- **Formula**: Bₑ = f(F) - Functions derive expected behaviors

- **Example**:
  ```
  Behavior(metric_name="Indoor Temperature", target_value=21, 
           category=THERMAL, tolerance=2, derived_from_function="living_room_id")
  ```

#### **3. Structure (S)**
- **Definition**: Physical/material components that enable behavior
- **Components**:
  - `name`: Element name (e.g., "exterior_wall_type_1")
  - `structure_type`: WALL, COLUMN, BEAM, SLAB, ROOF, FOUNDATION, PARTITION, MEP
  - `material_type`: Concrete, brick, wood, glass, etc.
  - `load_bearing`: Boolean (true if structural capacity required)
  - `dimensions`: {width, height, depth, area, thickness}
  - `acoustic_rating`: Sound transmission class (STC)
  - `thermal_properties`: U-value, R-value
  - `category`: "envelope", "frame", "partition", "mep"

- **Formula**: S → Bₛ (Structures exhibit actual behaviors)

- **Example**:
  ```
  Structure(name="Insulated Wall", structure_type=WALL, 
           material_type="brick", thermal_properties={U_value: 0.2})
  ```

#### **4. Actual Behavior (Bₛ)**
- **Definition**: Performance characteristics derived from structures
- **Calculation Process**: Bₛ = g(S) where structures exhibit measurable performance
- **Three calculation methods** (in priority order):
  1. **Physics-based** (from structure properties): U-values, STC ratings, etc.
  2. **Layout-based** (from room dimensions): Area, volume metrics
  3. **Conservative estimates**: 85% of target value fallback

- **Example**:
  ```
  For THERMAL behavior:
    U_value = 1 / R_value
    performance_ratio = actual_R_value / target_R_value
    actual_value = target_value × (0.7 + 0.3 × performance_ratio)
  ```

#### **5. Layout (L)**
- **Definition**: Spatial arrangement of rooms and circulation
- **Components**:
  - `P`: Position vectors {(x, y, z)} for each room center
  - `D`: Dimensions {width, height, depth} for each room
  - `A`: Adjacency matrix A[i,j] ∈ [-1, 1]
    - Positive values: desired adjacency
    - Negative values: desired separation
  - `C`: Circulation paths between rooms (A* pathfinding results)
  - `rooms`: Dictionary of Room objects with area, function, position
  - `circulation_efficiency`: Path efficiency metric

- **Formula**: L = h(S) - Structures arranged spatially

---

## MAIN TRANSFORMATION PIPELINE

### Transformation Chain

```
F → Bₑ → S → Bₛ → L → Evaluation → Reformulation → [Repeat]
     ↑__________________________________|
```

**Phase 1: F → Bₑ (Function to Expected Behavior)**
- Extract expected behaviors from function definitions
- Each function has associated performance requirements
- Behaviors defined based on function priorities and activities

**Phase 2: Bₑ → S (Expected Behavior to Structure)**
- Structure selection/generation to satisfy expected behaviors
- Multiple structure options evaluated for each behavior
- Structure compatibility checked across all behaviors

**Phase 3: S → Bₛ (Structure to Actual Behavior)**
- Physics calculations from structure properties
- Actual behavior calculation using material databases
- Comparison: does Bₛ ≈ Bₑ?

**Phase 4: S → L (Structure to Layout)**
- Spatial arrangement of structures
- Room positioning using force-directed algorithms
- Circulation path generation using A* pathfinding

**Phase 5: Evaluation**
- Multi-criteria scoring (functional, behavioral, structural, layout, sustainability)
- Composite score aggregation
- Convergence checking

**Phase 6: Reformulation (if Bₛ ≠ Bₑ)**
- Type 1: Modify structures (S' = S + ΔS)
- Type 2: Relax behavior targets (B'ₑ = Bₑ × tolerance)
- Type 3: Redefine functions (F' = Transform(F))

---

## COMPLEXITY CALCULATION & ADAPTATION

### Purpose
Dynamically adjust exploration parameters based on design problem complexity to optimize resource allocation

### Requirements Complexity (C_req)

**Formula**:
```
C_req = 0.15×text_complexity + 0.25×constraint_complexity + 
        0.30×room_complexity + 0.15×adjacency_complexity + 
        0.15×area_complexity
```

**Components**:
1. **text_complexity** = min(1.0, text_length / 500)
   - Normalized by 500 character baseline

2. **constraint_complexity** = min(1.0, constraint_count / 15)
   - Keywords: "must", "should", "require", "avoid", "ventilation", etc.

3. **room_complexity** = min(1.0, room_count_estimate / 10)
   - Extracted from keywords: "bedroom", "kitchen", "bathroom"
   - Also counts explicit numbers ("2 bedroom")

4. **adjacency_complexity** = min(1.0, adjacency_count / 8)
   - Keywords: "adjacent", "connected", "flow", "circulation"

5. **area_complexity** = min(1.0, area_spec_count / 5)
   - Regex pattern: `\d+-\d+ sqm|m²`

### FBSL Complexity (C_fbsl)

**Formula**:
```
C_fbsl = 0.25×function_complexity + 0.20×behavior_complexity + 
         0.25×room_complexity + 0.15×function_interdependency + 
         0.15×behavior_diversity
```

**Components**:
1. **function_complexity** = min(1.0, function_count / 15)
2. **behavior_complexity** = min(1.0, behavior_count / 20)
3. **room_complexity** = min(1.0, room_count / 12)
4. **function_interdependency** = interdependency_count / max_possible_interdependencies
   - Counts depends_on, conflicts_with, enables relationships
5. **behavior_diversity** = min(1.0, unique_categories / 6)
   - 6 main behavior categories

### Combined Complexity (C_overall)

**Formula**:
```
C_overall = 0.4 × C_req + 0.6 × C_fbsl
```

**Complexity Levels** (Classification):
- **Low**: C < 0.3 → Scale factor = 0.7
- **Medium**: 0.3 ≤ C < 0.6 → Scale factor = 1.0
- **High**: 0.6 ≤ C < 0.8 → Scale factor = 1.3
- **Very High**: C ≥ 0.8 → Scale factor = 1.5

### Adaptive Parameters

**Formula**:
```
adaptive_depth = base_depth × scale_factor(C)
adaptive_breadth = base_breadth × scale_factor(C) × component_scale
adaptive_max_nodes = base_max_nodes × scale_factor(C) × component_scale
adaptive_target_prototypes = base_target × scale_factor(C) × component_scale

component_scale = min(1.5, 1.0 + (room_count + function_count) / 20)
```

**Base Parameters** (for medium complexity):
- base_depth = 2
- base_breadth = 3
- base_max_nodes = 50
- base_aggregation_top_k = 3
- base_target_prototypes = 5

**Scaling by Component Count**:
- More rooms/functions → higher component_scale
- Example: 10 rooms + 8 functions → component_scale = 1.9×

---

## GRAPH OF THOUGHTS (GoT) MECHANISM

### Mathematical Foundation

**Graph Definition**:
```
G = (V, E)
V = {v₁, v₂, ..., vₙ}  (FBSL nodes representing design states)
E = {e₁, e₂, ..., eₘ}  (Transformation edges)
```

### Node Generation

**Formula**: 
```
f_node: P × C → N
where:
  P = problem state (initial requirements)
  C = context (constraints, precedents)
  N = new FBSL node (design alternative)
```

### Thought Expansion

**Four Transformation Strategies**:

1. **Functional Decomposition**
   - Splits functions by priority
   - Creates variants focused on high-priority functions
   - Exploration: different function orderings

2. **Behavioral Optimization**
   - Adjusts behavior tolerances
   - Relaxes constraints to improve satisfaction
   - Exploration: tolerance ranges

3. **Structural Variation**
   - Modifies material systems
   - Changes structural components
   - Exploration: material combinations

4. **Layout Permutation**
   - Generates alternative spatial arrangements
   - Explores room adjacency relationships
   - Exploration: layout configurations

### Path Scoring

**Formula**:
```
Score(path) = Σᵢ w(eᵢ) × q(nᵢ)

where:
  w(eᵢ) = edge weight (transformation cost)
  q(nᵢ) = node quality (composite score)
```

### GoT Expansion Algorithm

**Process**:
1. Add root (problem) node to graph
2. Initialize queue with (node, depth=0)
3. While queue not empty:
   - Pop node at current depth
   - Expand using 4 strategies → breadth children
   - Add children to graph with edges
   - Add children to queue with depth+1

**Expansion Control**:
- **Depth limit**: Configurable max_depth (typically 2-3)
- **Breadth**: Number of best children to expand (typically 2-5)
- **Node limit**: Maximum total nodes (adaptive, typically 20-100+)

### Score-Based Stopping Criteria

**Stopping Conditions**:
```
Stop if:
  (improvement < δ) AND (current_best > 0.7) AND (stagnation_count ≥ patience)
  OR
  (high_scoring_count ≥ 3)
  OR
  (depth reached)
  OR
  (node limit reached)

where:
  δ = stopping threshold (default 1e-3)
  ε = convergence threshold (default 1e-3)
  improvement = |current_best - previous_best|
  high_scoring_count = number of nodes with score ≥ 0.9 × best_score
  patience = stagnation tolerance (default 2)
```

**Adaptive Stopping Logic**:
- Stops faster if high scores already achieved (score > 0.7)
- Tolerates stagnation if multiple high-scoring alternatives exist
- More conservative stopping for lower complexity problems

---

## MULTI-CRITERIA SCORING SYSTEM

### Five Evaluation Criteria

#### **1. Functional Adequacy (S_f)**

**Formula**:
```
S_f = Σᵢ (wᵢ × Coverage(fᵢ)) / Σᵢ wᵢ

Coverage(fᵢ) = weighted_average_performance_ratio
             = Σⱼ(min(1, bₛⱼ/bₑⱼ)) / m
             where m = number of behaviors for function i
```

**Calculation**:
1. For each function, find related behaviors
2. Calculate performance ratio for each behavior: min(1, actual/target)
3. Average ratios to get Coverage ∈ [0, 1]
4. Weight by function priority
5. Aggregate across all functions

**Key Insight**: Measures DEGREE of function satisfaction, not just binary count

#### **2. Behavioral Performance (S_b)**

**Formula** (Geometric Mean):
```
S_b = exp(mean(log(min(1, bₛᵢ/bₑᵢ))))
    = [Πᵢ min(1, bₛᵢ/bₑᵢ)]^(1/m)

where m = number of behaviors
```

**Rationale**:
- Geometric mean more conservative than arithmetic
- All behaviors must perform reasonably well
- Single poor behavior significantly impacts score
- Stable even with many behaviors

**Behavior Handling**:
- **Positive behaviors** (more is better): ratio = actual/target
- **Negative behaviors** (less is better, e.g., cost): ratio = target/actual
- **Tolerance**: actual within target±tolerance is "satisfied"

#### **3. Structural Feasibility (S_s)**

**Formula**:
```
S_s = Σⱼ (Feasibility(sⱼ) × Compatibility(sⱼ, S\{sⱼ})) / |S|

where:
  Feasibility(sⱼ) = material_validity × dimensional_constraints
  Compatibility(sⱼ, S\{sⱼ}) = 1 - (conflicts / |S|)
```

**Calculation**:
1. Check material properties valid (U-values, STC, etc.)
2. Verify dimensions within structural constraints
3. Check compatibility with other structures
4. Penalize conflicts (incompatible materials, dimensional issues)
5. Average across all structures

#### **4. Layout Efficiency (S_l)**

**Formula**:
```
S_l = α × Compactness + β × Circulation + γ × Adjacency_Satisfaction

where α = 0.4, β = 0.3, γ = 0.3
```

**Components**:

a) **Compactness**:
```
Compactness = Total_Room_Area / Bounding_Box_Area

Total_Room_Area = Σᵢ (room_width × room_height)
Bounding_Box_Area = (max_x - min_x) × (max_y - min_y)

Range: [0, 1] (1 = perfectly compact, no wasted space)
```

b) **Circulation Efficiency**:
```
Circulation = Direct_Distance / Actual_Path_Length
           = avg(optimal_path / actual_path) across all circulation paths

Range: [0, 1] (1 = direct paths, no detours)
```

c) **Adjacency Satisfaction**:
```
Adjacency_Satisfaction = Satisfied_Adjacencies / Total_Required
                       = required_adj_met / required_adj_total

Range: [0, 1] (1 = all required adjacencies satisfied)
```

#### **5. Sustainability (S_sust)**

**Formula**:
```
S_sust = Σᵢ (wᵢ × sustainability_metricᵢ)

Metrics: energy_efficiency, material_sustainability, environmental_impact
```

### Composite Score Calculation

**Formula** (Rho-Parameter):
```
S_composite = [Σᵢ (wᵢ × Sᵢ^ρ)]^(1/ρ)

where:
  Sᵢ ∈ {S_f, S_b, S_s, S_l, S_sust}
  wᵢ = {0.3, 0.3, 0.2, 0.15, 0.05}  (default weights)
  ρ = compensation parameter
```

**Rho Parameter Interpretation**:
- **ρ → -∞**: Min operator (most conservative, no compensation)
- **ρ = -1**: Geometric mean (partially compensatory)
- **ρ = 0**: Weighted geometric mean: exp(Σ wᵢ × ln(Sᵢ))
- **ρ = 1**: Arithmetic mean (fully compensatory)
- **ρ → ∞**: Max operator (most lenient, full compensation)

**Default**: ρ = 1.0 (linear, fully compensatory)

---

## SPATIAL LAYOUT ALGORITHMS

### 1. Weighted Adjacency Matrix

**Purpose**: Define spatial relationship preferences between rooms

**Formula**:
```
w(i,j) = α × Functional_Dependency(i,j) + β × Traffic_Flow(i,j) + γ × Privacy_Requirement(i,j)

where α = 0.4, β = 0.35, γ = 0.25 (normalized: α + β + γ = 1)
```

**Components**:

a) **Functional Dependency**: How closely related are functions?
   - Required adjacency: 1.0
   - Preferred adjacency: 0.6
   - No requirement: 0.0
   - Functional incompatibility: -0.5

b) **Traffic Flow**: Expected movement between spaces
   - High traffic routes: 0.8-1.0
   - Moderate traffic: 0.4-0.6
   - Low traffic: 0.1-0.3
   - No traffic required: 0.0

c) **Privacy Requirement**: Need for spatial separation
   - High privacy need (negative, inverted):
     - Private spaces: -1.0
     - Semi-private: -0.5
   - Public spaces (no privacy need): 0.0

**Result Matrix**: w[i,j] ∈ [-1, 1]
- **Positive values**: Rooms should be adjacent
- **Negative values**: Rooms should be separated
- **Zero**: No preference

### 2. Force-Directed Layout Optimization

**Purpose**: Automatically position rooms based on adjacency weights

**Physics-Based Model**:

**Force Calculation**:
```
Force(rᵢ, rⱼ) = k_attraction × A[i,j] × d(rᵢ, rⱼ) - k_repulsion / d²(rᵢ, rⱼ)

where:
  k_attraction = 0.1 (attraction constant)
  k_repulsion = 100.0 (universal repulsion)
  A[i,j] = adjacency weight from matrix
  d(rᵢ, rⱼ) = Euclidean distance between room centers
```

**Position Update**:
```
pᵢ(t+1) = pᵢ(t) + η × Σⱼ Force(rᵢ, rⱼ)

where η = 0.01 (learning rate/step size)
```

**Convergence Criteria**:
- max_displacement < 0.01 meters, or
- iterations ≥ 200

**Algorithm**:
```
for iteration in 1..max_iterations:
  1. Calculate forces for all room pairs
  2. Update positions based on forces
  3. Clip positions to bounds
  4. Check convergence
  5. If converged: break
```

### 3. A* Pathfinding for Circulation

**Purpose**: Generate optimal circulation paths between rooms

**Cost Function**:
```
f(n) = g(n) + h(n)

where:
  g(n) = actual distance from start (Euclidean)
  h(n) = heuristic estimate to goal (Manhattan distance)
  f(n) = estimated total cost through node n
```

**Algorithm Steps**:
1. Convert world coordinates to grid (resolution 0.5m)
2. Mark obstacles (rooms as blocked cells)
3. Initialize open_set with start cell, f_score = 0
4. While open_set not empty:
   - Pop cell with lowest f_score
   - If cell = goal: reconstruct path
   - For each neighbor (8-directional):
     - Calculate tentative_g = g[current] + distance(current, neighbor)
     - If tentative_g < g[neighbor]:
       - Update g[neighbor] and f[neighbor]
       - Add to open_set
5. Path smoothing: Remove unnecessary waypoints via line-of-sight check

**Distance Metrics**:
- **Euclidean**: √((x₂-x₁)² + (y₂-y₁)²) for actual distances
- **Manhattan**: |x₂-x₁| + |y₂-y₁| for heuristic

**Path Smoothing**:
```
for each waypoint:
  try to connect to furthest visible waypoint
  if line_of_sight(current, furthest):
    skip intermediate waypoints
```

### 4. Circulation Efficiency Calculation

**Formula**:
```
Efficiency(path) = Direct_Distance / Actual_Path_Length

where:
  Direct_Distance = √[(x_end - x_start)² + (y_end - y_start)²]
  Actual_Path_Length = Σᵢ ||path[i+1] - path[i]||
```

**Range**: [0, 1]
- 1.0 = perfect straight path
- 0.5 = reasonable path with minor detours
- <0.3 = inefficient path with many turns

### 5. Compactness Score

**Formula**:
```
Compactness = Total_Used_Area / Bounding_Box_Area

Total_Used_Area = Σᵢ (room_width × room_height)
Bounding_Box = (max_x - min_x) × (max_y - min_y)
```

**Range**: [0, 1]
- 1.0 = minimal wasted space
- 0.7 = good efficiency
- <0.5 = significant wasted space

---

## NODE AGGREGATION

### Purpose
Combine multiple high-quality designs to create better alternatives by leveraging their strengths

### Selection Criteria

**High-Score Threshold**:
```
high_score_threshold = top_score × 0.9

High-scoring nodes: {n | score(n) ≥ high_score_threshold}
```

**Aggregation Condition**: 
- At least 2 high-scoring nodes must exist

### Aggregation Formula

**Formula**:
```
Aggregate(N₁, N₂, ..., Nₖ) = argmax[Σᵢ λᵢ × Compatibility(Nᵢ, Nⱼ) × Quality(Nᵢ)]

where:
  λᵢ = 1/(k-1)  (equal weighting)
  Quality(Nᵢ) = composite_score(Nᵢ)
  Compatibility(Nᵢ, Nⱼ) = 1 - (total_conflicts / total_elements)
```

### Compatibility Calculation

**Formula**:
```
total_conflicts = func_conflicts + behav_conflicts + struct_conflicts + layout_conflicts
total_elements = |Fᵢ ∪ Fⱼ| + |Bᵢ ∪ Bⱼ| + |Sᵢ ∪ Sⱼ| + |Lᵢ ∪ Lⱼ|
```

**Conflict Detection**:

1. **Function Conflicts**:
   - Check if fᵢ.conflicts_with contains fⱼ
   - Binary: 1 if conflict, 0 otherwise

2. **Behavior Conflicts**:
   - If |bᵢ.target - bⱼ.target| / max(bᵢ.target, bⱼ.target) > 0.5:
     - Conflict = 1
   - Otherwise: Conflict = 0

3. **Structure Conflicts**:
   - Incompatible materials: 0.3 penalty
   - Missing critical structures: 0.2 penalty

4. **Layout Conflicts**:
   - Area deviation penalty: |areaᵢ - areaⱼ| / avg_area
   - Room count difference > 2: 0.5 penalty
   - Incompatible adjacency matrices: penalize

### Aggregation Algorithm

**Process**:
1. Identify high-scoring nodes (≥ 90% of top score)
2. For each pair, calculate compatibility
3. Select top-k (typically 3-5) for aggregation
4. Merge FBSL components:
   - Functions: Union, prioritize higher-priority instances
   - Behaviors: Average targets, best actual values
   - Structures: Combine complementary structures
   - Layout: Force-directed merge of room positions

---

## ITERATIVE REFINEMENT PROCESS

### Three Reformulation Types (Based on Gero's Framework)

### **Type 1: Structure Modification** (Small Deviation)

**Applied When**: avg_deviation < 0.3

**Formula**:
```
S' = S + ΔS where ΔS minimizes |Bₛ - Bₑ|

Theoretical objective: find optimal structure additions to close behavior gap
```

**Algorithm**:
1. Calculate current deviation: Σ|bₛᵢ - bₑᵢ|
2. Generate candidate structures for unsatisfied behaviors
3. Try each candidate: measure new deviation
4. Select ΔS with minimum total deviation
5. Apply to node

**Candidate Structures**:
- **Thermal**: Insulation layer, material upgrade
- **Acoustic**: Sound partition with STC50+ rating
- **Lighting**: Window opening (1.5m × 2.0m standard)
- **Ventilation**: MEP duct system
- **Spatial**: Adjustable partition

**Expected Outcome**: Bs approaches Be through structure optimization

### **Type 2: Behavior Relaxation** (Moderate Deviation)

**Applied When**: 0.3 ≤ avg_deviation < 0.6

**Formula**:
```
For positive behaviors:
  B'ₑ = Bₑ × (1 + tolerance)

For negative behaviors (minimize):
  B'ₑ = Bₑ / (1 + tolerance)

Also increase tolerance:
  tolerance' = tolerance × 1.2
```

**Algorithm**:
1. For each unsatisfied behavior:
   - Expand target range by 20% tolerance increase
   - Recalculate satisfaction with new targets
2. Recalculate behavior satisfaction
3. Check if more behaviors now satisfied

**Example**:
```
Original: temperature_target = 21°C, tolerance = 2
New: temperature_target = 21°C, tolerance = 2.4 (20% increase)
Acceptable range: 18.6°C - 23.4°C (vs 19°C - 23°C)
```

**Expected Outcome**: More flexible targets allow existing structures to satisfy more behaviors

### **Type 3: Function Redefinition** (Large Deviation)

**Applied When**: avg_deviation ≥ 0.6

**Formula**:
```
F' = Transform(F) such that feasible(B'ₑ) = true

Priority reduction: priority' = priority × 0.8
```

**Algorithm**:
1. Identify low-priority functions causing the largest deviations
2. Reduce their priority: multiply by 0.8
3. This deprioritizes their associated behaviors
4. Recalculate composite scores with adjusted priorities

**Expected Outcome**: Redesign focus on achievable high-priority functions

### Refinement Loop

**Process**:
```
for iteration = 1 to max_iterations:
  1. Calculate Bₛ from S (structures exhibit behaviors)
  2. Identify unsatisfied behaviors
  3. Calculate current composite_score
  4. Check convergence: |score - prev_score| < ε
     If converged: break
  5. Calculate avg_deviation
  6. Apply reformulation:
     If avg_deviation < 0.3: Type 1 (Structure)
     Else if avg_deviation < 0.6: Type 2 (Behavior)
     Else: Type 3 (Function)
```

**Convergence Threshold**: ε = 0.01 (score change < 1%)
**Max Iterations**: 5 (default)

---

## BEHAVIOR CALCULATION

### Purpose
Derive actual behaviors (Bₛ) from structures to determine if design satisfies expected behaviors

### Three-Method Hierarchy

#### **Method 1: Physics-Based (Structures)**

**Calculation Order**:
1. **Thermal Performance** (Category: THERMAL)
   - Extract envelope structures (walls, roof, floor)
   - Get U-values from material properties
   - Calculate R-value: R = 1 / U
   - Average R-value: R_avg = Σ(R × area) / Σ(area)
   - Performance ratio: min(1.0, R_avg / target_R)
   - Actual value: target × (0.7 + 0.3 × performance_ratio)

2. **Acoustic Performance** (Category: ACOUSTIC)
   - Extract acoustic structures (walls, partitions)
   - Get STC ratings from properties or calculate:
     - Base STC from material
     - Add thickness effect: STC_adjustment = min(10, thickness × 50)
   - Average STC: STC_avg = mean([STC_structures])
   - Performance ratio: min(1.0, STC_avg / target_STC)
   - Actual value: target × performance_ratio

3. **Lighting Performance** (Category: LIGHTING)
   - Count window structures
   - Calculate window-to-floor ratio:
     - total_window_area = Σ(window_width × window_height)
     - total_floor_area = Σ(room_area)
     - ratio = total_window_area / total_floor_area
   - Daylight factor: DF = ratio × transmittance × 100
   - Performance ratio: min(1.0, DF / target_DF)
   - Actual value: target × performance_ratio

4. **Spatial Performance** (Category: SPATIAL)
   - For area metrics: sum room areas from layout
   - For privacy: count partitions and doors → privacy_score
   - For circulation: use circulation_efficiency metric

5. **Structural Performance** (Category: STRUCTURAL)
   - Check load-bearing structures present
   - Count components: foundation, columns/walls, beams, slabs
   - Completeness = components_present / total_required
   - Actual value: target × completeness

6. **Ventilation Performance** (Category: VENTILATION)
   - Check HVAC system present (MEP structures)
   - Score: 1.0 if HVAC, 0.85 if windows+vents, 0.75 if windows only, 0.4 if none
   - Actual value: target × ventilation_score

#### **Method 2: Layout-Based (Room Dimensions)**

**Used When**: No structures available but layout exists

**For Area Metrics**:
```
If behavior tied to function:
  related_rooms = rooms where function_id = behavior.derived_from_function
  total_area = Σ(room.area for room in related_rooms)
Else:
  total_area = Σ(room.area for all rooms)
```

**For Volume Metrics**:
```
total_volume = Σ(room.volume for all rooms where volume defined)
```

**For Count Metrics**:
```
room_count = len(rooms)
```

#### **Method 3: Conservative Estimate**

**Formula**:
```
actual_value = target_value × 0.85

Represents 85% achievement without structures
```

**Fallback**: actual_value = 1.0 (absolute fallback)

### Behavior Satisfaction Calculation

**Formula**:
```
actual_ratio = actual_value / target_value

For positive behaviors (more is better):
  satisfied = actual_ratio ≥ (1 - tolerance)
  satisfaction_degree = min(1.0, actual_ratio)

For negative behaviors (less is better, e.g., cost):
  satisfied = actual_ratio ≤ (1 + tolerance)
  satisfaction_degree = min(1.0, 1 / actual_ratio)
```

**Example**:
```
Temperature behavior:
  target = 21°C, tolerance = 0.1 (±10%)
  actual = 20.5°C
  ratio = 20.5 / 21 = 0.976
  satisfied = 0.976 ≥ (1 - 0.1) = 0.976 ≥ 0.9 → TRUE
  satisfaction_degree = 0.976
```

---

## KNOWLEDGE RETRIEVAL & EMBEDDINGS

### Component Embeddings

**FBSL Embedding Architecture**:
```
e_fbsl = [e_f || e_b || e_s || e_l]

where || = concatenation (preserves component identity)
```

**Components**:
1. **e_f**: Function embedding (from function names + categories)
2. **e_b**: Behavior embedding (from metric names + categories)
3. **e_s**: Structure embedding (from material + structure type)
4. **e_l**: Layout embedding (from room arrangement + circulation)

### Similarity Scoring

**Cosine Similarity** (for each component):
```
sim(q_component, d_component) = (q · d) / (||q|| × ||d||)

where || · || = L2 norm = √(Σ vᵢ²)
```

### Component-Weighted Relevance

**Formula**:
```
Relevance(q, d) = Σᵢ wᵢ × sim(qᵢ, dᵢ)

where:
  i ∈ {F, B, S, L}
  wᵢ ∈ {0.25, 0.25, 0.25, 0.25}  (default equal weights)
  sim(qᵢ, dᵢ) = cosine similarity for component i
```

### Precedent Retrieval from Knowledge Base

**Search Parameters**:
- Retrieve: Top-k similar spaces per function
- k = 3-5 precedents
- Similarity threshold:
  - High priority functions: > 0.7
  - Medium priority: > 0.5
  - Low priority: > 0.3

**Metadata Retrieved**:
- plan_id: Unique precedent identifier
- room_type: Function category
- translated_text: Natural language description
- embeddings: E_fbsl vector for further analysis

**Use Case**: Inform structure generation and layout validation with proven precedents

---

## AGENT IMPLEMENTATIONS

The system uses 7 specialized agents working in sequence:

### **1. Encoder Agent**
**Purpose**: Transform natural language requirements into structured FBSL problem nodes

**Process**:
- Input: User requirements (text)
- LLM Processing (Ollama with auto-model selection: prefers gemma3, falls back to llama3.1):
  - Extract rooms: "2 bedroom, 1 kitchen, 1 bathroom" → Function list
  - Extract area specs: "25-30 sqm living room" → spatial_requirements
  - Extract adjacencies: "kitchen adjacent to living room" → depends_on, required_adjacencies
  - Extract behaviors: "natural daylight", "ventilation" → Behavior list
- Finnish Floor Plan Mapping: Uses embeddings to enhance with precedents
- Output: Problem FBSL node with F, Bₑ, minimal S, initial L

**Key Methods**:
- `_test_llm_connection()`: Auto-detects Ollama models (HTTP + CLI fallback)
- `_parse_with_llm()`: Sends structured extraction prompt to LLM
- `_extract_functions()`: Creates Function objects from LLM output
- `_extract_behaviors()`: Creates Behavior objects with categories
- `_create_initial_layout()`: Synthesizes layout from room count estimate

### **2. Generalizer Agent**
**Purpose**: Decompose complex design problems into alternative design approaches

**Three Decomposition Strategies**:

a) **Decompose by Functional Zones**:
   - Categorizes functions into zones:
     - Private: bedrooms, studies (sleeping, privacy activities)
     - Social: living room, dining (socializing, entertainment)
     - Service: kitchen, bathroom (cooking, hygiene, cleaning)
     - Circulation: hallways, entries
   - Creates variants emphasizing zone organization
   - Variant 1: Compact (zones clustered)
   - Variant 2: Sequential (zones in order)
   - Variant 3: Modular (zones separated)

b) **Topology Variants**:
   - Linear topology: Rooms in sequence
   - Clustered topology: Rooms around central space
   - Grid topology: Regular grid arrangement
   - Free-form topology: Flexible arrangement

c) **Priority-Based Variants**:
   - Creates variants focusing on different function priorities
   - Variant 1: High-priority functions emphasized
   - Variant 2: Balanced approach
   - Variant 3: Secondary functions emphasized

**Output**: List of 4 alternative problem nodes, each with decomposed functions and deep-copied layouts

### **3. Research Agent**
**Purpose**: Retrieve relevant precedents and knowledge from Finnish floor plan database

**Process**:
1. For each function, generate composite embedding
2. Search FAISS index for similar spaces (top-k = 3-5)
3. Use component-weighted similarity:
   ```
   similarity = 0.25×func_sim + 0.25×behav_sim + 0.25×struct_sim + 0.25×layout_sim
   ```
4. Retrieve precedent metadata:
   - plan_id: Reference to floor plan
   - room_type: Function category
   - area, dimensions, adjacencies
   - thermal/acoustic properties

**Research Recommendations**:
- If avg_similarity > 0.7: Strong precedent available
- Suggest structure types based on precedent
- Recommend adjacency patterns
- Propose typical area ranges

**Output**: Precedent recommendations enhancing the node with validated design patterns

### **4. Encoder Agent (Re-run for Layout)**
After GoT exploration, if alternative nodes lack detailed layouts:
- Re-extracts room specifications from structure choices
- Synthesizes layout based on behavioral requirements
- Ensures all alternatives have complete L component

### **5. Scoring Agent** ✓ (Already documented in Multi-Criteria Scoring)

### **6. Layout Generation Agent**
**Purpose**: Convert FBSL specifications into spatial layouts

**Layout Synthesis Process**:
```
1. Extract room requirements from Functions
2. Calculate weighted adjacency matrix
3. Optimize room positions (force-directed)
4. Generate circulation paths (A*)
5. Calculate layout efficiency metrics
```

**Key Algorithms** (described in Spatial Algorithms section):
- Force-directed room positioning
- A* pathfinding for circulation
- Adjacency satisfaction calculation
- Compactness optimization

**Output**: Complete Layout with:
- Room positions (x, y, width, height)
- Circulation paths between rooms
- Layout efficiency score
- Floor plan visualization

### **7. Refinement Agent** ✓ (Already documented in Refinement Process)

### **8. Pipeline Orchestrator**
**Purpose**: Coordinate all agents and manage the complete workflow

**Orchestration Logic**:
```python
async def process_design_request(request):
    # Step 1: Encode
    problem_node = await encoder.encode(request)
    
    # Step 2: Research
    research = await research_agent.research(problem_node)
    
    # Step 3: Decompose
    alternatives = generalizer.decompose(problem_node)
    
    # Step 4: GoT Exploration (with complexity adaptation)
    complexity = complexity_calculator.calculate(request, problem_node)
    graph = await got_engine.generate_graph(
        problem_node,
        adaptive_depth=complexity.adaptive_depth,
        adaptive_breadth=complexity.adaptive_breadth,
        adaptive_max_nodes=complexity.adaptive_max_nodes
    )
    
    # Step 5: Extract leaf nodes from graph
    leaf_nodes = got_engine.get_leaf_nodes(graph)
    
    # Step 6: Score all nodes
    for node in leaf_nodes:
        scores = await scoring_agent.score_node(node)
    
    # Step 7: Prune low-scoring
    top_nodes = prune_by_score(leaf_nodes, threshold)
    
    # Step 8: Aggregation
    if high_scoring_count >= 2:
        aggregated = got_engine.aggregate_nodes(high_scoring)
        top_nodes.insert(0, aggregated)
    
    # Step 9: Refinement loop (per alternative)
    for node in top_nodes:
        refined_node, history = refiner.refine_node(node)
    
    # Step 10: Generate layouts
    for node in top_nodes:
        layout = await layout_agent.generate_layout(node)
    
    # Step 11: Final scoring
    for node in top_nodes:
        final_scores = await scoring_agent.score_node(node)
    
    # Step 12: Rank and return
    ranked = sort_by_composite_score(top_nodes)
    return ranked[0:3]  # Top 3 prototypes
```

---

---

## UTILITIES & SUPPORT SYSTEMS

### **Finnish Floor Plan Mapper (FinnishFBSLMapper)**
**Purpose**: Map Finnish architectural conventions to FBSL ontology

**Finnish Room Type Mapping**:
```
Room Type → Function Mapping:
- mh, bedroom → "provide_sleeping_space" (priority: 0.9)
- oh, olohuone → "provide_social_space" (priority: 0.85)
- keittiö, k → "provide_food_preparation" (priority: 0.95)
- kh, wc → "provide_bathing/sanitation" (priority: 0.9)
- vh → "provide_utility_space" (priority: 0.6)
- cl, varasto → "provide_storage" (priority: 0.5)
- eteinen, käytävä → "provide_circulation" (priority: 0.7)
```

**Behavior Mapping**:
- Extracts area ranges from precedent data
- Maps typical adjacency patterns
- Associates thermal/acoustic properties with materials

**Layout Synthesis**:
- Converts precedent data to Room objects
- Creates Layout with synthesized positions based on typical patterns
- Preserves area specifications and adjacency relationships

### **Vector Store Manager (VectorStoreManager)**
**Purpose**: Manage embeddings and similarity search infrastructure

**Components**:
1. **ChromaDB Integration**:
   - Persistent storage of new embeddings
   - Collection-based organization
   - Query interface for similarity search

2. **Finnish Embeddings Loading** (FinnishFloorPlanEmbeddingLoader):
   - Loads pre-computed embeddings from multi-modal dataset
   - Embedding types: spatial, text, visual, composite
   - Lazy-loading for performance
   - Statistics tracking: total_annotations, total_plans, room_type_distribution

3. **FAISS Index** (FaissClient):
   - Fast approximate nearest-neighbor search
   - Inner-product metric for relevance
   - Automatic index building/loading
   - Returns: {index, score, metadata}

**Similarity Search Process**:
```
1. Generate query embedding (F, B, S, L concatenated)
2. Search FAISS index for top-k similar embeddings
3. Return metadata: plan_id, room_type, area, adjacencies, properties
4. Optionally use component-weighted relevance for re-ranking
```

### **Embedding Loader (FinnishFloorPlanEmbeddingLoader)**
**Purpose**: Load and manage Finnish floor plan embeddings

**Data Structure**:
- Input: Multi-modal embeddings from Finnish floor plans
- Embedding types:
  - **spatial_embeddings**: Room geometry and arrangement
  - **text_embeddings**: Function descriptions and requirements
  - **visual_embeddings**: Visual features from floor plan images
  - **composite_embeddings**: Concatenation of all types for holistic similarity

**Statistics Available**:
- Total annotations per plan
- Room type distribution
- Area statistics per room type
- Common adjacency patterns

### **FAISS Client (FaissClient)**
**Purpose**: High-performance approximate nearest-neighbor search

**Index Structure**:
- Built from composite embeddings
- Uses inner-product metric (dot product similarity)
- Stores metadata: plan_id, room_type, area, properties

**Search Query**:
```python
results = faiss_client.search(embedding_vector, top_k=5)
# Returns: [{index, score, metadata}, ...]
```

**Score Interpretation**:
- Higher score = higher similarity (inner product of normalized vectors)
- Typical range: [0, 1] for normalized embeddings

---

## DATABASE SCHEMA

### **Projects Table**
```sql
CREATE TABLE projects (
    project_id UUID PRIMARY KEY,
    project_name VARCHAR(255),
    requirements TEXT,
    context JSONB,
    complexity_score FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### **FBSL Nodes Table**
```sql
CREATE TABLE fbsl_nodes (
    node_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects,
    parent_node_id UUID SELF-REFERENTIAL,
    node_type ENUM (problem, design_prototype, evaluation, research, refinement),
    
    -- Generation metadata
    generation_level INTEGER,
    iteration_number INTEGER,
    transformation_type ENUM (functional_decomposition, behavioral_optimization, 
                             structural_variation, layout_permutation, aggregation, refinement),
    
    -- FBSL Components (stored as JSONB)
    functions JSONB,      -- {func_id: {name, category, priority, activities, spatial_req, ...}}
    behaviors JSONB,      -- {behav_id: {metric_name, target, actual, category, ...}}
    structures JSONB,     -- {struct_id: {name, type, material, properties, ...}}
    layout JSONB,         -- {rooms: {...}, circulation: [...], efficiency: ...}
    
    -- Scores
    functional_score FLOAT,
    behavioral_score FLOAT,
    structural_score FLOAT,
    layout_score FLOAT,
    sustainability_score FLOAT,
    composite_score FLOAT,
    
    -- Quality metrics
    quality_score FLOAT,
    completeness FLOAT,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### **Evaluations Table**
```sql
CREATE TABLE evaluations (
    evaluation_id UUID PRIMARY KEY,
    node_id UUID REFERENCES fbsl_nodes,
    project_id UUID REFERENCES projects,
    
    -- Detailed scores by criterion
    functional_adequacy JSONB,    -- {coverage, satisfied_count, total_count, ...}
    behavioral_performance JSONB,  -- {geometric_mean, behavior_scores, ...}
    structural_feasibility JSONB,  -- {feasibility, compatibility, ...}
    layout_efficiency JSONB,       -- {compactness, circulation, adjacency, ...}
    sustainability JSONB,          -- {energy, materials, environment, ...}
    
    composite_score FLOAT,
    rank INTEGER,
    
    -- Analysis
    strengths TEXT[],
    weaknesses TEXT[],
    recommendations TEXT[],
    
    evaluated_at TIMESTAMP
)
```

### **Relationships**
```
One Project → Many FBSL Nodes (1:N)
One FBSL Node → One Evaluation (1:1)
FBSL Nodes form tree via parent_node_id
```

---

## COMPLETE PIPELINE WORKFLOW

### Phase 0: Input Processing

**Input**: Natural language requirements string

**Example**:
```
"2 bedroom apartment with kitchen, bathroom, living room. 
 Bedrooms should have natural light. Kitchen adjacent to living room.
 Living room area: 25-30 sqm. Bedrooms: 12-15 sqm each."
```

### Phase 1: Encoding & Research

**Step 1: Requirement Encoding** (Encoder Agent)
- Parse requirements via LLM
- Extract:
  - Functions: bedroom, kitchen, bathroom, living room
  - Behaviors: natural light, adjacency requirements, area specs
  - Spatial program
- Create initial FBSL node (Problem node)

**Step 2: Research** (Research Agent)
- Generate FBSL embedding
- Search knowledge base (Finnish floor plans)
- Retrieve similar spaces
- Enhance node with precedent findings
- Store embeddings for retrieval

### Phase 2: Design Space Generation

**Step 3: Complexity Calculation**
- Calculate C_req from requirements
- Calculate C_fbsl from problem node (if available)
- Determine C_overall and complexity_level
- Calculate adaptive parameters

**Step 4: GoT Graph Generation**
- Initialize root node (problem)
- Expand using 4 strategies:
  - Functional decomposition
  - Behavioral optimization
  - Structural variation
  - Layout permutation
- Score-based stopping when:
  - High scores plateaued, or
  - Multiple high-scoring alternatives exist, or
  - Max depth reached, or
  - Max nodes reached
- Graph G contains all generated FBSL nodes

**Step 5: Path Selection & Pruning**
- Find best paths through GoT using Score(path) = Σ w(eᵢ) × q(nᵢ)
- Extract top-k leaf nodes
- Score all leaf nodes using MCDA
- Prune low-scoring nodes:
  ```
  threshold = max(score[target-1], top_score × 0.8)
  keep: {n | score(n) ≥ threshold}
  ```

**Step 6: High-Score Aggregation**
- Identify high-scoring nodes: score ≥ 0.9 × top
- If ≥2 high-scoring exist:
  - Calculate compatibility matrix
  - Aggregate into single composite node
  - Score aggregated node
  - Add to alternatives

### Phase 3: Refinement Loop (Per Alternative)

**For Each Alternative Node**:

**Step 7: Expected Behaviors** (F → Bₑ)
- Already defined from functions

**Step 8: Structure Generation** (Bₑ → S)
- Generate/select structures for each behavior
- Ensure material compatibility

**Step 9: Behavior Calculation** (S → Bₛ)
- Calculate actual behaviors from structures
- Use 3-method hierarchy (physics → layout → estimate)
- Check satisfaction

**Step 10: Layout Generation** (S → L)
- Build weighted adjacency matrix from behaviors
- Force-directed optimization of room positions
- Generate A* circulation paths
- Calculate layout efficiency metrics

**Step 11: Convergence Loop**
```
for iteration = 1 to max_iterations:
  1. Calculate Bₛ from S
  2. Calculate composite_score
  3. Check: |score - prev_score| < 0.01
     If yes: converge → continue to Step 12
     If no: Apply reformulation
        avg_dev = |Bₛ - Bₑ|
        If avg_dev < 0.3: Type 1 (Structure)
        Else if avg_dev < 0.6: Type 2 (Behavior)
        Else: Type 3 (Function)
  prev_score = score
```

### Phase 4: Final Evaluation & Storage

**Step 12: Comprehensive Scoring**
- Score all final alternatives using full MCDA:
  - S_f: Functional adequacy
  - S_b: Behavioral performance (geometric mean)
  - S_s: Structural feasibility
  - S_l: Layout efficiency
  - S_sust: Sustainability
- Calculate composite: S_composite = [Σ(wᵢ × Sᵢ^ρ)]^(1/ρ)
- Rank alternatives

**Step 13: Pareto Front**
- Identify non-dominated solutions
- Maintain alternatives that excel in different criteria
- Enable trade-off analysis

**Step 14: Database Storage**
- Save project metadata
- Save FBSL nodes (F, B, S, L components)
- Save evaluations (scores, components, metadata)
- Create output report with top prototypes

### Output

**Deliverables**:
1. Top 3-5 design prototypes ranked by composite score
2. Each prototype includes:
   - FBSL components (functions, behaviors, structures, layout)
   - All 5 scores (functional, behavioral, structural, layout, sustainability)
   - Composite score
   - Floor plan visualization
   - Layout metrics (compactness, circulation efficiency, etc.)
3. Trade-off analysis showing Pareto front
4. Refinement history showing how each design converged

---

## PARAMETER REFERENCE

### Complexity Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| low_complexity_threshold | 0.3 | C_overall < 0.3 → low complexity |
| medium_complexity_threshold | 0.6 | 0.3 ≤ C_overall < 0.6 → medium |
| high_complexity_threshold | 0.8 | 0.6 ≤ C_overall < 0.8 → high |
| - | C_overall ≥ 0.8 → very_high |
| text_length_norm | 500 chars | Normalization baseline for text |
| constraint_count_norm | 15 | Normalization for constraints |
| room_count_norm | 10 | Normalization for rooms |
| adjacency_count_norm | 8 | Normalization for adjacencies |
| area_spec_norm | 5 | Normalization for area specs |
| function_count_norm | 15 | Normalization for functions |
| behavior_count_norm | 20 | Normalization for behaviors |
| room_count_norm_fbsl | 12 | Normalization for room count (FBSL) |
| behavior_diversity_norm | 6 | Number of behavior categories |

### GoT Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| max_depth | 1-4 | 2 (adaptive) | Maximum tree depth |
| breadth | 2-6 | 3 (adaptive) | Children per node |
| max_nodes | 20-200 | 50 (adaptive) | Total node budget |
| epsilon (ε) | 0.001-0.01 | 1e-3 | Convergence threshold |
| delta (δ) | 0.001-0.01 | 1e-3 | Stopping threshold |
| patience | 1-5 | 2 | Stagnation tolerance |
| high_score_threshold_mult | 0.8-0.95 | 0.9 | Multiplier for high scores |

### Scoring Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| w_functional_adequacy | 0.3 | Functional score weight |
| w_behavioral_performance | 0.3 | Behavioral score weight |
| w_structural_feasibility | 0.2 | Structural score weight |
| w_layout_efficiency | 0.15 | Layout score weight |
| w_sustainability | 0.05 | Sustainability weight |
| rho (ρ) | 1.0 | Compensation parameter |
| behavioral_tolerance_default | 0.1 (10%) | Default behavior tolerance |

### Spatial Algorithm Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| k_attraction | 0.1 | Force-directed attraction constant |
| k_repulsion | 100.0 | Force-directed repulsion constant |
| learning_rate (η) | 0.01 | Position update step size |
| max_iterations (layout) | 200 | Force-directed convergence limit |
| convergence_threshold (layout) | 0.01 m | Max displacement for convergence |
| grid_resolution | 0.5 m | A* pathfinding grid cell size |
| alpha (adjacency) | 0.4 | Functional dependency weight |
| beta (adjacency) | 0.35 | Traffic flow weight |
| gamma (adjacency) | 0.25 | Privacy requirement weight |

### Refinement Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_iterations | 5 | Maximum refinement iterations |
| convergence_threshold | 0.01 | Score difference for convergence |
| type1_threshold | 0.3 | avg_deviation < 0.3 → Type 1 |
| type2_threshold | 0.6 | 0.3 ≤ avg_deviation < 0.6 → Type 2 |
| type3_threshold | ≥ 0.6 | avg_deviation ≥ 0.6 → Type 3 |
| tolerance_multiplier_type1 | 1.5 | Structure attempt iterations |
| tolerance_multiplier_type2 | 1.2 | Behavior tolerance increase |
| priority_reduction_type3 | 0.8 | Function priority reduction |

### Material Properties Database

Example entries for behavior calculation:

```
Material: Concrete
  U-value: 2.0 W/m²K
  STC: 50 dB
  Density: 2400 kg/m³

Material: Brick
  U-value: 1.7 W/m²K
  STC: 45 dB
  Density: 1800 kg/m³

Material: Insulation
  U-value: 0.04 W/m²K
  STC: 20 dB
  Density: 30 kg/m³

Material: Glass
  U-value: 2.8 W/m²K
  STC: 30 dB
  Density: 2500 kg/m³
```

---

## Summary

This FBSL-KAGS framework implements a comprehensive computational design system that:

1. **Represents** architectural designs using 4-dimensional ontology (F, B, S, L)
2. **Explores** design space using Graph of Thought with adaptive parameters
3. **Evaluates** designs using multi-criteria scoring across 5 dimensions
4. **Optimizes** spatial layouts using physics-based algorithms
5. **Refines** iteratively through 3 reformulation types until convergence
6. **Ranks** alternatives using Pareto optimality

The system intelligently adapts all parameters based on problem complexity, automatically generating appropriate numbers of prototypes, adjusting exploration depth/breadth, and modifying pruning thresholds to match design problem characteristics.

All formulas, algorithms, and concepts are grounded in design theory (Gero's FBS ontology) while implemented with practical computational methods (force-directed layouts, A* pathfinding, MCDA scoring).
