# FBSL-KAGS Framework: Complete Formula Reference

## 1. FBSL Ontology

### Function Set
```
F = {f₁, f₂, ..., fₙ}
fᵢ = (name, category, priority, activities, spatial_requirements)
```

### Behavior Set
```
B = {b₁, b₂, ..., bₘ}
bᵢ = (metric_name, target_value, actual_value, tolerance, category, unit)
Bₑ = Expected behaviors (from functions)
Bₛ = Actual behaviors (from structures)
```

### Structure Set
```
S = {s₁, s₂, ..., sₖ} ∪ R(sᵢ, sⱼ)
R(sᵢ, sⱼ) = Structural relationships
```

### Layout Set
```
L = {P, D, A, C}
P = {(x, y, z) | x, y, z ∈ ℝ}  (Position vectors)
D = {(width, height, depth)}  (Dimensions)
A = A[i,j] ∈ [0,1]  (Adjacency matrix)
C = {path₁, path₂, ...}  (Circulation paths)
```

---

## 2. Graph of Thought

### Graph Definition
```
G = (V, E)
V = {v₁, v₂, ..., vₙ}  (FBSL nodes)
E = {e₁, e₂, ..., eₘ}  (Transformation edges)
```

### Node Generation
```
f_node: P × C → N
P = problem state
C = context
N = new FBSL node
```

### Thought Expansion
```
f_expand(n) = {n₁, n₂, ..., nₖ}
```

### Path Score
```
Score(path) = Σᵢ w(eᵢ) × q(nᵢ)
w(eᵢ) = edge weight (transformation cost)
q(nᵢ) = node quality (composite score)
```

---

## 3. Complexity Calculation

### Requirements Complexity
```
C_req = 0.15×text_complexity + 0.25×constraint_complexity + 
        0.30×room_complexity + 0.15×adjacency_complexity + 
        0.15×area_complexity

text_complexity = min(1.0, text_length / 500.0)
constraint_complexity = min(1.0, constraint_count / 15.0)
room_complexity = min(1.0, room_count_estimate / 10.0)
adjacency_complexity = min(1.0, adjacency_count / 8.0)
area_complexity = min(1.0, area_spec_count / 5.0)
```

### FBSL Complexity
```
C_fbsl = 0.25×function_complexity + 0.20×behavior_complexity + 
         0.25×room_complexity + 0.15×function_interdependency + 
         0.15×behavior_diversity

function_complexity = min(1.0, function_count / 15.0)
behavior_complexity = min(1.0, behavior_count / 20.0)
room_complexity = min(1.0, room_count / 12.0)
function_interdependency = interdependency_count / max_possible_interdependencies
behavior_diversity = min(1.0, behavior_categories / 6.0)
```

### Combined Complexity
```
C_overall = 0.4 × C_req + 0.6 × C_fbsl
```

### Adaptive Parameters
```
depth = base_depth × scale_factor(C_overall)
breadth = base_breadth × scale_factor(C_overall) × component_scale
max_nodes = base_max_nodes × scale_factor(C_overall) × component_scale
target_prototypes = base_target × scale_factor(C_overall) × component_scale

component_scale = min(1.5, 1.0 + (room_count + function_count) / 20.0)

scale_factor(C):
  if C < 0.3: return 0.7
  if C < 0.6: return 1.0
  if C < 0.8: return 1.3
  else: return 1.5
```

---

## 4. Node Aggregation

### Aggregation Formula
```
Aggregate(N₁, N₂, ..., Nₖ) = argmax[Σᵢ λᵢ × Compatibility(Nᵢ, Nⱼ) × Quality(Nᵢ)]

λᵢ = 1 / (k - 1)
Quality(Nᵢ) = composite_score(Nᵢ)
```

### Compatibility Calculation
```
Compatibility(Nᵢ, Nⱼ) = 1 - (total_conflicts / total_elements)

total_conflicts = func_conflicts + behav_conflicts + struct_conflicts + layout_conflicts
total_elements = |Fᵢ ∪ Fⱼ| + |Bᵢ ∪ Bⱼ| + |Sᵢ ∪ Sⱼ| + |Lᵢ ∪ Lⱼ|
```

### Conflict Detection
```
func_conflict = 1 if fᵢ.conflicts_with contains fⱼ
behav_conflict = 1 if |bᵢ.target - bⱼ.target| / max(bᵢ.target, bⱼ.target) > 0.5
struct_conflict = 0.3 if materials incompatible
layout_conflict = area_deviation if |areaᵢ - areaⱼ| / avg_area > 0.3
```

---

## 5. Scoring Mechanisms

### Functional Adequacy
```
S_f = Σᵢ (wᵢ × Coverage(fᵢ)) / Σᵢ wᵢ

Coverage(fᵢ) = satisfied_behaviors(fᵢ) / total_behaviors(fᵢ)
wᵢ = function priority
```

### Behavioral Performance
```
For positive behaviors:
S_b = [Πᵢ min(1, bₛᵢ/bₑᵢ)]^(1/m)

For negative behaviors:
S_b = [Πᵢ min(1, bₑᵢ/bₛᵢ)]^(1/m)

m = number of behaviors
```

### Structural Feasibility
```
S_s = Σⱼ (Feasibility(sⱼ) × Compatibility(sⱼ, S\{sⱼ})) / |S|

Feasibility(sⱼ) = material_validity × dimensional_constraints
Compatibility(sⱼ, S\{sⱼ}) = 1 - conflict_count / |S|
```

### Layout Efficiency
```
S_l = α × Compactness + β × Circulation + γ × Adjacency_Satisfaction

Compactness = Total_Used_Area / Bounding_Box_Area
Circulation = Actual_Path_Length / Optimal_Path_Length
Adjacency_Satisfaction = Satisfied_Adjacencies / Total_Required_Adjacencies

α = 0.4, β = 0.3, γ = 0.3
```

### Sustainability
```
S_sust = Σᵢ (wᵢ × sustainability_metricᵢ)
```

### Composite Score
```
S_composite = [Σᵢ (wᵢ × Sᵢ^ρ)]^(1/ρ)

Sᵢ ∈ {S_f, S_b, S_s, S_l, S_sust}
wᵢ ∈ {0.3, 0.3, 0.2, 0.15, 0.05}
ρ = 1.0 (linear, compensatory)
```

---

## 6. Spatial Algorithms

### Force-Directed Layout
```
Force(rᵢ, rⱼ) = k_attraction × A[i,j] × d(rᵢ, rⱼ) - k_repulsion / d²(rᵢ, rⱼ)

k_attraction = 0.1
k_repulsion = 100.0
A[i,j] = adjacency weight ∈ [-1, 1]
d(rᵢ, rⱼ) = √[(xᵢ-xⱼ)² + (yᵢ-yⱼ)²]
```

### Position Update
```
pᵢ(t+1) = pᵢ(t) + η × Σⱼ Force(rᵢ, rⱼ)

η = 0.01 (learning rate)
Convergence: max_displacement < 0.01 or iterations ≥ 200
```

### Weighted Adjacency Matrix
```
w(i,j) = α × Functional_Dependency(i,j) + β × Traffic_Flow(i,j) + γ × Privacy_Requirement(i,j)

α = 0.4, β = 0.35, γ = 0.25
Normalized: α + β + γ = 1
```

### Adjacency Extraction
```
Required adjacency: func_dep = 1.0, traffic = 0.8
Preferred adjacency: func_dep = 0.6, traffic = 0.5
Avoid adjacency: privacy = 1.0 (inverted to negative weight)
```

### A* Pathfinding
```
f(n) = g(n) + h(n)

g(n) = actual distance from start (Euclidean)
h(n) = heuristic estimate to goal (Manhattan)
grid_resolution = 0.5 meters
```

### Circulation Efficiency
```
Efficiency = Direct_Distance / Actual_Path_Length

Direct_Distance = √[(x_end - x_start)² + (y_end - y_start)²]
```

### Compactness
```
Compactness = Total_Used_Area / Bounding_Box_Area

Total_Used_Area = Σᵢ (room_areaᵢ)
Bounding_Box_Area = (max_x - min_x) × (max_y - min_y)
```

---

## 7. Embeddings and Retrieval

### FBSL Embedding
```
e_fbsl = [e_f || e_b || e_s || e_l]

|| = concatenation
dim(e_fbsl) = dim(e_f) + dim(e_b) + dim(e_s) + dim(e_l)
```

### Cosine Similarity
```
sim(q, d) = (q · d) / (||q|| × ||d||)

q = query embedding
d = document embedding
||v|| = √(Σᵢ vᵢ²)
```

### Component-Weighted Relevance
```
Relevance(q, d) = Σᵢ wᵢ × sim(qᵢ, dᵢ)

qᵢ, dᵢ = component embeddings (F, B, S, L)
wᵢ = component weights (default: 0.25 each)
```

---

## 8. Reformulation Types

### Type 1: Structure Modification
```
S' = S + ΔS where ΔS minimizes |Bₛ - Bₑ|

Applied when: avg_deviation < 0.3
avg_deviation = mean(|bₛᵢ - bₑᵢ| / bₑᵢ) for unsatisfied behaviors
```

### Type 2: Behavior Relaxation
```
B'ₑ = Bₑ × (1 + tolerance)  (for positive behaviors)
B'ₑ = Bₑ / (1 + tolerance)  (for negative behaviors)
tolerance' = tolerance × 1.2

Applied when: 0.3 ≤ avg_deviation < 0.6
```

### Type 3: Function Redefinition
```
F' = Transform(F) such that feasible(B'ₑ) = true

priority' = priority × 0.8

Applied when: avg_deviation ≥ 0.6
```

---

## 9. Convergence Criteria

### Score-Based Convergence
```
|S_composite(t) - S_composite(t-1)| < ε

ε = 0.01 (convergence threshold)
```

### Adaptive Stopping
```
Stop if: (improvement < δ) AND (current_best > 0.7)
OR: high_scoring_count ≥ 3

improvement = |current_best - previous_best|
δ = 1e-3 (stopping threshold)
high_scoring_count = |{n | score(n) ≥ 0.9 × current_best}|
```

### Pareto Optimality
```
Pareto_Set = {x | ¬∃y : ∀i fᵢ(y) ≥ fᵢ(x) ∧ ∃j fⱼ(y) > fⱼ(x)}

fᵢ ∈ {functional_adequacy, behavioral_performance, 
      structural_feasibility, layout_efficiency, sustainability}
```

---

## 10. Pruning and Selection

### Score Threshold
```
threshold = max(score[target_count-1], top_score × 0.8)

Keep: {n | score(n) ≥ threshold}
If |Keep| > target_count: Keep top target_count
```

### High-Score Identification
```
high_score_threshold = top_score × 0.9
high_scoring = {n | score(n) ≥ high_score_threshold}

Aggregate if: |high_scoring| ≥ 2
```

---

## 11. Quality Calculation

### Node Quality
```
Quality(N) = 0.2×completeness + 0.3×satisfaction_rate + 
             0.3×composite_score + 0.2×consistency

completeness = 0.25×(has_F + has_B + has_S + has_L)
satisfaction_rate = satisfied_behaviors / total_behaviors
consistency = functions_with_behaviors / total_functions
```

---

## 12. Database Queries

### Project Retrieval
```sql
SELECT * FROM projects WHERE project_id = ?
```

### Node Retrieval
```sql
SELECT * FROM fbsl_nodes 
WHERE project_id = ? AND node_type = 'design_prototype'
ORDER BY composite_score DESC
LIMIT ?
```

### Evaluation Retrieval
```sql
SELECT * FROM evaluations 
WHERE node_id = ?
ORDER BY evaluated_at DESC
```

---

## 13. Parameter Summary

### GoT Parameters
- max_depth: 1-3 (adaptive)
- breadth: 2-5 (adaptive)
- max_nodes: 20-100+ (adaptive)
- epsilon: 1e-3
- delta: 1e-3
- patience: 2

### Scoring Parameters
- w_f: 0.3
- w_b: 0.3
- w_s: 0.2
- w_l: 0.15
- w_sust: 0.05
- rho: 1.0

### Spatial Parameters
- k_attraction: 0.1
- k_repulsion: 100.0
- learning_rate: 0.01
- max_iterations: 200
- grid_resolution: 0.5m
- alpha (adjacency): 0.4
- beta (adjacency): 0.35
- gamma (adjacency): 0.25

### Refinement Parameters
- max_iterations: 5
- convergence_threshold: 0.01
- Type 1 threshold: 0.3
- Type 2 threshold: 0.6
- tolerance_multiplier: 1.5 (Type 1), 1.2 (Type 2)
- priority_reduction: 0.8 (Type 3)

### Complexity Parameters
- Low threshold: 0.3
- Medium threshold: 0.6
- High threshold: 0.8
- Base depth: 2
- Base breadth: 3
- Base max_nodes: 50
- Base target: 5

