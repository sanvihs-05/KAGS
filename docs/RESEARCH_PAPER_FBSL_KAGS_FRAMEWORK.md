# FBSL-KAGS Framework: A Knowledge-Augmented Graph-of-Thought System for Automated Architectural Layout Generation

## Abstract

This document provides a comprehensive technical description of the Function-Behavior-Structure-Layout Knowledge-Augmented Graph-of-Thought System (FBSL-KAGS), a computational framework for automated architectural layout generation that integrates design reasoning, knowledge retrieval, and multi-objective optimization.

---

## 1. Framework Architecture Overview

### 1.1 Core Ontology: FBSL (Function-Behavior-Structure-Layout)

The framework extends Gero's FBS (Function-Behavior-Structure) ontology to include Layout (L) as a fourth dimension, creating a complete representation for spatial design.

**Mathematical Representation:**

- **Function Set**: F = {f₁, f₂, ..., fₙ} where each fᵢ = (name, category, priority, activities, spatial_requirements)
- **Behavior Set**: B = {b₁, b₂, ..., bₘ} where bᵢ = (metric_name, target_value, actual_value, tolerance, category)
- **Structure Set**: S = {s₁, s₂, ..., sₖ} ∪ R(sᵢ, sⱼ) where R represents structural relationships
- **Layout Set**: L = {P, D, A, C} where:
  - P = {(x, y, z) | x, y, z ∈ ℝ} (position vectors)
  - D = {(width, height, depth)} (dimensions)
  - A = A[i,j] ∈ [0,1] (adjacency matrix)
  - C = {path₁, path₂, ...} (circulation paths)

### 1.2 Transformation Process

The FBSL transformation follows a causal chain:

```
F → Bₑ → S → Bₛ → L → Evaluation
     ↑__________|      |
     Reformulation     |
     ↑_________________|
```

Where:
- F → Bₑ: Functions derive expected behaviors
- Bₑ → S: Expected behaviors inform structure generation
- S → Bₛ: Structures exhibit actual behaviors
- S → L: Structures are spatially arranged into layouts
- Evaluation: Multi-criteria scoring
- Reformulation: Iterative refinement when Bₛ ≠ Bₑ

---

## 2. Graph of Thought (GoT) Mechanism

### 2.1 Mathematical Foundation

The GoT operates on a directed acyclic graph G = (V, E) where:
- V = {v₁, v₂, ..., vₙ} represents FBSL nodes (design states)
- E = {e₁, e₂, ..., eₘ} represents transformation edges

**Node Generation Function:**
```
f_node: P × C → N
```
Where P = problem state, C = context, N = new FBSL node

**Thought Expansion:**
```
f_expand(n) = {n₁, n₂, ..., nₖ}
```
Each nᵢ represents a different design alternative generated through transformation strategies.

**Graph Traversal Score:**
```
Score(path) = Σᵢ w(eᵢ) × q(nᵢ)
```
Where:
- w(eᵢ) = edge weight (transformation cost)
- q(nᵢ) = node quality (composite score)

### 2.2 Expansion Strategies

Four transformation strategies generate child nodes:

1. **Functional Decomposition**: Splits functions by priority, creating variants focused on high-priority functions
2. **Behavioral Optimization**: Relaxes behavior tolerances to improve overall satisfaction
3. **Structural Variation**: Modifies material systems and structural components
4. **Layout Permutation**: Generates alternative spatial arrangements

### 2.3 Adaptive Exploration Parameters

The system adapts exploration parameters based on design complexity:

**Complexity Score Calculation:**
```
C_overall = 0.4 × C_req + 0.6 × C_fbsl
```

Where:
- C_req = 0.15×text_complexity + 0.25×constraint_complexity + 0.30×room_complexity + 0.15×adjacency_complexity + 0.15×area_complexity
- C_fbsl = 0.25×function_complexity + 0.20×behavior_complexity + 0.25×room_complexity + 0.15×interdependency + 0.15×diversity

**Adaptive Parameters:**
```
depth = base_depth × scale_factor(C_overall)
breadth = base_breadth × scale_factor(C_overall) × component_scale
max_nodes = base_max_nodes × scale_factor(C_overall) × component_scale
target_prototypes = base_target × scale_factor(C_overall) × component_scale
```

Where scale_factor maps complexity levels:
- Low (C < 0.3): scale = 0.7
- Medium (0.3 ≤ C < 0.6): scale = 1.0
- High (0.6 ≤ C < 0.8): scale = 1.3
- Very High (C ≥ 0.8): scale = 1.5

### 2.4 Score-Based Stopping Criterion

Exploration terminates when:
```
(improvement < δ) AND (current_best > 0.7) OR (high_scoring_count ≥ 3)
```

Where:
- improvement = |current_best - previous_best|
- δ = convergence threshold (default: 1e-3)
- high_scoring_count = nodes with score ≥ 0.9 × current_best

---

## 3. Node Aggregation Mechanism

### 3.1 Aggregation Formula

Multiple high-quality nodes are combined using:

```
Aggregate(N₁, N₂, ..., Nₖ) = argmax[Σᵢ λᵢ × Compatibility(Nᵢ, Nⱼ) × Quality(Nᵢ)]
```

Where:
- λᵢ = 1/(k-1) (equal weighting)
- Compatibility(Nᵢ, Nⱼ) = 1 - (|Conflict(Lᵢ, Lⱼ)| / |Lᵢ ∪ Lⱼ|)
- Quality(Nᵢ) = composite_score(Nᵢ)

### 3.2 Compatibility Calculation

Compatibility considers four conflict dimensions:

```
Compatibility = 1 - (total_conflicts / total_elements)
```

Where conflicts are measured across:
1. **Function Conflicts**: Explicit conflicts + activity incompatibilities
2. **Behavior Conflicts**: Incompatible target values (>50% deviation)
3. **Structure Conflicts**: Incompatible material systems
4. **Layout Conflicts**: Area deviation (>30%) + room count mismatch (>2 rooms)

### 3.3 Selection Strategy

High-scoring nodes (score ≥ 0.9 × top_score) are aggregated together, with top_k determined adaptively:
```
aggregation_top_k = min(adaptive_k, len(high_scoring_nodes))
```

---

## 4. Scoring Mechanisms

### 4.1 Multi-Criteria Decision Analysis (MCDA)

Five criteria are evaluated:

#### 4.1.1 Functional Adequacy Score

```
S_f = Σᵢ (wᵢ × Coverage(fᵢ)) / Σᵢ wᵢ
```

Where:
- Coverage(fᵢ) = degree to which function fᵢ is satisfied ∈ [0,1]
- Coverage calculated as: satisfied_behaviors(fᵢ) / total_behaviors(fᵢ)
- wᵢ = function priority

#### 4.1.2 Behavioral Performance Score

For positive behaviors (more is better):
```
S_b = [Πᵢ min(1, bₛᵢ/bₑᵢ)]^(1/m)
```

For negative behaviors (less is better, e.g., cost):
```
S_b = [Πᵢ min(1, bₑᵢ/bₛᵢ)]^(1/m)
```

Where m = number of behaviors

#### 4.1.3 Structural Feasibility Score

```
S_s = Σⱼ (Feasibility(sⱼ) × Compatibility(sⱼ, S\{sⱼ})) / |S|
```

Where:
- Feasibility(sⱼ) = material_properties_validity × dimensional_constraints
- Compatibility(sⱼ, S\{sⱼ}) = 1 - conflict_count / |S|

#### 4.1.4 Layout Efficiency Score

```
S_l = α × Compactness + β × Circulation + γ × Adjacency_Satisfaction
```

Where:
- **Compactness** = Total_Used_Area / Bounding_Box_Area
- **Circulation** = Actual_Path_Length / Optimal_Path_Length
- **Adjacency_Satisfaction** = Satisfied_Adjacencies / Total_Required_Adjacencies
- α = 0.4, β = 0.3, γ = 0.3 (default weights)

#### 4.1.5 Sustainability Score

```
S_sust = Σᵢ (wᵢ × sustainability_metricᵢ)
```

Where metrics include energy efficiency, material sustainability, and environmental impact.

### 4.2 Composite Score Calculation

```
S_composite = [Σᵢ (wᵢ × Sᵢ^ρ)]^(1/ρ)
```

Where:
- ρ determines compensation degree
- ρ = 1: Linear (fully compensatory)
- ρ = -1: Geometric mean (partially compensatory)
- ρ → ∞: Min operator (non-compensatory, not implemented)

Default weights: w_f = 0.3, w_b = 0.3, w_s = 0.2, w_l = 0.15, w_sust = 0.05

---

## 5. Spatial Layout Algorithms

### 5.1 Force-Directed Layout Optimization

Room placement uses physics-based simulation:

**Force Calculation:**
```
Force(rᵢ, rⱼ) = k_attraction × A[i,j] × d(rᵢ, rⱼ) - k_repulsion / d²(rᵢ, rⱼ)
```

Where:
- k_attraction = 0.1 (attraction constant)
- k_repulsion = 100.0 (repulsion constant)
- A[i,j] = adjacency weight ∈ [-1, 1]
- d(rᵢ, rⱼ) = Euclidean distance between room centers

**Position Update:**
```
pᵢ(t+1) = pᵢ(t) + η × Σⱼ Force(rᵢ, rⱼ)
```

Where:
- η = 0.01 (learning rate)
- Iterations continue until max_displacement < 0.01 or max_iterations = 200

### 5.2 Weighted Adjacency Matrix Construction

```
w(i,j) = α × Functional_Dependency(i,j) + β × Traffic_Flow(i,j) + γ × Privacy_Requirement(i,j)
```

Where:
- α = 0.4 (functional dependency weight)
- β = 0.35 (traffic flow weight)
- γ = 0.25 (privacy requirement weight)
- Weights normalized: α + β + γ = 1

**Adjacency Extraction:**
- Required adjacencies: func_dep = 1.0, traffic = 0.8
- Preferred adjacencies: func_dep = 0.6, traffic = 0.5
- Avoid adjacencies: privacy = 1.0 (inverted to negative weight)

### 5.3 Circulation Path Generation

Uses A* pathfinding algorithm:

**Cost Function:**
```
f(n) = g(n) + h(n)
```

Where:
- g(n) = actual distance from start (Euclidean)
- h(n) = heuristic estimate to goal (Manhattan distance)

**Grid Resolution:** 0.5 meters

**Path Smoothing:** Line-of-sight optimization removes unnecessary waypoints

**Circulation Efficiency:**
```
Efficiency = Direct_Distance / Actual_Path_Length
```

---

## 6. Research Agent and Knowledge Retrieval

### 6.1 FBSL Embedding Generation

**Component Embeddings:**
Each FBSL component is embedded separately:
- e_f = embedding(Function)
- e_b = embedding(Behavior)
- e_s = embedding(Structure)
- e_l = embedding(Layout)

**Composite Embedding:**
```
e_fbsl = [e_f || e_b || e_s || e_l]
```

Where || denotes concatenation, preserving component-specific information.

**Embedding Dimensions:**
- Function embedding: variable (from Finnish floor plan embeddings)
- Behavior embedding: variable
- Structure embedding: variable
- Layout embedding: variable
- Composite: sum of component dimensions

### 6.2 Similarity Scoring

**Cosine Similarity:**
```
sim(q, d) = (q · d) / (||q|| × ||d||)
```

**Component-Weighted Relevance:**
```
Relevance(q, d) = Σᵢ wᵢ × sim(qᵢ, dᵢ)
```

Where:
- qᵢ, dᵢ = component embeddings (F, B, S, L)
- wᵢ = component weights (default: 0.25 each)

### 6.3 Precedent Retrieval

Retrieves top-k similar spaces from Finnish floor plan database:
- Search depth: 3-5 precedents per function
- Similarity threshold: > 0.5 for medium priority, > 0.7 for high priority
- Metadata includes: plan_id, room_type, translated_text, function

---

## 7. Iterative Refinement Process

### 7.1 Reformulation Types

Based on Gero's framework, three reformulation types:

#### 7.1.1 Type 1: Structure Modification

**Formula:**
```
S' = S + ΔS where ΔS minimizes |Bₛ - Bₑ|
```

**Decision Logic:**
- Applied when avg_deviation < 0.3 (small deviation)
- Tests structure modifications incrementally
- Selects ΔS that minimizes total deviation

**Structure Addition:**
- Thermal: insulation structure
- Acoustic: partition with STC50 rating
- Lighting: window opening (1.5m × 2.0m)
- Ventilation: MEP duct system
- Spatial: adjustable partition

#### 7.1.2 Type 2: Behavior Relaxation

**Formula:**
```
B'ₑ = Bₑ × (1 + tolerance)
```

**Implementation:**
- For positive behaviors: B'ₑ = Bₑ × (1 + tolerance)
- For negative behaviors: B'ₑ = Bₑ / (1 + tolerance)
- Also increases tolerance: tolerance' = tolerance × 1.2

**Decision Logic:**
- Applied when 0.3 ≤ avg_deviation < 0.6 (moderate deviation)

#### 7.1.3 Type 3: Function Redefinition

**Formula:**
```
F' = Transform(F) such that feasible(B'ₑ) = true
```

**Implementation:**
- Reduces function priority: priority' = priority × 0.8
- Applied when avg_deviation ≥ 0.6 (large deviation)

### 7.2 Refinement Loop

```
For iteration = 1 to max_iterations:
    1. Calculate Bₛ from S (S → Bₛ)
    2. Identify unsatisfied behaviors
    3. Calculate current score
    4. If |score - previous_score| < ε: converge
    5. Else: Apply reformulation (Type 1, 2, or 3)
```

**Convergence Threshold:** ε = 0.01
**Max Iterations:** 5 (default)

---

## 8. Convergence Criteria

### 8.1 Score-Based Convergence

```
|S_composite(t) - S_composite(t-1)| < ε
```

Where ε = 0.01 (default convergence threshold)

### 8.2 Adaptive Stopping

Convergence also occurs when:
- High scores achieved (score > 0.7) with no improvement
- Multiple high-scoring alternatives found (≥3 nodes ≥ 0.9 × best)
- Maximum iterations reached

### 8.3 Pareto Optimality

Maintains Pareto front for multi-objective optimization:

```
Pareto_Set = {x | ¬∃y : ∀i fᵢ(y) ≥ fᵢ(x) ∧ ∃j fⱼ(y) > fⱼ(x)}
```

Where fᵢ represents objective functions:
- f₁ = functional_adequacy
- f₂ = behavioral_performance
- f₃ = structural_feasibility
- f₄ = layout_efficiency
- f₅ = sustainability

---

## 9. Score-Based Pruning and Aggregation

### 9.1 Pruning Strategy

Low-scoring alternatives are pruned using:

**Score Threshold:**
```
threshold = max(score[target_count-1], top_score × 0.8)
```

**Selection:**
- Keep alternatives with score ≥ threshold
- If still too many, keep top target_count

### 9.2 High-Score Aggregation

**High-Score Identification:**
```
high_score_threshold = top_score × 0.9
```

**Aggregation:**
- All nodes with score ≥ high_score_threshold are aggregated
- Maximum 5 nodes aggregated together
- Requires ≥2 high-scoring nodes

---

## 10. Database Schema

### 10.1 Projects Table

**Schema:**
- project_id: UUID (primary key)
- project_name: VARCHAR(255)
- requirements: TEXT
- context: JSONB
- created_at: TIMESTAMP
- updated_at: TIMESTAMP

### 10.2 FBSL Nodes Table

**Schema:**
- node_id: UUID (primary key)
- project_id: UUID (foreign key → projects)
- parent_node_id: UUID (self-referential)
- node_type: ENUM (problem, design_prototype, evaluation, research, refinement)
- generation_level: INTEGER
- iteration_number: INTEGER
- functions: JSONB (serialized Function objects)
- behaviors: JSONB (serialized Behavior objects)
- structures: JSONB (serialized Structure objects)
- layout: JSONB (serialized Layout object)
- composite_score: FLOAT
- functional_score: FLOAT
- behavioral_score: FLOAT
- structural_score: FLOAT
- layout_score: FLOAT
- sustainability_score: FLOAT
- metadata: JSONB
- created_at: TIMESTAMP
- updated_at: TIMESTAMP

### 10.3 Evaluations Table

**Schema:**
- evaluation_id: UUID (primary key)
- node_id: UUID (foreign key → fbsl_nodes)
- project_id: UUID (foreign key → projects)
- functional_adequacy: JSONB (detailed scores)
- behavioral_performance: JSONB
- structural_feasibility: JSONB
- layout_efficiency: JSONB
- sustainability: JSONB
- composite_score: FLOAT
- rank: INTEGER
- strengths: TEXT[]
- weaknesses: TEXT[]
- recommendations: TEXT[]
- evaluated_at: TIMESTAMP

### 10.4 Relationships

- One project → Many FBSL nodes
- One FBSL node → One evaluation
- FBSL nodes form tree structure via parent_node_id

---

## 11. Complete Workflow Pipeline

### 11.1 Phase 1: Encoding and Research

**Step 1: Requirement Encoding**
- Input: Natural language requirements
- Process: LLM extraction → Spatial program → FBSL node
- Output: Problem node with F, B, S, L components

**Step 2: Research**
- Input: Problem node
- Process: Embedding generation → Similarity search → Precedent retrieval
- Output: Enhanced node with research findings

### 11.2 Phase 2: Design Space Generation

**Step 3: GoT Graph Generation**
- Input: Problem node
- Process:
  1. Calculate complexity
  2. Set adaptive parameters
  3. Expand nodes using strategies
  4. Score-based stopping
- Output: Graph G = (V, E) with FBSL nodes

**Step 4: Path Selection**
- Input: GoT graph
- Process: Find best paths using Score(path) = Σ w(eᵢ) × q(nᵢ)
- Output: Top-k leaf nodes

**Step 5: Scoring and Pruning**
- Input: Leaf nodes
- Process:
  1. Score all alternatives
  2. Prune low-scoring (score < threshold)
  3. Identify high-scoring (score ≥ 0.9 × top)
- Output: Pruned set of alternatives

**Step 6: Aggregation**
- Input: High-scoring alternatives
- Process: Aggregate using compatibility × quality formula
- Output: Aggregated prototype

### 11.3 Phase 3: Convergence Loop

**For each alternative:**
1. **F → Bₑ**: Expected behaviors already defined
2. **Bₑ → S**: Refinement generates/optimizes structures
3. **S → Bₛ**: Calculate actual behaviors from structures
4. **S → L**: Generate spatial layout
5. **Evaluation**: Score using MCDA
6. **Convergence Check**: |S_composite(t) - S_composite(t-1)| < ε
7. **Reformulation**: If not converged, apply Type 1/2/3

### 11.4 Phase 4: Final Scoring and Storage

**Step 7: Comprehensive Scoring**
- All alternatives scored using full MCDA
- Pareto front constructed
- Rankings assigned

**Step 8: Database Storage**
- Projects stored
- FBSL nodes stored
- Evaluations stored

---

## 12. Visualization Points for Research Paper

### 12.1 System Architecture Diagram

**Components to Show:**
1. Input layer: Natural language requirements
2. Encoding layer: Encoder Agent → FBSL node
3. Research layer: Research Agent → Knowledge retrieval
4. Generation layer: GoT Engine → Design alternatives
5. Refinement layer: Refinement Agent → Type 1/2/3 reformulations
6. Scoring layer: Scoring Agent → MCDA evaluation
7. Layout layer: Layout Agent → Spatial generation
8. Output layer: Prototypes with scores

**Connections:**
- Data flow arrows showing F → Bₑ → S → Bₛ → L
- Feedback loops showing reformulation paths
- Knowledge flow from research to encoding

### 12.2 FBSL Ontology Diagram

**Four Quadrants:**
1. **Function**: Circles with function names, priorities, activities
2. **Behavior**: Rectangles with metrics, targets, actuals
3. **Structure**: Hexagons with materials, dimensions
4. **Layout**: Spatial arrangement with rooms, adjacencies

**Connections:**
- F → Bₑ: Derivation arrows
- Bₑ → S: Generation arrows
- S → Bₛ: Calculation arrows
- S → L: Spatial arrangement

### 12.3 Graph of Thought Visualization

**Node Representation:**
- Size: Proportional to composite score
- Color: Generation level (depth)
- Shape: Node type (problem, prototype, etc.)

**Edge Representation:**
- Thickness: Transformation weight
- Color: Transformation type
- Direction: Parent → child

**Layout:**
- Hierarchical (top-down) or force-directed
- Highlight best paths
- Show aggregation nodes

### 12.4 Scoring Breakdown Visualization

**Radar/Spider Chart:**
- Five axes: Functional, Behavioral, Structural, Layout, Sustainability
- Multiple designs overlaid
- Pareto front highlighted

**Bar Chart:**
- Composite scores ranked
- Component scores stacked
- Threshold lines shown

### 12.5 Spatial Layout Visualization

**Floor Plan:**
- Room polygons with dimensions
- Circulation paths (A* results)
- Adjacency relationships (dashed lines)
- Site boundary
- Grid overlay

**Adjacency Graph:**
- Nodes: Rooms
- Edges: Adjacencies
- Edge thickness: Weight w(i,j)
- Edge color: Type (required/preferred/avoid)
- Layout: Force-directed or hierarchical

### 12.6 Refinement Process Flow

**Timeline/Sequence Diagram:**
- Initial state: Bₛ ≠ Bₑ
- Type 1: Structure modification
- Recalculation: New Bₛ
- Type 2: Behavior relaxation (if needed)
- Type 3: Function redefinition (if needed)
- Convergence: Bₛ ≈ Bₑ

**Deviation Plot:**
- X-axis: Iteration
- Y-axis: |Bₛ - Bₑ|
- Multiple behaviors shown
- Convergence threshold line

### 12.7 Complexity Adaptation Visualization

**Parameter Scaling:**
- X-axis: Complexity score
- Y-axis: Parameter value
- Multiple lines: depth, breadth, max_nodes, target_prototypes
- Threshold markers: low/medium/high/very_high

**Prototype Count vs Complexity:**
- Scatter plot
- X-axis: Complexity
- Y-axis: Number of prototypes generated
- Trend line showing adaptation

### 12.8 Database Schema Diagram

**Entity-Relationship Diagram:**
- Three main entities: Projects, FBSL_Nodes, Evaluations
- Relationships: One-to-many, foreign keys
- Attributes: Key fields highlighted
- JSONB fields: Shown as nested structures

### 12.9 Embedding Space Visualization

**t-SNE/UMAP Projection:**
- 2D/3D projection of FBSL embeddings
- Color: Function category or score
- Clusters: Similar designs grouped
- Trajectory: Design evolution path

### 12.10 Pareto Front Visualization

**2D Scatter Plot:**
- X-axis: Objective 1 (e.g., functional_adequacy)
- Y-axis: Objective 2 (e.g., layout_efficiency)
- Points: Design alternatives
- Pareto front: Highlighted boundary
- Dominated solutions: Grayed out

**3D Scatter Plot:**
- Three objectives simultaneously
- Interactive rotation
- Pareto surface highlighted

---

## 13. Key Parameters and Hyperparameters

### 13.1 GoT Parameters

- **max_depth**: 1-3 (adaptive based on complexity)
- **breadth**: 2-5 (adaptive based on complexity)
- **max_nodes**: 20-100+ (adaptive based on complexity)
- **epsilon**: 1e-3 (convergence threshold)
- **delta**: 1e-3 (stopping threshold, can override epsilon)
- **patience**: 2 (stagnation count before stopping)

### 13.2 Scoring Parameters

- **Weights**: w_f = 0.3, w_b = 0.3, w_s = 0.2, w_l = 0.15, w_sust = 0.05
- **Rho**: 1.0 (linear, compensatory)
- **Tolerance defaults**: 0.1-0.2 (behavior-dependent)

### 13.3 Spatial Algorithm Parameters

- **Force-directed:**
  - k_attraction: 0.1
  - k_repulsion: 100.0
  - learning_rate: 0.01
  - max_iterations: 200
  - convergence_threshold: 0.01

- **A* Pathfinding:**
  - grid_resolution: 0.5 meters
  - heuristic: Manhattan distance
  - smoothing: Line-of-sight optimization

- **Adjacency weights:**
  - α (functional): 0.4
  - β (traffic): 0.35
  - γ (privacy): 0.25

### 13.4 Refinement Parameters

- **max_iterations**: 5
- **convergence_threshold**: 0.01
- **Type 1 threshold**: avg_deviation < 0.3
- **Type 2 threshold**: 0.3 ≤ avg_deviation < 0.6
- **Type 3 threshold**: avg_deviation ≥ 0.6
- **Tolerance multiplier**: 1.5 (Type 1), 1.2 (Type 2)
- **Priority reduction**: 0.8 (Type 3)

### 13.5 Pruning Parameters

- **Quality threshold**: Adaptive (0.3-0.5 based on complexity)
- **Diversity threshold**: Adaptive (0.1-0.3 based on complexity)
- **Score threshold**: max(top_N, top_score × 0.8)
- **High-score threshold**: top_score × 0.9

### 13.6 Complexity Parameters

- **Low threshold**: 0.3
- **Medium threshold**: 0.6
- **High threshold**: 0.8
- **Base parameters**: depth=2, breadth=3, max_nodes=50, target=5
- **Component scale**: 1.0 + (room_count + function_count) / 20.0

---

## 14. Theoretical Contributions

### 14.1 FBSL Extension

Extends Gero's FBS ontology with Layout dimension, enabling explicit spatial reasoning in computational design.

### 14.2 Score-Driven Adaptation

Novel approach where exploration, pruning, and aggregation adapt based on actual performance scores rather than static heuristics.

### 14.3 Component-Preserving Embeddings

Concatenation-based embeddings preserve component-specific information for fine-grained similarity search.

### 14.4 Optimized Reformulation

Type 1 reformulation uses optimization to find ΔS that minimizes |Bₛ - Bₑ|, not just heuristic structure addition.

### 14.5 Multi-Objective Pareto Tracking

Maintains Pareto front across five objectives, enabling trade-off analysis in architectural design.

---

## 15. Computational Complexity

### 15.1 Time Complexity

- **GoT Generation**: O(b^d × n) where b = breadth, d = depth, n = node processing time
- **Scoring**: O(m × k) where m = behaviors, k = scoring criteria
- **Layout Generation**: O(r² × i) where r = rooms, i = force-directed iterations
- **A* Pathfinding**: O(|V| + |E|) where V = grid cells, E = edges

### 15.2 Space Complexity

- **GoT Graph**: O(b^d) nodes
- **Embeddings**: O(d_f + d_b + d_s + d_l) per node
- **Layout**: O(r) rooms with O(r²) adjacency matrix

---

## 16. Limitations and Future Work

### 16.1 Current Limitations

- Type 3 reformulation only reduces priority, doesn't transform functions
- No support for ρ → ∞ (min operator) in composite scoring
- 2D layouts only (no true 3D position vectors)
- Structure-behavior models are simplified
- No BERT-based component serialization

### 16.2 Future Enhancements

- Full function transformation in Type 3
- Advanced structure-behavior physics models
- Multi-story layout support
- Real-time embedding generation
- Component-weighted relevance scoring

---

## References

- Gero, J. S. (1990). Design prototypes: a knowledge representation schema for design. *AI Magazine*, 11(4), 26-36.
- Graph of Thought framework for design space exploration
- Multi-criteria decision analysis in architectural design
- Force-directed layout algorithms
- A* pathfinding for circulation design
- Pareto optimality in multi-objective optimization

