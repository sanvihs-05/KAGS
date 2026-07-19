# Visualization Specifications for FBSL-KAGS Research Paper

## 1. System Architecture Diagram

### Layout
- **Type**: Hierarchical block diagram (top-down)
- **Layers**: 8 horizontal layers with data flow arrows

### Components (Top to Bottom)

**Layer 1: Input**
- Box: "Natural Language Requirements"
- Example text: "2 bedroom apartment with kitchen..."

**Layer 2: Encoding**
- Box: "Encoder Agent"
- Sub-components:
  - LLM Extraction Module
  - Spatial Program Parser
  - FBSL Node Constructor
- Output arrow: "Problem Node (F, B, S, L)"

**Layer 3: Research**
- Box: "Research Agent"
- Sub-components:
  - Embedding Generator (e_fbsl = [e_f || e_b || e_s || e_l])
  - Vector Store (Finnish Floor Plans)
  - Similarity Search (Cosine)
- Output arrow: "Enhanced Node + Precedents"

**Layer 4: Generation**
- Box: "Graph of Thought Engine"
- Sub-components:
  - Complexity Calculator
  - Node Expander (4 strategies)
  - Path Finder (Score = Σ w(e) × q(n))
- Output arrow: "Design Alternatives"

**Layer 5: Refinement**
- Box: "Refinement Agent"
- Sub-components:
  - Type 1: S' = S + ΔS (optimize)
  - Type 2: B'ₑ = Bₑ × (1 + tolerance)
  - Type 3: F' = Transform(F)
- Feedback arrow: Back to Generation

**Layer 6: Scoring**
- Box: "Scoring Agent"
- Sub-components:
  - MCDA Calculator
  - Composite Score: [Σ(wᵢ × Sᵢ^ρ)]^(1/ρ)
  - Pareto Front Builder
- Output arrow: "Ranked Prototypes"

**Layer 7: Layout**
- Box: "Layout Agent"
- Sub-components:
  - Force-Directed Optimizer
  - A* Pathfinder
  - Adjacency Calculator
- Output arrow: "Spatial Layouts"

**Layer 8: Output**
- Box: "Design Prototypes"
- Sub-components:
  - Floor Plans (SVG)
  - Adjacency Graphs
  - Evaluation Reports

### Annotations
- Formula callouts next to relevant components
- Parameter values shown in small text
- Data flow arrows labeled with data types

---

## 2. FBSL Ontology Diagram

### Layout
- **Type**: Four-quadrant diagram with central connections

### Quadrants

**Top-Left: Function (F)**
- Shape: Circles
- Content: Function names, priorities (0-1), activities list
- Example: "provide_thermal_comfort" (priority: 0.9, activities: [comfort, rest])
- Color: Blue gradient (darker = higher priority)

**Top-Right: Behavior (B)**
- Shape: Rectangles
- Content: Metric name, target value, actual value, tolerance
- Example: "temperature: 22°C (target) / 18°C (actual) / ±0.1"
- Color: Green (satisfied) / Red (unsatisfied)

**Bottom-Left: Structure (S)**
- Shape: Hexagons
- Content: Structure name, material, dimensions
- Example: "thermal_insulation: insulation / 0.15m thickness"
- Color: Gray gradient

**Bottom-Right: Layout (L)**
- Shape: Spatial arrangement
- Content: Room polygons, positions, adjacencies
- Example: Room rectangles with labels, adjacency lines
- Color: Room type-based

### Connections
- **F → Bₑ**: Dashed arrows labeled "derives"
- **Bₑ → S**: Solid arrows labeled "informs"
- **S → Bₛ**: Dotted arrows labeled "exhibits"
- **S → L**: Thick arrows labeled "arranged as"

### Central Node
- Large circle: "FBSL Node"
- Shows transformation: F → Bₑ → S → Bₛ → L
- Reformulation loop shown as curved arrow

---

## 3. Graph of Thought Visualization

### Layout
- **Type**: Force-directed graph with hierarchical coloring

### Node Properties
- **Size**: Proportional to composite_score (min: 20px, max: 100px)
- **Color**: Generation level (depth)
  - Level 0: Dark blue
  - Level 1: Medium blue
  - Level 2: Light blue
  - Level 3: Cyan
- **Shape**: 
  - Circle: Problem node
  - Square: Design prototype
  - Diamond: Aggregated node
- **Border**: Thickness = 2px × quality_score

### Edge Properties
- **Thickness**: Proportional to transformation weight w(e)
- **Color**: Transformation type
  - Red: Functional decomposition
  - Green: Behavioral optimization
  - Blue: Structural variation
  - Purple: Layout permutation
  - Orange: Aggregation
- **Style**: 
  - Solid: High quality (q > 0.7)
  - Dashed: Medium quality (0.5 < q ≤ 0.7)
  - Dotted: Low quality (q ≤ 0.5)

### Highlighting
- **Best paths**: Thick yellow overlay
- **Aggregated nodes**: Pulsing animation or glow
- **High-scoring nodes**: Star icon overlay

### Layout Algorithm
- Initial: Hierarchical (root at top)
- Refinement: Force-directed for clarity
- Labels: Node IDs (first 8 chars) + composite score

---

## 4. Scoring Breakdown Visualization

### Radar Chart (Primary)

**Axes (5):**
1. Functional Adequacy (0-1)
2. Behavioral Performance (0-1)
3. Structural Feasibility (0-1)
4. Layout Efficiency (0-1)
5. Sustainability (0-1)

**Designs:**
- Multiple polygons overlaid
- Color: Unique per design
- Opacity: 0.6 for overlap visibility
- Legend: Design IDs + composite scores

**Reference Lines:**
- Threshold: 0.7 (dashed circle)
- Optimal: 1.0 (outer circle)

### Bar Chart (Secondary)

**X-axis**: Design prototypes (ranked by composite score)
**Y-axis**: Score (0-1)
**Stacked bars**: 
- Bottom: Functional (blue)
- Second: Behavioral (green)
- Third: Structural (gray)
- Fourth: Layout (purple)
- Top: Sustainability (orange)
**Total height**: Composite score
**Threshold line**: Horizontal at 0.7

---

## 5. Spatial Layout Visualization

### Floor Plan (Primary View)

**Coordinate System:**
- Origin: Bottom-left
- Units: Meters
- Scale: 1:100 or 1:200

**Room Representation:**
- **Polygons**: Filled rectangles with rounded corners
- **Color**: Room type (bedroom=blue, kitchen=yellow, etc.)
- **Labels**: Room name + area (e.g., "Bedroom 1\n13.5 m²")
- **Dimensions**: Width × Height shown inside or as callouts

**Circulation:**
- **Paths**: Colored lines (A* results)
  - Direct: Green
  - Corridor: Blue
  - Open: Yellow
- **Width**: Proportional to path importance
- **Arrows**: Direction of flow
- **Length**: Annotated in meters

**Adjacencies:**
- **Lines**: Dashed between adjacent rooms
- **Color**: 
  - Required: Green (thick)
  - Preferred: Yellow (medium)
  - Avoid: Red (thin, if violated)
- **Weight**: Line thickness = w(i,j)

**Site Boundary:**
- **Outline**: Thick black line
- **Setbacks**: Shaded areas
- **Dimensions**: Annotated

**Grid Overlay:**
- Optional: 1m × 1m grid
- Light gray, 50% opacity

### Adjacency Graph (Secondary View)

**Layout**: Force-directed or circular
**Nodes**: 
- Shape: Circles
- Size: Proportional to room area
- Label: Room name
- Color: Room type
**Edges**:
- Thickness: w(i,j) × 10 (normalized)
- Color: 
  - Positive (w > 0): Blue
  - Negative (w < 0): Red
- Style:
  - Required: Solid
  - Preferred: Dashed
  - Avoid: Dotted

---

## 6. Refinement Process Flow

### Sequence Diagram

**Timeline (X-axis)**: Iterations 1 → 5
**Layers (Y-axis)**: 
1. Initial State
2. Type 1 Reformulation
3. Recalculation
4. Type 2 Reformulation (if needed)
5. Type 3 Reformulation (if needed)
6. Converged State

**Flow:**
- Vertical arrows: State transitions
- Horizontal arrows: Iteration progression
- Annotations: Deviation values |Bₛ - Bₑ|

### Deviation Plot

**X-axis**: Iteration number (1-5)
**Y-axis**: |Bₛ - Bₑ| (absolute deviation)
**Lines**: One per behavior
**Colors**: Behavior category
**Reference**: 
- Convergence threshold: Horizontal line at ε = 0.01
- Target: Horizontal line at 0.0
**Markers**: 
- Type 1 applied: Triangle
- Type 2 applied: Square
- Type 3 applied: Diamond
- Converged: Star

---

## 7. Complexity Adaptation Visualization

### Parameter Scaling Plot

**X-axis**: Complexity Score (0-1)
**Y-axis**: Parameter Value (normalized 0-1)
**Lines** (4):
1. Depth (blue): 1 → 3
2. Breadth (green): 2 → 5
3. Max Nodes (red): 20 → 100
4. Target Prototypes (purple): 3 → 8

**Threshold Markers**:
- Vertical lines at 0.3, 0.6, 0.8
- Labels: "Low", "Medium", "High", "Very High"
- Shaded regions: Different complexity zones

### Prototype Count vs Complexity

**Type**: Scatter plot with trend line
**X-axis**: Complexity Score
**Y-axis**: Number of Prototypes Generated
**Points**: 
- Size: Proportional to final composite score
- Color: Complexity level
**Trend**: 
- Linear regression line
- Confidence interval (shaded)
- R² value shown

---

## 8. Database Schema Diagram

### Entity-Relationship Diagram

**Entities** (3 main):

1. **Projects** (Rectangle)
   - Primary Key: project_id (underlined)
   - Attributes: project_name, requirements, context, timestamps
   - Color: Light blue

2. **FBSL_Nodes** (Rectangle)
   - Primary Key: node_id (underlined)
   - Foreign Keys: project_id, parent_node_id (dashed underline)
   - Attributes: node_type, generation_level, scores, JSONB fields
   - Color: Light green
   - Self-relationship: parent_node_id → node_id (tree structure)

3. **Evaluations** (Rectangle)
   - Primary Key: evaluation_id (underlined)
   - Foreign Keys: node_id, project_id (dashed underline)
   - Attributes: scores (JSONB), rank, strengths, weaknesses
   - Color: Light yellow

**Relationships**:
- Projects → FBSL_Nodes: One-to-Many (1:N)
- FBSL_Nodes → Evaluations: One-to-One (1:1)
- FBSL_Nodes → FBSL_Nodes: One-to-Many (1:N, self-referential)

**Cardinality**: Shown with crow's foot notation
**JSONB Fields**: Shown as nested rectangles or noted as "JSONB"

---

## 9. Embedding Space Visualization

### t-SNE/UMAP Projection

**Type**: 2D scatter plot
**Points**: 
- Each point = one FBSL node
- Size: Composite score
- Color: 
  - Option 1: Function category
  - Option 2: Composite score (heatmap)
  - Option 3: Generation level

**Clusters**:
- Highlighted with convex hulls
- Labeled: "High-quality designs", "Spatial focus", etc.

**Trajectory**:
- Arrows showing design evolution
- From problem node → prototypes
- Thickness: Path quality

### Component Embedding Comparison

**Type**: Parallel coordinates or small multiples
**Show**: 
- Function embeddings (top row)
- Behavior embeddings (second row)
- Structure embeddings (third row)
- Layout embeddings (bottom row)
- Composite embeddings (overlay)

**Color**: Same design across all views

---

## 10. Pareto Front Visualization

### 2D Scatter Plot

**Axes**: 
- X: Functional Adequacy (0-1)
- Y: Layout Efficiency (0-1)

**Points**:
- All designs: Gray circles
- Pareto-optimal: Colored circles (size = composite score)
- Dominated: Small gray dots

**Pareto Front**:
- Line connecting Pareto-optimal points
- Shaded region: Feasible space
- Arrow: Direction of improvement

### 3D Scatter Plot

**Axes**:
- X: Functional Adequacy
- Y: Layout Efficiency  
- Z: Behavioral Performance

**Interactive Elements**:
- Rotation controls
- Zoom
- Point selection (shows design details)

**Pareto Surface**:
- Highlighted boundary
- Transparency: 50%

---

## 11. Formula Reference Diagram

### Layout
- **Type**: Sidebar or appendix page
- **Format**: Numbered equations with brief descriptions

### Key Formulas to Highlight

1. **FBSL Transformation**: F → Bₑ → S → Bₛ → L
2. **GoT Path Score**: Score(path) = Σᵢ w(eᵢ) × q(nᵢ)
3. **Aggregation**: Aggregate = argmax[Σᵢ λᵢ × Compatibility × Quality]
4. **Composite Score**: S_composite = [Σ(wᵢ × Sᵢ^ρ)]^(1/ρ)
5. **Force**: Force = k_a × A[i,j] × d - k_r / d²
6. **Adjacency**: w(i,j) = α×Func + β×Traffic + γ×Privacy
7. **A* Cost**: f(n) = g(n) + h(n)
8. **Type 1**: S' = S + ΔS (minimize |Bₛ - Bₑ|)
9. **Type 2**: B'ₑ = Bₑ × (1 + tolerance)
10. **Convergence**: |S(t) - S(t-1)| < ε

---

## 12. Workflow Sequence Diagram

### Horizontal Timeline

**Phases** (left to right):

1. **Encoding** (Column 1)
   - Input: Requirements text
   - Process: LLM → Spatial Program → FBSL Node
   - Output: Problem node

2. **Research** (Column 2)
   - Input: Problem node
   - Process: Embedding → Search → Retrieval
   - Output: Enhanced node

3. **Generation** (Column 3)
   - Input: Enhanced node
   - Process: GoT expansion → Path finding
   - Output: Alternatives

4. **Scoring** (Column 4)
   - Input: Alternatives
   - Process: MCDA → Pruning → Aggregation
   - Output: Ranked prototypes

5. **Refinement** (Column 5)
   - Input: Each prototype
   - Process: Type 1/2/3 → Convergence
   - Output: Refined prototypes

6. **Layout** (Column 6)
   - Input: Refined prototypes
   - Process: Force-directed → A* → Visualization
   - Output: Spatial layouts

7. **Storage** (Column 7)
   - Input: Final prototypes
   - Process: Database storage
   - Output: Stored designs

### Vertical Flow
- Arrows between phases
- Parallel processes shown (e.g., multiple alternatives refined simultaneously)
- Feedback loops: Refinement → Generation (if needed)

---

## 13. Parameter Sensitivity Analysis

### Heatmap

**X-axis**: Parameter values (e.g., rho: -1 to 2)
**Y-axis**: Different test cases
**Color**: Final composite score (heatmap)
**Annotations**: Optimal parameter values marked

### Line Plots

**Multiple subplots** (one per parameter):
- X-axis: Parameter value
- Y-axis: Composite score
- Lines: Different test cases
- Optimal: Marked with vertical line

---

## 14. Comparison Visualization

### Before/After Comparison

**Side-by-side**:
- Left: Without critical fixes (weighted average embeddings, heuristic reformulation)
- Right: With critical fixes (concatenated embeddings, optimized reformulation)

**Metrics Shown**:
- Embedding dimension
- Retrieval accuracy
- Refinement iterations
- Final scores

---

## 15. Technical Specifications for Figures

### Figure 1: System Architecture
- **Size**: Full page width
- **Format**: Vector (SVG/PDF)
- **Resolution**: 300 DPI minimum
- **Color**: CMYK for print

### Figure 2: FBSL Ontology
- **Size**: Half page
- **Format**: Vector
- **Layout**: Four quadrants with central node

### Figure 3: GoT Graph
- **Size**: Full page
- **Format**: Vector or high-res raster
- **Interactive**: Optional web version

### Figure 4: Scoring Breakdown
- **Size**: Half page (radar) + half page (bar chart)
- **Format**: Vector

### Figure 5: Spatial Layouts
- **Size**: Full page (floor plan) + half page (adjacency graph)
- **Format**: Vector
- **Scale**: Clearly marked

### Figure 6: Refinement Process
- **Size**: Full page
- **Format**: Vector
- **Timeline**: Clear iteration markers

### Figure 7: Complexity Adaptation
- **Size**: Half page (parameter scaling) + half page (prototype count)
- **Format**: Vector

### Figure 8: Database Schema
- **Size**: Half page
- **Format**: Vector
- **Style**: Standard ERD notation

### Figure 9: Embedding Space
- **Size**: Full page
- **Format**: High-res raster or vector
- **Interactive**: Optional 3D version

### Figure 10: Pareto Front
- **Size**: Half page (2D) + half page (3D)
- **Format**: Vector
- **3D**: Interactive optional

---

## 16. Caption Guidelines

Each figure should include:
1. **Figure number and title**
2. **Brief description** (1-2 sentences)
3. **Key formulas** referenced
4. **Parameter values** used
5. **Interpretation** (what it shows)

Example:
"Figure 3: Graph of Thought exploration for a 3-bedroom apartment design. Node size represents composite score, color indicates generation level. The highlighted path (yellow) shows the optimal design trajectory. Parameters: depth=2, breadth=4, max_nodes=75. The graph demonstrates how high-scoring alternatives (score > 0.7) trigger early stopping, resulting in 10 nodes instead of the maximum 75."

