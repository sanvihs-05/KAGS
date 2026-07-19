# Theoretical Framework Implementation Gap Analysis

## Overview
This document identifies what from the theoretical FBSL-KAGS framework is **NOT fully implemented** or **incompletely implemented**.

---

## 1. FBSL Embedding Generation ❌ **INCOMPLETE**

### Theoretical Formula:
```
e_fbsl = [e_f || e_b || e_s || e_l]
```
Where `||` denotes **concatenation** of component embeddings.

### Current Implementation:
**Location:** `backend/core/fbsl_models.py:508-530`

```python
def calculate_composite_embedding(self):
    # Uses WEIGHTED AVERAGE, not concatenation
    self.composite_embedding = np.average(
        np.array(embeddings), 
        axis=0, 
        weights=weights
    )
```

### Gap:
- ❌ Uses **weighted average** instead of **concatenation**
- ❌ Loses component-specific information by averaging
- ❌ Cannot distinguish between different component contributions

### Impact:
- Research agent cannot use component-specific embeddings for retrieval
- Cannot weight different components (F, B, S, L) differently in similarity search

---

## 2. Research Agent Embedding Generation ❌ **INCOMPLETE**

### Theoretical Formula:
```
e_component = BERT(serialize(component))
```

### Current Implementation:
**Location:** `backend/agents/research_agent.py`

- Uses Finnish embeddings from vector store
- Does NOT use BERT to serialize FBSL components
- Function embeddings come from Finnish floor plan data, not from serializing the Function object

### Gap:
- ❌ No BERT-based serialization of FBSL components
- ❌ Cannot generate embeddings from Function/Behavior/Structure/Layout objects directly
- ❌ Relies on external Finnish embeddings rather than component serialization

### Impact:
- Cannot generate embeddings for new FBSL components on-the-fly
- Limited to pre-computed Finnish embeddings

---

## 3. Relevance Scoring with Component Weighting ❌ **MISSING**

### Theoretical Formula:
```
Relevance(q, d) = Σᵢ wᵢ × sim(qᵢ, dᵢ)
```
Where each component (F, B, S, L) has its own similarity and weight.

### Current Implementation:
**Location:** `backend/agents/research_agent.py:21-90`

- Uses simple similarity search
- Does NOT weight components separately
- Does NOT compute component-wise similarities

### Gap:
- ❌ No component-weighted relevance scoring
- ❌ Cannot prioritize Function similarity over Structure similarity, etc.

### Impact:
- Cannot fine-tune retrieval based on which FBSL component is most important

---

## 4. Type 1 Reformulation - Optimization ❌ **INCOMPLETE**

### Theoretical Formula:
```
S' = S + ΔS where ΔS minimizes |Bₛ - Bₑ|
```

### Current Implementation:
**Location:** `backend/agents/refinement_agent.py:180-202`

```python
def _type1_reformulation(self, node, unsatisfied):
    # Adds structures but doesn't optimize ΔS
    if category == 'thermal':
        self._add_thermal_structure(node)
    # ...
```

### Gap:
- ❌ Adds structures **heuristically**, not through **optimization**
- ❌ Does NOT calculate optimal ΔS to minimize |Bₛ - Bₑ|
- ❌ No gradient-based or search-based optimization

### Impact:
- May add unnecessary structures
- May not find optimal structure modifications

---

## 5. Type 3 Reformulation - Function Transformation ❌ **INCOMPLETE**

### Theoretical Formula:
```
F' = Transform(F) such that feasible(B'ₑ) = true
```

### Current Implementation:
**Location:** `backend/agents/refinement_agent.py:223-239`

```python
def _type3_reformulation(self, node, unsatisfied):
    # Only reduces priority, doesn't transform function
    func.priority *= 0.8
```

### Gap:
- ❌ Only **reduces priority**, doesn't **transform/redefine** the function
- ❌ Does NOT create F' = Transform(F)
- ❌ Does NOT ensure feasible(B'ₑ) = true

### Impact:
- Cannot handle cases where functions need to be fundamentally redefined
- May get stuck when functions are infeasible

---

## 6. Type 2 Reformulation - Formula Mismatch ⚠️ **PARTIAL**

### Theoretical Formula:
```
B'ₑ = Bₑ × (1 + tolerance)
```

### Current Implementation:
**Location:** `backend/agents/refinement_agent.py:204-221`

```python
def _type2_reformulation(self, node, unsatisfied):
    # Modifies tolerance, not target value
    behav.tolerance *= 1.5
```

### Gap:
- ⚠️ Modifies **tolerance** instead of **target value Bₑ**
- ⚠️ Should modify `target_value = target_value × (1 + tolerance)`
- ⚠️ Current approach changes acceptance range, not the requirement itself

### Impact:
- May not fully relax requirements as theory suggests
- Tolerance increase is less effective than target value relaxation

---

## 7. Layout Position Vectors - 3D ❌ **NOT FULLY UTILIZED**

### Theoretical Formula:
```
P = Position vectors: {(x, y, z) | x, y, z ∈ ℝ}
```

### Current Implementation:
**Location:** `backend/core/spatial_algorithms.py`, `backend/agents/layout_agent.py`

- Uses 2D positions (x, y) only
- Height is stored separately, not as z-coordinate
- No 3D spatial reasoning

### Gap:
- ❌ No true 3D position vectors
- ❌ Cannot handle multi-story layouts
- ❌ Height treated as attribute, not spatial dimension

### Impact:
- Limited to single-story layouts
- Cannot optimize vertical relationships

---

## 8. Adjacency Matrix Range ❌ **NOT ENFORCED**

### Theoretical Formula:
```
A[i,j] ∈ [0,1] representing spatial relationships
```

### Current Implementation:
**Location:** `backend/core/spatial_algorithms.py:394-469`

```python
# Normalize to [-1, 1] range, not [0, 1]
adjacency_matrix = adjacency_matrix / max_val
```

### Gap:
- ❌ Uses **[-1, 1]** range (negative for separation)
- ❌ Theory specifies **[0, 1]** (0 = no adjacency, 1 = strong adjacency)
- ❌ Negative values represent separation, which is not in theory

### Impact:
- Inconsistent with theoretical framework
- May cause confusion in interpretation

---

## 9. Circulation Path Optimality ❌ **NOT VERIFIED**

### Theoretical Formula:
```
Circulation = Actual_Path_Length / Optimal_Path_Length
```

### Current Implementation:
**Location:** `backend/core/spatial_algorithms.py:652-679`

- Calculates actual path length
- Does NOT verify optimality (A* should be optimal, but not explicitly checked)
- Does NOT compare against theoretical optimal

### Gap:
- ❌ No verification that A* paths are truly optimal
- ❌ No comparison against theoretical minimum path length
- ❌ May not detect suboptimal circulation

### Impact:
- Cannot guarantee optimal circulation paths
- May accept inefficient layouts

---

## 10. Composite Score Rho Parameter - Min Operator ❌ **NOT IMPLEMENTED**

### Theoretical Formula:
```
S_composite = Σᵢ (wᵢ × Sᵢ^ρ)^(1/ρ)
Where ρ→∞ for min operator
```

### Current Implementation:
**Location:** `backend/agents/scoring_agent.py`

- Supports ρ = 1 (linear) and ρ = -1 (geometric mean)
- Does NOT support ρ → ∞ (min operator)
- No implementation of limit case

### Gap:
- ❌ Cannot use min operator (non-compensatory) scoring
- ❌ Limited to compensatory scoring methods

### Impact:
- Cannot enforce strict "all criteria must be met" scoring
- May accept designs with one very poor criterion

---

## Summary of Missing/Incomplete Features

| Feature | Status | Priority | Impact |
|---------|--------|----------|--------|
| FBSL Embedding Concatenation | ❌ Incomplete | High | Research retrieval accuracy |
| BERT Component Serialization | ❌ Missing | Medium | On-the-fly embedding generation |
| Component-Weighted Relevance | ❌ Missing | Medium | Fine-tuned retrieval |
| Type 1 Optimization (ΔS) | ❌ Incomplete | High | Optimal structure modification |
| Type 3 Function Transformation | ❌ Incomplete | High | Handling infeasible functions |
| Type 2 Target Value Relaxation | ⚠️ Partial | Medium | Proper requirement relaxation |
| 3D Position Vectors | ❌ Missing | Low | Multi-story layouts |
| Adjacency Matrix [0,1] Range | ❌ Mismatch | Low | Theoretical consistency |
| Circulation Optimality Verification | ❌ Missing | Medium | Path quality guarantee |
| Min Operator (ρ→∞) | ❌ Missing | Medium | Non-compensatory scoring |

---

## Recommendations

### High Priority:
1. **Implement FBSL embedding concatenation** instead of weighted average
2. **Add optimization to Type 1 reformulation** (minimize |Bₛ - Bₑ|)
3. **Implement proper Type 3 function transformation** (F' = Transform(F))

### Medium Priority:
4. **Add BERT-based component serialization** for embeddings
5. **Implement component-weighted relevance scoring**
6. **Fix Type 2 to modify target values** (B'ₑ = Bₑ × (1 + tolerance))
7. **Add circulation optimality verification**

### Low Priority:
8. **Support 3D position vectors** for multi-story
9. **Normalize adjacency matrix to [0,1]** range
10. **Implement min operator (ρ→∞)** for composite scoring

