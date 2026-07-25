# FBSL-KAGS: Multi-Agent System Architecture (Implementation Reference)

**Status:** Reflects the *current, verified* codebase (`backend/`), not the original design spec.
Every formula, threshold, and flow step below was read directly from source before writing.
Where the running system now differs from the earlier architecture document, a
**⚠ Changed from original spec** callout states the old value and why it changed.

FBSL extends Gero's **Function–Behavior–Structure (FBS)** ontology with an explicit
fourth layer — **Layout (L)** — carrying room coordinates, dimensions, and the
adjacency graph. The pipeline transforms unstructured natural-language requirements
into ranked, fully-specified floor-plan prototypes.

---

## 0. Reading Guide — What Changed Since the Original Spec

| Area | Original spec | Current implementation | Why |
|---|---|---|---|
| **Encoder LLM** | Ollama (gemma3/llama3.1) only | **Cloud-first chain**: Groq (or any OpenAI-compatible API) → Ollama → rule-based parser | Local 4 GB GPU couldn't fit models; 30–40 s reloads blew the 60 s timeout |
| **Generalizer** | 4 variants (zoning/topology/priority/structural) | **5 named Level-1 strategies** with real parameter deltas, expanded inside GoT | Label-only variants scored identically; now each changes physics/geometry |
| **Layout algorithm** | Force-directed particle simulation + A* on 0.5 m grid | **Zoned squarified treemap** (gap-free tiling) + **room-connectivity-graph** circulation | Force-directed leaves gaps by construction; free-space A* fails on a tiled plan |
| **Compactness** | `Total_Room_Area / Bounding_Box_Area` | **`min(W,H) / max(W,H)`** of the footprint | Old formula is ≈1.0 for any gap-free plan; can't tell a square from a corridor |
| **Circulation** | A* path length ratio on a grid | Shortest path on the **door-connectivity graph** (rooms sharing a ≥0.7 m wall) | Endpoints sit inside obstacles on a tiled plan; A* found no path → always 0.0 |
| **S_l weights** | `0.4·Compact + 0.3·Circ + 0.3·Adj` (3 terms) | `0.30·SpaceUtil + 0.25·Circ + 0.30·Adj + 0.15·Compact` (4 terms) | Space utilisation added as an explicit term |
| **Composite weights** | S_f .30 / S_b .30 / S_s .20 / S_l .15 / S_sust .05 | **S_f .25 / S_b .20 / S_s .20 / S_l .25 / S_sust .10** (normalised), `ρ = 1.0` | Layout weighted up now that L is real geometry |
| **Aggregation trigger** | score ≥ max×0.9 AND compatibility > 0.7 | within **0.75×top** AND **≥ 2 distinct design signatures** | Merging clones just reproduces the same design |
| **Adjacency satisfaction** | (implicit) | **Measured** on shared walls; a real 0.0 is kept, no 0.6 default | Old code hard-coded 0.6 "satisfaction" |
| **FBSL persistence** | fields existed, unused | **L is generated and stored**: coordinates, dims, adjacency matrices in `fbsl_data.json` per prototype | The L layer was declared but never populated |

---

## 1. Agents

### 1.1 Encoder Agent

**Role:** Entry point. Transforms raw natural-language requirements into an initial FBSL
problem node (Functions, expected Behaviors, placeholder Structures, initial room list).

**Input:** e.g. *"Design a 3-bedroom compact urban home, master bedroom ~16 sqm with
ensuite bathroom, open-plan kitchen and living room, mudroom connecting to the garage…"*

**Output — initial FBSL problem node:**
- **Functions (F):** one per room, with `priority ∈ [0.5, 0.95]`, activities, and
  `spatial_requirements {min_area, max_area, preferred_area, orientation}`
- **Expected Behaviors (Bₑ):** area behavior per room, plus lighting/ventilation/acoustic
  behaviors derived from qualitative cues ("quiet" → acoustic, "natural light" → lighting)
- **Initial Structures (S):** per-room partition + window (glazing ratio by room type),
  plus node-level HVAC (MEP) and reinforced-concrete foundation
- **Initial Layout (L):** room list with areas, no positions yet
- **Metadata:** `required_adjacencies` (resolved to canonical room *types*)

#### LLM Provider Chain ⚠ *Changed from original spec*

Three modes via `KAGS_LLM_PROVIDER` (default `auto`):

```
auto:   cloud (if key present) → Ollama → rule-based parser
openai: cloud only  (no fallback)
ollama: local only  (no fallback)
```

- **Cloud:** any OpenAI-compatible `/chat/completions` endpoint. `GROQ_API_KEY` alone
  auto-targets Groq (`llama-3.3-70b-versatile`, ~0.8 s/request). Configurable via
  `KAGS_LLM_API_KEY`, `KAGS_LLM_BASE_URL`, `KAGS_LLM_MODEL`, `KAGS_LLM_CLOUD_TIMEOUT` (20 s).
- **Ollama:** local `llama3.2` (default; smallest installed model for 4 GB GPUs).
- **Rule-based parser:** deterministic regex/lexicon extractor; the guaranteed floor.

Any cloud failure (timeout, rate-limit, malformed JSON) transparently falls through to the
next provider. `_call_cloud_llm` / `_call_ollama_llm` each raise on failure; a single
attempt-chain loop orchestrates the fallback.

#### Extraction post-processing (guards for real-LLM output)

- **Area sanitising:** a missing *or* non-positive `area_min/area_max` is treated as
  "unspecified" and replaced with a per-type default band. *(A literal `area_min: 0` from
  the model would otherwise set an area behavior target of 0, which collapses the entire
  geometric-mean S_b to 0.)*
- **Adjacency label resolution:** each adjacency's `room1/room2` (which the LLM often
  returns as a descriptive **name**, "Master Bedroom") is mapped to a canonical room
  **type** present in the design ("bedroom") — exact-type, exact-name, then substring
  match. Unresolvable pairs are dropped, never stored as guaranteed-unsatisfiable.
- **Rule-based fallback** additionally extracts adjacencies from connective phrases
  ("connected to", "attached", "open-plan X and Y", "separated from" → avoid).

---

### 1.2 Generalizer Agent → **5 Named Strategies (Level-1 GoT)** ⚠ *Changed from original spec*

**Role:** Seed the design space with genuinely different starting points. In the current
system this happens as **Level-1 expansion inside the Graph of Thoughts** (the standalone
`GeneralizerAgent` still exists for the non-GoT path, but the GoT path — the default —
uses the five named strategy seeds below). Each strategy changes parameters the physics
and geometry actually read, so each earns a different score.

| Strategy | `layout_aspect` | Real parameter delta |
|---|---|---|
| `functional_priority` | 1.5 | High-priority rooms scaled ×1.08, low-priority ×0.92 (area-neutral) |
| `performance_optimized` | 1.1 | Glazing ×0.90 (thermal), partitions → 0.15 m (acoustic), keeps HVAC |
| `structural_efficiency` | 1.35 | Rooms ×0.95, partitions → 0.08 m, non-load-bearing concrete → steel |
| `spatial_compactness` | 1.05 | Near-square footprint, rooms ×0.97 |
| `balanced` | 1.2 | Baseline parameters |

All area scalings stay inside the brief validator's ±10 % total-area grace so a strategy is
never auto-rejected for its emphasis.

**Level-2+ micro-transformations** specialise each strategy (names compose, e.g.
`spatial_compactness+linear_layout`):
- **Layout permutation** → `compact` (1.05), `linear` (2.4), `courtyard` (0.6) aspects
  (the parent's own aspect is skipped to avoid cloning it)
- **Behavioral** → `relaxed_tolerances`; `natural_ventilation` (removes MEP, enlarges
  glazing ×1.25 — a real S_b trade-off)
- **Functional** → `priority_focus` (keep priority > 0.7 functions)
- **Structural** → `alt_materials`

---

### 1.3 Research Agent (External RAG)

**Role:** Retrieve precedent Finnish floor plans from the FAISS vector store to ground
room sizing.

**Query process:** encode FBSL context → L2-normalise → `IndexFlatIP` inner-product search
(≡ cosine after normalisation) → map indices to metadata.

```
similarity(q, pᵢ) = cos(E(q), E(pᵢ)) = (q · pᵢ) / (‖q‖ · ‖pᵢ‖)
```

**Retrieval thresholds by function priority:** High > 0.7, Medium > 0.5, Low > 0.3.

**Precedent reconciliation:** similarity-weighted blend of stated vs precedent area,
`a* = λ·a_stated + (1−λ)·â_precedent` (λ = 0.6), clamped to each function's [min, max] band.

**Store:** `cubicasa_rag_store` — a **room-level** index built from CubiCasa5K by
`embeddings_generator/build_cubicasa_rag.py`. One record per room (34,319 rooms from 3,787
real plans), each with `{plan_id, room_type, area_m², neighbors}`; the area comes from the
room polygon (shoelace at CubiCasa's 100-units/metre scale). This replaced the legacy
`enhanced_multimodal_rag_store`, which indexed 215k OCR annotation *tokens* under a single
fake `plan_id` with no area — retrieval there returned nothing usable and reconciliation
was a permanent no-op.

**Verified working:** the Encoder attaches a query embedding to each Function
(`"<type> of <area> square metres"`); retrieval returns real same-type, similar-area
precedents (e.g. a 14 m² bedroom query → real bedrooms of 13.7–14.5 m²); reconciliation
then moves room areas toward the precedents (e.g. dining 13.0 → 15.9 m², living 37.5 → 35.3
m²) within the brief band. Two latent bugs were fixed to get here: a read-only-mmap segfault
in the FAISS index build (only a float32 store triggered it) and a torch/faiss OpenMP
runtime collision (`OMP: Error #15`).

**Five embedding types** (composite fusion, primary retrieval vector):
```
E_composite = 0.3·E_text + 0.4·E_arch + 0.2·E_spatial + 0.1·E_visual
```
Text (all-MiniLM-L6-v2), Architectural (domain features), Spatial (coordinates/adjacency),
Visual (CLIP ViT-B/32).

---

### 1.4 Scoring Agent (MCDA) ⚠ *Weights & several formulas changed*

**Role:** Evaluate every node across five dimensions, produce a composite ∈ [0, 1].

**Composite aggregation** — generalised power mean with compensation parameter `ρ`:
```
S_composite = ( Σᵢ wᵢ · Sᵢ^ρ )^(1/ρ)          (ρ = 1.0 in the pipeline → weighted mean)
   ρ = 0  →  weighted geometric mean  exp( Σ wᵢ · ln Sᵢ )
   ρ < 1  →  anti-compensatory (penalises weak dimensions)
```

**Current weights** (normalised to 1.0):

| Dimension | Weight |
|---|---|
| Functional Adequacy S_f | 0.25 |
| Behavioral Performance S_b | 0.20 |
| Structural Feasibility S_s | 0.20 |
| Layout Efficiency S_l | 0.25 |
| Sustainability S_sust | 0.10 |

*(Original spec: 0.30 / 0.30 / 0.20 / 0.15 / 0.05 with a plain weighted sum. Layout was
weighted up once L became real measured geometry.)*

**1. Functional Adequacy (S_f)** — degree of satisfaction, not a binary count:
```
Coverage(fᵢ) = mean over related behaviors of  min(1, B_actual / B_expected)
S_f          = Σᵢ (priorityᵢ · Coverage(fᵢ)) / Σᵢ priorityᵢ
```

**2. Behavioral Performance (S_b)** — geometric mean, so one bad behavior sinks the score:
```
S_b = exp( mean( ln( min(1, B_actualᵢ / B_expectedᵢ) ) ) )
```

**3. Structural Feasibility (S_s):**
```
start 1.0;  × 0.7 if no load-bearing structure;  × 0.8 if no envelope structure
```

**4. Layout Efficiency (S_l)** ⚠ *4-term, new geometry formulas*:
```
S_l = 0.30·SpaceUtilisation + 0.25·Circulation + 0.30·Adjacency + 0.15·Compactness

Compactness = min(W, H) / max(W, H)                 of the footprint bbox   (was Area/Bbox)
Circulation = mean( direct_dist / graph_path_dist ) over room-graph paths   (was A* on grid)
Adjacency   = satisfied_requirements / total_requirements                   (measured, brief-derived)
SpaceUtil   = used_area / total_area
```
Measured `Circulation`/`Adjacency` values (flagged `*_measured` in layout metadata) are
trusted verbatim, **including a genuine 0.0**; the old 0.8/0.6 defaults apply only to
layouts that never went through placement.

**5. Sustainability (S_sust)** — computed from the design's actual envelope physics and
geometry (no longer a flat baseline):
```
S_sust = 0.35·EnvelopeThermal + 0.25·FormFactor + 0.15·GlazingFit
       + 0.15·MaterialCarbon + 0.10·Passive

EnvelopeThermal = clip((U_poor − mean_U) / (U_poor − U_good))   area-weighted envelope U
FormFactor      = compactness  (min(W,H)/max(W,H) — compact plans lose less heat)
GlazingFit      = 1.0 in the cold-climate optimum window ratio [0.12, 0.22], falling off outside
MaterialCarbon  = 1 − area-weighted embodied-carbon (wood/timber low, concrete/steel high)
Passive         = 1.0 for natural ventilation (no mechanical MEP), else 0.5
```
Reuses the same material U-values as the BehaviorCalculator for consistency. Now
**layout-coupled** — an elongated (linear) footprint earns a lower S_sust than a compact
one. Verified live: five prototypes scored 0.523 / 0.493 / 0.389 / 0.470 / 0.504, with the
`linear_layout` variant lowest (poor form factor) — real variation replacing the old
uniform 0.500.

**Gate:** a node failing the brief validator (§1.9) has its composite forced to **0.0**
before ranking.

---

### 1.5 Layout Generation Agent ⚠ *Algorithm replaced*

**Role:** Turn the room list + adjacency requirements into actual coordinates, a
circulation graph, and layout metrics — the concrete **L** in FBSL.

**Placement — Zoned Squarified Treemap** *(replaces force-directed + SLSQP)*:
1. Group rooms into **service | social | private** zones by room type.
2. Lay zones out as left-to-right columns sized by area.
3. Squarify rooms within each zone (near-square tiles).
4. Footprint aspect ratio comes from `metadata['layout_aspect']` (variant-controlled,
   clamped [0.4, 4.0]) — this is what makes compact vs linear plans geometrically different.

Result: a **gap-free tiling** where every room's tile area equals its target area and
adjacency is physically real (rooms share walls). No overlaps, no gaps.

> *Why the change:* force-directed layout uses universal repulsion (`−k_rep/d²`), which by
> construction never lets rooms share a wall — leaving gaps that made compactness and
> adjacency metrics untrustworthy. The treemap tiles exactly.

**Adjacency-aware tiling:** the treemap is computed twice — once area-ordered, once with
required partners ordered consecutively — and whichever satisfies more of the brief's
adjacency requirements is kept.

**Weighted preference matrix** (still computed, used for the adjacency-graph visual):
```
w(i,j) = 0.4·Functional_Dependency + 0.35·Traffic_Flow + 0.25·Privacy      ∈ [−1, 1]
```

**Circulation — Room-Connectivity Graph** *(replaces free-space A*)*:
- Build a graph: an edge connects two rooms whose tiles share a wall ≥ 0.7 m (a doorway),
  weighted by centroid distance.
- For every room pair, `graph_path = shortest_path(room_graph)`; efficiency =
  `direct_distance / graph_path_length`, averaged over all pairs.
- Adjacent rooms route directly (efficiency 1.0); distant rooms pay for each detour — so
  compact and linear footprints earn genuinely different circulation scores.

**Persisted L (per Room):** `position_vector {x,y,z}`, `width`, `length`,
`actual_adjacencies` (rooms it shares a wall with), `required_adjacencies` (brief partners).
`Layout.to_dict()` also serialises `room_order` and both the required and actual adjacency
matrices.

---

### 1.6 Refinement Agent (Gero FBS Reformulation)

**Role:** Iteratively close the gap between actual behaviors (Bₛ, from physics) and expected
behaviors (Bₑ), using Gero's three reformulation types.

**Deviation:** `avg_deviation = mean( |Bₛᵢ − Bₑᵢ| / Bₑᵢ )`.

**Physics-based behavior calculation (S → Bₛ)** — key convention:
```
actual_value = target_value × performance_ratio
```
Direction is **pre-normalised inside the calculator**, so the scorer's `min(1, actual/target)`
is correct for every category (thermal, acoustic, lighting, ventilation). Examples:
- **Thermal:** weighted U-value → R-value → `ratio = min(1, R/target)`; `actual = target·(0.7 + 0.3·ratio)`
- **Acoustic:** composite STC (log combination of transmission paths) → `ratio = min(1, STC/target)`
- **Lighting:** Daylight Factor (split-flux, window ratio × area) → `ratio = min(1, DF/target)`
- **Ventilation:** ACH from mechanical duct sizing, or natural stack effect if no MEP

**Three reformulation types** (by deviation band):
| Deviation | Type | Action |
|---|---|---|
| < 0.3 | 1 — Structure Modification | Add insulation / acoustic partition / window / MEP to close the gap |
| 0.3 – 0.6 | 2 — Behavior Relaxation | `tolerance ×= 1.2` |
| ≥ 0.6 | 3 — Function Redefinition | `priority ×= 0.8` on the worst-offending low-priority functions |

> **Guard:** a `natural_ventilation` variant that deliberately dropped its MEP is **not**
> re-given mechanical ventilation by Type-1 refinement — that would silently revert it to
> the base design and collapse the variant space.

**Convergence:** loop until `|score − prev_score| < 0.01` (ε), then output.

---

### 1.7 Pruning Agent

**Role:** Remove low-quality nodes from the explored set.

```
prune_threshold = max_score × 0.70
```
Any valid node scoring below 70 % of the best is dropped; brief-violating nodes (composite
forced to 0.0) are always dropped. Pruning is **diversity-preserving**: among survivors, the
best of each distinct design signature is kept before admitting a second copy of any.

*(Live example, reference brief: 8 scored → 7 kept.)*

---

### 1.8 Aggregation Agent ⚠ *Trigger tightened*

**Role:** Merge complementary high-scoring designs into a composite `aggregated_hybrid`.

**Trigger:**
```
high_scoring = { nodes with score ≥ 0.75 × top_score }       (was ≥ 0.9 × max)
aggregate IFF  |high_scoring| ≥ 2  AND  ≥ 2 DISTINCT design signatures
```
The distinct-signature guard is the important change: merging identical clones just
reproduces the same design and would dishonestly inflate the count. When the high-scorers
are all the same design, aggregation is **skipped** (this is correct behavior, logged as
`Aggregation skipped: high-scoring candidates are the same design`).

**Compatibility** (for the merge itself):
```
Compatibility(Nᵢ, Nⱼ) = 1 − (total_conflicts / total_elements)
```
Conflicts: function `conflicts_with`, behavior target gap > 0.5, incompatible materials,
area deviation > 20 % / room-count diff > 2.

*(Live example: 5 high-scorers → one `aggregated_hybrid` at composite 0.935.)*

---

### 1.9 Brief Validator (Hard Gate) — *new component*

Derived once from the root problem node, applied to every alternative before ranking:
- **Expected room types** (Counter of types from the brief's rooms)
- **Total-area band** = Σ function [min, max] × (1 ± `AREA_GRACE` = 0.10)
- **Required adjacencies**

A node **fails** (→ composite 0.0) if it has no rooms, is missing/under-counts a required
room type, or its total area falls outside the band. Adjacency shortfalls are soft warnings.
This is what killed the old "108 m², 3-bedroom design ranked #1 for a 220–280 m²,
4-bedroom brief" class of fake result.

---

### 1.10 Pipeline Orchestrator

**Role:** Central coordinator — runs the phases below, controls GoT expansion, and selects
the final top-k.

**Complexity-adaptive parameters:**
```
C_overall = 0.4·C_req + 0.6·C_fbsl
```
| Level | depth | breadth | nodes | prototypes |
|---|---|---|---|---|
| Low (<0.3) | ×0.7 | ×0.7 | ×0.6 | ×0.6 |
| Medium (0.3–0.6) | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| High (0.6–0.8) | ×1.3 | ×1.3 | ×1.5 | ×1.3 |
| Very High (≥0.8) | ×1.5 | ×1.5 | ×2.0 | ×1.5 |
(base target prototypes = 5)

**GoT stopping criteria:**
```
Stop if (improvement < 0.001 AND score > 0.7 AND stagnation ≥ patience)
     OR high_scoring_count ≥ 3
     OR depth / node budget reached
```
**Guard:** expansion never stops before depth ≥ 2, so all five Level-1 strategies are
always expanded before stagnation checks apply.

**Diversity machinery:**
- **Design signature** = fingerprint of physics/geometry-driving params (aspect,
  ventilation strategy, mean glazing ratio, materials, room-type counts, area bucket).
- **Dedup** collapses clone leaves before scoring.
- **Diversity-greedy final ranking:** at each rank, pick the most *novel* remaining design
  (new strategy family / new footprint class / new signature), best-score-first within
  ties. #1 is still the best overall; #2/#3 become the best *different* designs, not copies.

---

## 2. Overall Pipeline Workflow

**Phase 0 — Input:** user natural-language brief.

**Phase 1 — Encoding & Knowledge Retrieval**
1. Requirement parsing (Encoder → cloud/Ollama/rule chain) → FBSL problem node
2. Brief spec built once (validator baseline)
3. Precedent retrieval (Research → FAISS) + area reconciliation

**Phase 2 — Design-Space Generation**
4. Complexity analysis → adaptive depth/breadth/nodes
5. GoT Level-1: **5 named strategies** seeded
6. GoT Level-2+: micro-transformations, round-robin interleaved expansion

**Phase 3 — Evaluation & Selection**
7. Score every leaf (5 dimensions → composite), apply brief-validator gate
8. Dedup clone signatures
9. **Prune** at `0.70 × max`
10. **Aggregate** distinct high-scorers → `aggregated_hybrid`

**Phase 4 — Refinement & Layout (per candidate)**
11. Convergence loop: S → Bₛ → refine (Type 1/2/3) → generate L (treemap + room-graph
    circulation) → re-score, until `|Δscore| < 0.01`

**Phase 5 — Final Output**
12. Re-score, Pareto front, **diversity-greedy** ranking
13. Select top-k (`--top_k`, default 3–5)
14. Package each prototype: **complete FBSL** (`fbsl_data.json`), all 5 scores, layout SVG,
    adjacency graph, MD/HTML report

---

## 3. External Systems

**FAISS Vector Store:** `IndexFlatIP` over Finnish floor-plan embeddings; L2-normalised
inner product ≡ cosine similarity. Five embedding types fused as in §1.3.

**LLM services:** cloud (Groq / any OpenAI-compatible) primary, Ollama (`llama3.2`) local
fallback, rule-based parser as the deterministic floor. Encoding temperature 0.1
(consistency); generative steps use higher temperature.

**PostgreSQL (optional persistence):** `projects`, `fbsl_nodes`
(functions/behaviors/structures/layout as JSONB, all six scores, `generation_level`),
`evaluations` (per-behavior breakdown, strengths/weaknesses). Independent of the
filesystem `outputs/<project_id>/prototypes/<rank>_<id>/` bundle, which always contains
`fbsl_data.json`, `metadata.json`, `layout.svg`, `adjacency.png`, and the reports.

---

## 4. FBSL Node — Complete Stored Representation

Every prototype persists the full node (`node.to_dict()` → `fbsl_data.json`):

```
FBSLLayoutNode
├─ functions{}     F: name, category, priority, activities, spatial_requirements
├─ behaviors{}     B: category, metric, target_value, actual_value (physics), tolerance
├─ structures{}    S: type, material, category, dimensions, load_bearing
├─ layout          L: rooms{ position_vector{x,y,z}, width, length,
│                             required_adjacencies[], actual_adjacencies[] },
│                     room_order[], adjacency_matrix[][], actual_adjacency_matrix[][],
│                     circulation_efficiency, compactness_score,
│                     adjacency_satisfaction_score, space_utilization_ratio
├─ scores          functional / behavioral / structural / layout / sustainability / composite
└─ metadata        variant_type, layout_aspect, brief_validation, convergence_history, …
```

---

## 5. Known Limitations (stated, not hidden)

1. **S_sust** is now computed from real envelope physics + geometry and is layout-coupled
   (see §1.4). Its absolute level is bounded by the pipeline's default envelope (mostly
   gypsum partitions + glazing + concrete foundation), so scores sit in a realistic
   ~0.39–0.52 band unless a design explicitly adds insulation / low-carbon materials.
2. **RAG** now retrieves real room precedents with areas from CubiCasa5K and grounds room
   sizing (see §1.3). Remaining nuance: reconciliation only *nudges* areas (λ = 0.6 toward
   the stated value) and is clamped to the brief band, so its effect is intentionally
   modest; adjacency-pattern mining from precedents is not yet used.
3. **Ollama on 4 GB GPUs** still times out at 60 s; the cloud-first chain is what makes the
   LLM path reliable. With no key and no reachable Ollama, extraction falls to the
   rule-based parser (functional, but coarser on per-room areas).
4. The **force-directed / A\*** code paths remain in the tree as dead/legacy references,
   superseded by the treemap and room-graph methods.

---

*Document generated from source verification of `backend/` on the current `main` branch.
Formulas quoted are the ones actually executed, cross-checked against
`scoring_agent.py`, `spatial_algorithms.py`, `layout_agent.py`, `graph_of_thoughts.py`,
`behavior_calculator.py`, `encoder_agent.py`, `brief_validator.py`, and
`orchestrator.py`.*
