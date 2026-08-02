# FBSL-KAGS — System Architecture and Design Reference

A complete, implementation-faithful description of the FBSL-KAGS pipeline: a
multi-agent system that turns a natural-language architectural brief into a set
of ranked, fully-specified floor-plan prototypes. Every formula and parameter
below is the one the code actually executes; each is stated together with the
reason it takes that form and that value. A comparison with the project's
original design is given once, at the end (§10).

---

## 1. Foundations: the FBSL ontology

FBSL extends **Gero's Function–Behavior–Structure (FBS)** ontology with an
explicit fourth layer, **Layout (L)**.

- **Function (F)** — *what the building must do*. One function per required room
  (`provide_bedroom`, `provide_kitchen`…), each with a priority ∈ [0.5, 0.95],
  a set of activities, and spatial requirements (min / preferred / max area).
- **Behavior (B)** — *how it must perform*. Expected behaviors (Bₑ) are targets
  (thermal 21 °C, daylight factor 3 %, acoustic STC 45, ventilation, area);
  actual behaviors (Bₛ) are what the current design achieves, computed from
  physics.
- **Structure (S)** — *what it is built from*. Walls, partitions, glazing,
  foundation, MEP — each with a material, dimensions, and a load-bearing flag.
- **Layout (L)** — *where everything is*. Room coordinates, dimensions, the
  adjacency graph, and circulation. This is the extension: classical FBS stops
  at S, but a floor plan is meaningless without concrete geometry, so L is a
  first-class layer carrying `position_vector`, room dimensions, and both the
  required and achieved adjacency matrices.

The pipeline is a transformation over these four layers: F → Bₑ (formulation),
Bₑ → S (synthesis), S → Bₛ (analysis, via physics), S+L → geometry
(realisation), and Bₑ ↔ Bₛ (evaluation), iterated until the design converges.

---

## 2. Agents

### 2.1 Encoder Agent

**Role.** Convert raw brief text into an initial FBSL problem node.

**LLM provider chain.** Extraction is attempted in a fixed order so that a fast,
capable model is used when available but the system never hard-fails:

```
auto (default):  cloud LLM  →  local Ollama  →  rule-based parser
```

- *Cloud* is any OpenAI-compatible chat endpoint. `GROQ_API_KEY` alone targets
  Groq's `llama-3.3-70b-versatile`. **Why Groq/70B:** a 70-billion-parameter
  model extracts room programs far more reliably than a small local model, and
  Groq's LPU serving returns in ≈1 s, so the latency cost of a large model
  disappears.
- *Ollama* runs `llama3.2` locally. **Why the small 3B model as the local
  tier:** on a 4 GB laptop GPU a larger model cannot fit and swapping models
  costs 30–40 s; the small model is the only one that stays responsive offline.
- *Rule-based parser* is deterministic and always succeeds — the guaranteed
  floor.

Each cloud/Ollama call raises on any failure (timeout, rate-limit, malformed
JSON) so the loop falls through cleanly. **Why a chain rather than one
provider:** the local path is unreliable on constrained hardware and a cloud
key may be absent or rate-limited; layering keeps extraction quality high when
possible and functional always.

**Extraction robustness.** Two guards handle real-LLM output:
- *Area sanitising* — a missing or non-positive `area_min/area_max` is replaced
  with a per-room-type default band. **Why:** models sometimes emit a literal
  `"area_min": 0`; because a zero-area room sets an area-behavior target of 0,
  it would otherwise collapse the whole behavioral score.
- *Adjacency label resolution* — each adjacency's endpoints (which a model often
  returns as a descriptive *name*, e.g. "Master Bedroom") are mapped to a
  canonical room *type* present in the design ("bedroom"). **Why:** the layout
  agent groups rooms by type, so a name-space label would never match and every
  requirement would read as unsatisfied.

**Stated total area.** The brief's explicit total ("Total area 210–250 sqm",
"within 250 sqm") is parsed — guarded by whole-design cue words so a single
room's area is never mistaken for the total — and stored as
`target_total_area`. The room program is then scaled to land inside that band —
up to the band minimum when it falls short, **down to the band maximum when it
overshoots** — each room clamped to its function's `[min_area, max_area]`.
**Why both directions:** rooms extracted individually sum to the *net* usable
area, typically 15–20 % below the *gross* figure a client states, so without the
up-scaling the design silently under-delivers. The down-scaling matters just as
much: RAG area reconciliation blends in precedent areas and can push the program
*above* the stated total, which fails the brief validator's area gate — and when
that happens to every candidate, all of them score 0.0 and the ranking collapses
entirely.

**Output.** An FBSL problem node: one Function + area Behavior + partition and
glazing Structures per room, node-level HVAC and foundation, an initial room
list, and `required_adjacencies` (resolved to canonical types). Each Function
also receives a 384-dim query embedding (see §2.2) so precedent retrieval can
fire.

---

### 2.2 Research Agent (RAG over the CubiCasa5K corpus)

**Role.** Ground the design in real precedent floor plans.

**Corpus.** `cubicasa_rag_store` — one record per **room** (34,319 rooms from
3,787 real Finnish plans), each `{plan_id, room_type, area_m², neighbors}`. The
area is measured from the room polygon in each plan's SVG (shoelace area at
CubiCasa's 100-units-per-metre scale). **Why room-level:** the pipeline reasons
per room (a bedroom's typical size, what a kitchen sits next to), so the
retrieval unit must be a room, not a whole plan or an annotation token.

**Retrieval.** Each Function's query embedding (`"<type> of <area> square
metres"`, encoded with `all-MiniLM-L6-v2`, 384-dim) is searched against a FAISS
`IndexFlatIP` index:

```
similarity(q, pᵢ) = cos(E(q), E(pᵢ)) = (q · pᵢ) / (‖q‖ · ‖pᵢ‖)
```

**Why cosine / inner-product on L2-normalised vectors:** after normalisation the
inner product equals cosine similarity, which measures semantic closeness
independent of vector magnitude — the right notion for "which real rooms are
like this one". **Why `IndexFlatIP` (exact) rather than an approximate index:**
34 k vectors is small enough for exact search in milliseconds, so there is no
reason to trade accuracy for speed.

**Area reconciliation.** For each room, the stated area is blended with a
similarity-weighted precedent estimate and clamped to the brief band:

```
â_precedent = Σᵢ simᵢ · areaᵢ / Σᵢ simᵢ
a*          = λ · a_stated + (1 − λ) · â_precedent        (λ = 0.6)
```

**Why a weighted blend:** the user's stated area should dominate (hence λ = 0.6,
majority weight) but be nudged toward what real plans of that type actually use,
so an under- or over-specified room is corrected toward realism. **Why
similarity weighting:** more-similar precedents should count more. **Why
clamped to the brief band:** grounding must never push a room outside the user's
own constraints.

**Adjacency prior (advisory).** Across all 3,787 plans the system computes, per
room-type pair, `P(a adjacent b | both present)`. High-probability pairs the
brief did not state (kitchen↔dining 0.96, sauna↔bathroom 0.94, bedroom↔bathroom
0.57…) are recorded as advisory knowledge. **Why advisory only, not fed into
placement:** the zoned layout already co-locates functionally related rooms, so
injecting these as soft placement constraints was measured to change adjacency
in only 1 of 3 tests — the knowledge is genuine and worth surfacing in reports,
but acting on it added complexity for no reliable benefit.

---

### 2.3 Design-Space Generation (Graph-of-Thoughts)

**Role.** Explore genuinely different designs rather than converging on one.

**Level 1 — five named strategies.** The root node is expanded into five
seeds, each changing parameters the physics and geometry actually read, so each
earns a different score:

| Strategy | Footprint aspect | Parameter change | Why |
|---|---|---|---|
| `functional_priority` | 1.5 | high-priority rooms ×1.08, low ×0.92 | serve the most important functions best |
| `performance_optimized` | 1.1 | glazing ×0.90, partitions 0.15 m | favour thermal/acoustic performance |
| `structural_efficiency` | 1.35 | rooms ×0.95, thin 0.08 m partitions, concrete→steel | minimise material |
| `spatial_compactness` | 1.05 | near-square footprint, rooms ×0.97 | maximise compactness |
| `balanced` | 1.2 | baseline | a neutral all-round design |

**Why five distinct aspect ratios:** the footprint aspect drives the treemap, so
giving each strategy its own aspect guarantees five geometrically different
plans (and five different design signatures). All area scalings stay inside the
validator's ±10 % grace so a strategy is never rejected merely for its emphasis.

**Level 2+ — micro-transformations** specialise each strategy; names compose
(e.g. `spatial_compactness+linear_layout`):
- *Layout permutation* → compact (aspect 1.05), linear (2.4), courtyard (0.6).
  **Why these three:** they span the meaningful footprint space — square,
  elongated, and wrapped — and the parent's own aspect is skipped to avoid
  producing a clone.
- *Behavioral* → relaxed tolerances; **natural ventilation** (removes mechanical
  MEP, enlarges glazing ×1.25) — a real physics trade-off.
- *Functional* → keep only priority > 0.7 functions.
- *Structural* → alternative materials.

**Expansion order and stopping.** Children are interleaved round-robin across
strategies (geometry-changing variants first) before the breadth cut, **so the
variants that actually change the plan are never the ones truncated away.**
Expansion never stops before depth 2 (every Level-1 strategy is fully expanded);
after that it stops when improvement < 0.001 with score > 0.7 for `patience = 2`
expansions, or when ≥ 3 high-scoring alternatives exist. **Why these criteria:**
once several good, distinct designs exist, further search yields diminishing
returns, and the depth-2 guard prevents the search from quitting before the
design space is even built.

**Deduplication.** A **design signature** fingerprints the physics/geometry
parameters (aspect, ventilation strategy, mean glazing ratio, materials,
room-type counts, area bucket). Leaves with identical signatures are collapsed
before scoring. **Why:** many GoT paths differ only in labels while sharing every
real parameter; without dedup the ranking fills with copies of one design.

---

### 2.4 Scoring Agent (Multi-Criteria Decision Analysis)

**Role.** Score every node on five dimensions and combine them.

**Composite aggregation.**

```
S_composite = ( Σᵢ wᵢ · Sᵢ^ρ )^(1/ρ)          with ρ = 1
```

**Why a generalised power mean:** the exponent ρ tunes how much a weak dimension
can be compensated by strong ones. At ρ = 1 it is the weighted arithmetic mean
(full compensation); at ρ = 0 it becomes the weighted geometric mean; at ρ < 1
it becomes anti-compensatory (a weak dimension drags the whole score down). The
pipeline runs ρ = 1 for a stable, interpretable weighted average, while keeping
the machinery to harden it later.

**Weights** (normalised to 1): S_f 0.25, S_b 0.20, S_s 0.20, S_l 0.25, S_sust
0.10. **Why this split:** functional adequacy and layout efficiency are the two
outcomes a client feels most directly, so they carry the most weight (0.25
each); behavioral performance and structural feasibility are important but more
of a floor than a differentiator (0.20 each); sustainability is a genuine but
secondary criterion here (0.10).

Both S_f and S_b score each behavior with a shared **performance function** that
rewards *exceeding* the target, not merely meeting it:

```
ratio = B_actual / B_expected
perf(ratio) = 0.85 · ratio                              if ratio ≤ 1
            = 0.85 + 0.15 · min(1, (ratio − 1) / 0.30)  if ratio > 1
```

**Why this shape:** a design that just meets every target scores 0.85, leaving
headroom (the top 0.15 band) to reward one that outperforms — reaching 1.0 at
30 % above target. **Why 0.85 / 0.30:** meeting the brief should already be
"good" (0.85, not a middling score), and 30 % over target is a realistic ceiling
for "excellent" in building performance, beyond which extra margin is wasted.
The behavior calculator is correspondingly uncapped (its performance ratio
clamps at 2× rather than 1×) so `actual` genuinely carries performance above
target. **Why this matters:** the old hard cap `min(1, ratio)` flattened
"meets" and "far exceeds" both to 1.0, discarding the very information that
distinguishes a good envelope from an adequate one.

**1. Functional Adequacy (S_f)** — degree to which functions are served:

```
Coverage(fᵢ) = mean over the function's behaviors of perf(B_actual / B_expected)
S_f          = Σᵢ priorityᵢ · Coverage(fᵢ) / Σᵢ priorityᵢ
```

**Why priority-weighted:** a shortfall in a high-priority room (a bedroom) should
hurt more than in a low-priority one (storage). An area behavior compares a
room's own area to its per-room target (matched by function id, else by the room
type in the metric name) — never the whole-house sum.

**2. Behavioral Performance (S_b)** — geometric mean over all behaviors:

```
S_b = exp( mean( ln( perf(B_actualᵢ / B_expectedᵢ) ) ) )
```

**Why geometric, not arithmetic:** the geometric mean is dominated by its
smallest term, so a single failing behavior (say daylight at 0.2) sinks the
score — which is correct, because a house that is warm but pitch-dark is not
"80 % good". An arithmetic mean would let strong behaviors mask a critical
failure.

**3. Structural Feasibility (S_s):**

```
S_s = 0.35·MaterialValidity + 0.25·DimensionalFeasibility
    + 0.20·SpanFeasibility + 0.20·LoadPath
```

- *MaterialValidity* — fraction of load-bearing elements using a structural
  material (concrete/steel/brick/wood/masonry); a gypsum or glass load-bearing
  wall is penalised. **Why weighted highest (0.35):** using a material that
  cannot carry load is the most fundamental feasibility failure.
- *DimensionalFeasibility* — load-bearing thickness ≥ 0.15 m, foundation depth
  ≥ 0.6 m. **Why these values:** standard minimums for a structural wall and a
  frost-safe foundation.
- *SpanFeasibility* — fraction of rooms whose shorter side ≤ 6 m. **Why 6 m and
  the shorter side:** floor structure spans the short direction, and ~6 m is the
  practical limit before intermediate support is needed; this makes S_s
  layout-coupled.
- *LoadPath* — a foundation plus vertical support must both be present.

S_s is a *feasibility check*, not a quality gradient: it stays high for any sound
design and drops sharply for an infeasible one (verified 1.00 / 0.72 / 0.14 for
feasible / marginal / infeasible).

**4. Layout Efficiency (S_l):**

```
S_l = 0.30·SpaceUtilisation + 0.25·Circulation + 0.30·Adjacency + 0.15·Compactness

Compactness    = min(W, H) / max(W, H)   of the footprint bounding box
Circulation    = mean( direct_distance / graph_path_distance ) over room pairs
Adjacency      = satisfied_requirements / total_requirements
SpaceUtilisation = used_area / total_area
```

**Why compactness as footprint squareness** (`min/max`, not room-area ÷ box):
for a gap-free tiled plan the area ratio is always ≈ 1 and cannot tell a square
from a corridor; the aspect ratio of the bounding box does exactly that — 1.0
for a square plan, → 0 for a long thin one — which is what "compact" means
thermally and circulation-wise. **Why circulation as a path-ratio on the room
graph:** movement in a house goes through doorways between rooms, so circulation
is the shortest walk over the graph of wall-sharing rooms versus the straight-
line distance; adjacent rooms score 1.0, distant rooms pay for every detour.
**Why the 0.30/0.25/0.30/0.15 split:** adjacency and space utilisation are what
make a plan livable (0.30 each), circulation efficiency slightly less (0.25),
and compactness is a secondary shaping factor (0.15).

**5. Sustainability (S_sust):**

```
S_sust = 0.35·EnvelopeThermal + 0.25·FormFactor + 0.15·GlazingFit
       + 0.15·MaterialCarbon + 0.10·Passive

EnvelopeThermal = clip( (U_poor − mean_U) / (U_poor − U_good) )   U_good 0.15, U_poor 1.20
FormFactor      = compactness (min(W,H)/max(W,H))
GlazingFit      = 1.0 inside the window-ratio optimum [0.12, 0.22], falling off outside
MaterialCarbon  = 1 − area-weighted embodied carbon (wood low, concrete/steel high)
Passive         = 1.0 for natural ventilation (no mechanical MEP), else 0.5
```

**Why these five terms:** operational heating dominates a building's lifetime
impact, so envelope insulation (0.35) and form factor (0.25 — a compact shape
loses less heat per floor area) lead; glazing balance, embodied carbon of
materials, and passive conditioning fill out the rest. **Why the cold-climate
constants** (U_good 0.15, window optimum 0.12–0.22): the corpus and target
context are Finnish, where heat retention is paramount, so the "good" envelope
and the ideal glazing fraction are set to Nordic norms.

**Gate.** A node that fails the brief validator (§2.9) has its composite forced
to 0.0 before ranking.

---

### 2.5 Layout Generation Agent

**Role.** Turn the room list and adjacency requirements into real coordinates,
a circulation graph, and layout metrics — the concrete L.

**Placement — zoned squarified treemap.**
1. Group rooms into **service | social | private** zones by type.
2. Lay the zones out as left-to-right columns sized by area.
3. Squarify rooms within each zone into near-square tiles.
4. Take the footprint aspect ratio from `layout_aspect` (variant-controlled,
   clamped [0.4, 4.0]).

**Why a treemap rather than force-directed placement:** a squarified treemap
tiles the footprint *exactly and gap-free*, so every room's drawn area equals
its target area and adjacency (shared walls) is physically real. Force-directed
layout uses universal repulsion, which by construction leaves gaps between
rooms — making its compactness and adjacency metrics untrustworthy. **Why
zoning:** grouping service/social/private mirrors how dwellings are actually
organised and gives the treemap a sensible coarse structure before fine tiling.

**Adjacency.** The brief's required pairs are honored by ordering partner rooms
consecutively within a zone; the treemap is computed twice (area-ordered vs
partner-ordered) and whichever satisfies more requirements is kept. Satisfaction
is then *measured* on the placed tiles (two rooms are adjacent when they share a
wall segment ≥ 0.3 m) and reported as the real `adjacency_satisfaction_score`.

**Weighted preference matrix** (used for the adjacency-graph visual):

```
w(i, j) = 0.4·FunctionalDependency + 0.35·TrafficFlow + 0.25·Privacy      ∈ [−1, 1]
```

**Why 0.4/0.35/0.25:** functional need to be near (kitchen–dining) is the
strongest driver, expected foot traffic next, and privacy (which pushes rooms
*apart*, hence it can be negative) last.

**Circulation — room-connectivity graph.** An edge connects two rooms whose
tiles share a wall ≥ 0.7 m (a doorway), weighted by centroid distance;
circulation efficiency is the mean of `direct / graph-path` over all room pairs.
**Why a graph, not free-space A\*:** on a gap-free plan every room is an obstacle
and both path endpoints sit *inside* obstacles, so grid A\* finds no path and
returns zero; movement really does pass through doorways, so the room graph is
the physically correct model. **Why the 0.7 m doorway threshold:** a shared wall
must be at least a door-width to be a real connection.

**Persisted L.** Every Room stores its `position_vector {x,y,z}`, width, length,
`actual_adjacencies` (rooms it shares a wall with), and `required_adjacencies`;
the Layout serialises `room_order` and both adjacency matrices.

---

### 2.6 Refinement Agent (Gero reformulation)

**Role.** Close the gap between expected (Bₑ) and actual (Bₛ) behaviors.

**Physics — how Bₛ is computed from S** (every calculator returns
`actual = target × performance_ratio`, so direction is pre-normalised and the
scorer's ratio is directly meaningful):
- *Thermal* — area-weighted envelope U-value → R-value; `ratio = R / target_R`
  (target 5.0). **Why R-value:** thermal resistance is what actually governs heat
  retention, and averaging by area weights big walls correctly.
- *Acoustic* — composite STC from material STC + a thickness bonus;
  `ratio = STC / 45`. **Why 45:** STC 45–50 is the normal target for separating
  living spaces.
- *Lighting* — **BRE (Lynes) average daylight factor**, per room and
  floor-area-weighted: `DF = (T·A_w·θ) / (A_total·(1−R²))` with T 0.70, visible
  sky angle θ 75°, mean interior reflectance R 0.50, and `A_total` the room's
  total *interior surface* (floor + ceiling + walls). `ratio = DF / 3`.
  **Why DF 3 %:** the accepted threshold for "well-lit" habitable rooms.
  **Why divide by interior surface, not floor area:** that is what makes room
  proportion and ceiling height matter — a taller or more elongated room needs
  more glazing for the same average daylight. The previous
  `DF = window_ratio × 0.75 × 100` ignored geometry entirely and overstated DF
  by ~5× (an 18 % glazing ratio read 13.5 %, an atrium-like figure, where this
  gives a realistic ~2.5 %). A supplementary factor applies the BRE limiting-depth
  rule `L/W + L/H_w ≤ 2/(1−R)`, scaling DF by `limit/actual` when a room is too
  deep to daylight its back half — the standard states this as pass/fail, so the
  continuous form is an approximation chosen to keep ranking smooth. No
  orientation, sun path, latitude or obstruction survey: comparative, not a
  compliance figure.
- *Ventilation* — air changes per hour from the **actual opening geometry**, per
  room and floor-area-weighted to the dwelling. Natural flow takes the greater of
  the wind- and buoyancy-driven single-sided rates (BS 5925 / CIBSE AM10):
  `Q_wind = 0.025·A_open·v` and `Q_stack = (C_d/3)·A_open·√(g·H·ΔT/T̄)`, with
  `A_open = 0.45 × glazed area`; a mechanical system adds its design supply rate
  (0.5 l/s·m²) at boost, and envelope leakage contributes a 0.15 ACH floor.
  `ratio = ACH / 4` against the purge criterion. **Why purge (4 ACH) and not
  background (~0.5):** rapid ventilation is the demanding case openable area is
  actually sized for; against a background target every glazed design saturates
  and the metric stops discriminating. Because a room's glazed area scales with
  its floor area, the resulting ACH is governed by *glazing ratio and ceiling
  height* — the parameters the GoT variants change — while windowless interior
  rooms, served by plant alone, correctly pull the dwelling figure down.
  **Superseded** a label lookup (HVAC → 1.0, windows → 0.75, else 0.40) that did
  no physics: it could not separate two naturally ventilated designs with
  different glazing, and gave any design carrying an HVAC object a perfect score
  regardless of its capacity.

**Reformulation types** (Gero), chosen by average deviation:

| Deviation | Type | Action | Why |
|---|---|---|---|
| < 0.3 | 1 — Structure modification | add insulation / partition / window / MEP | a small gap is closed by adding the right structure |
| 0.3–0.6 | 2 — Behavior relaxation | tolerance ×1.2 | a moderate gap may mean the target was too tight |
| ≥ 0.6 | 3 — Function redefinition | priority ×0.8 | a large gap means the problem is over-constrained; de-prioritise the offender |

A `natural_ventilation` variant is never re-given mechanical MEP by Type-1
refinement, which would silently undo its trade-off. The loop iterates until
`|score(t) − score(t−1)| < 0.01`. **Why 0.01:** below a 1 % score change further
iteration is not worth the compute.

---

### 2.7 Pruning Agent

```
prune_threshold = max_score × 0.70
```

Any valid node below 70 % of the best score is dropped; brief-violating nodes
(score 0) are always dropped. Pruning is diversity-preserving: the best of each
distinct signature is kept before a second copy of any. **Why 0.70:** it removes
clearly inferior designs while keeping the genuine trade-off variants (a linear
or natural-ventilation design scoring, say, 0.80 against a 0.95 winner) that a
tighter cut would wrongly discard.

---

### 2.8 Aggregation Agent

```
high_scoring = { nodes with score ≥ 0.75 × top_score }
aggregate  IFF  |high_scoring| ≥ 2  AND  ≥ 2 distinct design signatures
Compatibility(Nᵢ, Nⱼ) = 1 − conflicts / total_elements
```

The high-scorers are merged into an `aggregated_hybrid`. **Why require ≥ 2
*distinct* signatures:** merging copies of one design just reproduces it, so
aggregation only runs when there is real diversity to combine. **Why the 0.75
band:** with real variant physics, a genuinely different design scores
meaningfully below the winner, so a 0.90 cut would only ever catch clones; 0.75
admits the complementary designs worth fusing.

---

### 2.9 Brief Validator (hard gate)

Built once from the root node, applied to every alternative. A node **fails**
(composite → 0) if it has no rooms, is missing or under-counts a required room
type, or its total area falls outside the brief band. The band is the stated
total (± 10 %) when the brief gave one, otherwise the room-program sum
(± 10 %). **Why a hard gate:** an under-sized or incomplete design must never
out-rank a valid one, however well it scores on other dimensions. **Why ± 10 %
grace:** small, legitimate variation (circulation allowance, variant scaling)
should not fail an otherwise-correct design.

---

### 2.10 Pipeline Orchestrator

**Complexity-adaptive parameters.**

```
C_overall = 0.4·C_req + 0.6·C_fbsl
```

**Why 0.4/0.6:** the extracted FBSL structure (room count, behavior diversity,
interdependencies) is a more reliable complexity signal than raw text, so it is
weighted higher.

`C_overall` selects a **tier scale** applied to base parameters (depth 2,
breadth 3, nodes 50, prototypes 5):

| Level | depth | breadth | nodes | prototypes |
|---|---|---|---|---|
| Low (< 0.3) | ×0.7 | ×0.7 | ×0.6 | ×0.6 |
| Medium (0.3–0.6) | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| High (0.6–0.8) | ×1.3 | ×1.3 | ×1.5 | ×1.3 |
| Very High (≥ 0.8) | ×1.5 | ×1.5 | ×2.0 | ×1.5 |

A **second multiplier, `component_scale`, then scales breadth, nodes and
prototypes** (not depth) by the number of rooms + functions — a brief with the
same overall score but more parts warrants a wider search:

```
component_scale     = min(1.5, 1 + (room_count + function_count) / 20)
target_prototypes   = max(3,  int(5 × prototypes_tier_scale × component_scale))
breadth             = max(2,  int(3 × breadth_tier_scale   × component_scale))
max_nodes           = max(20, int(50 × nodes_tier_scale    × component_scale))
depth               = max(1,  int(2 × depth_tier_scale))          # no component_scale
```

**Why scale with complexity:** a studio needs little exploration; a large
multi-zone brief warrants a deeper, wider search and more candidate prototypes.

`target_prototypes` is the count when the request leaves `max_alternatives`
unset; an explicit `max_alternatives` instead caps the candidate pool directly.
The kept set is then handed to aggregation, which may add **one** merged hybrid,
so the final prototype count is `target_prototypes` (or fewer, after brief-gate
and diversity pruning) **+ 1** when a hybrid is produced.

*Worked example (the shipped sample):* Medium brief, `C_overall ≈ 0.39`,
8 rooms + 8 functions → tier ×1.0, `component_scale = min(1.5, 1 + 16/20) = 1.5`
→ `target_prototypes = int(5 × 1.0 × 1.5) = 7`; pruning keeps 7, aggregation adds
1 hybrid → **8 designs returned**.

**Final ranking** is diversity-greedy: at each position the most novel remaining
design (new strategy family / footprint class / signature) is chosen, best score
first within ties. **Why:** #1 is still the best design overall, but #2 and #3
become the best *different* designs rather than near-copies of the winner.

---

## 3. FBSL node — the stored representation

Every prototype persists the full node as `fbsl_data.json`:

```
FBSLLayoutNode
├─ functions{}   F: name, category, priority, activities, spatial_requirements
├─ behaviors{}   B: category, metric, target, actual (physics), tolerance
├─ structures{}  S: type, material, category, dimensions, load_bearing
├─ layout        L: rooms{ position_vector, width, length,
│                          required_adjacencies[], actual_adjacencies[] },
│                   room_order[], adjacency_matrix[][], actual_adjacency_matrix[][],
│                   circulation_efficiency, compactness_score,
│                   adjacency_satisfaction_score, space_utilization_ratio
├─ scores        functional / behavioral / structural / layout / sustainability / composite
└─ metadata      variant_type, layout_aspect, brief_validation, convergence_history,
                 precedents_count, rag_areas_reconciled, precedent_adjacencies, …
```

Every number in this file is independently checkable: the composite equals the
weighted sum of the five sub-scores; the room areas sum to the total; the tiles
are gap-free and non-overlapping; the compactness equals the bounding-box aspect
ratio; and each "satisfied" adjacency corresponds to two rooms that really share
a wall.

---

## 4. End-to-end workflow

- **Phase 0 — Input.** Natural-language brief.
- **Phase 1 — Encode & retrieve.** Encoder (cloud → Ollama → rule) builds the
  problem node and fits it to the stated total; the brief spec is captured;
  Research retrieves precedents and reconciles areas.
- **Phase 2 — Design space.** Complexity sets depth/breadth; GoT seeds the five
  strategies and expands them; clone signatures are collapsed.
- **Phase 3 — Evaluate & select.** Every leaf is scored on five dimensions
  (brief-validator gate), pruned at 0.70 × max, and distinct high-scorers are
  aggregated into a hybrid.
- **Phase 4 — Refine & lay out.** For each candidate the convergence loop
  computes Bₛ, refines, generates the L (treemap + room-graph circulation), and
  re-scores until stable.
- **Phase 5 — Output.** Re-score, apply diversity-greedy ranking, select the
  top-k, and package each prototype (complete FBSL, all five scores, a
  matplotlib-rendered **PNG** floor plan and adjacency graph — embedded in the
  API result as `data:` URIs, with the SVG layout retained as a fallback —
  and an MD/HTML report).

---

## 5. External systems

- **FAISS vector store** — `IndexFlatIP` over 34,319 room embeddings
  (`all-MiniLM-L6-v2`, 384-dim); L2-normalised inner product ≡ cosine.
- **LLM services** — cloud (Groq / any OpenAI-compatible) primary, local Ollama
  (`llama3.2`) fallback, rule-based parser as the deterministic floor. Encoding
  temperature 0.1 for consistent structured extraction.
- **PostgreSQL (optional)** — `projects`, `fbsl_nodes`
  (F/B/S/L as JSONB, all six scores, generation level), `evaluations`
  (per-behavior breakdown, strengths/weaknesses). The filesystem bundle under
  `outputs/<project>/prototypes/<rank>_<id>/` (`fbsl_data.json`, `metadata.json`,
  `layout.svg`, `adjacency.png`, reports) is independent of the database.

---

## 6. Regression protection

`tests/` holds a pytest suite (26 tests) covering: the five scoring dimensions
are real and discriminating; the encoder's parsing, lexicon, area/adjacency
guards, and stated-total fitting; treemap tiling, measured adjacency, and
room-graph circulation; design-signature dedup and the five strategies; and the
LLM provider fallback chain. `pytest tests/` runs green.

---

## 7. Known limitations (stated, not hidden)

1. **S_s** is a feasibility *check*, not a quality gradient — it is uniform-high
   across structurally-equivalent variants and drops only for genuinely
   infeasible ones. This is the correct behavior for feasibility, but it means
   S_s adds little ranking spread between sound designs.
2. **The refinement loop** is often idle: because the encoder + treemap already
   produce spec-meeting designs, most behaviors start satisfied, so Gero
   reformulation rarely fires. The machinery is exercised only when a design
   genuinely misses a target.
3. **Precedent adjacency** is surfaced as advisory knowledge but does not drive
   placement (zoning already captures most of it).
4. **Local Ollama** on a 4 GB GPU times out at 60 s; the cloud-first chain is
   what makes the LLM path reliable.

---

## 8. Summary of key parameters and their rationale

| Parameter | Value | Why this value |
|---|---|---|
| Composite ρ | 1.0 | interpretable weighted mean; machinery kept to harden later |
| Score weights | .25/.20/.20/.25/.10 | client-felt outcomes (F, L) lead; sustainability secondary |
| perf() meet / band / margin | 0.85 / 0.15 / 0.30 | meeting brief is "good"; 30 % over target is realistic "excellent" |
| Behavior ratio clamp | 2× | let `actual` carry over-performance without absurd values |
| S_l split | .30/.25/.30/.15 | adjacency & utilisation lead; compactness a shaping term |
| S_sust split | .35/.25/.15/.15/.10 | operational heat (envelope, form) dominates lifetime impact |
| S_s split | .35/.25/.20/.20 | wrong material is the worst feasibility failure |
| U_good / U_poor | 0.15 / 1.20 | Nordic cold-climate envelope references |
| Glazing optimum | 0.12–0.22 | balances daylight against heat loss in a cold climate |
| Max span | 6 m | practical floor span before intermediate support |
| Doorway threshold | 0.7 m | minimum shared-wall length for a real connection |
| w(i,j) split | .4/.35/.25 | function proximity > traffic > privacy |
| Adjacency prior threshold | 0.60 | keep only reliably-adjacent room-type pairs |
| RAG blend λ | 0.6 | user's stated area dominates, nudged by precedent |
| Prune threshold | 0.70 × max | drop inferior designs, keep real trade-offs |
| Aggregate band | 0.75 × top | admit complementary designs, exclude clones |
| Area grace | ±10 % | tolerate legitimate size variation |
| Complexity C weights | 0.4 req / 0.6 fbsl | structured FBSL is the stronger complexity signal |
| Layout aspects | 1.05–2.4 | span square → elongated footprints for real geometric diversity |

---

## 9. Physics and formula quick-reference

| Quantity | Formula | Why this form |
|---|---|---|
| Composite | `(Σ wᵢ Sᵢ^ρ)^(1/ρ)`, ρ=1 | tunable compensation; weighted mean at ρ=1 |
| perf(ratio) | 0.85·ratio (≤1); 0.85+0.15·min(1,(ratio−1)/0.30) (>1) | reward exceeding, not just meeting, target |
| S_f | Σ priorityᵢ·Coverage(fᵢ) / Σ priorityᵢ | priority-weighted degree of satisfaction |
| S_b | exp(mean(ln(perf(ratioᵢ)))) | geometric mean — one bad behavior sinks it |
| Compactness | min(W,H)/max(W,H) | footprint squareness distinguishes square from corridor |
| Circulation | mean(direct / graph-path) | doorway-graph walking efficiency |
| Similarity | cos(E(q),E(pᵢ)) | magnitude-independent semantic closeness |
| Area reconcile | λ·a_stated + (1−λ)·â_precedent | user-led, precedent-nudged |
| Thermal | ratio = R / target_R | resistance governs heat retention |
| Acoustic | ratio = STC / 45 | STC is the transmission-loss standard |
| Lighting | ratio = DF / 3 | daylight factor threshold for habitable rooms |

---

## 10. Comparison with the original design

The sections above describe the system as it runs today. The table below records
where that differs from the project's original architecture, and why each change
was made — consolidated here so the body reads as a straight description.

| Area | Original design | Current implementation | Reason for the change |
|---|---|---|---|
| **Encoder LLM** | Ollama (gemma3 / llama3.1) only | Cloud-first chain: Groq → Ollama → rule-based parser | Local models cannot fit a 4 GB GPU and time out; a fast cloud model makes extraction reliable, with graceful fallback |
| **LLM room vocabulary** | 10 fixed types | 18 types incl. garage, sauna, mudroom, office, entry, closet | the narrow enum silently dropped valid rooms |
| **Stated total area** | not parsed or enforced | parsed, room program scaled to it, validator enforces it | designs were coming out ~15 % under the size the user asked for |
| **Generalizer** | 4 label-only variants | 5 named strategies with real parameter deltas, expanded in GoT | label-only variants scored identically; now each changes physics/geometry |
| **Layout algorithm** | force-directed + A\* on a grid | zoned squarified treemap + room-connectivity circulation | force-directed leaves gaps (untrustworthy metrics); grid A\* finds no path on a tiled plan |
| **Compactness** | room-area ÷ bounding-box area | min(W,H) / max(W,H) | the old ratio is ≈1 for any gap-free plan and cannot tell a square from a corridor |
| **Circulation** | A\* path length on a grid | shortest path on the doorway graph | endpoints sit inside obstacles on a tiled plan, so grid A\* returned 0 |
| **S_l** | 0.4·Compact + 0.3·Circ + 0.3·Adj (3 terms) | 0.30·Util + 0.25·Circ + 0.30·Adj + 0.15·Compact (4 terms) | space utilisation added as an explicit term |
| **Composite weights** | .30/.30/.20/.15/.05 | .25/.20/.20/.25/.10 | layout weighted up now that L is real, measured geometry |
| **Behavioral scoring** | min(1, actual/target) | perf() rewards exceeding target | the cap discarded all performance above target, pinning S_b at 1.0 |
| **Area behavior** | summed the whole house vs a per-room target | compares a room's own area to its target | a 12× ratio falsely pinned S_f/S_b at 1.0 |
| **Structural feasibility (S_s)** | start 1.0, ×0.7/×0.8 for missing parts (always 1.0) | material validity + dimensions + span + load path | the old form never varied; now it is a real feasibility check |
| **Sustainability (S_sust)** | flat 0.5 + rare metadata bonuses | envelope thermal + form + glazing + carbon + passive | was a constant contributing no ranking signal; now layout-coupled |
| **Aggregation trigger** | score ≥ max×0.9 AND compatibility > 0.7 | score ≥ top×0.75 AND ≥ 2 distinct signatures | the 0.9 cut only ever matched clones; merging clones reproduces one design |
| **Adjacency satisfaction** | assumed / hard-coded 0.6 | measured on shared walls, brief-derived | the score must reflect the actual plan |
| **RAG corpus** | annotation tokens, single fake plan id, no areas | 34,319 room records from 3,787 real plans, with areas | the old store returned nothing usable; retrieval was inert |
| **FBSL persistence** | L fields declared but never filled | full L generated and stored (coordinates + adjacency matrices) | the layout layer existed only in the SVG, not in the data |
| **Tests** | none | 26-test pytest suite | no regression protection for any of the above |

**Net effect.** In the original system three of the five scoring dimensions were
constant (two through bugs, one hard-coded), the layout metrics were untrustworthy,
RAG returned nothing, the stated size was ignored, and the "variants" scored
identically — so the ranking was effectively arbitrary. In the current system all
five dimensions are computed from real physics or geometry and discriminate
between designs; the layout is gap-free with measured adjacency and circulation;
retrieval grounds room sizes in real precedents; the brief's stated total is
honored; and every number in the output can be independently re-derived from the
stored geometry.

---

*This document reflects the code on the current `main` branch and was written by
reading the implementation directly (`scoring_agent.py`, `behavior_calculator.py`,
`layout_agent.py`, `graph_of_thoughts.py`, `encoder_agent.py`,
`research_agent.py`, `brief_validator.py`, `spatial_algorithms.py`,
`orchestrator.py`, `build_cubicasa_rag.py`).*
