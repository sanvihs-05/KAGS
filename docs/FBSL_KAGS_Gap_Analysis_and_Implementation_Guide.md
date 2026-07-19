# FBSL-KAGS — Gap Analysis & Implementation Guide

**Goal:** move the project from "produces plausible-looking but faked/inconsistent results" to "produces real, reproducible, constraint-respecting results" — where every number and image is *computed* from one source of truth and can be re-derived on demand.

This guide is grounded in the **actual current `backend/` code** (not the old flat-file version). Every gap below cites the real file and line where the problem lives.

---

## Part 0 — How this was verified

I traced the live code paths, not the design docs. The gaps below are what the code *actually does today*, with evidence. This matters because the aim (the architecture PDFs) is sound — the failure is in the implementation, and only a few specific places.

---

## Part 1 — The single root cause

**Scores are not driven by the physics or the geometry.** They are driven by (a) the encoder's static `actual_value` estimates and (b) a completeness fallback. So the design can change and the score barely moves — which is exactly why results look faked and why the layout image can disagree with the score table.

Everything in Part 2 is a symptom of that root cause. The fix (Part 3) is to make **one `FBSLLayoutNode` flow through every stage, with each stage computing real values from structures/geometry and writing them back, gated by a validator** — the same node the renderer and scorer both read.

---

## Part 2 — The gaps (with evidence)

| # | Gap | Evidence (current code) | Why it fakes results |
|---|-----|------------------------|----------------------|
| G1 | **RAG is decorative** | `research_agent.py:119-135` — `enhance_node_with_research` writes only `node.metadata['research_findings']` / `recommended_adjacencies`. No area, adjacency, or target is changed. | Generation is not conditioned on retrieval. It's two disconnected modules, not RAG. Retrieval could be deleted and no output number would change. |
| G2 | **Scoring reads stale behaviors** | `scoring_agent.py:190-192, 251-252` — reads `behav.actual_value` directly. | The `actual_value` is whatever the encoder set (`area*0.9`, `ventilation=0.4`, `daylight=1.8`). Physics never touches it on the scoring path → behaviors are effectively hardcoded at encode time. |
| G3 | **~~Direction bug in satisfaction~~ — RETRACTED after tracing the calculator** | `scoring_agent.py:192,252` uses `min(1, actual/target)`, BUT `behavior_calculator.py` returns `actual_value = target × performance_ratio` for *every* category (thermal `0.7+0.3·min(1,R/R_t)`, acoustic `min(1,STC/45)`, lighting `min(1,DF/DF_t)`, ventilation `min(1,score)`). | Direction is **already handled inside the calculator** by normalising to `target × satisfaction` (higher = better, capped at 1). So `min(1, actual/target)` is *correct* once physics runs. The generic "direction bug" from the external FIX_GUIDE does **not** apply here — replacing it with closeness/lower-is-better primitives would double-handle direction and break scoring. Residual (minor, not a bug): the convention caps over-provision at 1.0, so it can't penalise glare/over-ventilation — a defensible first-order choice. |
| G4 | **Physics only half-wired** | `BehaviorCalculator.calculate_actual_behaviors` called at `orchestrator.py:772` (convergence loop) and `refinement_agent.py:52,233` — **not** before GoT breadth scoring/pruning (`orchestrator.py:~360`). | The alternatives that get ranked and pruned are scored on stale values. The real physics runs too late (or not at all) to affect selection. |
| G5 | **Score-faking fallback** | `orchestrator.py:705-721` — `_estimate_node_quality` = `0.25 x (# non-empty FBSL sections)`, used as the score fallback at `:365, :697`. | A structurally complete node scores 1.0 regardless of whether the design is good. This is "completeness," not "quality," masquerading as a score. |
| G6 | **Force-directed layout can't tile a plan** | `spatial_algorithms.py` (ForceDirectedLayout), used by `layout_agent.py`. Repulsion `k_rep/d²` never lets rooms share a wall. | Produces floating boxes with gaps, not a contiguous floor plan. Adjacency/compactness computed from it are unreliable → the image and the metrics disagree. |
| G7 | **No brief validator** | No `Σarea ∈ [A_min,A_max]` / bedroom-count / required-room gate in the pipeline (validator helpers exist in `fbsl_models.py`/`finnish_fbsl_mapper.py` but don't gate output). | An invalid design (wrong area, missing rooms) can be ranked #1. This is what produced the 108 m² / 3-bedroom "winner" in the old run. |
| G8 | **Compactness metric is degenerate** | `S_l` uses `room_area / bbox_area` (in `layout_agent.py` metrics). | That ratio ≈ `1 − circulation` for any packed plan → always ~0.88, can't tell a square from a corridor. Layout score barely discriminates. |
| G9 | **Unit inconsistency in physics** | Ventilation/lighting in `behavior_calculator.py` — verify the produced number is the one that reaches the score (the old engine printed 38.7 ACH while results said 0.48). | If the physics output and the scored value use different units/paths, the "computed" number is theatre. |

**What is already OK (don't rewrite these):**
- Scoring is genuinely *computed*, not random — the composite uses a real generalized mean (`scoring_agent.py:110-126`). The bug is the *inputs* (G2/G3), not the aggregation.
- `BehaviorCalculator` physics exists and is real (`behavior_calculator.py`) — it's just wired in too late (G4).
- The encoder now creates window/HVAC/foundation structures (recent fix) — so the physics *can* read them. This helps but does not fix G2/G3/G4.

---

## Part 3 — Implementation guide (ordered, do it bottom-up)

**Philosophy: get one honest end-to-end path working before any GoT breadth.** One brief → one validated FBSL node → one real layout → real behaviours → one real score. Breadth over a hollow core is what produced the faked look. Do these in order; each is independently testable.

### Step 1 — Wire physics into the scoring path (fixes G2, G4)

Before *any* node is scored, recompute its behaviours from its structures.

- In the GoT scoring path (`orchestrator.py:~355-366`), call `self.behavior_calculator.calculate_actual_behaviors(alt)` **before** `scoring_agent.score_node(alt)`, exactly as the convergence loop already does at `:772`.
- Net effect: `behav.actual_value` reflects the current structures/geometry, not the encoder's guess.

### Step 2 — ~~Fix the direction bug~~ RETRACTED (do nothing here)

After tracing `behavior_calculator.py`, there is **no direction bug to fix in the scorer**. Every calculator already normalises its output to `actual_value = target × performance_ratio` (higher = better, capped at 1), so `min(1, actual/target)` in `scoring_agent.py` is correct *once Step 1 makes the calculator run*. Introducing direction-aware primitives in the scorer would double-handle direction and break it.

**Leave `scoring_agent.py` unchanged.** The only residual is that over-provision (daylight glare, over-ventilation) is capped at 1.0 rather than penalised — a defensible first-order modelling choice. If you later want to penalise excess, do it *inside the relevant calculator* (turn its `min(1, x/target)` into a plateau), not in the scorer.

### Step 3 — Add the brief validator as a hard gate (fixes G7)

New gate run at every stage boundary (encode, after generalize, before scoring, before final ranking):

- room count and every required room present;
- `Σ area ∈ [A_min, A_max]` (e.g. 220–280 m²);
- required adjacencies present, "avoid" adjacencies absent.

On failure: repair (rescale areas / add missing room) or set `composite = 0`. **No invalid node may be ranked.** This one gate kills the "108 m² / 3-bedroom winner" class of bug.

### Step 4 — Replace the completeness fallback (fixes G5)

`_estimate_node_quality` (`orchestrator.py:705`) should not stand in for a real score. Options, in order of preference:
1. Ensure Steps 1–2 always produce a real score so the fallback is never hit; **or**
2. If a node genuinely can't be scored (missing layout), return `0.0` and log it — never a plausible number. Completeness is a *gate*, not a *score*.

### Step 5 — Replace force-directed layout with a rectangular dissection (fixes G6, G8)

Force-directed leaves gaps by construction. Swap it for a **zoned squarified treemap** (or a slicing tree / CP-ILP if you want adjacency *optimised* rather than emergent):

- group rooms into social / private / service zones;
- place zones as blocks, treemap rooms within each zone → tiles the footprint exactly, hits total area, keeps within-zone adjacencies naturally.

Then recompute metrics from the real geometry:
- `compactness = min(W,H)/max(W,H)` (footprint squareness) — replaces the degenerate `area/bbox`;
- `circulation = mean(Euclidean/Manhattan from social hub to each room)`;
- `adjacency = satisfied_required / total_required` via shared-wall detection.

Keep A* (`spatial_algorithms.py`) for corridor paths on the 0.5 m grid — that part is fine.

### Step 6 — Make RAG actually condition generation (fixes G1)

Turn the decorative retrieval into a real two-pass loop:

1. LLM pass 1 → draft FBSL + retrieval query;
2. FAISS retrieve precedents `R(T)` (the `faiss_client.search` at `research_agent.py:60` already works);
3. **Deterministic reconciliation** — seed initial room areas and adjacency weights from precedent statistics:
   - area: `a* = λ·a_stated + (1−λ)·(similarity-weighted precedent mean)`, `λ=1` if the brief gave an explicit number;
   - adjacency prior: `P(adj(i,j)) = Σ s_p·1[i~j in p] / Σ s_p`, fed into the layout adjacency matrix.
4. **Prove it works by ablation:** run with and without Step 3; the grounded areas/adjacencies must change. If they don't, the link is still dead.

If the Finnish index isn't reliably loaded, the honest alternative is to state "precedent retrieval is future work" rather than present decorative RAG as functional.

### Step 7 — Verify unit consistency in the physics (fixes G9)

For each behaviour in `behavior_calculator.py`, assert the number the function produces is the number that reaches the score. Residential sanity bands: ACH ≈ 0.35–1.0, DF ≈ 2–5 %, indoor design temp ~21 °C, U-wall ≤ 0.25. Any order-of-magnitude output (e.g. 38.7 ACH) is a unit bug.

---

## Part 4 — How to prove it actually works (for the viva/report)

1. **Physics golden tests** — hand-work one thermal, acoustic, lighting, ventilation example; assert each function reproduces it (±1 %).
2. **Validator test** — feed a deliberately too-small program; confirm it is rejected, not ranked.
3. **RAG ablation** — with/without reconciliation, the grounded numbers must differ.
4. **Score-derivation log** — for the winning design, print its five sub-scores and their inputs, so you can answer "show me how 0.95 came about."
5. **One integration test** — brief in → validation passes, composite ∈ [0,1], images exist, and the layout metrics match the numbers in the score table (same source of truth).
6. **Honesty line for the writeup** — "thermal and ventilation are rigorous steady-state calculations; acoustic and daylight are first-order; RAG grounds numeric fields via similarity-weighted fusion." Scoping is defensible; faked precision is not.

---

## Part 5 — Priority order (what to do first)

| Priority | Step | Why first |
|---|---|---|
| 1 | Step 1 (wire physics into scoring) | **DONE.** Without it, every other score is meaningless. Smallest change, biggest truth gain. |
| — | ~~Step 2 (direction bug)~~ | **RETRACTED** — no bug; direction handled in the calculator. |
| 2 | Step 3 (validator) | **DONE + golden-tested.** `backend/core/brief_validator.py`; gates GoT alternatives and the aggregated node in `orchestrator.py` (composite forced to 0 on violation). Test proves: valid passes; 45%-area design, missing-bedroom design, empty design all rejected. |
| 3 | Step 5 (real layout) | Makes layout metrics and images honest and consistent. |
| 4 | Step 4 (kill fake fallback) | **DONE.** Removed `_estimate_node_quality` and its 3 call sites in `orchestrator.py`. Unscoreable / failed-aggregation nodes now get `0.0` + a log, and `_select_top` uses the real score (a validator-zeroed node stays 0.0 instead of being rescued by the `or` fallback). |
| 5 | Step 6 (real RAG) | Highest-value research claim, but largest effort — do after the core is honest. |
| 6 | Step 7 (unit audit) | Ongoing; fold into the golden tests. |

Step 1 + Step 3 (validator) convert the pipeline from "faked" to "honest but simple." Steps 4–6 make it defensible and complete.

---

*Grounded in the current codebase: `backend/agents/{research,scoring,refinement}_agent.py`, `backend/pipeline/orchestrator.py`, `backend/core/{behavior_calculator,spatial_algorithms}.py`, `backend/agents/layout_agent.py`. Line numbers reflect the state at the time of writing — re-check before editing.*
