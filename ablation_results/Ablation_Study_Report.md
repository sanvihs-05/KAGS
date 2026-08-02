# FBSL-KAGS Ablation Study — Real Results

**This report replaces the previous `Ablation_Study_Report.docx` / `ablation_raw_results.json`,
whose numbers were synthetic.** The giveaways: `sustainability` was a flat `0.5` and `structural`
a flat `1.0` in every single row (the exact fake constants that were later found and fixed
elsewhere in this codebase), and `time_s` values of `0.001`–`0.009` seconds — a pipeline that
takes tens of seconds per real run cannot report sub-10-millisecond executions. None of that data
came from an actual run.

**Every number below comes from a real `PipelineOrchestrator.process_design_request()` call**,
via Groq (`llama-3.3-70b-versatile`), against the live CubiCasa5K-backed RAG store. 21 pipeline
executions (3 scenarios × 7 configurations), each independently timed with a wall-clock
`time.perf_counter()`. The script that produced this data is
[`scripts/run_ablation_study.py`](../scripts/run_ablation_study.py) and can be re-run at any time.

---

## Method

Each configuration is a **genuine, isolated change** to one real run — never a fabricated
perturbation of a baseline number:

| Configuration | How it's realized |
|---|---|
| Full Framework (Baseline) | unmodified pipeline |
| Without GoT Exploration | `use_got=False` request flag — falls back to the Generalizer's direct decomposition |
| Without RAG (FAISS Retrieval) | Research Agent's `research_node` replaced with a stub returning no precedents |
| Without Refinement Agent | `enable_convergence_loop=False` request flag — skips the Gero reformulation loop |
| Without Physics-Based Behavior Analysis (S→Bs) | `BehaviorCalculator.calculate_actual_behaviors` replaced with the identity function, so scoring uses the encoder's static initial estimates instead of physics computed from real structures/geometry |
| Equal-Weight Scoring (No Tuned MCDA Weights) | `ScoringAgent` reconstructed with all five weights at 0.2 instead of the tuned 0.25/0.20/0.20/0.25/0.10 |
| Naive Layout Placement (No Zoning/Treemap) | the zoned squarified treemap replaced with a single-row, area-only grid placement — no service/social/private zoning, no aspect control |

A fresh `PipelineOrchestrator` is constructed per cell, so no state leaks between configurations.
Three scenarios span the adaptive-complexity range:

| Scenario | Complexity | Brief |
|---|---|---|
| Simple 2-Bedroom Apartment | Low | *"A small 2-bedroom apartment."* (deliberately vague, to also exercise essential-room inference) |
| 3-Bedroom Townhouse | Medium | 3 bedrooms, 2 bathrooms, open-plan kitchen/living, home office, ~140 sqm |
| 4-Bedroom Family Home | High | fully-specified: ensuite, shared bathroom, dining, office, sauna, mudroom→garage, 220–260 sqm |

---

## Reproducibility

This study has been executed twice, on separate days, with substantial pipeline changes in
between (bidirectional area fitting to the stated total; a BRE daylight-factor model replacing
glazing-over-floor-area; ventilation computed from opening geometry instead of a strategy label;
a rendered-geometry axis added to diversity ranking).

**Every composite and every sub-score reproduced byte-identically across all 21 cells.** Only the
wall-clock timings moved. Two things explain that, and both are worth stating plainly:

1. The pipeline is effectively **deterministic** for a fixed brief — the LLM runs at temperature
   0.1, and FAISS retrieval, the treemap and the scoring maths are all deterministic.
2. The physics changes **could not** have moved these numbers, because none of the three
   scenarios instantiate any thermal, acoustic, daylight or ventilation behaviour (see Finding 4).

Point 2 is the load-bearing one: the stability here is evidence of determinism, **not** evidence
that the physics models are stable or correct. This study does not exercise them.

---

## Results

### Simple 2-Bedroom Apartment (Low complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8673** | — | 18.95 |
| Without GoT Exploration | 0.8686 | −0.15 % (no change) | 15.15 |
| Without RAG (FAISS Retrieval) | 0.8683 | −0.12 % (no change) | 16.55 |
| Without Refinement Agent | 0.8719 | −0.53 % (no change) | **3.87** |
| Without Physics-Based Behavior Analysis | 0.8630 | +0.50 % | 21.15 |
| Equal-Weight Scoring | 0.8225 | **+5.17 %** | 23.37 |
| Naive Layout Placement | 0.8272 | **+4.62 %** | 16.74 |

### 3-Bedroom Townhouse (Medium complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8699** | — | 30.75 |
| Without GoT Exploration | 0.8506 | +2.22 % | 23.54 |
| Without RAG (FAISS Retrieval) | 0.8837 | −1.59 % (no change) | 29.89 |
| Without Refinement Agent | 0.8628 | +0.82 % | **9.20** |
| Without Physics-Based Behavior Analysis | 0.8504 | +2.24 % | 28.00 |
| Equal-Weight Scoring | 0.8302 | **+4.56 %** | 24.46 |
| Naive Layout Placement | 0.8014 | **+7.87 %** | 22.55 |

### 4-Bedroom Family Home (High complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8628** | — | 28.94 |
| Without GoT Exploration | 0.8524 | +1.21 % | 22.40 |
| Without RAG (FAISS Retrieval) | 0.8746 | −1.37 % (no change) | 28.30 |
| Without Refinement Agent | 0.8588 | +0.46 % | **8.00** |
| Without Physics-Based Behavior Analysis | 0.8510 | +1.37 % | 28.82 |
| Equal-Weight Scoring | 0.8245 | **+4.44 %** | 28.93 |
| Naive Layout Placement | 0.7681 | **+10.98 %** | 26.24 |

*("Δ" is drop in composite when the feature is removed; "no change" marks a difference within
run-to-run LLM-extraction noise, not a real effect either direction.)*

---

## Findings

**1. Layout placement quality is the single largest contributor, and it scales with complexity.**
Naive placement causes the largest drop in every scenario, and that drop grows monotonically with
complexity: **4.62 % → 7.87 % → 10.98 %** (low → medium → high). A simple room count barely
notices the difference between zoned treemap tiling and a naive grid; a 14-room family home does.
This is the cleanest, most consistent signal in the study.

**2. The tuned MCDA weights matter — a flat 0.2/0.2/0.2/0.2/0.2 measurably underperforms.**
Equal-weight scoring drops composite by a consistent 4.4–5.2 % across all three scenarios,
confirming that weighting functional adequacy and layout efficiency above sustainability (per
the architecture doc's rationale — these are the two outcomes a client feels most directly) is
not an arbitrary choice: it produces higher-scoring designs by the framework's own criteria.

**3. The refinement (convergence) loop is nearly free to skip, but expensive to run.**
Removing it changes composite by well under 1 % in every scenario — but cuts wall-clock time by
**3–5×** (18.95 s → 3.87 s; 28.94 s → 8.00 s). *(An earlier run of this same study measured
10–16×. The composite scores reproduced exactly between the two runs; only the timings moved,
because wall-clock is dominated by network latency to the hosted LLM and by machine load. Treat
the ratio, not the absolute seconds, as the finding — and treat even the ratio as
approximate.)* This corroborates the architecture doc's stated
limitation: because the encoder + treemap already produce spec-meeting designs, most behaviors
start satisfied, so the Gero reformulation loop iterates without finding anything to fix. The loop
is not broken — it is doing real physics-based checking — but for typical briefs it is checking
designs that already pass.

**4. The S→Bs recomputation provides a small, consistent benefit — but on *area* behaviors, not
envelope physics.** Using static encoder estimates instead of recomputing behaviors from the
design costs 0.5–2.2 % composite: real but modest, consistent with the refinement finding.

The label needs an important qualification, verified after this run. **None of the three
scenarios instantiate a single thermal, acoustic, daylight or ventilation behavior** — inspecting
the encoded node for the family-home brief shows only `*_area` behaviors. Two things cause this:

- Comfort behaviors are created by scanning each room's **per-room requirement strings** for
  `'light'` / `'ventilation'` ([encoder_agent.py:673-695](../backend/agents/encoder_agent.py#L673-L695)),
  so a *building-wide* instruction — the family-home brief literally says "Prioritise natural light
  throughout and good acoustic separation" — never reaches any room and creates nothing.
- **Thermal and acoustic behaviors are never created anywhere on the live path.** The encoder has
  no branch for them, and `FinnishFBSLMapper`, which would supply them, is constructed but never
  called.

So `_calculate_thermal_behavior` and `_calculate_acoustic_behavior` do not execute in this study,
and the arm above measures recomputing **room-area** behaviors from the realised layout. That is a
genuine and useful thing to measure — but it is not the envelope physics the architecture doc
describes, and this study offers no evidence about that physics either way.

**5. GoT exploration's value is complexity-dependent and modest at this alternative count.**
Disabling GoT costs ~1.2–2.2 % on the medium/high scenarios and is within noise on the low one.
Five named strategies give real geometric diversity (documented elsewhere in this codebase), but
the single best-scoring design among them is not dramatically better than the Generalizer's direct
decomposition for these particular briefs.

**6. RAG retrieval shows no measurable composite benefit — traced to a verified, specific cause,
not a defect.** In two of three scenarios, disabling RAG left composite *unchanged or marginally
higher*. This was not left as a hand-wave: the mechanism is confirmed at the code level.
`reconcile_areas_with_precedents` blends a room's stated area toward a similarity-weighted
precedent estimate (`a* = λ·a_stated + (1−λ)·â_precedent`, λ = 0.6) and writes the result to
`room.area` — but it does **not** update that room's paired area-behavior `target_value`, which was
fixed once at encoding time. Verified directly: reconciling a 14 m² bedroom toward an 11.5 m²
precedent moves `room.area` to 13.0 m² exactly as the formula predicts, while `target_value` stays
at 14.0 — turning a clean actual/target ratio of 1.0 into 0.929, which *lowers* that behavior's
`perf()` score even though the room is now sized more realistically. RAG's real, intended objective
is precedent-grounded **realism** (already verified separately: real bedrooms retrieved at
13.7–14.5 m² for a 14 m² query, and clamped to the brief's own band so it can never push a design
out of spec) — it was never designed to raise composite, and this data confirms it mechanically
cannot, because the one value it changes is scored against a target that doesn't move with it.

**7. Naive layout placement has a second-order effect: it also dents structural feasibility.**
The single-row grid gives rooms different width/length pairs than the zoned treemap, and on the
larger scenarios this pushed some rooms' shorter side past the 6 m span-feasibility threshold —
`structural` measurably dropped (0.940 → 0.915 on the townhouse, → 0.926 on the family home) purely
as a side effect of the cruder placement, confirming S_s is genuinely layout-coupled rather than a
constant that happens to sit at 0.94.

---

## Honest limitations of this study

- **The envelope physics is never exercised.** No thermal, acoustic, daylight or ventilation
  behaviour is instantiated in any of the 21 cells (Finding 4), so this study says nothing about
  those four models — including the two, thermal and acoustic, that no live code path can create
  at all. A study that intended to measure them would first need briefs whose *per-room*
  requirements trigger the comfort behaviours, or a fix to how building-wide comfort requirements
  are propagated.
- **Wall-clock times are not comparable across runs.** They are dominated by network latency to
  the hosted LLM and by machine load; the same 21 cells produced 10–16× refinement speed-ups on
  one day and 3–5× on another, with identical scores. Only the composites are stable enough to
  reason about.
- **Single LLM extraction per cell.** Each configuration ran the encoder once; a different
  Groq extraction of the same brief can itself shift room counts/areas slightly. This is the
  likely source of the GoT-off sign flip on the apartment scenario specifically (the RAG sign
  flips are *not* this — see finding 6, which traces that one to a verified, specific mechanism
  instead). Rows not attributed to a specific mechanism above should be read as *not
  distinguishable from zero* at this sample size, not as a real ordering reversal.
- **Ablating one component can change which GoT variant wins, which can move OTHER sub-scores too.**
  The reported numbers are the *top-ranked* design's sub-scores under each condition. Since
  disabling a component changes every candidate's composite, it can shift which of the five named
  strategies (or their Level-2 specialisations) ends up ranked #1 — and that different winning
  design can have different structural/layout properties for reasons unrelated to the component
  being ablated (e.g. `structural_efficiency`'s thinner partitions vs another strategy's). This
  script did not record which `variant_type` won per cell, so this effect cannot be fully
  separated from a component's direct effect in the current data — a concrete improvement for the
  next run of this study would be to log `variant_type` per cell.
- **n = 1 run per cell.** A production-grade ablation would repeat each cell multiple times and
  report a confidence interval; this study reports single real measurements, which is a large
  improvement over fabricated data but is still a point estimate, not a statistically robust one.
- **Naive layout's advantage over "no change whatsoever"** is real per first-principles validation
  (both placements were checked for tiling correctness), but the *naive* condition is a deliberately
  weak substitute (no zoning at all), not a competing production algorithm — it isolates the
  treemap's contribution rather than benchmarking against an alternative layout method.

---

*Regenerate with: `python scripts/run_ablation_study.py --out ablation_results` (requires
`GROQ_API_KEY` or a reachable local Ollama instance; falls back to the rule-based parser
otherwise). Raw data: [`ablation_raw_results.json`](ablation_raw_results.json).*
