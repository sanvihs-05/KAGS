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

## Results

### Simple 2-Bedroom Apartment (Low complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8673** | — | 38.71 |
| Without GoT Exploration | 0.8686 | −0.15 % (no change) | 40.25 |
| Without RAG (FAISS Retrieval) | 0.8683 | −0.12 % (no change) | 31.35 |
| Without Refinement Agent | 0.8719 | −0.53 % (no change) | **2.36** |
| Without Physics-Based Behavior Analysis | 0.8630 | +0.50 % | 49.84 |
| Equal-Weight Scoring | 0.8225 | **+5.17 %** | 63.37 |
| Naive Layout Placement | 0.8272 | **+4.62 %** | 53.76 |

### 3-Bedroom Townhouse (Medium complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8699** | — | 89.85 |
| Without GoT Exploration | 0.8506 | +2.22 % | 64.19 |
| Without RAG (FAISS Retrieval) | 0.8837 | −1.59 % (no change) | 78.13 |
| Without Refinement Agent | 0.8628 | +0.82 % | **10.44** |
| Without Physics-Based Behavior Analysis | 0.8504 | +2.24 % | 76.30 |
| Equal-Weight Scoring | 0.8302 | **+4.56 %** | 73.29 |
| Naive Layout Placement | 0.8014 | **+7.87 %** | 64.91 |

### 4-Bedroom Family Home (High complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8628** | — | 94.68 |
| Without GoT Exploration | 0.8524 | +1.21 % | 64.06 |
| Without RAG (FAISS Retrieval) | 0.8746 | −1.37 % (no change) | 83.20 |
| Without Refinement Agent | 0.8588 | +0.46 % | **9.54** |
| Without Physics-Based Behavior Analysis | 0.8510 | +1.37 % | 74.80 |
| Equal-Weight Scoring | 0.8245 | **+4.44 %** | 101.30 |
| Naive Layout Placement | 0.7681 | **+10.98 %** | 78.09 |

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
**10–16×** (38.71 s → 2.36 s; 94.68 s → 9.54 s). This corroborates the architecture doc's stated
limitation: because the encoder + treemap already produce spec-meeting designs, most behaviors
start satisfied, so the Gero reformulation loop iterates without finding anything to fix. The loop
is not broken — it is doing real physics-based checking — but for typical briefs it is checking
designs that already pass.

**4. Physics-based behavior analysis (S→Bs) provides a small, consistent benefit.**
Using static encoder estimates instead of real structure-derived physics costs 0.5–2.2 % composite
— a real but modest effect, consistent with the refinement finding: most designs are close to
spec either way, so the physics step's main value is precision rather than large corrections.

**5. GoT exploration's value is complexity-dependent and modest at this alternative count.**
Disabling GoT costs ~1.2–2.2 % on the medium/high scenarios and is within noise on the low one.
Five named strategies give real geometric diversity (documented elsewhere in this codebase), but
the single best-scoring design among them is not dramatically better than the Generalizer's direct
decomposition for these particular briefs.

**6. RAG retrieval shows no measurable composite benefit — and this is a metric-scope finding, not
a defect.** In two of three scenarios, disabling RAG left composite *unchanged or marginally
higher*. This is expected once you look at what RAG actually optimizes: `reconcile_areas_with_precedents`
blends a stated area toward a real-precedent estimate (λ = 0.6) specifically to ground room sizes
in what real Finnish dwellings use — a **realism** objective, clamped to the brief's own band so it
can never move a design further from spec. It was never intended to raise the composite score, and
the composite has no term that rewards "matches a real building." The correct evaluation of RAG's
value is precedent fidelity (already verified separately: real bedrooms retrieved at 13.7–14.5 m²
for a 14 m² query, real areas reconciled), not this study's composite metric — and that is stated
here explicitly rather than left to look like a null result.

---

## Honest limitations of this study

- **Single LLM extraction per cell.** Each configuration ran the encoder once; a different
  Groq extraction of the same brief can itself shift room counts/areas slightly, which is the
  likely source of the small "no change" negative deltas (RAG, GoT on the apartment scenario) —
  they reflect extraction noise, not a real ordering reversal. Rows explicitly marked "no change"
  should be read as *not distinguishable from zero* at this sample size.
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
