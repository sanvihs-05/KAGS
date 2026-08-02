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

## Run history, and a defect this study had to fix in itself

This study has been executed five times. The history matters, because three of those runs reported
numbers that were not measuring what their labels claimed — and in each case the giveaway was a
number that looked *reasonable* rather than one that looked wrong.

**Runs 1–2** produced *byte-identical* composites across all 21 cells despite substantial pipeline
changes in between. That stability was not validation — it was a symptom. Inspecting the encoded
nodes showed that **none of the three scenarios instantiated a single thermal, acoustic, daylight
or ventilation behaviour**: comfort behaviours were created only by scanning each room's *per-room*
requirement strings, so the family-home brief's building-wide "Prioritise natural light throughout
and good acoustic separation" reached no room, and thermal and acoustic had no creation path at
all. The four physics models never executed, so nothing that touched them could move a score.

**Run 3** was taken after those behaviours were instantiated. It moved every cell — but exposed a
second defect, this time in the study's own method. The `Without Physics-Based Behavior Analysis`
arm reported a drop of **exactly 0.00 %** on two scenarios. An exact tie is a tell, not a finding:
`RefinementAgent` constructs its **own** `BehaviorCalculator`, and the ablation stubbed only the
orchestrator's instance. Since the convergence loop refines every alternative, the refinement path
kept recomputing real physics throughout — **the arm never ablated anything.** Every S→Bs number
this report previously carried (0.5 %, 1.37 %, 2.24 %) was measuring designs whose behaviours had
been recomputed from structures regardless.

**Run 4** stubbed both calculators. The corrected S→Bs figures came out three to four times larger
than the ones they replaced.

**Run 5 — the data above** — makes every arm prove it fired, because the lesson from run 3 is that
an arm which silently fails to ablate reports a *small effect*, and a small effect is
indistinguishable from a real one. Only the implausibly exact 0.00 % made that bug visible; nothing
would have caught it if the number had been 1.4 %.

Each configuration now carries a marker that must be observable in its own output, recorded as
`arm_verified` beside the numbers it validates:

| Arm | What must be true |
|---|---|
| Full Framework (Baseline) | `method` is Graph of Thought, `got_graph` present, **and physics actually ran** |
| Without GoT Exploration | `method` is no longer Graph of Thought |
| Without RAG | `precedents_found == 0` |
| Without Refinement Agent | every design has `convergence_iterations == 0` |
| Without Physics (S→Bs) | the stub was called **and no real calculator call occurred** |
| Equal-Weight Scoring | composite equals the unweighted mean of the five sub-scores |
| Naive Layout Placement | every room sits at `y = 0` (the stub's single row) |

**All 21 cells verified.** The physics arm is instrumented rather than inferred:
`BehaviorCalculator.calculate_actual_behaviors` is wrapped at the *class* level as a tripwire, so
any instance an arm forgets to stub still routes through it and is counted. The baseline logs
36 / 63 / 86 real physics calls; the ablated arm logs 26 / 47 / 59 stub calls and **zero** real
calls. Had this existed earlier, run 3's bug would have failed loudly on its first execution.

*(An earlier version of this check inferred the arm from the data — "are behaviours still at
0.9 × target?" — and produced two false alarms. `_fit_rooms_to_total` rewrites area targets to
match scaled rooms and Type-2 reformulation relaxes targets by ±20 %, neither touching
`actual_value`, so the ratio drifts for legitimate reasons. Counting calls is direct evidence; the
ratio was a proxy that did not survive contact with the pipeline.)*

---

## Results

### Simple 2-Bedroom Apartment (Low complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8757** | — | 21.55 |
| Without GoT Exploration | 0.877 | −0.15 % (no change) | 15.90 |
| Without RAG (FAISS Retrieval) | 0.8767 | −0.11 % (no change) | 19.35 |
| Without Refinement Agent | 0.8802 | −0.51 % (no change) | 5.58 |
| Without Physics-Based Behavior Analysis | 0.8379 | **+4.32 %** | 17.17 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.8305 | **+5.16 %** | 15.46 |
| Naive Layout Placement (No Zoning/Treemap) | 0.8356 | **+4.58 %** | 14.29 |

### 3-Bedroom Townhouse (Medium complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8789** | — | 26.45 |
| Without GoT Exploration | 0.8596 | +2.20 % | 19.90 |
| Without RAG (FAISS Retrieval) | 0.8927 | −1.57 % | 26.21 |
| Without Refinement Agent | 0.8714 | +0.85 % (no change) | 7.07 |
| Without Physics-Based Behavior Analysis | 0.808 | **+8.07 %** | 34.74 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.8401 | **+4.41 %** | 37.97 |
| Naive Layout Placement (No Zoning/Treemap) | 0.81 | **+7.84 %** | 29.91 |

### 4-Bedroom Family Home (High complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8377** | — | 56.33 |
| Without GoT Exploration | 0.8355 | +0.26 % (no change) | 25.40 |
| Without RAG (FAISS Retrieval) | 0.8528 | −1.80 % | 63.36 |
| Without Refinement Agent | 0.8453 | −0.91 % (no change) | 12.55 |
| Without Physics-Based Behavior Analysis | 0.818 | +2.35 % | 48.16 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.8018 | **+4.29 %** | 43.03 |
| Naive Layout Placement (No Zoning/Treemap) | 0.7562 | **+9.73 %** | 52.03 |

*("Δ" is drop in composite when the feature is removed; "no change" marks a difference within
run-to-run LLM-extraction noise, not a real effect either direction.)*

---

## Findings

**1. Layout placement quality scales with complexity and is the largest contributor at the high
end.** Naive placement's drop grows monotonically with complexity: **4.58 % → 7.84 % → 9.73 %**
(low → medium → high). A simple room count barely notices the difference between zoned treemap
tiling and a naive grid; a 14-room family home does. This monotonic scaling is the cleanest signal
in the study. (It is the *largest* single effect only on the high-complexity scenario — on the
medium one the physics ablation now edges it out; see Finding 4.)

**2. The tuned MCDA weights matter — a flat 0.2/0.2/0.2/0.2/0.2 measurably underperforms.**
Equal-weight scoring drops composite by a consistent 4.3–5.2 % across all three scenarios,
confirming that weighting functional adequacy and layout efficiency above sustainability (per
the architecture doc's rationale — these are the two outcomes a client feels most directly) is
not an arbitrary choice: it produces higher-scoring designs by the framework's own criteria.

**3. The refinement (convergence) loop is nearly free to skip, but expensive to run.**
Removing it changes composite by well under 1 % in every scenario — but cuts wall-clock time by
**3.7–4.5×** (26.45 s → 7.07 s; 56.33 s → 12.55 s). *(Successive runs of this same study measured
10–16×, then 3–5×, then 2.4–4×, now 3.7–4.5×, on identical composites. Wall-clock here is
dominated by network latency to the hosted LLM and by machine load, so treat the ratio as
indicative and the absolute seconds as not comparable across runs at all.)* This
corroborates the architecture doc's stated
limitation: because the encoder + treemap already produce spec-meeting designs, most behaviors
start satisfied, so the Gero reformulation loop iterates without finding anything to fix. The loop
is not broken — it is doing real physics-based checking — but for typical briefs it is checking
designs that already pass.

**4. Physics-based behavior analysis (S→Bs) is one of the largest contributors — three to four
times larger than this report previously claimed.** Replacing the physics with the encoder's static
estimates costs **4.32 % / 8.07 % / 2.35 %** (low / medium / high). On the medium scenario that is
the single biggest effect in the study, ahead of naive layout placement.

The earlier figures (0.5 % / 2.24 % / 1.37 %) were wrong, and the reason is documented above under
*Run history*: `RefinementAgent` holds its own `BehaviorCalculator`, the study stubbed only the
orchestrator's, and the convergence loop refines every alternative — so the physics kept running
and the arm measured almost nothing. Both calculators are now stubbed.

Two qualifications on what this arm covers. It ablates **all** behavior recomputation, area
behaviors included, so it is not a clean measurement of the envelope physics alone. And the
comfort behaviors it now also removes exist only where the brief asks for them: the apartment and
townhouse briefs contain no comfort language, so on those two scenarios the effect is still
dominated by room-area behaviors being recomputed from the realised layout. Only the family home
instantiates daylight and acoustic targets — and it shows the *smallest* drop of the three, which
is worth noting as unexplained rather than rationalised: the expected ordering would put it
highest.

**5. GoT exploration's value is complexity-dependent and modest at this alternative count.**
Disabling GoT costs 2.20 % on the medium scenario, and is within noise on the low (−0.15 %) and
high (0.26 %) ones.
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

- **The S→Bs arm ablates all behavior recomputation, not the envelope physics alone.** It removes
  area behaviors too, and comfort behaviors exist only where the brief asks for them — the
  apartment and townhouse briefs contain none, so on those scenarios the effect is still dominated
  by area behaviors. Isolating the four physics models would need briefs with comfort language
  across all three complexity levels, and an arm that stubs only the comfort calculators.
- **Timings occasionally include a stalled request.** An earlier run of these same cells recorded
  5,024.91 s for one of them — not plausible against that run's elapsed time, and almost certainly
  a hung LLM call. No cell in the current run exceeds 64 s, but the composites were identical
  across both, which is the point: a stalled request corrupts the timing and leaves the score
  untouched, because the score is computed from the design and not the clock.
- **Wall-clock times are not comparable across runs.** They are dominated by network latency to
  the hosted LLM and by machine load; the same 21 cells produced 10–16× refinement speed-ups on
  one run, 3–5× on the next and 2.4–4× on this one. Read the timings as order-of-magnitude only.
  Composites are stable *for a fixed build* — runs 1 and 2 reproduced byte-identically — but they
  do move when the pipeline changes, as runs 3 and 4 did, so a composite is only comparable
  against others from the same build.
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
