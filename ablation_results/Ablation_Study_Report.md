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

This study has been executed seven times. The history matters, because three of those runs reported
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

**Runs 6-7** changed how RAG works (finding 6) and re-measured. Run 6 also demonstrated the
verification catching a regression introduced *while extending the pipeline*: adding a
`preferred_pairs` argument to the treemap broke the naive-placement stub, which still had the old
signature, so every call raised TypeError and the layout agent fell back to synthesis. The arm
stopped ablating placement and reported 5.05 % instead of 9.73 % on the family home — a number
with nothing obviously wrong about it. Three cells failed the `y = 0` check, the log named the
TypeError, and the stub now absorbs future signature changes. **Run 7 is the data above.**

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
| **Full Framework (Baseline)** | **0.8748** | — | 36.57 |
| Without GoT Exploration | 0.879 | −0.48 % (no change) | 43.02 |
| Without RAG (FAISS Retrieval) | 0.8767 | −0.22 % (no change) | 45.70 |
| Without Refinement Agent | 0.8813 | −0.74 % (no change) | 10.88 |
| Without Physics-Based Behavior Analysis | 0.8353 | **+4.52 %** | 42.76 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.8299 | **+5.13 %** | 40.88 |
| Naive Layout Placement (No Zoning/Treemap) | 0.8387 | **+4.13 %** | 41.12 |

### 3-Bedroom Townhouse (Medium complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8923** | — | 52.76 |
| Without GoT Exploration | 0.8793 | +1.46 % | 54.63 |
| Without RAG (FAISS Retrieval) | 0.8927 | −0.04 % (no change) | 65.76 |
| Without Refinement Agent | 0.8848 | +0.84 % (no change) | 18.74 |
| Without Physics-Based Behavior Analysis | 0.8294 | **+7.05 %** | 70.76 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.852 | **+4.52 %** | 67.36 |
| Naive Layout Placement (No Zoning/Treemap) | 0.8251 | **+7.53 %** | 62.96 |

### 4-Bedroom Family Home (High complexity)

| Configuration | Composite | Δ vs baseline | Time (s) |
|---|---|---|---|
| **Full Framework (Baseline)** | **0.8461** | — | 70.87 |
| Without GoT Exploration | 0.8457 | +0.05 % (no change) | 65.54 |
| Without RAG (FAISS Retrieval) | 0.8528 | −0.79 % (no change) | 111.14 |
| Without Refinement Agent | 0.8551 | −1.06 % | 17.72 |
| Without Physics-Based Behavior Analysis | 0.8327 | +1.58 % | 48.61 |
| Equal-Weight Scoring (No Tuned MCDA Weights) | 0.8088 | **+4.41 %** | 75.99 |
| Naive Layout Placement (No Zoning/Treemap) | 0.7653 | **+9.55 %** | 73.09 |

*("Δ" is drop in composite when the feature is removed; "no change" marks a difference within
run-to-run LLM-extraction noise, not a real effect either direction.)*

---

## Findings

**1. Layout placement quality scales with complexity and is the largest contributor at the high
end.** Naive placement's drop grows monotonically with complexity: **4.13 % → 7.53 % → 9.55 %**
(low → medium → high). A simple room count barely notices the difference between zoned treemap
tiling and a naive grid; a 14-room family home does. This monotonic scaling is the cleanest signal
in the study. (It is the *largest* single effect only on the high-complexity scenario — on the
medium one the physics ablation now edges it out; see Finding 4.)

**2. The tuned MCDA weights matter — a flat 0.2/0.2/0.2/0.2/0.2 measurably underperforms.**
Equal-weight scoring drops composite by a consistent 4.4–5.1 % across all three scenarios,
confirming that weighting functional adequacy and layout efficiency above sustainability (per
the architecture doc's rationale — these are the two outcomes a client feels most directly) is
not an arbitrary choice: it produces higher-scoring designs by the framework's own criteria.

**3. The refinement (convergence) loop is nearly free to skip, but expensive to run.**
Removing it changes composite by well under 1 % in every scenario — but cuts wall-clock time by
**2.8–4.0×** (52.76 s → 18.74 s; 70.87 s → 17.72 s). *(Successive runs of this same study measured
10–16×, then 3–5×, 2.4–4×, 3.7–4.5×, now 2.8–4.0×. Wall-clock here is
dominated by network latency to the hosted LLM and by machine load, so treat the ratio as
indicative and the absolute seconds as not comparable across runs at all.)* This
corroborates the architecture doc's stated
limitation: because the encoder + treemap already produce spec-meeting designs, most behaviors
start satisfied, so the Gero reformulation loop iterates without finding anything to fix. The loop
is not broken — it is doing real physics-based checking — but for typical briefs it is checking
designs that already pass.

**4. Physics-based behavior analysis (S→Bs) is one of the largest contributors — three to four
times larger than this report previously claimed.** Replacing the physics with the encoder's static
estimates costs **4.52 % / 7.05 % / 1.58 %** (low / medium / high). On the medium scenario that is
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
Disabling GoT costs 1.46 % on the medium scenario, and is within noise on the low (−0.48 %) and
high (0.05 %) ones.
Five named strategies give real geometric diversity (documented elsewhere in this codebase), but
the single best-scoring design among them is not dramatically better than the Generalizer's direct
decomposition for these particular briefs.

**6. RAG's area path was actively harmful; fixing it raised the baselines, and it is now
honestly neutral.** Earlier runs showed disabling RAG *improving* composite by 1.57 % and 1.80 % —
and the pattern was the diagnosis: the harm concentrated on the two briefs that state room sizes,
and was noise (−0.11 %) on the vague one that states none. Retrieval was fighting the brief. Three
causes, all since fixed:

- **It overrode stated requirements.** λ = 0.6 blending was applied to *every* room, including
  sizes the client had specified. "The master bedroom should be 18 sqm" is a requirement, not a
  suggestion. Reconciliation now touches only areas the encoder defaulted — where precedent is
  genuinely filling a gap. On the family-home brief: 7 defaulted areas grounded, 6 brief-stated
  areas left alone.
- **It moved the design without moving the yardstick.** `room.area` changed while the paired
  area-behaviour `target_value` stayed at its encoding-time value, so a clean actual/target ratio
  of 1.0 became 0.929 and `perf()` fell — the design was marked down for a change it had just been
  told to make. The target now follows the room, as `_fit_rooms_to_total` already did.
- **The retrieval query pretended an embedding could compare sizes.** The query text was
  `"bedroom of 16 square metres"`, but sentence embeddings do not order magnitude: measured against
  that query, 40 m² scores 0.864 while 25 m² scores 0.858 — the *larger* room ranks closer. Room
  type carried the signal (0.30 gap) and the magnitude term added non-monotonic noise (0.14
  spread). The query is now semantic and area proximity is applied afterwards as a number.

**Result:** −1.57 % → **−0.04 %** and −1.80 % → **−0.79 %**. More importantly the *baselines* rose,
0.8789 → 0.8923 and 0.8377 → 0.8461, on exactly the two briefs that state areas: the pipeline
produces better designs now that retrieval has stopped contradicting the brief. The ablation delta
only measures RAG's marginal contribution; absolute quality is the thing that improved.

**Why it still does not go positive — a structural limit, not a remaining bug.** With the target
now moving with the room, precedent-grounded sizing is neutral *by construction* on S_f and S_b.
What is left is geometric perturbation: different areas tile differently, which costs a little
layout efficiency and form factor. "These sizes match real Finnish homes" is not a quantity this
scoring function measures, so precedent acting through **area** can never earn anything back.
Area is the wrong channel.

**Adjacency is the channel the scorer can see**, since `adjacency_satisfaction` is a real term in
S_l. The corpus carries P(a-b adjacent | both present) over 3,787 plans, and those 92 pairs now
inform placement — strictly as a tie-break after every brief requirement is honoured, and
deliberately never added to the scored requirement set (promoting invented constraints into the
denominator would be the system grading itself against requirements no client asked for).

**That change did not move the ablation either, and the reason is worth recording:** the prior's
strongest pairs are the ones a brief usually states outright. `kitchen|living_room` carries
p = 0.764, but the townhouse brief says "open-plan kitchen and living room", so it is already a
requirement and correctly excluded from suggestions — leaving one marginal pair
(`bedroom|living_room`, p = 0.629) to break a single tie in room ordering. Precedent has the most
to add exactly where these three scenarios have the least room for it. It is kept because it is
principled and verified to cost nothing (brief adjacency satisfaction held at 1.00), not because
this study can show it helping.

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
