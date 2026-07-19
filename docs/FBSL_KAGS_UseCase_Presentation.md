# FBSL-KAGS — Methodology and Use Case

**A Knowledge-Augmented Generative System that turns a natural-language design brief into a scored architectural floor plan, using the Function–Behavior–Structure–Layout (FBSL) ontology and a Graph-of-Thoughts exploration engine.**

This document explains how the system works — what each agent does and *why* — and illustrates it with one worked example carried through from start to finish.

---

## Contents

1. [Overview](#1-overview)
2. [The FBSL Representation](#2-the-fbsl-representation)
3. [The Agents and Their Roles](#3-the-agents-and-their-roles)
4. [Worked Example: A 4-Bedroom Family Home](#4-worked-example-a-4-bedroom-family-home)
5. [How the Layout Is Actually Generated](#5-how-the-layout-is-actually-generated)
6. [How Designs Are Scored](#6-how-designs-are-scored)
7. [How the System Improves a Weak Design](#7-how-the-system-improves-a-weak-design)
8. [Outputs](#8-outputs)
9. [What Works and What Is Still Limited](#9-what-works-and-what-is-still-limited)

---

## 1. Overview

Most generative layout tools jump straight from a text prompt to geometry, which means the result carries no reasoning about *why* rooms are where they are and no measurable notion of whether the design is any good. FBSL-KAGS is built around the opposite idea: before any geometry is drawn, the brief is turned into a structured design intent — what each space is *for*, how it should *perform*, and what it is *made of* — and only then is that intent laid out spatially and scored.

The pipeline is a collaboration between specialised agents:

```
NL brief → Research → Encoder → Complexity → ┌ Specializer → Layout → Scoring ┐ → Aggregation → Output
                                             └ Generalizer / Refinement       ┘
                                              (Graph-of-Thoughts loop)
```

Each agent hands the next a single shared object — an **FBSL node** — so the design intent stays legible and auditable the whole way through.

---

## 2. The FBSL Representation

FBSL is the common language every agent reads and writes. A design variant is one FBSL node with four parts:

- **Functions** — what a space must enable (e.g. *provide a kitchen*, with activities, priority, and a preferred area).
- **Behaviors** — measurable performance targets derived from functions (e.g. *kitchen ventilation ≥ 0.5 air changes per hour*, *bedroom daylight factor ≥ 2%*).
- **Structures** — the physical elements that deliver the behaviors (partitions, windows, an HVAC system, a foundation).
- **Layout** — the rooms and, once generated, their geometry.

The power of this split is that it makes design quality *checkable*. A behavior states a target; a structure is what makes it achievable; the scorer later compares the two. If a room needs daylight but no window structure exists, the system can see the gap — it isn't hidden inside a picture.

---

## 3. The Agents and Their Roles

**Research Agent.** Grounds the design in precedent. It searches a corpus of real residential floor plans (a Finnish plan library) and returns the five most relevant, supplying typical room adjacencies and proportion priors so the system isn't inventing conventions from scratch.

**Encoder Agent.** Translates the brief plus retrieved context into a structured FBSL node — creating the functions, behaviors, and structures for each room. This is where design intent first becomes explicit and machine-readable.

**Complexity Assessment.** Judges how hard the brief is and sizes the search accordingly. A simple studio doesn't warrant the same exploration budget as a twelve-room family home, so the system reads the brief's demands and the richness of the FBSL node and picks a search depth and breadth to match.

**Specializer.** Inside the exploration loop, it takes a design and produces several distinct children — varying room proportions, which adjacencies to prioritise, and structure choices — so the system explores genuinely different options rather than one.

**Layout Agent.** The spatial engine. It takes an FBSL node and produces an actual positioned floor plan (covered in §5).

**Scoring Agent.** Evaluates each laid-out design against its own stated intent across five criteria (§6), producing a single comparable score.

**Generalizer.** Prunes the exploration — keeping the strongest handful of variants at each level so the search stays focused.

**Refinement Agent.** When a design scores poorly, it diagnoses *why* and reformulates the FBSL node to fix it (§7).

**Aggregation & Final Output.** Ranks the survivors, selects the best, and writes out the drawings and records.

---

## 4. Worked Example: A 4-Bedroom Family Home

Everything below follows this brief:

> "Design a **4-bedroom family home**, approximately **185 m²**, for a **Nordic climate**. Open-plan kitchen with living/dining. Master bedroom with ensuite. A home office. Good daylight and cross-ventilation. Keep the bedrooms private from the kitchen."

**After Research**, the system has five Nordic precedents that confirm the usual pairings — kitchen next to dining, master next to its ensuite, an entry/mudroom buffer.

**After Encoding**, the brief has become an FBSL node with twelve functions (one per room), fourteen behaviors (area, daylight, and ventilation targets), and a set of structures. For every habitable room the Encoder creates a partition and a window; for the whole dwelling it adds one HVAC system and one load-bearing foundation. A representative slice:

```
Room: Living / Dining
  Function : provide open-plan living and dining, area target 40 m²
  Behavior : daylight factor ≥ 2%,  area ≈ 40 m²
  Structure: partition (gypsum board),  window (glazing, window-ratio 0.25)

Dwelling-level structures:
  HVAC ventilation system  (delivers the ventilation behavior)
  Reinforced-concrete foundation  (load-bearing)
```

The window's *window-ratio* is the value the scorer will later read to judge daylight; the HVAC system is what makes the ventilation behavior achievable; the foundation is what makes the structure count as physically sound. These aren't decorative — each one exists to satisfy a specific downstream check.

**After Complexity Assessment**, the brief is judged **HIGH** complexity — twelve rooms, a mix of privacy and openness constraints, and a rich FBSL node. The system therefore explores to a depth of two levels, five variants wide, which is enough to compare meaningfully different arrangements without exploding the search.

The exploration loop then produces variants, lays each one out, scores them, and keeps the best — the subject of the next two sections.

---

## 5. How the Layout Is Actually Generated

The Layout Agent turns one FBSL node into a positioned plan in a sequence of steps. The intent is to honour the design's adjacency preferences while keeping rooms at their target sizes and out of each other's way.

1. **Room sizing.** Each room starts as a rough square sized to its target area, with sensible width limits so it can be reshaped but not distorted.

2. **Adjacency weighting.** The system converts the design's relationships into a single preference per room pair. Pairs that should be together (kitchen–living, master–ensuite) get a strong positive weight; pairs that should be apart (bedrooms–kitchen) get a negative one; everything else is neutral. This one number per pair is what drives placement.

3. **Force-directed placement.** Rooms are treated like charged particles: positively-weighted pairs attract, everything repels enough to avoid piling up, and the system lets the arrangement settle. This produces a natural-looking first layout where related rooms cluster and private ones drift apart — for example, the bedrooms settle well away from the kitchen, as the brief asked.

4. **Constraint refinement.** A constrained optimiser then tidies the arrangement: it holds each room to its target area, pulls preferred neighbours flush against each other, and applies a heavy penalty for any overlap so no two rooms occupy the same space.

5. **Geometry.** Each room becomes a real polygon, from which the system can read exact adjacencies (which rooms share a wall), centroids (where to put labels), and gaps.

6. **Circulation.** For each important connection, a path-finder routes a corridor between rooms around obstacles, giving the plan its circulation.

7–9. **Assembly, metrics, drawings.** The rooms, corridors, and computed metrics — how efficiently space is used, how direct the circulation is, how many required adjacencies were actually met — are packaged into a layout, and the drawings are rendered (§8).

The result for the worked example: related rooms adjacent, bedrooms private, rooms at their intended sizes, and roughly three-quarters of the footprint used productively with good circulation.

---

## 6. How Designs Are Scored

Every laid-out design is judged against the intent it started from, on five criteria:

| Criterion | What it asks |
|---|---|
| **Functions** | Are all the required spaces present and served? |
| **Behaviors** | Do the measurable targets — daylight, ventilation, area — hold? |
| **Structures** | Is the design physically sound and properly enveloped? |
| **Layout** | Is space used well, with efficient circulation and satisfied adjacencies? |
| **Sustainability** | Do orientation and materials support performance? |

The five are combined so that a design can't hide a serious weakness behind strengths elsewhere — one badly-failing criterion pulls the whole score down. That is deliberate: a house with beautiful circulation but no daylight is not a good house.

This combining rule is also what surfaced a real bug in the system. The behavior score is built so that a single unmet target (say, a room with *no daylight at all*) drags it toward zero. Early on, the Encoder created only a bare partition for each room and no windows, so the daylight target read as completely unmet for every design — and the behavior score collapsed to **0.21** no matter how good the geometry was. The structural score was similarly stuck at **0.56**, because nothing in the design was marked load-bearing or as envelope, triggering two blanket penalties.

The fix was to make the Encoder create the structures the checks were looking for: a window per habitable room, an HVAC system, and a load-bearing foundation. With those in place the daylight and ventilation targets are met, the structural penalties disappear, and the scores move as expected:

| | Behaviors | Structures | Overall |
|---|---|---|---|
| Before | 0.21 | 0.56 | 0.48 |
| After | 0.87 | 1.00 | 0.86 |

The lesson is exactly the one FBSL is designed to make visible: the geometry was never the problem — the design was missing structures its own behaviors required, and because intent and structure are explicit, the system could pinpoint that.

---

## 7. How the System Improves a Weak Design

When a design scores below what the brief should allow, the Refinement Agent doesn't just nudge geometry — it reformulates the design intent, following a well-known model of how designers rethink problems:

- If the gap is **small**, adjust or add **structures** (this is what fixed the family home — adding the missing windows, HVAC, and foundation).
- If the gap is **moderate**, relax a **behavior** target that is over-constraining the design.
- If the gap is **large**, rethink a room's **function** entirely.

Choosing the lightest intervention that closes the gap keeps the design faithful to the brief while still improving it. For the worked example, structural additions alone lifted the overall score from **0.48 to 0.86**.

---

## 8. Outputs

The Final Output Agent selects the best-ranked design and produces:

- an **SVG floor plan** — rooms coloured by type, with labels, circulation paths, a legend, and a metrics panel;
- a **PNG adjacency graph** — a two-panel drawing pairing the spatial layout with a network view of which rooms connect, edges coloured by relationship type (critical, spatial, proximity, bridging);
- a **prototype record (JSON)** and a **Pareto comparison** of the surviving variants.

The two drawings are generated from the same underlying room geometry — one renders it as a plan, the other as a connectivity network — so they always agree.

---

## 9. What Works and What Is Still Limited

**Works well.** The FBSL split makes design quality measurable and, crucially, *diagnosable* — the daylight/structure bug above was found and fixed precisely because intent and structure are explicit rather than baked into an image. Related rooms cluster and private ones separate as briefs ask, and the Graph-of-Thoughts loop reliably surfaces a strong variant.

**Still limited.** Open-plan briefs with few partitions under-report adjacencies, because the adjacency check looks for shared walls that barely exist in such plans. Sustainability is not yet tied to the geometry — it reflects orientation and materials but not, say, actual solar exposure of the placed rooms. And complex briefs (vertical stacking, courtyards) show more run-to-run variation, since the placement step is sensitive to where rooms happen to start. These are the natural next areas of work.

---

*This walkthrough reflects the actual FBSL-KAGS pipeline (`backend/agents/`, `backend/core/`, `backend/utils/`). The 4-bedroom home is the running example; the before/after scores reflect the encoder structural fix.*
