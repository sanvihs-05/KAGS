"""Real ablation study: actually runs the FBSL-KAGS pipeline end-to-end for
every (scenario x configuration) cell and records genuinely measured metrics.

This REPLACES ablation_results/ablation_raw_results.json, whose numbers were
synthetic (sustainability flat 0.5, structural flat 1.0 -- the exact fake
constants fixed elsewhere in this codebase -- and time_s of 0.001-0.009s,
which is physically impossible for a pipeline that takes tens of seconds per
run). Every number this script produces comes from a real
PipelineOrchestrator.process_design_request() call.

Each ablation arm is a genuine, real change to the running pipeline for that
one call -- either an existing request flag (use_got, enable_convergence_loop)
or a scoped monkeypatch of one component on a freshly-constructed orchestrator
instance (never a permanent code change, and never shared across arms).

Usage:
    python scripts/run_ablation_study.py --out ablation_results
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)  # keep the run quiet; we print our own summary

from backend.pipeline.orchestrator import PipelineOrchestrator  # noqa: E402
from backend.agents.scoring_agent import ScoringAgent  # noqa: E402

SCENARIOS = {
    "simple_apartment": {
        "scenario": "Simple 2-Bedroom Apartment (vague brief)",
        "complexity": "Low",
        "brief": "A small 2-bedroom apartment.",
    },
    "townhouse": {
        "scenario": "3-Bedroom Townhouse",
        "complexity": "Medium",
        "brief": (
            "Design a 3-bedroom townhouse with two bathrooms, an open-plan "
            "kitchen and living room, and a small home office. Total area "
            "around 140 sqm."
        ),
    },
    "family_home": {
        "scenario": "4-Bedroom Family Home (detailed brief)",
        "complexity": "High",
        "brief": (
            "Design a 4-bedroom family home. The master bedroom should be "
            "18 sqm with an ensuite bathroom attached, and three further "
            "bedrooms of 12-14 sqm each sharing a common bathroom. Provide "
            "an open-plan kitchen and living room of about 40 sqm, with the "
            "kitchen connected to a separate dining area of 12 sqm. Include "
            "a quiet home office of 10 sqm, a sauna, a laundry, and a "
            "mudroom that connects to a garage. Prioritise natural light "
            "throughout and good acoustic separation between the bedrooms "
            "and living spaces. Total area 220-260 sqm."
        ),
    },
}


def _naive_grid_placement(room_specs, rooms, aspect=1.2, circulation_frac=0.0, required_pairs=None):
    """Ablation stand-in for the real zoned squarified treemap: places rooms
    left-to-right in a single row, sized only by area, with NO zoning
    (service/social/private) and no aspect-ratio control. This isolates what
    the layout agent's actual placement logic contributes."""
    positions = {}
    x = 0.0
    for rid, spec in room_specs.items():
        area = spec['area']
        side = area ** 0.5
        w, h = side, area / max(side, 1e-9)
        positions[rid] = {'x': x, 'y': 0.0, 'width': w, 'length': h}
        x += w
    return positions


def _empty_research(node, depth=3):
    """Ablation stand-in for the Research Agent: simulates the RAG store
    being unavailable / retrieval disabled. Returns the same shape
    research_node() normally returns, just empty."""
    return {'similar_spaces': [], 'room_precedents': {}, 'recommendations': []}


def _identity_behaviors(node):
    """Ablation stand-in for physics-based S->Bs analysis: returns the node
    unchanged, so scoring uses the encoder's static initial actual_value
    estimates instead of behaviors computed from real structures/geometry."""
    return node


def build_orchestrator(config_name: str) -> PipelineOrchestrator:
    """Fresh orchestrator per (scenario, config) cell -- no state leaks
    between arms. Applies the one real change this config ablates."""
    orch = PipelineOrchestrator(use_got=True)

    if config_name == "Without RAG (FAISS Retrieval)":
        orch.research.research_node = _empty_research
    elif config_name == "Without Physics-Based Behavior Analysis (S->Bs)":
        # BOTH calculators must be stubbed. RefinementAgent constructs its own
        # BehaviorCalculator, and the convergence loop refines every alternative,
        # so patching only the orchestrator's instance left the refinement path
        # recomputing real physics — the arm silently failed to ablate anything
        # and reported a ~0 % drop for designs that had had their behaviors
        # recomputed from structures after all.
        orch.behavior_calculator.calculate_actual_behaviors = _identity_behaviors
        orch.refiner.behavior_calculator.calculate_actual_behaviors = _identity_behaviors
    elif config_name == "Equal-Weight Scoring (No Tuned MCDA Weights)":
        orch.scoring = ScoringAgent(
            weights={'functional_adequacy': 0.2, 'behavioral_performance': 0.2,
                     'structural_feasibility': 0.2, 'layout_efficiency': 0.2,
                     'sustainability': 0.2},
            rho=1.0,
        )
    elif config_name == "Naive Layout Placement (No Zoning/Treemap)":
        orch.layout_agent._squarified_treemap_placement = _naive_grid_placement
    # "Full Framework (Baseline)", "Without GoT Exploration", and
    # "Without Refinement Agent" need no component patch -- they use the
    # orchestrator's own request-level flags (use_got / enable_convergence_loop).

    return orch


def request_overrides(config_name: str) -> Dict[str, Any]:
    if config_name == "Without GoT Exploration":
        return {"use_got": False}
    if config_name == "Without Refinement Agent":
        return {"enable_convergence_loop": False}
    return {}


CONFIGS = [
    "Full Framework (Baseline)",
    "Without GoT Exploration",
    "Without RAG (FAISS Retrieval)",
    "Without Refinement Agent",
    "Without Physics-Based Behavior Analysis (S->Bs)",
    "Equal-Weight Scoring (No Tuned MCDA Weights)",
    "Naive Layout Placement (No Zoning/Treemap)",
]


async def run_one(scenario_key: str, brief: str, config_name: str) -> Dict[str, Any]:
    orch = build_orchestrator(config_name)
    overrides = request_overrides(config_name)
    request = {
        "project_name": f"ablation_{scenario_key}_{config_name}",
        "requirements": brief,
        "context": {},
        **overrides,
    }
    t0 = time.perf_counter()
    result = await orch.process_design_request(request)
    elapsed = time.perf_counter() - t0

    if not result.get("success"):
        return {
            "config": config_name, "composite": None, "functional": None,
            "behavioral": None, "structural": None, "layout": None,
            "sustainability": None, "top_variant": None, "n_prototypes": 0,
            "time_s": round(elapsed, 2), "error": result.get("error"),
        }

    designs = result.get("designs", [])
    top = designs[0]["scores"] if designs else {}
    return {
        "config": config_name,
        "composite": round(top.get("composite", 0.0), 4),
        "functional": round(top.get("functional_adequacy", 0.0), 4),
        "behavioral": round(top.get("behavioral_performance", 0.0), 4),
        "structural": round(top.get("structural_feasibility", 0.0), 4),
        "layout": round(top.get("layout_efficiency", 0.0), 4),
        "sustainability": round(top.get("sustainability", 0.0), 4),
        # Which named strategy won under this condition -- ablating one
        # component can change which GoT variant ranks #1, and that different
        # winner can carry different structural/layout properties for reasons
        # unrelated to the component being ablated. Recording this lets a
        # reader separate "the component's direct effect" from "a different
        # design won" when interpreting cross-arm deltas.
        "top_variant": designs[0].get("variant_type", "N/A") if designs else None,
        "n_prototypes": len(designs),
        "time_s": round(elapsed, 2),
    }


async def run_scenario(scenario_key: str, meta: Dict[str, str]) -> Dict[str, Any]:
    print(f"\n=== {meta['scenario']} ({meta['complexity']}) ===", flush=True)
    results: List[Dict[str, Any]] = []
    baseline_composite = None
    for config_name in CONFIGS:
        print(f"  running: {config_name} ...", flush=True)
        r = await run_one(scenario_key, meta["brief"], config_name)
        if config_name == "Full Framework (Baseline)":
            baseline_composite = r["composite"]
        if baseline_composite and r["composite"] is not None:
            r["drop_pct"] = round(100 * (baseline_composite - r["composite"]) / baseline_composite, 2)
        else:
            r["drop_pct"] = None
        print(f"    composite={r['composite']} time_s={r['time_s']} "
              f"drop_pct={r['drop_pct']}", flush=True)
        results.append(r)
    return {"scenario": meta["scenario"], "complexity": meta["complexity"], "results": results}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ablation_results")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    study: Dict[str, Any] = {}
    for key, meta in SCENARIOS.items():
        study[key] = await run_scenario(key, meta)

    out_file = out_dir / "ablation_raw_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
