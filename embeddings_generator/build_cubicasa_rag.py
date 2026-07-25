"""
Build a room-level RAG store from the CubiCasa5K floor-plan dataset.

Why this exists
---------------
The legacy `enhanced_multimodal_rag_store` indexed 215k OCR annotation *tokens*
('VH', 'sink', 'UNDEFINED'), all under a single fake `plan_id='model'`, with no
structured area. The Research Agent's retrieval therefore returned nothing
useful and area-reconciliation was a permanent no-op.

This script indexes the RIGHT unit — one record per ROOM — parsed straight from
each plan's `model.svg`:
    { plan_id, room_type, area_m2, neighbors[], text }
Each room's `text` is a short natural-language description that the same
SentenceTransformer (all-MiniLM-L6-v2, 384-dim) embeds, and `area_m2` is a real
measurement (shoelace on the room polygon at CubiCasa's 100-units/metre scale).

Output (drop-in compatible with backend/utils/embedding_loader.py):
    <out_dir>/composite_embeddings.npy      float32  (N_rooms, 384)
    <out_dir>/consolidated_metadata.json    list[dict]  row-aligned with the .npy

Usage:
    python embeddings_generator/build_cubicasa_rag.py \
        --cubicasa cubicasa5k --out cubicasa_rag_store
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_cubicasa_rag")

# CubiCasa model.svg coordinate scale: ~100 units per metre (validated: sample
# bedrooms 12 m2, living rooms 18-31 m2, kitchens ~13 m2). area_m2 = shoelace / 100^2.
UNITS_PER_M = 100.0
AREA_SCALE = 1.0 / (UNITS_PER_M ** 2)

# Adjacency: two room polygons are neighbours if their boundaries lie within one
# wall-thickness of each other (~0.25 m => 25 model units).
ADJACENCY_MAX_DIST = 25.0

# Ignore rooms below this measured area (annotation slivers / fixtures).
MIN_ROOM_AREA_M2 = 1.5
# Ignore absurd areas (parse/scale outliers).
MAX_ROOM_AREA_M2 = 120.0

# Map the CubiCasa "Space <Type> <Modifier>" label to a canonical room_type that
# matches the pipeline's vocabulary. The primary token drives the mapping.
_TYPE_MAP = {
    "livingroom": "living_room",
    "bedroom": "bedroom",
    "bath": "bathroom",
    "bathroom": "bathroom",
    "toilet": "toilet",
    "wash": "bathroom",
    "sauna": "sauna",
    "kitchen": "kitchen",
    "eatingkitchen": "kitchen",
    "dining": "dining",
    "entry": "entry",
    "draughtlobby": "entry",
    "hall": "hallway",
    "hallway": "hallway",
    "closet": "closet",
    "walkin": "closet",
    "dressingroom": "closet",
    "storage": "storage",
    "garage": "garage",
    "carport": "garage",
    "office": "office",
    "library": "office",
    "den": "office",
    "outdoor": "balcony",
    "balcony": "balcony",
    "technicalroom": "utility",
    "utility": "utility",
    "recreationroom": "recreation",
    "room": "room",
}
# Types we never want as precedents (no useful program meaning).
_SKIP_TYPES = {"undefined", "userdefined", "below150cm", "elevated", "alcove"}

# Space elements may carry attributes (fill=, stroke=, style=) between the
# class and '>', and the room's floor polygon may not be the immediately
# adjacent node — so we locate each Space label by position and take the FIRST
# <polygon> that follows it (before the next Space label). A single strict
# "class then immediate polygon" regex silently dropped ~34% of rooms.
_SPACE_LABEL_RE = re.compile(r'class="Space ([^"]+)"')
_POLYGON_RE = re.compile(r'<polygon points="([^"]+)"')


def _canonical_type(raw_class: str) -> Optional[str]:
    """'Bath Shower' -> 'bathroom'; 'Kitchen Kitchenette' -> 'kitchen'."""
    tokens = raw_class.strip().split()
    if not tokens:
        return None
    primary = tokens[0].lower()
    if primary in _SKIP_TYPES:
        return None
    return _TYPE_MAP.get(primary, primary)


def _parse_points(points: str) -> List[Tuple[float, float]]:
    pts = []
    for p in points.split():
        if "," in p:
            try:
                x, y = p.split(",")
                pts.append((float(x), float(y)))
            except ValueError:
                continue
    return pts


def _shoelace_area(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _bbox(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _polys_adjacent(a_pts, b_pts) -> bool:
    """Cheap adjacency test: bounding boxes overlap/touch within a wall width.
    (Shapely would be exact but bbox-touch is a good, fast proxy for whether two
    rooms share a wall, and CubiCasa rooms are near-rectangular.)"""
    ax0, ay0, ax1, ay1 = _bbox(a_pts)
    bx0, by0, bx1, by1 = _bbox(b_pts)
    # horizontal gap and vertical gap between the two bboxes (negative = overlap)
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    # adjacent if they nearly touch on at least one axis while overlapping the other
    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_y = min(ay1, by1) - max(ay0, by0)
    if dx <= ADJACENCY_MAX_DIST and overlap_y > ADJACENCY_MAX_DIST:
        return True
    if dy <= ADJACENCY_MAX_DIST and overlap_x > ADJACENCY_MAX_DIST:
        return True
    return False


def parse_plan(svg_path: Path, plan_id: str) -> List[Dict]:
    """Parse one model.svg into a list of room records (no embeddings yet)."""
    try:
        svg = svg_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug("read failed %s: %s", svg_path, e)
        return []

    labels = list(_SPACE_LABEL_RE.finditer(svg))
    rooms = []
    for idx, m in enumerate(labels):
        rtype = _canonical_type(m.group(1))
        if rtype is None:
            continue
        # search window: from this Space label to the next one
        end = labels[idx + 1].start() if idx + 1 < len(labels) else len(svg)
        poly_m = _POLYGON_RE.search(svg, m.end(), end)
        if not poly_m:
            continue
        pts = _parse_points(poly_m.group(1))
        area = _shoelace_area(pts) * AREA_SCALE
        if not (MIN_ROOM_AREA_M2 <= area <= MAX_ROOM_AREA_M2):
            continue
        rooms.append({"room_type": rtype, "area_m2": round(area, 1), "_pts": pts})

    # adjacency within the plan
    for i, r in enumerate(rooms):
        neigh = []
        for j, other in enumerate(rooms):
            if i != j and _polys_adjacent(r["_pts"], other["_pts"]):
                neigh.append(other["room_type"])
        r["neighbors"] = sorted(set(neigh))

    # finalise records (drop the raw points; add plan_id + embedding text)
    out = []
    for r in rooms:
        neighbors = r["neighbors"]
        neigh_txt = (", adjacent to " + ", ".join(neighbors)) if neighbors else ""
        text = f"{r['room_type'].replace('_', ' ')} of {r['area_m2']:.0f} square metres{neigh_txt}"
        out.append({
            "plan_id": plan_id,
            "room_type": r["room_type"],
            "area": r["area_m2"],          # <-- structured area the reconciler reads
            "area_m2": r["area_m2"],
            "neighbors": neighbors,
            "text": text,
            "translated": text,
            "function": f"provide_{r['room_type']}",
        })
    return out


def build(cubicasa_dir: Path, out_dir: Path, variants: List[str], limit: Optional[int]) -> None:
    from sentence_transformers import SentenceTransformer

    # 1) Gather all plan folders across the requested variants.
    plan_dirs: List[Tuple[str, Path]] = []
    for variant in variants:
        vdir = cubicasa_dir / variant
        if not vdir.is_dir():
            logger.warning("variant folder missing, skipping: %s", vdir)
            continue
        for pdir in sorted(vdir.iterdir()):
            svg = pdir / "model.svg"
            if svg.is_file():
                plan_dirs.append((f"{variant}/{pdir.name}", svg))
    if limit:
        plan_dirs = plan_dirs[:limit]
    logger.info("Found %d plans across variants %s", len(plan_dirs), variants)

    # 2) Parse every plan into room records.
    records: List[Dict] = []
    n_empty = 0
    for k, (plan_id, svg) in enumerate(plan_dirs, 1):
        rooms = parse_plan(svg, plan_id)
        if not rooms:
            n_empty += 1
        records.extend(rooms)
        if k % 500 == 0:
            logger.info("  parsed %d/%d plans, %d rooms so far", k, len(plan_dirs), len(records))
    logger.info("Parsed %d rooms from %d plans (%d plans yielded no usable rooms)",
                len(records), len(plan_dirs), n_empty)
    if not records:
        raise SystemExit("No rooms parsed — check the dataset path / SVG schema.")

    # 3) Embed each room's text with the SAME model the pipeline queries with.
    logger.info("Loading SentenceTransformer all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["text"] for r in records]
    logger.info("Encoding %d room descriptions ...", len(texts))
    emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                       convert_to_numpy=True).astype("float32")
    logger.info("Embeddings shape: %s", emb.shape)

    # 4) Write the store (composite_embeddings.npy + consolidated_metadata.json).
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "composite_embeddings.npy", emb)
    # strip internal keys before persisting metadata
    meta = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    for gi, r in enumerate(meta):
        r["global_idx"] = gi
    with open(out_dir / "consolidated_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # 4b) Empirical adjacency prior (precedent knowledge for the Research Agent).
    prior = compute_adjacency_prior(records)
    with open(out_dir / "adjacency_prior.json", "w", encoding="utf-8") as f:
        json.dump(prior, f, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Adjacency prior: %d room-type pairs -> adjacency_prior.json", len(prior))

    # 5) Report — so the area scale can be sanity-checked empirically.
    _report(records, emb, out_dir)


def compute_adjacency_prior(records: List[Dict], min_support: int = 50) -> Dict:
    """Aggregate the per-room `neighbors` into an empirical adjacency prior:
    P(type_a adjacent type_b | a plan contains both types), over all plans.

    This is the architectural knowledge precedents carry that a brief often
    leaves unstated (kitchen↔dining, bedroom↔bathroom, sauna↔bathroom…). The
    Research Agent uses it to fill adjacency gaps the brief didn't specify.
    Only pairs with >= min_support co-occurring plans are kept (drop noise).
    """
    from collections import defaultdict, Counter
    plans = defaultdict(list)
    for r in records:
        plans[r["plan_id"]].append((r["room_type"], r.get("neighbors") or []))

    pair_adjacent = Counter()   # (a,b) -> #plans where an a and b share a wall
    pair_present = Counter()    # (a,b) -> #plans containing both types
    for rooms in plans.values():
        present = sorted({rt for rt, _ in rooms})
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pair_present[(a, b)] += 1
        seen = set()
        for rt, neigh in rooms:
            for nb in neigh:
                if rt != nb:
                    seen.add(tuple(sorted((rt, nb))))
        for pr in seen:
            pair_adjacent[pr] += 1

    prior = {}
    for pair, both in pair_present.items():
        if both >= min_support:
            prior[f"{pair[0]}|{pair[1]}"] = round(pair_adjacent.get(pair, 0) / both, 3)
    return prior


def _report(records: List[Dict], emb: np.ndarray, out_dir: Path) -> None:
    rt = Counter(r["room_type"] for r in records)
    logger.info("=" * 64)
    logger.info("STORE SUMMARY  ->  %s", out_dir)
    logger.info("  rooms: %d   plans: %d   embedding dim: %d",
                len(records), len({r["plan_id"] for r in records}), emb.shape[1])
    logger.info("  store size (npy): %.1f MB", (out_dir / "composite_embeddings.npy").stat().st_size / 1e6)
    logger.info("  room-type distribution (top 15):")
    for t, c in rt.most_common(15):
        areas = [r["area_m2"] for r in records if r["room_type"] == t]
        logger.info("    %-14s n=%-6d median area=%5.1f m2  (p10=%.1f p90=%.1f)",
                    t, c, statistics.median(areas),
                    np.percentile(areas, 10), np.percentile(areas, 90))
    logger.info("=" * 64)
    logger.info("Sanity check: bedroom/kitchen/living_room medians should look")
    logger.info("like real rooms (~10-14 / ~9-13 / ~16-25 m2). If they don't,")
    logger.info("the UNITS_PER_M scale is off.")


def main():
    ap = argparse.ArgumentParser(description="Build a room-level CubiCasa5K RAG store.")
    ap.add_argument("--cubicasa", default="cubicasa5k", help="path to cubicasa5k root")
    ap.add_argument("--out", default="cubicasa_rag_store", help="output store directory")
    ap.add_argument("--variants", nargs="+",
                    default=["colorful", "high_quality", "high_quality_architectural"],
                    help="which quality subsets to include (default: all three)")
    ap.add_argument("--limit", type=int, default=None, help="cap plans (for a quick test run)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cubicasa_dir = (root / args.cubicasa) if not Path(args.cubicasa).is_absolute() else Path(args.cubicasa)
    out_dir = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    build(cubicasa_dir, out_dir, args.variants, args.limit)


if __name__ == "__main__":
    main()
