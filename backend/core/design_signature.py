"""Design signature: a compact fingerprint of the parameters that actually
change a design's geometry or physics.

Two nodes with the same signature will produce the same treemap footprint,
the same behavior calculations, and therefore the same scores — they are the
same design, whatever their metadata labels say. The orchestrator uses this
to collapse clone leaves before ranking and to guarantee the final top-k
shows DISTINCT designs instead of three copies of the winner.
"""
from collections import Counter
from typing import Tuple


def design_signature(node) -> Tuple:
    """Fingerprint the physics/geometry-driving parameters of an FBSL node."""
    md = getattr(node, 'metadata', None) or {}

    aspect = round(float(md.get('layout_aspect', 1.2) or 1.2), 2)
    vent = md.get('ventilation_strategy', 'mechanical')

    ratios = []
    materials = set()
    structure_kinds = Counter()
    for s in getattr(node, 'structures', {}).values():
        dims = getattr(s, 'dimensions', None) or {}
        if 'window_ratio' in dims:
            try:
                ratios.append(float(dims['window_ratio']))
            except (TypeError, ValueError):
                pass
        mat = getattr(s, 'material_type', None)
        if mat:
            materials.add(str(mat))
        st = getattr(s, 'structure_type', None)
        structure_kinds[getattr(st, 'value', str(st))] += 1
    mean_wr = round(sum(ratios) / len(ratios), 2) if ratios else 0.0

    rooms = {}
    layout = getattr(node, 'layout', None)
    if layout is not None and getattr(layout, 'rooms', None):
        rooms = layout.rooms
    room_counts = tuple(sorted(
        Counter(getattr(r, 'room_type', 'space') for r in rooms.values()).items()
    ))
    # 10 m² buckets: refinement nudging areas a little should not count as a
    # "different design", but dropping a bedroom or 20 m² should.
    area_bucket = int(round(sum(float(getattr(r, 'area', 0) or 0) for r in rooms.values()) / 10.0))

    return (
        aspect,
        vent,
        mean_wr,
        tuple(sorted(materials)),
        tuple(sorted(structure_kinds.items())),
        room_counts,
        area_bucket,
    )


def variant_family(node) -> str:
    """Level-1 strategy family: first tag of the variant lineage."""
    md = getattr(node, 'metadata', None) or {}
    return (md.get('variant_type') or 'base').split('+')[0]


def footprint_class(node) -> str:
    """Coarse footprint shape from layout_aspect: square / moderate / linear."""
    md = getattr(node, 'metadata', None) or {}
    try:
        asp = float(md.get('layout_aspect', 1.2) or 1.2)
    except (TypeError, ValueError):
        asp = 1.2
    if asp < 1.15:
        return 'square'
    return 'moderate' if asp < 1.7 else 'linear'


def diversity_greedy_order(items, key=None):
    """Reorder best-first `items` so each pick maximizes novelty (new strategy
    family, new footprint class, new signature), breaking ties by the input
    (score) order. Position 0 is always the input's best item; positions 2-3
    become the best DIFFERENT designs instead of clones of the winner.
    """
    remaining = list(items)
    ordered = []
    seen_fam, seen_shape, seen_sigs = set(), set(), set()

    def _node(item):
        return key(item) if key else item

    while remaining:
        def _novelty(item):
            n = _node(item)
            return ((variant_family(n) not in seen_fam)
                    + (footprint_class(n) not in seen_shape)
                    + (design_signature(n) not in seen_sigs))
        best = max(_novelty(i) for i in remaining)
        pick = next(i for i in remaining if _novelty(i) == best)
        remaining.remove(pick)
        n = _node(pick)
        seen_fam.add(variant_family(n))
        seen_shape.add(footprint_class(n))
        seen_sigs.add(design_signature(n))
        ordered.append(pick)
    return ordered


def dedupe_by_signature(nodes, key=None):
    """Keep the first node per signature, preserving input order.

    `nodes` should already be sorted best-first so the survivor of each
    clone-group is its best representative.
    """
    seen = set()
    unique = []
    for n in nodes:
        node = key(n) if key else n
        sig = design_signature(node)
        if sig not in seen:
            seen.add(sig)
            unique.append(n)
    return unique
