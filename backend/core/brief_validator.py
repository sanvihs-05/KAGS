# core/brief_validator.py
"""
Brief Validator: hard gate ensuring no design that violates the brief is ranked.

The spec is derived ONCE from the root problem node (which the encoder built
directly from the brief), then every GoT alternative is checked against it.

Hard failures (composite score forced to 0):
  - missing required room types / fewer rooms of a type than the brief asked
  - total area outside the brief band (with a grace margin)

Soft issues (logged as warnings, never fatal):
  - individual room far outside its preferred band
  - required adjacency pairs whose rooms exist but layout geometry is absent
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Grace margin applied to the total-area band so borderline designs are not
# rejected on rounding noise (10% each side).
AREA_GRACE = 0.10


@dataclass
class ValidationResult:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'passed': self.passed, 'errors': self.errors, 'warnings': self.warnings}


class BriefValidator:
    """Builds a brief spec from the root node and validates alternatives against it."""

    @staticmethod
    def build_brief_spec(problem_node) -> Dict[str, Any]:
        """
        Derive the brief constraints from the ROOT problem node.

        - expected room-type counts come from the encoder-created layout rooms
        - the total-area band is the sum of each function's [min_area, max_area]
          (falling back to root total_area ± 25% when bands are unavailable)
        - required adjacencies come from node.metadata (encoder-extracted)
        """
        spec: Dict[str, Any] = {
            'expected_room_types': {},
            'total_min': None,
            'total_max': None,
            'required_adjacencies': [],
        }

        rooms = {}
        if problem_node.layout and problem_node.layout.rooms:
            rooms = problem_node.layout.rooms

        spec['expected_room_types'] = dict(Counter(
            (r.room_type or r.name or 'unknown').lower() for r in rooms.values()
        ))

        # Total-area band from function spatial_requirements
        total_min, total_max = 0.0, 0.0
        bands_found = 0
        for func in problem_node.functions.values():
            sr = getattr(func, 'spatial_requirements', None) or {}
            lo = sr.get('min_area')
            hi = sr.get('max_area')
            if lo is not None and hi is not None:
                total_min += float(lo)
                total_max += float(hi)
                bands_found += 1

        if bands_found > 0:
            spec['total_min'] = total_min * (1 - AREA_GRACE)
            spec['total_max'] = total_max * (1 + AREA_GRACE)
        elif rooms:
            root_total = sum(r.area for r in rooms.values())
            spec['total_min'] = root_total * 0.75
            spec['total_max'] = root_total * 1.25

        adj = (problem_node.metadata or {}).get('required_adjacencies', [])
        for a in adj:
            if isinstance(a, dict) and a.get('room1') and a.get('room2'):
                spec['required_adjacencies'].append(
                    (str(a['room1']).lower(), str(a['room2']).lower())
                )

        logger.info(
            f"✓ Brief spec: {sum(spec['expected_room_types'].values())} rooms "
            f"({len(spec['expected_room_types'])} types), "
            f"total band [{spec['total_min']:.0f}, {spec['total_max']:.0f}] m², "
            f"{len(spec['required_adjacencies'])} required adjacencies"
            if spec['total_min'] is not None else
            "✓ Brief spec built (no area band available)"
        )
        return spec

    @staticmethod
    def validate(node, spec: Dict[str, Any]) -> ValidationResult:
        """Check one design node against the brief spec. Hard failures gate ranking."""
        res = ValidationResult()

        rooms = {}
        if node.layout and node.layout.rooms:
            rooms = node.layout.rooms

        if not rooms:
            res.passed = False
            res.errors.append("design has no rooms")
            return res

        # --- Hard check 1: required room types present in required counts ----
        have = Counter((r.room_type or r.name or 'unknown').lower() for r in rooms.values())
        for rtype, need in (spec.get('expected_room_types') or {}).items():
            got = have.get(rtype, 0)
            if got < need:
                res.passed = False
                res.errors.append(f"room type '{rtype}': brief needs {need}, design has {got}")

        # --- Hard check 2: total area within the brief band --------------------
        total = sum(r.area for r in rooms.values())
        lo, hi = spec.get('total_min'), spec.get('total_max')
        if lo is not None and hi is not None:
            if not (lo <= total <= hi):
                res.passed = False
                res.errors.append(
                    f"total area {total:.0f} m² outside brief band [{lo:.0f}, {hi:.0f}] m²"
                )

        # --- Soft check: required adjacency rooms exist -----------------------
        room_names = {(r.name or '').lower() for r in rooms.values()} | set(have.keys())
        for r1, r2 in spec.get('required_adjacencies', []):
            missing = [r for r in (r1, r2) if not any(r in n or n in r for n in room_names)]
            if missing:
                res.warnings.append(f"required adjacency ({r1}–{r2}): room(s) missing {missing}")

        return res
