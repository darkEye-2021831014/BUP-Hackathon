"""
Deterministic guardrail validator for GridWise.

The LLM output is UNTRUSTED. This module is purely deterministic,
dependency-free Python that enforces the schema and semantic rules
in Section 2 step 3 and Section 4.1 of the problem spec.

If anything is invalid, the offending note(s) are downgraded to no_op
rather than crashing the whole request. This keeps the service robust
when the LLM hallucinates or returns malformed JSON.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .schemas import BatteryConfig

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return False


def _normalize_hours(hours: Any) -> List[int] | None:
    """Return sorted unique ints in [0,23] from list, or None if invalid."""
    if not isinstance(hours, list) or not hours:
        return None
    out = []
    seen = set()
    for h in hours:
        if isinstance(h, bool):
            return None
        if isinstance(h, int) and not isinstance(h, bool):
            v = h
        elif isinstance(h, float) and h.is_integer():
            v = int(h)
        else:
            return None
        if v < 0 or v > 23:
            return None
        if v in seen:
            return None
        seen.add(v)
        out.append(v)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Per-directive validators
# ---------------------------------------------------------------------------


def _validate_solar_reduction(adj: Any) -> Tuple[bool, Dict[str, Any] | None]:
    if not isinstance(adj, dict):
        return False, None
    hours = _normalize_hours(adj.get("hours"))
    if hours is None:
        return False, None
    factor = adj.get("factor")
    if not _is_finite_number(factor):
        return False, None
    factor = float(factor)
    if factor < 0.0 or factor > 1.0:
        return False, None
    return True, {"hours": hours, "factor": factor}


def _validate_minimum_battery_reserve(
    adj: Any, battery: BatteryConfig
) -> Tuple[bool, Dict[str, Any] | None]:
    if not isinstance(adj, dict):
        return False, None
    hours = _normalize_hours(adj.get("hours"))
    if hours is None:
        return False, None
    mev = adj.get("minimum_energy_kwh")
    if not _is_finite_number(mev):
        return False, None
    mev = float(mev)
    if mev < 0.0:
        return False, None
    # Clamp to capacity. Above capacity the directive is unsatisfiable in
    # those hours, so drop to no_op rather than produce an infeasible LP.
    if mev > battery.capacity_kwh:
        return False, None
    return True, {"hours": hours, "minimum_energy_kwh": mev}


def _validate_hours_only(adj: Any) -> Tuple[bool, Dict[str, Any] | None]:
    if not isinstance(adj, dict):
        return False, None
    hours = _normalize_hours(adj.get("hours"))
    if hours is None:
        return False, None
    return True, {"hours": hours}


def _validate_max_grid_window(adj: Any) -> Tuple[bool, Dict[str, Any] | None]:
    if not isinstance(adj, dict):
        return False, None
    hours = _normalize_hours(adj.get("hours"))
    if hours is None:
        return False, None
    mg = adj.get("max_grid_kwh")
    if not _is_finite_number(mg):
        return False, None
    mg = float(mg)
    if mg < 0.0:
        return False, None
    return True, {"hours": hours, "max_grid_kwh": mg}


# ---------------------------------------------------------------------------
# Top-level note validator
# ---------------------------------------------------------------------------


def _no_op(i: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": i,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _guardrail_no_op_for_note(i: int, item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return _no_op(
        i,
        f"Guardrail downgraded invalid note to no_op: {reason}",
    )


def validate_and_normalize_directives(
    raw_directives: List[Dict[str, Any]],
    battery: BatteryConfig,
    expected_n: int,
) -> List[Dict[str, Any]]:
    """Take LLM output (or fallback) and emit one valid entry per note.

    Drops any extras, fills missing indices with no_op, and force-rewrites
    applies/structured_adjustment per the spec:
      - non-no_op type -> applies must be true
      - no_op         -> applies false, structured_adjustment null
      - if a note's directive_type / structured_adjustment fails validation,
        it is downgraded to no_op (instead of failing the whole request)
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    for item in raw_directives:
        if not isinstance(item, dict):
            continue
        idx = item.get("note_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        if idx < 0 or idx >= expected_n:
            continue
        # First-write-wins per index.
        if idx in by_idx:
            continue
        by_idx[idx] = item

    result: List[Dict[str, Any]] = []
    for i in range(expected_n):
        item = by_idx.get(i)
        if item is None:
            result.append(_no_op(i, "Note was missing from interpreter output."))
            continue

        dtype = item.get("directive_type")
        adj = item.get("structured_adjustment")
        applies = item.get("applies")
        explanation = item.get("explanation")
        if not isinstance(explanation, str):
            explanation = ""

        # Unknown type -> no_op
        if not isinstance(dtype, str) or dtype not in ALLOWED_TYPES:
            result.append(_guardrail_no_op_for_note(i, item, "unknown directive_type"))
            continue

        # no_op handling
        if dtype == "no_op":
            result.append(
                {
                    "note_index": i,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": explanation or "Note does not affect today's schedule.",
                }
            )
            continue

        # Non-no_op requires applies=true
        if applies is not True:
            # Honor applies=true OR missing -> force true and continue
            # If applies explicitly false but type says something else -> no_op
            if applies is False:
                result.append(
                    _guardrail_no_op_for_note(
                        i,
                        item,
                        f"applies=false with non-no_op directive_type {dtype}",
                    )
                )
                continue

        # Per-type structured_adjustment validation
        valid = False
        normalized_adj: Dict[str, Any] | None = None
        if dtype == "solar_reduction":
            valid, normalized_adj = _validate_solar_reduction(adj)
        elif dtype == "minimum_battery_reserve":
            valid, normalized_adj = _validate_minimum_battery_reserve(adj, battery)
        elif dtype == "no_charge_window":
            valid, normalized_adj = _validate_hours_only(adj)
        elif dtype == "no_discharge_window":
            valid, normalized_adj = _validate_hours_only(adj)
        elif dtype == "max_grid_window":
            valid, normalized_adj = _validate_max_grid_window(adj)

        if not valid:
            result.append(
                _guardrail_no_op_for_note(
                    i,
                    item,
                    f"invalid structured_adjustment for {dtype}",
                )
            )
            continue

        result.append(
            {
                "note_index": i,
                "applies": True,
                "directive_type": dtype,
                "structured_adjustment": normalized_adj,
                "explanation": explanation or f"Applied {dtype} directive.",
            }
        )

    return result
