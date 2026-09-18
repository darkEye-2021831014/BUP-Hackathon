"""
Local judge / replay test for GridWise.

This script:
  1. Starts the FastAPI app in-process via httpx ASGITransport (no
     network needed).
  2. POSTs each of the 10 sample cases to /optimize-energy.
  3. Verifies directive_interpretation semantics (same directive_type,
     same hours set, factor within tolerance, etc.).
  4. Replays the returned hourly_plan against:
        - the interpreted directives (solar cap, charge/discharge
          windows, grid cap, min reserve)
        - the normal energy/battery rules (balance, bounds, rate
          limits, end-of-day neutrality)
        - the response's own totals (total_grid_kwh, total_cost_bdt,
          peak_grid_kwh must match recomputation from hourly_plan)
  5. Optionally runs the FULL pipeline through the LLM interpreter
     (which falls back to the deterministic regex interpreter when
     no keys are configured) and ensures the result is still valid.

Run: .venv/bin/python tests/test_judge.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.main import app  # noqa: E402
from app.replay import JUDGE_TOL, replay_check


# ---------------------------------------------------------------------------
# Directive semantics comparison
# ---------------------------------------------------------------------------


def _directive_signature(d: Dict[str, Any]) -> Tuple[str, frozenset, float]:
    """Return (type, hours_frozenset, numeric_value_or_-1) for comparison."""
    dtype = d.get("directive_type")
    adj = d.get("structured_adjustment") or {}
    hours = adj.get("hours", [])
    if not isinstance(hours, list):
        hours = []
    hset = frozenset(int(h) for h in hours)
    val = -1.0
    if dtype == "solar_reduction":
        val = float(adj.get("factor", 1.0))
    elif dtype == "minimum_battery_reserve":
        val = float(adj.get("minimum_energy_kwh", 0.0))
    elif dtype == "max_grid_window":
        val = float(adj.get("max_grid_kwh", 0.0))
    return (dtype, hset, val)


def _check_directives(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
) -> List[str]:
    errs: List[str] = []
    if len(expected) != len(actual):
        errs.append(
            f"directive_interpretation length {len(actual)} != expected {len(expected)}"
        )
        return errs
    for i, (e, a) in enumerate(zip(expected, actual)):
        if a.get("note_index") != e.get("note_index"):
            errs.append(
                f"note {i}: note_index {a.get('note_index')} != expected {e.get('note_index')}"
            )
        if a.get("applies") != e.get("applies"):
            errs.append(
                f"note {i}: applies {a.get('applies')} != expected {e.get('applies')}"
            )
        es = _directive_signature(e)
        asg = _directive_signature(a)
        if es[0] != asg[0]:
            errs.append(
                f"note {i}: directive_type {asg[0]} != expected {es[0]}"
            )
        if es[0] != "no_op":
            if es[1] != asg[1]:
                errs.append(
                    f"note {i}: hours {sorted(asg[1])} != expected {sorted(es[1])}"
                )
            if es[0] in ("solar_reduction", "maximum_battery_reserve", "max_grid_window"):
                if not math.isclose(es[2], asg[2], abs_tol=JUDGE_TOL):
                    errs.append(
                        f"note {i}: numeric value {asg[2]} != expected {es[2]}"
                    )
        else:
            # no_op must have null structured_adjustment
            if a.get("structured_adjustment") is not None:
                errs.append(
                    f"note {i}: no_op has non-null structured_adjustment"
                )
    return errs


# ---------------------------------------------------------------------------
# Replay validator (delegated to the shared app.replay module so the
# API's final_validator and this judge use the exact same logic).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)["cases"]


def run(use_bypass: bool) -> int:
    import asyncio

    cases_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    )
    cases = _load_cases(cases_path)

    return asyncio.run(_run_all(cases))


async def _run_all(cases: List[Dict[str, Any]]) -> int:
    n_ok = 0
    n_fail = 0
    summary: List[str] = []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
    ) as client:
        for case in cases:
            cid = case["id"]
            inp = case["input"]
            exp = case["expected_output"]

            payload = dict(inp)

            try:
                resp = await client.post(
                    "/optimize-energy", json=payload, timeout=60.0
                )
            except Exception as e:
                print(f"{cid}  EXC {type(e).__name__}: {e}")
                n_fail += 1
                summary.append(f"{cid}: exception {e}")
                continue
            if resp.status_code != 200:
                print(f"{cid}  HTTP {resp.status_code}  {resp.text[:200]}")
                n_fail += 1
                summary.append(f"{cid}: HTTP {resp.status_code}")
                continue

            body = resp.json()

            # 1) directive semantics
            d_errs = _check_directives(
                exp["directive_interpretation"],
                body["directive_interpretation"],
            )

            # 2) replay (shared with the API's final_validator)
            r_errs = replay_check(
                payload=inp,
                directive_interpretation=body["directive_interpretation"],
                plan=body["hourly_plan"],
                totals={
                    "total_grid_kwh": body["total_grid_kwh"],
                    "total_cost_bdt": body["total_cost_bdt"],
                    "peak_grid_kwh": body["peak_grid_kwh"],
                },
            )

            cost = body["total_cost_bdt"]
            if not d_errs and not r_errs:
                print(
                    f"{cid}  OK  cost={cost}  grid={body['total_grid_kwh']}  "
                    f"peak={body['peak_grid_kwh']}"
                )
                n_ok += 1
            else:
                print(f"{cid}  FAIL")
                for e in d_errs:
                    print(f"   dir: {e}")
                for e in r_errs:
                    print(f"   rep: {e}")
                n_fail += 1
                summary.append(f"{cid}: {len(d_errs)} dir, {len(r_errs)} replay")

    print("\n" + "=" * 60)
    print(f"PASS: {n_ok}/{len(cases)}  FAIL: {n_fail}/{len(cases)}")
    if summary:
        print("Failures:")
        for s in summary:
            print(f"  - {s}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bypass",
        action="store_true",
        help="Reserved: bypass the LLM (already done automatically when no keys)",
    )
    args = ap.parse_args()
    sys.exit(run(use_bypass=args.bypass))
