"""
Smoke test: bypass the LLM, feed the sample cases' expected directive
interpretation directly through the guardrail and optimizer, and
verify the LP can produce a feasible plan for each one.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.guardrails import validate_and_normalize_directives
from app.optimizer import optimize
from app.schemas import BatteryConfig, HourData, OptimizeRequest


def _make_battery(d):
    return BatteryConfig(**d)


def _make_hours(hours_raw):
    return [HourData(**h) for h in hours_raw]


def main():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    )
    with open(path) as f:
        data = json.load(f)

    failures = []
    for case in data["cases"]:
        cid = case["id"]
        inp = case["input"]
        exp = case["expected_output"]

        battery = _make_battery(inp["battery"])
        hours = _make_hours(inp["hours"])

        # Bypass LLM; use expected interpretation as input to guardrail
        raw_directives = exp["directive_interpretation"]
        validated = validate_and_normalize_directives(
            raw_directives=raw_directives,
            battery=battery,
            expected_n=len(inp["operator_notes"]),
        )

        try:
            result = optimize(hours, battery, validated)
            ok = "OK"
            err = ""
        except Exception as e:
            result = None
            ok = "FAIL"
            err = f"{type(e).__name__}: {e}"

        if result is not None:
            print(
                f"{cid} {ok}  cost={result['total_cost_bdt']}  "
                f"grid={result['total_grid_kwh']}  peak={result['peak_grid_kwh']}"
            )
        else:
            print(f"{cid} {ok}  {err}")
            failures.append((cid, err))

    print(f"\n{'=' * 60}")
    print(f"Smoke failures: {len(failures)}")
    for cid, err in failures:
        print(f"  {cid}: {err}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
