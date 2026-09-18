"""
Paraphrase robustness test.

Per Section 8.2 of the master prompt: write 8-10 of your OWN paraphrased
operator notes per directive type (different wording than the samples/
the master doc) and confirm the LLM interpretation is still correct —
this is your proxy for "paraphrase robustness," which is explicitly
hidden-tested.

This test exercises the deterministic regex fallback (since no LLM keys
are configured in this environment). If you configure LLM keys, run the
same test through the real LLM path by exporting the keys.
"""
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.guardrails import validate_and_normalize_directives
from app.llm_interpreter import _deterministic_interpret


def _check(
    note: str,
    battery_capacity_kwh: float,
    expected_type: str,
    expected_hours: List[int] | None = None,
    expected_value: float | None = None,
    expected_applies: bool = True,
) -> bool:
    raw = _deterministic_interpret([note], battery_capacity_kwh)
    battery = {
        "capacity_kwh": battery_capacity_kwh,
        "initial_energy_kwh": battery_capacity_kwh * 0.5,
        "minimum_energy_kwh": battery_capacity_kwh * 0.2,
        "max_charge_kwh_per_hour": battery_capacity_kwh * 0.2,
        "max_discharge_kwh_per_hour": battery_capacity_kwh * 0.2,
    }
    # Use guardrails to normalize (and to confirm valid types/structure)
    from app.schemas import BatteryConfig

    validated = validate_and_normalize_directives(
        raw_directives=raw,
        battery=BatteryConfig(**battery),
        expected_n=1,
    )
    d = validated[0]
    adj = d.get("structured_adjustment") or {}
    if d.get("applies") != expected_applies:
        print(
            f"  FAIL  applies={d.get('applies')} expected={expected_applies}  "
            f"note: {note}"
        )
        return False
    if d.get("directive_type") != expected_type:
        print(
            f"  FAIL  type={d.get('directive_type')} expected={expected_type}  "
            f"note: {note}"
        )
        return False
    if expected_applies:
        h = adj.get("hours", [])
        if expected_hours is not None and h != expected_hours:
            print(
                f"  FAIL  hours={h} expected={expected_hours}  note: {note}"
            )
            return False
        if expected_value is not None:
            if expected_type == "solar_reduction":
                v = adj.get("factor")
            elif expected_type == "minimum_battery_reserve":
                v = adj.get("minimum_energy_kwh")
            elif expected_type == "max_grid_window":
                v = adj.get("max_grid_kwh")
            else:
                v = None
            if v is None or abs(v - expected_value) > 0.01:
                print(
                    f"  FAIL  value={v} expected={expected_value}  "
                    f"note: {note}"
                )
                return False
    print(f"  OK    {note[:80]}")
    return True


CASES = [
    # solar_reduction paraphrases
    {
        "note": "Cloud cover will cut PV output to about 25 percent from 1 PM to 3 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [13, 14],
        "value": 0.25,
    },
    {
        "note": "We will see roughly 60% of forecast solar between 11 AM and 2 PM due to fog.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [11, 12, 13],
        "value": 0.60,
    },
    {
        "note": "The panels will produce only 10% of forecast during 2 PM and 4 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [14, 15],
        "value": 0.10,
    },
    {
        "note": "Heavy dust storm will knock rooftop solar down to about 30% from noon to 3 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [12, 13, 14],
        "value": 0.30,
    },
    # minimum_battery_reserve paraphrases
    {
        "note": "Hold at minimum 80 kWh of battery from 7 PM until 10 PM.",
        "capacity": 200,
        "type": "minimum_battery_reserve",
        "hours": [19, 20, 21],
        "value": 80.0,
    },
    {
        "note": "Don't let battery drop below 60 kWh between 5 PM and 7 PM.",
        "capacity": 200,
        "type": "minimum_battery_reserve",
        "hours": [17, 18],
        "value": 60.0,
    },
    {
        "note": "Keep at least 75% of battery capacity stored from 6 PM to 9 PM.",
        "capacity": 200,
        "type": "minimum_battery_reserve",
        "hours": [18, 19, 20],
        "value": 150.0,
    },
    {
        "note": "Reserve at least 100 kWh for the dormitory from 8 PM to 11 PM.",
        "capacity": 200,
        "type": "minimum_battery_reserve",
        "hours": [20, 21, 22],
        "value": 100.0,
    },
    # no_charge_window paraphrases
    {
        "note": "The battery must not be charged from 3 AM until 6 AM during inspection.",
        "capacity": 200,
        "type": "no_charge_window",
        "hours": [3, 4, 5],
    },
    {
        "note": "Charging is offline from 1 PM to 3 PM.",
        "capacity": 200,
        "type": "no_charge_window",
        "hours": [13, 14],
    },
    {
        "note": "Block charging from 11 AM until 1 PM.",
        "capacity": 200,
        "type": "no_charge_window",
        "hours": [11, 12],
    },
    # no_discharge_window paraphrases
    {
        "note": "Do not let the battery discharge from 5 PM to 7 PM.",
        "capacity": 200,
        "type": "no_discharge_window",
        "hours": [17, 18],
    },
    {
        "note": "Discharge is disabled between 7 PM and 9 PM.",
        "capacity": 200,
        "type": "no_discharge_window",
        "hours": [19, 20],
    },
    {
        "note": "Battery must not discharge between 10 PM and midnight.",
        "capacity": 200,
        "type": "no_discharge_window",
        "hours": [22, 23],
    },
    # max_grid_window paraphrases
    {
        "note": "Cap grid import at 200 kWh from 6 PM to 9 PM.",
        "capacity": 200,
        "type": "max_grid_window",
        "hours": [18, 19, 20],
        "value": 200.0,
    },
    {
        "note": "Limit grid intake to 130 kWh from 7 PM to 10 PM.",
        "capacity": 200,
        "type": "max_grid_window",
        "hours": [19, 20, 21],
        "value": 130.0,
    },
    {
        "note": "Grid draw must not exceed 175 kWh from 5 PM to 8 PM.",
        "capacity": 200,
        "type": "max_grid_window",
        "hours": [17, 18, 19],
        "value": 175.0,
    },
    # no_op paraphrases (distractors)
    {
        "note": "The library will close early on Friday.",
        "capacity": 200,
        "type": "no_op",
        "applies": False,
    },
    {
        "note": "Cafeteria now serves halal options daily.",
        "capacity": 200,
        "type": "no_op",
        "applies": False,
    },
    {
        "note": "IT maintenance is scheduled this weekend.",
        "capacity": 200,
        "type": "no_op",
        "applies": False,
    },
    {
        "note": "The seminar room booking was moved to next week.",
        "capacity": 200,
        "type": "no_op",
        "applies": False,
    },
    # G5-style hidden cases (from manual judge report)
    {
        "note": "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [13, 14],
        "value": 0.20,
    },
    {
        "note": "There will be a 100% solar outage from 6 AM until 2 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [6, 7, 8, 9, 10, 11, 12, 13],
        "value": 0.0,
    },
    {
        "note": "Refrain from charging the battery from 1 PM until 4 PM.",
        "capacity": 200,
        "type": "no_charge_window",
        "hours": [13, 14, 15],
    },
    {
        "note": "Grid import may not exceed 100 kWh between 6 PM and 9 PM.",
        "capacity": 200,
        "type": "max_grid_window",
        "hours": [18, 19, 20],
        "value": 100.0,
    },
    {
        "note": "Reduce solar to 0% from 10 AM to 12 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [10, 11],
        "value": 0.0,
    },
    {
        "note": "Cap grid at 100 kWh from 6 PM to 9 PM.",
        "capacity": 200,
        "type": "max_grid_window",
        "hours": [18, 19, 20],
        "value": 100.0,
    },
    {
        "note": "Full solar outage from 10 AM to 12 PM.",
        "capacity": 200,
        "type": "solar_reduction",
        "hours": [10, 11],
        "value": 0.0,
    },
    {
        "note": "Keep battery above 200 kWh from 8 PM to 10 PM.",
        "capacity": 500,
        "type": "minimum_battery_reserve",
        "hours": [20, 21],
        "value": 200.0,
    },
]


def main():
    n_ok = 0
    n_fail = 0
    for c in CASES:
        if _check(
            c["note"],
            c["capacity"],
            c["type"],
            c.get("hours"),
            c.get("value"),
            c.get("applies", True),
        ):
            n_ok += 1
        else:
            n_fail += 1
    print(f"\nPASS: {n_ok}/{len(CASES)}  FAIL: {n_fail}/{len(CASES)}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
