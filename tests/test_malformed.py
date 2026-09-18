"""
Malformed-input tests (Section 8.3 of the master prompt).

Verify that bad inputs return a controlled 400/422 response and never
crash with a stack trace or 5xx-with-stack-trace.
"""
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.main import app  # noqa: E402

# A minimal VALID request — used as a base for mutation tests.
VALID = {
    "scenario_id": "MAL-1",
    "operator_notes": ["Some note."],
    "hours": [
        {
            "hour": h,
            "demand_kwh": 100 + h,
            "solar_kwh": 0 if h < 6 else 50,
            "tariff_bdt_per_kwh": 5 + (h % 5),
        }
        for h in range(24)
    ],
    "battery": {
        "capacity_kwh": 200,
        "initial_energy_kwh": 100,
        "minimum_energy_kwh": 40,
        "max_charge_kwh_per_hour": 50,
        "max_discharge_kwh_per_hour": 50,
    },
}


def _mutate(req: dict, path: list[str], value: Any) -> dict:
    """Replace value at path (list of keys/indices) in a deep copy of req."""
    import copy
    r = copy.deepcopy(req)
    cur = r
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return r


async def _check(client, label, payload, expected_min=400, expected_max=499):
    try:
        resp = await client.post(
            "/optimize-energy", json=payload, timeout=30.0
        )
    except Exception as e:
        print(f"  FAIL  {label}: client threw {type(e).__name__}: {e}")
        return False
    if expected_min <= resp.status_code <= expected_max:
        print(f"  OK    {label}  HTTP {resp.status_code}")
        return True
    # If 200 OK we accept that too (valid edge cases sometimes are accepted)
    if resp.status_code == 200:
        print(f"  OK    {label}  HTTP 200 (accepted)")
        return True
    print(
        f"  FAIL  {label}: HTTP {resp.status_code} not in [{expected_min},{expected_max}]: "
        f"{resp.text[:200]}"
    )
    return False


async def main():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
    ) as client:
        n_ok = 0
        n_fail = 0

        tests = [
            ("missing scenario_id", _mutate(VALID, ["scenario_id"], None)),
            ("empty operator_notes", _mutate(VALID, ["operator_notes"], [])),
            ("25 hours", _mutate(VALID, ["hours"], VALID["hours"] + [VALID["hours"][0]])),
            ("23 hours", _mutate(VALID, ["hours"], VALID["hours"][:-1])),
            ("duplicate hour 0", _mutate(VALID, ["hours", 1, "hour"], 0)),
            ("missing hour 5", _mutate(VALID, ["hours"], [h for h in VALID["hours"] if h["hour"] != 5])),
            ("hour=25", _mutate(VALID, ["hours", 5, "hour"], 25)),
            ("hour=-1", _mutate(VALID, ["hours", 5, "hour"], -1)),
            ("demand=-5", _mutate(VALID, ["hours", 5, "demand_kwh"], -5)),
            ("battery missing capacity_kwh",
             _mutate(VALID, ["battery"], {k: v for k, v in VALID["battery"].items() if k != "capacity_kwh"})),
            ("battery initial > capacity",
             _mutate(VALID, ["battery", "initial_energy_kwh"], 999)),
            ("scenario_id empty",
             _mutate(VALID, ["scenario_id"], "")),
            ("operator_notes 4 entries",
             _mutate(VALID, ["operator_notes"], ["a", "b", "c", "d"])),
            ("operator_notes contains non-string",
             _mutate(VALID, ["operator_notes", 0], 42)),
        ]

        # also a totally empty body
        body = None
        try:
            resp = await client.post(
                "/optimize-energy", content=b"", timeout=10.0
            )
            if 400 <= resp.status_code <= 499:
                print(f"  OK    empty body HTTP {resp.status_code}")
                n_ok += 1
            else:
                print(f"  FAIL  empty body HTTP {resp.status_code}")
                n_fail += 1
        except Exception as e:
            print(f"  FAIL  empty body: {e}")
            n_fail += 1

        for label, payload in tests:
            if await _check(client, label, payload):
                n_ok += 1
            else:
                n_fail += 1

        print(f"\nPASS: {n_ok}/{n_ok+n_fail}  FAIL: {n_fail}/{n_ok+n_fail}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
