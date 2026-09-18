"""
Shared replay/validation logic used both by the API's final validator
and by the local judge test. Returns a list of error strings (empty
when the plan is valid).

The judge uses the same physical tolerance the rubric specifies
(0.01 kWh / 0.01 BDT).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

# Judge contract tolerance from the public sample pack: 0.01 kWh / 0.01 BDT.
JUDGE_TOL = 0.01


def compile_directive_effects(
    hours: Sequence[Mapping[str, Any]],
    battery: Mapping[str, Any],
    directives: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Walk the validated directives once and produce all per-hour bounds
    that both the optimizer and the replay validator need.

    Returns a dict with:
        eff_solar:    List[float] length N
        min_battery:  List[float] length N
        max_grid_cap: List[float] length N  (math.inf = no cap)
        no_charge:    set[int]
        no_discharge: set[int]
    """
    n = len(hours)
    eff_solar = [float(h["solar_kwh"]) for h in hours]
    max_grid_cap = [float("inf")] * n
    min_battery = [float(battery["minimum_energy_kwh"])] * n
    no_charge: set = set()
    no_discharge: set = set()

    for d in directives:
        if not d.get("applies"):
            continue
        dtype = d.get("directive_type")
        adj = d.get("structured_adjustment") or {}
        hours_set = adj.get("hours", [])
        if not isinstance(hours_set, list):
            continue
        if dtype == "solar_reduction":
            factor = float(adj.get("factor", 1.0))
            for h in hours_set:
                if 0 <= h < n:
                    eff_solar[h] *= factor
        elif dtype == "max_grid_window":
            cap = float(adj.get("max_grid_kwh"))
            for h in hours_set:
                if 0 <= h < n:
                    max_grid_cap[h] = min(max_grid_cap[h], cap)
        elif dtype == "minimum_battery_reserve":
            mev = float(adj.get("minimum_energy_kwh"))
            for h in hours_set:
                if 0 <= h < n:
                    if mev > min_battery[h]:
                        min_battery[h] = mev
        elif dtype == "no_charge_window":
            no_charge.update(int(h) for h in hours_set)
        elif dtype == "no_discharge_window":
            no_discharge.update(int(h) for h in hours_set)

    return {
        "eff_solar": eff_solar,
        "min_battery": min_battery,
        "max_grid_cap": max_grid_cap,
        "no_charge": no_charge,
        "no_discharge": no_discharge,
    }


def replay_check(
    payload: Mapping[str, Any],
    directive_interpretation: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    totals: Mapping[str, float],
    tol: float = JUDGE_TOL,
) -> List[str]:
    """Recompute every constraint from the raw plan and compare against the
    totals the response will report. Returns a list of error strings
    (empty if everything is consistent).
    """
    errs: List[str] = []
    if len(plan) != 24:
        errs.append(f"hourly_plan length {len(plan)} != 24")
        return errs
    if sorted(p["hour"] for p in plan) != list(range(24)):
        errs.append("hourly_plan missing some hours 0..23")
        return errs

    battery = payload["battery"]
    hours_lookup = {h["hour"]: h for h in payload["hours"]}
    demand = {h: float(hours_lookup[h]["demand_kwh"]) for h in range(24)}
    tariff = {h: float(hours_lookup[h]["tariff_bdt_per_kwh"]) for h in range(24)}

    fx = compile_directive_effects(
        hours=payload["hours"],
        battery=battery,
        directives=directive_interpretation,
    )
    eff_solar = fx["eff_solar"]
    no_charge = fx["no_charge"]
    no_discharge = fx["no_discharge"]
    min_reserve = fx["min_battery"]
    max_grid_cap = fx["max_grid_cap"]
    grid_capped_hours = {
        h for h in range(24) if max_grid_cap[h] < float("inf")
    }

    prev_e = float(battery["initial_energy_kwh"])
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0
    for p in plan:
        h = int(p["hour"])
        g = float(p["grid_kwh"])
        s = float(p["solar_used_kwh"])
        bk = float(p["battery_kwh"])
        action = p["battery_action"]
        e_after = float(p["battery_energy_after_kwh"])

        if action == "charge":
            charge, discharge = bk, 0.0
        elif action == "discharge":
            charge, discharge = 0.0, bk
        else:
            charge, discharge = 0.0, 0.0

        lhs = g + s + discharge
        rhs = demand[h] + charge
        if not math.isclose(lhs, rhs, abs_tol=tol):
            errs.append(
                f"hour {h}: balance {lhs} vs {rhs} (demand {demand[h]})"
            )

        if s > eff_solar[h] + tol:
            errs.append(
                f"hour {h}: solar_used {s} > eff_solar {eff_solar[h]}"
            )

        if e_after > float(battery["capacity_kwh"]) + tol:
            errs.append(
                f"hour {h}: battery {e_after} > capacity {battery['capacity_kwh']}"
            )
        if e_after < min_reserve[h] - tol:
            errs.append(
                f"hour {h}: battery {e_after} < reserve {min_reserve[h]}"
            )

        if charge > float(battery["max_charge_kwh_per_hour"]) + tol:
            errs.append(f"hour {h}: charge {charge} > max_charge_rate")
        if discharge > float(battery["max_discharge_kwh_per_hour"]) + tol:
            errs.append(f"hour {h}: discharge {discharge} > max_discharge_rate")

        if h in no_charge and charge > tol:
            errs.append(f"hour {h}: charge > 0 in no_charge_window")
        if h in no_discharge and discharge > tol:
            errs.append(f"hour {h}: discharge > 0 in no_discharge_window")

        if h in grid_capped_hours and g > max_grid_cap[h] + tol:
            errs.append(
                f"hour {h}: grid {g} > max_grid_cap {max_grid_cap[h]}"
            )

        expected_e = prev_e + charge - discharge
        if not math.isclose(e_after, expected_e, abs_tol=tol):
            errs.append(
                f"hour {h}: transition {e_after} vs {expected_e} (prev {prev_e})"
            )

        prev_e = e_after
        total_grid += g
        total_cost += g * tariff[h]
        peak_grid = max(peak_grid, g)

    if not math.isclose(
        prev_e, float(battery["initial_energy_kwh"]), abs_tol=tol
    ):
        errs.append(
            f"end-of-day battery {prev_e} != initial {battery['initial_energy_kwh']}"
        )

    if not math.isclose(total_grid, totals["total_grid_kwh"], abs_tol=tol):
        errs.append(
            f"total_grid_kwh {totals['total_grid_kwh']} != recompute {total_grid}"
        )
    if not math.isclose(total_cost, totals["total_cost_bdt"], abs_tol=tol):
        errs.append(
            f"total_cost_bdt {totals['total_cost_bdt']} != recompute {total_cost}"
        )
    if not math.isclose(peak_grid, totals["peak_grid_kwh"], abs_tol=tol):
        errs.append(
            f"peak_grid_kwh {totals['peak_grid_kwh']} != recompute {peak_grid}"
        )

    return errs
