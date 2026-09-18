"""
Pydantic v2 schemas matching the exact GridWise contract
from the problem statement, Section 3 (request) and Section 4 (response).
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator, model_validator


# ---------- Enums (allowed directive types and battery actions) ----------

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


# ---------- Request schema (Section 3) ----------


class HourData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: StrictInt = Field(..., ge=0, le=23)
    demand_kwh: StrictFloat = Field(..., ge=0)
    solar_kwh: StrictFloat = Field(..., ge=0)
    tariff_bdt_per_kwh: StrictFloat = Field(..., ge=0)


class BatteryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: StrictFloat = Field(..., gt=0)
    initial_energy_kwh: StrictFloat = Field(..., ge=0)
    minimum_energy_kwh: StrictFloat = Field(..., ge=0)
    max_charge_kwh_per_hour: StrictFloat = Field(..., ge=0)
    max_discharge_kwh_per_hour: StrictFloat = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "BatteryConfig":
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError(
                "initial_energy_kwh must be <= capacity_kwh"
            )
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError(
                "minimum_energy_kwh must be <= capacity_kwh"
            )
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            pass
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., min_length=1)
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourData] = Field(..., min_length=24, max_length=24)
    battery: BatteryConfig

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: List[str]) -> List[str]:
        for i, n in enumerate(v):
            if not isinstance(n, str) or not n.strip():
                raise ValueError(
                    f"operator_notes[{i}] must be a non-empty string"
                )
        return v

    @model_validator(mode="after")
    def _hours_complete(self) -> "OptimizeRequest":
        if len(self.hours) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        seen = set()
        for h in self.hours:
            if h.hour in seen:
                raise ValueError(f"duplicate hour value: {h.hour}")
            seen.add(h.hour)
        expected = set(range(24))
        if seen != expected:
            missing = expected - seen
            raise ValueError(f"hours must include 0..23, missing: {sorted(missing)}")
        return self


# ---------- Response schema (Section 4) ----------


class SolarReductionAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1)
    factor: float = Field(..., ge=0, le=1)


class MinimumBatteryReserveAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1)
    minimum_energy_kwh: float = Field(..., ge=0)


class NoChargeWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1)


class NoDischargeWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1)


class MaxGridWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1)
    max_grid_kwh: float = Field(..., ge=0)


StructuredAdjustment = Optional[
    Union[
        SolarReductionAdjustment,
        MinimumBatteryReserveAdjustment,
        NoChargeWindowAdjustment,
        NoDischargeWindowAdjustment,
        MaxGridWindowAdjustment,
        dict,  # for null -> no_op
    ]
]


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0)
    solar_used_kwh: float = Field(..., ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0)
    battery_energy_after_kwh: float = Field(..., ge=0)


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ---------- Health check ----------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
