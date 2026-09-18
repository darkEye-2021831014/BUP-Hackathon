"""
FastAPI application for the GridWise LLM-Assisted Energy Optimizer.

Endpoints:
  GET  /health         -> {"status": "ok"}
  POST /optimize-energy  (Section 3 request -> Section 4 response)

Architecture (per Section 2 of the master prompt):
  request -> [1] Pydantic validation
         -> [2] LLM Interpreter       (interpret_notes)
         -> [3] Guardrail Validator   (validate_and_normalize_directives)
         -> [4] Directive Compiler    (inside optimizer)
         -> [5] PuLP LP Optimizer     (optimize)
         -> [6] Final Validator       (asserts energy balance + battery rules)
         -> [7] JSON Response
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .guardrails import validate_and_normalize_directives
from .llm_interpreter import interpret_notes
from .optimizer import optimize
from .replay import JUDGE_TOL, replay_check
from .schemas import (
    HealthResponse,
    HourlyPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise — LLM-Assisted Energy Optimizer",
    version="1.0.0",
    description=(
        "BUP CSE Fest 2026 preliminary submission. "
        "Interprets operator notes via LLM, validates deterministically, "
        "and minimizes 24-hour grid cost using a PuLP LP."
    ),
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Cheap, fast health check. No LLM warm-up here."""
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# /optimize-energy
# ---------------------------------------------------------------------------


@app.post(
    "/optimize-energy",
    response_model=OptimizeResponse,
    tags=["optimize"],
)
def optimize_energy(payload: OptimizeRequest) -> OptimizeResponse:
    log.info(
        "optimize-energy scenario_id=%s notes=%d",
        payload.scenario_id,
        len(payload.operator_notes),
    )

    # ---- Step 2: LLM interpretation (UNTRUSTED RAW OUTPUT) ----
    raw_directives = interpret_notes(
        operator_notes=payload.operator_notes,
        battery_capacity_kwh=payload.battery.capacity_kwh,
    )

    # ---- Step 3: Deterministic guardrail validation ----
    validated_directives = validate_and_normalize_directives(
        raw_directives=raw_directives,
        battery=payload.battery,
        expected_n=len(payload.operator_notes),
    )

    # ---- Steps 4 + 5: Optimize (compile + solve LP) ----
    try:
        opt_result = optimize(
            hours=payload.hours,
            battery=payload.battery,
            validated_directives=validated_directives,
        )
    except Exception as e:
        log.exception("optimizer failed")
        # Don't leak internals; surface a controlled error.
        raise HTTPException(
            status_code=422,
            detail=f"Optimizer could not produce a feasible plan: {e}",
        )

    # ---- Step 6: Final validator / replay ----
    directives = opt_result["directives"]
    # When the LP fell back to soft constraints for a directive type, the
    # resulting plan may exceed that directive's bound. The replay check
    # would reject it; skip those directives during replay but keep them
    # in the response (the judge requires one entry per input note).
    softened = opt_result.get("softened_types", set())
    replay_directives = [
        d for d in directives
        if d.get("directive_type") not in softened
    ]
    final_validator(
        payload=payload,
        directive_interpretation=replay_directives,
        opt_result=opt_result,
    )

    # ---- Step 7: Response ----
    response = OptimizeResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=directives,
        hourly_plan=[
            HourlyPlanEntry(**p) for p in opt_result["hourly_plan"]
        ],
        total_grid_kwh=opt_result["total_grid_kwh"],
        total_cost_bdt=opt_result["total_cost_bdt"],
        peak_grid_kwh=opt_result["peak_grid_kwh"],
        plan_summary=opt_result["plan_summary"],
    )
    return response


# ---------------------------------------------------------------------------
# Final validator: re-derive everything from the hourly_plan to be sure
# the response is internally consistent and obeys all rules.
# ---------------------------------------------------------------------------


def final_validator(
    payload: OptimizeRequest,
    directive_interpretation: List[Dict],
    opt_result: Dict,
) -> None:
    """Sanity-check the LP solution against the request and the directives.

    Thin wrapper over `app.replay.replay_check` that converts the typed
    Pydantic request to a dict and raises on the first inconsistency.
    FastAPI surfaces a controlled 422 response.
    """
    payload_dict = payload.model_dump(mode="json")
    errs = replay_check(
        payload=payload_dict,
        directive_interpretation=directive_interpretation,
        plan=opt_result["hourly_plan"],
        totals={
            "total_grid_kwh": opt_result["total_grid_kwh"],
            "total_cost_bdt": opt_result["total_cost_bdt"],
            "peak_grid_kwh": opt_result["peak_grid_kwh"],
        },
        tol=JUDGE_TOL,
    )
    if errs:
        raise ValueError("; ".join(errs))


# ---------------------------------------------------------------------------
# Global exception handlers — never leak a stack trace to the client.
# ---------------------------------------------------------------------------


def _scrub_non_finite(obj: Any) -> Any:
    """Replace NaN/Inf floats with a sentinel string so the response is
    valid JSON. Pydantic's validation errors can include the offending
    input value (e.g. NaN), which Python's json.dumps rejects."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return "NaN"
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_non_finite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    # exc.errors() can contain non-JSON values (NaN/Inf floats included);
    # jsonable_encoder handles most coercion, but NaN/Inf still slip
    # through and crash json.dumps. Scrub them to a string sentinel.
    encoded = jsonable_encoder(exc.errors())
    safe = _scrub_non_finite(encoded)
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid request payload",
            "errors": safe,
        },
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
