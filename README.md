# GridWise — LLM-Assisted Energy Optimizer

[![Docker Pulls](https://img.shields.io/docker/pulls/voideye/gridwise)](https://hub.docker.com/r/voideye/gridwise)
[![Docker Image Size](https://img.shields.io/docker/image-size/voideye/gridwise/latest)](https://hub.docker.com/r/voideye/gridwise)
[![GitHub](https://img.shields.io/badge/GitHub-BUP--Hackathon-181717?logo=github)](https://github.com/darkEye-2021831014/BUP-Hackathon)

> **BUP CSE Fest 2026 · Preliminary Submission**
> FastAPI service that interprets natural-language operator notes for a
> 24-hour campus energy schedule and returns an LP-optimal
> grid/battery plan — built to the exact contract in the official
> problem statement.

---

## Table of contents

1. [At a glance](#at-a-glance)
2. [Run it in 30 seconds (Docker)](#run-it-in-30-seconds-docker)
3. [Run it from source](#run-it-from-source)
4. [API contract](#api-contract)
5. [Architecture](#architecture)
6. [Optimizer (LP formulation)](#optimizer-lp-formulation)
7. [LLM interpreter & deterministic fallback](#llm-interpreter--deterministic-fallback)
8. [Environment variables](#environment-variables)
9. [Verification & tests](#verification--tests)
10. [Security & no-secrets policy](#security--no-secrets-policy)
11. [Known limitations](#known-limitations)
12. [Credits & licenses](#credits--licenses)

---

## At a glance

| Item                  | Value                                                          |
|-----------------------|----------------------------------------------------------------|
| **Event**             | BUP CSE Fest 2026 — Hackathon Preliminary Round                |
| **Track**             | LLM-assisted optimization (24-hour campus microgrid)            |
| **API framework**     | FastAPI + Pydantic v2 (strict mode)                            |
| **Optimizer**         | Linear Program via PuLP, solved by bundled CBC                 |
| **LLM providers**     | Groq (primary) → Google Gemini (fallback) → regex (offline)    |
| **Endpoints**         | `POST /optimize-energy`, `GET /health`                         |
| **Docker image**      | [`voideye/gridwise:latest`](https://hub.docker.com/r/voideye/gridwise) |
| **Source**            | [github.com/darkEye-2021831014/BUP-Hackathon](https://github.com/darkEye-2021831014/BUP-Hackathon) |
| **Python**            | 3.11 (slim multi-stage image, non-root runtime)                |

---

## Run it in 30 seconds (Docker)

The published image bundles the API, the LP solver, and a non-root
runtime — no build step, no model files, no secrets baked in.

```bash
# 1. Pull
docker pull voideye/gridwise:latest

# 2. Run (with at least one LLM key; omit -e flags to use the
#    deterministic regex fallback)
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  voideye/gridwise:latest

# 3. Health check
curl -s http://localhost:8000/health
# {"status":"ok"}
```

Want a one-shot request to see a full response?

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @tests/sample-01.json | jq .
```

`docker history voideye/gridwise:latest` will confirm there are **no
`ENV KEY=…` lines** — secrets are only ever injected at runtime.

---

## Run it from source

Requirements: Python 3.11+ and a C toolchain if PuLP can't pull a wheel
for your platform (most x86_64 / arm64 Linux + macOS get wheels).

```bash
git clone https://github.com/darkEye-2021831014/BUP-Hackathon.git
cd BUP-Hackathon/API

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env and paste at least one key
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI / Swagger UI: <http://localhost:8000/docs>

---

## API contract

Both endpoints match the problem statement exactly.

### `GET /health` → `200 OK`

```json
{ "status": "ok" }
```

No LLM warm-up, no side effects. Used by Docker healthcheck and the
judge's connectivity probes.

### `POST /optimize-energy`

Request body — see `app/schemas.py` and `tests/sample-01.json`:

```json
{
  "scenario_id": "GRID-101",
  "operator_notes": [
    "Solar output will drop to about 20% from 1 PM to 3 PM.",
    "Do not charge the battery between 2 PM and 4 PM.",
    "The cafeteria menu changes tomorrow."
  ],
  "hours": [
    { "hour": 0,  "demand_kwh":  90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6 },
    "… 23 more hours …"
  ],
  "battery": {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100
  }
}
```

Validation rules (enforced strictly before the LLM is called):

- `scenario_id` — non-empty string
- `operator_notes` — 1–3 non-empty strings
- `hours` — exactly 24 entries, hours `0..23` each once, ascending
- `battery.initial_energy_kwh ≤ capacity_kwh`
- All numeric fields use **strict** typing — `"100"` (string) is
  rejected with HTTP 400 (no silent coercion)

Response body — see Section 4 of the problem statement:

```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    { "note_index": 0, "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": { "hours": [13, 14], "factor": 0.2 },
      "explanation": "…" }
  ],
  "hourly_plan": [
    { "hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0,
      "battery_action": "charge", "battery_kwh": 0.0,
      "battery_energy_after_kwh": 200.0 }
    /* … 23 more hours … */
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "applies solar reduction(s); respects no-charge window(s); …"
}
```

Failure modes:

| Condition                                     | HTTP | Notes                              |
|-----------------------------------------------|------|------------------------------------|
| Malformed JSON / missing required fields      | 400  | Pydantic validation error          |
| Wrong type (e.g. `"100"` for `demand_kwh`)    | 400  | Strict types — no coercion         |
| Semantically impossible to honor every note    | 200  | LP softens the conflicting notes; response still returned with `directive_interpretation` intact (schema requirement) |
| Unhandled internal error                      | 500  | Generic message — no stack trace leaked |

---

## Architecture

```
Client / Judge
   │  POST /optimize-energy  (request per Section 3 of the spec)
   ▼
[1] FastAPI Pydantic v2 (strict) validation ──── 400 on malformed input
   ▼
[2] LLM Interpreter (interpret_notes) ──── Groq primary, Gemini fallback,
   │                                            deterministic regex fallback,
   │                                            final no_op fallback.
   │   Returns RAW directive_interpretation JSON (untrusted).
   ▼
[3] Deterministic Guardrail Validator ──── pure Python, no LLM calls.
   │   - directive_type ∈ allowed enum  →  else no_op
   │   - note_index 0..N-1 in order
   │   - hours: unique ints 0..23 ascending
   │   - solar_reduction.factor ∈ [0, 1]
   │   - minimum_battery_reserve ≥ 0  and  ≤ capacity_kwh
   │   - max_grid_window.max_grid_kwh ≥ 0
   │   - applies=false only for no_op
   ▼
[4] Directive → Optimizer Constraint Compiler (inside optimize())
   ▼
[5] PuLP LP Optimizer (CBC) ──── minimize Σ_h grid[h] × tariff[h]
   ▼
[6] Final Validator / Replay ──── re-derives battery_action from
   │                                   charge/discharge, recomputes
   │                                   totals from hourly_plan,
   │                                   confirms every constraint.
   ▼
[7] JSON Response (exact Section 4 schema)
```

Cross-cutting:

- Global exception handler never leaks a stack trace.
- Structured logging that **never** logs API keys.
- 24-h neutrality (`E[23] == initial_energy_kwh`) is a hard LP
  constraint, not an after-the-fact patch.

---

## Optimizer (LP formulation)

Pure LP (no MILP / no binary variables) — CBC ships with PuLP for
manylinux. A tiny `1e-6` per-unit penalty on both `charge` and
`discharge` prevents the trivial "charge-and-discharge in the same
hour" degeneracy without affecting the judge-grade optimum (≈ 8 orders
of magnitude below the `0.01 BDT` judge tolerance).

Decision variables for each hour `h = 0..23`:

| Variable        | Bounds                                  | Meaning                          |
|-----------------|-----------------------------------------|----------------------------------|
| `grid[h]`       | `≥ 0`, optional cap from `max_grid_window` | kWh drawn from the grid     |
| `solar_used[h]` | `0 ≤ … ≤ effective_solar[h]`            | kWh of solar actually used       |
| `charge[h]`     | `≥ 0`, ≤ `max_charge_kwh_per_hour`      | kWh stored into the battery      |
| `discharge[h]`  | `≥ 0`, ≤ `max_discharge_kwh_per_hour`   | kWh pulled out of the battery    |
| `battery_energy[h]` | `active_min[h] ≤ … ≤ capacity_kwh`   | energy **after** hour `h`        |

Constraints per hour `h`:

1. Energy balance: `grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h]`
2. Battery transition: `E[h] == E[h-1] + charge[h] - discharge[h]` (with `E[-1] = initial_energy_kwh`)
3. Battery bounds: `active_min[h] ≤ E[h] ≤ capacity_kwh`
4. Rate limits: `charge[h] ≤ max_charge_kwh_per_hour`, `discharge[h] ≤ max_discharge_kwh_per_hour`
5. `no_charge_window` → `charge[h] == 0`
6. `no_discharge_window` → `discharge[h] == 0`
7. `max_grid_window` → `grid[h] ≤ max_grid_kwh` (smallest cap wins if multiple apply)
8. End-of-day neutrality: `E[23] == initial_energy_kwh`

Directive → constraint compilation lives in `app/optimizer.py` and
`app/replay.py` — both the optimizer and the replay validator share
the exact same logic.

If the directive set is collectively infeasible (e.g. grid cap too
low to serve peak demand), the LP is re-solved with the conflicting
soft caps (`max_grid_window`, `minimum_battery_reserve`) replaced by
penalized slack variables. The full directive list is still returned in
the response (the judge requires one entry per input note); the replay
validator skips softened directives but still checks the rest.

---

## LLM interpreter & deterministic fallback

Each request sends **all** operator notes in a single LLM call with
`temperature=0`, JSON-only system prompt, and short timeouts to stay
well inside the 5 s p95 budget.

Resolution order:

1. **Groq** — `llama-3.3-70b-versatile` (default). Fast (~hundreds of
   tokens/s), generous free tier.
2. **Google Gemini** — `gemini-2.0-flash` (default). Used if Groq errors
   or times out twice.
3. **Deterministic regex interpreter** (`app/llm_interpreter.py`) —
   used when both LLMs are unreachable or `LLM_PROVIDER=off`. Handles
   ~95 % of the paraphrases we tested, including word-numbers
   ("two hundred kWh"), word fractions ("one-fifth"), and end-exclusive
   time windows ("from 1 until 3").
4. **All-`no_op` fallback** — returns a feasible plan with one
   directive entry per note.

This layered approach guarantees a 200 response with a feasible plan
for every well-formed request, even if every LLM provider is down.

---

## Environment variables

All variables live in `.env.example` (no real values committed). The
Docker image does **not** bake any of them in.

| Variable            | Default                 | Purpose                                          |
|---------------------|-------------------------|--------------------------------------------------|
| `LLM_PROVIDER`      | `groq`                  | `groq`, `gemini`, or `off` (skip the LLM path)   |
| `GROQ_API_KEY`      | *(none)*                | Free-tier key from <https://console.groq.com>    |
| `GROQ_MODEL`        | `llama-3.3-70b-versatile` | Override the Groq model                        |
| `GEMINI_API_KEY`    | *(none)*                | Free-tier key from <https://ai.google.dev>       |
| `GEMINI_MODEL`      | `gemini-2.0-flash`      | Override the Gemini model                        |
| `LLM_TIMEOUT_S`     | `12`                    | Per-attempt HTTP timeout (seconds)               |
| `LLM_MAX_RETRIES`   | `1`                     | Retries per provider before falling back         |
| `PORT`              | `8000`                  | HTTP listen port                                 |
| `LOG_LEVEL`         | `INFO`                  | `DEBUG` / `INFO` / `WARNING`                     |

`.env` is git-ignored.

---

## Verification & tests

Five test scripts live under `tests/`. Each runs against the FastAPI
app in-process (no network needed) via `httpx.AsyncClient` +
`ASGITransport`.

| Script                       | Coverage                                                              |
|------------------------------|-----------------------------------------------------------------------|
| `tests/test_health.py`       | `GET /health` returns 200 + JSON                                       |
| `tests/test_judge.py`        | All public sample cases: directive semantics + replay (balance, battery, end-of-day neutrality, totals) |
| `tests/test_paraphrases.py`  | 29 paraphrased notes covering all 5 applicable directive types + distractors |
| `tests/test_malformed.py`    | 15 malformed-input cases — every one must return 4xx, never a 5xx with a stack trace |
| `tests/test_smoke.py`        | Bypasses the LLM; runs the LP solver only                              |

Run them all:

```bash
.venv/bin/python tests/test_health.py
.venv/bin/python tests/test_judge.py
.venv/bin/python tests/test_paraphrases.py
.venv/bin/python tests/test_malformed.py
.venv/bin/python tests/test_smoke.py
```

The replay validator used in `tests/test_judge.py` is the same
function (`app.replay.replay_check`) the API itself uses for its
final validator — guarantees the local tests and the running service
agree.

---

## Security & no-secrets policy

- `.env.example` ships **variable names only**.
- `.env` is in `.gitignore`.
- `Dockerfile` does not `ENV` or `ARG` any API key.
- The image does not contain `GROQ_API_KEY` or any other secret; the
  only way to provide one is at run time (`-e …` or your platform's
  secret manager).
- Logs are formatted to redact obvious secret-shaped strings.
- Global exception handler returns a generic `"Internal server error"`
  on unhandled failures — no stack trace, no prompt, no key.

---

## Known limitations

- **Linear battery.** The LP assumes lossless charge and discharge;
  no round-trip-efficiency term. (The problem statement uses the same
  linear model.)
- **No MILP.** Simultaneous charge + discharge in the same hour is
  discouraged with a tiny per-unit penalty (`1e-6`), not by integer
  variables. The penalty is 8 orders of magnitude below the judge
  tolerance for `total_cost_bdt`, so the judge-grade optimum is
  unchanged.
- **No telemetry.** Request bodies are sent to the LLM provider only
  for the duration of the call and are not retained by Groq / Gemini
  under their free-tier settings, but you should still avoid sending
  truly sensitive data through this service.

---

## Credits & licenses

Built with these open-source libraries (all permissive licenses):

| Library                                       | License        | Role                                  |
|-----------------------------------------------|----------------|---------------------------------------|
| [FastAPI](https://fastapi.tiangolo.com)       | MIT            | HTTP framework                        |
| [Uvicorn](https://www.uvicorn.org)            | BSD-3          | ASGI server                           |
| [Pydantic v2](https://docs.pydantic.dev)      | MIT            | Request/response validation           |
| [PuLP](https://coin-or.github.io/pulp)        | MIT            | LP modeling                           |
| [CBC](https://github.com/coin-or/Cbc)         | EPL 2.0        | LP solver (bundled with PuLP)         |
| [httpx](https://www.python-httpx.org)         | BSD-3          | HTTP client for LLM providers         |

LLM providers used for note interpretation:

- **Groq** — `llama-3.3-70b-versatile`
- **Google Gemini** — `gemini-2.0-flash`

No proprietary datasets or models are bundled. No secrets are required
to clone, install, or test locally — just optional API keys to enable
the real-LLM path.

---

## Links

- 🐳 Docker image: <https://hub.docker.com/r/voideye/gridwise>
- 🐙 Source code: <https://github.com/darkEye-2021831014/BUP-Hackathon>
- 📄 Problem statement: see
  `BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf`
  in the repo root
- 📊 Public sample cases: `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`
  in the repo root
- 📚 Rubric: `BUP_CSE_FEST_2026_Participant_Guide_&_Evaluation_Rubric_GridWise_LLM.pdf`
  in the repo root
