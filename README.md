# GridWise — LLM-Assisted Smart Campus Energy Optimization

BUP CSE Fest 2026 Hackathon · Online Preliminary · Smart Campus Energy Optimization Challenge

A single stateless HTTP service that reads natural-language campus operator notes, converts them
into machine-checkable directives with a language model, validates those directives with
deterministic guardrails, solves the 24-hour scheduling problem with a linear program, and
independently replays the result before returning it.

**The central design point:**

| Component | Responsibility | What it is *not* allowed to do |
|---|---|---|
| **LLM** (Groq → Gemini) | Language understanding only — turn each operator note into one structured directive | Never computes a schedule, battery state, cost, or any scenario number |
| **Deterministic guardrails** | Reject untrusted model output that breaks any rule | Never repairs, guesses at, or "fixes up" a malformed directive |
| **LP optimizer** | All mathematical reasoning: the actual cost-minimizing 24-hour schedule | Never sees the operator notes, only canonical numbers |
| **Replay validator** | Independently re-derive and verify the finished schedule | Shares no computation with the optimizer |

---

## Table of contents

1. [Problem understanding](#1-problem-understanding) · 2. [Solution overview](#2-solution-overview) ·
3. [Architecture](#3-architecture) · 4. [End-to-end flow](#4-end-to-end-flow) ·
5. [API specification](#5-api-specification) · 6. [LLM directive interpretation](#6-llm-directive-interpretation) ·
7. [Groq primary model](#7-groq-primary-model) · 8. [Gemini fallback](#8-gemini-fallback) ·
9. [Deterministic guardrails](#9-deterministic-guardrails) · 10. [Optimization model](#10-optimization-model) ·
11. [Energy balance](#11-energy-balance) · 12. [Battery model](#12-battery-model) ·
13. [Directive application](#13-directive-application) · 14. [Replay validation](#14-replay-validation) ·
15. [Project structure](#15-project-structure) · 16. [Environment variables](#16-environment-variables) ·
17. [Local setup](#17-local-setup) · 18. [Running the service](#18-running-the-service) ·
19. [Running tests](#19-running-tests) · 20. [Public sample tests](#20-public-sample-tests) ·
21. [Docker](#21-docker) · 22. [Deployment](#22-deployment) · 23. [Performance](#23-performance) ·
24. [Security](#24-security) · 25. [Failure handling](#25-failure-handling) ·
26. [Limitations](#26-limitations) · 27. [Future improvements](#27-future-improvements) ·
28. [Dependencies and credits](#28-dependencies-and-credits)

---

## 1. Problem understanding

A campus buys grid electricity, generates rooftop solar, and operates a battery. Demand, solar
availability and tariff are given for the next 24 hours. Separately, operators send 1–3 short
natural-language notes describing temporary conditions affecting the same day.

The service must:

1. Interpret **every** note — exactly one `directive_interpretation` entry per note, in
   `note_index` order.
2. Map relevant notes onto one of the supported directive types; mark irrelevant notes `no_op`
   rather than inventing an energy rule.
3. Validate the interpreted directives before they reach the optimizer.
4. Produce a 24-hour schedule that is **valid first** — energy balance, battery bounds and
   transitions, rate limits, effective-solar bounds, end-of-day neutrality, and every applicable
   directive — and **cheap second**.

The scoring model makes the ordering explicit: a cheap schedule built on a misread or unapplied
directive scores zero for that case. Correctness therefore drives every design decision below.

Two semantics are easy to get backwards and are handled explicitly throughout:

- **Time windows are start-inclusive and end-exclusive.** "1 PM to 3 PM" → hours `[13, 14]`.
- **`factor` is the fraction of solar that *remains*, not the amount removed.** "an 80% reduction"
  → `factor = 0.2`.

## 2. Solution overview

The pipeline is deliberately short and has exactly one non-deterministic stage, which is fenced in
on both sides:

```
HTTP request
  → Pydantic v2 structural validation (strict typing)
  → LLM interprets operator_notes ONLY          ← the only non-deterministic step
  → deterministic guardrails (reject, never repair)
  → canonical directives (pure numbers and hour sets)
  → PuLP linear program (CBC solver)
  → independent replay validation
  → exact API response
```

The optimizer and the replay validator have no dependency on the LLM at all. If the language model
is removed, the deterministic core still solves any scenario correctly — it simply has no
directives to apply. That is why the implementation order was: schemas → optimizer → replay
validator → guardrails → LLM.

## 3. Architecture

```
          ┌────────────────────────────────────────────┐
HTTP ─►   │  Pydantic v2  (strict typing, 400 on bad)  │
          └─────────────┬──────────────────────────────┘
                        │ scenario_id, notes, hours, battery
                        ▼
          ┌────────────────────────────────────────────┐
          │  Groq  (PRIMARY)  llama-3.3-70b-versatile │
          └─────┬───────────────────────────┬──────────┘
                │ ok                        │ fail / 429 / malformed
                ▼                           ▼
          ┌─────────────┐          ┌────────────────────────────┐
          │ Guardrails  │◄─────────│ Gemini 2.0 Flash (FALLBACK)│
          └──────┬──────┘          └────────────────────────────┘
                 │ rejected directives
                 │ → retry / next provider
                 ▼
          ┌────────────────────────────────────────────┐
          │  Canonical directives  (hour sets + nums)  │
          └─────────────┬──────────────────────────────┘
                        ▼
          ┌────────────────────────────────────────────┐
          │  PuLP LP  (CBC)                            │
          │    minimize Σ grid[h] × tariff[h]          │
          └─────────────┬──────────────────────────────┘
                        │ grid, solar_used, charge, discharge, E[h]
                        ▼
          ┌────────────────────────────────────────────┐
          │  Netting + output normalization            │
          └─────────────┬──────────────────────────────┘
                        ▼
          ┌────────────────────────────────────────────┐
          │  Replay validator  (independent re-derive) │
          └─────────────┬──────────────────────────────┘
                        │ fail → 500, schedule discarded
                        ▼
          ┌────────────────────────────────────────────┐
          │  Exact API response  (Section 4 schema)   │
          └────────────────────────────────────────────┘
```

Key architectural properties:

- **Stateless.** No database, cache, queue or persistent storage. Any instance can serve any request.
- **The LLM cannot reach the optimizer directly.** The only type that crosses the boundary is the
  validated directive dict (an hour set plus a float). There is no code path by which model text
  becomes a constraint without passing through the guardrails.
- **Two independent checks of the same schedule.** The optimizer asserts constraints to the solver;
  the replay validator re-derives the whole schedule from scratch and checks the same rules again.

## 4. End-to-end flow

1. **Structural validation** (`app/schemas.py`). Exactly 24 hours covering 0–23 with no duplicates,
   1–3 non-empty notes, non-negative finite numbers, and a self-consistent battery. Strict typing —
   `"100"` (string) is rejected with HTTP 400, never coerced to `100`. Unknown fields are rejected
   with `extra="forbid"`.
2. **Interpretation** (`app/llm_interpreter.py`). Only the notes and the battery capacity are sent to
   the model.
3. **Guardrails** (`app/guardrails.py`). Independent deterministic validation; any failure rejects the
   whole response and the interpreter moves on to the next provider attempt.
4. **Compilation** (`app/replay.compile_directive_effects`). Validated entries merge into one
   per-hour effect bundle. Where several directives touch the same hour, the **strictest** value
   wins — lowest `factor`, highest reserve, lowest grid cap, union of no-charge and no-discharge
   hours — so multiple simultaneous directives all remain active and the optimizer must satisfy
   their intersection.
5. **Optimization** (`app/optimizer.py`). A linear program over 120 continuous variables, solved by
   the bundled CBC solver.
6. **Normalization** (`app/optimizer.optimize`). Simultaneous charge/discharge is discouraged with a
   tiny `1e-6` per-unit penalty (≈ 8 orders of magnitude below the judge's `0.01 BDT` tolerance);
   the `E[h]` trajectory is rebuilt arithmetically so balance holds by construction; values are
   rounded once at the response boundary.
7. **Replay** (`app/replay.replay_check`). Everything is recomputed from the plan that is about to
   be returned. On any violation a controlled error is raised before the response is built.

## 5. API specification

### `GET /health`

```bash
curl -s http://localhost:8000/health
```
```json
{"status":"ok"}
```

### `POST /optimize-energy`

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario_id": "GRID-101",
    "operator_notes": [
      "Solar output will drop to about 20% from 1 PM to 3 PM.",
      "Do not charge the battery between 2 PM and 4 PM.",
      "The cafeteria menu changes tomorrow."
    ],
    "hours": [
      {"hour": 0,  "demand_kwh":  90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      "... 22 more entries (hours 1..23, no duplicates) ...",
      {"hour": 23, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 8}
    ],
    "battery": {
      "capacity_kwh": 500,
      "initial_energy_kwh": 200,
      "minimum_energy_kwh": 50,
      "max_charge_kwh_per_hour": 100,
      "max_discharge_kwh_per_hour": 100
    }
  }'
```

A runnable request body is available in `tests/sample-01.json`.

**Response**

```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
     "explanation": "Solar drops to 20% of normal during the stated window."},
    {"note_index": 1, "applies": true, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]},
     "explanation": "Battery charging is unavailable in this window."},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "This note does not affect today's energy schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0,
     "battery_action": "idle", "battery_kwh": 0.0,
     "battery_energy_after_kwh": 200.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "applies solar reduction(s); respects no-charge window(s); ..."
}
```

**Status codes**

| Code | Meaning |
|---|---|
| `200` | Successful health or optimization response |
| `400` | Malformed JSON, wrong type, or schema-invalid request |
| `422` | Optimizer could not produce a feasible plan (last-resort fall-back only) |
| `500` | Controlled internal error — generic body, no stack trace |
| `503` | *(reserved — never reached on the current deployment)* |

Errors return `{"detail": "..."}` with no stack trace and no configuration values.

## 6. LLM directive interpretation

The model is the only non-deterministic stage. It is given one job: turn each note into one
structured directive.

**What is sent** — the operator notes, and `battery_capacity_kwh`. Nothing else.

The capacity scalar is required because operators express reserves as a percentage of capacity
("keep at least 50% of the battery capacity in reserve"), which cannot be resolved into an absolute
kWh figure without it. Demand, solar and tariff series are never sent: the model has no use for them
and they would only add latency and hallucination surface.

**What comes back** — strictly this shape, validated by the guardrails:

```json
{"interpretations": [
  {"note_index": 0, "applies": true, "directive_type": "no_charge_window",
   "structured_adjustment": {"hours": [14, 15]}, "explanation": "..."}
]}
```

**Supported directive types**:

| `directive_type` | `structured_adjustment` | Effect |
|---|---|---|
| `solar_reduction` | `{"hours": [...], "factor": n}` | `effective_solar[h] = solar[h] * factor` |
| `minimum_battery_reserve` | `{"hours": [...], "minimum_energy_kwh": n}` | `E_after[h] >= max(base_min, n)` |
| `no_charge_window` | `{"hours": [...]}` | `charge[h] = 0` |
| `no_discharge_window` | `{"hours": [...]}` | `discharge[h] = 0` |
| `max_grid_window` | `{"hours": [...], "max_grid_kwh": n}` | `grid[h] <= n` |
| `no_op` | `null` | No change to the model |

The system prompt emphasizes, with worked examples: start-inclusive / end-exclusive windows;
`factor` as the *remaining* fraction; one interpretation per note in order; `no_op` for anything
that does not change today's electricity schedule; percentage-of-capacity reserve conversion; and an
explicit instruction never to invent scenario data or directive types.

Paraphrase robustness comes from the model plus a synonym section in the prompt, **not** from
phrase matching. No public sample phrase, scenario id or number appears anywhere in `app/`.

## 7. Groq primary model

Groq is the primary provider, chosen for its low inference latency, which is what makes the p95 ≤ 5s
target comfortable.

- Model is read from `GROQ_MODEL` — nothing is tied to a single hard-coded model name. The default
  is `llama-3.3-70b-versatile`; any Groq chat model with JSON-object output works.
- `temperature=0` for maximum determinism.
- `response_format={"type": "json_object"}` for structured output.
- Per-attempt timeout from `LLM_TIMEOUT_S` (default `12`); retry policy lives in the interpreter.
- Up to `LLM_MAX_RETRIES` attempts per provider (default `1`) before falling back.

## 8. Gemini fallback

`gemini-2.0-flash` is the fallback, used only when Groq fails: timeout, connection failure,
rate limit (429), any other provider error, or output that fails the guardrails after the retry.

- One attempt only after Groq exhausts its retries. There is no endless retry loop.
- `response_mime_type="application/json"`, `temperature=0`, timeout from `LLM_TIMEOUT_S`.

If both providers fail the deterministic regex interpreter (`app/llm_interpreter.py`) takes over —
it recognizes the most common phrasings including word-numbers ("two hundred kWh"), word fractions
("one-fifth"), and end-exclusive time windows ("from 1 until 3"). If even the regex cannot map a
note to a directive, the entry falls back to `no_op`. The service always returns `200` for any
well-formed request.

## 9. Deterministic guardrails

`app/guardrails.py` treats model output as hostile. It runs **independently of Pydantic** and
rejects — it never repairs, because a silently repaired directive is indistinguishable from an
invented one.

Checks performed:

| # | Check | # | Check |
|---|---|---|---|
| 1 | Output is a list | 10 | Hours are unique |
| 2 | Exactly one entry per note | 11 | Hours are in ascending order |
| 3 | `note_index` is an integer in range | 12 | Hours list is non-empty |
| 4 | No duplicate `note_index` | 13 | Numeric values are finite (no NaN/Inf) |
| 5 | No missing `note_index` | 14 | `0 <= factor <= 1` |
| 6 | Entries are in ascending order | 15 | `minimum_energy_kwh >= 0` |
| 7 | `directive_type` is one of the six | 16 | `minimum_energy_kwh <= battery capacity` |
| 8 | `no_op` ⇒ `applies = false` **and** `structured_adjustment = null` | 17 | `max_grid_kwh >= 0` |
| 9 | Non-`no_op` ⇒ `applies = true` and the **exact** key set for that type | 18 | No key outside the required shape (so no invented parameters) |

A guardrail failure is treated exactly like a provider failure: the interpreter moves on to the
next attempt or the fallback provider.

**Merging.** `compile_directive_effects` combines validated entries. Where several directives
cover the same hour the strictest value wins — lowest `factor`, highest reserve, lowest grid cap,
union of no-charge and no-discharge hours — so multiple simultaneous directives all remain active
and the optimizer must satisfy their intersection.

## 10. Optimization model

A plain linear program solved with **CBC**, the open-source COIN-OR solver bundled with PuLP. No
binary variables, no heuristics, no custom search.

**Decision variables**, per hour `h` (120 continuous variables total):

```
grid[h], solar_used[h], charge[h], discharge[h], battery_energy_after[h]   — all >= 0
```

**Objective** — the official one, and nothing else:

```
minimize  SUM over h of  grid[h] * tariff_bdt_per_kwh[h]
```

No battery-degradation term, carbon term, peak-shaving term or solar-preference term is added. Those
would change the optimum away from the one the judge computes.

**Variable bounds**

```
0 <= solar_used[h] <= effective_solar[h]
0 <= charge[h]     <= max_charge_kwh_per_hour      (0 in a no_charge_window)
0 <= discharge[h]  <= max_discharge_kwh_per_hour   (0 in a no_discharge_window)
0 <= grid[h]       <= max_grid_kwh                 (unbounded otherwise)
max(minimum_energy_kwh, directive_reserve[h]) <= battery_energy_after[h] <= capacity_kwh
```

Because directives are expressed as **bounds and linear constraints** rather than post-processing,
the solver optimizes within the feasible region rather than being corrected afterwards.

**Soft fallback.** If the directive set is collectively infeasible (e.g. a grid cap too low to
serve peak demand), the LP is re-solved with the conflicting soft caps (`max_grid_window`,
`minimum_battery_reserve`) replaced by penalized slack variables. The full directive list is
still returned in the response (the judge requires one entry per input note); the replay validator
skips softened directives but still checks the rest.

**Avoiding simultaneous charge and discharge.** A binary on/off variable per hour would turn this
into a MIP for no benefit: with no round-trip losses, charging and discharging in the same hour is
never profitable, only degenerate. The LP stays pure; the tiny `1e-6` per-unit penalty on both
`charge` and `discharge` removes the trivial degeneracy in the solver.

## 11. Energy balance

For every hour, exactly:

```
grid_kwh[h] + solar_used_kwh[h] + battery_discharge_kwh[h]
    = demand_kwh[h] + battery_charge_kwh[h]
```

Asserted as an equality constraint in the LP, then rebuilt arithmetically during output
normalization so it holds by construction in the returned numbers, then checked a third time by
the replay validator. Surplus solar is curtailed; grid export is not part of this challenge.

## 12. Battery model

```
h = 0 :  battery_energy_after[0] = initial_energy_kwh + charge[0] − discharge[0]
h > 0 :  battery_energy_after[h] = battery_energy_after[h−1] + charge[h] − discharge[h]
```

Bounds: `minimum_energy_kwh <= battery_energy_after[h] <= capacity_kwh`, raised for hours covered by
a `minimum_battery_reserve` directive. Rate limits are enforced per hour.

**End-of-day neutrality** is a hard equality constraint:

```
battery_energy_after[23] = initial_energy_kwh
```

The starting charge may shift energy between hours but can never be consumed as a free one-time
source.

`battery_action` semantics: `charge` and `discharge` both carry a non-negative magnitude in
`battery_kwh`, distinguished by the action; `idle` always has `battery_kwh = 0`.

## 13. Directive application

Applied deterministically **after** interpretation and **only** via the validated directive dicts:

| Directive | Applied as |
|---|---|
| `solar_reduction` | Upper bound on `solar_used[h]` = `solar[h] * factor` |
| `minimum_battery_reserve` | Lower bound on `battery_energy_after[h]` = `max(base, directive)` |
| `no_charge_window` | Upper bound on `charge[h]` = 0 |
| `no_discharge_window` | Upper bound on `discharge[h]` = 0 |
| `max_grid_window` | Upper bound on `grid[h]` = `max_grid_kwh` |
| `no_op` | Nothing |

All applicable directives stay active simultaneously — the optimizer satisfies their intersection.
Overlapping directives of the same type resolve to the strictest value, so none can overwrite
another. Organizer scoring scenarios are stated to be feasible; infeasible combinations are still
handled safely (the LP soft-falls-back as in §10) rather than crashing or returning an invalid
schedule.

## 14. Replay validation

`app/replay.py` is the last line of defence and deliberately shares no computation with the
optimizer. It takes the plan **as it will be returned** and re-derives everything:

1. 24 unique hours, 0–23 · 2. All values finite and non-negative · 3. `idle` ⇒ `battery_kwh = 0` ·
4. Effective solar recomputed from the base solar and the directives · 5. `solar_used <= effective_solar` ·
6. Energy balance every hour · 7. Charge rate limit · 8. Discharge rate limit ·
9. Battery transition replayed hour by hour · 10. `E_after >= max(base reserve, directive reserve)` ·
11. `E_after <= capacity` · 12. No charging in a no-charge window · 13. No discharging in a
no-discharge window · 14. Grid cap per hour · 15. End-of-day neutrality · 16. `total_grid_kwh`
recomputed · 17. `total_cost_bdt` recomputed · 18. `peak_grid_kwh` recomputed.

**If replay fails, a controlled error is raised before the response is built.** An invalid schedule
scores zero for that case and risks the reliability score; a controlled error costs only that one
case.

The tolerance used internally (`JUDGE_TOL`, default `1e-4`) is far tighter than the judge's `0.01`,
so the service fails before the judge would.

**Numeric precision.** Full precision is kept throughout the solver and normalization. Rounding
happens once, at the response boundary, to four decimal places — tight enough to avoid
floating-point artifacts in the JSON, loose enough that no rounding can disturb a `0.01` tolerance.
Totals are computed from the **rounded** plan, so `total_grid_kwh`, `total_cost_bdt` and
`peak_grid_kwh` always agree exactly with `hourly_plan`, which the judge treats as the source of
truth. The final replay runs on the rounded values.

**`plan_summary`** is generated deterministically from the directives and the plan
(`_summarize_plan`). The LLM is never asked to write it — a generated summary could make claims
the schedule does not support.

## 15. Project structure

```
app/
├── main.py                FastAPI app, endpoints, controlled error handling
├── schemas.py             Pydantic v2 models (strict typing, extra="forbid")
├── llm_interpreter.py     Groq → Gemini → regex → no_op pipeline + system prompt
├── guardrails.py          Deterministic validation + canonicalization
├── optimizer.py           PuLP LP construction, CBC solve, normalization, fallback
└── replay.py              Independent replay verification (shared with main + tests)

tests/
├── sample-01.json         A runnable request body for smoke testing
├── test_health.py         GET /health round-trip
├── test_judge.py          Public sample cases + directive semantics + replay
├── test_paraphrases.py    29 paraphrased notes covering all applicable directive types
├── test_malformed.py      15 malformed-input cases — every one must return 4xx
└── test_smoke.py          Bypasses the LLM; runs the LP solver only

Dockerfile · .dockerignore · .env.example · docker-compose.yml · requirements.txt · .gitignore
```

## 16. Environment variables

No secret has a default. Copy `.env.example` to `.env` and fill it in.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | no | `groq` | `groq`, `gemini`, or `off` (skip the LLM path entirely) |
| `GROQ_API_KEY` | yes (primary) | — | Groq credential from <https://console.groq.com> |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` | Groq model id |
| `GEMINI_API_KEY` | yes (fallback) | — | Gemini credential from <https://ai.google.dev> |
| `GEMINI_MODEL` | no | `gemini-2.0-flash` | Gemini model id |
| `LLM_TIMEOUT_S` | no | `12` | Per-attempt HTTP timeout (seconds) |
| `LLM_MAX_RETRIES` | no | `1` | Retries per provider before falling back |
| `PORT` | no | `8000` | HTTP listen port |
| `LOG_LEVEL` | no | `INFO` | Log verbosity |

At least one provider key must be set for full robustness. With neither set, the service uses the
deterministic regex interpreter and ultimately `no_op` for every note.

## 17. Local setup

From a clean environment:

```bash
git clone https://github.com/darkEye-2021831014/BUP-Hackathon.git
cd BUP-Hackathon/API

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (and optionally GEMINI_API_KEY)
```

Requires Python 3.11+.

## 18. Running the service

```bash
set -a && source .env && set +a        # load environment variables
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d @tests/sample-01.json | jq .
```

Interactive docs are at `http://localhost:8000/docs`.

## 19. Running tests

All tests run in-process against the FastAPI app via `httpx.AsyncClient` + `ASGITransport`. No
network access is required, no live LLM credentials are needed (the deterministic regex interpreter
is used automatically when no keys are set).

```bash
.venv/bin/python tests/test_health.py
.venv/bin/python tests/test_judge.py
.venv/bin/python tests/test_paraphrases.py
.venv/bin/python tests/test_malformed.py
.venv/bin/python tests/test_smoke.py
```

| File | Covers |
|---|---|
| `test_health.py` | `GET /health` round-trip |
| `test_judge.py` | Public sample cases + directive semantics + replay (balance, battery, end-of-day neutrality, totals) |
| `test_paraphrases.py` | 29 paraphrased notes covering all 5 applicable directive types + distractors |
| `test_malformed.py` | 15 malformed-input cases — every one must return 4xx, never 5xx with a stack trace |
| `test_smoke.py` | Bypasses the LLM; runs the LP solver only on the public cases |

The replay validator used in `test_judge.py` is the **same function** (`app.replay.replay_check`)
the API itself uses for its final validator — guarantees the local tests and the running service
agree on what "correct" means.

Prompt semantics — start-inclusive/end-exclusive windows, `factor` as the *remaining* fraction, and
`no_op` — are asserted by the model plus the synonym guidance in the system prompt rather than by
phrase matching, which is why no public sample phrase appears anywhere in `app/`.

## 20. Public sample tests

The bundled public-sample regression is `tests/test_judge.py`. It exercises every scenario
end-to-end through the HTTP API, replays each returned schedule, checks every GridWise rule,
compares the reported directives against the published reference semantics, and compares the
recalculated cost against the published reference optimum.

```bash
.venv/bin/python tests/test_judge.py
```

Expected output (one line per case):

```
SAMPLE-01  OK  cost=38365.0  grid=2692.5  peak=187.5
SAMPLE-02  OK  cost=42885.0  grid=2915.0  peak=180.0
...
PASS: 10/10  FAIL: 0/10
```

Schedules are never compared byte-for-byte — equivalent optima are accepted, as the Problem
Statement specifies. The replay validator also accepts the organizer's own reference schedules,
which confirms it is not over-strict.

## 21. Docker

**Pull and run** the published image (no build step required):

```bash
docker pull voideye/gridwise:latest

docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  voideye/gridwise:latest

curl -s http://localhost:8000/health   # {"status":"ok"}
```

**Push** (when re-publishing from source):

```bash
docker push voideye/gridwise:latest
```

**Build and run locally** (no registry needed):

```bash
docker build -t gridwise:1.0.0 .

docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  gridwise:1.0.0
```

| Property | Value |
|---|---|
| Source | <https://github.com/darkEye-2021831014/BUP-Hackathon> |
| Published image | [`voideye/gridwise:latest`](https://hub.docker.com/r/voideye/gridwise) |
| Base image | `python:3.11-slim` (multi-stage build) |
| Exposed port | `8000` (override with `PORT`) |
| Bind address | `0.0.0.0` |
| User | non-root (`gridwise`, uid `1000`) |
| Secrets | **none baked in** — passed at run time with `-e` |
| Health check | built-in `HEALTHCHECK` against `/health` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1` |

`.dockerignore` excludes `.env`, tests and caches, so no credential can reach the image.

## 22. Deployment

The service is a single stateless container with no database, queue or external dependency beyond
the LLM provider APIs, so it runs unchanged on any container platform.

Deployment requirements met:

- Both endpoints are publicly reachable over HTTP/HTTPS with **no** authentication, VPN, dashboard
  login or manual approval.
- Binds `0.0.0.0` on a configurable `PORT` (most platforms inject `PORT` automatically).
- `/health` is ready well inside 60 seconds of start — there is no model loading or warm-up.
- Secrets are supplied as platform environment variables, never committed and never in the image.
- Stateless, so horizontal scaling and restarts are safe.

Set `GROQ_API_KEY` (and optionally `GEMINI_API_KEY`) in the platform's secret manager, deploy the
image, and confirm `GET /health` and one `POST /optimize-energy` **from outside** your development
environment before submitting.

## 23. Performance

Targets: p95 ≤ 5s, hard timeout 30s.

| Stage | Typical | Note |
|---|---|---|
| Pydantic validation | < 1 ms | Strict typing |
| **LLM interpretation** | 0.3–1.5 s | The dominant cost; Groq is chosen for this reason |
| Guardrails | < 1 ms | Pure Python |
| **PuLP LP solve** | 3–10 ms | 120 variables, ~50 constraints |
| Normalization + replay | < 1 ms | |
| **Total** | **≈ 0.4–1.6 s** | Comfortably inside p95 ≤ 5s |

Design choices that keep latency down:

- Only the notes (plus one scalar) go to the model — a prompt of a few hundred tokens instead of a
  24-hour data series.
- Exactly **one** LLM call in the normal path. All notes are interpreted in a single request.
- Groq timeout (12 s) + one retry + Gemini timeout (12 s) bounds the absolute worst case inside the
  30 s limit.
- Provider clients are constructed once per process and reused, so no per-request connection setup.
- LP rather than MIP — no branch-and-bound.
- No database, cache, queue or cold-start-heavy dependency.

## 24. Security

- **No secret is ever hard-coded or committed.** All credentials come from environment variables;
  `.env` is in both `.gitignore` and `.dockerignore`.
- **No secret in the image.** Keys are passed at run time with `-e`.
- **No secret in logs.** No provider exception message is logged; provider exceptions are logged by
  exception *type* only, so an SDK cannot leak a key through an error string.
- **No secret or stack trace in responses.** Every error path returns a fixed, generic
  `{"detail": "..."}`. A catch-all handler converts any unexpected exception into a `500` with
  server-side logging only.
- **Non-root container user.**
- **Synthetic data only.** The service is stateless and stores nothing.

## 25. Failure handling

| Failure | Response |
|---|---|
| Malformed JSON | `400`, generic detail |
| Schema-invalid request (missing field, wrong type) | `400` with the offending field paths (values never echoed) |
| Groq timeout / 429 / connection error | Retry once, then fall back to Gemini |
| Groq output malformed or guardrail-rejected | Retry once, then fall back to Gemini |
| Gemini timeout / malformed / guardrail-rejected | Fall back to the deterministic regex interpreter |
| Both providers unavailable | Regex interpreter + `no_op` for any unmapped note — `200` |
| Directives collectively infeasible | Soft-fallback LP with penalized slack variables — `200` |
| Solver error | `422`, controlled message |
| **Replay validation fails** | Controlled error, schedule **discarded** before response |
| Unexpected exception | `500`, logged server-side, generic body |

The guiding rule: **never return a schedule that has not been independently verified, and never
invent a directive to paper over a provider failure.** Both would score zero on the affected case
while looking superficially successful.

## 26. Limitations

- **LLM availability caps service availability.** If both providers are unreachable the regex
  interpreter covers the most common phrasings, but uncommon paraphrases fall through to `no_op`.
  This is deliberate — the challenge mandates LLM interpretation, and a phrase-matching-only
  fallback would be non-compliant — but it does mean the LLM providers are a soft dependency for
  full paraphrase coverage.
- **Interpretation is not deterministic across runs.** Temperature is 0 and the prompt is fixed, but
  an identical note can occasionally yield a different directive. Everything downstream of the
  guardrails *is* fully deterministic: the same canonical directives always produce the same
  schedule.
- **Battery capacity is sent to the model.** One scalar, needed to resolve percentage-of-capacity
  reserves into kWh. No demand, solar or tariff data is ever sent, and the guardrails still verify
  every returned number independently.
- **Ties between equally optimal schedules are arbitrary.** CBC picks one optimal vertex. Cost is
  always optimal; the specific hourly actions and the resulting `peak_grid_kwh` may differ from the
  reference plan. The Problem Statement explicitly permits this.
- **No round-trip battery efficiency.** The Problem Statement defines lossless transitions, so none
  is modelled.
- **Single-day horizon only**; exactly 24 hours, no rolling window.

## 27. Future improvements

- A small deterministic pre-parser for unambiguous time expressions, used only to *cross-check* the
  LLM's hours and trigger a retry on disagreement — raising interpretation accuracy without
  replacing the model.
- Self-consistency: two low-temperature interpretations, accepted only when the canonical directives
  agree.
- An in-process cache keyed on the note text, cutting repeat-note latency to near zero across a
  hidden test set that reuses phrasings.
- A third provider tier for redundancy beyond Groq and Gemini.
- Explicit modelling of round-trip efficiency and degradation cost, if a future round specifies them.
- Prometheus metrics for per-stage latency and provider fallback rate.

## 28. Dependencies and credits

| Dependency | Role | Licence |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP framework | MIT |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server | BSD-3 |
| [Pydantic v2](https://docs.pydantic.dev/) | Request/response validation (strict typing) | MIT |
| [PuLP](https://coin-or.github.io/pulp) | LP modeling | MIT |
| [CBC](https://github.com/coin-or/Cbc) | LP solver (bundled with PuLP) | EPL 2.0 |
| [httpx](https://www.python-httpx.org/) | Test + harness HTTP client, LLM provider client | BSD-3 |
| [groq](https://github.com/groq/groq-python) | Primary LLM SDK | Apache-2.0 |
| [google-genai](https://github.com/googleapis/python-genai) | Fallback LLM SDK | Apache-2.0 |

**Models:** Groq (`llama-3.3-70b-versatile` by default, configurable) as primary; Google
`gemini-2.0-flash` as fallback. Both are used solely for operator-note interpretation.

**Challenge materials:** Problem Statement, Participant Guide & Evaluation Rubric, and Public
Sample Cases provided by the Department of CSE, Bangladesh University of Professionals, in
association with Poridhi, for BUP CSE Fest 2026.

**Repository:** <https://github.com/darkEye-2021831014/BUP-Hackathon>
**Docker image:** [`voideye/gridwise:latest`](https://hub.docker.com/r/voideye/gridwise) ·
publish with `docker push voideye/gridwise:latest`

An AI coding assistant was used during development. The architecture, the LLM / guardrail /
optimizer separation, the optimization model and the validation strategy are the team's own design.
