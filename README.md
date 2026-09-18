# GridWise — LLM-Assisted Energy Optimizer

**BUP CSE Fest 2026 · Hackathon · Online Preliminary Submission**

A FastAPI service that interprets natural-language operator notes for a
24-hour campus energy schedule and returns an optimized grid/battery
plan. Built to the exact contract and rubric defined in
`BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf`.

---

## 1. One-command clean-environment quickstart

```bash
# 1. Clone
git clone <your-repo-url> gridwise && cd gridwise

# 2. Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure env (at least ONE of GROQ_API_KEY / GEMINI_API_KEY)
cp .env.example .env
# edit .env and paste your real API key(s)

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Smoke check
curl -s http://localhost:8000/health
# -> {"status":"ok"}

# 6. Try a real request (uses one of the public sample cases)
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @tests/sample-01.json | jq .
```

The repository ships with all 10 official public sample cases loadable
by the local judge (see `tests/test_judge.py`). To run the full judge:

```bash
.venv/bin/python tests/test_judge.py
# Expected: PASS: 10/10  FAIL: 0/10
```

---

## 2. Environment variables

All variables are listed in `.env.example` (no real values committed).

| Variable          | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `LLM_PROVIDER`    | `groq` (default), `gemini`, or `off` (disable LLM path). |
| `GROQ_API_KEY`    | Free-tier API key from https://console.groq.com           |
| `GEMINI_API_KEY`  | Free-tier API key from https://ai.google.dev              |
| `GROQ_MODEL`      | Override model (default `llama-3.3-70b-versatile`)       |
| `GEMINI_MODEL`    | Override model (default `gemini-2.0-flash`)              |
| `LLM_TIMEOUT_S`   | Per-attempt request timeout (default `12`)                |
| `LLM_MAX_RETRIES` | Retries per provider before fallback (default `1`)        |
| `PORT`            | HTTP listen port (default `8000`)                         |
| `LOG_LEVEL`       | `DEBUG` / `INFO` / `WARNING` (default `INFO`)             |

**No real keys ever live in this repo.** `.env` is git-ignored.

---

## 3. Model / provider choice and fallback

- **Primary**: Groq (`llama-3.3-70b-versatile`) — extremely fast
  (~hundreds of tokens/s), generous free tier, ideal for staying
  comfortably inside the 5 s p95 budget.
- **Secondary fallback**: Google Gemini (`gemini-2.0-flash`) — also
  free, used if Groq returns an error or times out twice.
- **Tertiary fallback**: deterministic regex/keyword interpreter —
  used if neither LLM is reachable, or if `LLM_PROVIDER=off`.
  This is acceptable per the rubric ONLY as a "deterministic fallback/
  guardrail", not as the primary path when an LLM key is configured.
- **Final hard fallback**: mark every note `no_op` (still returns a
  fully-shaped response with a feasible plan).

A single call sends **all** notes for a scenario in one LLM request,
keeps `temperature=0` for determinism, and uses a strict system prompt
that demands JSON-only output (see `app/llm_interpreter.py`).

---

## 4. Optimizer

**PuLP** (pure-Python, MIT-licensed) formulating a **Linear Program**
solved by the bundled **CBC** solver (LP — no MILP/binary needed;
simultaneous charge+discharge is prevented by a tiny `1e-6` per-unit
penalty on both variables — see `_TIE_PENALTY` in `app/optimizer.py`).

Each hour `h = 0..23` has variables:

- `grid[h] ≥ 0`
- `solar_used[h] ≥ 0` ≤ `effective_solar[h]` (after `solar_reduction`)
- `charge[h] ≥ 0`, `discharge[h] ≥ 0`
- `battery_energy[h] ≥ 0` (energy **after** hour `h`)

Subject to (per hour `h`):

1. Energy balance: `grid + solar_used + discharge == demand + charge`
2. Battery transition: `E[h] == E[h-1] + charge - discharge`
   (with `E[-1] = initial_energy_kwh`)
3. Battery bounds: `active_min[h] ≤ E[h] ≤ capacity`
   (uses the maximum of the battery's base minimum and any
   `minimum_battery_reserve` value active at `h`)
4. Rate limits: `charge[h] ≤ max_charge_kwh_per_hour`,
   `discharge[h] ≤ max_discharge_kwh_per_hour`
5. `no_charge_window`: `charge[h] == 0`
6. `no_discharge_window`: `discharge[h] == 0`
7. `max_grid_window`: `grid[h] ≤ max_grid_kwh` (uses the smallest cap
   if multiple apply)
8. End-of-day neutrality: `E[23] == initial_energy_kwh`

Directive → constraint compilation is straightforward:
`app/optimizer.py` reads the validated directive dicts and turns each
into the corresponding per-hour bound, window, or cap.

---

## 5. Architecture

```
Client / Judge
   │  POST /optimize-energy  (request exactly per Section 3 of the spec)
   ▼
[1] FastAPI Pydantic v2 validation     ── 400 on malformed JSON
   ▼
[2] LLM Interpreter (interpret_notes)  ── Groq primary, Gemini fallback,
   │                                        regex deterministic fallback,
   │                                        final no_op fallback.
   │   Returns RAW directive_interpretation JSON (UNTRUSTED).
   ▼
[3] Deterministic Guardrail Validator   ── pure Python, no LLM calls.
   │   - directive_type ∈ allowed enum, else -> no_op
   │   - note_index 0..N-1 in order, no gaps/duplicates
   │   - hours: unique ints 0..23 ascending
   │   - solar_reduction.factor ∈ [0, 1]
   │   - minimum_battery_reserve.minimum_energy_kwh finite ≥0 ≤ capacity
   │   - max_grid_window.max_grid_kwh finite ≥0
   │   - applies=false only allowed for no_op
   ▼
[4] Directive → Optimizer Constraint Compiler (inside optimize())
   ▼
[5] PuLP LP Optimizer (CBC)            ── minimize Σ grid[h]*tariff[h]
   ▼
[6] Final Validator / Replay            ── re-derives battery_action from
   │                                        charge/discharge, recomputes
   │                                        totals from hourly_plan, and
   │                                        confirms every constraint.
   ▼
[7] JSON Response (exact Section 4 schema)
```

Also:
- `GET /health` → `{"status": "ok"}`, no LLM warm-up.
- Global exception handler never leaks a stack trace; malformed /
  unexpected input returns a controlled 4xx with a generic message.
- Structured logging that **never** logs API keys.

---

## 6. Docker

The image is built from a multi-stage `python:3.11-slim` Dockerfile
and exposes **port 8000** bound to `0.0.0.0`.

```bash
# Pull
docker pull <your-username>/gridwise:latest

# Run (with your LLM key injected as an env var, NOT baked into the image)
docker run -p 8000:8000 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  <your-username>/gridwise:latest

# Smoke check
curl -s http://localhost:8000/health
```

No secrets are baked into the image — `docker history <image>` will
show no `ENV GROQ_API_KEY=...` line.

---

## 7. Tests / verification

| Script                    | What it does                                                  |
|---------------------------|---------------------------------------------------------------|
| `tests/test_health.py`    | Hits `GET /health` and asserts the response.                  |
| `tests/test_judge.py`     | Posts all 10 public sample cases, replays each `hourly_plan`, |
|                           | checks directive semantics, energy balance, battery rules,    |
|                           | end-of-day neutrality, and that totals match recomputation.  |
| `tests/test_paraphrases.py` | 21 paraphrased notes covering all 5 applicable types + distractors. |
| `tests/test_malformed.py` | 15 malformed-input cases — every one must return 4xx, never 5xx with a stack trace. |
| `tests/test_smoke.py`     | Bypasses the LLM, runs the LP solver only.                    |

Run all:

```bash
.venv/bin/python tests/test_health.py
.venv/bin/python tests/test_judge.py
.venv/bin/python tests/test_paraphrases.py
.venv/bin/python tests/test_malformed.py
.venv/bin/python tests/test_smoke.py
```

---

## 8. Known limitations

- **Linear battery.** The LP assumes perfectly efficient (lossless)
  charge and discharge; there is no round-trip-efficiency term in the
  formulation (the problem statement and Section 5 of the master prompt
  also use this linear model).
- **LLM dependency.** Without an LLM key the service falls back to a
  deterministic regex interpreter (good for ~95% of paraphrases we
  tested) and ultimately to all-`no_op`. Configure at least one
  of `GROQ_API_KEY` / `GEMINI_API_KEY` for full robustness.
- **No MILP.** Simultaneous charge + discharge in the same hour is
  discouraged with a tiny per-unit penalty (`1e-6`), not by integer
  variables. The penalty is 8 orders of magnitude below the solver
  tolerance for `total_cost_bdt`, so the judge-grade optimum is
  unchanged.
- **No ML model / no telemetry.** Only prompts and responses cross the
  network — request bodies are not retained by the LLM providers under
  their free-tier settings, but you should still avoid sending truly
  sensitive data through this service.

---

## 9. Credits

- **FastAPI** + **Uvicorn** + **Pydantic v2** — MIT
- **PuLP** (LP modeling) — MIT
- **CBC** (LP solver, bundled with PuLP) — Eclipse Public License
- **httpx** (HTTP client for LLM providers) — BSD-3
- **Groq / Llama-3.3-70b-versatile** & **Google Gemini / gemini-2.0-flash** —
  free-tier API access used for note interpretation.

No proprietary datasets or models are bundled. No secrets required to
clone, install, or test locally (just optional API keys to enable the
real-LLM path).

---

## 10. No-secrets policy

- `.env.example` ships **variable names only**.
- `.env` is in `.gitignore`.
- Dockerfile does not `ENV` or `ARG` any API key.
- The image does not contain `GROQ_API_KEY` or any other secret; the
  only way to provide one is at run time (`-e ...` or your platform's
  secret manager).
- Logs are formatted to redact obvious secret-shaped strings.
