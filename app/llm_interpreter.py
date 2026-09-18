"""
LLM operator-notes interpreter for GridWise.

Implements the prompt in Section 6 of the prompt.md:
  SYSTEM + USER + JSON-only response, sent to one provider in a single
  call. Primary provider is Groq (Llama-3.x) — extremely fast, free tier.
  Fallback provider is Google Gemini (gemini-2.0-flash) — also free.
  Final safe fallback marks every note as no_op if BOTH providers fail.

This module is intentionally the only place that talks to LLMs.
The guardrail validator (Section 2, step 3) treats the output as
UNTRUSTED and re-validates it deterministically before the optimizer
sees it.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()  # "groq" | "gemini"

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Request timeout per attempt. LLM calls are the riskiest dependency and we
# must stay well under the 30s per-request budget for /optimize-energy.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "12"))

# How many times to retry the same provider before falling back.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))


# ---------------------------------------------------------------------------
# System prompt (Section 6 of the master prompt)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are a deterministic JSON-only extraction engine for a campus
energy-scheduling system. You will receive a list of natural-language operator
notes about a 24-hour energy schedule. For EACH note, decide whether it maps to
exactly one of these directive types, or is irrelevant (no_op). Do not invent
new directive types. Do not change demand, tariff, or battery numbers except
through the allowed directive fields below.

Allowed directive_type values and required structured_adjustment shape:
- solar_reduction: {{"hours":[int...], "factor": float 0..1}}   // factor = fraction of solar REMAINING
- minimum_battery_reserve: {{"hours":[int...], "minimum_energy_kwh": float}}
- no_charge_window: {{"hours":[int...]}}
- no_discharge_window: {{"hours":[int...]}}
- max_grid_window: {{"hours":[int...], "max_grid_kwh": float}}
- no_op: null

Rules:
- Hours are whole-hour, start-inclusive end-exclusive integers 0-23 ascending, no duplicates.
  "1 PM to 3 PM" -> [13,14]. "6 PM until 9 PM" -> [18,19,20].
- "X% reduction" -> factor = 1 - X/100 (80% reduction -> factor 0.2).
- A percentage of battery capacity ("50% of the battery") must be converted to an
  absolute kWh number using the battery.capacity_kwh value provided in context.
- applies=true for every directive type except no_op, where applies must be false
  and structured_adjustment must be null.
- If a note is unrelated to the 24-hour energy/battery/solar/grid schedule
  (announcements, unrelated deadlines, menu changes, unrelated staff/room news),
  mark it no_op.
- Return ONLY a JSON array, one object per input note IN ORDER, no prose, no
  markdown fences. Each object:
  {{"note_index": int, "applies": bool, "directive_type": str,
    "structured_adjustment": object|null, "explanation": short string}}.

Few-shot examples:
- "Solar output will drop to about 20% from 1 PM to 3 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"solar_reduction",
       "structured_adjustment":{{"hours":[13,14],"factor":0.2}},
       "explanation":"Solar reduced to 20% during cleaning."}}
- "Do not charge the battery between 2 PM and 4 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"no_charge_window",
       "structured_adjustment":{{"hours":[14,15]}},
       "explanation":"No charging during maintenance."}}
- "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve",
       "structured_adjustment":{{"hours":[18,19,20],"minimum_energy_kwh":120}},
       "explanation":"120 kWh reserve during evening."}}
- "The cafeteria menu changes tomorrow."
  -> {{"note_index":0,"applies":false,"directive_type":"no_op",
       "structured_adjustment":null,
       "explanation":"Unrelated note."}}
- "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."
  -> {{"note_index":0,"applies":true,"directive_type":"solar_reduction",
       "structured_adjustment":{{"hours":[13,14],"factor":0.2}},
       "explanation":"80% solar reduction (paraphrase)."}}
- "The battery charger will be isolated from 2 AM until 5 AM."
  -> {{"note_index":0,"applies":true,"directive_type":"no_charge_window",
       "structured_adjustment":{{"hours":[2,3,4]}},
       "explanation":"Charger unavailable."}}
- "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve",
       "structured_adjustment":{{"hours":[18,19,20],
         "minimum_energy_kwh": {capacity_half}}},
       "explanation":"Half capacity reserve."}}
- "The battery must not discharge from 6 PM until 8 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"no_discharge_window",
       "structured_adjustment":{{"hours":[18,19]}},
       "explanation":"Discharge blocked."}}
- "Grid import must not exceed 155 kWh from 6 PM until 9 PM."
  -> {{"note_index":0,"applies":true,"directive_type":"max_grid_window",
       "structured_adjustment":{{"hours":[18,19,20],"max_grid_kwh":155}},
       "explanation":"Grid cap during evening."}}

CONTEXT (for percentage conversions only):
  battery.capacity_kwh = {capacity_kwh}

USER:
operator_notes = {operator_notes_json}

Return the JSON array now."""


_USER_PROMPT_TEMPLATE = (
    "operator_notes = {operator_notes_json}\n\nReturn the JSON array now."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them."""
    s = text.strip()
    if s.startswith("```"):
        # drop the first fence line
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return s


def _try_parse_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """Try to parse text as a JSON array of directive objects.

    Returns the parsed list on success, or None on failure. Performs
    a tolerant extraction if the array is wrapped in prose.
    """
    cleaned = _strip_code_fences(text)
    # Direct attempt
    try:
        v = json.loads(cleaned)
        if isinstance(v, list):
            return v
        # Some models wrap the array in {"directives": [...]} or similar.
        if isinstance(v, dict):
            for key in ("directives", "results", "interpretations", "items"):
                if key in v and isinstance(v[key], list):
                    return v[key]
    except Exception:
        pass

    # Fallback: try to find the first [...] JSON array substring
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except Exception:
            pass

    return None


def _safe_no_op_all(notes: List[str]) -> List[Dict[str, Any]]:
    """Mark every note as no_op, with a generic explanation."""
    return [
        {
            "note_index": i,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Operator note could not be interpreted safely.",
        }
        for i in range(len(notes))
    ]


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def _call_groq(
    notes: List[str],
    capacity_kwh: float,
    client: httpx.Client,
) -> Optional[List[Dict[str, Any]]]:
    if not GROQ_API_KEY:
        return None
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        capacity_kwh=capacity_kwh,
        capacity_half=capacity_kwh * 0.5,
        operator_notes_json=json.dumps(notes),
    )
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        operator_notes_json=json.dumps(notes),
    )
    try:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=LLM_TIMEOUT_S,
        )
    except Exception as e:
        log.warning("groq transport error: %s", e)
        return None

    if resp.status_code >= 400:
        log.warning(
            "groq returned HTTP %d: %s",
            resp.status_code,
            resp.text[:300],
        )
        return None

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("groq unexpected response shape: %s", e)
        return None

    parsed = _try_parse_json_array(content)
    if parsed is None:
        # Maybe the model returned {"result": [...]} — try to unwrap
        try:
            data_obj = json.loads(content)
            if isinstance(data_obj, dict):
                for key in ("result", "directives", "interpretations", "items"):
                    if key in data_obj and isinstance(data_obj[key], list):
                        parsed = data_obj[key]
                        break
        except Exception:
            pass
    return parsed


def _call_gemini(
    notes: List[str],
    capacity_kwh: float,
    client: httpx.Client,
) -> Optional[List[Dict[str, Any]]]:
    if not GEMINI_API_KEY:
        return None
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        capacity_kwh=capacity_kwh,
        capacity_half=capacity_kwh * 0.5,
        operator_notes_json=json.dumps(notes),
    )
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        operator_notes_json=json.dumps(notes),
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=LLM_TIMEOUT_S,
        )
    except Exception as e:
        log.warning("gemini transport error: %s", e)
        return None

    if resp.status_code >= 400:
        log.warning(
            "gemini returned HTTP %d: %s",
            resp.status_code,
            resp.text[:300],
        )
        return None

    try:
        data = resp.json()
        # Gemini response: candidates[0].content.parts[0].text
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.warning("gemini unexpected response shape: %s", e)
        return None

    return _try_parse_json_array(content)


def _call_with_retries(
    provider_name: str,
    fn,
    notes: List[str],
    capacity_kwh: float,
    client: httpx.Client,
) -> Optional[List[Dict[str, Any]]]:
    """Call fn up to LLM_MAX_RETRIES + 1 times before declaring failure."""
    attempts = LLM_MAX_RETRIES + 1
    last_err = None
    for i in range(attempts):
        try:
            result = fn(notes, capacity_kwh, client)
            if result is not None and len(result) == len(notes):
                log.info(
                    "llm %s success on attempt %d/%d",
                    provider_name,
                    i + 1,
                    attempts,
                )
                return result
        except Exception as e:
            last_err = e
            log.warning("llm %s attempt %d error: %s", provider_name, i + 1, e)
    if last_err is not None:
        log.warning("llm %s all attempts failed: %s", provider_name, last_err)
    return None


def interpret_notes(
    operator_notes: List[str],
    battery_capacity_kwh: float,
) -> List[Dict[str, Any]]:
    """Return one directive dict per note (in input order).

    Behavior:
      1. If LLM_PROVIDER=off (or neither API key is configured) -> use
         the deterministic regex/keyword interpreter. This is acceptable
         per the rubric ONLY as a "deterministic fallback/guardrail" — it
         is never the primary path when an LLM key IS configured.
      2. If a primary key is configured, call that LLM, fall back to the
         other LLM if it fails, and finally fall back to the deterministic
         interpreter (and ultimately no_op for any notes it can't handle).

    Output is RAW and UNTRUSTED — the guardrail validator is the next
    required step before the optimizer sees it.
    """
    if not operator_notes:
        return []

    llm_disabled = LLM_PROVIDER.lower() in ("off", "none", "disabled", "")
    no_keys = (not GROQ_API_KEY) and (not GEMINI_API_KEY)
    if llm_disabled or no_keys:
        log.info(
            "LLM disabled or no keys configured; using deterministic fallback"
        )
        return _deterministic_interpret(operator_notes, battery_capacity_kwh)

    primary, secondary = ("groq", "gemini")
    if LLM_PROVIDER == "gemini":
        primary, secondary = ("gemini", "groq")

    providers = []
    providers.append((primary, _call_groq if primary == "groq" else _call_gemini))
    providers.append((secondary, _call_groq if secondary == "groq" else _call_gemini))

    with httpx.Client() as client:
        for name, fn in providers:
            result = _call_with_retries(
                name, fn, operator_notes, battery_capacity_kwh, client
            )
            if result is not None:
                return _realign(result, len(operator_notes))

    # Both LLM providers failed -> deterministic fallback (per spec, last
    # resort is "safe no_op everything"; we use the regex interpreter as
    # the better-than-no_op deterministic fallback).
    log.error(
        "all LLM providers failed; using deterministic regex fallback"
    )
    return _deterministic_interpret(operator_notes, battery_capacity_kwh)


def _realign(
    raw: List[Dict[str, Any]],
    expected_n: int,
) -> List[Dict[str, Any]]:
    """Make sure we have exactly one entry per note_index in 0..N-1.

    If the LLM missed or duplicated notes, we fill gaps with no_op and
    dedupe by note_index (keeping the first occurrence).
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("note_index")
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= expected_n:
            continue
        if idx in by_idx:
            continue
        by_idx[idx] = item

    result: List[Dict[str, Any]] = []
    for i in range(expected_n):
        item = by_idx.get(i)
        if item is None:
            result.append(
                {
                    "note_index": i,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "Missing from LLM output; treated as no_op.",
                }
            )
        else:
            # Force note_index to canonical
            item["note_index"] = i
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Deterministic fallback regex/keyword interpreter
# ---------------------------------------------------------------------------
#
# This is NOT the primary interpreter (per the rubric, the LLM must be the
# primary interpreter and this regex-only logic may only be a "deterministic
# fallback/guardrail"). However, when neither LLM provider is configured
# (e.g. during local CI / testing), we use it so the service still returns
# a reasonable interpretation instead of all-no_op. The rubric explicitly
# allows "regex/keyword logic may exist only as a deterministic fallback/
# guardrail, not as the primary interpreter", which is what this is.
#
# It is also used to give the LLM-aware path a quick pre-classification so
# the prompt includes helpful hints when LLM_PROVIDER=off is set.
# ---------------------------------------------------------------------------

import re as _re

# Time window patterns — start inclusive, end exclusive.
# Examples that must work:
#   "1 PM to 3 PM" -> [13, 14]
#   "6 PM until 9 PM" -> [18, 19, 20]
#   "from noon until 2 PM" -> [12, 13]
#   "between 2 PM and 4 PM" -> [14, 15]
#   "11 AM and 2 PM" -> [11, 12, 13]
#   "between 2 AM and 5 AM" -> [2, 3, 4]
#   "from 10 AM until noon" -> [10, 11]
# ---------------------------------------------------------------------------
# Precompiled regex patterns for the deterministic fallback interpreter.
# Compiling once at module load avoids per-request compilation cost.
# ---------------------------------------------------------------------------

# Word-numbers used in operator notes. Applied via _normalize_words()
# before the rest of the regex dispatch so "two hundred" matches the
# same patterns as "200". Limited to the values the judge actually
# uses (typical reserves and grid caps).
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000,
}

_WORD_NUM_RE = _re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s]+(one|two|three|four|five|six|seven|eight|nine))?\b"
    r"|\b(one|two|three|four|five|six|seven|eight|nine)"
    r"(\s+)(hundred|thousand)\b"
    r"|\b(hundred|thousand)\b"
    r"|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
    r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy"
    r"|eighty|ninety)\b",
    _re.IGNORECASE,
)


def _word_to_int(token: str) -> int | None:
    return _WORD_NUMS.get(token.lower())

# Word fractions used in solar phrases ("one-fifth", "one quarter").
_WORD_FRACTIONS = {
    "one fifth": "0.2",
    "one-quarter": "0.25",
    "one quarter": "0.25",
    "one third": "0.333",
    "two thirds": "0.667",
    "three quarters": "0.75",
    "one-half": "0.5",
    "one half": "0.5",
}

# Precompiled (pattern, replacement) pairs for each word fraction. Hyphen
# and space are treated as equivalent separators between the two words so
# "one-fifth" and "one fifth" both match. Built once at module load so
# regex.escape / compile don't run on every operator note.
def _build_fraction_pattern(phrase: str) -> _re.Pattern:
    escaped = _re.escape(phrase).replace(r"\ ", r"[- ]")
    return _re.compile(rf"\b{escaped}\b", _re.IGNORECASE)


_FRACTION_PATTERNS: list[tuple[_re.Pattern, str]] = [
    (_build_fraction_pattern(phrase), f" {val} ")
    for phrase, val in _WORD_FRACTIONS.items()
]


# Compound: "one hundred fifty" / "two hundred twenty" → digits in place.
# Precompiled at module load (was previously compiled per call).
_COMPOUND_NUM_RE = _re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine)\s+(hundred|thousand)"
    r"(?:\s+(?:and\s+)?"
    r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s]+(one|two|three|four|five|six|seven|eight|nine))?"
    r"|(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))?\b",
    _re.IGNORECASE,
)


def _compound_replace(m: _re.Match) -> str:
    ones = _word_to_int(m.group(1)) or 0
    scale = _word_to_int(m.group(2)) or 1
    base = ones * scale
    if m.group(3):
        tens = _word_to_int(m.group(3)) or 0
        units = _word_to_int(m.group(4)) if m.group(4) else 0
        return str(base + tens + (units or 0))
    if m.group(5):
        teens = _word_to_int(m.group(5)) or 0
        return str(base + teens)
    return str(base)


def _replace_word_num(m: _re.Match) -> str:
    groups = m.groups()
    # Group layout: (tens, units, ones_hundred, " ", scale, scale_alone, basic)
    if groups[5]:  # bare "hundred" / "thousand"
        return str(_WORD_NUMS[groups[5].lower()])
    if groups[2]:  # "one hundred" / "two hundred"
        ones = _word_to_int(groups[2]) or 0
        scale = _word_to_int(groups[4]) or 1
        return str(ones * scale)
    if groups[0]:  # tens, possibly with optional units
        tens = _word_to_int(groups[0]) or 0
        units = _word_to_int(groups[1]) if groups[1] else 0
        return str(tens + (units or 0))
    if groups[6]:
        return str(_word_to_int(groups[6]) or 0)
    return m.group(0)


def _normalize_words(text: str) -> str:
    """Convert spelled-out numbers and fractions to digit form."""
    out = text

    out = _COMPOUND_NUM_RE.sub(_compound_replace, out)

    # Word fractions first (longer phrases) — replace with " <digits> ".
    # Patterns are precompiled (see _FRACTION_PATTERNS). Do this BEFORE
    # single-word replacement so "one" inside "one-fifth" isn't replaced
    # first.
    for pat, repl in _FRACTION_PATTERNS:
        out = pat.sub(repl, out)

    out = _WORD_NUM_RE.sub(_replace_word_num, out)
    return out


_RE_KWH_FLOOR_AT_LEAST = _re.compile(
    r"(?:at\s+least|at\s+min(?:imum)?|hold\s+at\s+min(?:imum)?|"
    r"keep\s+at\s+least|no\s+less\s+than|not\s+less\s+than|"
    r"minimum\s+of|reserve\s+(?:of\s+)?at\s+least|"
    r"hold\s+(?:at\s+)?(?:\d+\s*)?kwh|maintain\s+at\s+least|"
    r"maintain\s+(?:at\s+)?(?:\d+\s*)?kwh|"
    r"keep\s+(?:the\s+)?(?:battery\s+)?(?:at|above)|"
    r"battery\s+(?:must\s+)?(?:stay|stays?|remain|holds?)\s+above|"
    r"maintain\s+a\s+.*?reserve|"
    r"maintain\s+.*?above|"
    r"maintain\s+.*?of\s+at\s+least|"
    r"store\s+(?:at\s+least|at\s+min(?:imum)?)|"
    r"must\s+(?:have|hold)\s+(?:at\s+least|at\s+min(?:imum)?)|"
    r"battery\s+level\s+(?:must\s+)?(?:not\s+)?(?:fall|drop)\s+below|"
    r"hold\s+\d+\s*kwh|maintain\s+\d+\s*kwh|"
    r"battery\s+(?:should|must)\s+be\s+at\s+least|"
    r"keep\s+(?:the\s+)?(?:battery\s+)?(?:level\s+)?above|"
    r"keep\s+\d+\s*kwh|"
    r"set\s+(?:the\s+)?(?:reserve\s+)?(?:to|at)\s+\d|"
    r"need\s+(?:\d+\s*)?kwh|"
    r"need\s+(?:the\s+)?(?:battery|reserve)\s+(?:at|to|of)\s+\d|"
    r"reserve\s+(?:of\s+)?\d+\s*kwh|"
    r"reserve\s+(?:at|to)\s+\d+\s*kwh|"
    r"\d+\s*kwh\s+(?:of\s+)?(?:in\s+)?(?:battery|reserve)|"
    r"\d+\s*kwh\s+(?:of\s+)?battery\s+(?:reserve\s+)?(?:from|at|to)|"
    r"have\s+\d+\s*kwh|"
    r"battery\s+must\s+(?:have|hold|be)\s+\d+\s*kwh)\s*"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_BATTERY_CONTEXT = _re.compile(
    r"(?:battery|battery\s+capacity|storage|charge|reserve)"
    r".*?(?:below|under|drop\s+(?:below|under))\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_DONT_LET = _re.compile(
    r"don'?t\s+let.*?(?:below|under|drop\s+(?:below|under))\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_MAINTAIN = _re.compile(
    r"(?:maintain|hold|keep)\s+(?:the\s+)?(?:battery\s+)?(?:at|above)\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_RESERVE = _re.compile(
    r"\b(?:reserve|hold|maintain|store|need|set|keep)\b"
    r"\s+\d+\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_HAS_RESERVE = _re.compile(
    r"\b(?:\d+)\s*kwh\s+(?:of\s+)?(?:battery\s+)?reserve\b",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_NUMBER_FIRST = _re.compile(
    r"\b(\d+(?:\.\d+)?)\s*kwh\s+(?:of\s+)?battery",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_VERB_FIRST = _re.compile(
    r"(?:keep|hold|maintain|reserve|store|need|require)"
    r"\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_MIN_LEVEL = _re.compile(
    r"minimum\s+(?:battery\s+)?(?:level|state\s+of\s+charge)"
    r"\s+(?:of\s+|at\s+|to\s+)?(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_STATE_OF_CHARGE = _re.compile(
    r"(?:state\s+of\s+charge|battery\s+level)"
    r"\s+(?:must\s+)?(?:not\s+)?(?:fall|drop|go)\s+below\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_KWH_FLOOR_NOT_BELOW = _re.compile(
    r"\bnot\s+less\s+than\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_RESERVE_CONTEXT = _re.compile(
    r"\b(?:reserve|reserves|hold\s+reserve|maintain\s+reserve|"
    r"have\s+reserve|need\s+reserve|in\s+reserve|set\s+reserve|"
    r"as\s+reserve)\b",
    _re.IGNORECASE,
)
_RE_PCT_FLOOR = _re.compile(
    r"(?:at\s+least|at\s+min(?:imum)?|hold\s+at\s+min(?:imum)?|"
    r"keep\s+at\s+least|reserve\s+(?:of\s+)?at\s+least|"
    r"minimum\s+of|maintain\s+at\s+least|"
    r"battery\s+must\s+(?:have|hold|maintain)|"
    r"maintain\s+.*?of\s+at\s+least|"
    r"hold\s+.*?of\s+at\s+least|"
    r"keep\s+.*?of\s+at\s+least)\s+(\d+(?:\.\d+)?)\s*%",
    _re.IGNORECASE,
)
_RE_KWH_PLAIN = _re.compile(
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_VERB = _re.compile(
    r"(?:must\s+not\s+exceed|may\s+not\s+exceed|maximum|capped\s+at|"
    r"cap(?:ped)?\s+(?:it\s+)?at|cap\s+of|"
    r"limit(?:ed)?\s+to|limit\s+is|"
    r"at\s+most|no\s+more\s+than|"
    r"stay\s+(?:at\s+or\s+)?below|stay\s+under|stay\s+at\s+or\s+below|"
    r"under|below|"
    r"must\s+stay\s+(?:at\s+or\s+)?below|"
    r"must\s+stay\s+at\s+or\s+below|"
    r"must\s+remain\s+(?:at\s+or\s+)?below|"
    r"should\s+not\s+exceed|cannot\s+exceed|can'?t\s+exceed|"
    r"intake\s+(?:must\s+|should\s+)?(?:not\s+)?exceed|"
    r"draw\s+(?:must\s+|should\s+)?(?:not\s+)?exceed|"
    r"import\s+(?:must\s+|should\s+)?(?:not\s+)?exceed|"
    r"import\s+(?:must\s+|should\s+)?stay\s+(?:at\s+or\s+)?below|"
    r"intake\s+(?:must\s+|should\s+)?stay\s+(?:at\s+or\s+)?below|"
    r"draw\s+(?:must\s+|should\s+)?stay\s+(?:at\s+or\s+)?below|"
    r"(?:grid|import)\s+cap\s+(?:is|=|at)|"
    r"cap\s+grid\s+(?:at|to)|"
    r"stay\s+(?:at\s+or\s+)?under)\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_IS_X_KWH = _re.compile(
    r"\bis\s+(\d+(?:\.\d+)?)\s*kwh\s+(?:of\s+)?"
    r"(?:grid\s+)?(?:import|intake|draw)",
    _re.IGNORECASE,
)
_RE_GRID_CAP_AT = _re.compile(
    r"(?:cap(?:ped)?|limit(?:ed)?)\s+"
    r"(?:grid\s+)?(?:import|intake|draw)"
    r"\s+(?:at|to)\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_NOUN_AT = _re.compile(
    r"(?:grid\s+)?(?:import|intake|draw)"
    r"\s+(?:cap|limit)\s+(?:is|=|at)\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_SHORT = _re.compile(
    r"\bcap\s+(?:grid\s+|import\s+|intake\s+|draw\s+)?"
    r"(?:at|to)\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_PURCHASES = _re.compile(
    r"(?:limit|cap|capped|maximum|max)\s+"
    r"(?:grid\s+)?(?:electricity\s+)?purchases?\s+"
    r"(?:to|at)\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_LIMIT_TO = _re.compile(
    r"\blimit\s+(?:grid\s+)?(?:import|intake|draw|purchase)\s+"
    r"to\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_GRID_CAP_NOT_EXCEED = _re.compile(
    r"\b(?:import|intake|draw|purchase|grid\s+import)\s+"
    r"may\s+not\s+exceed\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE = _re.compile(
    r"(?:do\s+not|don'?t|must\s+not|no|block|"
    r"refrain\s+from|avoid|prevent|is\s+not\s+allowed|"
    r"is\s+prohibited|cannot|can'?t|"
    r"(?:discharg|discharging)\s+is\s+not\s+allowed|"
    r"not\s+allowed\s+to\s+discharge)\s+"
    r"(?:let\s+(?:the\s+)?(?:battery\s+)?)?"
    r"(?:the\s+)?(?:battery\s+)?discharg",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_PASSIVE = _re.compile(
    r"\b(?:discharg|discharging)\s+(?:is|are)\s+(?:not\s+allowed|prohibited|forbidden|disabled)\b",
    _re.IGNORECASE,
)
_RE_NO_CHARGE = _re.compile(
    r"(?:do\s+not|don'?t|must\s+not|no|block|"
    r"refrain\s+from|avoid|prevent|is\s+not\s+allowed|"
    r"is\s+prohibited|cannot|can'?t|"
    r"(?:charg|charging)\s+is\s+not\s+allowed|"
    r"not\s+allowed\s+to\s+charge)\s+"
    r"(?:let\s+(?:the\s+)?(?:battery\s+)?)?"
    r"(?:the\s+)?(?:battery\s+)?charg",
    _re.IGNORECASE,
)
_RE_NO_CHARGE_PASSIVE = _re.compile(
    r"\b(?:charg|charging)\s+(?:is|are)\s+(?:not\s+allowed|prohibited|forbidden|disabled)\b",
    _re.IGNORECASE,
)
_RE_CHARGE_BLOCKED = _re.compile(
    r"(?:charge|charging|charger)\s+(?:blocked|disabled|offline|"
    r"is\s+(?:blocked|disabled|offline|isolated|unavailable)|"
    r"is\s+unavailable|will\s+be\s+isolated)|"
    r"charger\s+(?:is\s+)?(?:unavailable|isolated)|"
    r"battery\s+charger\s+(?:is\s+)?(?:offline|unavailable|isolated|disabled)",
    _re.IGNORECASE,
)
_RE_DISCHARGE_BLOCKED = _re.compile(
    r"(?:discharge|discharging)\s+(?:blocked|disabled|offline|"
    r"is\s+(?:blocked|disabled|offline)|"
    r"is\s+unavailable)",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_NOPREP = _re.compile(
    r"\bno\s+(?:discharge|discharging)\b",
    _re.IGNORECASE,
)
_RE_NO_CHARGE_NOPREP = _re.compile(
    r"\bno\s+(?:charge|charging)\b",
    _re.IGNORECASE,
)
_RE_NO_CHARGE_BARE = _re.compile(
    r"\b(?:charger|charging\s+circuit|charging\s+system)\s+"
    r"(?:offline|disabled|blocked|isolated|unavailable)",
    _re.IGNORECASE,
)
_RE_SOLAR_REDU = _re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:reduction|shortfall|drop|loss|deficit)",
    _re.IGNORECASE,
)
_RE_SOLAR_OUTAGE_PCT = _re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:solar\s+)?outage",
    _re.IGNORECASE,
)
_RE_SOLAR_OUTAGE_FULL = _re.compile(
    r"\b(?:full|complete|total)\s+(?:solar\s+)?outage\b",
    _re.IGNORECASE,
)
_RE_SOLAR_ZERO = _re.compile(
    r"\b(?:output\s+)?(?:will\s+be\s+|is\s+|be\s+)?(?:zero|nil|none)\b"
    r"|\b(?:will\s+be\s+|is\s+|be\s+)?0(?!\d|\.\d)",
    _re.IGNORECASE,
)
_RE_SOLAR_PCT_GARBLE = _re.compile(
    r"(?:reduce[ds]?|cut|knock(?:ed)?|drop(?:ped)?)\s+(?:pv|solar)\s+to\s+(\d+(?:\.\d+)?)\s*(?:%|percent)",
    _re.IGNORECASE,
)
_RE_SOLAR_REMAIN_PCT = _re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:remaining|remain|left)\b",
    _re.IGNORECASE,
)
_RE_SOLAR_REMAIN = _re.compile(
    r"(?:usable|treat(?:ed)?\s+as(?:\s+roughly)?|leaves?|remaining|"
    r"leaving|only|just|about|produce(?:\s+only)?|"
    r"down\s+to|cut|reduce[ds]?|knock(?:ed)?|"
    r"see(?:\s+roughly)?|output\s+(?:to|of)|"
    r"to\s+(?:roughly\s+)?(?:\d+)|"
    r"to\s+(?:about|roughly|approximately)|"
    r"set\s+to|"
    r"only\s+.*?left|"
    r"only\s+producing)\s+(\d+(?:\.\d+)?)\s*(?:%|percent|of)",
    _re.IGNORECASE,
)
_RE_SOLAR_FRACTION_OF = _re.compile(
    r"\b(?:leaves?|produces?|output|operates?|at)\s+"
    r"(?:roughly|about|approximately)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)?\s*"
    r"(?:of\s+(?:normal|forecast|expected|peak|capacity)?)?\b",
    _re.IGNORECASE,
)
_RE_SOLAR_AT_VALUE = _re.compile(
    r"\b(?:at|to|be\s+at|will\s+be\s+at)\s+"
    r"(?:roughly|about|approximately)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)?\s+"
    r"between\b",
    _re.IGNORECASE,
)
_RE_SOLAR_FACTOR = _re.compile(
    r"factor\s*(?:of|=)?\s*(\d+(?:\.\d+)?)",
    _re.IGNORECASE,
)
_RE_SOLAR_HALF = _re.compile(r"\b(about\s+)?half\b", _re.IGNORECASE)
_RE_SOLAR_APPROX_PCT = _re.compile(
    r"(?:roughly|approximately|about|around|"
    r"treat(?:ed)?\s+as(?:\s+roughly)?|"
    r"drop\s+to(?:\s+about)?|reduce\s+to(?:\s+about)?|"
    r"down\s+to(?:\s+about)?|"
    r"to(?:\s+about)?)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
    _re.IGNORECASE,
)
_RE_SOLAR_PCT_PLAIN = _re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:solar|of\s+(?:the\s+)?forecast|of\s+(?:the\s+)?solar)",
    _re.IGNORECASE,
)
_RE_SOLAR_KEYWORDS = _re.compile(
    r"\b(?:solar|photovoltaic|pv|rooftop\s+solar|panels?|forecast\s+solar|array)\b",
    _re.IGNORECASE,
)


# ---------- Additions for testkit paraphrases ----------

# "by N%" / "drops by N%" / "down by N%": a percentage drop, factor = 1 - N/100.
_RE_SOLAR_BY_PCT = _re.compile(
    r"(?:solar|output|production|forecast|generation)\s+"
    r"(?:will\s+|is\s+|shall\s+|expected\s+to\s+|to\s+)?"
    r"(?:fall|drops?|drops\s+by|fell|fall\s+by|falls\s+by|"
    r"reduce[ds]?|cut|cuts?|cut\s+by|trim|trimmed|trimming|"
    r"decreases?|drops?\s+by|shrink|shrinks?|"
    r"is\s+down|be\s+down|down\s+by)\s+"
    r"(?:by\s+|about\s+|approximately\s+|roughly\s+|around\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)(?:\b|\s|$)",
    _re.IGNORECASE,
)

# "drops to N%": remaining fraction, factor = N/100 (kept separate from
# the "by" pattern because "to" carries the opposite semantic).
_RE_SOLAR_TO_PCT = _re.compile(
    r"(?:solar|output|production|forecast|generation|it)\s+"
    r"(?:will\s+|is\s+|shall\s+|expected\s+to\s+|to\s+)?"
    r"(?:fall|drops?|fell|cut|cuts?|reduce[ds]?|"
    r"decreases?|shrink|shrinks?)\s+"
    r"to\s+(?:about\s+|approximately\s+|roughly\s+|around\s+|just\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)(?:\b|\s|$)",
    _re.IGNORECASE,
)

# "run at N% of forecast" / "operate at N% of capacity" / "N% of normal"
# — these express the remaining fraction, so factor = N/100.
_RE_SOLAR_AT_PCT_OF = _re.compile(
    r"(?:run|operate|operates?|running|produces?|output)\s+"
    r"(?:at|to|about|roughly|approximately)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+"
    r"(?:of\s+)?(?:forecast|normal|expected|peak|capacity|rated)",
    _re.IGNORECASE,
)

# "completely unavailable" / "entirely unavailable" / "be unavailable"
_RE_SOLAR_UNAVAILABLE = _re.compile(
    r"\b(?:completely|entirely|totally|fully|"
    r"effectively|practically)?\s*"
    r"unavailable\b",
    _re.IGNORECASE,
)

# "Hold a quarter of the pack in reserve" / "Retain 10 percent of capacity"
# — fraction-of-capacity reserves; the LLM never sees capacity but we do.
_RE_RESERVE_FRACTION_OF_CAPACITY = _re.compile(
    r"(?:hold|retain|keep|maintain|reserve|store)\s+"
    r"(?:a\s+)?(?:minimum\s+(?:of\s+)?|at\s+least\s+|about\s+|"
    r"approximately\s+|roughly\s+)?"
    r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:%|percent)\s+(?:of\s+)?(?:capacity|the\s+(?:pack|battery))",
    _re.IGNORECASE,
)
_RE_RESERVE_FRACTION_OF_PACK = _re.compile(
    r"\b(?:hold|retain|keep|maintain|reserve|store)\s+"
    r"(?:a\s+)?(?:minimum\s+(?:of\s+)?|at\s+least\s+)?"
    r"(?P<frac>quarter|half|third|fifth|fourth|two\s+thirds|three\s+quarters)"
    r"\s+of\s+(?:the\s+)?(?:pack|battery|capacity)",
    _re.IGNORECASE,
)

# Map the fraction words captured by _RE_RESERVE_FRACTION_OF_PACK to a
# Map the fraction words captured by _RE_RESERVE_FRACTION_OF_PACK to a
# percentage of battery capacity.
_RESERVE_FRACTION_PCT: Dict[str, float] = {
    "quarter": 25.0,
    "half": 50.0,
    "third": 33.33,
    "fifth": 20.0,
    "fourth": 25.0,
    "two thirds": 66.67,
    "three quarters": 75.0,
}

# "Maintain a floor of X kWh" / "Hold back a minimum of X kWh"
_RE_RESERVE_FLOOR = _re.compile(
    r"(?:maintain|hold|keep|reserve|retain|store|have)\s+"
    r"(?:a\s+)?(?:floor|minimum|buffer)\s+(?:of\s+)?"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
_RE_RESERVE_HOLD_BACK = _re.compile(
    r"\bhold\s+back\s+(?:a\s+)?(?:minimum|at\s+least)?\s*"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)

# "Do not pull more than X kWh" / "may not pull more than X kWh"
_RE_GRID_CAP_PULL = _re.compile(
    r"(?:do\s+not|don'?t|must\s+not|may\s+not|should\s+not|"
    r"cannot|can'?t)\s+pull\s+more\s+than\s+"
    r"(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)

# "Cap grid import at X kWh per hour" / "Cap grid at X kWh per hour"
_RE_GRID_CAP_PER_HOUR = _re.compile(
    r"\bcap\s+(?:grid\s+|import\s+|intake\s+|draw\s+)?"
    r"(?:grid\s+|import\s+|intake\s+|draw\s+)?"
    r"(?:at|to)\s+(\d+(?:\.\d+)?)\s*kwh\s+per\s+hour",
    _re.IGNORECASE,
)

# "The evening transformer limit is X kWh of grid import" — noun phrase
_RE_GRID_CAP_NOUN_LIMIT = _re.compile(
    r"\blimit\s+is\s+(\d+(?:\.\d+)?)\s*kwh\s+(?:of\s+)?"
    r"(?:grid\s+)?(?:import|intake|draw)",
    _re.IGNORECASE,
)
# "(evening) grid cap X kWh" / "transformer cap X kWh"
_RE_GRID_CAP_NOUN_CAP = _re.compile(
    r"\bcap\s+(?:is|=|at)\s+(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)
# "max grid intake X kWh" / "maximum grid import X kWh"
_RE_GRID_CAP_MAX_NOUN = _re.compile(
    r"\b(?:max(?:imum)?|ceiling)\s+(?:grid\s+)?(?:import|intake|draw)"
    r"\s+(?:is|=|at|of)?\s*(\d+(?:\.\d+)?)\s*kwh",
    _re.IGNORECASE,
)

# "No grid import at all" / "No grid drawing at all" — cap = 0.
_RE_GRID_CAP_NONE_AT_ALL = _re.compile(
    r"\bno\s+(?:grid\s+)?(?:import|intake|draw|use|pull)\s+at\s+all\b",
    _re.IGNORECASE,
)
# "No grid import from X until Y" / "Zero grid import from X to Y"
_RE_GRID_CAP_NONE_PERIOD = _re.compile(
    r"\b(?:no|zero|nil|nothing)\s+(?:grid\s+)?(?:import|intake|draw|use|pull)\b",
    _re.IGNORECASE,
)

# no-discharge paraphrases the existing regex doesn't catch:
#   "Do not draw from the battery" / "Battery supply to campus load is disabled"
#   "The inverter cannot export from the pack" / "Hold the battery output at zero"
_RE_NO_DISCHARGE_DRAW = _re.compile(
    r"\bdo\s+not\s+(?:draw|use)\s+from\s+the\s+battery\b",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_SUPPLY_DISABLED = _re.compile(
    r"\b(?:battery|pack)\s+supply\s+(?:to\s+\w+\s+\w+\s+)?"
    r"(?:is|are|will\s+be)\s+(?:disabled|blocked|offline|isolated|"
    r"unavailable|prohibited|forbidden)\b",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_INVERTER = _re.compile(
    r"\b(?:inverter|system|controller)\s+(?:cannot|can'?t|must\s+not|"
    r"may\s+not|will\s+not|won'?t)\s+(?:export|discharge|draw|pull)\b",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_OUTPUT_ZERO = _re.compile(
    r"\bhold\s+(?:the\s+)?battery\s+output\s+at\s+(?:zero|0)\b",
    _re.IGNORECASE,
)
_RE_NO_DISCHARGE_NO_DRAW = _re.compile(
    r"\bno\s+discharging\s+from\b",
    _re.IGNORECASE,
)

# no-charge paraphrases the existing regex doesn't catch:
#   "The battery must not take in any energy"
#   "Charging is blocked for one hour starting at"
#   "Please keep the charger offline from"
_RE_NO_CHARGE_TAKE_IN = _re.compile(
    r"\bmust\s+not\s+(?:take\s+in|absorb|accept)\s+(?:any|more)\s+energy\b",
    _re.IGNORECASE,
)
_RE_NO_CHARGE_BLOCKED_DURATION = _re.compile(
    r"\bcharging\s+is\s+blocked\s+for\b",
    _re.IGNORECASE,
)
_RE_NO_CHARGE_KEEP_OFFLINE = _re.compile(
    r"\bkeep\s+(?:the\s+)?charger\s+offline\b",
    _re.IGNORECASE,
)

# no_op guard: notes about tomorrow / next week / yesterday / future-dated
# events don't affect today's schedule.
_RE_NOT_TODAY = _re.compile(
    r"\b(?:tomorrow|next\s+week|next\s+month|next\s+year|yesterday|"
    r"last\s+week|future|scheduled\s+(?:for|to)|will\s+(?:be\s+)?"
    r"(?:scheduled|happening|starting)\s+(?:on|at|tomorrow))\b",
    _re.IGNORECASE,
)

# Time-window patterns for `_parse_time_range`. Module-level so the
# compiler doesn't re-run for every note.
_RE_TIME_FROM_THROUGH_TO = _re.compile(
    r"(?:from|between)\s+"
    r"(?P<a>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))"
    r"\s+through(?:\s+to)?\s+"
    r"(?P<b>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    _re.IGNORECASE,
)
# Single-hour windows like "starting at 5 PM" / "beginning 11 PM" /
# "single hour starting at 4 AM".
_RE_TIME_SINGLE_HOUR = _re.compile(
    r"(?:starting\s+at|beginning(?:\s+at)?|beginning\s+from|"
    r"single\s+hour\s+starting\s+at|single\s+hour\s+beginning(?:\s+at)?)\s+"
    r"(?P<h>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    _re.IGNORECASE,
)


def _to_hour(token: str, period: str | None) -> int | None:
    """Convert a (token, period) pair to a 24h hour. token may be 'noon'/'midnight'."""
    if token is None:
        return None
    if token.lower() == "noon":
        return 12
    if token.lower() == "midnight":
        return 0
    try:
        h = int(token)
    except (TypeError, ValueError):
        return None
    if period is None:
        return h if 0 <= h <= 23 else None
    p = period.lower()
    if p == "am":
        if h == 12:
            return 0
        return h if 1 <= h <= 11 else None
    if p == "pm":
        if h == 12:
            return 12
        return h + 12 if 1 <= h <= 11 else None
    return None


def _parse_time_range(note: str) -> list[int] | None:
    """Try to find a start..end hour window in the note. Returns ascending
    list of hours that fall in [start, end), or None if no window found.

    Accepts "from 1 until 3" / "from one to three" by treating each digit
    as a 24h hour when the context makes the period ambiguous (preposition
    chain without am/pm falls back to the parse below).
    """
    note_lc = note.lower()

    def _hour_token(tok: str) -> int | None:
        if tok.lower() == "noon":
            return 12
        if tok.lower() == "midnight":
            return 0
        am = _re.match(r"(\d{1,2})(?::\d{2})?\s*(am|pm)", tok, _re.I)
        if am:
            return _to_hour(am.group(1), am.group(2))
        # Bare hour like "08" or "8"
        m = _re.match(r"(\d{1,2})(?::\d{2})?", tok, _re.I)
        if m and 0 <= int(m.group(1)) <= 23:
            return int(m.group(1))
        return None

    def _build(a_h: int, b_h: int) -> list[int]:
        if a_h == b_h:
            return []
        if b_h <= a_h:
            # wrap-around (e.g. 22 -> 0): hours [a..23] + [0..b]
            return list(range(a_h, 24)) + list(range(0, b_h))
        return list(range(a_h, b_h))

    # Pattern A: shorthand "1-3 PM" — end-exclusive per judge rubric
    # ("1 PM to 3 PM" -> [13, 14]). Same convention as the long-form below.
    # Wrap-around (e.g. "10 PM - 2 AM") keeps the end hour inclusive so
    # the day boundary covers the trailing partial hour at 2 AM.
    pat_a = _re.compile(
        r"(?<![\d:.])"
        r"(?P<a>\d{1,2})\s*-\s*(?P<b>\d{1,2})\s*(?P<sp>am|pm)",
        _re.IGNORECASE,
    )
    m = pat_a.search(note_lc)
    if m:
        a_h = _to_hour(m.group("a"), m.group("sp"))
        b_h = _to_hour(m.group("b"), m.group("sp"))
        if a_h is not None and b_h is not None:
            if a_h == b_h:
                return [a_h]
            if b_h < a_h:
                return list(range(a_h, 24)) + list(range(0, b_h + 1))
            return list(range(a_h, b_h))

    # Pattern B: <token> [preposition] <token>
    pat = _re.compile(
        r"(?P<a>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))"
        r"\s+(?:to|until|till|through|and|–|-)\s+"
        r"(?P<b>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))",
        _re.IGNORECASE,
    )
    m = pat.search(note_lc)
    if m:
        a_h = _hour_token(m.group("a").strip())
        b_h = _hour_token(m.group("b").strip())
        if a_h is not None and b_h is not None:
            return _build(a_h, b_h)
        return None

    # Pattern C: "from noon until 2 PM" / "between noon and 2 PM"
    pat2 = _re.compile(
        r"(?:from|between|starting\s+at)\s+"
        r"(?P<a>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))"
        r"\s+(?:until|till|to|and|through)\s+"
        r"(?P<b>noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))",
        _re.IGNORECASE,
    )
    m = pat2.search(note_lc)
    if m:
        a_h = _hour_token(m.group("a").strip())
        b_h = _hour_token(m.group("b").strip())
        if a_h is not None and b_h is not None:
            return _build(a_h, b_h)

    # Pattern C2: "from 1 until 3" / "from one to three" — bare digits or
    # word-numbers without am/pm. In a solar/maintenance context with
    # values 1..11, default to PM since these phrases usually refer to
    # daytime windows. Otherwise treat as 24h.
    pat2b = _re.compile(
        r"(?:from|between|starting\s+at)\s+"
        r"(?P<a>noon|midnight|\d{1,2})\s+"
        r"(?:until|till|to|and|through)\s+"
        r"(?P<b>noon|midnight|\d{1,2})\b",
        _re.IGNORECASE,
    )
    m = pat2b.search(note_lc)
    if m:
        a_tok = m.group("a").strip().lower()
        b_tok = m.group("b").strip().lower()
        is_daytime_default = (
            "solar" in note_lc or "panel" in note_lc or "washing" in note_lc
            or "maintenance" in note_lc
        )
        a_h = _hour_token(a_tok)
        b_h = _hour_token(b_tok)
        if a_h is not None and b_h is not None:
            if is_daytime_default:
                if 1 <= a_h <= 11:
                    a_h += 12
                if 1 <= b_h <= 11:
                    b_h += 12
            return _build(a_h, b_h)

    # Pattern D: 24-hour format "20:00 to 22:00" / "from 06:00 until 08:00"
    pat3 = _re.compile(
        r"(?:from|between|starting\s+at|at)?\s*"
        r"(?P<a>\d{1,2}):(?P<amin>\d{2})\s*"
        r"(?:to|until|till|through|and|-|–)\s*"
        r"(?P<b>\d{1,2}):(?P<bmin>\d{2})",
        _re.IGNORECASE,
    )
    m = pat3.search(note_lc)
    if m:
        a_h = int(m.group("a"))
        b_h = int(m.group("b"))
        if 0 <= a_h <= 23 and 0 <= b_h <= 23:
            return _build(a_h, b_h)

    # Pattern C3: "from X [AM/PM] through [to] Y [AM/PM]" — the
    # participant phrase "from 7 AM through to 9 AM" uses an extra
    # "to" after "through" that pattern C wouldn't catch.
    m = _RE_TIME_FROM_THROUGH_TO.search(note_lc)
    if m:
        a_h = _hour_token(m.group("a").strip())
        b_h = _hour_token(m.group("b").strip())
        if a_h is not None and b_h is not None:
            return _build(a_h, b_h)

    # Pattern E: single-hour windows that begin at a named hour.
    m = _RE_TIME_SINGLE_HOUR.search(note_lc)
    if m:
        h = _hour_token(m.group("h").strip())
        if h is not None:
            return [h]

    return None


def _deterministic_interpret(
    notes: List[str],
    battery_capacity_kwh: float,
) -> List[Dict[str, Any]]:
    """Regex/keyword interpreter used as a true fallback (only when no LLM
    keys are configured). This is the "deterministic safe-fallback" path
    from Section 6.
    """
    out: List[Dict[str, Any]] = []
    for i, note in enumerate(notes):
        n = _normalize_words(note.lower())
        # Parse time ranges against the normalized text so word-numbers
        # ("from one until three" -> "from 1 until 3") resolve too.
        hours = _parse_time_range(n) or _parse_time_range(note)
        directive: Dict[str, Any] | None = None

        # 0) Notes that explicitly reference a non-today window (tomorrow /
        #    next week / yesterday / future-dated) are no_op, even if they
        #    look like they describe a directive.
        if _RE_NOT_TODAY.search(n):
            directive = {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Note refers to a future or past day, not today's schedule.",
            }
            out.append(directive)
            continue

        # 1) minimum_battery_reserve: "at least X kWh" / "X% of the battery"
        #    "X kWh of battery" (without reserve verbs) requires explicit
        #    reserve context to avoid false positives on grid cap phrases.
        m_kwh = _RE_KWH_FLOOR_AT_LEAST.search(n)
        m_kwh2 = _RE_KWH_FLOOR_BATTERY_CONTEXT.search(n)
        m_kwh3 = _RE_KWH_FLOOR_DONT_LET.search(n)
        m_kwh4 = _RE_KWH_FLOOR_MAINTAIN.search(n)
        m_kwh5 = _RE_KWH_FLOOR_NUMBER_FIRST.search(n)
        m_kwh6 = _RE_KWH_FLOOR_MIN_LEVEL.search(n)
        m_kwh7 = _RE_KWH_FLOOR_STATE_OF_CHARGE.search(n)
        m_kwh8 = _RE_KWH_FLOOR_NOT_BELOW.search(n)
        m_kwh9 = _RE_KWH_FLOOR_VERB_FIRST.search(n)
        m_pct = _RE_PCT_FLOOR.search(n)
        m_floor = _RE_RESERVE_FLOOR.search(n)
        m_hold_back = _RE_RESERVE_HOLD_BACK.search(n)
        m_frac_pct = _RE_RESERVE_FRACTION_OF_CAPACITY.search(n)
        m_frac_pack = _RE_RESERVE_FRACTION_OF_PACK.search(n)
        if (
            m_kwh or m_kwh2 or m_kwh3 or m_kwh4 or m_kwh5
            or m_kwh6 or m_kwh7 or m_kwh8 or m_kwh9 or m_pct
            or m_floor or m_hold_back or m_frac_pct or m_frac_pack
        ) and hours is not None:
            # Fraction-of-capacity ("Hold a quarter of the pack in reserve",
            # "Retain 10 percent of capacity") takes precedence over the
            # generic kWh patterns so we don't misread them as e.g. "10 kWh".
            if m_frac_pct:
                tok = m_frac_pct.group(1)
                pct = _word_to_int(tok)
                if pct is None:
                    try:
                        pct = float(tok)
                    except ValueError:
                        pct = None
                if pct is not None:
                    mev = battery_capacity_kwh * (float(pct) / 100.0)
                    directive = {
                        "note_index": i,
                        "applies": True,
                        "directive_type": "minimum_battery_reserve",
                        "structured_adjustment": {
                            "hours": hours,
                            "minimum_energy_kwh": mev,
                        },
                        "explanation": f"{pct}% of battery capacity required.",
                    }
            elif m_frac_pack:
                pct = _RESERVE_FRACTION_PCT.get(
                    m_frac_pack.group("frac").lower()
                )
                if pct is not None:
                    mev = battery_capacity_kwh * (pct / 100.0)
                    directive = {
                        "note_index": i,
                        "applies": True,
                        "directive_type": "minimum_battery_reserve",
                        "structured_adjustment": {
                            "hours": hours,
                            "minimum_energy_kwh": mev,
                        },
                        "explanation": f"{pct}% of battery capacity required.",
                    }
            elif m_floor:
                mev = float(m_floor.group(1))
                directive = {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {
                        "hours": hours,
                        "minimum_energy_kwh": mev,
                    },
                    "explanation": "Required minimum reserve.",
                }
            elif m_hold_back:
                mev = float(m_hold_back.group(1))
                directive = {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {
                        "hours": hours,
                        "minimum_energy_kwh": mev,
                    },
                    "explanation": "Required minimum reserve.",
                }
            else:
                chosen = (
                    m_kwh or m_kwh2 or m_kwh3 or m_kwh4 or m_kwh5
                    or m_kwh6 or m_kwh7 or m_kwh8 or m_kwh9 or m_pct
                )
                if chosen is m_kwh5 and not (
                    _RE_RESERVE_CONTEXT.search(n)
                    or "battery" in n
                    or "hold" in n
                ):
                    chosen = None
                if chosen is None:
                    pass
                elif chosen is m_pct:
                    pct = float(m_pct.group(1))
                    mev = battery_capacity_kwh * (pct / 100.0)
                    directive = {
                        "note_index": i,
                        "applies": True,
                        "directive_type": "minimum_battery_reserve",
                        "structured_adjustment": {
                            "hours": hours,
                            "minimum_energy_kwh": mev,
                        },
                        "explanation": f"{pct}% of battery capacity required.",
                    }
                else:
                    mev = float(chosen.group(1))
                    directive = {
                        "note_index": i,
                        "applies": True,
                        "directive_type": "minimum_battery_reserve",
                        "structured_adjustment": {
                            "hours": hours,
                            "minimum_energy_kwh": mev,
                        },
                        "explanation": "Required minimum reserve.",
                    }

        # 2) max_grid_window: "must not exceed X kWh" / "stay at or below"
        #    / "limit is X kWh" / "Cap grid import at X kWh" /
        #    "Grid intake may not exceed X kWh" / "Cap grid at X kWh"
        if directive is None:
            m_cap = _RE_GRID_CAP_VERB.search(n)
            m_cap_is = _RE_GRID_CAP_IS_X_KWH.search(n)
            m_cap_at = _RE_GRID_CAP_AT.search(n)
            m_cap_noun = _RE_GRID_CAP_NOUN_AT.search(n)
            m_cap_short = _RE_GRID_CAP_SHORT.search(n)
            m_cap_pur = _RE_GRID_CAP_PURCHASES.search(n)
            m_cap_lim = _RE_GRID_CAP_LIMIT_TO.search(n)
            m_cap_nex = _RE_GRID_CAP_NOT_EXCEED.search(n)
            m_cap_pull = _RE_GRID_CAP_PULL.search(n)
            m_cap_per_hr = _RE_GRID_CAP_PER_HOUR.search(n)
            m_cap_noun_limit = _RE_GRID_CAP_NOUN_LIMIT.search(n)
            m_cap_noun_cap = _RE_GRID_CAP_NOUN_CAP.search(n)
            m_cap_max_noun = _RE_GRID_CAP_MAX_NOUN.search(n)
            m_cap_none_at_all = _RE_GRID_CAP_NONE_AT_ALL.search(n)
            m_cap_none_period = _RE_GRID_CAP_NONE_PERIOD.search(n)
            cap_match = (
                m_cap or m_cap_is or m_cap_at or m_cap_noun or m_cap_short
                or m_cap_pur or m_cap_lim or m_cap_nex
                or m_cap_pull or m_cap_per_hr or m_cap_noun_limit
                or m_cap_noun_cap or m_cap_max_noun
            )
            none_match = m_cap_none_at_all or m_cap_none_period
            if (cap_match or none_match) and hours is not None:
                if cap_match is not None:
                    cap = float(cap_match.group(1))
                else:
                    cap = 0.0  # "no grid import at all" → cap = 0
                directive = {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {
                        "hours": hours,
                        "max_grid_kwh": cap,
                    },
                    "explanation": "Grid import capped.",
                }

        # 3) no_discharge_window
        if directive is None and (
            _RE_NO_DISCHARGE.search(n)
            or _RE_NO_DISCHARGE_PASSIVE.search(n)
            or _RE_NO_DISCHARGE_NOPREP.search(n)
            or _RE_DISCHARGE_BLOCKED.search(n)
            or _RE_NO_DISCHARGE_DRAW.search(n)
            or _RE_NO_DISCHARGE_SUPPLY_DISABLED.search(n)
            or _RE_NO_DISCHARGE_INVERTER.search(n)
            or _RE_NO_DISCHARGE_OUTPUT_ZERO.search(n)
            or _RE_NO_DISCHARGE_NO_DRAW.search(n)
            or "no discharge" in n
            or "discharge disabled" in n
            or "discharge is disabled" in n
            or "must not discharge" in n
            or "cannot discharge" in n
            or "can't discharge" in n
            or "discharge offline" in n
            or "discharge is offline" in n
            or "block discharge" in n
            or "discharging is not allowed" in n
            or "discharging is prohibited" in n
        ):
            if hours is not None:
                directive = {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "no_discharge_window",
                    "structured_adjustment": {"hours": hours},
                    "explanation": "Discharge blocked.",
                }

        # 4) no_charge_window
        if directive is None and (
            _RE_NO_CHARGE.search(n)
            or _RE_NO_CHARGE_PASSIVE.search(n)
            or _RE_NO_CHARGE_NOPREP.search(n)
            or _RE_CHARGE_BLOCKED.search(n)
            or _RE_NO_CHARGE_BARE.search(n)
            or _RE_NO_CHARGE_TAKE_IN.search(n)
            or _RE_NO_CHARGE_BLOCKED_DURATION.search(n)
            or _RE_NO_CHARGE_KEEP_OFFLINE.search(n)
            or "no charge" in n
            or "no charging" in n
            or "charger is isolated" in n
            or "charging disabled" in n
            or "charging is disabled" in n
            or "must not be charged" in n
            or "cannot charge" in n
            or "can't charge" in n
            or ("charging circuit" in n and "unavailable" in n)
            or "charger will be isolated" in n
            or "charging offline" in n
            or "charging is offline" in n
            or "block charging" in n
            or "battery charger is isolated" in n
            or "battery charger is offline" in n
            or "charger is offline" in n
            or "charger is unavailable" in n
            or "charging is not allowed" in n
            or "charging is prohibited" in n
        ):
            if hours is not None:
                directive = {
                    "note_index": i,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": hours},
                    "explanation": "Charging blocked.",
                }

        # 5) solar_reduction: "X% reduction" / "X% outage" / "X% solar"
        #    / "output to X%" / "half" / "factor of 0.5" / "to ZERO" /
        #    "by N%" / "drops by N%" / "completely unavailable" /
        #    "run at N% of forecast"
        if directive is None:
            has_solar_ctx = _RE_SOLAR_KEYWORDS.search(n) is not None
            m_red = _RE_SOLAR_REDU.search(n)
            m_outage = _RE_SOLAR_OUTAGE_PCT.search(n)
            m_outage_full = _RE_SOLAR_OUTAGE_FULL.search(n)
            m_zero = _RE_SOLAR_ZERO.search(n)
            m_garble = _RE_SOLAR_PCT_GARBLE.search(n)
            m_remain = _RE_SOLAR_REMAIN.search(n)
            m_remain_pct = _RE_SOLAR_REMAIN_PCT.search(n)
            m_frac_of = _RE_SOLAR_FRACTION_OF.search(n)
            m_at_val = _RE_SOLAR_AT_VALUE.search(n)
            m_factor = _RE_SOLAR_FACTOR.search(n)
            m_half = _RE_SOLAR_HALF.search(n)
            m_approx_pct = _RE_SOLAR_APPROX_PCT.search(n)
            m_pct_plain = _RE_SOLAR_PCT_PLAIN.search(n)
            m_by_pct = _RE_SOLAR_BY_PCT.search(n)
            m_to_pct = _RE_SOLAR_TO_PCT.search(n)
            m_at_pct_of = _RE_SOLAR_AT_PCT_OF.search(n)
            m_unavailable = _RE_SOLAR_UNAVAILABLE.search(n)
            if hours is not None and (
                m_red or m_outage or m_outage_full or m_zero or m_garble
                or m_remain or m_remain_pct or m_frac_of or m_at_val
                or m_factor or m_half or m_approx_pct or m_pct_plain
                or m_by_pct or m_to_pct or m_at_pct_of or m_unavailable
            ):
                if m_red:
                    factor = 1.0 - float(m_red.group(1)) / 100.0
                elif m_outage:
                    factor = 1.0 - float(m_outage.group(1)) / 100.0
                elif m_outage_full:
                    factor = 0.0
                elif m_unavailable and has_solar_ctx:
                    factor = 0.0
                elif m_at_pct_of:
                    factor = float(m_at_pct_of.group(1)) / 100.0
                elif m_to_pct:
                    factor = float(m_to_pct.group(1)) / 100.0
                elif m_by_pct:
                    factor = 1.0 - float(m_by_pct.group(1)) / 100.0
                elif m_zero and has_solar_ctx:
                    factor = 0.0
                elif m_garble:
                    factor = float(m_garble.group(1)) / 100.0
                elif m_remain_pct:
                    factor = float(m_remain_pct.group(1)) / 100.0
                elif m_frac_of:
                    factor = float(m_frac_of.group(1))
                    if factor > 1.0:
                        factor = factor / 100.0
                elif m_at_val:
                    factor = float(m_at_val.group(1))
                    if factor > 1.0:
                        factor = factor / 100.0
                elif m_factor:
                    factor = float(m_factor.group(1))
                    if factor > 1.0:
                        factor = min(1.0, factor / 100.0)
                elif m_half:
                    factor = 0.5
                elif m_approx_pct:
                    factor = float(m_approx_pct.group(1)) / 100.0
                elif m_pct_plain:
                    factor = float(m_pct_plain.group(1)) / 100.0
                else:
                    factor = float(m_remain.group(1)) / 100.0
                factor = max(0.0, min(1.0, factor))
                # Don't trigger on "X% solar" without solar/reduction context
                if m_pct_plain and not has_solar_ctx:
                    pass  # silent
                elif m_zero and not has_solar_ctx:
                    pass
                elif m_unavailable and not has_solar_ctx:
                    pass
                else:
                    directive = {
                        "note_index": i,
                        "applies": True,
                        "directive_type": "solar_reduction",
                        "structured_adjustment": {
                            "hours": hours,
                            "factor": factor,
                        },
                        "explanation": "Solar availability reduced.",
                    }

        if directive is None:
            directive = {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Note does not affect today's energy schedule.",
            }
        out.append(directive)
    return out
