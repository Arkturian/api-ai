"""OpenAI Decisions API (GA 22.10.2026) — Kandidat als Alternative zu Jev.

Auftrag Alex 26.10.2026: vollstaendig anbinden, gegen Jev messen.

Upstream: POST https://api.openai.com/v1/decisions, Modell `gpt-6-luna`
(laut Guide das einzige). Fragen sind eine LISTE mit `name`, Typen
`predicate` | `choice` (`choices: [{value, description}]`) | `score`
(`levels: [{label, description}]`, 2-10, Index ab 0). Antwort:
`answers` in Fragenreihenfolge, je `{type, name, ...}`; `type` kann auch
`refusal` sein. Abgerechnet wird nur Eingabe: 0,10 USD je Mio, gecacht
0,01 USD je Mio (Prefix >= 1024 Tokens, Ankuendigung 22.10.).
Gemessen 26.10.: Antwortform wie oben, `usage.input_tokens_details`
traegt `cached_tokens` und `cache_write_tokens`.

Kostenschutz wie bei Jev: kein confirm_api_billing, aber ein Tagesbudget je
Host gegen Endlosschleifen. Der Schluessel ist OPENAI_API_KEY aus der .env
des Dienstes und wird nie geloggt oder zurueckgegeben.

Uebersetzer jev<->decisions: damit dieselbe Frage an beide Motoren gehen
kann (A/B ueber /ai/jev?engine=openai) und Aufrufer nichts umbauen muessen.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

UPSTREAM_URL = "https://api.openai.com/v1/decisions"
TYPEN = ("predicate", "choice", "score")
MAX_FRAGEN = 400


def default_model() -> str:
    return os.getenv("DECISIONS_DEFAULT_MODEL", "gpt-6-luna")


def schluessel() -> Optional[str]:
    return (os.getenv("OPENAI_API_KEY") or "").strip() or None


def _preise() -> tuple:
    return (float(os.getenv("DECISIONS_PRICE_PER_1M_INPUT_USD", "0.10")),
            float(os.getenv("DECISIONS_PRICE_PER_1M_CACHED_USD", "0.01")))


def kosten(usage: dict) -> float:
    """Gecachte Tokens sind in input_tokens enthalten (Guide) — also
    abziehen und zum Cache-Preis rechnen. Cache-Schreiben kostet laut Guide
    den normalen Satz, ist also im ungecachten Anteil richtig."""
    usage = usage or {}
    gesamt = int(usage.get("input_tokens") or 0)
    gecacht = int(((usage.get("input_tokens_details") or {}).get("cached_tokens")) or 0)
    voll, cache = _preise()
    return (max(gesamt - gecacht, 0) * voll + gecacht * cache) / 1_000_000.0


# ── Tageszaehler (eine Datei je Tag und Host) ───────────────────────────

def _dir() -> Path:
    return Path(os.getenv("DECISIONS_USAGE_DIR", "/var/lib/api-ai"))


def _datei() -> Path:
    return _dir() / f"decisions_usage_{datetime.now().strftime('%Y-%m-%d')}.json"


def _grenzen() -> tuple:
    return (int(os.getenv("DECISIONS_DAILY_MAX_REQUESTS", "50000")),
            float(os.getenv("DECISIONS_DAILY_MAX_USD", "2")))


@contextlib.contextmanager
def _sperre():
    _dir().mkdir(parents=True, exist_ok=True)
    with open(_dir() / ".decisions_usage.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _lesen() -> dict:
    p = _datei()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            pass
    return {"day": datetime.now().strftime("%Y-%m-%d"), "requests": 0, "questions": 0,
            "input_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "errors": 0, "by_caller": {}}


def _schreiben(d: dict) -> None:
    p = _datei()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    tmp.replace(p)


def rueckstellung() -> str:
    jetzt = datetime.now().astimezone()
    return (jetzt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class DecisionsBudget(Exception):
    def __init__(self, requests, cost_usd, max_req, max_usd):
        super().__init__("decisions daily budget")
        self.detail = {"error": "decisions_daily_budget_exceeded", "requests_today": requests,
                       "cost_usd_today": cost_usd, "max_requests_per_day": max_req,
                       "max_usd_per_day": max_usd, "resets_at": rueckstellung(),
                       "hint": "Schutz gegen Endlosschleifen; Grenzen per "
                               "DECISIONS_DAILY_MAX_REQUESTS/DECISIONS_DAILY_MAX_USD."}


def budget_pruefen() -> None:
    max_req, max_usd = _grenzen()
    with _sperre():
        d = _lesen()
    if (max_req > 0 and d["requests"] >= max_req) or (max_usd > 0 and d["cost_usd"] >= max_usd):
        raise DecisionsBudget(d["requests"], round(d["cost_usd"], 6), max_req, max_usd)


def buchen(caller: str, fragen: int, usage: Optional[dict], cost_usd: float,
           fehler: bool = False) -> None:
    usage = usage or {}
    eingabe = int(usage.get("input_tokens") or 0)
    gecacht = int(((usage.get("input_tokens_details") or {}).get("cached_tokens")) or 0)
    with _sperre():
        d = _lesen()
        for ziel in (d, d["by_caller"].setdefault(caller, {"requests": 0, "questions": 0,
                                                           "input_tokens": 0, "cached_tokens": 0,
                                                           "cost_usd": 0.0, "errors": 0})):
            ziel["requests"] += 1
            ziel["questions"] += int(fragen or 0)
            ziel["input_tokens"] += eingabe
            ziel["cached_tokens"] += gecacht
            ziel["cost_usd"] += float(cost_usd or 0.0)
            if fehler:
                ziel["errors"] += 1
        d["last_updated"] = datetime.now().isoformat()
        _schreiben(d)


def status() -> dict:
    max_req, max_usd = _grenzen()
    with _sperre():
        d = _lesen()
    voll, cache = _preise()
    d["budget_exhausted"] = bool((max_req > 0 and d["requests"] >= max_req)
                                 or (max_usd > 0 and d["cost_usd"] >= max_usd))
    d["resets_at"] = rueckstellung()
    d["cost_usd"] = round(d["cost_usd"], 8)
    d["limits"] = {"max_requests_per_day": max_req, "max_usd_per_day": max_usd}
    d["price_per_1m_input_usd"] = voll
    d["price_per_1m_cached_input_usd"] = cache
    d["scope"] = "per_host"
    return d


# ── Aufruf ──────────────────────────────────────────────────────────────

async def aufrufen(body: dict, timeout_s: float = 15.0, versuche: int = 3) -> tuple:
    """-> (status_code, json_body). 429 und 5xx mit Backoff (retry-after
    beachtet); alles andere geht beim ersten Mal zurueck."""
    key = schluessel()
    if not key:
        return 0, None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    code, data = 0, None
    for versuch in range(versuche):
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(UPSTREAM_URL, json=body, headers=headers)
        code = r.status_code
        try:
            data = r.json()
        except Exception:
            data = {"raw": (r.text or "")[:500]}
        if (code == 429 or code >= 500) and versuch < versuche - 1:
            warte = None
            ra = (r.headers or {}).get("retry-after")
            if ra:
                try:
                    warte = min(float(ra), 10.0)
                except ValueError:
                    warte = None
            if warte is None:
                warte = min(0.5 * (2 ** versuch) + random.uniform(0, 0.25), 5.0)
            logger.warning("decisions upstream %s, Versuch %d, warte %.2fs", code, versuch + 1, warte)
            await asyncio.sleep(warte)
            continue
        return code, data
    return code, data


# ── Uebersetzer Jev <-> Decisions ───────────────────────────────────────

def jev_zu_decisions(state: Any, fragen: dict, model: Optional[str] = None) -> dict:
    """Jev-Body {state, questions: {id: {type: noul|choice|score, ...}}}
    in einen Decisions-Body. Strukturierter state geht als `input_json`
    (Guide: nur in user-Nachrichten), Text als String."""
    if isinstance(state, str):
        eingabe: Any = state
    else:
        eingabe = [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
    liste = []
    for name, q in fragen.items():
        typ = q.get("type")
        neu: dict = {"name": name, "instructions": q.get("instructions")}
        if typ == "noul":
            neu["type"] = "predicate"
        elif typ == "choice":
            neu["type"] = "choice"
            neu["choices"] = [{"value": str(k), "description": str(v)}
                              for k, v in (q.get("criteria") or {}).items()]
        elif typ == "score":
            neu["type"] = "score"
            neu["levels"] = [{"label": str(s), "description": str(s)}
                             for s in (q.get("criteria") or [])]
        else:
            raise ValueError(f"unbekannter Jev-Typ {typ!r}")
        liste.append(neu)
    return {"model": model or default_model(), "input": eingabe, "questions": liste}


def decisions_zu_jev(antwort: dict) -> dict:
    """Decisions-Antwort in Jev-Form: answers als Map id -> {...},
    Wahrscheinlichkeiten als Map, bei score zusaetzlich `legend`."""
    aus: dict = {}
    for a in (antwort or {}).get("answers") or []:
        name, typ = a.get("name"), a.get("type")
        if typ == "predicate":
            aus[name] = {"type": "noul", "noul": a.get("probability")}
        elif typ == "choice":
            aus[name] = {"type": "choice", "choice": a.get("choice"),
                         "confidence": a.get("confidence"),
                         "probabilities": {p.get("value"): p.get("probability")
                                           for p in a.get("probabilities") or []}}
        elif typ == "score":
            probs = a.get("probabilities") or []
            aus[name] = {"type": "score", "score": a.get("score"),
                         "confidence": a.get("confidence"),
                         "legend": {str(p.get("value")): p.get("label") for p in probs},
                         "probabilities": {str(p.get("value")): p.get("probability") for p in probs}}
        else:
            # refusal und kuenftige Typen: nicht still verwerfen
            aus[name] = {k: v for k, v in a.items() if k != "name"}
    usage = (antwort or {}).get("usage") or {}
    return {"model": (antwort or {}).get("model"), "answers": aus,
            "usage": {"input_tokens": usage.get("input_tokens", 0),
                      "output_tokens": usage.get("output_tokens", 0),
                      "cached_tokens": ((usage.get("input_tokens_details") or {})
                                        .get("cached_tokens", 0))}}
