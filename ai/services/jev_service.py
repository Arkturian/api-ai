"""Jev — TypeSafe System One (Anforderung Content #5083, Jev/Alex 28.09.2026).

Preis belegt: docs.typesafe.ai/models.md, 0,042 USD je Mio Eingabetoken,
Ausgabe frei (jev-1.13.0 = jev-latest).

Durchreichen statt nachbauen: api-ai validiert nur grob, ruft
POST https://api.typesafe.ai/v1/systemone und ergaenzt Kosten und Latenz.
Upstream-Doku: https://docs.typesafe.ai/api.md (401, 422, 429, 529).

Kostenschutz: kein confirm_api_billing (0,042 USD je Mio Eingabetoken,
Ausgabe frei, laut Jev), stattdessen ein Tagesbudget gegen Endlosschleifen
(Anfragen und USD) und Nutzung je Aufrufer. Zaehler je Host, eine Datei
je Tag unter JEV_USAGE_DIR. Der Schluessel wird nie geloggt oder
zurueckgegeben.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

UPSTREAM_URL = "https://api.typesafe.ai/v1/systemone"
_SCHLUESSEL_DATEI = "/home/alex/.secrets/typesafe-api.key"


def _preis_je_mio() -> float:
    return float(os.getenv("JEV_PRICE_PER_1M_INPUT_USD", "0.042"))


def default_model() -> str:
    return os.getenv("JEV_DEFAULT_MODEL", "jev-latest")


def schluessel() -> Optional[str]:
    k = (os.getenv("TYPESAFE_API_KEY") or "").strip()
    if k:
        return k
    try:
        return Path(_SCHLUESSEL_DATEI).read_text().strip() or None
    except OSError:
        return None


# ── Tageszaehler ────────────────────────────────────────────────────────

def _dir() -> Path:
    return Path(os.getenv("JEV_USAGE_DIR", "/var/lib/api-ai"))


def _datei() -> Path:
    return _dir() / f"jev_usage_{datetime.now().strftime('%Y-%m-%d')}.json"


def _grenzen() -> tuple:
    return (int(os.getenv("JEV_DAILY_MAX_REQUESTS", "50000")),
            float(os.getenv("JEV_DAILY_MAX_USD", "2")))


@contextlib.contextmanager
def _sperre():
    _dir().mkdir(parents=True, exist_ok=True)
    with open(_dir() / ".jev_usage.lock", "w") as f:
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
    return {"day": datetime.now().strftime("%Y-%m-%d"), "requests": 0, "input_tokens": 0,
            "output_tokens": 0, "cost_usd": 0.0, "errors": 0, "by_caller": {}, "models_seen": {}}


def _schreiben(d: dict) -> None:
    p = _datei()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    tmp.replace(p)


def budget_pruefen() -> None:
    """Vor dem Aufruf: Tagesgrenze erreicht -> JevBudget."""
    max_req, max_usd = _grenzen()
    with _sperre():
        d = _lesen()
    if (max_req > 0 and d["requests"] >= max_req) or (max_usd > 0 and d["cost_usd"] >= max_usd):
        raise JevBudget(d["requests"], round(d["cost_usd"], 6), max_req, max_usd)


def buchen(caller: str, model: Optional[str], input_tokens: int, output_tokens: int,
           cost_usd: float, fehler: bool = False) -> None:
    with _sperre():
        d = _lesen()
        d["requests"] += 1
        d["input_tokens"] += int(input_tokens or 0)
        d["output_tokens"] += int(output_tokens or 0)
        d["cost_usd"] += float(cost_usd or 0.0)
        if fehler:
            d["errors"] += 1
        c = d["by_caller"].setdefault(caller, {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                                              "cost_usd": 0.0, "errors": 0})
        c["requests"] += 1
        c["input_tokens"] += int(input_tokens or 0)
        c["output_tokens"] += int(output_tokens or 0)
        c["cost_usd"] += float(cost_usd or 0.0)
        if fehler:
            c["errors"] += 1
        if model:
            d["models_seen"][model] = d["models_seen"].get(model, 0) + 1
        d["last_updated"] = datetime.now().isoformat()
        _schreiben(d)


def status() -> dict:
    max_req, max_usd = _grenzen()
    with _sperre():
        d = _lesen()
    d["cost_usd"] = round(d["cost_usd"], 8)
    d["limits"] = {"max_requests_per_day": max_req, "max_usd_per_day": max_usd}
    d["price_per_1m_input_usd"] = _preis_je_mio()
    d["scope"] = "per_host"
    return d


class JevBudget(Exception):
    def __init__(self, requests, cost_usd, max_req, max_usd):
        super().__init__("jev daily budget")
        self.detail = {"error": "jev_daily_budget_exceeded", "requests_today": requests,
                       "cost_usd_today": cost_usd, "max_requests_per_day": max_req,
                       "max_usd_per_day": max_usd,
                       "hint": "Schutz gegen Endlosschleifen; Grenzen per JEV_DAILY_MAX_REQUESTS/JEV_DAILY_MAX_USD."}


# ── Aufruf ──────────────────────────────────────────────────────────────

async def aufrufen(body: dict, timeout_s: float = 10.0, versuche: int = 3) -> tuple:
    """-> (status_code, json_body, headers). 429/529 mit Backoff, retry-after
    wird beachtet; alles andere geht beim ersten Mal zurueck."""
    key = schluessel()
    if not key:
        return 0, None, {}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for versuch in range(versuche):
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(UPSTREAM_URL, json=body, headers=headers)
        if r.status_code in (429, 529) and versuch < versuche - 1:
            warte = None
            ra = (r.headers or {}).get("retry-after")
            if ra:
                try:
                    warte = min(float(ra), 10.0)
                except ValueError:
                    warte = None
            if warte is None:
                warte = min(0.5 * (2 ** versuch) + random.uniform(0, 0.25), 5.0)
            logger.warning("jev upstream %s, Versuch %d, warte %.2fs", r.status_code, versuch + 1, warte)
            await asyncio.sleep(warte)
            continue
        try:
            data = r.json()
        except Exception:
            data = {"raw": (r.text or "")[:500]}
        return r.status_code, data, dict(r.headers or {})
    return r.status_code, data, {}


def kosten(input_tokens: int) -> float:
    return int(input_tokens or 0) * _preis_je_mio() / 1_000_000.0
