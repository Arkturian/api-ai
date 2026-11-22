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


class ZaehlerUnlesbar(Exception):
    """#3693 (Steward, 11572): Der Zaehler des Tages ist beschaedigt oder nicht
    lesbar. Dann geht nichts hinaus, und es faengt nichts wieder bei null an."""

    def __init__(self, datei: Path, grund: str, dienst: str = "decisions"):
        super().__init__(f"{dienst}-Zaehler unlesbar: {grund}")
        self.detail = {"error": f"{dienst}_zaehler_unlesbar", "grund": grund,
                       "hint": "Der Tageszaehler ist beschaedigt oder nicht lesbar. Bis ein Mensch "
                               f"{datei.name} prueft, bleibt {dienst} zu."}


def _beiseitelegen(p: Path) -> None:
    """Kopie zur Pruefung, einmal je Datei. Das Original bleibt liegen: Verschoben
    saehe der naechste Aufruf keine Datei und begaenne den Tag bei null."""
    try:
        if any(p.parent.glob(p.name + ".unlesbar-*")):
            return
        ziel = p.with_name(f"{p.name}.unlesbar-{datetime.now().strftime('%H%M%S')}")
        ziel.write_bytes(p.read_bytes())
    except OSError as e:
        logger.error("decisions: Zaehler nicht beiseitegelegt (%s)", type(e).__name__)


def _lesen() -> dict:
    p = _datei()
    if not p.exists():
        # Neuer Tag: die einzige Lage, in der der Zaehler bei null beginnt.
        return {"day": datetime.now().strftime("%Y-%m-%d"), "requests": 0, "questions": 0,
                "input_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "errors": 0, "by_caller": {}}
    try:
        d = json.loads(p.read_text())
        if not isinstance(d, dict) or not isinstance(d.get("by_caller"), dict):
            raise ValueError("keine Zaehlerform")
        float(d["cost_usd"]), int(d["requests"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        grund = "nicht_lesbar" if isinstance(e, OSError) else "beschaedigt"
        logger.critical("decisions: Tageszaehler %s ist %s (%s), Decisions bleibt zu",
                        p.name, grund, type(e).__name__)
        _beiseitelegen(p)
        raise ZaehlerUnlesbar(p, grund)
    return d


def _schreiben(d: dict) -> None:
    p = _datei()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    tmp.replace(p)


def _aufrufer_grenzen(env: str, caller: Optional[str]) -> Optional[tuple]:
    """Eigene Tagesgrenze je gepruefter Kennung (Alex 18.11.2026: storage-api
    bekommt eine eigene Kennung, damit Kosten und Rechte getrennt sind).
    Env als JSON, z. B. {"service:storage-api": {"requests": 2000, "usd": 0.1}}.
    Kaputtes JSON gilt als nicht gesetzt (die Hostgrenze bleibt)."""
    if not caller:
        return None
    try:
        tabelle = json.loads(os.getenv(env) or "{}")
        g = tabelle.get(caller) if isinstance(tabelle, dict) else None
    except ValueError:
        logger.error("%s ist kein gueltiges JSON, ignoriert", env)
        return None
    if not isinstance(g, dict):
        return None
    return int(g.get("requests") or 0), float(g.get("usd") or 0.0)


def rueckstellung() -> str:
    jetzt = datetime.now().astimezone()
    return (jetzt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class DecisionsBudget(Exception):
    def __init__(self, requests, cost_usd, max_req, max_usd, caller: Optional[str] = None):
        super().__init__("decisions daily budget")
        self.detail = {"error": "decisions_caller_budget_exceeded" if caller else "decisions_daily_budget_exceeded",
                       **({"caller": caller} if caller else {}), "requests_today": requests,
                       "cost_usd_today": cost_usd, "max_requests_per_day": max_req,
                       "max_usd_per_day": max_usd, "resets_at": rueckstellung(),
                       "hint": "Schutz gegen Endlosschleifen; Grenzen per "
                               "DECISIONS_DAILY_MAX_REQUESTS/DECISIONS_DAILY_MAX_USD."}


def budget_pruefen(caller: Optional[str] = None) -> None:
    max_req, max_usd = _grenzen()
    with _sperre():
        d = _lesen()
    if (max_req > 0 and d["requests"] >= max_req) or (max_usd > 0 and d["cost_usd"] >= max_usd):
        raise DecisionsBudget(d["requests"], round(d["cost_usd"], 6), max_req, max_usd)
    eigen = _aufrufer_grenzen("DECISIONS_CALLER_LIMITS", caller)
    if eigen:
        c = (d.get("by_caller") or {}).get(caller) or {}
        r, u = int(c.get("requests") or 0), float(c.get("cost_usd") or 0.0)
        if (eigen[0] > 0 and r >= eigen[0]) or (eigen[1] > 0 and u >= eigen[1]):
            raise DecisionsBudget(r, round(u, 6), eigen[0], eigen[1], caller=caller)


def max_zeichen() -> int:
    """#3693: Groesse je Aufruf, fuer /ai/decisions und /ai/jev?engine=openai.
    Vorgabe wie Jev (400 000), Kundeninstanzen z. B. 32 000."""
    try:
        wert = int((os.getenv("DECISIONS_MAX_ZEICHEN") or "").strip())
    except ValueError:
        return 400_000
    return wert if wert > 0 else 400_000


def schaetzung(zeichen: int, versuche: int) -> float:
    """Obergrenze der Kosten: jedes Zeichen als ein Token, voller Eingabepreis,
    jeder Versuch zahlt. Decisions rechnet nur Eingabe ab."""
    voll, _cache = _preise()
    return max(int(zeichen), 0) * voll / 1_000_000.0 * max(int(versuche), 1)


def _reserviert(z: dict) -> tuple:
    return int(z.get("reserved_requests") or 0), float(z.get("reserved_usd") or 0.0)


def reservieren(caller: str, betrag: float) -> None:
    """#3693: Unter der Sperre nur aufrufen, wenn gebucht + reserviert + Schaetzung
    die Grenze nicht ueberschreiten. Sonst DecisionsBudget (429). Eine Reservierung,
    die ein abgestuerzter Prozess hinterlaesst, verfaellt mit dem Tag; sie sperrt
    nur mehr, nie weniger."""
    max_req, max_usd = _grenzen()
    eigen = _aufrufer_grenzen("DECISIONS_CALLER_LIMITS", caller)
    with _sperre():
        d = _lesen()
        rr, ru = _reserviert(d)
        if (max_req > 0 and d["requests"] + rr + 1 > max_req) or \
                (max_usd > 0 and d["cost_usd"] + ru + betrag > max_usd):
            raise DecisionsBudget(d["requests"], round(d["cost_usd"], 6), max_req, max_usd)
        c = d["by_caller"].setdefault(caller, {"requests": 0, "questions": 0, "input_tokens": 0,
                                               "cached_tokens": 0, "cost_usd": 0.0, "errors": 0})
        if eigen:
            cr, cu = _reserviert(c)
            if (eigen[0] > 0 and int(c.get("requests") or 0) + cr + 1 > eigen[0]) or \
                    (eigen[1] > 0 and float(c.get("cost_usd") or 0.0) + cu + betrag > eigen[1]):
                raise DecisionsBudget(int(c.get("requests") or 0), round(float(c.get("cost_usd") or 0.0), 6),
                                      eigen[0], eigen[1], caller=caller)
        for z in (d, c):
            zr, zu = _reserviert(z)
            z["reserved_requests"], z["reserved_usd"] = zr + 1, zu + betrag
        _schreiben(d)


def abrechnen(caller: str, betrag: float, fragen: int, usage: Optional[dict], cost_usd: float,
              fehler: bool = False) -> None:
    """Gibt die Reservierung frei und bucht die tatsaechlichen Kosten in einem Schritt."""
    with _sperre():
        d = _lesen()
        c = d["by_caller"].setdefault(caller, {"requests": 0, "questions": 0, "input_tokens": 0,
                                               "cached_tokens": 0, "cost_usd": 0.0, "errors": 0})
        for z in (d, c):
            zr, zu = _reserviert(z)
            z["reserved_requests"], z["reserved_usd"] = max(zr - 1, 0), max(zu - betrag, 0.0)
        _schreiben(d)
    buchen(caller, fragen, usage, cost_usd, fehler=fehler)


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
    try:
        with _sperre():
            d = _lesen()
    except ZaehlerUnlesbar as e:
        return {**e.detail, "budget_exhausted": True, "resets_at": rueckstellung(),
                "limits": {"max_requests_per_day": max_req, "max_usd_per_day": max_usd,
                           "max_chars_per_request": max_zeichen()}, "scope": "per_host"}
    voll, cache = _preise()
    d["budget_exhausted"] = bool((max_req > 0 and d["requests"] >= max_req)
                                 or (max_usd > 0 and d["cost_usd"] >= max_usd))
    d["resets_at"] = rueckstellung()
    d["cost_usd"] = round(d["cost_usd"], 8)
    d["limits"] = {"max_requests_per_day": max_req, "max_usd_per_day": max_usd,
                   "max_chars_per_request": max_zeichen()}
    d["reserved_usd"] = round(float(d.get("reserved_usd") or 0.0), 8)
    d["reserved_requests"] = int(d.get("reserved_requests") or 0)
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
