"""POST /ai/decisions — OpenAI Decisions API (Auftrag Alex 26.10.2026).

Nativer Vertrag von OpenAI, durchgereicht: `input` (Text oder
Nachrichtenliste), `questions` als Liste mit `name`, optional
`prompt_cache_key`, `prompt_cache_options`. Ergaenzt um `cost_usd` und
`latency_ms`. Gleiches Tor wie /ai/jev (gepruefetes Agenten-JWT, siehe
main.py), gleiches Tagesbudget-Muster.

Wer heute Jev ruft und vergleichen will, braucht diese Route nicht:
`POST /ai/jev?engine=openai` nimmt den Jev-Body und antwortet in Jev-Form.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ai.services import decisions_service as s

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_ZEICHEN = 1_000_000


class DecisionsRequest(BaseModel):
    input: Union[str, List[Any]] = Field(description="Gemeinsame Evidenz: Text oder Nachrichtenliste (Responses-Format).")
    questions: List[Dict[str, Any]] = Field(description="1-400 Fragen {name, type: predicate|choice|score, instructions, choices|levels}")
    model: Optional[str] = Field(default=None, description="Vorgabe gpt-6-luna")
    prompt_cache_key: Optional[str] = Field(default=None, max_length=256)
    prompt_cache_options: Optional[Dict[str, Any]] = None


def fragen_pruefen(fragen: List[Dict[str, Any]]) -> None:
    if not fragen:
        raise HTTPException(422, {"error": "questions_required"})
    if len(fragen) > s.MAX_FRAGEN:
        raise HTTPException(422, {"error": "too_many_questions", "count": len(fragen), "max": s.MAX_FRAGEN})
    namen = set()
    for i, q in enumerate(fragen):
        name = q.get("name") if isinstance(q, dict) else None
        if not isinstance(name, str) or not name:
            raise HTTPException(422, {"error": "question_name_required", "index": i})
        if name in namen:
            raise HTTPException(422, {"error": "duplicate_question_name", "name": name})
        namen.add(name)
        if q.get("type") not in s.TYPEN:
            raise HTTPException(422, {"error": "invalid_question_type", "name": name, "allowed": list(s.TYPEN)})
        if q.get("instructions") in (None, "", []):
            raise HTTPException(422, {"error": "instructions_required", "name": name})
        if q["type"] == "choice" and not (isinstance(q.get("choices"), list) and q["choices"]):
            raise HTTPException(422, {"error": "choices_required", "name": name})
        if q["type"] == "score":
            lv = q.get("levels")
            if not isinstance(lv, list) or not 2 <= len(lv) <= 10:
                raise HTTPException(422, {"error": "score_levels_2_to_10", "name": name})


async def ausfuehren(body: dict, caller: str, versuche: int = 3) -> dict:
    """Gemeinsamer Kern fuer /ai/decisions und /ai/jev?engine=openai:
    Groesse, Schluessel, Budget, Aufruf, Buchung. -> Upstream-Antwort
    plus cost_usd und latency_ms."""
    groesse = len(json.dumps(body, ensure_ascii=False))
    grenze = min(_MAX_ZEICHEN, s.max_zeichen())
    if groesse > grenze:
        raise HTTPException(413, {"error": "request_too_large", "chars": groesse, "max_chars": grenze})
    if not s.schluessel():
        raise HTTPException(503, {"error": "openai_key_missing"})
    # #3693: Kosten vor dem Aufruf reservieren, unter der Sperre. So haelt die
    # Grenze auch bei gleichzeitigen Aufrufen und knapp unter dem Deckel.
    betrag = s.schaetzung(groesse, versuche)
    try:
        s.reservieren(caller, betrag)
    except s.DecisionsBudget as e:
        from datetime import datetime as _dt
        warte = max(1, int((_dt.fromisoformat(e.detail["resets_at"]) - _dt.now().astimezone()).total_seconds()))
        raise HTTPException(429, e.detail, headers={"Retry-After": str(warte)})
    n = len(body.get("questions") or [])
    t0 = time.monotonic()
    try:
        code, data = await s.aufrufen(body, versuche=versuche)
    except BaseException as e:
        # Nicht gesendet (Verbindung kam nicht zustande): 0. Sonst kann OpenAI die
        # Anfrage berechnet haben (Zeitueberschreitung, Abbruch): Schaetzung buchen.
        nicht_gesendet = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout))
        s.abrechnen(caller, betrag, n, None, 0.0 if nicht_gesendet else betrag, fehler=True)
        if isinstance(e, asyncio.CancelledError):
            raise
        raise HTTPException(502, {"error": "decisions_upstream_unreachable", "exc": type(e).__name__})
    latenz = int((time.monotonic() - t0) * 1000)
    if code != 200:
        s.abrechnen(caller, betrag, n, None, 0.0, fehler=True)
        status = code if 400 <= code < 500 else 502
        raise HTTPException(status, {"error": "decisions_upstream_error", "upstream_status": code,
                                     "upstream_body": data, "latency_ms": latenz})
    usage = (data or {}).get("usage") or {}
    preis = s.kosten(usage)
    s.abrechnen(caller, betrag, n, usage, preis)
    logger.info("decisions ok caller=%s q=%d in=%s cached=%s %dms", caller, n, usage.get("input_tokens"),
                (usage.get("input_tokens_details") or {}).get("cached_tokens"), latenz)
    return {**data, "cost_usd": preis, "latency_ms": latenz}


def aufrufer(authorization: Optional[str], x_agent_name: Optional[str]) -> str:
    """Etikett fuers Nutzungslog, nie Recht — wie bei /ai/jev."""
    from ai.routes import jev_routes as _jev
    caller = ""
    if isinstance(authorization, str) and authorization:
        caller = _jev._jwt_sub_geprueft(authorization) or "(jwt-ungueltig)"
    if not caller and isinstance(x_agent_name, str):
        caller = x_agent_name.strip()[:64]
    if caller.startswith("agent:"):
        caller = caller[len("agent:"):]
    return caller or "(unbekannt)"


@router.post("/decisions")
async def decisions_endpoint(req: DecisionsRequest,
                             x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name"),
                             authorization: Optional[str] = Header(default=None),
                             max_attempts: int = Query(default=3, ge=1, le=3)):
    """Typisierte Urteile ueber die OpenAI Decisions API (gpt-6-luna)."""
    fragen_pruefen(req.questions)
    body: dict = {"model": req.model or s.default_model(), "input": req.input, "questions": req.questions}
    if req.prompt_cache_key:
        body["prompt_cache_key"] = req.prompt_cache_key
    if req.prompt_cache_options:
        body["prompt_cache_options"] = req.prompt_cache_options
    return await ausfuehren(body, aufrufer(authorization, x_agent_name),
                            versuche=max_attempts if isinstance(max_attempts, int) else 3)


@router.get("/decisions/cost-status")
async def decisions_cost_status():
    """Tageszaehler dieses Hosts: Anfragen, Fragen, Tokens (davon gecacht),
    Kosten, je Aufrufer, Grenzen."""
    return s.status()
