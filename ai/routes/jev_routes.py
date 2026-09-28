"""POST /ai/jev — Jev (TypeSafe System One), Anforderung Content #5083."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from ai.services import jev_service as s

logger = logging.getLogger(__name__)
router = APIRouter()

_TYPEN = ("noul", "choice", "score")
# Upstream: 64k Tokens je Anfrage, 32k fuer state + laengste Frage. Hier nur
# eine grobe Zeichenbremse gegen offensichtlich zu Grosses (~4 Zeichen je
# Token); den genauen Entscheid trifft Jev selbst (4xx wird durchgereicht).
_MAX_ZEICHEN = 400_000


class JevRequest(BaseModel):
    state: Union[str, Dict[str, Any], list] = Field(description="Zu beurteilender Zustand: Text oder strukturierte Daten.")
    questions: Dict[str, Any] = Field(description="Map frage_id -> {type: noul|choice|score, instructions, criteria?}")
    model: Optional[str] = Field(default=None, description="Vorgabe jev-latest; feste Version erlaubt, z. B. jev-1.13.0")


def _pruefen(req: JevRequest) -> None:
    if not req.questions:
        raise HTTPException(422, {"error": "questions_required"})
    for qid, q in req.questions.items():
        if not isinstance(q, dict) or q.get("type") not in _TYPEN:
            raise HTTPException(422, {"error": "invalid_question_type", "question": qid, "allowed": list(_TYPEN)})
        if q.get("instructions") in (None, "", [], {}):
            raise HTTPException(422, {"error": "instructions_required", "question": qid})
        if q["type"] == "choice":
            crit = q.get("criteria")
            if not isinstance(crit, dict) or not crit:
                raise HTTPException(422, {"error": "criteria_required", "question": qid})
            if len(crit) > 255:
                raise HTTPException(422, {"error": "too_many_choice_options", "question": qid,
                                          "options": len(crit), "max": 255})
        if q["type"] == "score":
            crit = q.get("criteria")
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise HTTPException(422, {"error": "score_levels_2_to_10", "question": qid})


@router.post("/jev")
async def jev_endpoint(req: JevRequest, x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name")):
    """Typisierte Urteile (noul/choice/score) ueber TypeSafe System One.
    Antwort unveraendert, ergaenzt um cost_usd und latency_ms. X-Agent-Name
    (vom MCP-Gateway aus dem geprueften JWT) dient nur der Zuordnung im
    Nutzungslog, nie als Berechtigung."""
    _pruefen(req)
    body = {"state": req.state, "model": req.model or s.default_model(), "questions": req.questions}
    groesse = len(json.dumps(body, ensure_ascii=False))
    if groesse > _MAX_ZEICHEN:
        raise HTTPException(413, {"error": "request_too_large", "chars": groesse, "max_chars": _MAX_ZEICHEN,
                                  "hint": "Jev: 64k Tokens je Anfrage, davon 32k fuer state + laengste Frage."})
    if not s.schluessel():
        raise HTTPException(503, {"error": "typesafe_key_missing"})
    try:
        s.budget_pruefen()
    except s.JevBudget as e:
        raise HTTPException(429, e.detail)
    caller = (x_agent_name or "").strip()[:64] or "(unbekannt)"
    t0 = time.monotonic()
    try:
        code, data, _hdr = await s.aufrufen(body)
    except Exception as e:   # Netzfehler / Timeout
        s.buchen(caller, None, 0, 0, 0.0, fehler=True)
        raise HTTPException(502, {"error": "jev_upstream_unreachable", "exc": type(e).__name__})
    latenz = int((time.monotonic() - t0) * 1000)
    if code != 200:
        s.buchen(caller, None, 0, 0, 0.0, fehler=True)
        status = code if 400 <= code < 500 else 502
        raise HTTPException(status, {"error": "jev_upstream_error", "upstream_status": code,
                                     "upstream_body": data, "latency_ms": latenz})
    usage = (data or {}).get("usage") or {}
    kosten = s.kosten(usage.get("input_tokens", 0))
    s.buchen(caller, data.get("model"), usage.get("input_tokens", 0), usage.get("output_tokens", 0), kosten)
    logger.info("jev ok caller=%s model=%s in=%s out=%s %dms", caller, data.get("model"),
                usage.get("input_tokens"), usage.get("output_tokens"), latenz)
    return {**data, "cost_usd": kosten, "latency_ms": latenz}


@router.get("/jev/cost-status")
async def jev_cost_status():
    """Tageszaehler dieses Hosts: Anfragen, Tokens, Kosten, je Aufrufer,
    gesehene Jev-Versionen, Grenzen."""
    return s.status()
