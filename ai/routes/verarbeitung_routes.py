"""#4095: POST /ai/verarbeitung — zustandslose Verarbeitung nach der Konfiguration der Instanz.

Aufrufer schicken Prompt, System, optional ein JSON-Schema und Grenzen. Den
Anbieter waehlt KI_VERARBEITUNG (ai/services/verarbeitung.py), nie der Aufrufer:
Felder wie `model` werden mit 422 abgewiesen.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from ai import auth_guard as _ag
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.routes import kette_routes as kr
from ai.routes import text_ai_routes as t
from ai.services import kette as k
from ai.services import verarbeitung as v

logger = logging.getLogger("api-ai.verarbeitung")
router = APIRouter()


class VerarbeitungRumpf(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    prompt: str = Field(min_length=1, max_length=200000)
    system: Optional[str] = Field(default=None, max_length=20000)
    json_schema: Optional[dict] = Field(default=None, alias="schema")
    budget_s: float = Field(default=60.0, ge=5)
    # Wirkt nur beim Schluesselweg (openai); auf den Abo-Wegen meldet die Antwort es.
    max_ausgabe: Optional[int] = Field(default=None, ge=16, le=32000)

    @model_validator(mode="after")
    def _grenzen(self):
        hoechstens = k.budget_hoechstens()
        if not self.budget_s <= hoechstens:
            raise ValueError(f"budget_s hoechstens {hoechstens:g} s")
        return self


async def _openai(body: VerarbeitungRumpf, system: Optional[str], modell: Optional[str]):
    p = t.Prompt(prompt=body.prompt, system=system, max_tokens=body.max_ausgabe)
    return await t._openai_einmal(p, modell)


@router.get("/verarbeitung/stand", dependencies=[Depends(_ag.betreiber_oder_instanz)])
async def verarbeitung_stand():
    """Welche KI_VERARBEITUNG wirkt (fuer Rollout und Messung, #4095). Nicht in
    /health: das zeigt sonst, welche Abos ein Kunde hat (#3639)."""
    return v.stand()


@router.post("/verarbeitung")
async def verarbeitung_endpoint(body: VerarbeitungRumpf,
                                authorization: Optional[str] = Header(default=None),
                                x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name")):
    aufrufer = kr._aufrufer(authorization, x_agent_name)
    try:
        kf = v.konfig()
    except HTTPException as e:
        logger.warning("verarbeitung: caller=%s %s", aufrufer, (e.detail or {}).get("error"))
        raise
    system = body.system
    if body.json_schema is not None:
        system = (system + "\n\n" if system else "") + v.schema_anweisung(body.json_schema)

    nicht = []
    try:
        if kf.anbieter == "openai":
            out = await _openai(body, system, kf.modell)
            weg, modell, versucht = "openai", getattr(out, "model", None) or kf.modell, None
        else:
            weg_fest, vorgabe = v.ANBIETER[kf.anbieter]
            reihe = [weg_fest] if weg_fest else None
            wahl = {weg_fest: {"model": kf.modell or vorgabe}} if weg_fest else {}
            erg = await k.laufen(kr.rufer(body.prompt, system), wahl, body.budget_s, False,
                                 t.CLI_FRIST.set, t.CLI_FRIST.reset, reihe=reihe)
            out, weg, modell, versucht = erg["out"], erg["weg"], erg["model"], erg["versucht"]
            if body.max_ausgabe is not None:
                nicht.append("max_ausgabe")
    except HTTPException as e:
        logger.info("verarbeitung: caller=%s anbieter=%s status=%s", aufrufer, kf.anbieter, e.status_code)
        raise

    text = getattr(out, "response", None) or ""
    antwort = {"text": text, "anbieter": kf.anbieter, "weg": weg, "modell": modell}
    if versucht is not None:
        antwort["versucht"] = versucht
    if nicht:
        antwort["nicht_angewendet"] = nicht
    if body.json_schema is not None:
        try:
            antwort["daten"] = v.json_lesen(text, body.json_schema)
        except ValueError as e:
            logger.info("verarbeitung: caller=%s anbieter=%s antwort_kein_json %s", aufrufer, kf.anbieter, e)
            raise HTTPException(status_code=502, detail={
                "error": "antwort_kein_json", "grund": str(e), "anbieter": kf.anbieter,
                "weg": weg, "modell": modell, "text": text[:2000]}) from None
    logger.info("verarbeitung: caller=%s anbieter=%s weg=%s model=%s", aufrufer, kf.anbieter, weg, modell)
    return antwort
