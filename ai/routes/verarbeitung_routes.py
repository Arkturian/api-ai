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


async def _kunde(body: "VerarbeitungRumpf", mk: "v.MandantKonfig", aufrufer: str) -> dict:
    """#4135 A1.3: Schluesselweg eines Kunden-Mandanten. Schluessel je Aufruf aus dem
    Tresor (Zweck verarbeitung), Tages- und Monatsdeckel des Mandanten, eigene Zaehler.
    Jeder Fehler endet hier (503/429/502) -- kein anderer Anbieter, kein Kern-Schluessel."""
    from ai.services import openai_text as ot
    from ai.services import tresor
    system = body.system
    if body.json_schema is not None:
        system = (system + "\n\n" if system else "") + v.schema_anweisung(body.json_schema)
    wert = await tresor.lesen(mk.verweis, mk.mandant, zweck="verarbeitung", aufrufer=aufrufer, anbieter="openai")
    marke = ot.KONTO.set(ot.Konto(mandant=mk.mandant, schluessel=wert, tag_usd=mk.tag_usd, monat_usd=mk.monat_usd))
    try:
        out = await _openai(body, system, mk.modell)
    finally:
        ot.KONTO.reset(marke)
        del wert
    text = getattr(out, "response", None) or ""
    modell = getattr(out, "model", None) or mk.modell
    antwort = {"text": text, "anbieter": "openai", "weg": "openai", "modell": modell,
               "konto": {"mandant": mk.mandant, "art": "schluessel_des_mandanten"}}
    if body.json_schema is not None:
        try:
            antwort["daten"] = v.json_lesen(text, body.json_schema)
        except ValueError as e:
            raise HTTPException(status_code=502, detail={
                "error": "antwort_kein_json", "grund": str(e), "anbieter": "openai", "weg": "openai",
                "modell": modell, "text": text[:2000]}) from None
    logger.info("verarbeitung: caller=%s mandant=%s anbieter=openai(kunde) model=%s", aufrufer, mk.mandant, modell)
    return antwort


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
    # #4135 (Plan 6567 A1): Mandant NUR aus dem geprueften JWT (Middleware).
    from ai.services import aufruf
    mandant = aufruf.mandant()
    try:
        if mandant is None:
            if v.jwt_pflicht() == "an":
                raise HTTPException(status_code=401, detail={
                    "error": "jwt_required",
                    "hint": "/ai/verarbeitung verlangt ein geprueftes JWT (Dienst oder Agent)."})
            logger.warning("verarbeitung OHNE JWT caller=%s (JWT-Pflicht: beobachten, Kern-Einstellung)", aufrufer)
            kf = v.konfig()
        elif mandant != v.betreiber():
            mk = v.mandant_konfig(mandant)          # 503 ohne/ungueltige Zuweisung, "keine"
            return await _kunde(body, mk, aufrufer)
        else:
            kf = v.konfig()
    except HTTPException as e:
        d = e.detail if isinstance(e.detail, dict) else {}
        logger.warning("verarbeitung: caller=%s mandant=%s %s", aufrufer, mandant or "-", d.get("error"))
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
    logger.info("verarbeitung: caller=%s mandant=%s anbieter=%s weg=%s model=%s", aufrufer, mandant or "-",
                kf.anbieter, weg, modell)
    return antwort
