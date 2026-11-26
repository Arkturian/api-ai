"""#3658: POST /ai/kette — Text ueber die Abos in der Reihenfolge aus KI_KETTE.

Ohne KI_KETTE antwortet der Pfad 503 `kette_aus`, und zwar in der Middleware
`_kette_tor` (main.py) vor dem Lesen des Rumpfs. So startet er auf aiserver und
den Kundeninstanzen nie einen Modellaufruf. Vertrag: #3658 (11327, 11330, 11331).
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.routes import text_ai_routes as t
from ai.services import kette as k

logger = logging.getLogger("api-ai.kette")
router = APIRouter()


class WegWahl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weg: Literal["agy", "codex", "claude"]
    model: Optional[str] = Field(default=None, max_length=64)
    effort: Optional[str] = Field(default=None, max_length=16)


class KetteRumpf(BaseModel):
    """Nur Text. Bilder, Werkzeuge, Persona und Verlauf gibt es hier nicht (422)."""
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=200000)
    system: Optional[str] = Field(default=None, max_length=20000)
    wege: Optional[List[WegWahl]] = None
    budget_s: float = Field(default=k.BUDGET_VORGABE_S, ge=5)   # Obergrenze: k.budget_hoechstens()
    nur_erster_weg: bool = False
    # Angenommen, wirken auf den Abo-Wegen aber nicht (11330): Antwort sagt es.
    max_tokens: Optional[int] = Field(default=None, ge=1, le=100000)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def _wege_pruefen(self):
        hoechstens = k.budget_hoechstens()
        if not self.budget_s <= hoechstens:
            raise ValueError(f"budget_s hoechstens {hoechstens:g} s ({k.BUDGET_HOECHSTENS_ENV})")
        namen = [w.weg for w in (self.wege or [])]
        if len(namen) != len(set(namen)):
            raise ValueError("jeder Weg hoechstens einmal")
        for w in self.wege or []:
            if w.weg == "claude" and w.model and "haiku" not in w.model.lower():
                raise ValueError("claude laeuft in der Kette nur mit Haiku")
        return self


def _aufrufer(authorization: Optional[str], x_agent_name: Optional[str]) -> str:
    """Wer ruft? Nur Etikett fuers Journal, nie Recht (Codepilot 26.11.2026:
    Dienste wie service:codepilot-api und service:project-api sollen im Journal
    zuordenbar sein). Wie /ai/jev: geprueftes JWT vor dem Kopf des Gateways;
    ein vorgelegtes, aber ungueltiges JWT faellt nicht auf den Kopf zurueck."""
    if not isinstance(authorization, str):
        authorization = None
    if not isinstance(x_agent_name, str):
        x_agent_name = None
    wer = ""
    if authorization and authorization.lower().startswith("bearer "):
        from ai.services import jwt_pruefung as jp
        claims = jp.claims_oder_none(authorization)
        wer = str((claims or {}).get("sub") or "").strip()[:64] or "(jwt-ungueltig)"
    if not wer:
        wer = (x_agent_name or "").strip()[:64]
    if wer.startswith("agent:"):
        wer = wer[len("agent:"):]
    return wer or "(unbekannt)"


@router.post("/kette")
async def kette_endpoint(body: KetteRumpf,
                         authorization: Optional[str] = Header(default=None),
                         x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name")):
    aufrufer = _aufrufer(authorization, x_agent_name)
    wahl = {w.weg: w.model_dump() for w in (body.wege or [])}

    async def rufen(weg: str, modell: str, effort: Optional[str]):
        if weg == "agy":
            p = t.Prompt(prompt=body.prompt, system=body.system, effort=effort)
            return await t.gemini_endpoint(p, model=modell, api_key="placeholder")
        if weg == "codex":
            p = t.Prompt(prompt=body.prompt, system=body.system, effort=effort, tools=False)
            return await t.chatgpt_endpoint(p, model=modell, api_key="placeholder", request=None, x_api_key=None)
        p = t.Prompt(prompt=body.prompt, system=body.system, effort=effort)
        return await t._claude_mit_rechten(p, modell, direkt=True)

    try:
        erg = await k.laufen(rufen, wahl, body.budget_s, body.nur_erster_weg,
                             t.CLI_FRIST.set, t.CLI_FRIST.reset)
    except HTTPException as e:
        logger.info("kette: caller=%s status=%s", aufrufer, e.status_code)
        raise
    out = erg["out"]
    antwort = out.model_dump() if hasattr(out, "model_dump") else dict(out)
    antwort.update(routed_to=erg["weg"], model=erg["model"], versucht=erg["versucht"])
    nicht = [n for n in ("max_tokens", "temperature") if getattr(body, n) is not None]
    if nicht:
        antwort["nicht_angewendet"] = nicht
    logger.info("kette: caller=%s routed_to=%s model=%s versucht=%s", aufrufer, erg["weg"], erg["model"],
                ",".join(f"{v['weg']}:{v['code']}" for v in erg["versucht"]))
    return antwort
