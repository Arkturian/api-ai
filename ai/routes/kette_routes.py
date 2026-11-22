"""#3658: POST /ai/kette — Text ueber die Abos in der Reihenfolge aus KI_KETTE.

Ohne KI_KETTE antwortet der Pfad 503 `kette_aus`, und zwar in der Middleware
`_kette_tor` (main.py) vor dem Lesen des Rumpfs. So startet er auf aiserver und
den Kundeninstanzen nie einen Modellaufruf. Vertrag: #3658 (11327, 11330, 11331).
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter
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
    budget_s: float = Field(default=k.BUDGET_VORGABE_S, ge=5, le=k.BUDGET_HOECHSTENS_S)
    nur_erster_weg: bool = False
    # Angenommen, wirken auf den Abo-Wegen aber nicht (11330): Antwort sagt es.
    max_tokens: Optional[int] = Field(default=None, ge=1, le=100000)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def _wege_pruefen(self):
        namen = [w.weg for w in (self.wege or [])]
        if len(namen) != len(set(namen)):
            raise ValueError("jeder Weg hoechstens einmal")
        for w in self.wege or []:
            if w.weg == "claude" and w.model and "haiku" not in w.model.lower():
                raise ValueError("claude laeuft in der Kette nur mit Haiku")
        return self


@router.post("/kette")
async def kette_endpoint(body: KetteRumpf):
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

    erg = await k.laufen(rufen, wahl, body.budget_s, body.nur_erster_weg,
                         t.CLI_FRIST.set, t.CLI_FRIST.reset)
    out = erg["out"]
    antwort = out.model_dump() if hasattr(out, "model_dump") else dict(out)
    antwort.update(routed_to=erg["weg"], model=erg["model"], versucht=erg["versucht"])
    nicht = [n for n in ("max_tokens", "temperature") if getattr(body, n) is not None]
    if nicht:
        antwort["nicht_angewendet"] = nicht
    logger.info("kette: routed_to=%s model=%s versucht=%s", erg["weg"], erg["model"],
                ",".join(f"{v['weg']}:{v['code']}" for v in erg["versucht"]))
    return antwort
