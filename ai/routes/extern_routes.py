"""#3656: POST /ai/extern/hydrogen — nur Text, nur Hydrogen, je Aufrufer gezaehlt.

Der Schluessel wird in der Middleware `_extern_tor` (main.py) vor dem Lesen des
Rumpfs geprueft. Hier steht das Ergebnis in `request.state.extern`.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai.routes import text_ai_routes as t
from ai.services import ki_wege

logger = logging.getLogger("api-ai.extern")
router = APIRouter()

MODELL_VORGABE = "gemini-4-hydrogen-low"
_MODELL = re.compile(r"gemini-4-hydrogen(-(low|medium|high|xhigh))?")
PROMPT_MAX = 40000
SYSTEM_MAX = 8000


class ExternText(BaseModel):
    """Bewusst schmal: keine Bilder, keine Werkzeuge, keine Persona, kein Verlauf.
    Jedes weitere Feld ergibt 422, damit nichts still mitlaeuft."""
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=PROMPT_MAX)
    system: Optional[str] = Field(default=None, max_length=SYSTEM_MAX)
    model: Optional[str] = None

    @field_validator("model")
    @classmethod
    def _nur_hydrogen(cls, v):
        if v is None:
            return v
        if not _MODELL.fullmatch(v):
            raise ValueError("nur gemini-4-hydrogen, -low, -medium, -high oder -xhigh")
        return v


@router.post("/extern/hydrogen", response_model=t.AIResponse)
async def extern_hydrogen(body: ExternText, request: Request):
    zugang = getattr(request.state, "extern", None)
    if not zugang:
        # Nie ohne die Middleware erreichbar; falls doch, zu statt offen.
        raise HTTPException(status_code=503, detail={"error": "extern_aus"})
    kennung, tag_max = zugang
    modell = body.model or MODELL_VORGABE
    ki_wege.tag_buchen(f"extern_hydrogen_{kennung}", tag_max)
    # tools bleibt leer: /ai/gemini kennt den Schalter nicht, Werkzeuge sind
    # auf api-ai ohnehin aus (#2937).
    prompt = t.Prompt(prompt=body.prompt, system=body.system)
    out = await t.gemini_endpoint(prompt, model=modell, api_key="placeholder")
    out.routed_to = "agy"
    logger.info("extern hydrogen aufrufer=%s modell=%s", kennung, modell)
    return out
