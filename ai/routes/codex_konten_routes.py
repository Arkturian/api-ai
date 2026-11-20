"""GET /ai/codex/konten — Stand der eigenen codex-Konten (#3165).

Fuer Login (Abstimmung 20.11.: Login und api-ai waehlen nach derselben Regel,
Login muss wissen, wohin api-ai gewechselt hat). Lesbar fuer die Agenten aus
API_AI_CODEX_KONTEN_LESER (Vorgabe: Login), geprueft am JWT, und fuer Admins
(Menschen-JWT ueber auth-api /me). Keine Pfade, keine Token.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from ai.services import codex_konten

logger = logging.getLogger(__name__)
router = APIRouter()


def _leser() -> set:
    roh = os.getenv("API_AI_CODEX_KONTEN_LESER", "Login")
    return {n.strip() for n in roh.split(",") if n.strip()}


async def _darf_lesen(authorization: Optional[str]) -> None:
    from ai.services import jwt_pruefung
    claims = jwt_pruefung.pruefen(authorization)       # 401/503 bei falschem Token
    if claims is None:
        raise HTTPException(status_code=401, detail={"error": "authentication_required"})
    sub = str(claims.get("sub") or "")
    if claims.get("type") == "agent" and sub.startswith("agent:") and sub[len("agent:"):] in _leser():
        return
    from ai.routes.verbrauch_routes import _profil, ist_admin
    profil = await _profil(authorization)
    if profil and ist_admin(profil):
        return
    raise HTTPException(status_code=403, detail={"error": "not_allowed"})


@router.get("/codex/konten")
async def konten_stand(authorization: Optional[str] = Header(default=None)):
    """`aktiv`, je Konto `name, email, used_pct, resets_at, state, gemessen_um`,
    die letzten 10 Wechsel `{zeit, von, nach, von_pct, nach_pct, grund}` und die
    letzten 10 Ereignisse (wechsel, kein_ziel, anmeldung_abgelehnt)."""
    await _darf_lesen(authorization)
    if not codex_konten.aktiv_modus():
        return JSONResponse({"aktiv": None, "konten": [], "wechsel": [], "ereignisse": [],
                             "hinweis": "Kontomodus aus: codex laeuft mit der Anmeldung des Hosts"},
                            headers={"Cache-Control": "no-store"})
    return JSONResponse(await asyncio.to_thread(codex_konten.uebersicht),
                        headers={"Cache-Control": "no-store"})


async def mess_schleife() -> None:
    """Alle MESS_ABSTAND_S eine Runde. Mehrere Prozesse: die Runde haelt ein
    flock, der zweite Prozess faellt dann aus."""
    from ai.services import codex_melden
    await asyncio.sleep(30)
    while True:
        try:
            erg = await asyncio.to_thread(codex_konten.runde, codex_melden.melden)
            if erg.get("aktion") not in ("bleibt", "laeuft_schon"):
                logger.info("codex-konten: %s", erg)
        except Exception as e:                                # noqa: BLE001
            logger.warning("codex-konten: Runde gescheitert: %s", type(e).__name__)
        await asyncio.sleep(codex_konten.MESS_ABSTAND_S)
