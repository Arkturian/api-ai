"""Verbrauch je Aufrufer lesen (Auftrag Alex 13.11.2026 ueber CloudV2).

GET /ai/verbrauch/bilder — Bilder je Wiener Tag und Agent, fuer das
Info-Fenster im Portal. Nur Admins: Die Route zeigt alle Agenten. Gleiches
Tor wie cloud-api `/api/usage/report`: Menschen-JWT, von der auth-api per
`/me` bestaetigt, `is_superadmin` oder `agentos_role == "admin"`.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from ai.services import bild_verbrauch as bv

logger = logging.getLogger(__name__)
router = APIRouter()

_ME_CACHE: dict = {}
_ME_TTL_S = 60.0


def _me_url() -> str:
    """Dieselbe auth-api, gegen die api-ai auch Realtime-Grants tauscht —
    eine Kundeninstanz mit eigener auth-api setzt das an einer Stelle."""
    eigen = (os.getenv("AUTH_API_ME_URL") or "").strip()
    if eigen:
        return eigen
    from ai.services import realtime_grant_verifier as v
    return v.GRANT_URL.rsplit("/", 1)[0] + "/me"


async def _profil(authorization: Optional[str]) -> Optional[dict]:
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    jetzt = time.time()
    treffer = _ME_CACHE.get(token)
    if treffer and treffer[0] > jetzt:
        return treffer[1]
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(_me_url(), headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as e:
        logger.warning("verbrauch: auth-api nicht erreichbar (%s)", type(e).__name__)
        raise HTTPException(status_code=503, detail={"error": "auth_unreachable"})
    if r.status_code != 200:
        return None
    profil = r.json() or {}
    if len(_ME_CACHE) > 500:
        _ME_CACHE.clear()
    _ME_CACHE[token] = (jetzt + _ME_TTL_S, profil)
    return profil


def ist_admin(profil: dict) -> bool:
    return bool(profil.get("is_superadmin")) or profil.get("agentos_role") == "admin"


def _zeit(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        t = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail={"error": "invalid_time", "value": v})
    return t if t.tzinfo else t.replace(tzinfo=bv.WIEN)


@router.get("/verbrauch/bilder")
async def bilder(agent: Optional[str] = Query(default=None, description="Agentenname oder Mail; leer = alle"),
                 von: Optional[str] = Query(default=None, description="ISO-Zeit, einschliesslich; ohne Zone = Wien"),
                 bis: Optional[str] = Query(default=None, description="ISO-Zeit, ausschliesslich; ohne Zone = Wien"),
                 authorization: Optional[str] = Header(default=None)):
    """Bilder je Tag (Europe/Vienna), Agent und Art.

    Zeile: `t`, `agent`, `art` (vorerst `bild`), `anzahl`, `abo`, `bezahlt`,
    `kosten_usd`; dazu `kosten_unbekannt`, wenn ein bezahltes Bild ohne
    Preisangabe gezaehlt wurde. `seit` ist die erste erfasste Erzeugung
    ueberhaupt: vorher wurde nicht gezaehlt."""
    profil = await _profil(authorization)
    if profil is None:
        raise HTTPException(status_code=401, detail={"error": "authentication_required"})
    if not ist_admin(profil):
        raise HTTPException(status_code=403, detail={"error": "admin_only"})
    t_von, t_bis = _zeit(von), _zeit(bis)
    if t_von and t_bis and t_von >= t_bis:
        raise HTTPException(status_code=422, detail={"error": "von_not_before_bis"})
    import asyncio
    res = await asyncio.to_thread(bv.bericht, (agent or "").strip() or None, t_von, t_bis)
    return JSONResponse(res, headers={"Cache-Control": "no-store"})
