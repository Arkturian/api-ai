"""#4135 (Plan 6567 A1.3): Kundenschluessel aus dem Tresor der auth-api lesen.

Vertrag (#4120, auth-api app/api/v1/tresor.py):
  GET /api/v1/tresor/{verweis}?mandant=…&zweck=…&aufrufer=…  -> {"wert", "anbieter", …}
Nur mit dem Dienst-JWT `service:api-ai-arkserver` (Recht `tresor:lesen`, Leserliste
TRESOR_LESER; Alex' Ja 26.11. 21:47Z). Der Ausweis liegt in /etc/api-ai/dienst-ausweis.env
(API_AI_DIENST_CLIENT_ID, API_AI_DIENST_SECRET, API_AI_DIENST_TOKEN_URL).

Der Wert wird je Aufruf gelesen (jeder Abruf mit Grund im Audit der auth-api) und
nie gespeichert, protokolliert oder in einer Fehlermeldung genannt.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import HTTPException

logger = logging.getLogger("api-ai.tresor")

_TOKEN: dict = {"wert": None, "bis": 0.0}
_SPERRE = asyncio.Lock()


def _token_url() -> Optional[str]:
    return (os.getenv("API_AI_DIENST_TOKEN_URL") or "").strip() or None


def _basis() -> Optional[str]:
    eigen = (os.getenv("TRESOR_URL") or "").strip()
    if eigen:
        return eigen.rstrip("/")
    u = _token_url()
    return u.split("/api/", 1)[0] if u and "/api/" in u else None


def _nicht_verfuegbar(grund: str, mandant: str) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "error": "konto_nicht_verfuegbar", "grund": grund, "mandant": mandant,
        "hint": "Der Schluessel des Mandanten ist nicht lesbar. Es laeuft kein anderer Anbieter."})


async def _dienst_token(mandant: str) -> str:
    async with _SPERRE:
        if _TOKEN["wert"] and _TOKEN["bis"] > time.time() + 60:
            return _TOKEN["wert"]
        cid = (os.getenv("API_AI_DIENST_CLIENT_ID") or "").strip()
        sec = (os.getenv("API_AI_DIENST_SECRET") or "").strip()
        url = _token_url()
        if not (cid and sec and url):
            raise _nicht_verfuegbar("dienstausweis_fehlt", mandant)
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(url, data={"grant_type": "client_credentials",
                                            "client_id": cid, "client_secret": sec})
        except httpx.HTTPError as e:
            logger.warning("tresor: Token nicht erhalten (%s)", type(e).__name__)
            raise _nicht_verfuegbar("auth_nicht_erreichbar", mandant) from None
        if r.status_code != 200:
            logger.warning("tresor: Token abgelehnt (%s)", r.status_code)
            raise _nicht_verfuegbar("dienstausweis_abgelehnt", mandant)
        d = r.json() or {}
        tok = d.get("access_token")
        if not isinstance(tok, str) or not tok:
            raise _nicht_verfuegbar("dienstausweis_abgelehnt", mandant)
        _TOKEN["wert"] = tok
        _TOKEN["bis"] = time.time() + float(d.get("expires_in") or 600)
        return tok


async def lesen(verweis: str, mandant: str, zweck: str, aufrufer: str, anbieter: str) -> str:
    """Wert eines Tresor-Eintrags des Mandanten. 503 `konto_nicht_verfuegbar` mit Grund,
    wenn er nicht lesbar ist oder zu einem anderen Anbieter gehoert."""
    tok = await _dienst_token(mandant)                 # zuerst der Ausweis
    basis = _basis()
    if not basis:
        raise _nicht_verfuegbar("tresor_adresse_fehlt", mandant)
    url = f"{basis}/api/v1/tresor/{quote(verweis, safe='')}"
    params = {"mandant": mandant, "zweck": zweck, "aufrufer": (aufrufer or "unbekannt")[:200]}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, params=params, headers={"Authorization": f"Bearer {tok}"})
    except httpx.HTTPError as e:
        logger.warning("tresor: nicht erreichbar (%s) mandant=%s", type(e).__name__, mandant)
        raise _nicht_verfuegbar("tresor_nicht_erreichbar", mandant) from None
    if r.status_code == 404:
        raise _nicht_verfuegbar("tresor_eintrag_unbekannt", mandant)
    if r.status_code in (401, 403):
        _TOKEN["wert"] = None
        logger.warning("tresor: Lesen abgelehnt (%s) mandant=%s", r.status_code, mandant)
        raise _nicht_verfuegbar("tresor_lesen_abgelehnt", mandant)
    if r.status_code != 200:
        logger.warning("tresor: Fehler %s mandant=%s", r.status_code, mandant)
        raise _nicht_verfuegbar("tresor_fehler", mandant)
    d = r.json() or {}
    if str(d.get("anbieter") or "").lower() != anbieter:
        raise _nicht_verfuegbar("tresor_anbieter_passt_nicht", mandant)
    wert = d.get("wert")
    if not isinstance(wert, str) or not wert.strip():
        raise _nicht_verfuegbar("tresor_eintrag_leer", mandant)
    return wert.strip()
