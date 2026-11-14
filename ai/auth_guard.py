"""Gemeinsame Zugangspruefungen (#2938, Befund 59).

`get_api_key()` in den Routen liefert nur einen Platzhalter und prueft nichts.
Hier stehen die echten Pruefungen, die Betreiber-, Status- und Abfrage-Routen
brauchen:

- `betreiber`         nur mit Betreiber-Schluessel (X-API-KEY = REALTIME_GRANT_SERVICE_KEY)
- `betreiber_oder_lokal`  Schluessel oder ein Dienst auf diesem Host
- `angemeldet`        Schluessel oder gepruefter Foederations-JWT

"Auf diesem Host": direkt auf 127.0.0.1:8000 ohne Proxy-Koepfe, oder ueber das
lokale nginx, das X-Real-IP selbst setzt (ueberschreibt, nicht anhaengt) und
dort Loopback eintraegt. X-Forwarded-For allein zaehlt nie, den Kopf kann der
Aufrufer mitbringen.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, Request

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def ist_betreiber(x_api_key: Optional[str]) -> bool:
    erwartet = os.environ.get("REALTIME_GRANT_SERVICE_KEY", "")
    return bool(erwartet and isinstance(x_api_key, str) and x_api_key
                and hmac.compare_digest(x_api_key, erwartet))


def von_diesem_host(request: Optional[Request]) -> bool:
    if request is None:
        return True
    try:
        host = request.client.host if request.client else None
    except Exception:
        host = ""
    # uvicorn (proxy_headers, vertraut 127.0.0.1) setzt den Client auf None,
    # wenn JEDER Eintrag in X-Forwarded-For vertrauenswuerdig ist: das ist
    # genau der lokale Dienst ueber das lokale nginx. Kommt jemand von aussen,
    # haengt nginx dessen Adresse an, und uvicorn nennt sie als Client.
    if host is None and request.headers.get("x-forwarded-for") is not None:
        host = "127.0.0.1"
    if host not in _LOOPBACK:
        return False
    real_ip = request.headers.get("x-real-ip")
    if real_ip is not None:
        return real_ip.strip() in _LOOPBACK
    return request.headers.get("x-forwarded-for") is None


def betreiber(x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")) -> str:
    if ist_betreiber(x_api_key):
        return "betreiber"
    raise HTTPException(status_code=401, detail={
        "error": "operator_key_required",
        "hint": "Betreiber-Daten bzw. -Schalter: X-API-KEY des Hosts mitschicken."})


def betreiber_oder_lokal(request: Request,
                         x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")) -> str:
    if ist_betreiber(x_api_key):
        return "betreiber"
    if von_diesem_host(request):
        return "lokal"
    raise HTTPException(status_code=401, detail={
        "error": "operator_or_local_required",
        "hint": "Nur fuer Dienste auf diesem Host oder mit X-API-KEY des Hosts."})


def angemeldet(authorization: Optional[str] = Header(default=None),
               x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")) -> str:
    if ist_betreiber(x_api_key):
        return "betreiber"
    from ai.services import bild_verbrauch as bv
    claims = bv._claims_geprueft(authorization)
    name = bv.name_aus_claims(claims) if claims else None
    if name:
        return name
    raise HTTPException(status_code=401, detail={
        "error": "auth_required",
        "hint": "Bearer-JWT der Foederation oder X-API-KEY des Hosts mitschicken."})
