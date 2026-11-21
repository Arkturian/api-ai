"""Gehoeren Aussteller, Schluesselquelle und Grant-Adresse zusammen? (#3604)

Die drei Werte haben als Vorgabe die auth-api des Betreibers. Das ist auf
unserem Kern richtig. Auf einer Kundeninstanz, die nur einen Teil davon setzt,
fiel der Rest still auf uns zurueck: Ein Realtime-Tausch oder eine Admin-Pruefung
schickte das Bearer-Token eines Kundennutzers an unsere auth-api.

Die Regel:

- Keine der drei Variablen gesetzt: `vorgabe` (unser Kern).
- Alle drei gesetzt, jede zeigt auf die eigene Instanz: `eigen`.
- Alles andere: `unvollstaendig`. Dann verlaesst kein Bearer-Token den Dienst
  und kein Token gilt als geprueft.

„Zeigt auf uns“ heisst NICHT „endet auf arkturian.com“. Kundeninstanzen tragen
dort eigene Namen (z. B. `ele.arkturian.com`), und ihr Aussteller ist genau
dieser Name. Eine Adresse unter arkturian.com ist deshalb nur erlaubt, wenn sie
der eigene Aussteller-Host ist.

Grenze: Fehlen alle drei, sieht eine Kundeninstanz aus wie unser Kern. Das
faengt die Vorabpruefung des Rollouts ab, nicht dieser Code.

Protokolliert werden nur Variablennamen, nie Werte.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, TypedDict
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ISSUER_ENV = "REALTIME_AUTH_ISSUER"
JWKS_ENV = "REALTIME_AUTH_JWKS_URL"
GRANT_ENV = "REALTIME_AUTH_GRANT_URL"
# Optional, aber sie tragen Token oder Vertrauen: In einer eigenen Instanz
# duerfen auch sie nicht auf uns zeigen.
ZUSATZ_ENVS = ("AUTH_API_ME_URL", "AGENTOS_JWT_REVOCATION_URL")

VORGABEN = {
    ISSUER_ENV: "auth-api.arkturian.com",
    JWKS_ENV: "https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json",
    GRANT_ENV: "https://auth-api.arkturian.com/api/v1/auth/realtime-grant",
}

VORGABE = "vorgabe"
EIGEN = "eigen"
UNVOLLSTAENDIG = "unvollstaendig"
FEHLERCODE = "realtime_auth_unvollstaendig"

# Nie der Name einer Kundeninstanz (Cloud-Security, #3604).
_BETREIBER_HOSTS = frozenset({"arkturian.com", "www.arkturian.com", "arkserver.arkturian.com"})

_gemeldet = False


class Befund(TypedDict):
    zustand: str
    fehlt: List[str]
    zeigt_auf_betreiber: List[str]


def _host(wert: str) -> Optional[str]:
    roh = wert if "//" in wert else f"//{wert}"
    try:
        host = urlsplit(roh).hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def _unter_arkturian(host: str) -> bool:
    return host == "arkturian.com" or host.endswith(".arkturian.com")


def _ist_betreiber_aussteller(host: Optional[str]) -> bool:
    if not host:
        return True
    return (host in _BETREIBER_HOSTS
            or host.endswith(".arkserver.arkturian.com")
            or (host.startswith("auth-api.") and host.endswith(".arkturian.com")))


def _url_zeigt_auf_betreiber(url: str, eigener_host: Optional[str]) -> bool:
    try:
        teile = urlsplit(url)
    except ValueError:
        return True
    if teile.scheme not in ("http", "https"):
        return True          # nicht pruefbar, also nicht vertrauen
    host = _host(url)
    if not host:
        return True
    if not _unter_arkturian(host):
        return False
    return host != eigener_host or _ist_betreiber_aussteller(eigener_host)


def befund() -> Befund:
    """Liest die Umgebung bei jedem Aufruf. Enthaelt nur Variablennamen."""
    namen = (ISSUER_ENV, JWKS_ENV, GRANT_ENV)
    if not any(n in os.environ for n in namen):
        return {"zustand": VORGABE, "fehlt": [], "zeigt_auf_betreiber": []}

    werte = {n: (os.environ.get(n) or "").strip() for n in namen}
    if all(werte[n] == VORGABEN[n] for n in namen):
        return {"zustand": VORGABE, "fehlt": [], "zeigt_auf_betreiber": []}

    fehlt = [n for n in namen if not werte[n]]
    eigener_host = _host(werte[ISSUER_ENV]) if werte[ISSUER_ENV] else None
    betreiber: List[str] = []
    if werte[ISSUER_ENV] and _ist_betreiber_aussteller(eigener_host):
        betreiber.append(ISSUER_ENV)
    for n in (JWKS_ENV, GRANT_ENV):
        if werte[n] and _url_zeigt_auf_betreiber(werte[n], eigener_host):
            betreiber.append(n)
    for n in ZUSATZ_ENVS:
        wert = (os.environ.get(n) or "").strip()
        if wert and _url_zeigt_auf_betreiber(wert, eigener_host):
            betreiber.append(n)

    zustand = UNVOLLSTAENDIG if fehlt or betreiber else EIGEN
    return {"zustand": zustand, "fehlt": fehlt, "zeigt_auf_betreiber": betreiber}


def zustand() -> str:
    return befund()["zustand"]


def dienst_basis(env: str, vorgabe: str) -> Optional[str]:
    """Basisadresse eines Nachbardienstes, oder None, wenn es ihn hier nicht gibt.

    Die Vorgabe (unsere Adresse) gilt nur auf unserem Kern. Eine Instanz mit
    eigener auth-api braucht einen eigenen Wert; ohne ihn oder mit einem Wert,
    der auf uns zeigt, ist das Werkzeug aus, statt Daten an uns zu schicken."""
    wert = (os.environ.get(env) or "").strip().rstrip("/")
    if zustand() == VORGABE:
        return wert or vorgabe.rstrip("/")
    if not wert:
        return None
    issuer = (os.environ.get(ISSUER_ENV) or "").strip()
    eigener_host = _host(issuer) if issuer else None
    return None if _url_zeigt_auf_betreiber(wert, eigener_host) else wert


def unvollstaendig() -> bool:
    """True, wenn gerade kein Bearer-Token hinaus darf. Meldet das einmal je
    Prozess ins Journal."""
    b = befund()
    if b["zustand"] != UNVOLLSTAENDIG:
        return False
    global _gemeldet
    if not _gemeldet:
        _gemeldet = True
        logger.error("realtime_auth unvollstaendig: fehlt=%s zeigt_auf_betreiber=%s; "
                     "Tausch und Token-Pruefung sind aus",
                     ",".join(b["fehlt"]) or "-", ",".join(b["zeigt_auf_betreiber"]) or "-")
    return True
