"""Eine eigene api-ai-Instanz muss gegen ihre EIGENE auth-api tauschen.

Befund (Cloud, Frage 1 in #4850): Für agentos1 sollen Aussteller,
Schlüsselquelle und Tausch-Endpunkt auf die Kunden-Instanz zeigen.
Zwei davon waren per Umgebung setzbar, der dritte nicht — und ohne
ihn tauscht eine fremde Instanz weiter gegen arkturian.

Der Kommentar über den beiden anderen sagt es bereits: „Ein fremder
Aussteller mit unserer Schlüsselquelle wäre eine Prüfung, die nur so
aussieht." Derselbe Satz gilt für den Endpunkt — er war nur
übersehen worden.
"""

import importlib
import os

import pytest


def _neu_laden(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import ai.services.realtime_grant_verifier as m
    return importlib.reload(m)


def test_vorgabe_bleibt_arkturian(monkeypatch):
    """Wer nichts setzt, tauscht weiter wie bisher."""
    m = _neu_laden(monkeypatch, REALTIME_AUTH_GRANT_URL=None)
    assert m.GRANT_URL == (
        "https://auth-api.arkturian.com/api/v1/auth/realtime-grant"
    )


def test_eigene_instanz_kann_umlenken(monkeypatch):
    m = _neu_laden(
        monkeypatch,
        REALTIME_AUTH_GRANT_URL="https://agentos.example/api/v1/auth/realtime-grant",
    )
    assert m.GRANT_URL == "https://agentos.example/api/v1/auth/realtime-grant"


def test_alle_drei_sind_gemeinsam_setzbar(monkeypatch):
    """Aussteller, Schlüsselquelle UND Endpunkt. Fehlt einer, ist es
    keine eigene Instanz, sondern eine halbe — und die ist schlimmer
    als keine, weil sie funktioniert und das Falsche tut."""
    m = _neu_laden(
        monkeypatch,
        REALTIME_AUTH_ISSUER="agentos.oneal.arkturian.com",
        REALTIME_AUTH_JWKS_URL="https://agentos.example/jwks.json",
        REALTIME_AUTH_GRANT_URL="https://agentos.example/grant",
    )
    assert m.AUTH_ISSUER == "agentos.oneal.arkturian.com"
    assert m.JWKS_URL == "https://agentos.example/jwks.json"
    assert m.GRANT_URL == "https://agentos.example/grant"


def test_aufraeumen(monkeypatch):
    """Modul wieder in den Vorgabezustand, damit die uebrige Suite
    nicht auf einem umgebogenen Verifier laeuft."""
    m = _neu_laden(monkeypatch, REALTIME_AUTH_GRANT_URL=None,
                   REALTIME_AUTH_ISSUER=None, REALTIME_AUTH_JWKS_URL=None)
    assert "arkturian" in m.GRANT_URL
