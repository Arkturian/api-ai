"""#4257: Dienst-JWT-Rechte ai:realtime (tschepp-voice-token) und ai:tts:narrate (tschepp-redaktion).

Ziel: ein Dienst-JWT mit ai:realtime bekommt ohne Grant-Tausch ein eigenes
Budget-Profil dienst:<sub> mit den Grenzen aus DIENST_TAGESGRENZEN; ai:tts:narrate
bucht Zeichen je Tag vor dem Anbieter (429 mit Klartext). Gegenfälle: ein Token
ohne das Recht der Route 403, das Realtime-Recht an einer fremden Route 403,
ohne Dienst bleibt der Grant-Tausch (401 ohne Bearer).
"""
import asyncio
import json
import types

import pytest
from fastapi import HTTPException

from ai.routes import realtime_routes as rr
from ai.services import dienst_jwt as dj
from tests.test_3958_dienst_jwt import _kopf, _token, client, netz  # noqa: F401  (Fixtures)

VOICE = "service:tschepp-voice-token"
REDAKTION = "service:tschepp-redaktion"


def test_recht_fuer_neue_routen():
    for p in ("/ai/realtime/token", "/ai/realtime/usage", "/ai/realtime/session/heartbeat", "/ai/realtime/session/end"):
        assert dj.recht_fuer("POST", p) == "ai:realtime", p
    assert dj.recht_fuer("POST", "/ai/realtime/live/sdp") is None
    assert dj.recht_fuer("POST", "/ai/tts/narrate") == "ai:tts:narrate"
    assert dj.recht_fuer("GET", "/ai/tts/narrate/rid-1") == "ai:tts:narrate"
    assert dj.recht_fuer("POST", "/ai/tts") is None
    assert "ai:realtime" in dj.DIENST_RECHTE and "ai:tts:narrate" in dj.DIENST_RECHTE


def _dep(scope, dienst=None, authorization=None):
    anfrage = types.SimpleNamespace(state=types.SimpleNamespace(dienst=dienst) if dienst else types.SimpleNamespace())
    return asyncio.run(rr.require_realtime_grant(scope)(anfrage, authorization))


def test_realtime_dienst_bekommt_eigenes_profil(monkeypatch):
    monkeypatch.setenv("DIENST_TAGESGRENZEN", json.dumps({VOICE: {"realtime_eur": 5, "realtime_parallel": 1}}))
    g = _dep("mint", {"sub": VOICE, "recht": "ai:realtime"})
    assert g.profile_id == f"dienst:{VOICE}" and g.sub == VOICE
    assert g.daily_budget_eur == 5.0 and g.max_parallel_sessions == 1 and g.monthly_budget_eur is None
    assert g.has_scope("usage")


def test_realtime_dienst_vorgabe_und_fremder_scope(monkeypatch):
    monkeypatch.delenv("DIENST_TAGESGRENZEN", raising=False)
    g = _dep("usage", {"sub": VOICE, "recht": "ai:realtime"})
    assert g.daily_budget_eur == dj.VORGABE_GRENZEN["realtime_eur"]
    with pytest.raises(HTTPException) as e:
        _dep("devlog", {"sub": VOICE, "recht": "ai:realtime"})
    assert e.value.status_code == 403


def test_ohne_dienst_bleibt_der_grant_tausch():
    with pytest.raises(HTTPException) as e:
        _dep("mint", None, None)
    assert e.value.status_code == 401 and e.value.detail["error"] == "realtime_user_jwt_required"


def test_narrate_zeichen_grenze(monkeypatch, tmp_path):
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path))
    monkeypatch.setenv("DIENST_TAGESGRENZEN", json.dumps({REDAKTION: {"narrate_zeichen": 100}}))
    marke = dj.AKTUELL.set({"sub": REDAKTION, "recht": "ai:tts:narrate"})
    try:
        dj.narrate_buchen(60)
        with pytest.raises(HTTPException) as e:
            dj.narrate_buchen(60)
        assert e.value.status_code == 429
        d = e.value.detail
        assert d["error"] == "dienst_tagesgrenze" and d["feld"] == "narrate_zeichen" and d["grenze"] == 100
        assert "Retry-After" in e.value.headers
    finally:
        dj.AKTUELL.reset(marke)
    dj.narrate_buchen(10_000)        # ohne Dienst im Kontext: nichts gebucht, kein Fehler


def test_realtime_recht_an_fremder_route_403(client):
    r = client.post("/ai/transcribe", headers=_kopf(_token(sub=VOICE, permissions=["ai:realtime"])),
                    files={"file": ("a.mp3", b"x", "audio/mpeg")})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "dienst_recht_fehlt"


def test_token_ohne_realtime_recht_an_token_route_403(client):
    r = client.post("/ai/realtime/token", headers=_kopf(_token()), json={})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "dienst_recht_fehlt"


def test_realtime_dienst_kommt_bis_zur_route(client):
    """Mit ai:realtime kein 401/403 der Anmeldung: die Route läuft bis zum
    Billing-Tor (ohne confirm_api_billing dort abgelehnt, nichts gebucht)."""
    r = client.post("/ai/realtime/token", headers=_kopf(_token(sub=VOICE, permissions=["ai:realtime"])), json={})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "api_billing_confirmation_required", r.text
