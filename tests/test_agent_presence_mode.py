"""companion_mode='agent-presence' — die Sprachsitzung IST der Agent
(Owner-Entscheid Alex, 03.09.: je Agent ein Realtime-Agent, der dieser
Agent ist; Arcturian bleibt die foederationsweite Stimme).

Gegen die echten Fehler gehalten: (1) die Stimme erfindet nichts, was der
Stand nicht traegt; (2) sie verspricht keine Weitergabe (Rueckweg nicht
gebaut); (3) sie hat genau ein Werkzeug — den eigenen Stand nachladen;
(4) der Stand kommt mit dem Bearer des Menschen, nie mit einem Dienst-Token.
"""

import asyncio

from ai.routes import realtime_routes as rr

CTX = {"stand": "2026-09-02T06:26:44.773Z", "derivative_count": 8, "ref": "abc",
       "block": "[PRESENCE-CONTEXT ref=abc of=CloudV2 stand=02.09._06:26]\n"
                "▸ 02.09. 06:01 CloudV2 hat die Linse ausgeliefert. ⟨intern: …⟩"}


def test_modus_ist_registriert():
    assert "agent-presence" in rr.SUPPORTED_COMPANION_MODES


def test_prompt_ist_der_agent_nicht_der_erzaehler():
    p = rr._agent_presence_instructions("CloudV2", CTX, "de")
    assert p.startswith("Du bist CloudV2.")
    assert "nicht als Erzaehler" in p
    assert "in erster Person" in p
    assert "Stand 06:26" in p and "8 Ableitungen" in p
    assert CTX["block"] in p                       # der Stand steht drin
    assert "nie vorlesen" in p                      # Marker sind Struktur


def test_prompt_haelt_die_drei_grenzen():
    p = rr._agent_presence_instructions("CloudV2", CTX, "de")
    assert "Das habe ich nicht im Stand von 06:26" in p      # nichts erfinden
    assert "nicht sichtbar" in p and "abgeschlossene" in p    # Laufendes fehlt
    assert "noch nicht an meine Arbeit weitergeben" in p      # Rueckweg nicht gebaut
    assert "frage_kleinhirn mit agent='CloudV2'" in p         # Nachladen


def test_ohne_stand_sagt_die_stimme_das_offen():
    p = rr._agent_presence_instructions("CloudV2", None, "de")
    assert "keiner geladen" in p and "keinen Stand meiner Arbeit" in p
    assert "Stand unbekannt" in p


def test_genau_ein_werkzeug_der_eigene_stand():
    assert [t["name"] for t in rr._agent_presence_tools()] == ["frage_kleinhirn"]


def test_stand_kommt_mit_dem_bearer_des_menschen(monkeypatch):
    seen = {}

    class _Resp:
        status_code = 200
        def json(self): return CTX

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None, params=None):
            seen["url"] = url; seen["headers"] = headers; return _Resp()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst-token-NICHT-benutzen")
    out = asyncio.run(rr._fetch_presence_context("CloudV2", "Bearer mensch-1"))
    assert out is CTX
    assert seen["url"].endswith("/api/sessions/CloudV2/presence-context")
    assert seen["headers"] == {"Authorization": "Bearer mensch-1"}
    # ohne Bearer: kein Aufruf, kein Stand — nie anonym
    assert asyncio.run(rr._fetch_presence_context("CloudV2", "")) is None


def test_stand_fehler_kostet_die_sitzung_nicht(monkeypatch):
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): raise RuntimeError("cloud-api weg")
    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    assert asyncio.run(rr._fetch_presence_context("CloudV2", "Bearer x")) is None
