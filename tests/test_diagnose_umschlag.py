"""Verwerfungen fahren im TRANSPORT-Umschlag, nicht im Ergebnis.

Mein erster Vorschlag war ein Zusatzfeld IM Domain-Ergebnis.
Tschepp-Codex2 hat ihn am Browservertrag widerlegt: Der
Details-Validator führt eine geschlossene Feldliste und hätte es als
`invalid_tool_response` abgewiesen; die Suchvalidatoren erlauben
zusätzliche Felder und hätten es als `function_call_output` ans
Modell zurückgegeben. Das Modell hätte meine Auditdaten gesehen —
und bei jedem weiteren Sprechzug mitbezahlt.

Deshalb neben `result`. Der BFF schält den Umschlag ohnehin ab: Er
protokolliert die Diagnose und gibt dem Browser nur das unveränderte
`result`.

Warum überhaupt: Verwerfe ich ein erfundenes Feld des Modells, steht
in oneals Protokoll ein Aufruf OHNE dieses Feld. Niemand könnte
rekonstruieren, dass das Modell es gesagt hat — genau die Lücke,
wegen der die Gespräche jetzt gespeichert werden.
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    async def _ja(auth, scope):
        return object()

    monkeypatch.setattr(rr, "exchange_and_verify", _ja)
    app = FastAPI()
    app.include_router(rr.router, prefix="/ai")
    return TestClient(app, raise_server_exceptions=True)


def _ruf(client, tool, args, sid="vs_pf_1"):
    return client.post(
        f"/ai/realtime/tool/{tool}",
        json={"call_id": "c1", "arguments": args},
        headers={"Authorization": "Bearer x", "X-Session-ID": sid},
    )


# ───────────────────────────── verworfene Kriterien

def test_verworfenes_kriterium_steht_im_umschlag(client, monkeypatch):
    async def _suche(args, sid, verfeinern, diagnose=None):
        if diagnose is not None:
            diagnose["dropped_criteria"] = ["bekleidung"]
        return {"status": "matches", "count": 3}

    monkeypatch.setattr(rr, "_tool_product_search", _suche)
    d = _ruf(client, "find_products", {"bekleidung": "ja"}).json()
    assert d["diagnostics"] == {"dropped_criteria": ["bekleidung"]}
    # und NICHT im Ergebnis — sonst säh's das Modell
    assert "diagnostics" not in d["result"]
    assert "dropped_criteria" not in d["result"]


def test_ohne_verwerfung_kein_feld(client, monkeypatch):
    """Ein immer vorhandenes leeres Feld erzeugt Rauschen und verleitet
    dazu, es zu ignorieren."""
    async def _suche(args, sid, verfeinern, diagnose=None):
        return {"status": "matches", "count": 3}

    monkeypatch.setattr(rr, "_tool_product_search", _suche)
    assert "diagnostics" not in _ruf(client, "find_products", {}).json()


def test_verworfene_position_steht_im_umschlag(client, monkeypatch):
    async def _details(args, sid, diagnose=None):
        if diagnose is not None:
            diagnose["dropped_position"] = 0
        return {"status": "details", "name": "Sonus"}

    monkeypatch.setattr(rr, "_tool_product_details", _details)
    d = _ruf(client, "product_details", {"position": 0}).json()
    assert d["diagnostics"] == {"dropped_position": 0}
    assert "dropped_position" not in d["result"]


def test_das_ergebnis_bleibt_unveraendert(client, monkeypatch):
    """Der Browser bekommt nach dem Abschälen genau das, was oneal
    geliefert hat — Wort für Wort."""
    original = {"status": "matches", "count": 3,
                "__app_command__": {"name": "show_product_results"}}

    async def _suche(args, sid, verfeinern, diagnose=None):
        if diagnose is not None:
            diagnose["dropped_criteria"] = ["x"]
        return dict(original)

    monkeypatch.setattr(rr, "_tool_product_search", _suche)
    assert _ruf(client, "find_products", {}).json()["result"] == original


# ───────────────────── die Handler füllen den Behälter wirklich

def test_suchhandler_meldet_unbekannte_felder(monkeypatch):
    import asyncio

    from ai.services import realtime_session_scope as sc
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://o.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "k" * 32)
    monkeypatch.setattr(sc, "SCOPE_PATH", "/tmp/scope_diag_test.json")
    sc.merken("vs1", "O'Neal", 2027)

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "matches", "count": 1}

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _A()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _C)
    diag: dict = {}
    asyncio.run(rr._tool_product_search(
        {"bekleidung": "ja", "sport": ["MX"]}, "vs1", False, diag))
    assert diag["dropped_criteria"] == ["bekleidung"]


def test_detailhandler_meldet_die_unbrauchbare_position(monkeypatch):
    import asyncio
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://o.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "k" * 32)

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "details"}

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _A()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _C)
    diag: dict = {}
    asyncio.run(rr._tool_product_details({"position": 0}, "vs1", diag))
    assert diag["dropped_position"] == 0
