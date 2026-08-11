"""Der Mint-Pfad wird EINMAL wirklich durchlaufen (#1038).

Am 2026-08-11 um 19:42 warf jede Arcturian-Sitzung einen 500 —
`arcturian_resolver` gelesen, bevor es zugewiesen war. **160 Tests waren
gruen.** Keiner von ihnen rief `mint_realtime_token` auf: Sie pruefen
Bausteine (Werkzeuge, Nutzlasten, Persona-Text) einzeln und korrekt, die
Funktion, die sie zusammensetzt, war unbetreten.

Ein statischer Waechter deckt seither die Fehlerklasse ab. Er ersetzt
aber nicht, dass der Weg einmal von Anfang bis Ende gegangen wird — ein
Aufrufpfad, den keine Pruefung betritt, ist ein Pfad, dessen Zustand
niemand kennt.

Der Aufruf nach OpenAI wird abgefangen: Geprueft wird der eigene Code,
nicht der Anbieter, und ein Test darf kein Geld kosten.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.routes import realtime_routes as rr


class _OpenAIAntwort:
    status_code = 200
    text = ""

    def json(self):
        return {"client_secret": {"value": "ek_test", "expires_at": 0},
                "id": "sess_test"}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """App nur mit diesem Router, Torwaechter offen, OpenAI abgefangen."""

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _OpenAIAntwort()
        async def get(self, *a, **k): return _OpenAIAntwort()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    # Ein Schluessel muss gesetzt sein, sonst bricht der Rumpf vor dem
    # Zusammensetzen ab — der Wert ist beliebig, der Aufruf nach OpenAI
    # ist abgefangen.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-nicht-echt")
    # Der Budget-Waechter schreibt nach /var/lib/api-ai — dort darf ein
    # Testlauf nicht hin. Auf eine temporaere Datei umlegen, damit die
    # Reservierung echt laeuft statt uebersprungen zu werden: Sie ist
    # Teil des Pfads, den dieser Test gerade durchlaufen soll.
    from ai.services import realtime_budget_guard as guard
    monkeypatch.setattr(guard, "RESERVATIONS_PATH", tmp_path / "res.json")

    app = FastAPI()
    app.include_router(rr.router, prefix="/ai")

    # Die beiden Abhaengigkeiten des Endpunkts durch offene Doubles
    # ersetzen — geprueft wird der Rumpf, nicht die Torwaechter. Die
    # haben eigene Tests (`test_operator_endpoints_are_closed`).
    from ai.routes.realtime_routes import get_api_key
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    # Den Vollmachts-Torwaechter ueber den Abhaengigkeitsbaum finden,
    # nicht ueber eine Namensvermutung: `require_realtime_grant("mint")`
    # erzeugt beim Import eine Closure, deren Objekt der Schluessel ist.
    # Die erste Fassung dieses Tests riet und traf nicht — die Anfrage
    # endete mit 401, erreichte den Arcturian-Zweig nie und war trotzdem
    # gruen, weil sie nur auf "kein 500" prueft.
    for route in app.routes:
        for dep in getattr(getattr(route, "dependant", None), "dependencies", []):
            if getattr(dep.call, "__qualname__", "").startswith("require_realtime_grant"):
                app.dependency_overrides[dep.call] = lambda: _Grant()
    # `raise_server_exceptions=True`: Ein 500 soll die echte
    # Rueckverfolgung zeigen, nicht eine nackte Statuszahl. Genau daran
    # habe ich beim Bauen dieses Tests dreimal Zeit verloren.
    return TestClient(app, raise_server_exceptions=True)


class _Grant:
    """Vollmachts-Double mit den Feldern, die der Rumpf wirklich liest.

    Kein `__getattr__`-Fallback auf None mehr: Der Budget-Waechter
    rechnet mit `max_parallel_sessions` und `daily_budget_eur`, und ein
    None ergab dort `'>=' not supported between int and NoneType` — ein
    500, der wie ein Produktionsfehler aussah und einer des Tests war.
    """

    profile_id = "agentos-test"
    sub = "test-subject"
    tenant_id = "arkturian"
    scopes = ("mint",)
    max_parallel_sessions = 4
    daily_budget_eur = 5.0
    model = None
    voice = None


@pytest.mark.parametrize("resolver", [
    None,
    "agentos.arcturian-action.v2",
    "agentos.arcturian-action.v3",
])
def test_arcturian_mint_wirft_keinen_500(client, resolver):
    """Genau der Aufruf, der zwischen 19:42 und 20:59 jedes Mal 500 warf.

    Alle drei Fassungen, weil der Fehler an der versionsabhaengigen
    Anweisung hing — v1 (Feld weggelassen) haette ihn genauso ausgeloest.
    """
    # `confirm_api_billing` ist die eigene Kostenbremse und default-deny
    # (403 mit Preisangabe). Sie gehoert dorthin — hier wird sie
    # bestaetigt, weil der Aufruf nach OpenAI abgefangen ist und der
    # Test nichts kostet.
    payload = {"companion_mode": "arcturian", "language": "de",
               "read_tools": True, "confirm_api_billing": True}
    if resolver:
        payload["arcturian_resolver"] = resolver
    r = client.post("/ai/realtime/token", json=payload)
    # 200 verlangen, nicht bloss "kein 500": Eine Anfrage, die vorher
    # abgewiesen wird, betritt den Zweig nicht und beweist nichts. Genau
    # daran ist die erste Fassung dieses Tests gescheitert — sie war
    # gruen, waehrend der Fehler wieder eingebaut war.
    assert r.status_code == 200, (
        f"Mint scheitert fuer resolver={resolver}: "
        f"HTTP {r.status_code} {r.text[:300]}"
    )
    d = r.json()
    # Und der Beleg, dass wirklich der Arcturian-Zweig lief: Diese
    # Felder entstehen ausschliesslich dort.
    assert d.get("resolver_followup"), "Arcturian-Zweig wurde nicht durchlaufen"
    assert d.get("persona_sha256"), "Persona wurde nicht zusammengesetzt"
    if resolver:
        assert d.get("arcturian_resolver") == resolver
