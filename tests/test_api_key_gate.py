"""Zugangssperre per `X-API-KEY` (#1184).

Alexanders Entscheidung vom 2026-08-18: Statischer Schluessel, alles
ohne ihn gesperrt. Legitime Traeger sind die MCP-Server, KIs die er
selbst ausstattet, und das admin-Dashboard.

Zwei Eigenschaften, die dieser Test festhaelt, weil sie beide leicht
verlorengehen:

**Ohne gesetzten Schluessel wird NICHT gesperrt.** Sonst waere der
Dienst in der Sekunde des Ausrollens tot, bevor irgendein Aufrufer den
Schluessel hat. Gemessen wurden am 2026-08-17 vier Aufrufer, von denen
drei ihn erst bekommen muessen.

**Der GCP-Budget-Webhook bleibt offen.** Google kann keinen Schluessel
von uns tragen. Gemessen am 2026-08-18: 30 Aufrufe in sieben Tagen,
zuletzt 08:58 — er feuert, entgegen der Annahme, GCP sei ohnehin tot.
Ihn zu sperren hiesse, die Kostenwarnungen abzuwuergen, die es nur
gibt, weil im Mai 209,56 EUR unbemerkt abflossen.
"""

import inspect

import main


def test_ohne_gesetzten_schluessel_wird_nicht_gesperrt():
    quelle = inspect.getsource(main._require_api_key)
    assert 'erwartet = os.getenv(_API_KEY_ENV)' in quelle
    assert 'if not erwartet:' in quelle
    # Und er reicht dann durch, statt abzuweisen.
    kopf = quelle.split('if not erwartet:')[1].split('\n')[1]
    assert 'return await call_next(request)' in kopf


def test_fehlende_konfiguration_ist_laut():
    """Eine Sperre, die mangels Schluessel nicht greift, muss beim Start
    warnen — sonst haelt man sie fuer geschlossen."""
    quelle = inspect.getsource(main)
    assert 'ist NICHT gesetzt' in quelle
    assert 'Zugangssperre (#1184) ist AUS' in quelle


def test_gcp_webhook_bleibt_offen():
    assert "/ai/gemini/gcp-budget-webhook" in main._OFFENE_PFADE
    quelle = inspect.getsource(main._require_api_key)
    assert 'pfad in _OFFENE_PFADE' in quelle


def test_vergleich_ist_zeitkonstant():
    """`==` auf Schluesseln verraet ueber die Laufzeit, an welcher
    Stelle sie sich unterscheiden."""
    quelle = inspect.getsource(main._require_api_key)
    assert 'hmac.compare_digest' in quelle
    assert 'geliefert == erwartet' not in quelle


def test_abweisung_nennt_den_grund_ohne_den_schluessel_zu_verraten():
    quelle = inspect.getsource(main._require_api_key)
    assert 'status_code=401' in quelle
    assert 'api_key_required' in quelle
    # Der erwartete Wert darf NIE in der Antwort oder im Protokoll stehen.
    assert 'erwartet' not in quelle.split('JSONResponse')[1]


def test_nur_ai_pfade_sind_betroffen():
    """`/health` und `/docs` muessen erreichbar bleiben — sonst meldet
    jede Ueberwachung den Dienst als tot."""
    quelle = inspect.getsource(main._require_api_key)
    assert 'not pfad.startswith("/ai/")' in quelle


# ----------------------------------------------------- Verhalten, nicht Form

def _client(monkeypatch, schluessel=None):
    """App mit echter Middleware; OpenAI wird nie erreicht, weil die
    Sperre vorher greift bzw. der Pfad frei ist."""
    from fastapi.testclient import TestClient
    if schluessel is None:
        monkeypatch.delenv("API_ACCESS_KEY", raising=False)
    else:
        monkeypatch.setenv("API_ACCESS_KEY", schluessel)
    return TestClient(main.app, raise_server_exceptions=False)


def test_mit_schluessel_gesetzt_wird_ohne_kopf_abgewiesen(monkeypatch):
    c = _client(monkeypatch, "geheim-test")
    r = c.post("/ai/chatgpt", json={"prompt": "x"})
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "api_key_required"


def test_falscher_schluessel_wird_abgewiesen(monkeypatch):
    c = _client(monkeypatch, "geheim-test")
    r = c.post("/ai/chatgpt", json={"prompt": "x"},
               headers={"X-API-KEY": "falsch"})
    assert r.status_code == 401


def test_richtiger_schluessel_kommt_durch(monkeypatch):
    """Kommt durch heisst: NICHT 401. Was der Endpunkt danach tut
    (Kostenbremse, fehlender Anbieterschluessel), ist nicht Sache der
    Sperre."""
    c = _client(monkeypatch, "geheim-test")
    r = c.post("/ai/chatgpt", json={"prompt": "x"},
               headers={"X-API-KEY": "geheim-test"})
    assert r.status_code != 401


def test_webhook_kommt_auch_ohne_schluessel_durch(monkeypatch):
    """Der Fall, der eine pauschale Sperre gefaehrlich macht."""
    c = _client(monkeypatch, "geheim-test")
    r = c.post("/ai/gemini/gcp-budget-webhook", json={})
    assert r.status_code != 401


def test_health_bleibt_erreichbar(monkeypatch):
    c = _client(monkeypatch, "geheim-test")
    assert c.get("/health").status_code != 401
