"""#4135 (Plan 6567 A1.2): /ai/verarbeitung je Mandant aus dem geprueften JWT.

Ein Kunden-Mandant (z. B. lesser) bekommt auf dem Kern nur seine eigene
Zuweisung (Schluesselweg) oder "keine KI" -- nie die Abos des Betreibers
(KI_VERARBEITUNG), nie dessen OpenAI-Schluessel. Ohne JWT: beobachten bzw. 401.
"""
import asyncio
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import jwt_pruefung
from ai.services import verarbeitung as v


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    def verboten(*a, **kw):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **kw):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("API_ACCESS_KEY", v.JWT_PFLICHT_ENV, v.BETREIBER_ENV, "OPENAI_TEXT_DAILY_MAX_USD"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv(v.ENV, "hydrogen")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-kern-schluessel")
    monkeypatch.setenv(v.MANDANTEN_DIR_ENV, str(tmp_path / "mandanten"))
    monkeypatch.setenv("KOSTENBUCH_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path / "auf"))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "verbrauch"))
    monkeypatch.setenv("KI_KETTE", "agy,codex,claude")

    tokens = {"Bearer lesser": {"sub": "service:conversation-api", "tenant": "lesser"},
              "Bearer ark": {"sub": "service:conversation-api", "tenant": "arkturian"}}
    monkeypatch.setattr(jwt_pruefung, "claims_oder_none", lambda a: tokens.get(a))
    return tmp_path


@pytest.fixture
def wege(monkeypatch):
    gerufen = []

    async def agy(prompt, model=None, api_key=None, request=None):
        gerufen.append("agy")
        return t.AIResponse(response="antwort", model=model)

    async def nie(*a, **kw):
        raise AssertionError("anderer Weg gerufen")

    async def kern_openai(*a, **kw):
        raise AssertionError("OpenAI-Schluessel des Kerns gerufen")
    monkeypatch.setattr(t, "gemini_endpoint", agy)
    monkeypatch.setattr(t, "chatgpt_endpoint", nie)
    monkeypatch.setattr(t, "_claude_mit_rechten", nie)
    monkeypatch.setattr(t, "deepseek_endpoint", nie)
    monkeypatch.setattr(t, "_openai_einmal", kern_openai)
    return gerufen


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def _post(client, token=None, **kopf):
    h = dict(kopf)
    if token:
        h["Authorization"] = token
    return client.post("/ai/verarbeitung", json={"prompt": "Ordne ein"}, headers=h)


def _zuweisen(tmp_path, mandant, inhalt):
    d = tmp_path / "mandanten"
    d.mkdir(exist_ok=True)
    (d / f"{mandant}.json").write_text(json.dumps(inhalt) if not isinstance(inhalt, str) else inhalt)


def test_kunde_ohne_zuweisung_bekommt_nie_die_abos_des_betreibers(client, wege):
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "verarbeitung_nicht_zugewiesen"
    assert wege == []


def test_kunde_mit_keine_ki(client, wege, umgebung):
    _zuweisen(umgebung, "lesser", {"verarbeitung": "keine", "urteil": "keine"})
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "verarbeitung_keine_ki" and wege == []


@pytest.mark.parametrize("anbieter", ["hydrogen", "codex", "kette", "claude"])
def test_kunde_darf_keinen_abo_weg_zuweisen(client, wege, umgebung, anbieter):
    _zuweisen(umgebung, "lesser", {"verarbeitung": {"anbieter": anbieter, "verweis": "lesser-openai"}})
    r = _post(client, "Bearer lesser")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["grund"] == "anbieter_nicht_erlaubt" and wege == []


def test_kunde_mit_schluesselweg_nie_der_kernschluessel(client, wege, umgebung):
    _zuweisen(umgebung, "lesser", {"verarbeitung": {"anbieter": "openai", "verweis": "lesser-openai",
                                                     "modell": "gpt-5.4-mini"}})
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "konto_nicht_verfuegbar" and wege == []


@pytest.mark.parametrize("inhalt,grund", [
    ("{kaputt", "kein_json"),
    ({"verarbeitung": {"anbieter": "openai"}}, "verweis_fehlt"),
    ({"verarbeitung": {"anbieter": "openai", "verweis": "x", "modell": "a b"}}, "modell_ungueltig"),
    ({"urteil": "keine"}, "verarbeitung_fehlt"),
])
def test_ungueltige_zuweisung(client, wege, umgebung, inhalt, grund):
    _zuweisen(umgebung, "lesser", inhalt)
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["grund"] == grund and wege == []


def test_betreiber_mit_jwt_nimmt_die_kern_einstellung(client, wege):
    r = _post(client, "Bearer ark")
    assert r.status_code == 200, r.text
    assert r.json()["anbieter"] == "hydrogen" and wege == ["agy"]


def test_ohne_jwt_beobachten_laeuft_mit_kern_einstellung(client, wege, caplog):
    with caplog.at_level("WARNING"):
        r = _post(client, **{"X-Agent-Name": "service:issue-api"})
    assert r.status_code == 200 and wege == ["agy"]
    assert any("verarbeitung OHNE JWT" in m for m in caplog.messages)


def test_ohne_jwt_mit_pflicht_401(client, wege, monkeypatch):
    monkeypatch.setenv(v.JWT_PFLICHT_ENV, "an")
    r = _post(client, **{"X-Agent-Name": "service:issue-api"})
    assert r.status_code == 401 and r.json()["detail"]["error"] == "jwt_required" and wege == []


def test_gefaelschter_mandant_im_kopf_zaehlt_nicht(client, wege, monkeypatch):
    """Ungepruftes JWT oder ein Kopf machen niemanden zum Kunden oder Betreiber."""
    monkeypatch.setenv(v.JWT_PFLICHT_ENV, "an")
    r = _post(client, "Bearer gefaelscht", **{"X-Caller-Tenant": "arkturian"})
    assert r.status_code == 401 and wege == []


def test_betreiber_ist_einstellbar(client, wege, monkeypatch):
    monkeypatch.setenv(v.BETREIBER_ENV, "lesser")
    monkeypatch.setenv("KI_VERARBEITUNG", "hydrogen")
    # Auf einer Kundeninstanz ist der eigene Mandant der Betreiber; arkturian waere dort fremd.
    assert _post(client, "Bearer lesser").status_code == 200
    assert _post(client, "Bearer ark").json()["detail"]["error"] == "verarbeitung_nicht_zugewiesen"


def test_kostenbuch_traegt_den_mandanten(client, wege, umgebung):
    _post(client, "Bearer ark")
    from ai.services import kostenbuch as kb
    kb.KONTEXT.set({"pfad": "/ai/x", "aufrufer": "a", "quelle": "jwt", "tenant": "lesser"})
    kb.buchen(anbieter="x", art="geld", usd=0.1)
    zeilen = [json.loads(z) for p in (umgebung / "kb").glob("*.jsonl") for z in p.read_text().splitlines()]
    assert zeilen[-1]["tenant"] == "lesser"
