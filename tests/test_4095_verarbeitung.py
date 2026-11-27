"""#4095: POST /ai/verarbeitung — Anbieter aus KI_VERARBEITUNG, nie vom Aufrufer.

Gegenproben aus dem Issue: openai ruft nie DeepSeek, hydrogen erzeugt keinen
bezahlten Aufruf, ohne Konfiguration kommt eine klare Meldung statt eines Ersatzes.
"""
import asyncio
import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import kette as k
from ai.services import openai_text as ot
from ai.services import verarbeitung as v

SCHLUESSEL = "betreiber-test-schluessel"


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    def verboten(*a, **kw_):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **kw_):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("API_ACCESS_KEY", "KETTE_AGY_TAG_MAX", "KETTE_CODEX_TAG_MAX", "KETTE_CLAUDE_TAG_MAX",
              "KETTE_BUDGET_HOECHSTENS_S", v.ENV, "OPENAI_API_KEY", "OPENAI_TEXT_DAILY_MAX_USD"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path))
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", SCHLUESSEL)
    # Die Kette der Instanz steht bewusst anders herum: die Rolle muss ihre eigene Reihe nehmen.
    monkeypatch.setenv(k.KETTE_ENV, "codex,agy,claude")

    async def kein_deepseek(*a, **kw):
        raise AssertionError("DeepSeek gerufen")
    monkeypatch.setattr(t, "deepseek_endpoint", kein_deepseek)


class Aufrufe:
    def __init__(self, monkeypatch):
        self.liste = []
        self.plan = {}

        def lauf(weg, prompt, model):
            self.liste.append((weg, model, prompt.system))
            aus = self.plan.get(weg, "antwort")
            if isinstance(aus, BaseException):
                raise aus
            return t.AIResponse(response=aus, model=model or weg)

        async def agy(prompt, model=None, api_key=None, request=None):
            return lauf("agy", prompt, model)

        async def codex(prompt, model=None, api_key=None, request=None, x_api_key=None):
            return lauf("codex", prompt, model)

        async def claude(prompt, model=None, api_key="placeholder", direkt=False):
            return lauf("claude", prompt, model)

        async def openai(prompt, model):
            return lauf("openai", prompt, model)
        monkeypatch.setattr(t, "gemini_endpoint", agy)
        monkeypatch.setattr(t, "chatgpt_endpoint", codex)
        monkeypatch.setattr(t, "_claude_mit_rechten", claude)
        monkeypatch.setattr(t, "_openai_einmal", openai)

    @property
    def wege(self):
        return [a[0] for a in self.liste]


@pytest.fixture
def auf(monkeypatch):
    return Aufrufe(monkeypatch)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def _post(client, **rumpf):
    rumpf.setdefault("prompt", "Ordne ein: Login kaputt")
    return client.post("/ai/verarbeitung", json=rumpf)


def test_ohne_konfiguration_klare_meldung_kein_anbieter(client, auf):
    r = _post(client)
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "verarbeitung_nicht_konfiguriert"
    assert auf.liste == []


@pytest.mark.parametrize("wert,grund", [
    ("deepseek", "anbieter_unbekannt"),
    ('{"anbieter": "hydrogen", "modell": "a b"}', "modell_ungueltig"),
    ('{"anbieter": ', "kein_json"),
    ("[1]", "anbieter_unbekannt"),
])
def test_ungueltige_konfiguration(client, auf, monkeypatch, wert, grund):
    monkeypatch.setenv(v.ENV, wert)
    r = _post(client)
    assert r.status_code == 503
    assert r.json()["detail"] == {**r.json()["detail"], "error": "verarbeitung_konfig_ungueltig", "grund": grund}
    assert auf.liste == []


def test_hydrogen_nur_agy_auch_wenn_kette_anders_steht(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "hydrogen")
    r = _post(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert auf.liste[0][:2] == ("agy", "gemini-4-hydrogen-low") and auf.wege == ["agy"]
    assert (d["anbieter"], d["weg"], d["text"]) == ("hydrogen", "agy", "antwort")


def test_hydrogen_faellt_nie_auf_einen_anderen_weg(client, auf, monkeypatch):
    """Betreiber mit Hydrogen erzeugt keinen bezahlten Aufruf und keinen Wechsel."""
    monkeypatch.setenv(v.ENV, '{"anbieter": "hydrogen", "modell": "gemini-4-hydrogen-high"}')
    auf.plan["agy"] = HTTPException(status_code=502, detail="agy kaputt")
    r = _post(client)
    assert r.status_code == 503
    assert auf.wege == ["agy"] and auf.liste[0][1] == "gemini-4-hydrogen-high"


def test_codex_nur_codex(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "codex")
    assert _post(client).status_code == 200
    assert auf.wege == ["codex"]


def test_kette_folgt_ki_kette(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "kette")
    auf.plan["codex"] = HTTPException(status_code=502, detail="aus")
    r = _post(client)
    assert r.status_code == 200 and auf.wege == ["codex", "agy"]


def test_openai_ruft_nie_deepseek_und_keine_kette(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, '{"anbieter": "openai", "modell": "gpt-5.4-mini"}')
    r = _post(client, max_ausgabe=400)
    assert r.status_code == 200, r.text
    assert auf.liste == [("openai", "gpt-5.4-mini", None)]
    assert "nicht_angewendet" not in r.json()


def test_aufrufer_nennt_kein_modell(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "hydrogen")
    r = _post(client, model="deepseek-v4-pro")
    assert r.status_code == 422 and auf.liste == []


def test_max_ausgabe_auf_abo_wird_gemeldet(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "hydrogen")
    assert _post(client, max_ausgabe=400).json()["nicht_angewendet"] == ["max_ausgabe"]


SCHEMA = {"type": "object", "required": ["intent", "title"],
          "properties": {"intent": {"type": "string"}, "title": {"type": "string"}}}


@pytest.mark.parametrize("antwort", [
    '{"intent": "issue", "title": "Login"}',
    '```json\n{"intent": "issue", "title": "Login"}\n```',
    'Hier: {"intent": "issue", "title": "Login"} fertig',
])
def test_schema_liefert_daten(client, auf, monkeypatch, antwort):
    monkeypatch.setenv(v.ENV, "hydrogen")
    auf.plan["agy"] = antwort
    r = _post(client, schema=SCHEMA, system="Du ordnest ein.")
    assert r.status_code == 200, r.text
    assert r.json()["daten"] == {"intent": "issue", "title": "Login"}
    system = auf.liste[0][2]
    assert system.startswith("Du ordnest ein.") and '"required": ["intent", "title"]' in system


@pytest.mark.parametrize("antwort,grund", [
    ("kein json", "kein_json"),
    ('{"intent": "issue"}', "pflichtfelder_fehlen:title"),
    ("[1, 2]", "kein_json"),
])
def test_schema_verletzt_gibt_502_statt_stillem_ersatz(client, auf, monkeypatch, antwort, grund):
    monkeypatch.setenv(v.ENV, "hydrogen")
    auf.plan["agy"] = antwort
    r = _post(client, schema=SCHEMA)
    assert r.status_code == 502
    d = r.json()["detail"]
    assert d["error"] == "antwort_kein_json" and d["grund"] == grund and d["anbieter"] == "hydrogen"


def test_budget_obergrenze(client, auf, monkeypatch):
    monkeypatch.setenv(v.ENV, "hydrogen")
    assert _post(client, budget_s=500).status_code == 422 and auf.liste == []


def test_stand_nur_mit_schluessel(client, monkeypatch):
    monkeypatch.delenv(v.BETREIBER_ENV, raising=False)
    monkeypatch.setenv(v.ENV, '{"anbieter": "openai"}')
    assert client.get("/ai/verarbeitung/stand").status_code == 401
    r = client.get("/ai/verarbeitung/stand", headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200
    assert r.json() == {"konfiguriert": True, "gueltig": True, "anbieter": "openai", "modell": None,
                        "modell_aus_konfig": False, "betreiber": {"mandant": "arkturian", "quelle": "vorgabe"},
                        "openai": {"schluessel": False, "tagesgrenze_usd": None}}


@pytest.mark.parametrize("wert,erwartet", [
    (None, {"konfiguriert": False, "gueltig": False, "fehler": "verarbeitung_nicht_konfiguriert", "grund": None,
            "betreiber": {"mandant": "arkturian", "quelle": "vorgabe"}}),
    ("hydrogen", {"konfiguriert": True, "gueltig": True, "anbieter": "hydrogen",
                  "modell": "gemini-4-hydrogen-low", "modell_aus_konfig": False, "betreiber": {"mandant": "arkturian", "quelle": "vorgabe"}}),
    ("quatsch", {"konfiguriert": True, "gueltig": False, "fehler": "verarbeitung_konfig_ungueltig",
                 "grund": "anbieter_unbekannt", "betreiber": {"mandant": "arkturian", "quelle": "vorgabe"}}),
])
def test_stand_werte(monkeypatch, wert, erwartet):
    monkeypatch.delenv(v.BETREIBER_ENV, raising=False)
    if wert:
        monkeypatch.setenv(v.ENV, wert)
    assert v.stand() == erwartet


def test_stand_kette_zeigt_reihe(monkeypatch):
    monkeypatch.setenv(v.ENV, "kette")
    assert v.stand()["reihe"] == ["codex", "agy", "claude"]


def test_openai_ohne_schluessel_echter_weg_503(client, monkeypatch):
    """Ohne Attrappe: der echte Schluesselweg verweigert ohne Schluessel, nichts geht hinaus."""
    monkeypatch.setenv(v.ENV, "openai")

    async def nie(*a, **kw):
        raise AssertionError("OpenAI gerufen")
    monkeypatch.setattr(ot.httpx.AsyncClient, "post", nie)
    r = _post(client)
    assert r.status_code == 503 and r.json()["detail"]["error"] == "openai_schluessel_fehlt"
