"""#3658: POST /ai/kette — Google -> ChatGPT -> Claude, nur ueber die Abos.

Die drei Wege sind hier Attrappen. Ein Test prueft eigens, dass die Frist der
Kette bis in den CLI-Prozess reicht.
"""
import asyncio
import json
import os
import subprocess
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import kette as k
from ai.services import ki_wege as kw


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    def verboten(*a, **kw_):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **kw_):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL", "REALTIME_AUTH_GRANT_URL", "API_ACCESS_KEY",
              "GEMINI_FORWARD_URL", "GEMINI_KERN_URL", "KETTE_AGY_TAG_MAX", "KETTE_CODEX_TAG_MAX",
              "KETTE_CLAUDE_TAG_MAX"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv(k.KETTE_ENV, "agy,codex,claude")


def morgen():
    return (datetime.now().astimezone() + timedelta(hours=5)).replace(microsecond=0).isoformat()


class Wege:
    """Attrappe: je Weg eine Liste von Ausgaengen (Text oder Ausnahme)."""

    def __init__(self, monkeypatch):
        self.plan = {"agy": ["agy-text"], "codex": ["codex-text"], "claude": ["claude-text"]}
        self.aufrufe = []
        self.fristen = []

        async def agy(prompt, model=None, api_key=None, request=None):
            return self._lauf("agy", prompt, model)

        async def codex(prompt, model=None, api_key=None, request=None, x_api_key=None):
            return self._lauf("codex", prompt, model)

        async def claude(prompt, model=None, api_key="placeholder", direkt=False):
            assert direkt is True
            return self._lauf("claude", prompt, model)
        monkeypatch.setattr(t, "gemini_endpoint", agy)
        monkeypatch.setattr(t, "chatgpt_endpoint", codex)
        monkeypatch.setattr(t, "_claude_mit_rechten", claude)

    def _lauf(self, weg, prompt, model):
        self.aufrufe.append((weg, model, prompt.effort, prompt.tools))
        self.fristen.append(t.CLI_FRIST.get())
        ausgang = self.plan[weg].pop(0) if self.plan[weg] else "x"
        if isinstance(ausgang, BaseException):
            raise ausgang
        return t.AIResponse(response=ausgang, model=model)


@pytest.fixture
def wege(monkeypatch):
    return Wege(monkeypatch)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def KONT(bis=None):
    return HTTPException(status_code=429, detail={"error": "kontingent", "weg": "x", "resets_at": bis})


FEHLT = HTTPException(status_code=500, detail={"error": "cli_fehlt", "weg": "x", "resets_at": None})
AUS = HTTPException(status_code=504, detail="Gemini CLI timeout - prompt may be too long")


def test_ohne_kette_aus_vor_dem_rumpf(client, wege, monkeypatch):
    monkeypatch.delenv(k.KETTE_ENV)
    r = client.post("/ai/kette", content=b"{kaputt", headers={"Content-Type": "application/json"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "kette_aus") and wege.aufrufe == []


def test_erster_weg(client, wege):
    r = client.post("/ai/kette", json={"prompt": "x"})
    d = r.json()
    assert r.status_code == 200 and (d["response"], d["routed_to"], d["model"]) == ("agy-text", "agy", "gemini-4-hydrogen-low")
    assert d["versucht"] == [{"weg": "agy", "code": "ok"}] and len(wege.aufrufe) == 1


@pytest.mark.parametrize("fehler,code", [(KONT(), "kontingent"), (FEHLT, "fehlt"), (AUS, "aussetzer"),
                                         (HTTPException(status_code=502, detail={"error": "x"}), "aussetzer"),
                                         (RuntimeError("weg"), "aussetzer")])
def test_wechsel_zum_naechsten(client, wege, fehler, code):
    wege.plan["agy"] = [fehler]
    d = client.post("/ai/kette", json={"prompt": "x"}).json()
    assert (d["routed_to"], d["response"]) == ("codex", "codex-text")
    assert d["versucht"] == [{"weg": "agy", "code": code}, {"weg": "codex", "code": "ok"}]


def test_leere_antwort_ist_ein_aussetzer(client, wege):
    wege.plan["agy"] = ["   "]
    d = client.post("/ai/kette", json={"prompt": "x"}).json()
    assert d["routed_to"] == "codex" and d["versucht"][0]["code"] == "aussetzer"


def test_fehler_der_anfrage_wechselt_nicht(client, wege):
    wege.plan["agy"] = [HTTPException(status_code=422, detail={"error": "unknown_model"})]
    r = client.post("/ai/kette", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "unknown_model")
    assert [a[0] for a in wege.aufrufe] == ["agy"]


def test_alle_leer_429_mit_fruehestem_reset(client, wege):
    frueh = (datetime.now().astimezone() + timedelta(hours=2)).replace(microsecond=0).isoformat()
    wege.plan = {"agy": [KONT(morgen())], "codex": [KONT(frueh)], "claude": [KONT()]}
    r = client.post("/ai/kette", json={"prompt": "x"})
    d = r.json()["detail"]
    assert r.status_code == 429 and (d["error"], d["resets_at"]) == ("kontingent", frueh)
    assert 0 < int(r.headers["Retry-After"]) <= 7200


@pytest.mark.parametrize("plan,code", [
    ({"agy": [AUS], "codex": [FEHLT], "claude": [KONT()]}, "wege_fehler"),
    ({"agy": [FEHLT], "codex": [KONT()], "claude": [FEHLT]}, "cli_fehlt"),
])
def test_ende_der_kette(client, wege, plan, code):
    wege.plan = plan
    r = client.post("/ai/kette", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, code)


def test_nur_erster_weg(client, wege):
    wege.plan["agy"] = [KONT(morgen())]
    r = client.post("/ai/kette", json={"prompt": "x", "nur_erster_weg": True})
    assert r.status_code == 429 and [a[0] for a in wege.aufrufe] == ["agy"]


def test_leerer_weg_wird_uebersprungen(client, wege):
    kw.ausgang("agy", "kontingent", morgen())
    d = client.post("/ai/kette", json={"prompt": "x"}).json()
    assert d["routed_to"] == "codex" and [a[0] for a in wege.aufrufe] == ["codex"]
    assert d["versucht"][0] == {"weg": "agy", "code": "kontingent", "uebersprungen": True}


def test_reihenfolge_kommt_nie_vom_aufrufer(client, wege, monkeypatch):
    monkeypatch.setenv(k.KETTE_ENV, "agy,codex")
    wege.plan["agy"] = [KONT()]
    d = client.post("/ai/kette", json={"prompt": "x", "wege": [{"weg": "claude"}, {"weg": "agy"}]}).json()
    assert d["routed_to"] == "codex" and "claude" not in [a[0] for a in wege.aufrufe]


def test_modell_und_staerke_je_weg(client, wege):
    wege.plan["agy"] = [KONT()]
    b = {"prompt": "x", "wege": [{"weg": "agy", "model": "gemini-4-hydrogen-medium"},
                                 {"weg": "codex", "model": "gpt-5.6-luna", "effort": "medium"}]}
    client.post("/ai/kette", json=b)
    assert wege.aufrufe == [("agy", "gemini-4-hydrogen-medium", None, None), ("codex", "gpt-5.6-luna", "medium", False)]


def test_claude_nur_haiku(client, wege):
    r = client.post("/ai/kette", json={"prompt": "x", "wege": [{"weg": "claude", "model": "opus"}]})
    assert r.status_code == 422 and wege.aufrufe == []
    wege.plan = {"agy": [KONT()], "codex": [KONT()], "claude": ["ok"]}
    d = client.post("/ai/kette", json={"prompt": "x"}).json()
    assert d["routed_to"] == "claude" and wege.aufrufe[-1][1] == "haiku"


@pytest.mark.parametrize("rumpf", [{"prompt": "x", "image_paths": ["/etc/passwd"]}, {"prompt": "x", "tools": True},
                                   {"prompt": "x", "budget_s": 500}, {"prompt": ""},
                                   {"prompt": "x", "wege": [{"weg": "agy"}, {"weg": "agy"}]},
                                   {"prompt": "x", "wege": [{"weg": "grok"}]}])
def test_ungueltiger_rumpf(client, wege, rumpf):
    assert client.post("/ai/kette", json=rumpf).status_code == 422 and wege.aufrufe == []


def test_deckel_je_weg(client, wege, monkeypatch):
    monkeypatch.setenv("KETTE_AGY_TAG_MAX", "1")
    wege.plan["agy"] = ["a1", "a2"]
    assert client.post("/ai/kette", json={"prompt": "x"}).json()["routed_to"] == "agy"
    d = client.post("/ai/kette", json={"prompt": "x"}).json()
    assert d["routed_to"] == "codex" and d["versucht"][0]["code"] == "kontingent"


def test_direkte_gemini_aufrufe_zaehlen_mit_und_sperren_nie(client, monkeypatch):
    async def agy(prompt, model=None, api_key=None, request=None):
        return t.AIResponse(response="ok", model="m")
    # Die Route /ai/gemini selbst ist echt; nur der Lauf ist ersetzt.
    monkeypatch.setattr(t, "_run_cli_with_pgid", lambda cmd, **kw_: subprocess.CompletedProcess(
        cmd, 0, json.dumps({"status": "SUCCESS", "response": "ok"}), ""))
    monkeypatch.setenv("KETTE_AGY_TAG_MAX", "1")
    codes = [client.post("/ai/gemini", json={"prompt": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 200] and kw.tag_stand("google_hydrogen") == 3


def test_budget_erschoepft(client, wege, monkeypatch):
    uhr = [1000.0]
    monkeypatch.setattr(k, "_uhr", lambda: uhr[0])

    def langsam(*a):
        uhr[0] += 40
        raise AUS
    wege.plan["agy"] = [langsam]
    orig = wege._lauf

    def lauf(weg, prompt, model):
        if weg == "agy":
            wege.aufrufe.append((weg, model, None, None))
            langsam()
        return orig(weg, prompt, model)
    wege._lauf = lauf
    r = client.post("/ai/kette", json={"prompt": "x", "budget_s": 30})
    assert (r.status_code, r.json()["detail"]["error"]) == (504, "budget")
    assert [a[0] for a in wege.aufrufe] == ["agy"]


def test_frist_reicht_bis_zur_cli(monkeypatch):
    """Die Frist der Kette kuerzt die Zeitgrenze des CLI-Prozesses (killpg am Ende)."""
    gesehen = {}

    class Proc:
        pid = 4242

        def communicate(self, input=None, timeout=None):
            gesehen["timeout"] = timeout
            return "", ""
        returncode = 0
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw_: Proc())
    marke = t.CLI_FRIST.set(k._uhr() + 7)
    try:
        t._run_cli_with_pgid(["true"], env={}, timeout=300)
    finally:
        t.CLI_FRIST.reset(marke)
    assert 5 <= gesehen["timeout"] <= 7
    t._run_cli_with_pgid(["true"], env={}, timeout=300)
    assert gesehen["timeout"] == 300


def test_frist_ist_waehrend_des_wegs_gesetzt(client, wege):
    client.post("/ai/kette", json={"prompt": "x", "budget_s": 20})
    assert wege.fristen[0] is not None and t.CLI_FRIST.get() is None


def test_nicht_angewendet(client, wege):
    d = client.post("/ai/kette", json={"prompt": "x", "max_tokens": 400, "temperature": 0.2}).json()
    assert d["nicht_angewendet"] == ["max_tokens", "temperature"]


def test_leeres_claude_wird_vermerkt(client, wege):
    bis = morgen()
    wege.plan = {"agy": [KONT()], "codex": [KONT()],
                 "claude": [HTTPException(status_code=503, detail={"error": "claude_account_unavailable",
                                                                   "grund": "limit", "resets": bis})]}
    client.post("/ai/kette", json={"prompt": "x"})
    assert kw.gesperrt("claude") == (True, bis)
    wege.aufrufe.clear()
    wege.plan = {"agy": [KONT()], "codex": [KONT()], "claude": ["nie"]}
    r = client.post("/ai/kette", json={"prompt": "x"})
    assert r.status_code == 429 and "claude" not in [a[0] for a in wege.aufrufe]
