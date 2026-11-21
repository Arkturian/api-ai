"""#3645: feste Fehlercodes je CLI-Weg, Ausgaenge in /ai/wege."""
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import ki_wege as kw

AGY_QUOTA = "Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 42h15m28s."
CODEX_QUOTA = ("You’ve hit your usage limit. Visit https://chatgpt.com/settings/usage to purchase more credits "
               "or try again at Nov 21st, 2026 1:52 PM.")
GEMINI_LEER = "You have exhausted your capacity on this model. Your quota will reset after 9h15m42s."
# Erkennung der cloud-api (sprechstile.py, f0737382), muss bis zu ihrem Umbau weiter greifen.
KONTINGENT_CLOUD = re.compile(r"usage limit|rate limit|quota|out of credits|insufficient_quota|"
                              r"resource_exhausted|exhausted your", re.I)
GEMINI_502_CLOUD = re.compile(r"Upstream (?:502|429) from POST \S*/ai/gemini|agy_error", re.I)


@pytest.fixture(autouse=True)
def keine_cli(monkeypatch, tmp_path):
    def verboten(*a, **k):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL", "REALTIME_AUTH_GRANT_URL", "API_ACCESS_KEY",
              "CLAUDE_ROUTE", "GEMINI_FORWARD_URL", "AGY_RUN_AS_USER"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("CLI_HOME", str(tmp_path / "home"))


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def cli(monkeypatch):
    class Cli:
        rc, aus, err, wirft = 0, "", "", None

    def lauf(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        if Cli.wirft:
            raise Cli.wirft
        return subprocess.CompletedProcess(cmd, Cli.rc, Cli.aus, Cli.err)
    monkeypatch.setattr(t, "_run_cli_with_pgid", lauf)
    return Cli


AGY_OK = json.dumps({"status": "SUCCESS", "response": "pong", "usage": {"input_tokens": 1, "output_tokens": 1}})
CODEX_OK = "\n".join([json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "pong"}}),
                      json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})])


def agy_fehler(m):
    return json.dumps({"status": "ERROR", "error": m, "response": ""})


def codex_fehler(m):
    return json.dumps({"type": "turn.failed", "error": {"message": m}})


@pytest.mark.parametrize("text,code,zustand", [
    (AGY_QUOTA, "kontingent", "kontingent"),
    ('{"short_error":"RESOURCE_EXHAUSTED (code 429): x"}', "kontingent", "kontingent"),
    (CODEX_QUOTA, "kontingent", "kontingent"),
    (GEMINI_LEER, "kontingent", "kontingent"),
    ("Rate limit reached for requests", "kontingent", "kontingent"),
    ("insufficient_quota", "kontingent", "kontingent"),
    ("authentication failed or timed out", "cli_fehlt", "fehler"),
    ("Not logged in", "cli_fehlt", "nicht_angemeldet"),
    ("access token could not be refreshed", "cli_fehlt", "nicht_angemeldet"),
])
def test_einordnen(text, code, zustand):
    assert kw.einordnen(text) == (code, zustand)


@pytest.mark.parametrize("text", ["", "Error: empty prompt.", 'invalid model selection (--model "x")', "connection reset"])
def test_anderes_bleibt_ungeordnet(text):
    assert kw.einordnen(text) is None


def test_reset_agy_relativ():
    j = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    assert kw.resets_aus(AGY_QUOTA, j) == (j + timedelta(hours=42, minutes=15, seconds=28)).isoformat()


@pytest.mark.parametrize("uhr,h,mi", [("1:52 PM", 13, 52), ("12:05 AM", 0, 5), ("12:05 PM", 12, 5), ("11:59 PM", 23, 59)])
def test_reset_codex_absolut(uhr, h, mi):
    w = kw.resets_aus(f"try again at Nov 21st, 2026 {uhr}.")
    assert w == datetime(2026, 11, 21, h, mi).astimezone().replace(microsecond=0).isoformat()


@pytest.mark.parametrize("text", ["usage limit", "try again at Xyz 21st, 2026 1:52 PM", "try again at Feb 31st, 2026 1:52 PM",
                                  "Resets in soon", "Resets in 99999999999h"])
def test_reset_unbekannt_ist_none(text):
    assert kw.resets_aus(text) is None


WIEN = timezone(timedelta(hours=1))


@pytest.mark.parametrize("text,jetzt,soll", [
    (GEMINI_LEER, datetime(2026, 11, 21, 12, 0, tzinfo=WIEN), "2026-11-21T21:15:42+01:00"),
    ("Upgrade or try again at Nov 19th, 2026 7:20 PM.", datetime(2026, 11, 19, 12, 0, tzinfo=WIEN), "2026-11-19T19:20:00+01:00"),
    ("try again at Nov 19th 7:20 PM", datetime(2026, 11, 15, 23, 30, tzinfo=WIEN), "2026-11-19T19:20:00+01:00"),
    ("try again at Jan 2nd 9:00 AM", datetime(2026, 12, 30, 12, 0, tzinfo=WIEN), "2027-01-02T09:00:00+01:00"),
    ("try again at 10:25 PM", datetime(2026, 11, 21, 12, 0, tzinfo=WIEN), "2026-11-21T22:25:00+01:00"),
    ("try again at 10:25 PM", datetime(2026, 11, 21, 23, 0, tzinfo=WIEN), "2026-11-22T22:25:00+01:00"),
    ("try again at 12:05 AM", datetime(2026, 11, 21, 23, 0, tzinfo=WIEN), "2026-11-22T00:05:00+01:00"),
    ("try again in 3 hours", datetime(2026, 11, 15, 23, 30, tzinfo=WIEN), "2026-11-16T02:30:00+01:00"),
    ("try again in 1 day 2 hours", datetime(2026, 11, 15, 23, 30, tzinfo=WIEN), "2026-11-17T01:30:00+01:00"),
    ("try again in 42m", datetime(2026, 11, 15, 12, 0, tzinfo=WIEN), "2026-11-15T12:42:00+01:00"),
])
def test_reset_formate_der_cloud_api(text, jetzt, soll):
    assert kw.resets_aus(text, jetzt) == soll


@pytest.mark.parametrize("text", ["try again at Nov 31st, 2026 7:20 PM", "try again at sometime later", "try again at 25:00"])
def test_reset_unsinn_ist_none(text):
    assert kw.resets_aus(text) is None


def test_gemini_stderr_kontingent(client, cli):
    cli.rc, cli.err = 1, GEMINI_LEER
    r = client.post("/ai/gemini", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"]) == (500, "kontingent", "agy") and d["resets_at"]


def test_gemini_nicht_installiert(client, cli):
    cli.wirft = FileNotFoundError("agy")
    r = client.post("/ai/gemini", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"], d["resets_at"]) == (500, "cli_fehlt", "agy", None)
    assert "Gemini CLI not installed on server" in d["hint"]


def test_chatgpt_nicht_installiert(client, cli):
    cli.wirft = FileNotFoundError("codex")
    r = client.post("/ai/chatgpt", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"]) == (500, "cli_fehlt", "codex")


def test_gemini_kontingent_bleibt_502_und_erkennbar(client, cli):
    cli.aus = agy_fehler(AGY_QUOTA)
    r = client.post("/ai/gemini", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"], d["alt_error"]) == (502, "kontingent", "agy", "agy_error")
    assert d["resets_at"] and d["agy_message"] == AGY_QUOTA
    wie_mcp = f"Upstream 502 from POST http://127.0.0.1:8000/ai/gemini: {json.dumps(d)}"
    assert KONTINGENT_CLOUD.search(wie_mcp) and GEMINI_502_CLOUD.search(wie_mcp)


def test_gemini_anmeldung_ist_cli_fehlt(client, cli):
    cli.aus = agy_fehler("authentication failed or timed out")
    d = client.post("/ai/gemini", json={"prompt": "x"}).json()["detail"]
    assert (d["error"], d["alt_error"]) == ("cli_fehlt", "agy_error")


def test_gemini_anderer_fehler_bleibt_agy_error(client, cli):
    cli.aus = agy_fehler("Error: empty prompt.")
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (502, "agy_error")


def test_chatgpt_kontingent_bleibt_502_und_erkennbar(client, cli):
    cli.aus = codex_fehler(CODEX_QUOTA)
    r = client.post("/ai/chatgpt", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"]) == (502, "kontingent", "codex")
    assert d["resets_at"].startswith("2026-11-21T13:52")
    assert "Codex CLI upstream error" in d["hint"]
    assert KONTINGENT_CLOUD.search(f"Upstream 502 from POST x: {json.dumps(d)}")


def test_chatgpt_stderr_kontingent(client, cli):
    cli.rc, cli.err = 1, CODEX_QUOTA
    r = client.post("/ai/chatgpt", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (500, "kontingent")


def test_chatgpt_anderer_fehler_bleibt_text(client, cli):
    cli.aus = codex_fehler("model overloaded")
    r = client.post("/ai/chatgpt", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]) == (502, "Codex CLI upstream error: model overloaded")


def wege(monkeypatch, client, auth=None):
    monkeypatch.setattr(main, "_cli_status", lambda n: {"ok": True, "auth": auth or {"ok": None, "reason": "not_readable_by_service"}})
    monkeypatch.setattr(kw, "agy", lambda: kw._weg("fehler", grund="anmeldung_nicht_lesbar"))
    r = client.get("/ai/wege", headers={"X-API-KEY": "dienst"})
    assert r.status_code == 200
    return r.json()["wege"]


def test_erfolg_macht_nicht_lesbar_zu_ok(monkeypatch, client, cli):
    assert wege(monkeypatch, client)["agy"]["zustand"] == "fehler"
    cli.aus = AGY_OK
    assert client.post("/ai/gemini", json={"prompt": "x"}).status_code == 200
    cli.aus = CODEX_OK
    assert client.post("/ai/chatgpt", json={"prompt": "x"}).status_code == 200
    w = wege(monkeypatch, client)
    for n in ("agy", "codex"):
        assert w[n]["zustand"] == "ok" and w[n]["letzter_erfolg"] and "grund" not in w[n]


def test_kontingent_in_wege_mit_bis(monkeypatch, client, cli):
    cli.aus = agy_fehler(AGY_QUOTA)
    client.post("/ai/gemini", json={"prompt": "x"})
    w = wege(monkeypatch, client)["agy"]
    assert w["zustand"] == "kontingent" and w["bis"]


def test_abgelaufenes_kontingent_gilt_als_angemeldet(monkeypatch, client):
    kw.ausgang("codex", "kontingent", "2020-01-01T00:00:00+00:00")
    w = wege(monkeypatch, client, {"ok": True})["codex"]
    assert (w["zustand"], w["bis"]) == ("angemeldet", None)


def test_kontingent_ohne_bis_laeuft_aus(monkeypatch, client):
    kw.ausgang("codex", "kontingent")
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "kontingent"
    a = kw._ausgang_lesen("codex"); a["zeit"] = time.time() - kw._KONTINGENT_OHNE_BIS_GILT_S - 1
    kw._ausgang_pfad("codex").write_text(json.dumps(a))
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "angemeldet"


def test_fehlschlag_laeuft_aus(monkeypatch, client):
    kw.ausgang("codex", "nicht_angemeldet")
    w = wege(monkeypatch, client, {"ok": True})["codex"]
    assert (w["zustand"], w["grund"]) == ("nicht_angemeldet", "letzter_aufruf")
    a = kw._ausgang_lesen("codex"); a["zeit"] = time.time() - kw._FEHLER_GILT_S - 1
    kw._ausgang_pfad("codex").write_text(json.dumps(a))
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "angemeldet"


def anmeldung(zeit=None):
    p = kw.Path(os.environ["CODEX_HOME"]) / "auth.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    if zeit:
        os.utime(p, (zeit, zeit))


def test_gebuchtes_fehlt_schlaegt_keine_neue_installation(monkeypatch, client):
    kw.ausgang("codex", "fehlt")
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "angemeldet"


def test_gebuchtes_fehlt_gilt_wenn_die_platte_nichts_sagt(monkeypatch, client):
    kw.ausgang("codex", "fehlt")
    assert wege(monkeypatch, client)["codex"]["zustand"] == "fehlt"


def test_neue_anmeldung_verwirft_alten_fehlschlag(monkeypatch, client):
    anmeldung(time.time() - 600)
    kw.ausgang("codex", "nicht_angemeldet")
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "nicht_angemeldet"
    anmeldung(time.time() + 5)
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "angemeldet"


def test_neue_anmeldung_agy(monkeypatch, client, tmp_path):
    monkeypatch.setattr(kw.shutil, "which", lambda n: "/usr/bin/agy")
    kw.ausgang("agy", "nicht_angemeldet")
    t = kw.Path(os.environ["CLI_HOME"]).joinpath(".gemini", "antigravity-cli")
    t.mkdir(parents=True)
    (t / "antigravity-oauth-token").write_text("x")
    os.utime(t / "antigravity-oauth-token", (time.time() + 5, time.time() + 5))
    monkeypatch.setattr(main, "_cli_status", lambda n: {"ok": True, "auth": {"ok": True}})
    assert client.get("/ai/wege", headers={"X-API-KEY": "dienst"}).json()["wege"]["agy"]["zustand"] == "angemeldet"


def test_token_erneuerung_hebt_kontingent_nicht_auf(monkeypatch, client):
    kw.ausgang("codex", "kontingent", (datetime.now().astimezone() + timedelta(hours=3)).isoformat())
    anmeldung(time.time() + 5)
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] == "kontingent"


def test_fehlende_cli_schlaegt_alten_erfolg(monkeypatch, client):
    kw.ausgang("codex", "ok")
    monkeypatch.setattr(main, "_cli_status", lambda n: {"ok": False, "error": "codex: not found on PATH"})
    r = client.get("/ai/wege", headers={"X-API-KEY": "dienst"})
    w = r.json()["wege"]["codex"]
    assert w["zustand"] == "fehlt" and w["letzter_erfolg"]


@pytest.mark.parametrize("auth,soll", [({"ok": False, "reason": "expired", "has_refresh_token": True}, "angemeldet"),
                                       ({"ok": False, "reason": "expired", "has_refresh_token": False}, "nicht_angemeldet")])
def test_abgelaufener_zugriffstoken(auth, soll):
    assert kw.codex(lambda n: {"ok": True, "auth": auth})["zustand"] == soll


@pytest.mark.parametrize("roh", ["{", "[]", '{"art":"kontingent","bis":"unsinn","zeit":"x"}', '{"art":"kontingent","bis":5}'])
def test_kaputte_ausgangsdatei_wirft_nicht(monkeypatch, client, roh):
    p = kw._ausgang_pfad("codex"); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(roh)
    assert wege(monkeypatch, client, {"ok": True})["codex"]["zustand"] in ("angemeldet", "kontingent")


def test_buchung_scheitert_trifft_aufruf_nicht(monkeypatch, client, cli):
    monkeypatch.setenv("VERBRAUCH_DIR", "/proc/nicht-schreibbar")
    cli.aus = CODEX_OK
    assert client.post("/ai/chatgpt", json={"prompt": "x"}).status_code == 200


def test_ausgang_speichert_weder_text_noch_fremden_weg(monkeypatch, client, cli):
    cli.aus = codex_fehler(CODEX_QUOTA + " geheimer-prompt")
    client.post("/ai/chatgpt", json={"prompt": "geheimer-prompt"})
    roh = kw._ausgang_pfad("codex").read_text()
    assert "geheimer" not in roh and oct(kw._ausgang_pfad("codex").stat().st_mode & 0o777) == "0o600"


def test_nur_bekannte_wege_werden_gebucht():
    d = kw._ausgang_pfad("codex").parent
    d.mkdir(parents=True, exist_ok=True)
    vorher = sorted(x.name for x in d.iterdir())
    kw.ausgang("../x", "ok")
    kw.ausgang("fremd", "ok")
    assert sorted(x.name for x in d.iterdir()) == vorher
