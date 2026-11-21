"""#3640: Textweg ueber den OpenAI-Schluessel der Instanz, #3639: /ai/wege."""
import json
import logging
import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ai import auth_guard as ag
from ai.services import ki_wege, openai_text as ot

KEY = "sk-kunde-geheim-1234567890"
INSTANZ = "instanzschluessel"
BETREIBER = "betreiberschluessel"


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    for n in ("REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL", "REALTIME_AUTH_GRANT_URL",
              "REALTIME_GRANT_SERVICE_KEY", ot.ANFRAGEN_ENV, "AGY_RUN_AS_USER"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("API_ACCESS_KEY", INSTANZ)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv(ot.GRENZE_ENV, "0.5")
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(ot, "_monatsdeckel_greift", lambda: False)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False, headers={"X-API-KEY": INSTANZ})


@pytest.fixture
def anonym():
    return TestClient(main.app, raise_server_exceptions=False)


def antwort(text="pong", status="completed", usage=None, **extra):
    return {"id": "resp_1", "model": "gpt-6-luna", "status": status,
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": text}]}] if text else [],
            "usage": usage if usage is not None else {
                "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 100,
                "output_tokens_details": {"reasoning_tokens": 40}, "total_tokens": 1100}, **extra}


class Netz:
    def __init__(self):
        self.anfragen, self.antworten, self.fehler = [], [], None

    def install(self, monkeypatch):
        netz = self

        class Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                netz.anfragen.append((url, json, headers))
                if netz.fehler:
                    raise netz.fehler
                code, daten, kopf = netz.antworten.pop(0) if netz.antworten else (200, antwort(), {})
                return httpx.Response(code, json=daten, headers=kopf)

        monkeypatch.setattr(ot.httpx, "AsyncClient", Client)
        return self


@pytest.fixture
def netz(monkeypatch):
    return Netz().install(monkeypatch)


def heute():
    return ot.stand()


def test_erfolg_rumpf_und_antwort(client, netz):
    r = client.post("/ai/openai", json={"prompt": "Sag pong", "system": "Kurz.", "max_tokens": 300})
    assert r.status_code == 200
    d = r.json()
    assert (d["response"], d["model"], d["tokens_used"], d["thinking_tokens"], d["finish_reason"], d["effort_applied"]) \
        == ("pong", "gpt-6-luna", 1100, 40, "stop", "low")
    url, rumpf, kopf = netz.anfragen[0]
    assert url == "https://api.openai.com/v1/responses"
    assert rumpf == {"model": "gpt-6-luna", "input": "Sag pong", "instructions": "Kurz.", "max_output_tokens": 300,
                     "reasoning": {"effort": "low"}, "store": False}
    assert kopf == {"Authorization": f"Bearer {KEY}"}


def test_kosten_mit_cache_und_buchung(client, netz):
    client.post("/ai/openai", json={"prompt": "x"})
    soll = (800 * .10 + 200 * .01 + 100 * .50) / 1e6
    assert heute()["heute_usd"] == pytest.approx(soll, abs=1e-6)
    assert (heute()["heute_anfragen"], heute()["heute_erfolg"]) == (1, 1)


def test_ohne_schluessel_503_ohne_netz(monkeypatch, client, netz):
    monkeypatch.delenv("OPENAI_API_KEY")
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "openai_schluessel_fehlt")
    assert netz.anfragen == []


@pytest.mark.parametrize("wert", [None, "", "0", "-1", "nan", "inf", "abc"])
def test_ohne_gueltige_grenze_503_ohne_netz(monkeypatch, client, netz, wert):
    monkeypatch.delenv(ot.GRENZE_ENV, raising=False)
    if wert is not None:
        monkeypatch.setenv(ot.GRENZE_ENV, wert)
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "openai_text_grenze_fehlt")
    assert netz.anfragen == []


def test_tagesgrenze_429_kontingent_ohne_netz(monkeypatch, client, netz):
    monkeypatch.setenv(ot.GRENZE_ENV, "0.0001")
    r = client.post("/ai/openai", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["grund"], d["weg"]) == (429, "kontingent", "tagesgrenze", "openai_schluessel")
    assert d["resets_at"] and "quota" in d["hint"] and int(r.headers["retry-after"]) > 0
    assert netz.anfragen == []


def test_anfragengrenze(monkeypatch, client, netz):
    monkeypatch.setenv(ot.ANFRAGEN_ENV, "1")
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 200
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 429
    assert len(netz.anfragen) == 1


def test_reservierung_haelt_die_grenze_unter_gleichzeitigkeit(monkeypatch):
    monkeypatch.setenv(ot.GRENZE_ENV, "0.001")
    ot.reservieren(0.0006)
    with pytest.raises(Exception) as e:
        ot.reservieren(0.0006)
    assert e.value.status_code == 429


def test_monatsdeckel_sperrt(monkeypatch, client, netz):
    monkeypatch.setattr(ot, "_monatsdeckel_greift", lambda: True)
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["grund"]) == (429, "monatsdeckel")
    assert netz.anfragen == []


def test_monatsdeckel_nicht_pruefbar_bleibt_zu(monkeypatch, client, netz):
    def kaputt():
        raise RuntimeError("x")
    monkeypatch.setattr(ot, "_monatsdeckel_greift", kaputt)
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 503
    assert netz.anfragen == []


@pytest.mark.parametrize("code,typ", [("insufficient_quota", "insufficient_quota"),
                                      ("credit_balance_exhausted", "insufficient_quota"),
                                      ("project_spend_limit_exceeded", "x"), (None, "insufficient_quota")])
def test_abrechnungsfehler_ist_kontingent(client, netz, code, typ):
    netz.antworten.append((429, {"error": {"code": code, "type": typ, "message": "m"}}, {}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"], r.json()["detail"]["grund"]) == (429, "kontingent", "openai_guthaben")
    assert (heute()["heute_usd"], heute()["heute_anfragen"]) == (0, 0)


def test_rate_limit_mit_retry(client, netz):
    netz.antworten.append((429, {"error": {"type": "rate_limit_error", "message": "m"}}, {"retry-after": "7"}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["grund"], r.headers["retry-after"]) == (429, "openai_rate_limit", "7") and d["resets_at"]


@pytest.mark.parametrize("status,fehler", [(401, "openai_schluessel_abgelehnt"), (403, "openai_schluessel_abgelehnt"),
                                           (400, "openai_text_fehler"), (500, "openai_text_fehler")])
def test_fremder_fehlertext_kommt_nicht_durch(client, netz, caplog, status, fehler):
    netz.antworten.append((status, {"error": {"code": "c", "type": "t", "message": f"Incorrect key {KEY}"}}, {}))
    with caplog.at_level(logging.DEBUG):
        r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (502, fehler)
    assert KEY not in r.text and KEY not in caplog.text
    assert heute()["heute_usd"] == 0


def test_timeout_behaelt_reservierung(client, netz):
    netz.fehler = httpx.ReadTimeout("t")
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (504, "openai_text_timeout")
    assert heute()["heute_usd"] > 0


def test_verbindungsfehler_storniert(client, netz):
    netz.fehler = httpx.ConnectError("c")
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 502
    assert heute()["heute_usd"] == 0


def test_abbruch_nach_senden_behaelt_reservierung(client, netz):
    netz.fehler = httpx.RemoteProtocolError("r")
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 502
    assert heute()["heute_usd"] > 0


def test_gekuerzte_antwort_meldet_length(client, netz):
    netz.antworten.append((200, antwort(status="incomplete", incomplete_details={"reason": "max_output_tokens"}), {}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["finish_reason"]) == (200, "length")


def test_leere_antwort_502_aber_gebucht(client, netz):
    netz.antworten.append((200, antwort(text="", status="incomplete", incomplete_details={"reason": "max_output_tokens"}), {}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (502, "openai_text_leer")
    assert heute()["heute_usd"] > 0


def test_ohne_usage_bleibt_hoechstbetrag(client, netz):
    netz.antworten.append((200, antwort(usage={}), {}))
    client.post("/ai/openai", json={"prompt": "x"})
    assert heute()["heute_usd"] == pytest.approx(ot.hoechstbetrag("gpt-6-luna", 1, 1000), abs=1e-6)


@pytest.mark.parametrize("rumpf,code,fehler", [
    ({"prompt": "x", "model": "gpt-6-sol"}, 422, "modell_nicht_freigegeben"),
    ({"prompt": "x", "effort": "max"}, 422, "invalid_effort"),
    ({"prompt": "x" * (ot.MAX_EINGABE_ZEICHEN + 1)}, 422, "prompt_zu_lang"),
    ({"prompt": ""}, 422, "prompt_leer"),
    ({"prompt": "x", "tools": True}, 403, "tools_disabled"),
    ({"prompt": "x", "response_format": {"type": "json_object"}}, 422, "response_format_unsupported_on_endpoint"),
    ({"prompt": "x", "conversation_history": [{"role": "user", "content": "a"}]}, 422, "conversation_history_unsupported_on_endpoint"),
    ({"prompt": "x", "persona_variant": "default"}, 422, "persona_unsupported_on_endpoint"),
    ({"prompt": "x", "model_fallback": True}, 422, "model_fallback_unsupported_on_endpoint"),
    ({"prompt": "x", "image_paths": ["/tmp/a.png"]}, 422, "images_unsupported_on_endpoint"),
])
def test_abgewiesen_ohne_netz(client, netz, rumpf, code, fehler):
    r = client.post("/ai/openai", json=rumpf)
    assert (r.status_code, r.json()["detail"]["error"]) == (code, fehler)
    assert netz.anfragen == [] and heute()["heute_anfragen"] == 0


def test_system_zaehlt_zur_laenge(client, netz):
    r = client.post("/ai/openai", json={"prompt": "x", "system": "y" * ot.MAX_EINGABE_ZEICHEN})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "prompt_zu_lang")


@pytest.mark.parametrize("wunsch,soll", [(999999, ot.MAX_AUSGABE_TOKENS), (1, ot.MIN_AUSGABE_TOKENS), (-5, 1000)])
def test_ausgabe_ist_gedeckelt(client, netz, wunsch, soll):
    client.post("/ai/openai", json={"prompt": "x", "max_tokens": wunsch})
    assert netz.anfragen[0][1]["max_output_tokens"] == soll


def test_idempotenz_replay_ohne_zweiten_aufruf(client, netz):
    b = {"prompt": "x", "request_id": "sprechstil-0001"}
    assert client.post("/ai/openai", json=b).status_code == 200
    r = client.post("/ai/openai", json=b)
    assert r.status_code == 200 and r.json()["response"] == "pong"
    assert len(netz.anfragen) == 1 and heute()["heute_anfragen"] == 1


def test_idempotenz_anderer_rumpf_409(client, netz):
    client.post("/ai/openai", json={"prompt": "x", "request_id": "sprechstil-0002"})
    assert client.post("/ai/openai", json={"prompt": "y", "request_id": "sprechstil-0002"}).status_code == 409


def test_idempotenz_nach_timeout_kein_zweiter_aufruf(client, netz):
    netz.fehler = httpx.ReadTimeout("t")
    b = {"prompt": "x", "request_id": "sprechstil-0003"}
    assert client.post("/ai/openai", json=b).status_code == 504
    netz.fehler = None
    assert client.post("/ai/openai", json=b).status_code == 409
    assert len(netz.anfragen) == 1


def test_idempotenz_nach_tagesgrenze_wiederholbar(monkeypatch, client, netz):
    monkeypatch.setenv(ot.GRENZE_ENV, "0.0001")
    b = {"prompt": "x", "request_id": "sprechstil-0004"}
    assert client.post("/ai/openai", json=b).status_code == 429
    monkeypatch.setenv(ot.GRENZE_ENV, "0.5")
    assert client.post("/ai/openai", json=b).status_code == 200


def test_kundeninstanz_verlangt_instanzschluessel(anonym, netz):
    assert anonym.post("/ai/openai", json={}).status_code == 401
    assert anonym.post("/ai/openai", json={"prompt": "x"}, headers={"X-API-KEY": "falsch"}).status_code == 401
    assert netz.anfragen == []
    assert anonym.post("/ai/openai", json={"prompt": "x"}, headers={"X-API-KEY": INSTANZ}).status_code == 200


def test_ohne_api_access_key_nicht_offen(monkeypatch, anonym, netz):
    """P1 (Cloud-Security): Das Bezahltor laesst ohne API_ACCESS_KEY alles durch."""
    monkeypatch.delenv("API_ACCESS_KEY")
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", BETREIBER)
    assert anonym.post("/ai/openai", json={"prompt": "x"}).status_code == 401
    assert anonym.post("/ai/openai", json={}).status_code == 401
    assert netz.anfragen == []
    assert anonym.post("/ai/openai", json={"prompt": "x"}, headers={"X-API-KEY": BETREIBER}).status_code == 200


def test_ganz_ohne_schluessel_503(monkeypatch, anonym, netz):
    monkeypatch.delenv("API_ACCESS_KEY")
    r = anonym.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "operator_key_not_configured")
    assert netz.anfragen == []


@pytest.mark.parametrize("methode,pfad", [("get", "/ai/openai/cost-status"), ("post", "/ai/openai")])
def test_beide_schluessel_gesetzt(monkeypatch, anonym, netz, methode, pfad):
    """P2 (Cloud-Security): Der Betreiberschluessel darf nicht am Bezahltor scheitern."""
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", BETREIBER)
    json_ = {"json": {"prompt": "x"}} if methode == "post" else {}
    for kopf in (INSTANZ, BETREIBER, f"  {INSTANZ}  "):
        assert getattr(anonym, methode)(pfad, headers={"X-API-KEY": kopf}, **json_).status_code == 200, kopf
    for kopf in ("falsch", "\u00e4".encode("latin-1")):
        assert getattr(anonym, methode)(pfad, headers={"X-API-KEY": kopf}, **json_).status_code == 401
    assert getattr(anonym, methode)(pfad, **json_).status_code == 401


def test_betreiberschluessel_oeffnet_keinen_anderen_bezahlpfad(monkeypatch, anonym):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", BETREIBER)
    for pfad in ("/ai/m3", "/ai/deepseek", "/ai/genimage", "/ai/openaix"):
        r = anonym.post(pfad, json={}, headers={"X-API-KEY": BETREIBER})
        assert r.status_code in (401, 404), pfad
        if pfad != "/ai/openaix":
            assert r.json()["detail"]["error"] == "api_key_required"


def test_jede_openai_route_hat_ihr_tor():
    from fastapi.routing import APIRoute
    routen = [r for r in main.app.routes if isinstance(r, APIRoute)
              and (r.path == "/ai/openai" or r.path.startswith("/ai/openai/"))]
    assert len(routen) == 3
    for r in routen:
        assert any(d.dependency is ag.betreiber_oder_instanz for d in r.dependencies), r.path


def test_kaputtes_bis_wirft_nicht(monkeypatch, client):
    ot._dir().mkdir(parents=True, exist_ok=True)
    ot._schreiben(ot._ausgangsdatei(), {"letzter_ausgang": "kontingent", "bis": "2026-11-21T10:00:00"})
    assert ot.weg_zustand()["zustand"] == "kontingent"
    ot._schreiben(ot._ausgangsdatei(), {"letzter_ausgang": "kontingent", "bis": "unsinn"})
    assert ot.weg_zustand()["zustand"] == "kontingent"


def test_ist_bezahlpfad():
    assert main._ist_bezahlpfad("/ai/openai") and main._ist_bezahlpfad("/ai/openai/cost-status")
    assert not main._ist_bezahlpfad("/ai/openaix")


def test_cost_status_nicht_von_kennung_verdeckt(monkeypatch, client):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    r = client.get("/ai/openai/cost-status", headers={"X-API-KEY": "dienst"})
    assert r.status_code == 200 and r.json()["grenze_usd"] == 0.5


@pytest.mark.parametrize("pfad", ["/ai/openai/cost-status", "/ai/openai/sprechstil-0001", "/ai/wege"])
def test_status_nicht_oeffentlich(monkeypatch, anonym, pfad):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    assert anonym.get(pfad, headers={"X-Real-IP": "203.0.113.9"}).status_code == 401


@pytest.mark.parametrize("pfad", ["/ai/openai/cost-status", "/ai/wege"])
def test_dieser_host_ist_keine_ausnahme(monkeypatch, anonym, pfad):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    monkeypatch.setattr(ag, "von_diesem_host", lambda r: True)
    assert anonym.get(pfad).status_code == 401


def test_wege_mit_instanzschluessel(monkeypatch, anonym):
    monkeypatch.setattr(main, "_cli_status", lambda n: {"ok": False, "error": "codex: not found on PATH"})
    assert anonym.get("/ai/wege", headers={"X-API-KEY": INSTANZ}).status_code == 200
    assert anonym.get("/ai/wege", headers={"X-API-KEY": "falsch"}).status_code == 401
    assert anonym.get("/ai/wege", headers={"X-API-KEY": "\u00e4".encode("latin-1")}).status_code == 401


def test_wege_ohne_jeden_schluessel_503(monkeypatch, anonym):
    monkeypatch.delenv("API_ACCESS_KEY")
    r = anonym.get("/ai/wege")
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "operator_key_not_configured")


def test_fehlender_schluessel_verbraucht_keine_kennung(monkeypatch, client, netz):
    b = {"prompt": "x", "request_id": "sprechstil-0005"}
    monkeypatch.delenv("OPENAI_API_KEY")
    assert client.post("/ai/openai", json=b).status_code == 503
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert client.post("/ai/openai", json=b).status_code == 200


@pytest.mark.parametrize("wert", ["inf", "nan", "1e999", "Wed, 21 Oct 2026 07:28:00 GMT", "9999999999"])
def test_unsinniges_retry_after_storniert(client, netz, wert):
    netz.antworten.append((429, {"error": {"type": "rate_limit_error"}}, {"retry-after": wert}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["grund"]) == (429, "openai_rate_limit")
    assert "retry-after" not in r.headers and heute()["heute_usd"] == 0


@pytest.mark.parametrize("daten", [[], "x", {"error": "x"}])
def test_seltsamer_fehlerrumpf(client, netz, daten):
    netz.antworten.append((500, daten, {}))
    assert client.post("/ai/openai", json={"prompt": "x"}).status_code == 502
    assert heute()["heute_usd"] == 0


@pytest.mark.parametrize("daten", [[], "x", {"usage": "kaputt", "output": []}, {"usage": {"input_tokens": "x"}}])
def test_unlesbare_antwort_bleibt_gebucht(client, netz, daten):
    netz.antworten.append((200, daten, {}))
    r = client.post("/ai/openai", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (502, "openai_text_antwort_unlesbar")
    assert heute()["heute_usd"] > 0


def test_health_verraet_keine_wege():
    roh = json.dumps(main.health_check())
    assert "openai_schluessel" not in roh and "fingerabdruck" not in roh and "wege" not in roh


def wege(monkeypatch, client, status=None):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    monkeypatch.setattr(main, "_cli_status", lambda n: status or {"ok": False, "error": "codex: not found on PATH"})
    r = client.get("/ai/wege", headers={"X-API-KEY": "dienst"})
    assert r.status_code == 200 and KEY not in r.text
    return r.json()["wege"]


def test_wege_form_und_fingerabdruck(monkeypatch, client):
    monkeypatch.setattr(ki_wege.shutil, "which", lambda n: None)
    monkeypatch.setattr(ki_wege.cli_dienst, "aktiv", lambda: False)
    w = wege(monkeypatch, client)
    assert w["agy"] == {"zustand": "fehlt", "bis": None, "letzter_erfolg": None}
    assert w["codex"]["zustand"] == "fehlt"
    o = w["openai_schluessel"]
    assert (o["zustand"], o["grenze_usd"], o["heute_usd"]) == ("angemeldet", 0.5, 0)
    assert o["fingerabdruck"] == ot.fingerabdruck() and len(o["fingerabdruck"]) == 8


@pytest.mark.parametrize("auth,soll", [({"ok": True}, "angemeldet"), ({"ok": False, "reason": "absent"}, "nicht_angemeldet"),
                                       ({"ok": False, "reason": "expired"}, "nicht_angemeldet"),
                                       ({"ok": None, "reason": "not_readable_by_service"}, "fehler")])
def test_wege_codex(monkeypatch, client, auth, soll):
    assert wege(monkeypatch, client, {"ok": True, "auth": auth})["codex"]["zustand"] == soll


def test_wege_agy_ohne_token(monkeypatch, client, tmp_path):
    monkeypatch.setattr(ki_wege.shutil, "which", lambda n: "/usr/bin/agy")
    monkeypatch.setenv("CLI_HOME", str(tmp_path))
    assert wege(monkeypatch, client)["agy"]["zustand"] == "nicht_angemeldet"
    t = tmp_path.joinpath(".gemini", "antigravity-cli")
    t.mkdir(parents=True)
    (t / "antigravity-oauth-token").write_text("x")
    assert wege(monkeypatch, client)["agy"]["zustand"] == "angemeldet"


def test_wege_schluessel_zustaende(monkeypatch, client, netz):
    assert wege(monkeypatch, client)["openai_schluessel"]["zustand"] == "angemeldet"
    client.post("/ai/openai", json={"prompt": "x"})
    w = wege(monkeypatch, client)["openai_schluessel"]
    assert w["zustand"] == "ok" and w["letzter_erfolg"]
    netz.antworten.append((429, {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}}, {}))
    client.post("/ai/openai", json={"prompt": "x"})
    assert wege(monkeypatch, client)["openai_schluessel"]["zustand"] == "kontingent"
    netz.antworten.append((500, {"error": {}}, {}))
    client.post("/ai/openai", json={"prompt": "x"})
    assert wege(monkeypatch, client)["openai_schluessel"]["zustand"] == "fehler"
    monkeypatch.delenv(ot.GRENZE_ENV)
    w = wege(monkeypatch, client)["openai_schluessel"]
    assert (w["zustand"], w["grund"]) == ("fehler", "tagesgrenze_fehlt")
    monkeypatch.delenv("OPENAI_API_KEY")
    w = wege(monkeypatch, client)["openai_schluessel"]
    assert (w["zustand"], w["fingerabdruck"]) == ("fehlt", None)


def test_wege_tagesgrenze_erreicht(monkeypatch, client, netz):
    monkeypatch.setenv(ot.ANFRAGEN_ENV, "1")
    client.post("/ai/openai", json={"prompt": "x"})
    w = wege(monkeypatch, client)["openai_schluessel"]
    assert (w["zustand"], w["grund"]) == ("kontingent", "tagesgrenze") and w["bis"]


def test_dateien_nur_fuer_den_dienst(client, netz):
    client.post("/ai/openai", json={"prompt": "geheimer kundentext"})
    d = Path(os.environ["VERBRAUCH_DIR"])
    for f in d.glob("openai_text*.json"):
        assert oct(f.stat().st_mode & 0o777) == "0o600"
        assert "geheimer" not in f.read_text() and KEY not in f.read_text()
