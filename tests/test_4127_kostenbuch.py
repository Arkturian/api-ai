"""#4127: Kostenbuch je Aufruf und GET /ai/kosten.

Alex, 26.11.2026: „Ich brauche volle Kostentransparenz“. Vorher gab es nur
Monatszaehler ohne Aufrufer; STT-Minuten zaehlte niemand.
"""
import json
import types
from datetime import date

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import audio_ai_routes as ar
from ai.routes import verbrauch_routes as vr
from ai.services import bild_verbrauch as bv
from ai.services import kostenbuch as kb

SCHLUESSEL = "betreiber-test-schluessel"


@pytest.fixture(autouse=True)
def ablage(monkeypatch, tmp_path):
    monkeypatch.setenv("KOSTENBUCH_DIR", str(tmp_path / "kostenbuch"))
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path / "aufrufer"))
    monkeypatch.setenv("VERBRAUCH_BASIS_DIR", str(tmp_path))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "verbrauch"))
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", SCHLUESSEL)
    monkeypatch.delenv("API_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_BILLING_TOR", raising=False)
    return tmp_path


def _buch(tmp_path):
    out = []
    for p in sorted((tmp_path / "kostenbuch").glob("*.jsonl")):
        out += [json.loads(z) for z in p.read_text().splitlines()]
    return out


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


# --- Einordnung --------------------------------------------------------------

@pytest.mark.parametrize("pfad,aufrufer,erwartet", [
    ("/ai/verarbeitung", "service:issue-api", "prompt_agent"),
    ("/ai/deepseek", "service:issue-api", "prompt_agent"),
    ("/ai/verarbeitung", "AiApi", "verarbeitung"),
    ("/ai/kette", "service:project-api", "verarbeitung"),
    ("/ai/gemini", "service:conversation-api", "conversation"),
    ("/ai/transcribe", "service:conversation-api", "conversation"),
    ("/ai/jev", "agent:cloud-api-jev", "kleinhirn"),
    ("/ai/gemini", "Cloud", "kleinhirn"),
    ("/ai/transcribe", "service:comm-api", "stt"),
    ("/ai/tts/minimax", "Assistant", "tts"),
    ("/ai/genvideo/flow/status/abc", "DaVinci", "bild_video"),
    ("/ai/decisions", "service:storage-api", "jev_decisions"),
    ("/ai/realtime/usage", "agent:productfinder-internal-demo", "realtime"),
    ("/ai/chatgpt", "ANONYM", "text_direkt"),
    ("/ai/irgendwas", None, "rest"),
])
def test_weg(pfad, aufrufer, erwartet):
    assert kb.weg(pfad, aufrufer) == erwartet


# --- Buchen mit dem Kontext der Middleware ------------------------------------

def test_middleware_gibt_pfad_und_aufrufer_an_die_buchung(client, ablage):
    async def probe():
        kb.buchen(anbieter="test", art="geld", usd=0.5)
        return {"ok": True}
    main.app.router.add_api_route("/ai/_probe_4127", probe, methods=["POST"])
    try:
        r = client.post("/ai/_probe_4127", headers={"X-Agent-Name": "service:comm-api"})
    finally:
        main.app.router.routes[:] = [x for x in main.app.router.routes if getattr(x, "path", "") != "/ai/_probe_4127"]
    assert r.status_code == 200
    (z,) = _buch(ablage)
    assert (z["pfad"], z["aufrufer"], z["quelle"], z["usd"]) == ("/ai/_probe_4127", "service:comm-api", "header", 0.5)


def test_ausserhalb_eines_aufrufs_unbekannt(ablage):
    kb.buchen(anbieter="x", art="abo", tokens_ein=3)
    (z,) = _buch(ablage)
    assert z["aufrufer"] == "unbekannt" and z["pfad"] is None and z["art"] == "abo"


def test_buchen_wirft_nie(monkeypatch, ablage):
    monkeypatch.setenv("KOSTENBUCH_DIR", "/proc/nicht/schreibbar")
    kb.buchen(anbieter="x", art="geld", usd=1)          # kein Fehler


# --- STT: Minuten werden endlich gezaehlt ---------------------------------------

@pytest.fixture
def whisper(monkeypatch):
    class _Tr:
        async def create(self, **kw):
            return types.SimpleNamespace(text="hallo", language="german", duration=90.0)

    class _Client:
        def __init__(self, **kw):
            self.audio = types.SimpleNamespace(transcriptions=_Tr())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ar, "AsyncOpenAI", _Client)
    monkeypatch.setattr(ar, "_normalize_audio_for_openai", lambda data, base: (data, ".mp3"))
    monkeypatch.setattr(ar, "_audio_duration_seconds", lambda data: 120.0)


def test_stt_minuten_und_geld_im_kostenbuch(client, whisper, ablage):
    r = client.post("/ai/transcribe", data={"model": "gpt-4o-mini-transcribe"},
                    headers={"X-Agent-Name": "service:comm-api"},
                    files={"file": ("a.mp3", b"ID3-nicht-leer", "audio/mpeg")})
    assert r.status_code == 200, r.text
    (z,) = _buch(ablage)
    assert z["anbieter"] == "openai_stt" and z["art"] == "geld" and z["minuten"] == 2.0
    assert z["usd"] == pytest.approx(0.006) and z["aufrufer"] == "service:comm-api"
    assert z["genauigkeit"] == "gezaehlt"


def test_stt_ohne_eigene_dauer_nimmt_die_des_anbieters(client, whisper, monkeypatch, ablage):
    monkeypatch.setattr(ar, "_audio_duration_seconds", lambda data: None)
    r = client.post("/ai/transcribe", files={"file": ("a.mp3", b"ID3-nicht-leer", "audio/mpeg")})
    assert r.status_code == 200
    (z,) = _buch(ablage)
    assert z["minuten"] == 1.5 and z["usd"] == pytest.approx(0.009)


# --- Bilder und Flow --------------------------------------------------------------

def test_bild_toepfe_getrennt(ablage):
    bv.buchen(agent="DaVinci", quelle="jwt", art="video", model="veo-3.1-fast", routed_via="google-flow",
              kosten_usd=0.0)
    bv.buchen(agent="3DTischler", quelle="jwt", model="codex-imagegen", routed_via="subscription", kosten_usd=None)
    bv.buchen(agent="Design", quelle="jwt", model="gpt-image-2.6-sunburst", routed_via="api", kosten_usd=0.08)
    bv.buchen(agent="Design", quelle="jwt", model="minimax-image-01", routed_via="api", kosten_usd=0.01)
    z = {(x["anbieter"], x["art"]): x for x in _buch(ablage)}
    assert z[("google_flow", "abo")]["credits"] == 10 and z[("google_flow", "abo")]["aufrufer"] == "DaVinci"
    assert z[("capsolver", "geld")]["usd"] == 0.003 and z[("capsolver", "geld")]["genauigkeit"] == "geschaetzt"
    assert z[("codex", "abo")]["usd"] == 0
    assert z[("openai_bild", "geld")]["usd"] == 0.08
    # MiniMax bucht sein eigener Zaehler, hier nicht doppelt
    assert len(z) == 4


# --- Bericht ------------------------------------------------------------------------

def _schreiben(pfad, zeilen):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text("".join(json.dumps(z) + "\n" for z in zeilen))


@pytest.fixture
def daten(ablage):
    _schreiben(ablage / "aufrufer" / "2026-11-25.jsonl", [
        {"pfad": "/ai/transcribe", "caller": "service:comm-api", "quelle": "header", "status": 200},
        {"pfad": "/ai/transcribe", "caller": "service:comm-api", "quelle": "header", "status": 502},
        {"pfad": "/ai/chatgpt", "caller": "127.0.0.1", "quelle": "ip", "status": 200},
        {"pfad": "/ai/verarbeitung", "caller": "service:issue-api", "quelle": "jwt", "status": 200},
        {"pfad": "/ai/gemini/gcp-budget-webhook", "caller": "34.98.140.64", "quelle": "ip", "status": 200},
    ])
    _schreiben(ablage / "kostenbuch" / "2026-11-25.jsonl", [
        {"pfad": "/ai/transcribe", "aufrufer": "service:comm-api", "quelle": "header", "anbieter": "openai_stt",
         "art": "geld", "minuten": 2.0, "usd": 0.012, "genauigkeit": "gezaehlt"},
        {"pfad": "/ai/chatgpt", "aufrufer": "127.0.0.1", "quelle": "ip", "anbieter": "codex", "art": "abo",
         "tokens_ein": 1000, "tokens_aus": 10, "gegenwert_usd": 0.5},
        {"pfad": "/ai/verarbeitung", "aufrufer": "service:issue-api", "quelle": "jwt", "anbieter": "agy",
         "art": "abo", "tokens_ein": 200, "tokens_aus": 30},
    ])
    _schreiben(ablage / "deepseek_usage_2026-11.json", [])
    (ablage / "deepseek_usage_2026-11.json").write_text(json.dumps({"total_cost_usd": 0.03, "request_count": 30}))


def _z(b, feld="weg"):
    return {(x[feld], x["anbieter"], x["art"]): x for x in b["zeilen"]}


def test_bericht_nach_weg(daten):
    b = kb.bericht(date(2026, 11, 25), date(2026, 11, 25), "weg")
    z = _z(b)
    stt = z[("stt", "openai_stt", "geld")]
    assert stt["minuten"] == 2.0 and stt["betrag"] == 0.012 and stt["waehrung"] == "USD"
    a = {x["weg"]: x for x in b["aufrufe"]}
    assert a["stt"]["aufrufe"] == 2 and a["stt"]["aufrufe_ok"] == 1
    # Abo: nur Mengen, kein Betrag; Gegenwert getrennt
    pa = z[("prompt_agent", "agy", "abo")]
    assert pa["tokens_ein"] == 200 and "betrag" not in pa
    td = z[("text_direkt", "codex", "abo")]
    assert "betrag" not in td and td["gegenwert_usd"] == 0.5
    assert "rest" not in a                  # Google-Webhook ist keine Last
    assert b["summe"]["betrag"] == 0.012 and b["summe"]["kurs_usd_je_eur"] == 1.05
    assert b["toepfe"] == [{"anbieter": "openai_stt", "betrag": 0.012, "waehrung": "USD"}]
    assert b["erfasst_bis"] is None or isinstance(b["erfasst_bis"], str)
    m = b["monatszaehler"][0]
    assert m["anbieter"] == "deepseek" and m["betrag"] == 0.03 and m["waehrung"] == "USD"


def test_bericht_tagesreihe(daten):
    b = kb.bericht(date(2026, 11, 24), date(2026, 11, 25), "weg", "tag")
    stt = _z(b)[("stt", "openai_stt", "geld")]
    assert stt["reihe"] == [{"tag": "2026-11-25", **{k: stt["reihe"][0][k] for k in stt["reihe"][0] if k != "tag"}}]
    assert stt["reihe"][0]["betrag"] == 0.012
    abo = _z(b)[("prompt_agent", "agy", "abo")]
    assert "betrag" not in abo["reihe"][0]
    a = {x["weg"]: x for x in b["aufrufe"]}
    assert a["stt"]["reihe"] == [{"tag": "2026-11-25", "aufrufe": 2, "aufrufe_ok": 1}]


def test_bericht_anonym_nicht_verteilt(daten):
    b = kb.bericht(date(2026, 11, 25), date(2026, 11, 25), "aufrufer")
    z = _z(b, "aufrufer")
    assert z[("ANONYM", "codex", "abo")]["gegenwert_usd"] == 0.5
    a = {x["aufrufer"]: x for x in b["aufrufe"]}
    assert a["ANONYM"]["aufrufe"] == 1 and "127.0.0.1" not in a
    l = {x["art"]: x for x in b["luecken"]}
    assert l["ohne_kennung"]["anteil_aufrufe"] == 0.25


def test_bericht_nach_anbieter(daten):
    b = kb.bericht(date(2026, 11, 25), date(2026, 11, 25), "anbieter")
    assert {(x["anbieter"], x["art"]) for x in b["zeilen"]} == {("openai_stt", "geld"), ("codex", "abo"), ("agy", "abo")}
    assert b["aufrufe"] == []


def test_deckel_aus_der_umgebung(monkeypatch):
    monkeypatch.setenv("JEV_DAILY_MAX_USD", "3")
    assert kb.deckel("typesafe_jev") == {"wert": 3.0, "einheit": "USD", "zeitraum": "tag"}
    monkeypatch.delenv("OPENAI_TEXT_DAILY_MAX_USD", raising=False)
    assert kb.deckel("openai_text") is None and kb.deckel("codex") is None


def test_bericht_nennt_luecke_vor_dem_kostenbuch(daten):
    b = kb.bericht(date(2026, 11, 20), date(2026, 11, 25), "tag")
    arten = [x["art"] for x in b["luecken"]]
    assert "vor_kostenbuch" in arten and b["kostenbuch_seit"] == "2026-11-25"


@pytest.mark.parametrize("von,bis,gruppe", [
    (date(2026, 11, 26), date(2026, 11, 25), "weg"),
    (date(2025, 1, 1), date(2026, 11, 25), "weg"),
    (date(2026, 11, 25), date(2026, 11, 25), "modell"),
])
def test_bericht_grenzen(von, bis, gruppe):
    with pytest.raises(ValueError):
        kb.bericht(von, bis, gruppe)


def test_bericht_takt_nur_tag():
    with pytest.raises(ValueError):
        kb.bericht(date(2026, 11, 25), date(2026, 11, 25), "weg", "stunde")


# --- Zugang zu GET /ai/kosten ---------------------------------------------------------

def test_kosten_ohne_anmeldung_401(client, daten):
    assert client.get("/ai/kosten").status_code == 401


def test_kosten_mit_betreiberschluessel(client, daten):
    r = client.get("/ai/kosten?von=2026-11-25&bis=2026-11-25", headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200 and r.json()["summe"]["betrag"] == 0.012
    assert r.headers["cache-control"] == "no-store"


def test_kosten_falscher_schluessel_401(client, daten):
    assert client.get("/ai/kosten", headers={"X-API-KEY": "falsch"}).status_code == 401


def test_kosten_mit_recht_im_jwt(client, daten, monkeypatch):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "pruefen", lambda a: {"sub": "agent:Cloud", "permissions": [vr.RECHT_KOSTEN]})

    async def kein_profil(a):
        raise AssertionError("mit Recht kein /me noetig")
    monkeypatch.setattr(vr, "_profil", kein_profil)
    r = client.get("/ai/kosten?gruppe=aufrufer&von=2026-11-25&bis=2026-11-25",
                   headers={"Authorization": "Bearer x"})
    assert r.status_code == 200 and r.json()["gruppe"] == "aufrufer"


def test_kosten_jwt_ohne_recht_und_kein_admin_403(client, daten, monkeypatch):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "pruefen", lambda a: {"sub": "agent:Story", "permissions": ["ai:chatgpt"]})

    async def profil(a):
        return {"agentos_role": "user"}
    monkeypatch.setattr(vr, "_profil", profil)
    r = client.get("/ai/kosten", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403 and r.json()["detail"]["required"] == "ai:kosten:lesen"


def test_kosten_admin_mensch(client, daten, monkeypatch):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "pruefen", lambda a: {"sub": "17"})

    async def profil(a):
        return {"is_superadmin": True}
    monkeypatch.setattr(vr, "_profil", profil)
    assert client.get("/ai/kosten", headers={"Authorization": "Bearer x"}).status_code == 200


def test_health_nennt_die_faehigkeit():
    assert "kosten" in main.health_check()["faehigkeiten"]


def test_kosten_mit_tagesreihe_ueber_http(client, daten):
    r = client.get("/ai/kosten?von=2026-11-25&bis=2026-11-25&takt=tag", headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200 and r.json()["takt"] == "tag" and "reihe" in r.json()["zeilen"][0]


def test_kosten_ungueltige_gruppe_422(client, daten):
    r = client.get("/ai/kosten?gruppe=modell", headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 422
