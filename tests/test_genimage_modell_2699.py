"""OpenAI schaltet die alten Bildmodelle ab (#2699, gelesen 11.11.2026):
gpt-image-1 seit 23.10.2026, gpt-image-1.5 und -1-mini am 01.12.2026,
gpt-image-2 am 10.02.2027. Vorgabe ist gpt-image-2.6-sunburst, die alten
Namen bleiben als Alias, und die Antwort nennt das echte Modell."""
import asyncio
import base64
import types

import pytest

from ai.routes import image_ai_routes as g
from ai.services import openai_cost_tracker as oct_mod


# --- Namen: automatisch das aktuelle Modell (Alex, 11.11.2026) ---------

from ai.services import openai_bildmodell_katalog as kat

# Auszug aus GET /v1/models, gelesen 11.11.2026
KATALOG_11_11 = [
    {"id": "chatgpt-image-latest", "created": 1765925279, "shutdown_date": "2026-12-01"},
    {"id": "gpt-image-1-mini", "created": 1758845821, "shutdown_date": "2026-12-01"},
    {"id": "gpt-image-1.5", "created": 1764030620, "shutdown_date": "2026-12-01"},
    {"id": "gpt-image-2", "created": 1776399795, "shutdown_date": "2027-02-10"},
    {"id": "gpt-image-2-2026-04-21", "created": 1776399994, "shutdown_date": "2027-02-10"},
    {"id": "gpt-image-2.5-flare", "created": 1788563147, "shutdown_date": None},
    {"id": "gpt-image-2.5-sunburst", "created": 1788563162, "shutdown_date": None},
    {"id": "gpt-image-2.5-sunburst-2026-09-08", "created": 1788851010, "shutdown_date": None},
    {"id": "gpt-image-2.6-flare", "created": 1793222647, "shutdown_date": None},
    {"id": "gpt-image-2.6-sunburst", "created": 1793222670, "shutdown_date": None},
]


@pytest.fixture
def katalog(monkeypatch):
    def setzen(wahl):
        monkeypatch.setattr(kat, "_stand", dict(wahl))
        monkeypatch.setattr(kat, "_gelesen_um", __import__("time").time())  # gilt als frisch
    return setzen


def test_waehlt_heute_2_6():
    import datetime as dt
    assert kat.waehlen(KATALOG_11_11, dt.date(2026, 11, 11)) == {
        "qualitaet": "gpt-image-2.6-sunburst", "schnell": "gpt-image-2.6-flare"}


def test_neue_version_wird_ohne_codeaenderung_genommen():
    import datetime as dt
    neu = KATALOG_11_11 + [
        {"id": "gpt-image-2.10-sunburst", "created": 1800000000, "shutdown_date": None},
        {"id": "gpt-image-2.10-sunburst-2027-01-01", "created": 1800000001, "shutdown_date": None},
    ]
    w = kat.waehlen(neu, dt.date(2026, 11, 11))
    assert w["qualitaet"] == "gpt-image-2.10-sunburst"   # 2.10 > 2.6, nicht alphabetisch
    assert w["schnell"] == "gpt-image-2.6-flare"


def test_modell_kurz_vor_abschaltung_faellt_raus():
    import datetime as dt
    alt = [m for m in KATALOG_11_11 if not m["id"].startswith("gpt-image-2.")]
    w = kat.waehlen(alt, dt.date(2027, 1, 20))   # gpt-image-2 endet in 21 Tagen
    assert w == {}


def test_fehlende_stufe_nimmt_die_andere():
    import datetime as dt
    nur = [{"id": "gpt-image-3", "created": 1, "shutdown_date": None}]
    assert kat.waehlen(nur, dt.date(2026, 11, 11)) == {
        "qualitaet": "gpt-image-3", "schnell": "gpt-image-3"}


@pytest.mark.parametrize("name,stufe", [
    ("gpt-image", "qualitaet"), ("gpt-image-2", "qualitaet"),
    ("gpt-image-1.5", "qualitaet"), ("gpt-image-1", "qualitaet"),
    ("gpt-image-2.6-sunburst", "qualitaet"), ("gpt-image-1-mini", "schnell"),
    ("gpt-image-2.6-flare", "schnell"), ("gpt-image-fast", "schnell"),
])
def test_jeder_name_landet_beim_aktuellen_modell_seiner_stufe(katalog, name, stufe):
    katalog({"qualitaet": "gpt-image-9-sunburst", "schnell": "gpt-image-9-flare"})
    erwartet = {"qualitaet": "gpt-image-9-sunburst", "schnell": "gpt-image-9-flare"}[stufe]
    assert kat.aufloesen(name) == erwartet
    assert g.is_openai_image_model(name)


def test_andere_namen_bleiben(katalog):
    katalog({"qualitaet": "gpt-image-9-sunburst", "schnell": "gpt-image-9-flare"})
    assert kat.aufloesen("dall-e-3") == "dall-e-3"
    assert kat.aufloesen("codex-imagegen") == "codex-imagegen"
    assert g.MODEL_MAPPING["gpt-image-2-abo"] == "codex-imagegen"


def test_ohne_katalog_gilt_der_rueckfall(monkeypatch):
    monkeypatch.setattr(kat, "_stand", {})
    assert kat.aufloesen("gpt-image-2") == "gpt-image-2.6-sunburst"
    assert kat.aufloesen("gpt-image-1-mini") == "gpt-image-2.6-flare"


def test_katalogfehler_verhindert_kein_bild(monkeypatch):
    import httpx

    class _Kaputt:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("weg")

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(httpx, "AsyncClient", _Kaputt)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(kat, "_stand", {})
    monkeypatch.setattr(kat, "_letzter_versuch", 0.0)
    asyncio.run(kat.aktualisieren())
    assert kat.aktuell("qualitaet") == kat.RUECKFALL["qualitaet"]


def test_vorgabe_ist_das_aktuelle_modell():
    assert g.ImageGenRequest(prompt="x").model == "gpt-image"
    assert not any(k.startswith("gpt-image-") and v.startswith("gpt-image")
                   for k, v in g.MODEL_MAPPING.items()), "keine festen Versionen mehr"


def test_modellliste_zeigt_aktuelle_aufloesung(katalog):
    katalog({"qualitaet": "gpt-image-9-sunburst", "schnell": "gpt-image-9-flare"})
    antwort = asyncio.run(g.list_image_models())
    ids = {m["id"]: m for m in antwort["models"]}
    assert ids["gpt-image"]["resolves_to"] == "gpt-image-9-sunburst"
    assert ids["gpt-image-fast"]["resolves_to"] == "gpt-image-9-flare"
    assert antwort["openai_catalog"]["quelle"] == "openai_models"


# --- Groessen ------------------------------------------------------------

@pytest.mark.parametrize("w,h,ok", [
    (3840, 2160, True), (2160, 3840, True), (2048, 2048, True),
    (1024, 1024, True), (2560, 1440, True), (1920, 1088, True),
    (3840, 3840, False),          # ueber dem Pixelbudget (API: "pixel budget")
    (1000, 1000, False),          # keine Vielfachen von 16
    (3840, 1024, False),          # Seitenverhaeltnis ueber 3:1
    (640, 640, False),            # unter 655 360 Pixeln
    (4096, 1536, False),          # Kante ueber 3840
])
def test_freie_groesse_2_6(w, h, ok):
    assert g.freie_groesse_zulaessig("gpt-image-2.6-sunburst", w, h) is ok
    assert g.freie_groesse_zulaessig("gpt-image-2.6-flare", w, h) is ok


def test_freie_groesse_ab_2_6_auch_fuer_kuenftige():
    assert not g.freie_groesse_zulaessig("dall-e-3", 1920, 1088)
    assert not g.freie_groesse_zulaessig("gpt-image-2", 1920, 1088)
    assert g.freie_groesse_zulaessig("gpt-image-2.10-sunburst", 1920, 1088)
    assert g.freie_groesse_zulaessig("gpt-image-3", 1920, 1088)


# --- Aufruf an OpenAI ----------------------------------------------------

class _Antwort:
    status_code = 200

    def json(self):
        return {"created": 1, "data": [{"b64_json": base64.b64encode(b"png").decode()}],
                "usage": {"input_tokens": 100, "output_tokens": 4000,
                          "input_tokens_details": {"text_tokens": 100, "image_tokens": 0}}}


def _openai_abfangen(monkeypatch):
    gesendet = {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **k):
            gesendet["url"], gesendet["json"] = url, json
            return _Antwort()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    async def _speichern(**kw):
        return types.SimpleNamespace(id=7, file_url="https://s/7")
    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "save_file_and_record", _speichern)

    getrackt = []
    monkeypatch.setattr(oct_mod.openai_cost_tracker, "track_image",
                        lambda **kw: getrackt.append(kw))
    return gesendet, getrackt


def test_4k_transparent_geht_unveraendert_an_2_6(monkeypatch):
    gesendet, getrackt = _openai_abfangen(monkeypatch)
    asyncio.run(g.generate_with_openai_image(
        prompt="Logo", model="gpt-image-2.6-sunburst", collection_id="c", link_id=None,
        width=3840, height=2160,
        background="transparent"))
    j = gesendet["json"]
    assert j["model"] == "gpt-image-2.6-sunburst"
    assert j["size"] == "3840x2160"
    assert j["background"] == "transparent" and j["output_format"] == "png"
    # Kosten aus den Token: 100 * 5 $/M + 4000 * 30 $/M = 0,1205 $
    assert getrackt[0]["cost_usd"] == pytest.approx(0.1205)


def test_freie_groesse_wird_nicht_auf_ein_preset_gerundet(monkeypatch):
    gesendet, _ = _openai_abfangen(monkeypatch)
    asyncio.run(g.generate_with_openai_image(
        prompt="x", model="gpt-image-2.6-flare", collection_id="c", link_id=None,
        width=2560, height=1440))
    assert gesendet["json"]["size"] == "2560x1440"


# --- Kosten --------------------------------------------------------------

def test_kosten_aus_usage():
    u = {"input_tokens": 1200, "output_tokens": 10000,
         "input_tokens_details": {"text_tokens": 200, "image_tokens": 1000}}
    # 200*5 + 1000*8 + 10000*30 = 309 000 / 1e6
    assert oct_mod.kosten_aus_usage("gpt-image-2.6-flare", u) == pytest.approx(0.309)


def test_kosten_ohne_usage_oder_fremdes_modell_sind_none():
    assert oct_mod.kosten_aus_usage("gpt-image-2.6-sunburst", None) is None
    assert oct_mod.kosten_aus_usage("dall-e-3", {"output_tokens": 5}) is None
    assert oct_mod.kosten_aus_usage("gpt-image-2.6-sunburst", {"output_tokens": 0}) is None


def test_kosten_sind_gedeckelt_und_nie_negativ():
    riesig = {"input_tokens": 0, "output_tokens": 10**12}
    assert oct_mod.kosten_aus_usage("gpt-image-2.6-sunburst", riesig) == oct_mod._MAX_KOSTEN_JE_BILD_USD
    t = oct_mod.OpenAICostTracker()
    assert t._cost_for_call("gpt-image-2.6-sunburst", 1, -3.0)[0] == 0.0


def test_kuenftiges_modell_rechnet_mit_dem_neuesten_tokenpreis():
    u = {"input_tokens": 100, "output_tokens": 4000}
    assert oct_mod.kosten_aus_usage("gpt-image-2.10-sunburst", u) == pytest.approx(0.1205)
    assert oct_mod.kosten_aus_usage("gpt-image-2", u) is None   # aelter: kein Tokenpreis


def test_ohne_token_gilt_der_satz_je_bild():
    t = oct_mod.OpenAICostTracker()
    usd, _ = t._cost_for_call("gpt-image-2.6-sunburst", 2)
    assert usd == pytest.approx(0.35)


# --- Ende zu Ende: der Endpunkt fragt das aktuelle Modell an -------------

@pytest.mark.parametrize("name,erwartet", [
    ("gpt-image-2", "gpt-image-9-sunburst"),
    ("gpt-image-1.5", "gpt-image-9-sunburst"),
    ("gpt-image-2.6-sunburst", "gpt-image-9-sunburst"),
    (None, "gpt-image-9-sunburst"),           # Vorgabe
    ("gpt-image-1-mini", "gpt-image-9-flare"),
])
def test_endpunkt_schickt_das_aktuelle_modell_an_openai(monkeypatch, katalog, name, erwartet):
    katalog({"qualitaet": "gpt-image-9-sunburst", "schnell": "gpt-image-9-flare"})
    gesendet, _ = _openai_abfangen(monkeypatch)
    rumpf = {"prompt": "x", "route": "api", "confirm_api_billing": True}
    if name:
        rumpf["model"] = name
    antwort = asyncio.run(g.generate_image_endpoint(g.ImageGenRequest(**rumpf), api_key="k"))
    assert gesendet["json"]["model"] == erwartet
    inhalt = antwort if isinstance(antwort, dict) else __import__("json").loads(antwort.body)
    assert inhalt["actual_model"] == erwartet
