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


# --- Namen ---------------------------------------------------------------

@pytest.mark.parametrize("alt,neu", [
    ("gpt-image-2", "gpt-image-2.6-sunburst"),
    ("gpt-image-1.5", "gpt-image-2.6-sunburst"),
    ("gpt-image-1", "gpt-image-2.6-sunburst"),
    ("gpt-image-1-mini", "gpt-image-2.6-flare"),
    ("gpt-image-2.6-sunburst", "gpt-image-2.6-sunburst"),
    ("gpt-image-2.6-flare", "gpt-image-2.6-flare"),
])
def test_alte_namen_zeigen_auf_den_nachfolger(alt, neu):
    assert g.MODEL_MAPPING[alt] == neu
    assert g.is_openai_image_model(neu)


def test_kein_abgeschaltetes_modell_wird_mehr_angefragt():
    abgeschaltet = {"gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"}
    assert not abgeschaltet & set(g.MODEL_MAPPING.values())


def test_vorgabe_ist_sunburst():
    assert g.ImageGenRequest(prompt="x").model == "gpt-image-2.6-sunburst"
    assert g.OPENAI_BILD_VORGABE == "gpt-image-2.6-sunburst"


def test_modellliste_nennt_nachfolger_und_alias():
    liste = asyncio.run(g.list_image_models())["models"]
    ids = {m["id"]: m for m in liste}
    assert "gpt-image-2.6-sunburst" in ids and "gpt-image-2.6-flare" in ids
    assert ids["gpt-image-2"]["alias_of"] == "gpt-image-2.6-sunburst"
    assert "gpt-image-1" not in ids


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


def test_freie_groesse_nur_fuer_2_6():
    assert not g.freie_groesse_zulaessig("dall-e-3", 1920, 1088)


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


def test_ohne_token_gilt_der_satz_je_bild():
    t = oct_mod.OpenAICostTracker()
    usd, _ = t._cost_for_call("gpt-image-2.6-sunburst", 2)
    assert usd == pytest.approx(0.35)
