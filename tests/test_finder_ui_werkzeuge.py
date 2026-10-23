"""Productfinder Stufe 2 (Post #5634, OnealServ 23.10.2026): vier reine
Oberflaechenwerkzeuge, die der Finder-Browser selbst ausfuehrt. Opt-in per
`finder_ui_tools`; ohne Flag bleibt die Sitzung wie in Stufe 1 (erklaeren,
nicht bedienen). Dazu die feste Anrede „du"."""
import asyncio
import inspect

import pytest
from fastapi import HTTPException

from ai.routes import realtime_routes as rr

UI_NAMEN = ["set_grouping", "toggle_favorite", "open_my_selection",
            "show_outfit_image"]


def test_werkzeuge_sind_klein_und_ohne_produkt_ids():
    defs = rr._product_finder_ui_tools()
    assert [d["name"] for d in defs] == UI_NAMEN
    gruppe = defs[0]["parameters"]
    assert gruppe["required"] == ["dimension"]
    assert gruppe["properties"]["dimension"]["enum"] == [
        "product_type", "product_line", "color", "price", "model_year",
        "availability", "size"]
    for d in defs[1:]:
        assert d["parameters"]["properties"] == {}
    assert set(UI_NAMEN) == rr.FINDER_UI_TOOL_NAMES


def test_client_werkzeuge_sind_keine_serverwerkzeuge():
    for name in UI_NAMEN:
        assert name not in rr.READ_TOOL_NAMES
        assert name not in rr.PRODUCT_TOOL_NAMES


def test_flag_ist_opt_in():
    assert rr.RealtimeTokenRequest(session_id="s").finder_ui_tools is False


def test_mint_zweig_haengt_werkzeuge_nur_mit_flag_an():
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert "if request.finder_ui_tools:" in quelle
    assert "_product_finder_ui_tools()" in quelle
    assert "ui_tools=request.finder_ui_tools" in quelle


@pytest.mark.parametrize("name", UI_NAMEN)
def test_realtime_tool_weist_client_werkzeug_laut_ab(name):
    with pytest.raises(HTTPException) as e:
        asyncio.run(rr.realtime_tool_call(
            tool_name=name, body=rr.RealtimeToolCall(arguments={}),
            x_session_id="s", authorization=None))
    assert e.value.status_code == 400
    assert e.value.detail["error"] == "client_tool"


def test_prompt_ohne_flag_erklaert_nur():
    text = rr._product_finder_prompt("de")
    assert "Behaupte nie, du haettest selbst etwas angeklickt" in text
    assert "Vier Dinge kannst du" not in text


def test_prompt_mit_flag_bedient_ehrlich_und_ohne_werkzeugnamen():
    text = rr._product_finder_prompt("de", ui_tools=True)
    assert "Vier Dinge kannst du" in text
    assert "NUR aus der Antwort" in text
    assert "NUR auf ausdruecklichen" in text
    assert "Behaupte nie, du haettest selbst etwas angeklickt" not in text
    for name in UI_NAMEN:
        assert name not in text


@pytest.mark.parametrize("ui", [False, True])
def test_anrede_ist_du_ohne_sie_beispiele(ui):
    """OnealServ-Test 23.10.: Sie und du wechselten in einer Sitzung.
    Ein einziger Beispielsatz in Sie reicht, damit das Modell kippt."""
    text = rr._product_finder_prompt("de", ui_tools=ui)
    assert "ANREDE" in text and "Du duzt den Kunden" in text
    for sie in ("Ihnen", "Oeffnen Sie", "Schauen Sie", "meinen Sie",
                "Ihr Warenkorb"):
        assert sie not in text, sie
