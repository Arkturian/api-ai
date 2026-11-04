"""Konfigurator oder Auswahl (Alex ueber OnealServ, 04.11.2026).

`presentation` sagt dem Finder, ob er die Outfit-Figur oder die
Gruppen-/Listenansicht zeigt. Es ist eine Anzeigeabsicht, KEIN
Suchkriterium: oneal hat extra=forbid, ein durchgereichtes Feld gaebe
422, mein Fehlerzweig daraus `unavailable` — und der Kunde hoerte
„Katalog nicht erreichbar".
"""
import asyncio

import pytest

from ai.routes import realtime_routes as rr
from ai.services import realtime_session_scope as sc
from tests.test_product_search_handler import _Antwort, _Client


@pytest.fixture
def an(monkeypatch):
    monkeypatch.setenv(rr.FINDER_PRESENTATION_ENV, "on")


def test_ohne_schalter_weder_feld_noch_regel(monkeypatch):
    """Bis der Finder das Feld vor seinem BFF entfernt, darf das Modell
    es nicht schicken: oneal hat extra=forbid."""
    monkeypatch.delenv(rr.FINDER_PRESENTATION_ENV, raising=False)
    for w in rr._product_finder_tools("O'Neal")[:2]:
        assert "presentation" not in w["parameters"]["properties"]
    t = rr._product_finder_prompt("de", "O'Neal", ui_tools=True)
    assert "presentation" not in t
    assert "KOMPLETTE AUSRUESTUNG" in t and "per_category_limit" in t


@pytest.mark.parametrize("name", ["find_products", "refine_search"])
def test_beide_suchwerkzeuge_fuehren_presentation(an, name):
    t = {w["name"]: w for w in rr._product_finder_tools("O'Neal")}[name]
    feld = t["parameters"]["properties"]["presentation"]
    assert feld["enum"] == ["outfit", "selection"]
    assert "Konfigurator" in feld["description"]
    assert "presentation" not in t["parameters"].get("required", [])


def test_presentation_ist_kein_kriterium(an):
    assert "presentation" not in rr._product_finder_kriterien_felder()


@pytest.fixture
def katalog(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SCOPE_PATH", str(tmp_path / "scope.json"))
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "x" * 32)
    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    _Client.gesehen, _Client.fehler = [], None
    _Client.antwort = _Antwort(200, {"status": "matches", "count": 3})
    sc.merken("vs1", "O'Neal", 2028)


@pytest.mark.parametrize("verfeinern", [False, True])
def test_presentation_erreicht_den_katalog_nie(an, katalog, verfeinern):
    diagnose = {}
    asyncio.run(rr._tool_product_search(
        {"presentation": "outfit", "color": "blau", "per_category_limit": 4},
        "vs1", verfeinern, diagnose))
    n = _Client.gesehen[-1]["json"]
    assert "presentation" not in n["criteria"]
    assert "presentation" not in n
    assert n["criteria"]["color"] == "blau"
    # Kein „erfundenes Feld" im Protokoll — das Modell hat es richtig gesagt.
    assert "presentation" not in diagnose.get("dropped_criteria", [])


def test_persona_unterscheidet_konfigurator_und_auswahl(an):
    t = rr._product_finder_prompt("de", "O'Neal", ui_tools=True)
    assert "`presentation`" in t
    assert "`outfit`" in t and "`selection`" in t
    i = t.index("`selection`: ")
    assert "OHNE `per_category_limit`" in t[i:i + 500]
