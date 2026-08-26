"""Nur Schema-Felder an oneal weiterreichen.

Anlass: Das Werkzeugschema traegt `additionalProperties: false`, aber
ohne `strict: true` ist das fuer OpenAI ein HINWEIS, keine Schranke.
Erfindet das Modell ein Feld, antwortet oneal mit 422, mein
Fehlerzweig macht daraus `unavailable` — und der Kunde hoert „Katalog
gerade nicht erreichbar". Ein erfundenes Wort sperrte den Katalog.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


def test_die_liste_kommt_aus_dem_schema():
    """Keine zweite, handgepflegte Aufzaehlung — die driftet."""
    schema = rr._product_finder_tools(None)[0]["parameters"]
    aus_schema = set(schema["properties"]) - rr.PRODUCT_SERVER_CONTROLLED
    assert rr._product_finder_kriterien_felder() == aus_schema


def test_neues_schemafeld_wird_automatisch_erlaubt(monkeypatch):
    """Wer dem Schema ein Feld hinzufuegt, darf hier nichts nachziehen
    muessen — sonst wird das neue Feld lautlos verworfen."""
    echt = rr._product_finder_tools

    def mit_extra(marke=None):
        t = echt(marke)
        t[0]["parameters"]["properties"]["fit"] = {"type": "string"}
        return t

    monkeypatch.setattr(rr, "_product_finder_tools", mit_extra)
    assert "fit" in rr._product_finder_kriterien_felder()


def test_marke_ist_nie_ein_kriterium():
    """Bei ungebundener Marke FUEHRT das Schema `brand`, damit das
    Modell danach fragen kann — den Wert setzt trotzdem der Scope.
    Liesse ich ihn durch, koennte ein Gespraech die Marke wechseln,
    an der Bindung vorbei."""
    schema = rr._product_finder_tools(None)[0]["parameters"]
    assert "brand" in schema["properties"], "Vorbedingung des Tests"
    assert "brand" not in rr._product_finder_kriterien_felder()


@pytest.mark.parametrize("feld", ["brand", "collection_year", "selection_token"])
def test_serverfelder_bleiben_draussen(feld):
    assert feld not in rr._product_finder_kriterien_felder()


def test_bekannte_kriterien_sind_vollstaendig():
    erlaubt = rr._product_finder_kriterien_felder()
    for f in ("sport", "category", "product_type", "price_max", "size"):
        assert f in erlaubt
