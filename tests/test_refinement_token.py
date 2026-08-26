"""Das Auswahl-Token haelt der Server, nicht das Modell (Issue #1398).

oneal entfernt das Token aus dem Werkzeugergebnis (`0e3ea84`), damit
es den Modellkontext nie erreicht. Der Browser ergaenzt es nicht — am
Client-Code geprueft. Ein Pflichtfeld dafuer im Modellschema waere
damit ein Vertrag, den niemand erfuellen kann: Beide Aenderungen sind
einzeln richtig, zusammen brechen sie die sprachgesteuerte
Verfeinerung.

Dieselbe Doktrin wie bei Marke und Jahr, ein Satz:
**Was das Modell nicht wissen soll, haelt der Server.**
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr           # noqa: E402
from ai.services import realtime_session_scope as sc  # noqa: E402


@pytest.fixture(autouse=True)
def eigener_speicher(tmp_path, monkeypatch):
    # Der echte Name ist SCOPE_PATH und der Wert ein String, kein Path.
    # Beim ersten Anlauf hatte ich ihn geraten — `monkeypatch` hat das
    # nicht durchgehen lassen und die Tests brachen ab, statt still
    # gegen den PRODUKTIONSSPEICHER zu laufen. Genau dieser Fehler ist
    # mir beim Budgetwaechter schon einmal passiert.
    monkeypatch.setattr(sc, "SCOPE_PATH", str(tmp_path / "scope.json"))
    assert "/var/lib" not in sc.SCOPE_PATH


# ───────────────────────────────── das Schema

def test_das_modell_sieht_kein_token_mehr():
    for t in rr._product_finder_tools("O'Neal"):
        p = t["parameters"]
        assert "selection_token" not in p["properties"], t["name"]
        assert "selection_token" not in (p.get("required") or []), t["name"]


def test_refine_verlangt_gar_nichts_mehr():
    """Ein `required`, das das Modell nicht erfuellen kann, ist
    schlimmer als keins: Es erzeugt erfundene Werte."""
    refine = [t for t in rr._product_finder_tools(None)
              if t["name"] == "refine_search"][0]
    assert not refine["parameters"].get("required")


# ───────────────────────────────── merken und lesen

def test_token_ueberlebt_bis_zur_verfeinerung():
    sc.merken(session_id="s1", brand="O'Neal", collection_year=2027)
    assert sc.token_merken("s1", "st_abc") is True
    assert sc.token_lesen("s1") == "st_abc"


def test_ohne_sitzung_kein_heimatloses_token():
    """Ein Token ohne Scope gehoerte beim naechsten Lesen niemandem."""
    assert sc.token_merken("unbekannt", "st_abc") is False
    assert sc.token_lesen("unbekannt") is None


def test_leeres_token_ist_kein_fehler():
    """Eine Suche ohne Treffer liefert keins — normal, nicht kaputt."""
    sc.merken(session_id="s1", brand=None, collection_year=2027)
    assert sc.token_merken("s1", None) is False
    assert sc.token_merken("s1", "") is False


def test_sitzungen_teilen_ihr_token_nicht():
    """Sonst verfeinerte ein Kunde die Trefferliste eines anderen."""
    sc.merken(session_id="s1", brand="O'Neal", collection_year=2027)
    sc.merken(session_id="s2", brand="O'Neal", collection_year=2027)
    sc.token_merken("s1", "st_eins")
    assert sc.token_lesen("s2") is None


def test_neues_token_ersetzt_das_alte():
    sc.merken(session_id="s1", brand=None, collection_year=2027)
    sc.token_merken("s1", "st_alt")
    sc.token_merken("s1", "st_neu")
    assert sc.token_lesen("s1") == "st_neu"


def test_abgelaufener_scope_gibt_kein_token(monkeypatch):
    sc.merken(session_id="s1", brand=None, collection_year=2027)
    sc.token_merken("s1", "st_abc")
    monkeypatch.setattr(sc.time, "time", lambda: 1e12)
    assert sc.token_lesen("s1") is None
    # und schreiben laesst sich auch nichts mehr
    assert sc.token_merken("s1", "st_neu") is False


def test_marke_bleibt_neben_dem_token_erhalten():
    """Das Token darf den Scope nicht ueberschreiben — sonst liefe
    eine gebundene Sitzung ploetzlich ueber den ganzen Katalog."""
    sc.merken(session_id="s1", brand="O'Neal", collection_year=2027)
    sc.token_merken("s1", "st_abc")
    scope = sc.lesen("s1")
    assert scope["brand"] == "O'Neal"
    assert scope["collection_year"] == 2027


# ───────────────────────── der Handler benutzt es auch

def test_handler_liest_das_token_aus_dem_scope():
    import inspect
    quelle = inspect.getsource(rr._tool_product_search)
    assert "realtime_session_scope.token_lesen(session_id)" in quelle
    assert '(args or {}).get("selection_token")' not in quelle, \
        "Token darf NICHT mehr aus den Modellargumenten kommen"


def test_handler_merkt_das_token_aus_dem_ergebnis():
    import inspect
    quelle = inspect.getsource(rr._tool_product_search)
    assert "token_merken(session_id, token)" in quelle
    assert "__app_command__" in quelle
