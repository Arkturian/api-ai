"""Kategorien: feste Werte statt freier Woerter.

Anlass (Alex' erster echter Kundendialog, 2026-08-27 00:02): Das
Modell fuellte `category` mit dem gehoerten Wort — „Bekleidung". oneal
verlangt kanonische Slugs, antwortete 422, mein Fehlerzweig machte
daraus `unavailable`, und der Sprecher sagte „Katalog nicht
erreichbar". Ein deutsches Substantiv legte den Katalog lahm.

Der Feldfilter von vorhin greift hier NICHT: `category` IST ein
bekanntes Feld. Die Luecke lag eine Ebene tiefer — beim WERT.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


@pytest.fixture(autouse=True)
def cache_leer(monkeypatch):
    rr._kategorien_cache["werte"] = None
    rr._kategorien_cache["geholt"] = 0.0
    monkeypatch.delenv(rr.ONEAL_API_KEY_ENV, raising=False)
    yield
    rr._kategorien_cache["werte"] = None
    rr._kategorien_cache["geholt"] = 0.0


# ───────────────────────────────── das Schema kennt feste Werte

def test_category_hat_ein_enum():
    for t in rr._product_finder_tools("O'Neal"):
        cat = t["parameters"]["properties"]["category"]
        assert cat["items"].get("enum"), t["name"]


def test_das_gehoerte_wort_steht_nicht_drin():
    """Der konkrete Fall aus dem Gespraech."""
    enum = rr._product_finder_tools(None)[0]["parameters"]["properties"]["category"]["items"]["enum"]
    for wort in ("Bekleidung", "Schutz", "Helme", "bekleidung"):
        assert wort not in enum


def test_echte_slugs_sind_drin():
    enum = rr._oneal_kategorien()
    for slug in ("helmets-mx", "jerseys-offroad", "gloves", "goggles"):
        assert slug in enum


# ─────────────────────── Innenleben gehoert nicht ins Gespraech

def test_ersatzteile_und_buchhaltung_bleiben_draussen():
    """`z-spare-parts-helmets` ist mit 1357 Eintraegen die GROESSTE
    Kategorie. Gaebe ich sie dem Modell, schluege eine Frage nach
    Helmen zuerst Ersatzvisiere vor."""
    enum = rr._oneal_kategorien()
    for slug in ("z-spare-parts-helmets", "z-merchandise", "displays",
                 "old-brands", "revenue-without-material-usage"):
        assert slug not in enum


@pytest.mark.parametrize("slug,erwartet", [
    ("helmets-mx", True), ("gloves", True),
    ("z-spare-parts-boots", False), ("displays", False),
    ("", False),
])
def test_kundenkategorie_filter(slug, erwartet):
    assert rr._ist_kundenkategorie(slug) is erwartet


# ───────────────────────────── faellt nie aus, cacht aber

def test_ohne_erreichbaren_katalog_kommt_die_rueckfallliste():
    """Ein Mint darf nicht daran scheitern, dass ein Katalogdienst
    langsam ist — das Werkzeug bliebe ohne Kategorien und der Kunde
    hoerte denselben Satz wie beim echten Fehler."""
    assert rr._oneal_kategorien() == list(rr.ONEAL_KATEGORIEN_RUECKFALL)


def test_leere_antwort_wird_nicht_uebernommen(monkeypatch):
    """Ein leeres Regal ist kein geraeumter Laden.

    Uebernaehme ich eine leere Liste, verschwaenden alle Kategorien aus
    dem Werkzeug — und zwar gecacht fuer eine Stunde.
    """
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return []

    monkeypatch.setattr(rr.httpx, "get", lambda *a, **k: _A())
    assert rr._oneal_kategorien() == list(rr.ONEAL_KATEGORIEN_RUECKFALL)


def test_fehler_beim_abruf_ist_kein_absturz(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    def _kaputt(*a, **k):
        raise RuntimeError("Netz weg")

    monkeypatch.setattr(rr.httpx, "get", _kaputt)
    assert rr._oneal_kategorien() == list(rr.ONEAL_KATEGORIEN_RUECKFALL)


def test_lebende_liste_gewinnt_und_wird_gefiltert(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return ["helmets-mx", "z-spare-parts-helmets", "neue-kategorie"]

    monkeypatch.setattr(rr.httpx, "get", lambda *a, **k: _A())
    werte = rr._oneal_kategorien()
    assert werte == ["helmets-mx", "neue-kategorie"]
    assert "z-spare-parts-helmets" not in werte


def test_zweiter_aufruf_holt_nicht_nochmal(monkeypatch):
    """Sonst haenge ich an jedem Mint einen Fremdaufruf an."""
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")
    rufe = []

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return ["gloves"]

    def _zaehl(*a, **k):
        rufe.append(1)
        return _A()

    monkeypatch.setattr(rr.httpx, "get", _zaehl)
    rr._oneal_kategorien()
    rr._oneal_kategorien()
    assert len(rufe) == 1


# ───────────────────────────────────── die Anweisung sagt es auch

def test_die_persona_sagt_was_bei_keinem_treffer_zu_tun_ist():
    """Ein Enum verhindert erfundene Werte nur, wenn OpenAI streng
    prueft — und `strict` ist nicht gesetzt. Die Anweisung ist die
    zweite Sicherung, nicht Zierrat."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "KATEGORIEN" in t
    assert "LASS ES WEG" in t
