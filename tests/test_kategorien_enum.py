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
    """Nur die Suchwerkzeuge tragen Kriterien. `product_details` waehlt
    nicht aus, es fragt nach dem bereits gewaehlten Produkt."""
    mit_kriterien = [t for t in rr._product_finder_tools("O'Neal")
                     if "category" in t["parameters"]["properties"]]
    assert {t["name"] for t in mit_kriterien} == {"find_products", "refine_search"}
    for t in mit_kriterien:
        assert t["parameters"]["properties"]["category"]["items"].get("enum"), t["name"]


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

@pytest.mark.ohne_kundendaten
def test_ersatzteile_und_buchhaltung_bleiben_draussen(monkeypatch):
    """`z-spare-parts-helmets` ist mit 1357 Eintraegen die GROESSTE
    Kategorie. Gaebe ich sie dem Modell, schluege eine Frage nach
    Helmen zuerst Ersatzvisiere vor."""
    # Der Server liefert sie mit `relevant_only=true` gar nicht mehr.
    # Belegt wird jetzt, dass ich danach FRAGE — nicht, dass ich eine
    # Ausschlussliste eines Kunden pflege.
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://o.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")
    gesehen = {}

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"slug": "helmets-mx"}]}

    def _merk(url, params=None, headers=None, timeout=None):
        gesehen["params"] = params
        return _A()

    monkeypatch.setattr(rr.httpx, "get", _merk)
    rr._oneal_kategorien()
    assert gesehen["params"] == {"relevant_only": "true"}


@pytest.mark.ohne_kundendaten
def test_der_server_filtert_jetzt_statt_ich(monkeypatch):
    """Bis 2026-08-27 stand hier eine Ausschlussliste mit Slugs eines
    Kunden — dieselbe Sorte Leck wie die Abschrift daneben, nur
    kleiner und deshalb übersehen, auch von meinem eigenen Wächter.

    oneal beantwortet die Frage seit `c18a3de` selbst. Ich filtere
    nicht mehr, ich frage präziser: `relevant_only=true`.
    """
    assert not hasattr(rr, "_ist_kundenkategorie")
    assert not hasattr(rr, "_KATEGORIE_AUSSCHLUSS")
    import inspect
    quelle = inspect.getsource(rr._oneal_kategorien)
    assert '"relevant_only": "true"' in quelle


# ───────────────────────────── faellt nie aus, cacht aber

@pytest.mark.ohne_kundendaten
def test_ohne_katalog_gibt_es_keine_kategorien():
    """Ein Mint darf nicht daran scheitern, dass ein Katalogdienst
    langsam ist — das Werkzeug bliebe ohne Kategorien und der Kunde
    hoerte denselben Satz wie beim echten Fehler."""
    # Seit dem Kundenschnitt gibt es KEINE Abschrift mehr. Eine leere
    # Liste heisst „nicht beschaffbar" — das Werkzeug bietet dann kein
    # Kategoriefeld an, statt mit einer veralteten Liste zu suchen.
    assert rr._oneal_kategorien() == []


@pytest.mark.ohne_kundendaten
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
    # Seit dem Kundenschnitt gibt es KEINE Abschrift mehr. Eine leere
    # Liste heisst „nicht beschaffbar" — das Werkzeug bietet dann kein
    # Kategoriefeld an, statt mit einer veralteten Liste zu suchen.
    assert rr._oneal_kategorien() == []


@pytest.mark.ohne_kundendaten
def test_fehler_beim_abruf_ist_kein_absturz(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    def _kaputt(*a, **k):
        raise RuntimeError("Netz weg")

    monkeypatch.setattr(rr.httpx, "get", _kaputt)
    # Seit dem Kundenschnitt gibt es KEINE Abschrift mehr. Eine leere
    # Liste heisst „nicht beschaffbar" — das Werkzeug bietet dann kein
    # Kategoriefeld an, statt mit einer veralteten Liste zu suchen.
    assert rr._oneal_kategorien() == []


@pytest.mark.ohne_kundendaten
def test_lebende_liste_gewinnt_und_wird_gefiltert(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return ["helmets-mx", "neue-kategorie"]

    monkeypatch.setattr(rr.httpx, "get", lambda *a, **k: _A())
    assert rr._oneal_kategorien() == ["helmets-mx", "neue-kategorie"]


@pytest.mark.ohne_kundendaten
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


# ═══════════ die ECHTE Antwortform, nicht die angenommene ═══════════

@pytest.mark.ohne_kundendaten
def test_die_echte_antwortform_von_oneal_wird_gelesen(monkeypatch):
    """Aufgezeichnet aus der Produktion am 2026-08-27 00:10.

    Meine erste Fassung parste `items`/`categories` — oneal liefert
    aber `data`. Ergebnis: HTTP 200, null Eintraege, und mein
    Leer-Schutz machte daraus lautlos die Rueckfallliste. Der Abruf
    war „erfolgreich" und wirkungslos.

    Bitterer Teil: Mein erster Test benutzte eine nackte Liste — also
    die Form, die ich ANGENOMMEN hatte. Ein Test gegen die eigene
    Annahme prueft nichts.
    """
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_API_KEY_ENV, "k")

    class _A:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [
                {"id": 17177, "name": "ADV Pants", "slug": "adv-pants",
                 "name_de": "ADV Pants", "name_en": None},
                {"id": 29, "name": "Bags / Backpacks",
                 "slug": "bags---backpacks",
                 "name_de": "Taschen / Rucksäcke", "name_en": None},
            ]}

    monkeypatch.setattr(rr.httpx, "get", lambda *a, **k: _A())
    werte = rr._oneal_kategorien()
    assert werte == ["adv-pants", "bags---backpacks"]
    # Frueher stand hier zusaetzlich „nicht die Rueckfallliste". Seit
    # dem Kundenschnitt gibt es keine mehr — ein leeres Ergebnis waere
    # jetzt eine leere Liste, und die ist von einem gelungenen Abruf
    # ohnehin unterscheidbar.
    assert werte, "leere Liste heisst: der Abruf ist wirkungslos"


def test_verworfene_kriterien_von_oneal_werden_protokolliert():
    """oneal verwirft seit `e851e91` unbekanntes Vokabular still und
    meldet es in `hints.ignored_criteria`. Das ist kein Fehler — der
    Kunde bekommt Treffer. Es ist der Fruehwarnwert dafuer, dass mein
    Werkzeugschema und ihr Vokabular auseinanderlaufen. Steht er nur
    in ihrem Log, sieht ihn niemand, der mein Schema pflegt.
    """
    import inspect
    quelle = inspect.getsource(rr._tool_product_search)
    assert "ignored_criteria" in quelle
