"""Komplette Ausrüstung: EINE Suche über mehrere Kategorien (#1446).

Befund aus Alex' Sitzung `bfd02d1b` (22:51): „Zeig mir eine komplette
Montur" ließ das Modell fünf einzelne `find_products` feuern — jede
Auswahl ersetzte die vorige, am Ende stand ein einzelner Stiefel im
Hero. Es fehlte die Möglichkeit, mehrere Kategorien in EINER Auswahl
zu halten.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


@pytest.mark.parametrize("werkzeug", ["find_products", "refine_search"])
def test_beide_suchwerkzeuge_koennen_set_auswahl(werkzeug):
    t = [x for x in rr._product_finder_tools("O'Neal")
         if x["name"] == werkzeug][0]
    f = t["parameters"]["properties"]["per_category_limit"]
    assert f == {"type": "integer", "minimum": 1, "maximum": 10}


def test_grenzen_stimmen_mit_dem_server_ueberein():
    """oneals `ProductSelectionCriteriaPatch` führt 1–10. Eine weitere
    Spanne hier hieße, dass das Modell Werte setzen darf, die drüben
    mit 422 enden — und der Kunde hört „Katalog nicht erreichbar"."""
    f = rr._product_finder_tools(None)[0]["parameters"]["properties"]
    assert f["per_category_limit"]["minimum"] == 1
    assert f["per_category_limit"]["maximum"] == 10


def test_es_geht_als_kriterium_mit():
    """Laut Vertrag liegt es INNERHALB `criteria`, damit `refine_search`
    es vom Basis-Token erbt — mein vorhandener Kriterienpfad reicht es
    deshalb unverändert durch."""
    assert "per_category_limit" in rr._product_finder_kriterien_felder()


def test_details_bekommt_kein_set_feld():
    t = [x for x in rr._product_finder_tools(None)
         if x["name"] == "product_details"][0]
    assert "per_category_limit" not in t["parameters"]["properties"]


# ───────────────────────────────────────────── Persona

def test_die_persona_verbietet_die_kette_von_einzelsuchen():
    """Der eigentliche Fehler aus dem Log."""
    p = rr._product_finder_prompt("de", "O'Neal")
    assert "KOMPLETTE AUSRUESTUNG" in p
    assert "NIEMALS mehrere Suchen nacheinander" in p
    assert "jede ersetzt die vorige" in p


def test_die_persona_nennt_KEINE_kundenkategorien():
    """Der Auftrag schlug vor, fünf konkrete Slugs in den Text zu
    schreiben. Das wäre O'Neals Sortimentsstruktur im api-ai-Kern —
    genau die Sorte Kundendatum, die am 2026-08-27 mit `7f57a58`
    entfernt wurde, und der Leak-Wächter würde sie sofort wieder
    melden.

    Die Kategorien stehen ohnehin live im Werkzeug-Enum; die Persona
    beschreibt das MUSTER, nicht den Inhalt.
    """
    p = rr._product_finder_prompt("de", "O'Neal")
    for slug in ("helmets-mx", "jerseys-offroad", "pants-mx",
                 "gloves", "boots-mx", "protection-mx"):
        assert slug not in p, slug


def test_der_sprecher_nennt_gruppen_statt_einer_gesamtzahl():
    p = rr._product_finder_prompt("de", "O'Neal")
    assert "Gruppen" in p


def test_der_leak_waechter_bleibt_gruen():
    """Die schärfste Prüfung dieser Änderung: Sie darf den Kern nicht
    wieder mit Kundendaten füllen."""
    import subprocess
    import sys
    wurzel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, os.path.join(wurzel, "scripts", "tenant_leak_check.py")],
        capture_output=True, text=True, cwd=wurzel,
    )
    assert r.returncode == 0, r.stdout
