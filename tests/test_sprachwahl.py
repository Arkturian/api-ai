"""Der Sprecher folgt der im Finder gewählten Sprache.

Anlass (Alex, 2026-08-27 11:20): „Reagiert der Agent automatisch in
der Sprache, die ich am Anfang wähle?" — Nein, tat er nicht.
`_product_finder_prompt` nahm den Parameter `language` entgegen und
benutzte ihn NIE. Die Persona war durchgehend deutsch; das Modell
antwortete in der Sprache, in der die Anweisung geschrieben ist.

Der Parameter war da, die Signatur sah richtig aus, und nichts
schlug fehl. Eine unbenutzte Zusicherung ist schwerer zu sehen als
eine fehlende.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


@pytest.mark.parametrize("code,name", [
    ("de", "German"), ("en", "English"), ("sl", "Slovenian"),
    ("it", "Italian"), ("es", "Spanish"),
])
def test_alle_finder_sprachen_sind_bekannt(code, name):
    """Der Finder bietet fünf an. Spanisch fehlte in JEDER Tabelle des
    Moduls — ein spanischer Kunde wählte seine Sprache und bekam
    Deutsch."""
    assert rr._sprachname(code) == name


@pytest.mark.parametrize("code,name", [
    ("de", "German"), ("en", "English"), ("es", "Spanish"),
])
def test_die_anweisung_nennt_die_sprache(code, name):
    t = rr._product_finder_prompt(code, "O'Neal")
    assert f"Sprich und antworte auf {name}" in t


def test_die_sprache_steht_ganz_vorn():
    """Weit hinten in einer langen Anweisung wird sie überlesen — vom
    Modell wie vom Menschen."""
    t = rr._product_finder_prompt("es", "O'Neal")
    assert t.index("SPRACHE") < 200


def test_unbekannte_sprache_faellt_auf_deutsch():
    """Eine Entscheidung, kein Zufall: Der Außendienst dieses Kunden
    spricht Deutsch, und eine Sitzung in einer unverständlichen
    Sprache ist schlimmer als eine in der falschen."""
    for roh in ("pt", "xx", "", None, "  "):
        assert rr._sprachname(roh) == "German"


def test_regionalkennung_wird_akzeptiert():
    """`de-AT`, `es-MX` kommen aus Browsern. Ohne diese Kürzung fiele
    ein österreichischer Nutzer auf die Vorgabe zurück — was zufällig
    stimmte — und ein mexikanischer auf Deutsch."""
    assert rr._sprachname("es-MX") == "Spanish"
    assert rr._sprachname("de-AT") == "German"
    assert rr._sprachname("EN-GB") == "English"


def test_der_kunde_darf_wechseln_der_agent_nicht_von_selbst():
    """Wechselt der Kunde, folgt der Sprecher. Von sich aus die
    Sprache zu wechseln wäre im Verkaufsgespräch ein Fehler."""
    t = rr._product_finder_prompt("it", "O'Neal")
    assert "folge ihm" in t
    assert "beginne nicht von dir aus" in t


def test_eine_tabelle_nicht_mehrere():
    """Es gab vier Kopien im Modul, alle ohne Spanisch. Diese hier ist
    die, die der Produktfinder benutzt — und sie ist vollständig."""
    assert set(rr.SPRACHNAMEN) == {"de", "en", "sl", "it", "es"}
    import inspect
    quelle = inspect.getsource(rr._product_finder_prompt)
    assert "_sprachname(language)" in quelle


def test_parameter_wird_wirklich_benutzt():
    """Der eigentliche Fehler: Die Signatur sah richtig aus, der Wert
    kam nie an. Zwei verschiedene Sprachen müssen zwei verschiedene
    Anweisungen ergeben."""
    a = rr._product_finder_prompt("de", "O'Neal")
    b = rr._product_finder_prompt("es", "O'Neal")
    assert a != b
