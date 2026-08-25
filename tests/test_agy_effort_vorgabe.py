"""Effort-Vorgabe für agy (Meldung Automation, 2026-08-22).

`agy models` führt ausschließlich suffigierte Slugs — `gemini-3.7-flash`
ohne Suffix existiert dort nicht. Ein Aufruf ohne `effort` kann deshalb
nie zu einer gültigen Modellwahl komponiert werden und endet zwangsläufig
in `requires --effort (available: low, medium, high)`.

Getroffen hat es `conversation-api`: rund 2.800 Fehlschläge am Tag, seit
mindestens dem 15.08., und weil dieser Dienst keine Rückfallkette hat
(#1185), fiel die Anreicherung still aus — niemand bemerkte es acht Tage
lang.

Der Einwand gegen eine Vorgabe ist ernst zu nehmen: Sie trifft eine
Wahl, die der Aufrufer nicht getroffen hat. Nur ist die Alternative
hier nicht seine Wahl, sondern ein sicherer Fehlschlag.
"""

import json

import pytest

from ai.routes import text_ai_routes as t


@pytest.fixture
def katalog(tmp_path, monkeypatch):
    def schreiben(efforts):
        p = tmp_path / "models.json"
        p.write_text(json.dumps({"providers": {"gemini": {"efforts": efforts}}}))
        monkeypatch.setattr(t, "_MODELS_JSON_PFAD", p)
        return p
    return schreiben


def test_katalog_vorgabe_wird_genommen(katalog):
    katalog({"default": "medium", "available": ["low", "medium", "high"]})
    assert t._agy_effort_vorgabe("gemini-3.7-flash") == "medium"


def test_der_katalog_gewinnt_gegen_die_notfallkonstante(katalog):
    """Die Wahrheit steht in `agy models` und kommt über Automation.

    Eine Konstante im Code, die den Katalog überstimmt, wäre wieder die
    driftende Tabelle, die 2026-07-29 aus genau diesem Modul entfernt
    wurde (Post #4383).
    """
    katalog({"default": "high"})
    assert t._agy_effort_vorgabe("gemini-3.7-flash") == "high"
    assert t._AGY_EFFORT_NOTFALL != "high"


def test_ohne_katalog_greift_die_notfallkonstante(tmp_path, monkeypatch):
    """Ein fehlender Katalog darf die 2.800 Fehlschläge nicht zurückholen."""
    monkeypatch.setattr(t, "_MODELS_JSON_PFAD", tmp_path / "gibtsnicht.json")
    assert t._agy_effort_vorgabe("gemini-3.7-flash") == t._AGY_EFFORT_NOTFALL


def test_kaputter_katalog_wirft_nicht(tmp_path, monkeypatch):
    p = tmp_path / "models.json"
    p.write_text("{kein json")
    monkeypatch.setattr(t, "_MODELS_JSON_PFAD", p)
    assert t._agy_effort_vorgabe("gemini-3.7-flash") == t._AGY_EFFORT_NOTFALL


def test_per_model_wird_beachtet(katalog):
    """`gemini-3.1-pro` kann nur low/high — eine pauschale `medium` wäre
    dort wieder ein sicherer Fehlschlag, nur mit anderem Wortlaut.

    Wir raten dann NICHT, welcher der erlaubten Werte gemeint war: agy
    nennt dem Aufrufer die gültige Liste besser als wir.
    """
    katalog({"default": "medium", "per_model": {"gemini-3.1-pro": ["low", "high"]}})
    assert t._agy_effort_vorgabe("gemini-3.1-pro") is None
    assert t._agy_effort_vorgabe("gemini-3.7-flash") == "medium"


def test_per_model_das_die_vorgabe_erlaubt_nimmt_sie(katalog):
    katalog({"default": "high", "per_model": {"gemini-3.1-pro": ["low", "high"]}})
    assert t._agy_effort_vorgabe("gemini-3.1-pro") == "high"


def test_antwort_traegt_das_feld():
    """Die Vorgabe darf nicht STILL sein — sonst tauscht sie einen
    lauten Fehler gegen eine unsichtbare Wahl."""
    felder = t.AIResponse.model_fields
    assert "effort_applied" in felder
    assert felder["effort_applied"].default is None


def test_variable_ist_vor_jeder_verzweigung_gebunden():
    """Der Fehler, der mich in diesem Dienst schon zweimal erwischt hat.

    `_effort_nachgereicht` wird nur in EINEM Zweig zugewiesen und
    hinterher in der Antwort gelesen. Ohne Bindung davor ist das ein
    NameError im Betrieb — beim Import unsichtbar, beim ersten echten
    Aufruf ein 500.
    """
    import inspect

    quelle = inspect.getsource(t.gemini_endpoint)
    zuweisung = quelle.index("_effort_nachgereicht: Optional[str] = None")
    erste_lesung = quelle.index("effort_applied=_effort_nachgereicht")
    assert zuweisung < erste_lesung
