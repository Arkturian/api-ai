"""Praesenz-Historie ist Begegnung, nicht Wahrheit (#4914 q-936e78536864, Punkt 8).

Der Verlauf frueherer Sprachsitzungen (Vorwort) sagt, was FRUEHER gesagt
wurde; was jetzt gilt, sagt nur der frische Kontext aus den Lesewerkzeugen.
Bei Konflikt gewinnt der frische Kontext — als Anweisung im Prompt, nicht
als Hoffnung.
"""

from ai.routes import realtime_routes as rr


def test_vorwort_ist_gedaechtnis_frischer_kontext_gewinnt():
    p = rr._companion_arcturian_prompt("de")
    assert "Gespraechs" in p and "gedaechtnis" in p
    assert "FRUEHER gesagt wurde" in p
    assert "frische Kontext" in p
    # Werkzeugnamen gehoeren NICHT in den Grundprompt (Opt-in-Vertrag) —
    # der Satz nennt sie deshalb nicht.
    assert "frage_kleinhirn" not in p
    assert "gilt der frische Kontext" in p
    # Der Satz steht VOR dem Sprachblock, also im Regelteil, nicht im Nachsatz.
    assert p.index("gilt der frische Kontext") < p.index("SPRACHE:")
