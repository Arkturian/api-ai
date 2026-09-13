"""Zeichen-Alignment -> Woerter. Anlass 13.09.2026, Aufnahme 125030
(Reel #1818): sechs Tokens wie 'gehen?\\n\\nUnd' — die Wortgrenze kannte
nur das Leerzeichen, nicht den Absatz."""
from ai.services import tts_service as t


def _align(text, dt=0.1):
    chars = list(text)
    starts = [round(i * dt, 3) for i in range(len(chars))]
    ends = [round((i + 1) * dt, 3) for i in range(len(chars))]
    return chars, starts, ends


def test_absatz_trennt_wie_leerzeichen():
    w = t.gruppiere_woerter(*_align("gehen?\n\nUnd ich"))
    assert [x["word"] for x in w] == ["gehen?", "Und", "ich"]
    assert w[0] == {"word": "gehen?", "start": 0.0, "end": 0.6}
    assert w[1] == {"word": "Und", "start": 0.8, "end": 1.1}


def test_alle_weissraeume_trennen():
    w = t.gruppiere_woerter(*_align("a\tb\r\nc  d"))
    assert [x["word"] for x in w] == ["a", "b", "c", "d"]


def test_letztes_wort_ohne_schlussweissraum_und_mit():
    assert [x["word"] for x in t.gruppiere_woerter(*_align("ab cd"))] == ["ab", "cd"]
    assert [x["word"] for x in t.gruppiere_woerter(*_align("ab cd\n"))] == ["ab", "cd"]
    assert t.gruppiere_woerter(*_align("ab cd\n"))[-1]["end"] == 0.5


def test_zeitversatz_wird_addiert():
    w = t.gruppiere_woerter(*_align("ab cd"), time_offset=10.0)
    assert w[0]["start"] == 10.0 and w[1]["end"] == 10.5


def test_leer():
    assert t.gruppiere_woerter([], [], []) == []
