"""`/health` sagt, WAS diese Instanz kann — nicht welchen Commit sie trägt.

Anlass (OnealServ-Codex, 2026-08-27): Der laufende Produktfinder-BFF
zeigt auf die Kunden-Instanz, deren `/health` antwortete, aber keinen
nachprüfbaren Stand meldete. Die Frage „läuft dort schon der neue
Code" war von außen nicht zu beantworten — und eine Probe gegen den
falschen Host wäre keine Freigabe gewesen.

Ein Commit-Hash wäre die naheliegende Antwort und die schlechtere:
Der Deploy packt ein tar aus, das mitgelieferte `.git` bleibt alt.
Genau daran habe ich am 2026-08-26 eine Stunde verloren. Was man
wissen will, ist nicht „welcher Commit", sondern „kann diese Instanz
das Werkzeug".
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "x")


def test_health_nennt_die_realtime_werkzeuge():
    from main import health_check
    d = health_check()
    assert "realtime_tools" in d
    assert "cart_details" in d["realtime_tools"]
    assert "find_products" in d["realtime_tools"]


def test_die_liste_ist_sortiert_und_damit_vergleichbar():
    """Zwei Instanzen lassen sich nur vergleichen, wenn die Reihenfolge
    nicht vom Zufall abhängt."""
    from main import health_check
    w = health_check()["realtime_tools"]
    assert w == sorted(w)


def test_health_bleibt_gesund_wenn_die_werkzeuge_fehlen(monkeypatch):
    """Ein Gesundheitsbericht, der selbst krank werden kann, ist
    keiner. Fällt der Import aus, bleibt die Liste leer — der
    Health-Check antwortet trotzdem."""
    import main
    monkeypatch.setattr(main, "_realtime_faehigkeiten", lambda: [])
    d = main.health_check()
    assert d["status"] == "healthy"
    assert d["realtime_tools"] == []


def test_die_liste_kommt_aus_der_echten_werkzeugmenge():
    """Keine zweite Aufzählung — sonst meldet `/health` etwas anderes,
    als der Dispatch tatsächlich kann."""
    import inspect

    import main
    quelle = inspect.getsource(main._realtime_faehigkeiten)
    assert "READ_TOOL_NAMES" in quelle
