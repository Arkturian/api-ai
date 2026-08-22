"""Woher stammt die Sperrentscheidung? (#1283, Nebenbefund)

CloudV2 fragte `cost-status` auf arkturian ab und sah dort nebeneinander:
`0,20 von 30,00 EUR`, `budget_exceeded: false` — und `requests_blocked:
true`. Beides stimmte: arkturian ist nicht der Master, seine Zahlen sind
Anzeige, die Sperre kommt von arkserver. Es SIEHT nur aus wie ein
Widerspruch, und diese Optik hat bei der Suche Zeit gekostet.

Ein Bericht, der zwei Wahrheiten aus verschiedenen Quellen ohne Etikett
nebeneinanderstellt, ist kein vollständiger Bericht.
"""

import pytest

from ai.services.openai_realtime_cost_tracker import OpenAIRealtimeCostTracker


@pytest.fixture
def zaehler(monkeypatch):
    """Frischer Zustand — der Zaehler ist ein SINGLETON.

    `OpenAIRealtimeCostTracker()` liefert immer dasselbe Objekt. Ohne
    das Zuruecksetzen traegt ein Test die harte Sperre des vorigen
    weiter; genau das ist hier passiert: allein gruen, im vollen Lauf
    rot, weil `decision_source` dann `hard_cap` statt `master` war.
    Ein Test, dessen Ergebnis von der Reihenfolge abhaengt, misst die
    Reihenfolge und nicht die Sache.
    """
    z = OpenAIRealtimeCostTracker()
    monkeypatch.setattr(z, "_maybe_reload_from_file", lambda: None)
    vorher = dict(z._usage_data)
    z._usage_data.pop("openai_realtime_hard_cap_active", None)
    yield z
    z._usage_data.clear()
    z._usage_data.update(vorher)


def test_ohne_master_ist_die_entscheidung_lokal(zaehler, monkeypatch):
    monkeypatch.setattr(zaehler, "master_url", "")
    monkeypatch.setattr(zaehler, "shared_secret", "")
    st = zaehler.get_status()
    assert st["decision_source"] == "local"
    assert st["master"] is None
    assert st["note"] is None


def test_mit_master_traegt_der_bericht_dessen_zahlen(zaehler, monkeypatch):
    """Der eigentliche Punkt: die fremden Zahlen sind SICHTBAR.

    Ohne sie muss der Leser raten, gegen welchen Deckel er gerade
    gelaufen ist — und die Zahl daneben, die er sieht, ist die falsche.
    """
    monkeypatch.setattr(zaehler, "master_url", "http://arkserver.example:8000")
    monkeypatch.setattr(zaehler, "shared_secret", "egal")
    monkeypatch.setattr(
        zaehler, "_fetch_master_status",
        lambda: {"would_block": True, "total_cost_eur": 35.12,
                 "monthly_budget_eur": 35.0, "month": "2026-08"},
    )
    st = zaehler.get_status()
    assert st["requests_blocked"] is True
    assert st["decision_source"] == "master"
    assert st["master"]["total_cost_eur"] == 35.12
    assert st["master"]["monthly_budget_eur"] == 35.0
    assert st["master"]["host"] == "arkserver.example"
    assert "Master" in st["note"]


def test_master_url_wird_nicht_im_klartext_ausgeliefert(zaehler, monkeypatch):
    """Eine URL kann Zugangsdaten tragen — der Host genuegt.

    Der Bericht geht an Frontends und in Protokolle. `user:pass@host`
    zwischen Doppelpunkt und Klammeraffe ist genau die Stelle, die ein
    Namensfilter uebersieht.
    """
    monkeypatch.setattr(
        zaehler, "master_url", "http://dienst:GEHEIM@arkserver.example:8000")
    monkeypatch.setattr(zaehler, "shared_secret", "egal")
    monkeypatch.setattr(zaehler, "_fetch_master_status", lambda: {"would_block": False})
    st = zaehler.get_status()
    assert st["master"]["host"] == "arkserver.example"
    assert "GEHEIM" not in repr(st)
    assert "dienst" not in repr(st)


def test_harte_sperre_nennt_sich_beim_namen(zaehler, monkeypatch):
    """Sonst liest sich der Notschalter wie ein erschoepftes Budget."""
    monkeypatch.setattr(zaehler, "master_url", "http://arkserver.example:8000")
    monkeypatch.setattr(zaehler, "shared_secret", "egal")
    zaehler._usage_data["openai_realtime_hard_cap_active"] = True
    st = zaehler.get_status()
    assert st["decision_source"] == "hard_cap"
    assert st["requests_blocked"] is True


def test_bestehende_felder_bleiben_unveraendert(zaehler, monkeypatch):
    """Kein Umbau eines Vertrags, den Frontends schon lesen."""
    monkeypatch.setattr(zaehler, "master_url", "")
    monkeypatch.setattr(zaehler, "shared_secret", "")
    st = zaehler.get_status()
    for feld in ("provider", "month", "total_cost_eur", "total_cost_usd",
                 "monthly_budget_eur", "usage_percentage", "remaining_eur",
                 "request_count", "session_count", "by_model",
                 "budget_exceeded", "requests_blocked"):
        assert feld in st


def test_unerreichbarer_master_wird_ausgewiesen_nicht_verschwiegen(zaehler, monkeypatch):
    """Die Fassung davor hing an einem Zwischenspeicher, den ein
    anderer Aufruf nebenbei füllte. War er leer — harte Sperre als
    Kurzschluss, oder ein gescheiterter Abruf — behauptete der Bericht
    die Herkunft `master` und zeigte keine einzige ihrer Zahlen. Diese
    beiden Tests waren beim ersten Lauf rot und haben genau das gefunden.
    """
    def kaputt():
        raise RuntimeError("Netz weg")
    monkeypatch.setattr(zaehler, "master_url", "http://arkserver.example:8000")
    monkeypatch.setattr(zaehler, "shared_secret", "egal")
    monkeypatch.setattr(zaehler, "_fetch_master_status", kaputt)
    st = zaehler.get_status()
    assert st["decision_source"] == "master"
    assert st["master"]["reachable"] is False
    assert st["master"]["total_cost_eur"] is None
