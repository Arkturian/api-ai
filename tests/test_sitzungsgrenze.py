"""Kostengrenze je EINZELNER Sprachsitzung (#1283, CloudV2s Punkt 3).

„Ein Monatstopf merkt erst nach dem Schaden, dass eine einzelne Sitzung
teuer war." Im August waren es 963 Antworten in 16 Sitzungen — der
Deckel schlug zu, als das Geld weg war, nicht als eine Sitzung entglitt.

Zwei Dinge, die diese Prüfungen festhalten:

* Ohne Zahl wird nur GEZÄHLT. Zählen ist die Vorbedingung fürs Deckeln,
  nicht die halbe Umsetzung — deckeln kann man nur, was man misst.
* Durchgesetzt wird am Herzschlag, nicht beim Verrechnen. Laufendes
  Audio lässt sich nicht mitten im Satz abschneiden; das steht seit
  Codex' v1-Regel im Docstring von `confirm_usage_charge` und gilt
  weiter.
"""

import pytest

from ai.services import realtime_budget_guard as g


@pytest.fixture(autouse=True)
def frischer_zustand(tmp_path, monkeypatch):
    """Zustandsdatei umlenken — mit Nachweis, dass es gegriffen hat.

    Erster Anlauf patchte `STATE_PATH`; so heisst die Konstante nicht.
    Die Tests liefen daraufhin gegen
    `/var/lib/api-ai/realtime_reservations.json`, also gegen den ECHTEN
    Budgetzustand, und scheiterten nur, weil die Dateirechte es
    verboten. Bei laxeren Rechten haetten sie Alex' laufende
    Reservierungen ueberschrieben — eine Testsuite, die Produktion
    anfasst, ohne dass jemand es merkt.

    Deshalb `raising=True` (Tippfehler im Namen fliegt sofort auf) und
    eine ausdrueckliche Zusicherung darunter.
    """
    ziel = tmp_path / "reservations.json"
    monkeypatch.setattr(g, "RESERVATIONS_PATH", ziel)
    assert g.RESERVATIONS_PATH == ziel
    assert "/var/lib" not in str(g.RESERVATIONS_PATH)
    monkeypatch.delenv(g.SESSION_BUDGET_ENV, raising=False)


def _laden(profil="p", nutzer="u", vid="vs1", eur=1.0):
    return g.confirm_usage_charge(
        profile_id=profil, user_id=nutzer, voice_session_id=vid, cost_eur=eur
    )


# ─────────────────────────────────────────────────────────── Schalter

@pytest.mark.parametrize("wert", ["", "0", "-2", "viel"])
def test_ohne_brauchbare_zahl_keine_grenze(monkeypatch, wert):
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, wert)
    assert g._session_budget_eur() == 0.0


def test_zahl_wird_gelesen(monkeypatch):
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.50")
    assert g._session_budget_eur() == 2.5


# ─────────────────────────────────────────────────────────── Zählen

def test_kosten_werden_je_sitzung_getrennt_gefuehrt():
    """Der eigentliche Mangel: Es gab diese Zahl bisher gar nicht.

    `confirm_usage_charge` bekam die voice_session_id schon immer
    übergeben und hat sie nur protokolliert.
    """
    _laden(vid="vs1", eur=1.0)
    _laden(vid="vs2", eur=0.25)
    s = _laden(vid="vs1", eur=0.5)
    assert s["session_eur"] == pytest.approx(1.5)
    assert s["daily_total_eur"] == pytest.approx(1.75)


def test_ohne_sitzungskennung_entsteht_kein_sammeleintrag():
    """Sonst vermengt ein leerer Schlüssel mehrere Sitzungen zu einer —
    und die Grenze träfe irgendwann die falsche."""
    s = g.confirm_usage_charge(
        profile_id="p", user_id="u", voice_session_id="", cost_eur=3.0
    )
    assert s["session_eur"] is None
    assert s["daily_total_eur"] == pytest.approx(3.0)


# ────────────────────────────────────────────────────── Durchsetzung

def _sitzung_anlegen(vid="vs1"):
    with g._locked_state() as state:
        pv = g._profile_view(state, "p", g._today_str())
        uv = g._user_view(pv, "u")
        uv.setdefault("active_sessions", []).append(vid)


def test_ohne_grenze_schlaegt_das_herz_weiter():
    _sitzung_anlegen()
    _laden(eur=999.0)
    assert g.refresh_lease("p", "u", "vs1") is True


def test_unter_der_grenze_schlaegt_das_herz_weiter(monkeypatch):
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.00")
    _sitzung_anlegen()
    _laden(eur=1.50)
    assert g.refresh_lease("p", "u", "vs1") is True


def test_ueber_der_grenze_wird_der_herzschlag_verweigert(monkeypatch):
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.00")
    _sitzung_anlegen()
    _laden(eur=2.50)
    assert g.refresh_lease("p", "u", "vs1") is False


def test_die_grenze_trifft_nur_die_teure_sitzung(monkeypatch):
    """Eine entglittene Sitzung darf die daneben nicht mitreissen."""
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.00")
    _sitzung_anlegen("teuer")
    _sitzung_anlegen("guenstig")
    _laden(vid="teuer", eur=5.0)
    _laden(vid="guenstig", eur=0.1)
    assert g.refresh_lease("p", "u", "teuer") is False
    assert g.refresh_lease("p", "u", "guenstig") is True


def test_verrechnen_wirft_weiterhin_nicht(monkeypatch):
    """Codex' v1-Regel bleibt: über der Grenze wird verbucht, nicht
    abgebrochen. Wer hier eine Ausnahme einbaut, schneidet einen
    laufenden Satz ab."""
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "0.01")
    s = _laden(eur=99.0)
    assert s["session_eur"] == pytest.approx(99.0)
