"""Der Budget-Fehler muss das ECHTE Fenster nennen (2026-08-18).

Alexanders Arcturian-Stimme startete fuenf Tage lang nicht. Das Portal
zeigte „Tageslimit erreicht — morgen wieder verfuegbar", weil mein
Dienst `daily_budget_exceeded` schickte. Das Fenster stand aber auf
`monthly`: 32,21 EUR von 30,00 EUR verbraucht, offen erst am
1. September.

Der Wachter arbeitete exakt wie eingestellt. Falsch war die
BESCHRIFTUNG — `_today_str()` liefert je nach Konfiguration einen Tag
ODER einen Monat, und der Kommentar sagt selbst „name kept for compat".
Eine Beschriftung, die eine Umstellung nicht mitgemacht hat, altert
lautlos.

Damals blieb der Fehlercode stehen, weil ein ausgeliefertes iOS-Geraet
ihn als geschlossenes Enum prueft. Seit #1314 liegt AppDevs Zustimmung
vor und der Code heisst `budget_exceeded` — siehe
`tests/test_budget_fehlercode.py` fuer die Regeln, die dabei
gelten.
"""

from ai.services import realtime_budget_guard as g


def test_fehler_traegt_fenster_zahlen_und_zeitpunkt():
    e = g.DailyBudgetExceeded("alex-default", 32.206290476190496, 30.0)
    f = e.public_fields
    assert f["window"] == g.BUDGET_WINDOW
    assert f["used_eur"] == 32.21
    assert f["limit_eur"] == 30.0
    assert f["resets_at"]


def test_rohwerte_bleiben_im_protokoll_nicht_in_der_antwort():
    """Profil und ungerundete Betraege gehoeren nicht zum Browser."""
    e = g.DailyBudgetExceeded("alex-default", 32.206290476190496, 30.0)
    assert "alex-default" in e.audit_detail
    assert "alex-default" not in str(e.public_fields)
    # gerundet, nicht die 15 Nachkommastellen aus der Zustandsdatei
    assert e.public_fields["used_eur"] == 32.21


def test_monatsfenster_zeigt_auf_den_monatsersten(monkeypatch):
    monkeypatch.setattr(g, "BUDGET_WINDOW", "monthly")
    r = g._window_reset_at()
    assert r.endswith(("+02:00", "+01:00"))
    tag = r.split("T")[0]
    assert tag.endswith("-01"), f"Monatsfenster oeffnet nicht am Ersten: {r}"


def test_tagesfenster_zeigt_auf_mitternacht(monkeypatch):
    monkeypatch.setattr(g, "BUDGET_WINDOW", "daily")
    r = g._window_reset_at()
    assert "T00:00:00" in r


def test_router_reicht_die_felder_durch():
    """Ohne diese Zeile bleibt der Fehler so nichtssagend wie vorher."""
    import inspect
    from ai.routes import realtime_routes as rr
    quelle = inspect.getsource(rr)
    assert '**(exc.public_fields or {})' in quelle
    # Und das Protokoll-Detail darf NICHT mitgehen.
    stelle = quelle.index('**(exc.public_fields or {})')
    fenster = quelle[stelle - 200:stelle + 100]
    assert "audit_detail" not in fenster
