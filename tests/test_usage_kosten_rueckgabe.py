"""`/ai/realtime/usage` meldet die Kosten DIESER Zeile zurück.

Anlass (OnealServ-Codex, #1413): Der BFF schreibt die Kosten je
Sitzung mit. Ohne einen Wert von mir müsste er die Preistabelle
kopieren — und eine zweite Preistabelle driftet gegen meine. Dann
stünden im Protokoll Kosten, die mit der Abrechnung nichts zu tun
haben.

`None` heißt „nichts verbucht" (Nullmeldung oder Dublette). Eine 0.0
wäre von „hat nichts gekostet" nicht zu unterscheiden — und genau
diese Verwechslung hat uns gestern Nacht eine Stunde gekostet, als
ein leerer Topf wie „günstig" aussah statt wie „nicht gemessen".
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


def test_der_endpunkt_gibt_die_zeilenkosten_zurueck():
    import inspect
    quelle = inspect.getsource(rr.realtime_usage_report)
    assert 'status["cost_eur"]' in quelle


def test_null_meldung_liefert_None_nicht_null_euro():
    """Der Unterschied zwischen „nichts verbucht" und „kostete nichts"."""
    import inspect
    quelle = inspect.getsource(rr.realtime_usage_report)
    assert "zeilen_eur = None" in quelle
    assert "if zeilen_eur is not None else None" in quelle


def test_der_wert_stammt_aus_derselben_rechnung_wie_der_waechter():
    """Eine zweite Rechnung driftet. Der zurückgegebene Wert MUSS der
    sein, mit dem auch der Budgetwächter belastet wurde."""
    import inspect
    quelle = inspect.getsource(rr.realtime_usage_report)
    assert "zeilen_eur = float(per_row_eur)" in quelle
    assert "cost_eur=zeilen_eur" in quelle


def test_die_preistabelle_liegt_nur_an_einer_stelle():
    """Wenn oneal den Wert von mir nimmt, gibt es genau eine Tabelle."""
    from ai.services import openai_realtime_cost_tracker as t
    assert hasattr(t, "OPENAI_REALTIME_PRICING")
    import inspect
    quelle = inspect.getsource(rr.realtime_usage_report)
    assert "_cost_for_session" in quelle
