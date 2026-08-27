"""`X-Usage-Response: minimal` — nur, was der Aufrufer braucht.

Die breite Antwort trägt Gesamtausgaben, Monatsbudget und Tokenzahlen
über ALLE Profile. Hinter dem `usage`-Scope ist das nicht gefährlich,
aber es ist mehr als der Vertrag verlangt — und dieselbe Form, die im
August zugeschlagen hat, als `/ai/*/cost-status` Ausgaben und Budget
des Eigentümers offen auslieferte.

Opt-in, nicht Vorgabe: Ein Beschneiden bräche jeden bestehenden
Aufrufer, und ich weiß nicht sicher, wer sie heute liest.
"""

import inspect
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402

QUELLE = inspect.getsource(rr.realtime_usage_report)


def test_der_kopf_existiert():
    assert 'alias="X-Usage-Response"' in QUELLE


def test_minimal_liefert_genau_drei_felder():
    assert '"accepted": status["accepted"]' in QUELLE
    assert '"deduped": status["deduped"]' in QUELLE
    assert '"cost_eur": kosten' in QUELLE


def test_die_breite_form_bleibt_die_vorgabe():
    """Ohne Kopf ändert sich für bestehende Aufrufer nichts."""
    i_minimal = QUELLE.index('== "minimal"')
    i_return_status = QUELLE.rindex("return status")
    assert i_return_status > i_minimal, "breite Form muss der Rückfall sein"


def test_schreibweise_und_leerzeichen_stoeren_nicht():
    assert '(x_usage_response or "").strip().lower()' in QUELLE


def test_die_schmale_form_traegt_keine_gesamtsummen():
    """Der eigentliche Zweck. Stünde `total_cost_eur` darin, hätte das
    Beschneiden nichts gebracht."""
    start = QUELLE.index('== "minimal"')
    block = QUELLE[start:start + 400]
    for feld in ("total_cost_eur", "monthly_budget_eur", "by_model",
                 "request_count", "session_count"):
        assert feld not in block, feld
