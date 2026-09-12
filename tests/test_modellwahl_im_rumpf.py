"""Modellwahl im Rumpf wird nicht mehr still verworfen.

Gemessen am 12.09.: ein Aufruf an /ai/chatgpt mit
`{"model": "gpt-5.6-terra"}` IM RUMPF lief auf der Vorgabe und meldete
`model: "codex-default"` zurueck. Pydantic verwirft unbekannte Felder
still, das Modell ist ein Abfrageparameter. Der Aufrufer hielt eine
ganze Auswertung fuer das Ergebnis eines Modells, das nie lief — der
Fehler faellt nur auf, wenn jemand das Antwortfeld `model` liest.

Ziel: Rumpf wirkt. Gegenfall: der Abfrageparameter behaelt Vorrang.
Nachbarfall: ohne beides bleibt es bei None, also der CLI-Vorgabe.
"""

import inspect

import pytest

from ai.routes import text_ai_routes as t


def test_prompt_kennt_das_feld():
    assert "model" in t.Prompt.model_fields
    assert t.Prompt(prompt="x").model is None
    assert t.Prompt(prompt="x", model="gpt-6-astra").model == "gpt-6-astra"


@pytest.mark.parametrize("funktion", [
    t.claude_endpoint, t.chatgpt_endpoint, t.grok_endpoint,
    t.gemini_endpoint, t.gemini_vision_endpoint, t.m3_endpoint,
    t.deepseek_endpoint,
])
def test_jeder_endpunkt_loest_die_modellwahl_auf(funktion):
    """Haelt die Verdrahtung. Ein Endpunkt ohne diese Zeile verwirft
    die Wahl des Aufrufers wieder still."""
    quelle = inspect.getsource(funktion)
    assert "model = model or prompt.model" in quelle


@pytest.mark.parametrize("funktion", [
    t.claude_endpoint, t.chatgpt_endpoint, t.grok_endpoint,
    t.gemini_endpoint, t.gemini_vision_endpoint, t.m3_endpoint,
    t.deepseek_endpoint,
])
def test_abfrageparameter_behaelt_vorrang(funktion):
    """`model or prompt.model` — nicht umgekehrt. Der Abfrageparameter
    ist der dokumentierte Weg; wer ihn setzt, meint ihn."""
    quelle = inspect.getsource(funktion)
    assert "prompt.model or model" not in quelle
