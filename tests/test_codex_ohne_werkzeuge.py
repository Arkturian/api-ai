"""tools=false: reiner Textaufruf ohne Shell (CloudV2/cloud-api, 11.11.2026).

Der echte Fehler: Prosa-Zusammenfassung "in 150 Woertern" bei effort high.
codex zaehlte die Woerter per Shell nach (python, dann python3), drei
Modellrunden, 42k Eingabe-Tokens, 90 s, cloud-api brach nach 30 s mit 502 ab.
Gemessen: mit features.shell_tool=false und unified_exec=false antwortet das
Modell auf "fuehre uname -n aus" mit KEIN_WERKZEUG statt den Befehl auszufuehren.
"""
import asyncio
import types

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t

ERFOLG = ('{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n'
          '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}\n')
AUS = ["-c", "features.shell_tool=false", "-c", "features.unified_exec=false"]


def _verdrahten(monkeypatch):
    laeufe = []

    def fake_run(cmd, env=None, timeout=None, input=None):
        laeufe.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout=ERFOLG, stderr="")

    class Slot:
        def release(self): pass

    async def slot(_name): return Slot()

    monkeypatch.setattr(t, "_run_cli_with_pgid", fake_run)
    monkeypatch.setattr(t, "_acquire_cli_slot", slot)
    monkeypatch.setattr(t, "_codex_katalog", lambda: None)
    monkeypatch.setattr(t, "_codex_config_effort", lambda *a, **k: None)
    monkeypatch.setattr(t, "_download_storage_images", lambda _p: [])
    from ai.services import codex_cost_tracker as _cct
    monkeypatch.setattr(_cct.codex_cost_tracker, "track_usage", lambda *a, **k: None)
    return laeufe


def _enthaelt(cmd, teil):
    return any(cmd[i:i + len(teil)] == teil for i in range(len(cmd)))


def test_vorgabe_laesst_werkzeuge_an(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="hallo"), "gpt-5.6-luna", "x"))
    assert not _enthaelt(laeufe[0], AUS)
    assert "features.shell_tool=false" not in laeufe[0]


def test_tools_false_schaltet_shell_ab(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="Fasse in 150 Woertern zusammen", effort="low",
                                           tools=False), "gpt-5.6-luna", "x"))
    assert _enthaelt(laeufe[0], AUS)
    # Prompt bleibt das letzte Argument (stdin-Marker), die Flags stehen davor.
    assert laeufe[0][-1] == "-"


def test_tools_false_verbindet_sich_mit_sandbox(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="x", sandbox="read-only", tools=False),
                                  "gpt-5.6-luna", "x"))
    assert _enthaelt(laeufe[0], ["-s", "read-only"]) and _enthaelt(laeufe[0], AUS)


@pytest.mark.parametrize("endpunkt", ["grok", "gemini"])
def test_andere_cli_pfade_weisen_tools_false_ab(endpunkt):
    with pytest.raises(HTTPException) as e:
        t._weise_tools_aus_ab(t.Prompt(prompt="x", tools=False), endpoint=endpunkt)
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "tools_switch_unsupported_on_endpoint"
    t._weise_tools_aus_ab(t.Prompt(prompt="x"), endpoint=endpunkt)  # Vorgabe: kein Fehler
