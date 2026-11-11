"""tools=false: reiner Textaufruf ohne Shell (CloudV2/cloud-api, 11.11.2026).

Der echte Fehler: Prosa-Zusammenfassung "in 150 Woertern" bei effort high.
codex zaehlte die Woerter per Shell nach (python, dann python3), drei
Modellrunden, 42k Eingabe-Tokens, 90 s, cloud-api brach nach 30 s mit 502 ab.
Nur Shell aus reichte nicht: das Modell zaehlte dann per Code-Modus (JS-exec)
weiter. Gemessen mit allen fuenf Schaltern aus: null Werkzeugaufrufe.
"""
import asyncio
import types

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t

ERFOLG = ('{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n'
          '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}\n')
AUS = ["-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
       "-c", "features.code_mode_host=false", "-c", "features.apps=false",
       "-c", "features.plugins=false"]


umgebungen = []


def _verdrahten(monkeypatch, tmp_path=None):
    laeufe = []
    umgebungen.clear()
    if tmp_path is not None:
        haupt = tmp_path / "codex"
        haupt.mkdir()
        (haupt / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(haupt))
        monkeypatch.delenv("CODEX_TEXT_HOME", raising=False)
        monkeypatch.delenv("CODEX_RUN_AS_USER", raising=False)

    def fake_run(cmd, env=None, timeout=None, input=None):
        laeufe.append(list(cmd))
        umgebungen.append(dict(env or {}))
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


def test_vorgabe_laesst_werkzeuge_an(monkeypatch, tmp_path):
    laeufe = _verdrahten(monkeypatch, tmp_path)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="hallo"), "gpt-5.6-luna", "x"))
    assert not _enthaelt(laeufe[0], AUS)
    assert "features.shell_tool=false" not in laeufe[0]


def test_tools_false_schaltet_shell_ab(monkeypatch, tmp_path):
    laeufe = _verdrahten(monkeypatch, tmp_path)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="Fasse in 150 Woertern zusammen", effort="low",
                                           tools=False), "gpt-5.6-luna", "x"))
    assert _enthaelt(laeufe[0], AUS)
    # Prompt bleibt das letzte Argument (stdin-Marker), die Flags stehen davor.
    assert laeufe[0][-1] == "-"


def test_tools_false_verbindet_sich_mit_sandbox(monkeypatch, tmp_path):
    laeufe = _verdrahten(monkeypatch, tmp_path)
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


# --- Minimal-Instruktion und eigenes Text-Home (CloudV2, 11.11.) ---------

import os
from pathlib import Path


def test_tools_false_nimmt_minimal_instruktion_und_ephemeral(monkeypatch, tmp_path):
    laeufe = _verdrahten(monkeypatch, tmp_path)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="x", tools=False), "gpt-5.6-luna", "x"))
    cmd = laeufe[0]
    assert "--ephemeral" in cmd
    anweisung = next(a for a in cmd if a.startswith("model_instructions_file="))
    assert Path(anweisung.split("=", 1)[1]).is_file()
    assert _enthaelt(cmd, ["-c", "include_environment_context=false"])


def test_tools_false_laeuft_im_text_home_mit_link_auf_die_anmeldung(monkeypatch, tmp_path):
    _verdrahten(monkeypatch, tmp_path)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="x", tools=False), "gpt-5.6-luna", "x"))
    haupt = tmp_path / "codex"
    assert umgebungen[0]["CODEX_HOME"] == str(haupt) + "-text"
    link = Path(str(haupt) + "-text") / "auth.json"
    assert link.is_symlink() and os.readlink(link) == str(haupt / "auth.json")


def test_vorgabe_bleibt_im_hauptverzeichnis(monkeypatch, tmp_path):
    laeufe = _verdrahten(monkeypatch, tmp_path)
    asyncio.run(t._chatgpt_einmal(t.Prompt(prompt="x"), "gpt-5.6-luna", "x"))
    assert umgebungen[0]["CODEX_HOME"] == str(tmp_path / "codex")
    assert "--ephemeral" not in laeufe[0]
    assert not Path(str(tmp_path / "codex") + "-text").exists()


def test_ohne_anmeldung_kein_text_home(tmp_path):
    assert t._codex_text_home(str(tmp_path / "leer")) is None


def test_ersetzte_anmeldung_wird_an_die_quelle_zurueckgeschrieben(tmp_path):
    """codex schreibt ein erneuertes Token womoeglich per Umbenennen: dann
    steht im Text-Home eine Datei statt des Links. Die neuere Anmeldung muss
    zurueck an die Quelle, sonst gilt dort ein verbrauchtes Refresh-Token."""
    konto = tmp_path / "konten" / "a"
    konto.mkdir(parents=True)
    (konto / "auth.json").write_text("alt")
    haupt = tmp_path / "codex"
    haupt.mkdir()
    (haupt / "auth.json").symlink_to(konto / "auth.json")
    text = Path(t._codex_text_home(str(haupt)))
    link = text / "auth.json"
    link.unlink()
    link.write_text("neu")
    os.utime(konto / "auth.json", (1, 1))
    assert t._codex_text_home(str(haupt)) == str(text)
    assert (konto / "auth.json").read_text() == "neu"        # Kontodatei erneuert
    assert (haupt / "auth.json").is_symlink()                 # Kontowechsel-Link bleibt
    assert link.is_symlink() and link.read_text() == "neu"
