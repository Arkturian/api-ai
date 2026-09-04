"""/health prueft das codex-CLI (Anlass 2026-09-04: arkturian 75 min tot,
/health „healthy“, Katalog „einsatzbereit“). Gegen den echten Fehler
gehalten: ein CLI, das beim Start wirft, muss als ok=false erscheinen."""

import subprocess

import main


class _R:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _reset():
    main._cli_status_cache.clear()


def test_kaputtes_cli_ist_nicht_ok(monkeypatch):
    _reset()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R(
        1, "", "Error: Missing optional dependency @openai/codex-linux-x64. Reinstall Codex"))
    st = main._cli_status("codex")
    assert st["ok"] is False and st["version"] is None
    assert "Missing optional dependency" in st["error"]


def test_gesundes_cli_meldet_version(monkeypatch):
    _reset()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R(0, "codex-cli 0.153.2\n"))
    st = main._cli_status("codex")
    assert st == {**st, "ok": True, "version": "0.153.2", "error": None}


def test_fehlendes_cli_ist_nicht_ok(monkeypatch):
    _reset()
    def boom(*a, **k): raise FileNotFoundError("codex")
    monkeypatch.setattr(subprocess, "run", boom)
    assert main._cli_status("codex")["ok"] is False


def test_cache_haelt_monitore_vom_cli_fern(monkeypatch):
    _reset()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (calls.append(1), _R(0, "codex-cli 1.0.0"))[1])
    main._cli_status("codex"); main._cli_status("codex")
    assert len(calls) == 1


def test_health_traegt_das_feld(monkeypatch):
    _reset()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R(0, "codex-cli 0.153.2"))
    h = main.health_check()
    assert h["clis"]["codex"]["ok"] is True and h["clis"]["codex"]["version"] == "0.153.2"
