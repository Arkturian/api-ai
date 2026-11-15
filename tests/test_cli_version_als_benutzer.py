"""Statusseite: `codex --version` darf nicht als root in Alex' codex-Home
schreiben (Automation, 15.11.: root-eigenes tmp/arg0-linux/codex-arg0*)."""
from types import SimpleNamespace

from ai.routes import dashboard_routes as db


def _fang(monkeypatch):
    gesehen = []
    monkeypatch.setattr(db.subprocess, "run",
                        lambda cmd, **k: gesehen.append(list(cmd)) or SimpleNamespace(stdout="x 1.0", stderr=""))
    return gesehen


def test_codex_version_laeuft_als_benutzer(monkeypatch):
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")
    gesehen = _fang(monkeypatch)
    assert db._cli_version("codex") == "x 1.0"
    assert gesehen[0][:4] == ["sudo", "-n", "-u", "alex"]
    assert gesehen[0][-1] == "--version"


def test_andere_clis_unveraendert(monkeypatch):
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")
    gesehen = _fang(monkeypatch)
    db._cli_version("claude")
    assert gesehen[0] == ["claude", "--version"]
