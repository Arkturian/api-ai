"""codex laeuft als alex, wenn CODEX_RUN_AS_USER gesetzt ist.

Anlass (Automation, 13.09.): der Dienst (root, CODEX_HOME=/home/alex/.codex)
machte 12 425 Pfade in Alex' Home root-eigen, zuletzt config.toml 0600 —
codex fuer alex unbrauchbar. Ziel: Praefix sudo -n -u alex mit
weitergereichter Umgebung. Gegenfall: ohne Variable unveraendert.
Nachbarfall: andere CLIs (claude, agy) bleiben unangetastet; HOME=/root
wird nicht an alex weitergegeben.
"""

from ai.routes import text_ai_routes as t


def test_ohne_variable_unveraendert(monkeypatch):
    monkeypatch.delenv("CODEX_RUN_AS_USER", raising=False)
    cmd = ["codex", "exec", "--json", "-"]
    assert t.codex_als_benutzer(cmd, {"PATH": "/usr/bin"}) == cmd


def test_praefix_mit_umgebung(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")
    fake = tmp_path / "codex"; fake.write_text("#!/bin/sh\n"); fake.chmod(0o755)
    env = {"PATH": str(tmp_path) + ":/usr/bin", "CODEX_HOME": "/home/alex/.codex", "NO_COLOR": "1", "HOME": "/root"}
    out = t.codex_als_benutzer(["codex", "exec", "--json", "-"], env)
    assert out[:6] == ["sudo", "-n", "-u", "alex", "-H", "env"]
    assert "CODEX_HOME=/home/alex/.codex" in out and "NO_COLOR=1" in out
    assert not any(x.startswith("HOME=") for x in out)          # root-HOME nicht an alex
    assert str(fake) in out                                     # aufgeloester Pfad, nicht "codex"
    assert out[-3:] == ["exec", "--json", "-"]


def test_andere_clis_bleiben(monkeypatch):
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")
    assert t.codex_als_benutzer(["claude", "-p", "x"], {"PATH": "/usr/bin"}) == ["claude", "-p", "x"]


def test_verdrahtung():
    import inspect
    assert "cmd = codex_als_benutzer(cmd, env)" in inspect.getsource(t._run_cli_with_pgid)
    assert 'codex_als_benutzer(["codex", "debug", "models"], env)' in inspect.getsource(t._codex_katalog)
