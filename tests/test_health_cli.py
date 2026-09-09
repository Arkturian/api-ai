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


def _jwt(exp):
    import base64, json as _j
    b = lambda o: base64.urlsafe_b64encode(_j.dumps(o).encode()).rstrip(b"=").decode()
    return f"{b({'alg':'none'})}.{b({'exp': exp})}.sig"


def test_codex_auth_abgelaufen_ist_nicht_ok(tmp_path, monkeypatch):
    """pdrei 09.09.: Binary 0.153.4 ok, Anmeldung vom 14.07. -> 502 beim Aufruf."""
    import json as _j, time as _t
    (tmp_path / "auth.json").write_text(_j.dumps({"auth_mode": "chatgpt", "last_refresh": "2026-07-14T10:00:00Z",
        "tokens": {"access_token": _jwt(int(_t.time()) - 86400 * 40), "refresh_token": "r"}}))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    a = main._codex_auth_status()
    assert a["ok"] is False and "abgelaufen" in a["reason"]
    assert a["last_refresh"].startswith("2026-07-14") and a["has_refresh_token"] is True
    assert "r" != a.get("refresh_token") and "access_token" not in a      # nie ein Token im Health


def test_codex_auth_gueltig_ist_ok(tmp_path, monkeypatch):
    import json as _j, time as _t
    (tmp_path / "auth.json").write_text(_j.dumps({"auth_mode": "chatgpt", "last_refresh": "2026-09-04T07:07:23Z",
        "tokens": {"access_token": _jwt(int(_t.time()) + 86400 * 5), "refresh_token": "r"}}))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    a = main._codex_auth_status()
    assert a["ok"] is True and a["access_token_expires_at"].endswith("Z")


def test_codex_auth_fehlt(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "leer"))
    assert main._codex_auth_status()["ok"] is False


def test_health_codex_traegt_auth(tmp_path, monkeypatch):
    import json as _j, time as _t
    _reset()
    (tmp_path / "auth.json").write_text(_j.dumps({"auth_mode": "chatgpt",
        "tokens": {"access_token": _jwt(int(_t.time()) + 3600)}}))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R(0, "codex-cli 0.153.4"))
    h = main.health_check()
    assert h["clis"]["codex"]["auth"]["ok"] is True
