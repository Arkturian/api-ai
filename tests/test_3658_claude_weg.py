"""#3658: Der Claude-Weg meldet dauerhafte Zustaende mit festem Code.

Am Ziel (22.11.) endete die letzte Stufe als Aussetzer: Die CLI lag fuer den
Dienstbenutzer nicht ausfuehrbar, und die Anmeldung war nicht lesbar.
"""
import asyncio
import subprocess

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t
from ai.services import kette as k


@pytest.fixture(autouse=True)
def keine_cli(monkeypatch):
    def verboten(*a, **kw):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)


def lauf(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_unlesbare_anmeldung_ist_kein_konto(monkeypatch, tmp_path):
    datei = tmp_path / ".credentials.json"
    datei.write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    echt = type(datei).read_text

    def gesperrt(self, *a, **kw):
        if self.name == ".credentials.json":
            raise PermissionError("nein")
        return echt(self, *a, **kw)
    monkeypatch.setattr(type(datei), "read_text", gesperrt)
    with pytest.raises(HTTPException) as e:
        t._claude_konto_vorab()
    d = e.value.detail
    assert (e.value.status_code, d["error"], d["grund"]) == (503, "claude_account_unavailable", "anmeldung_nicht_lesbar")
    assert k.einordnen(e.value) == ("fehlt", None)


@pytest.mark.parametrize("stderr,art", [
    ("/usr/local/bin/claude: line 9: /home/alex/.local/share/claude/versions/2.1.173: Permission denied", "fehlt"),
    ("bash: claude: command not found", "fehlt"),
    ("Claude AI usage limit reached|1795500000", "kontingent"),
    ("Invalid API key · Please run /login", "fehlt"),
    ("irgendwas anderes", "aussetzer"),
])
def test_cli_fehler_mit_festem_code(monkeypatch, tmp_path, stderr, art):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "v"))
    monkeypatch.setattr(t, "_run_cli_with_pgid",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", stderr))
    with pytest.raises(HTTPException) as e:
        asyncio.run(t._claude_mit_rechten(t.Prompt(prompt="x"), "haiku", direkt=True))
    assert k.einordnen(e.value)[0] == art
