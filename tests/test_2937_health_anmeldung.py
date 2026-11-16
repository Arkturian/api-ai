"""#2937: Als apiai ist /home/alex 0700. /health meldete dann ok=false,
reason=absent fuer eine Anmeldung, die da ist (os.path.exists schweigt bei
fehlendem Leserecht)."""
import os

import main


def test_nicht_lesbar_ist_nicht_fehlend(monkeypatch, tmp_path):
    zu = tmp_path / "zu"
    (zu / ".codex").mkdir(parents=True)
    (zu / ".codex" / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(zu / ".codex"))
    echt = os.stat

    def stat(p, *a, **k):
        if str(p).startswith(str(zu)):
            raise PermissionError(13, "Permission denied")
        return echt(p, *a, **k)

    monkeypatch.setattr(main.os, "stat", stat)
    monkeypatch.setattr(main.os.path, "exists", lambda p: False if str(p).startswith(str(zu)) else os.path.lexists(p))
    s = main._codex_auth_status()
    assert s["ok"] is None and s["reason"] == "not_readable_by_service"


def test_wirklich_fehlend_bleibt_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "leer"))
    s = main._codex_auth_status()
    assert s["ok"] is False and s["reason"] == "absent"
