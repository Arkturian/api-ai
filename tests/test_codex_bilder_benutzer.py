"""Bilder fuer codex, wenn codex als anderer Benutzer laeuft (11.10.).

Gegen den echten Fehler gehalten: der Dienst (root) legte das Bild unter
/root/... ab, codex lief als alex und konnte `-i <pfad>` nicht lesen. Das
Modell meldete "kein Bild", der Dienst "attached 1 image(s)".
"""
import os
import stat
import types

import pytest

from ai.routes import text_ai_routes as t


def _bild(tmp_path, name="bild.jpg"):
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0jpeg")
    return str(p)


def _fremder_benutzer(monkeypatch, tmp_path):
    """Zielbenutzer mit anderer uid als der Prozess; chown wird gezaehlt."""
    ziel = tmp_path / "vision-ziel"
    monkeypatch.setenv("CODEX_RUN_AS_USER", "zielnutzer")
    monkeypatch.setenv("CODEX_VISION_DIR", str(ziel))
    import pwd
    monkeypatch.setattr(pwd, "getpwnam", lambda n: types.SimpleNamespace(pw_uid=os.geteuid() + 1, pw_gid=4242))
    chowns = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: chowns.append((str(p), u, g)))
    return ziel, chowns


def test_ohne_variable_bleiben_pfade_unveraendert(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_RUN_AS_USER", raising=False)
    b = _bild(tmp_path)
    assert t.bilder_fuer_codex_benutzer([b]) == ([b], [])


def test_gleicher_benutzer_keine_kopie(monkeypatch, tmp_path):
    import pwd
    monkeypatch.setenv("CODEX_RUN_AS_USER", pwd.getpwuid(os.geteuid()).pw_name)
    b = _bild(tmp_path)
    assert t.bilder_fuer_codex_benutzer([b]) == ([b], [])


def test_fremder_benutzer_bekommt_private_kopie(monkeypatch, tmp_path):
    ziel, chowns = _fremder_benutzer(monkeypatch, tmp_path)
    b = _bild(tmp_path)
    neu, kopien = t.bilder_fuer_codex_benutzer([b, "https://example.org/x.png"])
    assert neu[1] == "https://example.org/x.png"          # URLs bleiben stehen
    assert neu[0] != b and neu[0].startswith(str(ziel)) and neu[0].endswith(".jpg")
    assert kopien == [neu[0]]
    assert open(neu[0], "rb").read() == open(b, "rb").read()
    assert stat.S_IMODE(os.stat(ziel).st_mode) == 0o700     # nur der Zielbenutzer
    assert stat.S_IMODE(os.stat(neu[0]).st_mode) == 0o600
    uid = os.geteuid() + 1
    assert (str(ziel), uid, 4242) in chowns and (neu[0], uid, 4242) in chowns


def test_unbekannter_benutzer_wird_laut(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_RUN_AS_USER", "gibt-es-nicht-xyz")
    with pytest.raises(RuntimeError):
        t.bilder_fuer_codex_benutzer([_bild(tmp_path)])


def test_kopierfehler_wird_nicht_verschluckt(monkeypatch, tmp_path):
    _fremder_benutzer(monkeypatch, tmp_path)
    def kaputt(p, u, g):
        raise PermissionError("chown")
    monkeypatch.setattr(os, "chown", kaputt)
    with pytest.raises(PermissionError):
        t.bilder_fuer_codex_benutzer([_bild(tmp_path)])
