"""#2936 (Befund 57): lokale image_paths in /ai/chatgpt und /ai/claude,
Werkzeuge/Sandbox als Vorgabe, agy als root mit skip-permissions,
anonymer Upload ueber /ai/gemini/vision, persona_variant-Traversal."""
import asyncio
import base64
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t

SCHLUESSEL = "betreiber-2936"


class Fang:
    def __init__(self):
        self.cmd = None
        self.env = None

    def __call__(self, cmd, env=None, **k):
        self.cmd, self.env = list(cmd), dict(env or {})
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")


@pytest.fixture
def fang(monkeypatch, tmp_path):
    f = Fang()
    monkeypatch.setattr(t, "_run_cli_with_pgid", f)
    monkeypatch.setattr(t, "_download_storage_images", lambda _p: [])
    monkeypatch.setenv("CLI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", SCHLUESSEL)
    monkeypatch.delenv("GEMINI_FORWARD_URL", raising=False)
    monkeypatch.delenv("CLAUDE_ROUTE", raising=False)
    monkeypatch.delenv("AGY_RUN_AS_USER", raising=False)
    return f


# 1. lokale image_paths ------------------------------------------------------

@pytest.mark.parametrize("pfad", ["/ai/chatgpt", "/ai/claude"])
def test_lokale_pfade_von_aussen_403(fang, pfad):
    r = TestClient(main.app).post(pfad, json={"prompt": "x", "image_paths": ["/etc/shadow"]},
                                  headers={"X-Real-IP": "1.2.3.4"})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "local_paths_not_allowed"
    assert fang.cmd is None


@pytest.mark.parametrize("pfad", ["/ai/chatgpt", "/ai/claude"])
def test_lokale_pfade_auch_lokal_nur_medien_unter_wurzeln(fang, monkeypatch, tmp_path, pfad):
    monkeypatch.setattr(t, "_darf_lokale_pfade", lambda r: True)
    wurzel = tmp_path / "disk"
    wurzel.mkdir()
    (wurzel / "notiz.txt").write_text("x")
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(wurzel))
    c = TestClient(main.app)
    for bild, grund in (("/etc/shadow", "path_outside_allowed_roots"),
                        (str(wurzel / "notiz.txt"), "not_a_media_file"),
                        (str(wurzel / "fehlt.jpg"), "file_not_found")):
        r = c.post(pfad, json={"prompt": "x", "image_paths": [bild]})
        assert r.status_code == 422, r.text
        assert grund in r.json()["detail"]["reason"]
    assert fang.cmd is None


def test_root_kopie_nur_fuer_erlaubte_dateien(monkeypatch, tmp_path):
    import getpass
    import pwd
    fremd = "root" if getpass.getuser() != "root" else "nobody"
    monkeypatch.setenv("CODEX_RUN_AS_USER", fremd)
    monkeypatch.setenv("CODEX_VISION_DIR", str(tmp_path / "vision"))
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(tmp_path / "disk"))
    monkeypatch.setattr(os, "chown", lambda *a, **k: None)
    geheim = tmp_path / "geheim.jpg"
    geheim.write_bytes(b"x")
    with pytest.raises(HTTPException) as e:
        t.bilder_fuer_codex_benutzer([str(geheim)])
    assert e.value.status_code == 422
    assert not any((tmp_path / "vision").glob("aiimg_*"))


# 2. Werkzeuge und Sandbox -----------------------------------------------------

def test_tools_ohne_betreiber_403(fang):
    r = TestClient(main.app).post("/ai/chatgpt", json={"prompt": "x", "tools": True})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "tools_require_operator_key"
    assert fang.cmd is None


def test_tools_mit_betreiber_laufen(fang):
    TestClient(main.app).post("/ai/chatgpt", json={"prompt": "x", "tools": True},
                              headers={"X-API-KEY": SCHLUESSEL})
    assert fang.cmd is not None and "features.shell_tool=false" not in " ".join(fang.cmd)


def test_vorgabe_ohne_werkzeuge_und_read_only(fang):
    TestClient(main.app).post("/ai/chatgpt", json={"prompt": "x"})
    kette = " ".join(fang.cmd)
    assert "-s read-only" in kette and "dangerously-bypass" not in kette
    assert "features.shell_tool=false" in kette


def test_volle_sandbox_ohne_betreiber_403(fang):
    r = TestClient(main.app).post("/ai/chatgpt", json={"prompt": "x", "sandbox": "voll"})
    assert r.status_code == 403
    assert fang.cmd is None


def test_claude_ueber_codex_erbt_die_rechte(fang, monkeypatch):
    """claude->codex ruft chatgpt_endpoint als Funktion (request=None):
    das darf die Rechte des aeusseren Aufrufs nicht auf 'lokal' heben."""
    monkeypatch.setenv("CLAUDE_ROUTE", "codex")
    r = TestClient(main.app).post("/ai/claude", json={"prompt": "x", "tools": True})
    assert r.status_code == 403
    assert fang.cmd is None


# 3. agy ----------------------------------------------------------------------

def test_agy_ohne_skip_permissions_mit_sandbox(fang):
    TestClient(main.app).post("/ai/gemini", json={"prompt": "hallo"})
    assert "--dangerously-skip-permissions" not in fang.cmd
    assert "--sandbox" in fang.cmd


def test_agy_laeuft_als_benutzer_ohne_schluessel(fang, monkeypatch, tmp_path):
    home = tmp_path / "alexhome"
    home.mkdir()
    monkeypatch.setattr(t, "_agy_benutzer", lambda: ("alex", str(home)))
    monkeypatch.setattr(t, "_agy_arbeitsbereich", lambda h, u: str(home))
    monkeypatch.setenv("GOOGLE_API_KEY", "geheim")
    TestClient(main.app).post("/ai/gemini", json={"prompt": "hallo"})
    assert fang.cmd[:5] == ["sudo", "-n", "-u", "alex", "-H"]
    vor_agy = fang.cmd[:fang.cmd.index("--output-format")]
    assert not any("GOOGLE" in x or "GEMINI_API" in x for x in vor_agy)
    assert f"HOME={home}" in vor_agy


# 4. /ai/gemini/vision ---------------------------------------------------------

def _png():
    return "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode()


def test_vision_ohne_anmeldung_401(fang):
    r = TestClient(main.app).post("/ai/gemini/vision", json={"prompt": {"text": "x", "images": [_png()]}})
    assert r.status_code == 401


@pytest.mark.parametrize("bild,fehler", [
    ("data:text/html;base64," + base64.b64encode(b"<script>").decode(), "vision_mime_not_allowed"),
    ("data:image/png;base64," + base64.b64encode(b"<html>kein bild").decode(), "vision_not_an_image"),
])
def test_vision_nur_echte_bilder(fang, bild, fehler):
    r = TestClient(main.app).post("/ai/gemini/vision", json={"prompt": {"text": "x", "images": [bild]}},
                                  headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 422 and r.json()["detail"]["error"] == fehler


# 5. persona_variant -------------------------------------------------------------

@pytest.mark.parametrize("schlecht", ["../../etc", "a/b", "A", "x" * 40, "default?x=1", "%2e%2e"])
def test_persona_variant_nur_name(fang, schlecht):
    r = TestClient(main.app).post("/ai/chatgpt", json={"prompt": "x", "persona_variant": schlecht})
    assert r.status_code == 422
    assert fang.cmd is None


def test_persona_gut_bleibt_erlaubt():
    assert t.Prompt(prompt="x", persona_variant="default").persona_variant == "default"
    with pytest.raises(ValueError):
        t.persona_dir_for("claude", "../x")
    assert asyncio.run(t.get_persona_bundle("chatgpt", "../x")) == {"rendered": None, "allowed_tools": []}
