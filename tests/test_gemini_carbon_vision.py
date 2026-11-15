"""/ai/gemini fuer Storage-Analysen ueber Gemini 4 Carbon (Alex 14.11.2026,
Post 6000): lokale Pfade nur von demselben Host und nur Medien unter den
erlaubten Wurzeln, Carbon als Vorgabe fuer Vision, und nie ein Google-
Schluessel in der Umgebung von agy (sonst kippt es in den bezahlten Modus)."""
import asyncio
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t


class Fang:
    def __init__(self):
        self.cmd = None
        self.env = None
        self.dateien_waehrend = []

    def __call__(self, cmd, env=None, **k):
        self.cmd, self.env = list(cmd), dict(env or {})
        self.eingabe = k.get("input")
        p = self.eingabe or ""
        self.dateien_waehrend = [w[1:] for w in p.split() if w.startswith("@")]
        self.existierten = [os.path.exists(d) for d in self.dateien_waehrend]
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")


@pytest.fixture
def fang(monkeypatch, tmp_path):
    f = Fang()
    monkeypatch.setattr(t, "_run_cli_with_pgid", f)
    monkeypatch.setattr(t, "_download_storage_images", lambda _p: [])
    monkeypatch.setenv("CLI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GEMINI_FORWARD_URL", raising=False)
    monkeypatch.delenv("GEMINI_VISION_MODEL", raising=False)
    return f


def _modell(cmd):
    return cmd[cmd.index("--model") + 1] if "--model" in cmd else None


def test_agy_bekommt_keinen_google_schluessel(fang, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "geheim")
    monkeypatch.setenv("GEMINI_API_KEY", "geheim2")
    TestClient(main.app).post("/ai/gemini", json={"prompt": "hallo"})
    assert fang.env is not None
    assert "GOOGLE_API_KEY" not in fang.env and "GEMINI_API_KEY" not in fang.env


def test_vision_ohne_modell_nimmt_carbon(fang):
    TestClient(main.app).post("/ai/gemini", json={
        "prompt": "Was zeigt das?", "image_paths": ["https://api-storage.arkturian.com/storage/media/1"]})
    assert _modell(fang.cmd) == "gemini-4-carbon-medium"


def test_url_im_prompt_reicht_fuer_carbon(fang):
    TestClient(main.app).post("/ai/gemini", json={
        "prompt": "Analysiere https://api-storage.arkserver.arkturian.com/storage/media/5?width=600&format=jpg"})
    assert _modell(fang.cmd) == "gemini-4-carbon-medium"


def test_text_ohne_bild_bleibt_ohne_modell(fang):
    TestClient(main.app).post("/ai/gemini", json={"prompt": "hallo"})
    assert _modell(fang.cmd) is None


def test_ausdruecklicher_modellwunsch_gewinnt(fang):
    TestClient(main.app).post("/ai/gemini?model=gemini-3.8-flash-high", json={
        "prompt": "x", "image_paths": ["https://api-storage.arkturian.com/storage/media/1"]})
    assert _modell(fang.cmd) == "gemini-3.8-flash-high"


def test_lokale_pfade_von_aussen_403(fang, tmp_path):
    bild = tmp_path / "a.jpg"
    bild.write_bytes(b"JPG")
    r = TestClient(main.app).post("/ai/gemini", json={"prompt": "x", "image_paths": [str(bild)]},
                                  headers={"X-Real-IP": "1.2.3.4"})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "local_paths_not_allowed"
    assert fang.cmd is None


@pytest.mark.parametrize("host,kopf,erwartet", [
    ("127.0.0.1", {}, True), ("::1", {}, True),
    ("127.0.0.1", {"x-real-ip": "1.2.3.4"}, False),
    ("127.0.0.1", {"x-forwarded-for": "1.2.3.4"}, False),
    ("8.8.8.8", {}, False), ("testclient", {}, False),
])
def test_wer_lokale_pfade_darf(host, kopf, erwartet):
    req = SimpleNamespace(client=SimpleNamespace(host=host), headers=kopf)
    assert t._darf_lokale_pfade(req) is erwartet


def _direkt(pfade):
    from ai.routes.text_ai_routes import Prompt
    return asyncio.run(t.gemini_endpoint(Prompt(prompt="Analysiere", image_paths=pfade), model=None,
                                         api_key="x", request=None))


def test_lokales_bild_wird_in_den_arbeitsbereich_kopiert_und_danach_geloescht(fang, monkeypatch, tmp_path):
    wurzel = tmp_path / "disk"
    wurzel.mkdir()
    bild = wurzel / "frame_10.jpg"
    bild.write_bytes(b"JPG")
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(wurzel))
    with pytest.raises(HTTPException):
        _direkt([str(bild)])  # Fang liefert returncode 1
    assert len(fang.dateien_waehrend) == 1
    kopie = fang.dateien_waehrend[0]
    assert kopie.startswith(str(tmp_path / "home" / ".aiapi-neutral" / ".vision"))
    assert fang.existierten == [True] and not os.path.exists(kopie)
    assert str(bild) not in fang.eingabe
    assert _modell(fang.cmd) == "gemini-4-carbon-medium"


@pytest.mark.parametrize("name,inhalt,grund", [
    ("ausserhalb", None, "path_outside_allowed_roots"),
    ("notiz.txt", b"x", "not_a_media_file"),
    ("fehlt.jpg", None, "file_not_found"),
])
def test_lokale_pfade_werden_geprueft(fang, monkeypatch, tmp_path, name, inhalt, grund):
    wurzel = tmp_path / "disk"
    wurzel.mkdir()
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(wurzel))
    if name == "ausserhalb":
        p = tmp_path / "geheim.jpg"
        p.write_bytes(b"x")
    else:
        p = wurzel / name
        if inhalt is not None:
            p.write_bytes(inhalt)
    with pytest.raises(HTTPException) as e:
        _direkt([str(p)])
    assert e.value.status_code == 422 and grund in e.value.detail["reason"]
    assert fang.cmd is None


def test_symlink_aus_der_wurzel_heraus_zaehlt_als_ausserhalb(fang, monkeypatch, tmp_path):
    wurzel = tmp_path / "disk"
    wurzel.mkdir()
    geheim = tmp_path / "geheim.jpg"
    geheim.write_bytes(b"x")
    (wurzel / "link.jpg").symlink_to(geheim)
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(wurzel))
    with pytest.raises(HTTPException) as e:
        _direkt([str(wurzel / "link.jpg")])
    assert "path_outside_allowed_roots" in e.value.detail["reason"]


def test_prompt_geht_ueber_stdin_nicht_in_argv(fang):
    """#2937: sudo protokolliert argv ins Journal - der Kundentext gehoert nicht dorthin."""
    TestClient(main.app).post("/ai/gemini", json={"prompt": "GEHEIMER-KUNDENTEXT-4711"})
    assert fang.cmd is not None
    assert "-p" not in fang.cmd and "--print" not in fang.cmd
    assert not any("GEHEIMER-KUNDENTEXT-4711" in a for a in fang.cmd)
    assert "GEHEIMER-KUNDENTEXT-4711" in fang.eingabe
