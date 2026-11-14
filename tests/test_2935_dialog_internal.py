"""#2935 (Befund 56): Pfadtraversierung als root in /ai/dialog/*, offenes
/internal/notify-cli-update, Stored XSS auf /ai/status."""
import json
import os

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ai.routes import dialog_routes as d
from ai.routes import internal_routes as ir
from ai.routes import dashboard_routes as db

SCHLUESSEL = "betreiber-test"


@pytest.fixture
def app(monkeypatch, tmp_path):
    status = tmp_path / "dialog_status"
    status.mkdir()
    monkeypatch.setattr(d, "STATUS_DIR", str(status))
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", SCHLUESSEL)
    for reg in (d.TEMP_DIALOG_REGISTRY, d.TEMP_DIALOG_STATUS, d.TEMP_DIALOG_RESULTS, d.TEMP_DIALOG_HISTORY):
        reg.clear()
    from ai.services import bild_verbrauch as bv
    jwts = {"Bearer jwt-a": {"sub": "agent:A"}, "Bearer jwt-b": {"sub": "agent:B"}}
    monkeypatch.setattr(bv, "_claims_geprueft", lambda a: jwts.get(a))
    a = FastAPI()
    a.include_router(d.router, prefix="/ai/dialog")
    a.include_router(ir.router, prefix="/internal")
    a.include_router(db.router, prefix="/ai")
    a.state.tmp = tmp_path
    return TestClient(a)


GEHEIM = "../../geheim"


def _geheim(tmp_path):
    (tmp_path / "geheim.json").write_text(json.dumps({"token": "SEHR-GEHEIM"}))
    (tmp_path / "geheim.chunks.json").write_text("[]")
    return tmp_path / "geheim.json"


@pytest.mark.parametrize("methode,pfad", [
    ("get", "/ai/dialog/status"), ("post", "/ai/dialog/clear_temp"),
    ("get", "/ai/dialog/tempchunk"), ("post", "/ai/dialog/cancel"),
])
def test_dialog_ohne_anmeldung_401(app, methode, pfad):
    r = getattr(app, methode)(pfad, params={"id": "abc", "index": 0})
    assert r.status_code == 401


def test_traversierung_liest_nichts(app):
    datei = _geheim(app.app.state.tmp)
    r = app.get("/ai/dialog/status", params={"id": GEHEIM}, headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 422
    assert "SEHR-GEHEIM" not in r.text
    assert datei.exists()


def test_traversierung_loescht_nichts(app):
    datei = _geheim(app.app.state.tmp)
    r = app.post("/ai/dialog/clear_temp", params={"id": GEHEIM}, headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 422
    assert datei.exists() and (app.app.state.tmp / "geheim.chunks.json").exists()


@pytest.mark.parametrize("schlecht", ["../x", "a/b", "a" * 65, "", "a.b", "x\x00y", "%2e%2e"])
def test_kennung_streng(schlecht):
    with pytest.raises(HTTPException):
        d._status_pfad(schlecht, ".json")


def test_hilfsfunktionen_schreiben_nicht_ausserhalb(app, tmp_path):
    d.set_dialog_status(GEHEIM, phase="x")
    d.register_temp_dialog_chunks(GEHEIM, [{"index": 0, "path": "/etc/passwd"}])
    assert not (tmp_path / "geheim.json").exists()
    assert not (tmp_path / "geheim.chunks.json").exists()


def test_fremder_auftrag_ist_unsichtbar_eigener_sichtbar(app):
    d._besitzer_setzen("job1", "A")
    d.set_dialog_status("job1", phase="queued")
    fremd = app.get("/ai/dialog/status", params={"id": "job1"}, headers={"Authorization": "Bearer jwt-b"})
    assert fremd.status_code == 404
    eigen = app.get("/ai/dialog/status", params={"id": "job1"}, headers={"Authorization": "Bearer jwt-a"})
    assert eigen.status_code == 200 and eigen.json()["phase"] == "queued"
    betreiber = app.get("/ai/dialog/status", params={"id": "job1"}, headers={"X-API-KEY": SCHLUESSEL})
    assert betreiber.status_code == 200
    assert app.post("/ai/dialog/clear_temp", params={"id": "job1"},
                    headers={"Authorization": "Bearer jwt-b"}).status_code == 404


def test_kennung_kann_nicht_uebernommen_werden(app):
    d._besitzer_setzen("job2", "A")
    with pytest.raises(HTTPException) as e:
        d._besitzer_setzen("job2", "B")
    assert e.value.status_code == 409


def test_tempchunk_folgt_keinem_symlink_aus_tmp(app, tmp_path, monkeypatch):
    d._besitzer_setzen("job3", "A")
    ziel = tmp_path / "ausserhalb.mp3"
    ziel.write_bytes(b"x")
    import tempfile
    link = os.path.join(tempfile.gettempdir(), f"dlg_link_{os.getpid()}.mp3")
    try:
        if os.path.lexists(link):
            os.remove(link)
        os.symlink("/etc/hostname", link)
        d.register_temp_dialog_chunks("job3", [{"index": 0, "path": link}])
        r = app.get("/ai/dialog/tempchunk", params={"id": "job3", "index": 0},
                    headers={"Authorization": "Bearer jwt-a"})
        assert r.status_code == 403
    finally:
        if os.path.lexists(link):
            os.remove(link)


def test_notify_cli_update_ohne_schluessel_401_und_nichts_geschrieben(app, monkeypatch):
    geschrieben, gestartet = [], []
    monkeypatch.setattr(ir, "_write_curated_models", lambda h, m: geschrieben.append(h))
    monkeypatch.setattr(ir, "_spawn", lambda c: gestartet.append(c))
    r = app.post("/internal/notify-cli-update", json={"host": "x", "models": {"claude": {}}})
    assert r.status_code == 401
    r = app.post("/internal/notify-cli-update", json={"host": "x"}, headers={"X-API-KEY": "falsch"})
    assert r.status_code == 401
    assert geschrieben == [] and gestartet == []
    r = app.post("/internal/notify-cli-update", json={"host": "x", "models": {"claude": {}}},
                 headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200 and geschrieben == ["x"]


def test_status_seite_escaped_katalog(monkeypatch):
    karte = db._provider_card("<b>p</b>", {
        "available": ['<img src=x onerror=alert(1)>'], "default": "<script>x</script>",
        "cli_version_recorded": "1<i>", "cli_version_live": "2", "discovery_method": "<u>m</u>"})
    assert "<img" not in karte and "<script>" not in karte and "<b>p" not in karte
    assert "&lt;img src=x onerror=alert(1)&gt;" in karte
