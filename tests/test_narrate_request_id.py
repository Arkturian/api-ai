"""narrate request_id: dauerhafter Status, payloadgebundene Idempotenz.

Vertrag: Content #4976, p-b12b0d4d6062 (Story-Codex, 13.09.). Der Fall,
der alles motiviert: Client-Timeout, Server spricht zu Ende — der Client
darf NIE ein zweites Mal bezahlen, ohne dass ein Mensch es entscheidet.

Ziel: done wird ohne Anbieteraufruf wiederholt. Gegenfall: running wird
nicht zweimal gesprochen (409), failed nach dem Sprechen auch nicht.
Nachbarfall: failed VOR dem Sprechen darf neu; verwaistes running darf
neu und sagt es; andere Nutzlast unter derselben Kennung ist ein Fehler.
"""

import json
import types
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from ai.routes import narration_routes as r
from ai.services import narrate_jobs as nj
from ai.services import narration_service as n


@pytest.fixture(autouse=True)
def _ablage(monkeypatch, tmp_path):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))


def _req(rid="story-szene-1:cue-3", text="Mira trat durch das Tor."):
    return n.NarrationRequest(
        text=text,
        character=n.NarrationCharacter(name="Erz", voice_id="v"),
        config=n.NarrationConfig(preprocessing=False),
        save_options={},
        request_id=rid,
    )


def _antwort(text="Mira trat durch das Tor."):
    return n.NarrationResponse(audio_id=4711, audio_url="https://s/4711", duration_seconds=1.2,
                               dramatic_script=text, original_text=text, saved=True)


def _service_mit(monkeypatch, verhalten):
    """verhalten: async callable(req) -> NarrationResponse oder raise."""
    gezaehlt = {"aufrufe": 0}

    async def generate(self, req):
        gezaehlt["aufrufe"] += 1
        return await verhalten(req)

    monkeypatch.setattr(n.NarrationService, "generate", generate)
    return gezaehlt


# ---------------------------------------------------- Hash

def test_hash_bindet_inhalt_nicht_speicheroption():
    a, b = _req(), _req()
    b.save_options = {"is_public": False}
    assert nj.payload_hash(a) == nj.payload_hash(b)
    c = _req(text="Anderer Text.")
    assert nj.payload_hash(a) != nj.payload_hash(c)


def test_kennung_form():
    assert nj.request_id_ok("story-szene-1:cue-3")
    assert not nj.request_id_ok("kurz")
    assert not nj.request_id_ok("mit leerzeichen und mehr")
    assert not nj.request_id_ok("x" * 129)


# ---------------------------------------------------- Ziel: Replay

@pytest.mark.asyncio
async def test_done_wird_ohne_anbieteraufruf_wiederholt(monkeypatch):
    async def ok(req):
        return _antwort()
    z = _service_mit(monkeypatch, ok)
    erste = await r.narrate(_req(), api_key="x")
    assert erste.replayed is False and erste.request_id == "story-szene-1:cue-3"
    zweite = await r.narrate(_req(), api_key="x")
    assert zweite.replayed is True and zweite.audio_id == 4711
    assert z["aufrufe"] == 1                      # nur einmal gesprochen
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "done" and st["result"]["audio_id"] == 4711


# ---------------------------------------------------- Gegenfaelle

@pytest.mark.asyncio
async def test_running_wird_nicht_zweimal_gesprochen(monkeypatch):
    nj.anlegen("story-szene-1:cue-3", nj.payload_hash(_req()))
    async def darf_nicht(req):
        raise AssertionError("zweites Sprechen")
    _service_mit(monkeypatch, darf_nicht)
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(), api_key="x")
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "narration_in_progress"
    assert exc.value.headers.get("Retry-After") == "5"


@pytest.mark.asyncio
async def test_fehler_nach_dem_sprechen_wird_nicht_automatisch_wiederholt(monkeypatch):
    async def bricht_beim_speichern(req):
        nj.stufe(req.request_id, "save")       # Sprechen war schon durch
        raise RuntimeError("storage down")
    z = _service_mit(monkeypatch, bricht_beim_speichern)
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(), api_key="x")
    assert exc.value.status_code == 500
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "failed" and st["failed_stage"] == "save"
    with pytest.raises(HTTPException) as exc2:
        await r.narrate(_req(), api_key="x")
    assert exc2.value.status_code == 409
    assert exc2.value.detail["error"] == "narration_failed_after_tts"
    assert z["aufrufe"] == 1


@pytest.mark.asyncio
async def test_andere_nutzlast_unter_derselben_kennung_ist_ein_fehler(monkeypatch):
    async def ok(req):
        return _antwort()
    _service_mit(monkeypatch, ok)
    await r.narrate(_req(), api_key="x")
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(text="Ganz anderer Satz."), api_key="x")
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "request_id_payload_mismatch"


# ---------------------------------------------------- Nachbarfaelle

@pytest.mark.asyncio
async def test_fehler_vor_dem_sprechen_darf_neu(monkeypatch):
    laeufe = {"n": 0}
    async def erst_gesperrt_dann_ok(req):
        laeufe["n"] += 1
        if laeufe["n"] == 1:
            nj.stufe(req.request_id, "tts")
            raise HTTPException(status_code=429, detail={"error": "monthly_api_cap_reached"})
        return _antwort()
    _service_mit(monkeypatch, erst_gesperrt_dann_ok)
    with pytest.raises(HTTPException):
        await r.narrate(_req(), api_key="x")
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "failed" and st["failed_stage"] == "pre_tts"
    zweite = await r.narrate(_req(), api_key="x")
    assert zweite.replayed is False and laeufe["n"] == 2


@pytest.mark.asyncio
async def test_verwaistes_running_sagt_es_und_darf_neu(monkeypatch):
    nj.anlegen("story-szene-1:cue-3", nj.payload_hash(_req()))
    d = nj.lesen("story-szene-1:cue-3")
    d["updated_at"] = (datetime.now() - timedelta(minutes=11)).isoformat()
    (nj.jobs_dir() / "story-szene-1:cue-3.json").write_text(json.dumps(d))
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "running" and st["stale"] is True
    async def ok(req):
        return _antwort()
    z = _service_mit(monkeypatch, ok)
    await r.narrate(_req(), api_key="x")
    assert z["aufrufe"] == 1


@pytest.mark.asyncio
async def test_status_unbekannt_und_preview_sind_404():
    for rid in ("gibt-es-nicht-1", "preview", "kurz"):
        with pytest.raises(HTTPException) as exc:
            await r.narrate_status(rid)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_ungueltige_kennung_422(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(rid="kurz"), api_key="x")
    assert exc.value.status_code == 422 and exc.value.detail["error"] == "invalid_request_id"


@pytest.mark.asyncio
async def test_ohne_kennung_wie_bisher(monkeypatch):
    async def ok(req):
        return _antwort()
    z = _service_mit(monkeypatch, ok)
    a = _req(rid=None)
    await r.narrate(a, api_key="x")
    await r.narrate(a, api_key="x")
    assert z["aufrufe"] == 2 and not list(nj.jobs_dir().glob("*.json")) if nj.jobs_dir().exists() else True
