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

def test_hash_bindet_inhalt_UND_speicheroption():
    """Review q-d98ba93140f0, Punkt 2: dieselbe Kennung mit anderer
    Sichtbarkeit bekaeme sonst beim Replay die falsche Datei."""
    a, b = _req(), _req()
    b.save_options = {"is_public": False}
    assert nj.payload_hash(a) != nj.payload_hash(b)
    c = _req(text="Anderer Text.")
    assert nj.payload_hash(a) != nj.payload_hash(c)
    d = _req(); d.respeak_stale = True
    assert nj.payload_hash(a) == nj.payload_hash(d)      # Steuerung, kein Inhalt


def test_kennung_form():
    assert nj.request_id_ok("story-szene-1:cue-3")
    assert not nj.request_id_ok("kurz")
    assert not nj.request_id_ok("mit leerzeichen und mehr")
    assert not nj.request_id_ok("x" * 129)
    # Pfadschutz (Review Punkt 3): kein fuehrender Punkt, kein "..", kein Trenner
    assert not nj.request_id_ok("..abcdefgh")
    assert not nj.request_id_ok(".versteckt1")
    assert not nj.request_id_ok("a/b/cdefghij")
    assert not nj.request_id_ok("ab..cdefghij")


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


def _verwaist(rid="story-szene-1:cue-3"):
    nj.anlegen(rid, nj.payload_hash(_req()))
    d = nj.lesen(rid)
    d["updated_at"] = (datetime.now() - timedelta(minutes=11)).isoformat()
    (nj.jobs_dir() / f"{rid}.json").write_text(json.dumps(d))


@pytest.mark.asyncio
async def test_verwaistes_running_spricht_NIE_von_selbst_neu(monkeypatch):
    """Review Punkt 1: 10 Minuten beweisen weder Abschluss noch
    Nichtabrechnung. Ein nackter POST bleibt 409 — mit `stale: true`."""
    _verwaist()
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "running" and st["stale"] is True
    async def darf_nicht(req):
        raise AssertionError("automatisch neu gesprochen")
    _service_mit(monkeypatch, darf_nicht)
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(), api_key="x")
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "narration_in_progress" and exc.value.detail["stale"] is True


@pytest.mark.asyncio
async def test_neusprechen_nach_stale_nur_ausdruecklich(monkeypatch):
    _verwaist()
    async def ok(req):
        return _antwort()
    z = _service_mit(monkeypatch, ok)
    req = _req(); req.respeak_stale = True
    a = await r.narrate(req, api_key="x")
    assert z["aufrufe"] == 1 and a.replayed is False
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["state"] == "done" and st.get("respoken_from_stale") is True


def test_reservierung_ist_atomar():
    """Review Punkt 3: O_EXCL — genau einer gewinnt."""
    h = nj.payload_hash(_req())
    assert nj.reservieren("story-szene-1:cue-3", h) is not None
    assert nj.reservieren("story-szene-1:cue-3", h) is None


def test_aufraeumen_laesst_grabstein(monkeypatch):
    """Review Punkt 1: nach 30 Tagen faellt das Ergebnis, nicht die Kennung."""
    import os, time
    nj.reservieren("story-szene-1:cue-3", "h")
    nj.abschliessen("story-szene-1:cue-3", {"audio_id": 1})
    p = nj.jobs_dir() / "story-szene-1:cue-3.json"
    alt = time.time() - 31 * 86400
    os.utime(p, (alt, alt))
    assert nj.aufraeumen() == 1
    d = nj.lesen("story-szene-1:cue-3")
    assert d["tombstone"] is True and "result" not in d


@pytest.mark.asyncio
async def test_grabstein_gibt_410(monkeypatch):
    nj.reservieren("story-szene-1:cue-3", nj.payload_hash(_req()))
    d = nj.lesen("story-szene-1:cue-3"); d.update({"state": "done", "tombstone": True}); nj.schreiben(d)
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(), api_key="x")
    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_fehler_in_der_aufbereitung_ist_stufe_prepare(monkeypatch):
    """Review Punkt 4: pre_tts liegt VOR der dramaturgischen Aufbereitung."""
    async def bricht_in_prepare(req):
        nj.stufe(req.request_id, "prepare")
        raise RuntimeError("LLM down")
    _service_mit(monkeypatch, bricht_in_prepare)
    with pytest.raises(HTTPException):
        await r.narrate(_req(), api_key="x")
    st = await r.narrate_status("story-szene-1:cue-3")
    assert st["failed_stage"] == "prepare"
    # prepare gilt als vor dem Sprechen: neu erlaubt
    async def ok(req):
        return _antwort()
    z = _service_mit(monkeypatch, ok)
    await r.narrate(_req(), api_key="x")
    assert z["aufrufe"] == 1


@pytest.mark.asyncio
async def test_mix_kennung_ist_im_sprech_status_unbekannt():
    nj.reservieren("story-szene-1:cue-3", "h", kind="mix")
    with pytest.raises(HTTPException) as exc:
        await r.narrate_status("story-szene-1:cue-3")
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc2:
        await r.narrate(_req(), api_key="x")
    assert exc2.value.detail["error"] == "request_id_belongs_to_other_endpoint"


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


# ---------------------------------------------------- Restpunkte q-d98ba93140f0

def test_reservierung_schreibt_inhalt_atomar():
    """Restpunkt 1: nie eine leere Datei sichtbar. Der Verlierer liest
    sofort einen vollstaendigen Eintrag."""
    h = nj.payload_hash(_req())
    assert nj.reservieren("story-szene-1:cue-3", h) is not None
    assert nj.reservieren("story-szene-1:cue-3", h) is None
    d = nj.lesen("story-szene-1:cue-3")
    assert d and d["state"] == "running" and d["payload_hash"] == h


def test_gleichzeitige_erstansprueche_genau_einer_gewinnt():
    import threading
    h = nj.payload_hash(_req()); gewinner = []
    def lauf():
        if nj.reservieren("story-szene-1:cue-3", h) is not None:
            gewinner.append(1)
    ts = [threading.Thread(target=lauf) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(gewinner) == 1


def test_gleichzeitige_wiederansprueche_genau_einer_gewinnt():
    h = nj.payload_hash(_req())
    nj.reservieren("story-szene-1:cue-3", h)
    nj.scheitern("story-szene-1:cue-3", 429, {"error": "cap"})
    import threading
    gewinner = []
    def lauf():
        if nj.uebernehmen("story-szene-1:cue-3", h, "narrate", lambda a: a.get("state") == "failed"):
            gewinner.append(1)
    ts = [threading.Thread(target=lauf) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(gewinner) == 1


@pytest.mark.asyncio
async def test_verlorener_anspruch_ohne_eintrag_ist_409(monkeypatch):
    """Restpunkt 1, der Kern: Anspruch verloren und (noch) nichts lesbar
    -> nie sprechen."""
    monkeypatch.setattr(nj, "reservieren", lambda *a, **k: None)
    monkeypatch.setattr(nj, "lesen", lambda rid: None)
    async def darf_nicht(req):
        raise AssertionError("gesprochen")
    _service_mit(monkeypatch, darf_nicht)
    with pytest.raises(HTTPException) as exc:
        await r.narrate(_req(), api_key="x")
    assert exc.value.status_code == 409 and exc.value.detail.get("claim") == "conflict"


@pytest.mark.asyncio
async def test_status_traegt_stufen_semantik():
    nj.reservieren("story-szene-1:cue-3", "h")
    st = await r.narrate_status("story-szene-1:cue-3")
    assert "prepare" in st["stage_semantics"] and "Abo" in st["stage_semantics"]["prepare"]
