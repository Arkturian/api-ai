"""request_id fuer /ai/chatgpt und /ai/genimage (Issue #1817, Story).

Ziel: Replay liefert das gespeicherte Ergebnis ohne zweiten Lauf — beim
Abo-Textpfad ohne Kontingent, beim Bildpfad ohne 0,175 USD. Gegenfall:
laufend -> 409; Fehler NACH dem Aufruf -> 409 (Geld/Kontingent koennen
weg sein). Nachbarfall: Fehler VOR dem Aufruf (Sperre 403/429, 422) darf
neu; `cost-status` bleibt erreichbar neben der Statusroute; ohne
request_id unveraendert.
"""

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t
from ai.routes import image_ai_routes as g
from ai.services import narrate_jobs as nj


@pytest.fixture(autouse=True)
def _ablage(monkeypatch, tmp_path):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))


# ---------------------------------------------------- /ai/chatgpt

def _chatgpt_stub(monkeypatch, verhalten):
    z = {"n": 0}
    async def einmal(prompt, model=None, api_key="x"):
        z["n"] += 1
        return await verhalten(prompt)
    monkeypatch.setattr(t, "_chatgpt_einmal", einmal)
    return z


@pytest.mark.asyncio
async def test_chatgpt_replay_ohne_zweiten_lauf(monkeypatch):
    async def ok(p):
        return t.AIResponse(response="Szene …", model="gpt-5.6-sol", tokens_used=12)
    z = _chatgpt_stub(monkeypatch, ok)
    p = t.Prompt(prompt="Szene 29 ausarbeiten", model="gpt-5.6-sol", request_id="szene-29-entwurf-1")
    a = await t.chatgpt_endpoint(p, None, "x")
    b = await t.chatgpt_endpoint(p, None, "x")
    assert a.response == b.response == "Szene …" and z["n"] == 1
    st = await t.chatgpt_status("szene-29-entwurf-1")
    assert st["kind"] == "chatgpt" and st["state"] == "done" and st["result"]["model"] == "gpt-5.6-sol"


@pytest.mark.asyncio
async def test_chatgpt_ohne_kennung_wie_bisher(monkeypatch):
    async def ok(p):
        return t.AIResponse(response="x", model="codex-default")
    z = _chatgpt_stub(monkeypatch, ok)
    p = t.Prompt(prompt="x")
    await t.chatgpt_endpoint(p, None, "x"); await t.chatgpt_endpoint(p, None, "x")
    assert z["n"] == 2


@pytest.mark.asyncio
async def test_chatgpt_fehler_nach_dem_aufruf_wird_nicht_wiederholt(monkeypatch):
    async def bricht(p):
        raise HTTPException(status_code=502, detail={"error": "codex_failed"})
    z = _chatgpt_stub(monkeypatch, bricht)
    p = t.Prompt(prompt="x", request_id="szene-29-entwurf-2")
    with pytest.raises(HTTPException):
        await t.chatgpt_endpoint(p, None, "x")
    with pytest.raises(HTTPException) as exc:
        await t.chatgpt_endpoint(p, None, "x")
    assert exc.value.status_code == 409 and exc.value.detail["error"] == "chatgpt_failed_after_call"
    assert z["n"] == 1


@pytest.mark.asyncio
async def test_chatgpt_sperre_vor_dem_aufruf_darf_neu(monkeypatch):
    laeufe = {"n": 0}
    async def erst_422(p):
        laeufe["n"] += 1
        if laeufe["n"] == 1:
            raise HTTPException(status_code=422, detail={"error": "response_format_unsupported_on_endpoint"})
        return t.AIResponse(response="ok", model="gpt-5.6-sol")
    _chatgpt_stub(monkeypatch, erst_422)
    p = t.Prompt(prompt="x", request_id="szene-29-entwurf-3")
    with pytest.raises(HTTPException):
        await t.chatgpt_endpoint(p, None, "x")
    assert (await t.chatgpt_status("szene-29-entwurf-3"))["failed_stage"] == "pre_tts"
    assert (await t.chatgpt_endpoint(p, None, "x")).response == "ok" and laeufe["n"] == 2


def test_cost_status_bleibt_erreichbar():
    """Die Statusroute /chatgpt/{request_id} darf /chatgpt/cost-status nicht verdecken."""
    from main import app
    pfade = [getattr(r, "path", "") for r in app.routes]
    assert pfade.index("/ai/chatgpt/cost-status") < pfade.index("/ai/chatgpt/{request_id}")
    import asyncio
    with pytest.raises(HTTPException) as e:
        asyncio.run(t.chatgpt_status("cost-status"))
    assert e.value.status_code == 404


# ---------------------------------------------------- /ai/genimage

def _bild_stub(monkeypatch, verhalten):
    z = {"n": 0}
    async def einmal(request, api_key="x"):
        z["n"] += 1
        return await verhalten(request)
    monkeypatch.setattr(g, "_generate_image_einmal", einmal)
    return z


@pytest.mark.asyncio
async def test_genimage_replay_ohne_anbieteraufruf(monkeypatch):
    async def ok(r):
        return {"id": 125000, "image_url": "https://s/125000", "file_url": "https://s/125000", "model": "gpt-image-2"}
    z = _bild_stub(monkeypatch, ok)
    r = g.ImageGenRequest(prompt="Ein Tor", model="gpt-image-2", confirm_api_billing=True, request_id="shot-38-v1-bild")
    a = await g.generate_image_endpoint(r, "x")
    b = await g.generate_image_endpoint(r, "x")
    assert a["id"] == b["id"] == 125000 and b["replayed"] is True and z["n"] == 1
    st = await g.genimage_status("shot-38-v1-bild")
    assert st["kind"] == "image" and st["state"] == "done"


@pytest.mark.asyncio
async def test_genimage_andere_nutzlast_409_und_fehler_nach_aufruf_409(monkeypatch):
    async def ok(r):
        return {"id": 1, "image_url": "u", "file_url": "u"}
    _bild_stub(monkeypatch, ok)
    r = g.ImageGenRequest(prompt="Ein Tor", request_id="shot-38-v1-bild")
    await g.generate_image_endpoint(r, "x")
    with pytest.raises(HTTPException) as e:
        await g.generate_image_endpoint(g.ImageGenRequest(prompt="Anderes Tor", request_id="shot-38-v1-bild"), "x")
    assert e.value.detail["error"] == "request_id_payload_mismatch"

    async def bricht(r):
        raise HTTPException(status_code=502, detail={"error": "openai_upstream_error"})
    z = _bild_stub(monkeypatch, bricht)
    r2 = g.ImageGenRequest(prompt="Ein Tor", request_id="shot-39-v1-bild")
    with pytest.raises(HTTPException):
        await g.generate_image_endpoint(r2, "x")
    with pytest.raises(HTTPException) as e2:
        await g.generate_image_endpoint(r2, "x")
    assert e2.value.status_code == 409 and e2.value.detail["error"] == "image_failed_after_call" and z["n"] == 1


@pytest.mark.asyncio
async def test_genimage_billing_tor_403_darf_neu(monkeypatch):
    laeufe = {"n": 0}
    async def erst_403(r):
        laeufe["n"] += 1
        if laeufe["n"] == 1:
            raise HTTPException(status_code=403, detail={"error": "api_billing_confirmation_required"})
        return {"id": 2, "image_url": "u", "file_url": "u"}
    _bild_stub(monkeypatch, erst_403)
    r = g.ImageGenRequest(prompt="x", request_id="shot-40-v1-bild")
    with pytest.raises(HTTPException):
        await g.generate_image_endpoint(r, "x")
    assert (await g.genimage_status("shot-40-v1-bild"))["failed_stage"] == "pre_tts"
    assert (await g.generate_image_endpoint(r, "x"))["id"] == 2


def test_genimage_models_bleibt_erreichbar():
    """Die Statusroute /genimage/{request_id} darf /genimage/models nicht
    verdecken. Am Ziel gemessen 13.09.: /ai/genimage/models -> 404
    unknown_request_id, weil die Kennungsroute davor stand."""
    from main import app
    pfade = [getattr(r, "path", "") for r in app.routes]
    assert pfade.index("/ai/genimage/models") < pfade.index("/ai/genimage/{request_id}")
    import asyncio
    with pytest.raises(HTTPException) as e:
        asyncio.run(g.genimage_status("models"))
    assert e.value.status_code == 404


# ------------------------------- signierte Referenzen (Content #5764, 04.11.)

_S = "https://api-storage.arkturian.com/storage/media/"


def _refs(sig=None, ids=(128981, 128982)):
    q = f"?exp={sig}&kid=k1&sig=abc{sig}" if sig else ""
    return [f"{_S}{i}{q}" for i in ids]


@pytest.mark.asyncio
async def test_genimage_signierte_referenzen_gleiche_kennung_darf_neu(monkeypatch):
    """Echter Ablauf vom 04.11.: erster Lauf mit unsignierten Links ->
    422 vor dem Anbieter; Neuanlauf derselben request_id mit frisch
    signierten Links muss laufen (nicht request_id_payload_mismatch), und
    beim Anbieter kommen die SIGNIERTEN Links an, nicht die gekuerzten."""
    monkeypatch.delenv("STORAGE_API_URL", raising=False)
    gesehen = []
    async def lauf(r):
        gesehen.append(list(r.reference_image_urls))
        if len(gesehen) == 1:
            raise HTTPException(status_code=422, detail={"error": "reference_image_fetch_failed"})
        return {"id": 9, "image_url": "u", "file_url": "u"}
    _bild_stub(monkeypatch, lauf)
    rid = "arkturian-daily-test-bild"
    mk = lambda refs: g.ImageGenRequest(prompt="x", model="gpt-image-2", confirm_api_billing=True,
                                        request_id=rid, reference_image_urls=refs)
    with pytest.raises(HTTPException):
        await g.generate_image_endpoint(mk(_refs()), "x")
    out = await g.generate_image_endpoint(mk(_refs(1000)), "x")
    assert out["id"] == 9 and gesehen[1] == _refs(1000)
    # spaeterer Lauf mit neuer Signatur: Replay, kein zweites Bild
    again = await g.generate_image_endpoint(mk(_refs(2000)), "x")
    assert again["replayed"] is True and len(gesehen) == 2


@pytest.mark.asyncio
async def test_genimage_andere_referenz_bleibt_mismatch(monkeypatch):
    monkeypatch.delenv("STORAGE_API_URL", raising=False)
    async def ok(r):
        return {"id": 1, "image_url": "u", "file_url": "u"}
    _bild_stub(monkeypatch, ok)
    rid = "arkturian-daily-test2-bild"
    await g.generate_image_endpoint(g.ImageGenRequest(prompt="x", request_id=rid, reference_image_urls=_refs(1)), "x")
    for refs in (_refs(1, ids=(128981, 128999)),                       # anderes Objekt
                 [_S + "128981?v=neu&exp=1&kid=k1&sig=a", _S + "128982?exp=1&kid=k1&sig=a"],  # anderer Inhalt
                 ["https://fremd.example/storage/media/128981", _S + "128982"]):            # fremder Host
        with pytest.raises(HTTPException) as e:
            await g.generate_image_endpoint(g.ImageGenRequest(prompt="x", request_id=rid, reference_image_urls=refs), "x")
        assert e.value.detail["error"] == "request_id_payload_mismatch"


def test_ohne_signatur_nur_storage_und_nur_die_drei():
    assert g._ohne_signatur(_S + "1?v=c&exp=9&kid=k&sig=s") == _S + "1?v=c"
    fremd = "https://fremd.example/x?exp=9&sig=s"
    assert g._ohne_signatur(fremd) == fremd


# --- Am Ziel gefunden 14.11. (Post 5771): arkserver-Umgebung --------------
# Die Tests oben loeschen STORAGE_API_URL und liefen damit nur gegen die
# Vorgabe. arkserver hat api-storage.arkserver…, die signierten Links lauten
# auf den kanonischen Namen -> die Signatur blieb im Hash.

_ARKSERVER = "https://api-storage.arkserver.arkturian.com"


def test_ohne_signatur_kanonischer_host_bei_arkserver_umgebung(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", _ARKSERVER)
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)
    monkeypatch.delenv("STORAGE_SIGNATUR_HOSTS", raising=False)
    assert g._ohne_signatur(_S + "128981?exp=9&kid=k&sig=s") == _S + "128981"
    assert g._ohne_signatur(_S + "1?v=c&exp=9&kid=k&sig=s") == _S + "1?v=c"
    eigen = _ARKSERVER + "/storage/media/1?exp=9&kid=k&sig=s"
    assert g._ohne_signatur(eigen) == _ARKSERVER + "/storage/media/1"
    fremd = "https://fremd.example/storage/media/1?exp=9&sig=s"
    assert g._ohne_signatur(fremd) == fremd
    anderer_port = "https://api-storage.arkturian.com:8443/storage/media/1?exp=9&sig=s"
    assert g._ohne_signatur(anderer_port) == anderer_port


def test_signatur_host_gibt_keinen_schluessel(monkeypatch):
    """Die neue Liste gilt nur fuer den Hash, nie fuer den Schluessel."""
    from ai.clients import storage_abruf as sa
    monkeypatch.setenv("STORAGE_API_URL", _ARKSERVER)
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)
    assert sa.darf_schluessel_sehen(_S + "128981") is False
    assert sa.darf_schluessel_sehen(_ARKSERVER + "/storage/media/1") is True


@pytest.mark.asyncio
async def test_genimage_signierte_referenzen_neuanlauf_bei_arkserver_umgebung(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", _ARKSERVER)
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)
    monkeypatch.delenv("STORAGE_SIGNATUR_HOSTS", raising=False)
    gesehen = []
    async def lauf(r):
        gesehen.append(list(r.reference_image_urls))
        if len(gesehen) == 1:
            raise HTTPException(status_code=422, detail={"error": "reference_image_fetch_failed"})
        return {"id": 9, "image_url": "u", "file_url": "u"}
    _bild_stub(monkeypatch, lauf)
    rid = "arkturian-daily-test3-bild"
    mk = lambda refs: g.ImageGenRequest(prompt="x", model="gpt-image-2", confirm_api_billing=True,
                                        request_id=rid, reference_image_urls=refs)
    with pytest.raises(HTTPException):
        await g.generate_image_endpoint(mk(_refs()), "x")
    out = await g.generate_image_endpoint(mk(_refs(1000)), "x")
    assert out["id"] == 9 and gesehen[1] == _refs(1000)
