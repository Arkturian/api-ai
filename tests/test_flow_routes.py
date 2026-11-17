"""/ai/genvideo/flow (Google Flow ueber useapi.net, 14.11.2026)."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.routes import flow_routes as fr


class Upstream:
    def __init__(self):
        self.aufrufe = []
        self.job = {"jobid": "j1-user:1-bot:google-flow", "status": "created"}
        self.uploads = 0

    def handler(self, req: httpx.Request):
        self.aufrufe.append((req.method, str(req.url)))
        u = str(req.url)
        if req.method == "POST" and u.endswith("/videos"):
            self.gesendet = json.loads(req.content)
            return httpx.Response(201, json=self.job)
        if req.method == "GET" and "/jobs/" in u:
            return httpx.Response(200, json=self.job)
        if req.method == "GET" and u.startswith("https://video.example/"):
            return httpx.Response(200, content=b"\x00mp4")
        if req.method == "POST" and u.endswith("/storage/upload"):
            self.uploads += 1
            return httpx.Response(200, json={"id": 900 + self.uploads})
        return httpx.Response(404, json={"error": "unerwartet", "url": u})


@pytest.fixture
def up(monkeypatch, tmp_path):
    u = Upstream()
    monkeypatch.setenv("USEAPI_TOKEN", "test-token")
    monkeypatch.setenv("FLOW_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "verbrauch"))
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(
        transport=httpx.MockTransport(u.handler)))
    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "storage_api_key", lambda: "k")
    u.tmp = tmp_path
    return u


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(fr.router, prefix="/ai")
    return TestClient(app)


def _verbrauch(tmp_path):
    zeilen = []
    for p in (tmp_path / "verbrauch").glob("bilder_*.jsonl"):
        zeilen += [json.loads(z) for z in p.read_text().splitlines() if z.strip()]
    return zeilen


def test_ohne_freigabe_403_und_kein_aufruf(up, client):
    r = client.post("/ai/genvideo/flow", json={"prompt": "x"})
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "api_billing_confirmation_required"
    assert up.aufrufe == []


def test_falsche_dauer_vor_jedem_aufruf(up, client):
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "duration": 10, "confirm_api_billing": True})
    assert r.status_code == 422
    assert r.json()["detail"]["allowed"] == [4, 6, 8]
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "model": "veo-3.1-quality", "duration": 6,
                                               "confirm_api_billing": True})
    assert r.status_code == 422
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "resolution": "360p", "confirm_api_billing": True})
    assert r.status_code == 422
    assert up.aufrufe == []


def test_ohne_token_503(up, client, monkeypatch):
    monkeypatch.delenv("USEAPI_TOKEN")
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "confirm_api_billing": True})
    assert r.status_code == 503
    assert up.aufrufe == []


def test_max_credits_bremst(up, client):
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "model": "veo-3.1-quality",
                                               "max_credits": 50, "confirm_api_billing": True})
    assert r.status_code == 422
    assert r.json()["detail"]["estimated_credits"] == 100
    assert up.aufrufe == []


def test_auftrag_geht_async_mit_hochformat(up, client):
    r = client.post("/ai/genvideo/flow", headers={"X-Agent-Name": "3DTischler"},
                    json={"prompt": "Katze", "aspect_ratio": "9:16", "duration": 6, "confirm_api_billing": True})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["job_id"] == up.job["jobid"]
    assert b["estimated_credits"] == 10
    assert up.gesendet == {"prompt": "Katze", "model": "veo-3.1-fast", "aspectRatio": "portrait",
                           "duration": 6, "resolution": "720p", "count": 1, "async": True}
    akte = fr._lesen(up.job["jobid"])
    assert akte["agent"] == "3DTischler" and akte["quelle"] == "header"


def _fertig(up, url="https://video.example/a.mp4"):
    m = {"mediaGenerationId": "m1"}
    if url:
        m["videoUrl"] = url
    up.job = {"jobid": up.job["jobid"], "status": "completed", "request": {"model": "veo-3.1-fast"},
              "response": {"media": [m], "remainingCredits": 990}}


def test_status_sichert_und_bucht_genau_einmal(up, client):
    client.post("/ai/genvideo/flow", json={"prompt": "x", "confirm_api_billing": True})
    _fertig(up)
    jid = up.job["jobid"]
    a = client.get(f"/ai/genvideo/flow/status/{jid}").json()
    b = client.get(f"/ai/genvideo/flow/status/{jid}").json()
    assert up.uploads == 1
    assert a["clips"][0]["storage_object_id"] == 901 == b["clips"][0]["storage_object_id"]
    assert a["clips"][0]["storage_url"].endswith("/storage/media/901")
    assert a["remaining_credits"] == 990
    z = _verbrauch(up.tmp)
    assert len(z) == 1 and z[0]["art"] == "video" and z[0]["routed_via"] == "google-flow"


def test_fehlender_link_wird_spaeter_gesichert(up, client):
    client.post("/ai/genvideo/flow", json={"prompt": "x", "confirm_api_billing": True})
    jid = up.job["jobid"]
    _fertig(up, url=None)
    a = client.get(f"/ai/genvideo/flow/status/{jid}").json()
    assert up.uploads == 0 and "hint" in a
    _fertig(up)
    b = client.get(f"/ai/genvideo/flow/status/{jid}").json()
    assert up.uploads == 1 and b["clips"][0]["storage_object_id"] == 901
    assert len(_verbrauch(up.tmp)) == 1


def test_gleichzeitige_abfragen_verklemmen_nicht_und_sichern_einmal(up, monkeypatch):
    jid = up.job["jobid"]
    _fertig(up)
    zaehler = {"n": 0}

    async def langsam(url, c, l, name):
        zaehler["n"] += 1
        await asyncio.sleep(0.8)
        return {"storage_object_id": 7, "storage_url": "u"}

    monkeypatch.setattr(fr, "_video_sichern", langsam)

    async def beide():
        return await asyncio.wait_for(asyncio.gather(
            fr._abschliessen(jid, up.job, sichern=True, collection_id=None, link_id=None),
            fr._abschliessen(jid, up.job, sichern=True, collection_id=None, link_id=None)), 10)

    # In einem Faden: ein blockierendes flock haelt die Ereignisschleife an,
    # dann greift auch wait_for nicht mehr — der Test soll scheitern, nicht haengen.
    import threading
    fehler = []
    t = threading.Thread(target=lambda: fehler.append(None) if asyncio.run(beide()) else None, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "Verklemmung: zweite Abfrage blockiert die Ereignisschleife"
    assert zaehler["n"] == 1
    assert len(_verbrauch(up.tmp)) == 1


def test_fehlgeschlagen_bucht_nicht(up, client):
    jid = up.job["jobid"]
    up.job = {"jobid": jid, "status": "failed", "error": "policy", "code": 400}
    r = client.get(f"/ai/genvideo/flow/status/{jid}").json()
    assert r["status"] == "failed" and r["error"] == "policy"
    assert _verbrauch(up.tmp) == []


def test_flow_fehler_596_sagt_neu_verknuepfen(up, client, monkeypatch):
    def h(req):
        return httpx.Response(596, json={"error": "session"})
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(transport=httpx.MockTransport(h)))
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "confirm_api_billing": True})
    assert r.status_code == 502
    assert "neu" in r.json()["detail"]["hint"]


def test_konto_kuerzt_adresse(up, client, monkeypatch):
    def h(req):
        if str(req.url).endswith("/accounts"):
            return httpx.Response(200, json={"alex@gmail.com": {"health": "ok", "secret": "x"}})
        return httpx.Response(200, json={})
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(transport=httpx.MockTransport(h)))
    from ai import auth_guard as ag
    client.app.dependency_overrides[ag.angemeldet] = lambda: "test"
    b = client.get("/ai/genvideo/flow/account").json()
    assert b["accounts"] == [{"account": "a***@gmail.com", "health": "ok"}]
    assert "alex@" not in json.dumps(b)


def test_credits_schaetzung():
    assert fr.credits_schaetzung("omni-flash-1.2", 15, "720p", 1) == 60
    assert fr.credits_schaetzung("omni-flash-1.2", 8, "360p", 1) == 8
    assert fr.credits_schaetzung("veo-3.1-fast", 8, "720p", 2) == 20


# --- Bilder -------------------------------------------------------------------

class BildUpstream(Upstream):
    def handler(self, req):
        u = str(req.url)
        self.aufrufe.append((req.method, u))
        if req.method == "POST" and u.endswith("/images"):
            self.gesendet = json.loads(req.content)
            import base64
            return httpx.Response(200, json={"jobId": "ji1", "media": [
                {"image": {"generatedImage": {"mediaGenerationId": "g1", "seed": 5,
                                              "fifeUrl": "https://img.example/a.png"}}},
                {"image": {"generatedImage": {"mediaGenerationId": "g2", "seed": 6,
                                              "encodedImage": base64.b64encode(b"PNG").decode()}}}]})
        if req.method == "GET" and u.startswith("https://img.example/"):
            return httpx.Response(200, content=b"PNG", headers={"content-type": "image/jpeg"})
        if req.method == "POST" and u.endswith("/storage/upload"):
            self.uploads += 1
            self.typen = getattr(self, "typen", []) + [req.content.split(b"Content-Type: ")[1][:10]]
            return httpx.Response(200, json={"id": 700 + self.uploads})
        return httpx.Response(404)


@pytest.fixture
def bup(monkeypatch, tmp_path):
    u = BildUpstream()
    monkeypatch.setenv("USEAPI_TOKEN", "t")
    monkeypatch.setenv("FLOW_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "verbrauch"))
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(
        transport=httpx.MockTransport(u.handler)))
    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "storage_api_key", lambda: "k")
    u.tmp = tmp_path
    return u


def test_bild_ohne_freigabe_und_falsche_werte_ohne_aufruf(bup, client):
    assert client.post("/ai/genimage/flow", json={"prompt": "x"}).status_code == 403
    assert client.post("/ai/genimage/flow", json={"prompt": "x", "model": "imagen-4",
                                                   "confirm_api_billing": True}).status_code == 422
    assert client.post("/ai/genimage/flow", json={"prompt": "x", "aspect_ratio": "auto",
                                                   "confirm_api_billing": True}).status_code == 422
    assert bup.aufrufe == []


def test_bild_count_eins_gesendet_und_beide_formen_gesichert(bup, client):
    r = client.post("/ai/genimage/flow", headers={"X-Agent-Name": "Story"},
                    json={"prompt": "Haus", "aspect_ratio": "16:9", "confirm_api_billing": True})
    assert r.status_code == 200, r.text
    assert bup.gesendet == {"prompt": "Haus", "model": "nano-banana-2", "aspectRatio": "16:9", "count": 1}
    b = r.json()
    assert [i["storage_object_id"] for i in b["images"]] == [701, 702]
    assert b["image_url"].endswith("/storage/media/701") and b["routed_via"] == "google-flow"
    assert bup.typen[0].startswith(b"image/jpeg")
    z = _verbrauch(bup.tmp)
    assert len(z) == 2 and {x["agent"] for x in z} == {"Story"} and z[0]["art"] == "bild"


def test_bild_hochskaliert_und_faellt_sonst_auf_1k_zurueck(bup, client, monkeypatch):
    import base64
    alt = bup.handler
    zaehler = {"n": 0}

    def h(req):
        if str(req.url).endswith("/images/upscale"):
            zaehler["n"] += 1
            if json.loads(req.content)["mediaGenerationId"] == "g2":
                return httpx.Response(503, json={"error": "weg"})
            assert json.loads(req.content)["resolution"] == "4k"
            return httpx.Response(200, json={"encodedImage": base64.b64encode(b"JPG4K").decode()})
        return alt(req)
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(transport=httpx.MockTransport(h)))
    r = client.post("/ai/genimage/flow", json={"prompt": "x", "count": 2, "upscale": "4k", "confirm_api_billing": True})
    b = r.json()
    assert zaehler["n"] == 2
    a, z = b["images"]
    assert a["upscaled"] == "4k" and a["original_storage_object_id"] == 701 and a["storage_object_id"] == 702
    assert b["image_url"].endswith("/storage/media/702")
    assert "upscale_error" in z and z["storage_object_id"] == 703
    assert [x["storage_id"] for x in _verbrauch(bup.tmp)] == [702, 703]


def test_bild_falscher_upscale_ohne_aufruf(bup, client):
    r = client.post("/ai/genimage/flow", json={"prompt": "x", "upscale": "8k", "confirm_api_billing": True})
    assert r.status_code == 422 and bup.aufrufe == []


# --- Vertrauliche Ergebnisse (EDERA-EDWARD, 17.11.2026) ------------------------

def _formular(req: httpx.Request) -> dict:
    """Textfelder aus dem multipart-Upload (ohne die Datei)."""
    import re as _re
    felder = {}
    for name, wert in _re.findall(rb'name="([^"]+)"\r\n\r\n([^\r]*)\r\n', req.content):
        felder[name.decode()] = wert.decode()
    return felder


def _mit_formular(u):
    alt = u.handler
    u.formulare = []

    def h(req):
        if req.method == "POST" and str(req.url).endswith("/storage/upload"):
            f = _formular(req)
            u.formulare.append(f)
            r = alt(req)
            d = r.json()
            if f.get("grant_scope"):
                d["grant_scope"] = f["grant_scope"]
            return httpx.Response(200, json=d)
        return alt(req)
    return h


def _als_jwt_agent(monkeypatch, name="EDERA-EDWARD"):
    from ai.services import bild_verbrauch as bv
    monkeypatch.setattr(bv, "aufrufer", lambda a, x: (name, "jwt"))


def test_video_privat_bindet_an_den_angemeldeten_agenten(up, client, monkeypatch):
    h = _mit_formular(up)
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(
        transport=httpx.MockTransport(h)))
    _als_jwt_agent(monkeypatch)
    r = client.post("/ai/genvideo/flow", json={"prompt": "x", "private": True, "confirm_api_billing": True})
    assert r.status_code == 200, r.text
    assert "private" not in up.gesendet  # geht nicht an Google
    _fertig(up)
    b = client.get(f"/ai/genvideo/flow/status/{up.job['jobid']}").json()
    f = up.formulare[0]
    assert f["is_public"] == "false" and f["private"] == "true"
    assert f["grant_scope"] == "cloud-session:EDERA-EDWARD"
    assert b["clips"][0]["private"] is True and b["clips"][0]["grant_scope"] == "cloud-session:EDERA-EDWARD"


def test_video_ohne_private_bleibt_oeffentlich_und_ungebunden(up, client, monkeypatch):
    h = _mit_formular(up)
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(
        transport=httpx.MockTransport(h)))
    _als_jwt_agent(monkeypatch)
    client.post("/ai/genvideo/flow", json={"prompt": "x", "confirm_api_billing": True})
    _fertig(up)
    b = client.get(f"/ai/genvideo/flow/status/{up.job['jobid']}").json()
    f = up.formulare[0]
    assert f["is_public"] == "true" and "private" not in f and "grant_scope" not in f
    assert "private" not in b["clips"][0]


def test_bild_privat_auch_beim_hochskalieren_und_nur_mit_jwt_gebunden(bup, client, monkeypatch):
    import base64
    alt = bup.handler

    def mit_upscale(req):
        if str(req.url).endswith("/images/upscale"):
            return httpx.Response(200, json={"encodedImage": base64.b64encode(b"JPG").decode()})
        return alt(req)
    bup.handler = mit_upscale
    h = _mit_formular(bup)
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: httpx.AsyncClient(
        transport=httpx.MockTransport(h)))
    # Nur Kopfzeile, kein JWT: privat ja, Bindung nein (Kopf ist faelschbar).
    r = client.post("/ai/genimage/flow", headers={"X-Agent-Name": "EDERA-EDWARD"},
                    json={"prompt": "x", "private": True, "upscale": "4k", "confirm_api_billing": True})
    assert r.status_code == 200, r.text
    assert len(bup.formulare) == 4  # 2 Bilder (Mock liefert zwei) + 2 Hochskalierte
    assert all(f["private"] == "true" and f["is_public"] == "false" for f in bup.formulare)
    assert all("grant_scope" not in f for f in bup.formulare)
    assert all(i["private"] is True and "hint" in i for i in r.json()["images"])


def test_scope_fuer_wie_storage_mcp():
    assert fr.scope_fuer("EDERA-EDWARD") == "cloud-session:EDERA-EDWARD"
    assert fr.scope_fuer("Ä b") == "cloud-session-x:" + "Ä b".encode().hex()
    assert fr.privat_ablage(True, "A", "header") == {"privat": True, "grant_scope": None}
    assert fr.privat_ablage(False, "A", "jwt") == {}
