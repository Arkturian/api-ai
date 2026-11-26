"""#4132: Storage-Uploads im Auftrag eines Agenten tragen dessen Mandanten.

Storage, 26.11.2026: Bilder von SaulGoodman (edera) und Design kamen ohne
`origin_tenant` an; nach #4087 sieht der Mandant sie dann nicht. Der Kopf
`X-Caller-Tenant` kommt NUR aus einem gepruften JWT, nie aus einem Kopf des
Aufrufers.
"""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ai.clients import storage_client as sc
from ai.routes import flow_routes as fr
from ai.services import aufruf


@pytest.fixture(autouse=True)
def ablage(monkeypatch, tmp_path):
    monkeypatch.setenv("KOSTENBUCH_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path / "auf"))
    monkeypatch.delenv("API_ACCESS_KEY", raising=False)


def _mit(tenant):
    return aufruf.KONTEXT.set({"pfad": "/ai/x", "aufrufer": "a", "quelle": "jwt", "tenant": tenant})


@pytest.mark.parametrize("tenant,erwartet", [
    ("edera", {"X-Caller-Tenant": "edera"}),
    ("arkturian", {"X-Caller-Tenant": "arkturian"}),
    (None, {}),
    ("Edera X", {}),
    ("", {}),
])
def test_herkunft_kopf(tenant, erwartet):
    m = _mit(tenant)
    try:
        assert aufruf.herkunft_kopf() == erwartet
    finally:
        aufruf.KONTEXT.reset(m)


def test_ohne_aufruf_kein_kopf():
    assert aufruf.herkunft_kopf() == {}


def test_angabe_aus_der_akte_geht_vor():
    assert aufruf.herkunft_kopf("edera") == {"X-Caller-Tenant": "edera"}
    assert aufruf.herkunft_kopf("nicht gültig") == {}


# --- Middleware: Mandant nur aus dem gepruften JWT ------------------------------

@pytest.fixture
def probe():
    gesehen = []

    async def ziel():
        gesehen.append(aufruf.mandant())
        return {"ok": True}
    main.app.router.add_api_route("/ai/_probe_4132", ziel, methods=["POST"])
    yield gesehen
    main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != "/ai/_probe_4132"]


def test_mandant_aus_geprueftem_jwt_auch_neben_agentenname(probe, monkeypatch):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "claims_oder_none",
                        lambda a: {"sub": "agent:SaulGoodman", "tenant": "edera"} if a == "Bearer gut" else None)
    c = TestClient(main.app, raise_server_exceptions=False)
    assert c.post("/ai/_probe_4132", headers={"Authorization": "Bearer gut", "X-Agent-Name": "SaulGoodman"}).status_code == 200
    assert probe == ["edera"]


def test_ungeprueftes_jwt_oder_kopf_setzt_keinen_mandanten(probe, monkeypatch):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "claims_oder_none", lambda a: None)
    c = TestClient(main.app, raise_server_exceptions=False)
    c.post("/ai/_probe_4132", headers={"Authorization": "Bearer gefaelscht", "X-Caller-Tenant": "edera"})
    c.post("/ai/_probe_4132", headers={"X-Agent-Name": "Design", "X-Caller-Tenant": "edera"})
    assert probe == [None, None]


# --- Upload-Wege schicken den Kopf ------------------------------------------------

class _Antwort:
    def __init__(self, daten):
        self._d = daten
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._d


def test_storage_client_schickt_herkunft(monkeypatch):
    gesendet = {}

    async def post(self, url, **kw):
        gesendet.update(kw.get("headers") or {})
        return _Antwort({"id": 7, "object_key": "k", "original_filename": "a.png", "file_url": "u",
                         "mime_type": "image/png", "file_size_bytes": 1, "checksum": "c",
                         "created_at": "2026-11-26T00:00:00", "grant_scope": "cloud-session:SaulGoodman"})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    m = _mit("edera")
    try:
        asyncio.run(sc.save_file_and_record(b"x", "a.png", context="image-generation",
                                            private=True, grant_scope="cloud-session:SaulGoodman"))
    finally:
        aufruf.KONTEXT.reset(m)
    assert gesendet.get("X-Caller-Tenant") == "edera" and "X-API-KEY" in gesendet


def test_storage_client_ohne_jwt_ohne_herkunft(monkeypatch):
    gesendet = {}

    async def post(self, url, **kw):
        gesendet.update(kw.get("headers") or {})
        return _Antwort({"id": 8, "object_key": "k", "original_filename": "a.png", "file_url": "u",
                         "mime_type": "image/png", "file_size_bytes": 1, "checksum": "c",
                         "created_at": "2026-11-26T00:00:00"})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    asyncio.run(sc.save_file_and_record(b"x", "a.png"))
    assert "X-Caller-Tenant" not in gesendet


def test_flow_spaet_gesichertes_video_nimmt_den_mandanten_der_akte(monkeypatch):
    """Videos werden beim Abfragen des Status gesichert, oft ohne den Kontext des Auftrags."""
    gesendet = {}

    class _C:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            gesendet.update(kw.get("headers") or {})
            return _Antwort({"id": 9})
    monkeypatch.setattr(fr, "_client", lambda timeout=60.0: _C())
    asyncio.run(fr._datei_sichern(None, b"video", "video/mp4", None, None, "v.mp4", mandant="edera"))
    assert gesendet.get("X-Caller-Tenant") == "edera"
