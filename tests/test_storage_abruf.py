"""Issue #2654: Abruf von Storage-Adressen in /ai/transcribe und Vision.

Gefunden am 06.11.2026. Beide Wege schickten den Dienstschluessel an jede
Adresse, die `/storage/media/` enthielt, und folgten Umleitungen mit Schluessel.
Eine private Datei endete in /ai/transcribe als 502, in Vision lief das Modell
still ohne Bild weiter. Kein echter Schluessel, kein Netz: MockTransport.
"""

import json
import logging

import httpx
import pytest
from fastapi import HTTPException

from ai.clients import storage_abruf as s
from ai.routes import audio_ai_routes as a
from ai.routes import text_ai_routes as t

SCHLUESSEL = "test-schluessel-nicht-echt"
EIGEN = "https://api-storage.arkturian.com/storage/media/133118"
SIGNIERT = EIGEN + "?exp=1793962102&kid=k1&sig=abcdef"
FREMD = "https://fremder-host.example/storage/media/1"
PRIVAT = json.dumps({"detail": {"error": "This object is private", "code": "private_media"}}).encode()


@pytest.fixture(autouse=True)
def _umgebung(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)
    for modul in (s, a, t):
        monkeypatch.setattr(modul, "storage_api_key", lambda: SCHLUESSEL)


def _netz(monkeypatch, antwort):
    """Ersetzt das Netz fuer beide httpx-Clients, merkt sich (Adresse, Schluessel)."""
    gesehen = []

    def handler(request):
        gesehen.append((str(request.url), request.headers.get("x-api-key")))
        return antwort(request)

    transport = httpx.MockTransport(handler)
    echt_async, echt_sync = httpx.AsyncClient, httpx.Client
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: echt_async(transport=transport, **k))
    monkeypatch.setattr(httpx, "Client", lambda **k: echt_sync(transport=transport, **k))
    return gesehen


def _audio(request):
    return httpx.Response(200, content=b"AUDIO")


def _umleitung_nach_draussen(request):
    if request.url.host == "api-storage.arkturian.com":
        return httpx.Response(302, headers={"location": "https://abfluss.example/x"})
    return httpx.Response(200, content=b"AUDIO")


def _privat(request):
    return httpx.Response(403, content=PRIVAT, headers={"content-type": "application/json"})


# ---------------------------------------------------- /ai/transcribe

@pytest.mark.asyncio
async def test_transcribe_eigener_storage_bekommt_den_schluessel(monkeypatch):
    gesehen = _netz(monkeypatch, _audio)
    f = await a._fetch_audio_from_url(EIGEN)
    assert await f.read() == b"AUDIO"
    assert gesehen == [(EIGEN, SCHLUESSEL)]


@pytest.mark.asyncio
async def test_transcribe_fremder_host_bekommt_ihn_nicht(monkeypatch):
    gesehen = _netz(monkeypatch, _audio)
    await a._fetch_audio_from_url(FREMD)
    assert gesehen == [(FREMD, None)]


@pytest.mark.asyncio
async def test_transcribe_umleitung_traegt_ihn_nicht_nach_draussen(monkeypatch):
    gesehen = _netz(monkeypatch, _umleitung_nach_draussen)
    await a._fetch_audio_from_url(EIGEN)
    assert gesehen[0][1] == SCHLUESSEL
    assert "abfluss.example" in gesehen[1][0] and gesehen[1][1] is None


@pytest.mark.asyncio
async def test_transcribe_signierte_adresse_ohne_schluessel(monkeypatch):
    gesehen = _netz(monkeypatch, _audio)
    await a._fetch_audio_from_url(SIGNIERT)
    assert gesehen == [(SIGNIERT, None)]


@pytest.mark.asyncio
async def test_transcribe_private_datei_ist_403_mit_hinweis(monkeypatch):
    _netz(monkeypatch, _privat)
    with pytest.raises(HTTPException) as e:
        await a._fetch_audio_from_url(EIGEN)
    assert e.value.status_code == 403
    assert e.value.detail["error"] == "private_media"
    assert "assets_download" in e.value.detail["hint"]


@pytest.mark.asyncio
async def test_transcribe_anderer_403_bleibt_502_und_verraet_keine_signatur(monkeypatch):
    """Nachbarfall: abgelaufene Signatur ist nicht private_media."""
    _netz(monkeypatch, lambda r: httpx.Response(403, json={"detail": "Access denied"}))
    with pytest.raises(HTTPException) as e:
        await a._fetch_audio_from_url(SIGNIERT)
    assert e.value.status_code == 502
    assert "sig=" not in str(e.value.detail)


@pytest.mark.asyncio
async def test_transcribe_groessengrenze_gilt_weiter(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_URL_MAX_MB", "0")
    _netz(monkeypatch, _audio)
    with pytest.raises(HTTPException) as e:
        await a._fetch_audio_from_url(EIGEN)
    assert e.value.status_code == 413


@pytest.mark.asyncio
async def test_transcribe_endlose_umleitung_bricht_ab(monkeypatch):
    aufrufe = _netz(monkeypatch, lambda r: httpx.Response(302, headers={"location": "https://kreis.example/x"}))
    with pytest.raises(HTTPException) as e:
        await a._fetch_audio_from_url("https://kreis.example/start")
    assert e.value.status_code == 502
    assert len(aufrufe) == s.MAX_SPRUENGE + 1


# ---------------------------------------------------- Vision

def _bild(request):
    return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})


@pytest.fixture
def _vision_ordner(monkeypatch, tmp_path):
    monkeypatch.setenv("CLI_HOME", str(tmp_path))
    return tmp_path / ".aiapi-neutral" / ".vision"


def test_vision_fremder_host_im_prompt_bekommt_den_schluessel_nicht(monkeypatch, _vision_ordner):
    gesehen = _netz(monkeypatch, _bild)
    t._download_storage_images(f"Was zeigt {FREMD} ?")
    assert gesehen == [(FREMD, None)]


def test_vision_umleitung_traegt_ihn_nicht_nach_draussen(monkeypatch, _vision_ordner):
    gesehen = _netz(monkeypatch, _umleitung_nach_draussen)
    t._download_storage_images(f"Was zeigt {EIGEN} ?")
    assert gesehen[0][1] == SCHLUESSEL and gesehen[1][1] is None


def test_vision_eigenes_bild_wird_lokal(monkeypatch, _vision_ordner):
    _netz(monkeypatch, _bild)
    (url, pfad), = t._download_storage_images(f"Was zeigt {EIGEN} ?")
    assert url == EIGEN and open(pfad, "rb").read() == b"\x89PNG"


def test_vision_private_datei_bricht_ab_und_raeumt_auf(monkeypatch, _vision_ordner):
    """Vorher: Warnung im Protokoll, das Modell antwortete ohne Bild."""
    zweite = EIGEN.replace("133118", "2")

    def antwort(request):
        return _privat(request) if request.url.path.endswith("/2") else _bild(request)

    _netz(monkeypatch, antwort)
    with pytest.raises(HTTPException) as e:
        t._download_storage_images(f"{EIGEN} und {zweite}")
    assert e.value.status_code == 403 and e.value.detail["error"] == "private_media"
    assert list(_vision_ordner.iterdir()) == []


# ---------------------------------------------------- Protokoll

def test_signatur_steht_nicht_im_protokoll():
    rec = logging.LogRecord("httpx", logging.INFO, "", 0, 'HTTP Request: %s %s "%s %d %s"',
                            ("GET", httpx.URL(SIGNIERT), "HTTP/1.1", 200, "OK"), None)
    assert s.AnfrageAusProtokoll().filter(rec) is True
    zeile = rec.getMessage()
    assert "sig=" not in zeile and EIGEN in zeile and "200 OK" in zeile
