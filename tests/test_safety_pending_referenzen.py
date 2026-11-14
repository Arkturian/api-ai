"""Frisch hochgeladene Referenzen: Storage antwortet ~1 min mit 451
safety_pending (Tschepp 14.11.2026). Das ist kein Urteil — genimage und
Flow warten kurz, statt mit reference_image_fetch_failed abzubrechen.
Ein echtes Sperr-Urteil (451 ohne safety_pending) bleibt ein Fehler."""
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import HTTPException

import ai.routes.image_ai_routes as g
import ai.routes.flow_routes as fr
from ai.clients import storage_abruf as sa

_REQ = httpx.Request("GET", "https://api-storage.arkturian.com/storage/media/1")
PENDING = httpx.Response(451, json={"detail": {"code": "safety_pending", "ai_safety_status": "processing"}},
                         request=_REQ)
URTEIL = httpx.Response(451, json={"detail": {"error": "unsafe"}}, request=_REQ)
BILD = httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}, request=_REQ)


def test_erkennung():
    assert sa.ist_safety_pending(451, PENDING.content)
    assert not sa.ist_safety_pending(451, URTEIL.content)
    assert not sa.ist_safety_pending(200, PENDING.content)
    assert issubclass(sa.SicherheitAusstehend, sa.GesperrteDatei)


def _folge(monkeypatch, antworten):
    it = iter(antworten)
    aufrufe = []

    async def hole(client, url, key, *a, **k):
        aufrufe.append(url)
        return next(it)
    monkeypatch.setattr(g, "_hole_referenz", hole)
    monkeypatch.setattr(g, "_SAFETY_TAKT_S", 0)
    return aufrufe


@pytest.mark.asyncio
async def test_genimage_wartet_bis_freigegeben(monkeypatch):
    aufrufe = _folge(monkeypatch, [PENDING, PENDING, BILD])
    out = await g._download_reference_images(["https://api-storage.arkturian.com/storage/media/1"])
    assert len(out) == 1 and len(aufrufe) == 3


@pytest.mark.asyncio
async def test_genimage_gibt_nach_frist_422_mit_hinweis(monkeypatch):
    """422, nicht 503: genimage mit request_id wertet 422 als Fehler VOR dem
    Aufruf, ein Neuanlauf mit derselben Kennung bleibt moeglich (vgl. 14.11.)."""
    _folge(monkeypatch, [PENDING] * 50)
    monkeypatch.setattr(g, "SAFETY_WARTEN_S", 0)
    with pytest.raises(HTTPException) as e:
        await g._download_reference_images(["https://api-storage.arkturian.com/storage/media/1"])
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "reference_safety_pending"
    from ai.services import narrate_jobs as nj
    assert e.value.status_code in nj._VOR_AUFRUF_STATUS


@pytest.mark.asyncio
async def test_genimage_urteil_bleibt_fehler_ohne_warten(monkeypatch):
    aufrufe = _folge(monkeypatch, [URTEIL, BILD])
    with pytest.raises(HTTPException) as e:
        await g._download_reference_images(["https://api-storage.arkturian.com/storage/media/1"])
    assert e.value.status_code == 422 and len(aufrufe) == 1


@pytest.mark.asyncio
async def test_oeffne_wirft_ausstehend_bzw_gesperrt(monkeypatch):
    async def frei(url):
        return url, {}, {}
    monkeypatch.setattr(sa, "ziel_pruefen_async", frei)
    for antwort, art in ((PENDING, sa.SicherheitAusstehend), (URTEIL, sa.GesperrteDatei)):
        c = httpx.AsyncClient(transport=httpx.MockTransport(lambda r, a=antwort: a), follow_redirects=False)
        with pytest.raises(sa.GesperrteDatei) as e:
            async with sa.oeffne(c, "https://api-storage.arkturian.com/storage/media/1"):
                pass
        assert type(e.value) is art


@pytest.mark.asyncio
async def test_flow_referenz_wartet(monkeypatch):
    zaehler = {"n": 0}

    class R:
        headers = {"content-type": "image/png"}

        async def aiter_bytes(self, n):
            yield b"PNG"

    @asynccontextmanager
    async def oeffne(c, url, mit_schluessel=False):
        zaehler["n"] += 1
        if zaehler["n"] < 3:
            raise sa.SicherheitAusstehend(url)
        yield R()

    async def flow(method, pfad, **k):
        return {"mediaGenerationId": {"mediaGenerationId": "m1"}}
    monkeypatch.setattr(sa, "oeffne", oeffne)
    monkeypatch.setattr(fr, "_flow", flow)
    monkeypatch.setattr(fr, "_SAFETY_TAKT_S", 0)
    assert await fr._bild_hochladen("4711") == "m1"
    assert zaehler["n"] == 3
