"""Vertrauliche Ablage bei /ai/genimage (3DNerd, 12.10.).

Gegen den echten Fehler gehalten: `is_public=True` war in allen fuenf
Ablagewegen fest verdrahtet. Ein Produktbild aus einem vertraulichen
Kunden-Screenshot waere per Storage-ID ohne Anmeldung abrufbar gewesen.
"""
import asyncio
import inspect
import re

import pytest

from ai.clients import storage_client as sc
from ai.routes import image_ai_routes as r
from ai.services import codex_imagegen as ci


class _Antwort:
    def raise_for_status(self):
        return None

    def json(self):
        return {"id": 7, "object_key": "k", "original_filename": "x.png", "file_url": "u"}


def _client(store):
    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, files=None, data=None, headers=None):
            store["data"] = dict(data)
            return _Antwort()
    return _C


def test_client_schickt_private_und_nie_oeffentlich(monkeypatch):
    store = {}
    monkeypatch.setattr(sc.httpx, "AsyncClient", _client(store))
    # is_public=True vom Aufrufer darf private NICHT aushebeln.
    asyncio.run(sc.save_file_and_record(b"x", "x.png", is_public=True, private=True))
    assert store["data"]["private"] == "true"
    assert store["data"]["is_public"] == "false"


def test_client_ohne_private_bleibt_wie_bisher(monkeypatch):
    store = {}
    monkeypatch.setattr(sc.httpx, "AsyncClient", _client(store))
    asyncio.run(sc.save_file_and_record(b"x", "x.png", is_public=True))
    assert "private" not in store["data"] and store["data"]["is_public"] == "true"


GENERATOREN = [
    r.generate_with_openai_image, r.generate_with_higgsfield,
    r.generate_with_minimax, r.generate_with_gemini,
    ci.generate_with_codex_imagegen,
]


@pytest.mark.parametrize("fn", GENERATOREN, ids=lambda f: f.__name__)
def test_jeder_ablageweg_nimmt_private_und_reicht_es_weiter(fn):
    sig = inspect.signature(fn)
    assert "private" in sig.parameters and sig.parameters["private"].default is False
    quelle = inspect.getsource(fn)
    # Kein Weg darf das Ergebnis fest oeffentlich ablegen.
    assert "is_public=True" not in quelle
    assert re.search(r"is_public=not private,\s*private=private,", quelle)


def test_handler_reicht_private_an_alle_fuenf_wege():
    quelle = inspect.getsource(r._generate_image_einmal)
    for aufruf in ("generate_with_higgsfield(", "generate_with_gemini(", "generate_with_minimax(",
                   "generate_with_codex_imagegen(", "generate_with_openai_image("):
        block = quelle[quelle.index("await " + aufruf) if ("await " + aufruf) in quelle
                       else quelle.index("await _ci." + aufruf):][:700]
        assert "private=bool(request.private)" in block, aufruf
    assert '"private": bool(request.private)' in quelle


def test_vorgabe_ist_oeffentlich_wie_bisher():
    req = r.ImageGenRequest(prompt="x")
    assert req.private is False
    assert r.ImageGenRequest(prompt="x", private=True).private is True
