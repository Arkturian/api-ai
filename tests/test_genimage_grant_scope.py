"""grant_scope und ttl_hours bei /ai/genimage (cloud-api/3DNerd, 11.10.).

Gegen den echten Fehler gehalten: api-ai legte jedes Ergebnis ohne Scope
unter dem eigenen Upload-Konto ab. Ein privates Ergebnis war damit fuer
den Menschen, der es bestellt hat, nicht lesbar — der Storage signiert nur
fuer Objekte mit Bindung an den angefragten Scope.
"""
import asyncio
import inspect
import re

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ai.clients import storage_client as sc
from ai.routes import image_ai_routes as g
from ai.services import codex_imagegen as c


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


def test_client_schickt_scope_und_verfall_als_formularfelder(monkeypatch):
    store = {}
    monkeypatch.setattr(sc.httpx, "AsyncClient", _client(store))
    asyncio.run(sc.save_file_and_record(b"x", "x.png", private=True, grant_scope="cloud-user:17", ttl_hours=24))
    assert store["data"]["grant_scope"] == "cloud-user:17" and store["data"]["ttl_hours"] == "24"


def test_client_ohne_angabe_schickt_keines_der_felder(monkeypatch):
    store = {}
    monkeypatch.setattr(sc.httpx, "AsyncClient", _client(store))
    asyncio.run(sc.save_file_and_record(b"x", "x.png"))
    assert "grant_scope" not in store["data"] and "ttl_hours" not in store["data"]


def _req(**kw):
    basis = dict(prompt="ein Tor", model="gpt-image-2")
    basis.update(kw)
    return g.ImageGenRequest(**basis)


def _verdrahten(monkeypatch):
    aufrufe = {"abo": [], "api": []}

    async def fake_abo(**kw):
        aufrufe["abo"].append(kw)
        return {"id": 1, "image_url": "u", "file_url": "u", "storage_object_id": 1,
                "width": 1024, "height": 1024, "provider": "codex-imagegen", "billing": "subscription"}

    async def fake_api(**kw):
        aufrufe["api"].append(kw)
        return {"id": 2, "image_url": "p", "file_url": "p", "storage_object_id": 2,
                "width": 1024, "height": 1024, "provider": "openai"}

    monkeypatch.setattr(c, "generate_with_codex_imagegen", fake_abo)
    monkeypatch.setattr(g, "generate_with_openai_image", fake_api)
    monkeypatch.setattr(g, "_check_openai_billing_gate", lambda *a, **k: None)
    return aufrufe


@pytest.mark.parametrize("scope", ["Cloud-User:17", "cloud-user", "cloud-user:", ":17",
                                   "cloud-user:alex@example.org", "cloud user:17", "x" * 42 + ":1"])
def test_ungueltiger_scope_422_bevor_ein_bild_entsteht(monkeypatch, scope):
    aufrufe = _verdrahten(monkeypatch)
    with pytest.raises(HTTPException) as e:
        asyncio.run(g._generate_image_einmal(_req(route="subscription", grant_scope=scope), "x"))
    assert e.value.status_code == 422 and e.value.detail["error"] == "invalid_grant_scope"
    assert aufrufe["abo"] == [] and aufrufe["api"] == []


def test_scope_und_verfall_erreichen_den_abo_weg_und_die_antwort(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    out = asyncio.run(g._generate_image_einmal(
        _req(route="subscription", private=True, grant_scope=" cloud-user:17 ", ttl_hours=48), "x"))
    kw = aufrufe["abo"][0]
    assert kw["grant_scope"] == "cloud-user:17" and kw["ttl_hours"] == 48 and kw["private"] is True
    assert out["grant_scope"] == "cloud-user:17" and out["ttl_hours"] == 48


def test_scope_erreicht_auch_den_bezahlten_weg(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    asyncio.run(g._generate_image_einmal(
        _req(route="api", confirm_api_billing=True, grant_scope="cloud-user:17"), "x"))
    assert aufrufe["api"][0]["grant_scope"] == "cloud-user:17"


def test_ohne_scope_bleibt_alles_wie_bisher(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    out = asyncio.run(g._generate_image_einmal(_req(route="subscription"), "x"))
    assert aufrufe["abo"][0]["grant_scope"] is None and aufrufe["abo"][0]["ttl_hours"] is None
    assert out["grant_scope"] is None


@pytest.mark.parametrize("wert", [0, -1, 8761])
def test_verfall_ausserhalb_der_grenzen_wird_abgewiesen(wert):
    with pytest.raises(ValidationError):
        _req(ttl_hours=wert)


GENERATOREN = [g.generate_with_openai_image, g.generate_with_higgsfield,
               g.generate_with_minimax, g.generate_with_gemini, c.generate_with_codex_imagegen]


@pytest.mark.parametrize("fn", GENERATOREN, ids=lambda f: f.__name__)
def test_jeder_ablageweg_reicht_scope_und_verfall_weiter(fn):
    sig = inspect.signature(fn)
    assert "grant_scope" in sig.parameters and "ttl_hours" in sig.parameters
    assert re.search(r"grant_scope=grant_scope,\s*ttl_hours=ttl_hours,", inspect.getsource(fn))


def test_muster_ist_das_des_storage():
    # storage/signing.py SCOPE_RE, gelesen 11.10. (dev @ 90e8be2)
    assert g._SCOPE_RE.pattern == r"^[a-z][a-z0-9-]{0,40}:[A-Za-z0-9._-]{1,120}$"
    assert g._SCOPE_RE.match("cloud-user:17") and g._SCOPE_RE.match("cloud-session:AiApi")
