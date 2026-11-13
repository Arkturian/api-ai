"""Bilder je Aufrufer (Auftrag Alex 13.11.2026 ueber CloudV2).

Vorher fuehrte api-ai nur eine Monatssumme ohne Aufrufer; das
Info-Fenster eines Agenten konnte keine Bilder zeigen.
"""
import asyncio
import json
import os
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from ai.routes import image_ai_routes as g
from ai.routes import verbrauch_routes as vr
from ai.services import bild_verbrauch as bv
from ai.services import codex_imagegen as c


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


# --- Wer ruft? -----------------------------------------------------------

def test_aufrufer_jwt_gewinnt_gegen_den_kopf(monkeypatch):
    monkeypatch.setattr(bv, "_claims_geprueft", lambda a: {"sub": "agent:Marketing"} if a else None)
    assert bv.aufrufer("Bearer x", "Gefaelscht") == ("Marketing", "jwt")


def test_aufrufer_mensch_ist_die_mail(monkeypatch):
    monkeypatch.setattr(bv, "_claims_geprueft", lambda a: {"sub": "17", "email": "Alex@Example.com"})
    assert bv.aufrufer("Bearer x", None) == ("alex@example.com", "jwt")


def test_aufrufer_ohne_gueltiges_jwt(monkeypatch):
    monkeypatch.setattr(bv, "_claims_geprueft", lambda a: None)
    assert bv.aufrufer("Bearer kaputt", "agent:Story") == ("Story", "header")
    assert bv.aufrufer(None, None) == (bv.UNBEKANNT, "keine")


# --- Buchen und Bericht ----------------------------------------------------

def test_bericht_je_wiener_tag_mit_abo_bezahlt_und_kosten():
    bv.buchen(agent="Marketing", quelle="jwt", model="codex-imagegen", routed_via="subscription",
              kosten_usd=None, jetzt=_ts("2026-11-13T10:00:00+00:00"))
    bv.buchen(agent="Marketing", quelle="jwt", model="gpt-image-2.6-sunburst", routed_via="api",
              kosten_usd=0.21, jetzt=_ts("2026-11-13T11:00:00+00:00"))
    # 23:30 UTC ist in Wien schon der naechste Tag
    bv.buchen(agent="Marketing", quelle="jwt", model="codex-imagegen", routed_via="subscription",
              kosten_usd=None, jetzt=_ts("2026-11-13T23:30:00+00:00"))
    bv.buchen(agent="Story", quelle="header", model="minimax-image-01", routed_via=None,
              kosten_usd=None, jetzt=_ts("2026-11-13T12:00:00+00:00"))
    r = bv.bericht()
    assert r["seit"] == "2026-11-13T10:00:00+00:00"
    assert r["zeilen"] == [
        {"t": "2026-11-13", "agent": "Marketing", "art": "bild", "anzahl": 2, "abo": 1, "bezahlt": 1, "kosten_usd": 0.21},
        {"t": "2026-11-13", "agent": "Story", "art": "bild", "anzahl": 1, "abo": 0, "bezahlt": 1, "kosten_usd": 0.0,
         "kosten_unbekannt": 1},
        {"t": "2026-11-14", "agent": "Marketing", "art": "bild", "anzahl": 1, "abo": 1, "bezahlt": 0, "kosten_usd": 0.0},
    ]


def test_bericht_filter_und_seit_bleibt_der_erste_eintrag():
    for t, a in [("2026-11-01T09:00:00+00:00", "A"), ("2026-11-13T09:00:00+00:00", "B")]:
        bv.buchen(agent=a, quelle="jwt", model="m", routed_via="subscription", kosten_usd=None, jetzt=_ts(t))
    r = bv.bericht(agent="B", von=datetime(2026, 11, 10, tzinfo=timezone.utc))
    assert [z["agent"] for z in r["zeilen"]] == ["B"]
    assert r["seit"].startswith("2026-11-01")                 # nicht vom Filter verschoben
    assert bv.bericht(bis=datetime(2026, 11, 1, tzinfo=timezone.utc))["zeilen"] == []


def test_ohne_eintraege_seit_ist_none():
    assert bv.bericht() == {"seit": None, "zeilen": []}


def test_buchen_wirft_nie(monkeypatch, tmp_path):
    f = tmp_path / "datei"
    f.write_text("x")
    monkeypatch.setenv("VERBRAUCH_DIR", str(f))                # kein Verzeichnis
    bv.buchen(agent="A", quelle="jwt", model="m", routed_via="api", kosten_usd=1.0)


# --- genimage bucht ------------------------------------------------------

def _verdrahten(monkeypatch):
    async def fake_abo(**kw):
        return {"id": 7, "image_url": "u", "file_url": "u", "storage_object_id": 7, "request_id": "codex_x",
                "width": 1024, "height": 1024, "billing": "subscription"}

    async def fake_api(**kw):
        return {"id": 8, "image_url": "p", "file_url": "p", "storage_object_id": 8, "cost_usd": 0.1234}

    monkeypatch.setattr(c, "generate_with_codex_imagegen", fake_abo)
    monkeypatch.setattr(g, "generate_with_openai_image", fake_api)
    monkeypatch.setattr(g, "_check_openai_billing_gate", lambda *a, **k: None)
    monkeypatch.setattr(bv, "_claims_geprueft", lambda a: {"sub": "agent:Marketing"} if a else None)


def _zeilen():
    d = os.environ["VERBRAUCH_DIR"]
    out = []
    for p in sorted(os.listdir(d)):
        out += [json.loads(z) for z in open(os.path.join(d, p))]
    return out


def test_genimage_bucht_abo_und_bezahlt_beim_aufrufer(monkeypatch):
    _verdrahten(monkeypatch)
    asyncio.run(g.generate_image_endpoint(g.ImageGenRequest(prompt="x", model="gpt-image-2"), "k",
                                          authorization="Bearer t", x_agent_name=None))
    asyncio.run(g.generate_image_endpoint(
        g.ImageGenRequest(prompt="x", model="gpt-image-2", route="api", confirm_api_billing=True), "k",
        authorization=None, x_agent_name="Story"))
    z = _zeilen()
    assert [(r["agent"], r["quelle"], r["routed_via"], r["kosten_usd"], r["storage_id"]) for r in z] == [
        ("Marketing", "jwt", "subscription", 0.0, 7),
        ("Story", "header", "api", 0.1234, 8),
    ]


def test_genimage_replay_zaehlt_nicht_doppelt(monkeypatch):
    _verdrahten(monkeypatch)
    from ai.services import narrate_jobs as nj
    gespeichert = {}
    monkeypatch.setattr(nj, "vorab", lambda rid, art, h: gespeichert.get(rid))
    monkeypatch.setattr(nj, "abschliessen", lambda rid, out: gespeichert.__setitem__(rid, out))
    req = g.ImageGenRequest(prompt="x", model="gpt-image-2", request_id="job-1")
    for _ in range(2):
        asyncio.run(g.generate_image_endpoint(req, "k", authorization="Bearer t", x_agent_name=None))
    assert len(_zeilen()) == 1 and _zeilen()[0]["request_id"] == "job-1"


def test_interner_direktaufruf_ist_intern(monkeypatch):
    """speech_service / audio_drama rufen den Endpunkt direkt als Funktion."""
    _verdrahten(monkeypatch)
    asyncio.run(g.generate_image_endpoint(g.ImageGenRequest(prompt="x", model="gpt-image-2"), "k"))
    assert _zeilen()[0]["agent"] == bv.INTERN and _zeilen()[0]["quelle"] == "intern"


def test_fehlschlag_bucht_nichts(monkeypatch):
    _verdrahten(monkeypatch)

    async def kaputt(**kw):
        raise HTTPException(status_code=504, detail={"error": "codex_imagegen_timeout"})
    monkeypatch.setattr(c, "generate_with_codex_imagegen", kaputt)
    with pytest.raises(HTTPException):
        asyncio.run(g.generate_image_endpoint(g.ImageGenRequest(prompt="x", model="gpt-image-2", route="subscription"),
                                              "k", authorization="Bearer t", x_agent_name=None))
    assert not os.path.isdir(os.environ["VERBRAUCH_DIR"]) or _zeilen() == []


# --- Leseroute -----------------------------------------------------------

def _lies(monkeypatch, profil, **q):
    async def fake_profil(a):
        return profil
    monkeypatch.setattr(vr, "_profil", fake_profil)
    return asyncio.run(vr.bilder(agent=q.get("agent"), von=q.get("von"), bis=q.get("bis"),
                                 authorization="Bearer t"))


def test_leseroute_nur_fuer_admins(monkeypatch):
    with pytest.raises(HTTPException) as e:
        _lies(monkeypatch, None)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException) as e:
        _lies(monkeypatch, {"email": "x@y", "agentos_role": "developer"})
    assert e.value.status_code == 403
    bv.buchen(agent="A", quelle="jwt", model="m", routed_via="subscription", kosten_usd=None,
              jetzt=_ts("2026-11-13T09:00:00+00:00"))
    for profil in ({"agentos_role": "admin"}, {"is_superadmin": True}):
        r = _lies(monkeypatch, profil)
        assert json.loads(r.body)["zeilen"][0]["agent"] == "A"


def test_leseroute_zeit_ohne_zone_ist_wien(monkeypatch):
    bv.buchen(agent="A", quelle="jwt", model="m", routed_via="subscription", kosten_usd=None,
              jetzt=_ts("2026-11-12T23:30:00+00:00"))      # Wien: 13.11. 00:30
    r = json.loads(_lies(monkeypatch, {"agentos_role": "admin"}, von="2026-11-13T00:00:00").body)
    assert [z["t"] for z in r["zeilen"]] == ["2026-11-13"]


def test_me_adresse_folgt_der_konfigurierten_auth_api(monkeypatch):
    monkeypatch.delenv("AUTH_API_ME_URL", raising=False)
    assert vr._me_url().endswith("/api/v1/auth/me")
