"""Sicherheitsbefund Jev 28.09.2026: /ai/jev war ohne Anmeldung nutzbar
(API_ACCESS_KEY auf keinem Host gesetzt, Schluesseltor #1755 unscharf).
Jetzt: gepruefetes Agenten-JWT Pflicht, 401 VOR der Validierung und vor
jedem Anbieteraufruf; gilt auch fuer /ai/jev/cost-status."""
import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import jev_routes as j


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_USAGE_DIR", str(tmp_path))
    monkeypatch.setattr(j, "_jwt_sub_geprueft", lambda a: "agent:Jev" if a == "Bearer gut" else None)
    aufrufe = []
    async def fake_aufrufen(body, **k):
        aufrufe.append(body)
        return 200, {"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1}}, {}
    from ai.services import jev_service as s
    monkeypatch.setattr(s, "aufrufen", fake_aufrufen)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    c = TestClient(main.app)
    c.aufrufe = aufrufe
    return c


GUT = {"state": "t", "questions": {"a": {"type": "noul", "instructions": "?"}}}


@pytest.mark.parametrize("hdr", [{}, {"Authorization": "Bearer gefaelscht"}, {"Authorization": "Basic abc"},
                                 {"X-Agent-Name": "Jev"}, {"X-API-KEY": "irgendwas"}])
def test_ohne_gueltiges_jwt_401_vor_validierung(client, hdr):
    r = client.post("/ai/jev", json={"state": "x", "questions": {}}, headers=hdr)   # waere 422
    assert r.status_code == 401 and r.json()["detail"]["error"] == "jwt_required"
    assert client.aufrufe == []


def test_cost_status_braucht_jwt(client):
    assert client.get("/ai/jev/cost-status").status_code == 401
    assert client.get("/ai/jev/cost-status", headers={"Authorization": "Bearer gut"}).status_code == 200


def test_mit_gueltigem_jwt_durch(client):
    r = client.post("/ai/jev", json=GUT, headers={"Authorization": "Bearer gut"})
    assert r.status_code == 200 and len(client.aufrufe) == 1
    assert client.post("/ai/jev", json={"state": "x", "questions": {}},
                       headers={"Authorization": "Bearer gut"}).status_code == 422


def test_andere_pfade_unberuehrt(client):
    assert client.get("/health").status_code == 200
