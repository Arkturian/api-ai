"""#3188: „Bearer “ ohne Token ergab auf den Realtime-Grant-Wegen 500 statt 401.

Gemessen am 16.11.2026 beim Cutover #2937: httpx lehnte den weitergereichten
Header ab (LocalProtocolError), und das fing niemand.
"""
import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from ai.routes import realtime_routes as rr
from ai.services import realtime_grant_verifier as rv


class _KeinNetz:
    """Jeder Aufbau eines Clients ist ein Fehler: auth-api darf nie gefragt werden."""

    def __init__(self, *a, **kw):
        raise AssertionError("auth-api wurde trotz ungueltigem Bearer angefragt")


@pytest.fixture
def ohne_netz(monkeypatch):
    monkeypatch.setenv(rv.SERVICE_KEY_ENV, "dienst-attrappe")
    monkeypatch.setenv(rv.REALTIME_PROFILE_ID_ENV, "p1")
    monkeypatch.setattr(rv.httpx, "AsyncClient", _KeinNetz)


@pytest.mark.parametrize("kopf", ["Bearer ", "Bearer  ", "bearer \x01abc", "Bearer a b"])
@pytest.mark.asyncio
async def test_ungueltiger_bearer_401_ohne_aufruf(ohne_netz, kopf):
    with pytest.raises(rv.GrantDenied) as e:
        await rv._exchange_user_jwt_for_grant(kopf)
    assert e.value.status_code == 401
    assert e.value.error_code == "realtime_user_jwt_required"


@pytest.mark.asyncio
async def test_local_protocol_error_wird_401(monkeypatch):
    """Zweite Sperre: was httpx trotzdem ablehnt, ist ein Fehler des Aufrufers."""
    monkeypatch.setenv(rv.SERVICE_KEY_ENV, "dienst-attrappe")

    class _Lehnt:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            raise httpx.LocalProtocolError("Illegal header value")

    monkeypatch.setattr(rv.httpx, "AsyncClient", _Lehnt)
    with pytest.raises(rv.GrantDenied) as e:
        await rv._exchange_user_jwt_for_grant("Bearer abc.def.ghi")
    assert e.value.status_code == 401


def test_route_leerer_bearer_401_nicht_500(ohne_netz):
    app = FastAPI()

    @app.post("/hb")
    async def hb(grant=Depends(rr.require_realtime_grant("mint"))):
        return {"ok": True}

    r = TestClient(app, raise_server_exceptions=False).post(
        "/hb", headers={"Authorization": "Bearer "})
    assert r.status_code == 401, r.text
    assert r.json()["detail"]["error"] in ("realtime_user_jwt_required",)
