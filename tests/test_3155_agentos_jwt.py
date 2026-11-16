"""#3155 (#2972/#2970): api-ai prueft JWTs mit agentos_jwt v1.0.1.

Vorher: PyJWKClient ohne Widerruf und ohne Typ-Allowlist. Jedes von
auth-api signierte Token galt als angemeldet, auch ein widerrufenes
Agenten-Token oder ein Realtime-Grant.
"""
import hashlib
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from ai import agentos_jwt as aj
from ai import auth_guard as ag
from ai.services import bild_verbrauch as bv
from ai.services import jwt_pruefung as jp
from ai.services import realtime_grant_verifier as rv

ISS = "auth-api.arkturian.com"
SCHLUESSEL = rsa.generate_private_key(public_exponent=65537, key_size=2048)
FREMD = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks():
    k = RSAAlgorithm.to_jwk(SCHLUESSEL.public_key(), as_dict=True)
    k.update(kid="auth-primary", use="sig", alg="RS256")
    return {"keys": [k]}


def _token(typ="agent", kid="auth-primary", key=SCHLUESSEL, **extra):
    jetzt = int(time.time())
    claims = {"sub": "agent:Comm", "iss": ISS, "iat": jetzt - 10, "exp": jetzt + 600,
              "jti": "jti-gut", "type": typ, **extra}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def netz(monkeypatch, tmp_path):
    """auth-api als Attrappe: JWKS und Widerrufsliste ueber den fetch-Haken."""
    zustand = {"down": False, "widerrufen": []}

    def fetch(url, params):
        if zustand["down"]:
            raise OSError("auth-api weg")
        if url.endswith("jwks.json"):
            return _jwks()
        if url.endswith("revoked-agent-jtis"):
            return {"revoked_jtis": [{"jti": j, "revoked_at": "2026-11-16T08:00:00Z"}
                                     for j in zustand["widerrufen"]],
                    "as_of": "2026-11-16T09:00:00Z", "min_agent_iat": None, "owner_min_iat": {}}
        raise AssertionError(url)

    echt = jp.Verifier

    class MitAttrappe(echt):
        def __init__(self, **kw):
            super().__init__(fetch=fetch, **kw)

    monkeypatch.setattr(jp, "Verifier", MitAttrappe)
    monkeypatch.setenv("AGENTOS_JWT_STATE_DIR", str(tmp_path / "agentos_jwt"))
    jp._zuruecksetzen()
    yield zustand
    jp._zuruecksetzen()


@pytest.fixture
def client(netz):
    app = FastAPI()

    @app.get("/x")
    def x(wer: str = Depends(ag.angemeldet)):
        return {"wer": wer}

    return TestClient(app)


def _get(client, token):
    return client.get("/x", headers={"Authorization": f"Bearer {token}"})


# --- Pruefpunkte aus #3152 --------------------------------------------------------

def test_gueltiges_agenten_token_200(client):
    r = _get(client, _token())
    assert r.status_code == 200 and r.json() == {"wer": "Comm"}


def test_unsigniertes_token_mit_typ_liste_401_nicht_500(client):
    roh = jwt.encode({"sub": "agent:X", "iss": ISS, "type": ["agent"], "iat": 1, "exp": 9999999999},
                     key="", algorithm="none")
    assert _get(client, roh).status_code == 401


def test_widerrufene_jti_401(client, netz):
    netz["widerrufen"] = ["jti-gut"]
    r = _get(client, _token())
    assert r.status_code == 401


def test_unbekannte_kid_401(client):
    assert _get(client, _token(kid="fremd", key=FREMD)).status_code == 401


def test_fremder_schluessel_mit_bekannter_kid_401(client):
    assert _get(client, _token(key=FREMD)).status_code == 401


@pytest.mark.parametrize("typ,extra", [
    ("realtime_grant", {"aud": "ai-realtime:tscheppaschlucht"}),
    ("oauth_access", {"aud": "chatgpt-connector"}),
    ("ephemeral_capability", {}),
])
def test_nicht_angenommene_klasse_401(client, typ, extra):
    assert _get(client, _token(typ=typ, **extra)).status_code == 401


def test_delegated_nur_mit_aud_ai_api(client):
    assert _get(client, _token(typ="delegated", aud="ai-api", sub="17", email="a@b.at")).status_code == 200
    assert _get(client, _token(typ="delegated", aud="content-api")).status_code == 401


def test_agenten_token_mit_aud_401(client):
    assert _get(client, _token(aud="ai-api")).status_code == 401


def test_auth_api_nie_erreicht_503_nicht_401(client, netz):
    netz["down"] = True
    r = _get(client, _token())
    assert r.status_code == 503


def test_etikett_wirft_nie(netz):
    netz["down"] = True
    assert bv.aufrufer(f"Bearer {_token()}", "Story") == ("Story", "header")


def test_state_dir_nicht_anlegbar_503_und_etikett_none(monkeypatch, tmp_path):
    sperre = tmp_path / "datei"
    sperre.write_text("x")
    monkeypatch.setenv("AGENTOS_JWT_STATE_DIR", str(sperre / "unter"))
    jp._zuruecksetzen()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        jp.pruefen(f"Bearer {_token()}")
    assert e.value.status_code == 503
    assert jp.claims_oder_none(f"Bearer {_token()}") is None


def test_ohne_bearer_bleibt_betreiber_oder_401(client):
    assert client.get("/x").status_code == 401


# --- Stimme: realtime_grant nur dort, nur mit passender aud ------------------------

def _grant(aud="ai-realtime:p1", **extra):
    return _token(typ="realtime_grant", aud=aud, profile_id="p1", scopes=["realtime:connect"],
                  limits={"max_parallel_sessions": 2, "daily_budget_eur": 5}, **extra)


def test_grant_mit_passender_aud(netz):
    g = rv._verify_grant_jwt(_grant(), {"p1"})
    assert g.profile_id == "p1" and g.max_parallel_sessions == 2


def test_grant_mit_fremder_aud(netz):
    with pytest.raises(rv.GrantWrongProfile):
        rv._verify_grant_jwt(_grant(aud="ai-realtime:p2"), {"p1"})


def test_agenten_token_ist_kein_grant(netz):
    with pytest.raises(rv.GrantUnverifiable):
        rv._verify_grant_jwt(_token(), {"p1"})


# --- Vendoring ----------------------------------------------------------------------

def test_vendorte_datei_unveraendert():
    pfad = Path(aj.__file__)
    assert hashlib.sha256(pfad.read_bytes()).hexdigest() == \
        "ff39755338d7e96d24e2b6a5bdfb6f16e4b6b5f07e7cd541f706f2c6dee2063d"
    assert aj.__version__ == "1.0.1"


def test_kein_alter_pruefweg_mehr():
    import ai.services.realtime_grant_verifier as m
    assert not hasattr(m, "_jwks") and not hasattr(m, "PyJWKClient")
