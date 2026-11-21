"""#3604: Eine halbe REALTIME_AUTH-Konfiguration faellt nicht auf uns zurueck.

Jeder Fall setzt Dienstschluessel und Profil, damit der alte Code wirklich bis
zum Netz gekommen waere. `kein_netz` schlaegt fehl, sobald ein Client entsteht.
"""
import asyncio
import importlib.util
import itertools
import logging
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ai.routes import jev_routes, verbrauch_routes
from ai.services import auth_konfig as ak
from ai.services import bild_verbrauch as bv
from ai.services import jwt_pruefung as jp
from ai.services import realtime_grant_verifier as rv

LOKAL = {
    ak.ISSUER_ENV: "kunde.example.com",
    ak.JWKS_ENV: "http://127.0.0.1:8010/api/v1/auth/.well-known/jwks.json",
    ak.GRANT_ENV: "http://127.0.0.1:8010/api/v1/auth/realtime-grant",
}
DREI = (ak.ISSUER_ENV, ak.JWKS_ENV, ak.GRANT_ENV)
HALB = [c for r in (1, 2) for c in itertools.combinations(DREI, r)]
BEARER = "Bearer kopf.inhalt.signatur"


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    for n in (*DREI, *ak.ZUSATZ_ENVS):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("REALTIME_PROFILE_ID", "kunde")
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienstschluessel")
    monkeypatch.setattr(ak, "_gemeldet", False)
    verbrauch_routes._ME_CACHE.clear()
    jp._zuruecksetzen()
    yield
    jp._zuruecksetzen()


@pytest.fixture
def kein_netz(monkeypatch):
    def verboten(*a, **k):
        raise AssertionError("Aufruf nach aussen")
    monkeypatch.setattr(httpx, "AsyncClient", verboten)
    monkeypatch.setattr(httpx, "get", verboten)


def setzen(monkeypatch, **werte):
    for k, v in werte.items():
        monkeypatch.setenv(k, v)


def test_genau_sechs_halbe_kombinationen():
    assert len(HALB) == 6


@pytest.mark.parametrize("namen", HALB)
def test_halb_tauscht_nicht(monkeypatch, kein_netz, namen):
    setzen(monkeypatch, **{n: LOKAL[n] for n in namen})
    assert ak.zustand() == ak.UNVOLLSTAENDIG
    assert ak.befund()["fehlt"] == [n for n in DREI if n not in namen]
    with pytest.raises(rv.AuthKonfigUnvollstaendig) as e:
        asyncio.run(rv.exchange_and_verify(BEARER, "mint"))
    assert e.value.status_code == 503
    assert e.value.error_code == "realtime_auth_unvollstaendig"


@pytest.mark.parametrize("namen", HALB)
def test_halb_schuetzt_auch_den_direkten_tausch(monkeypatch, kein_netz, namen):
    setzen(monkeypatch, **{n: LOKAL[n] for n in namen})
    with pytest.raises(rv.AuthKonfigUnvollstaendig):
        asyncio.run(rv._exchange_user_jwt_for_grant(BEARER))


@pytest.mark.parametrize("namen", HALB)
def test_halb_schickt_kein_bearer_an_me(monkeypatch, kein_netz, namen):
    setzen(monkeypatch, **{n: LOKAL[n] for n in namen})
    with pytest.raises(HTTPException) as e:
        asyncio.run(verbrauch_routes._profil(BEARER))
    assert e.value.status_code == 503
    assert e.value.detail == {"error": "realtime_auth_unvollstaendig"}


@pytest.mark.parametrize("namen", HALB)
def test_halb_prueft_kein_token(monkeypatch, namen):
    setzen(monkeypatch, **{n: LOKAL[n] for n in namen})
    monkeypatch.setattr(jp, "allgemein", lambda: pytest.fail("Pruefer gebaut"))
    with pytest.raises(HTTPException) as e:
        jp.pruefen(BEARER)
    assert e.value.status_code == 503


def test_halb_bricht_etiketten_nicht(monkeypatch):
    """Jev und Bildbuchung laufen weiter, nur ohne geprueften Namen."""
    setzen(monkeypatch, **{ak.GRANT_ENV: LOKAL[ak.GRANT_ENV]})
    assert jev_routes._jwt_sub_geprueft(BEARER) is None
    assert bv.aufrufer(BEARER, "AgentKopf") == ("AgentKopf", "header")


class _Antwort:
    status_code = 200

    def json(self):
        return {"approved": False, "reason_code": "not_enabled"}


def _fake_client(ziele):
    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None):
            ziele.append(url)
            return _Antwort()

    return Client


def _frisches_modul(monkeypatch):
    """Eigene Kopie: GRANT_URL wird beim Import gelesen, und ein reload
    wuerde die Fehlerklassen der uebrigen Suite austauschen."""
    pfad = Path(rv.__file__)
    spec = importlib.util.spec_from_file_location("rv_kopie_3604", pfad)
    modul = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, modul)  # fuer @dataclass
    spec.loader.exec_module(modul)
    return modul


def test_alle_drei_lokal_tauscht_lokal(monkeypatch):
    setzen(monkeypatch, **LOKAL)
    assert ak.zustand() == ak.EIGEN
    ziele = []
    m = _frisches_modul(monkeypatch)
    monkeypatch.setattr(m.httpx, "AsyncClient", _fake_client(ziele))
    with pytest.raises(m.GrantDenied):
        asyncio.run(m.exchange_and_verify(BEARER, "mint"))
    assert ziele == [LOKAL[ak.GRANT_ENV]]


def test_keiner_gesetzt_bleibt_vorgabe(monkeypatch):
    assert ak.zustand() == ak.VORGABE
    ziele = []
    m = _frisches_modul(monkeypatch)
    monkeypatch.setattr(m.httpx, "AsyncClient", _fake_client(ziele))
    with pytest.raises(m.GrantDenied):
        asyncio.run(m.exchange_and_verify(BEARER, "mint"))
    assert ziele == [ak.VORGABEN[ak.GRANT_ENV]]


def test_alle_drei_ausdruecklich_vorgabe(monkeypatch):
    setzen(monkeypatch, **ak.VORGABEN)
    assert ak.zustand() == ak.VORGABE


@pytest.mark.parametrize("name", DREI)
def test_einzelner_wert_gleich_vorgabe_ist_halb(monkeypatch, name):
    """Die Vorgabe gilt nur, wenn KEINER gesetzt ist."""
    setzen(monkeypatch, **{name: ak.VORGABEN[name]})
    assert ak.zustand() == ak.UNVOLLSTAENDIG


@pytest.mark.parametrize("name", DREI)
def test_leerer_wert_zaehlt_als_fehlend(monkeypatch, name):
    setzen(monkeypatch, **{**LOKAL, name: "  "})
    b = ak.befund()
    assert b["zustand"] == ak.UNVOLLSTAENDIG and b["fehlt"] == [name]


def test_alle_drei_leer_faellt_nicht_auf_vorgabe(monkeypatch):
    setzen(monkeypatch, **{n: "" for n in DREI})
    assert ak.zustand() == ak.UNVOLLSTAENDIG


@pytest.mark.parametrize("issuer,url", [
    ("ele.arkturian.com", "http://127.0.0.1:8010/x"),
    ("ele.arkturian.com", "https://ele.arkturian.com/api/v1/auth/x"),
    ("agentos.oneal.arkturian.com", "https://agentos.oneal.arkturian.com/x"),
    ("https://ele.arkturian.com", "https://ELE.arkturian.com./x"),
    ("agentos1", "http://localhost:8010/x"),
])
def test_kundennamen_unter_arkturian_sind_eigen(monkeypatch, issuer, url):
    setzen(monkeypatch, **{ak.ISSUER_ENV: issuer, ak.JWKS_ENV: url, ak.GRANT_ENV: url})
    assert ak.zustand() == ak.EIGEN


@pytest.mark.parametrize("issuer,feld,wert", [
    ("kunde.example.com", ak.GRANT_ENV, "https://auth-api.arkturian.com/api/v1/auth/realtime-grant"),
    ("kunde.example.com", ak.JWKS_ENV, "https://auth-api.arkturian.com/jwks.json"),
    ("ele.arkturian.com", ak.GRANT_ENV, "https://p3.arkturian.com/grant"),
    ("kunde.example.com", ak.GRANT_ENV, "https://arkturian.com/grant"),
    ("kunde.example.com", ak.GRANT_ENV, "ftp://kunde.example.com/grant"),
    ("kunde.example.com", ak.GRANT_ENV, "kunde.example.com/ohne-schema"),
    ("kunde.example.com", "AUTH_API_ME_URL", "https://auth-api.arkturian.com/api/v1/auth/me"),
    ("kunde.example.com", "AGENTOS_JWT_REVOCATION_URL", "https://auth-api.arkturian.com/r"),
])
def test_zeigt_auf_betreiber(monkeypatch, issuer, feld, wert):
    setzen(monkeypatch, **{**LOKAL, ak.ISSUER_ENV: issuer, feld: wert})
    b = ak.befund()
    assert b["zustand"] == ak.UNVOLLSTAENDIG and feld in b["zeigt_auf_betreiber"]


@pytest.mark.parametrize("issuer", ["auth-api.arkturian.com", "auth-api.arkserver.arkturian.com",
                                    "https://auth-api.arkturian.com", "arkturian.com"])
def test_aussteller_des_betreibers_mit_lokalen_adressen(monkeypatch, issuer):
    setzen(monkeypatch, **{**LOKAL, ak.ISSUER_ENV: issuer})
    assert ak.befund()["zeigt_auf_betreiber"] == [ak.ISSUER_ENV]


def test_journal_einmal_und_ohne_werte(monkeypatch, caplog):
    wert = "https://geheim.example/grant"
    setzen(monkeypatch, **{ak.GRANT_ENV: wert})
    with caplog.at_level(logging.ERROR, logger=ak.logger.name):
        assert ak.unvollstaendig() and ak.unvollstaendig()
    zeilen = [r.getMessage() for r in caplog.records if r.name == ak.logger.name]
    assert len(zeilen) == 1
    assert ak.ISSUER_ENV in zeilen[0] and "geheim" not in zeilen[0]


@pytest.mark.parametrize("env,erwartet", [
    ({}, "vorgabe"),
    (LOKAL, "eigen"),
    ({ak.ISSUER_ENV: LOKAL[ak.ISSUER_ENV]}, "unvollstaendig"),
])
def test_health_und_token_route(monkeypatch, env, erwartet):
    import main
    monkeypatch.setattr(main, "_cli_status", lambda name: {"ok": True})
    setzen(monkeypatch, **env)
    assert main.health_check()["realtime_auth"] == erwartet
    if erwartet == "unvollstaendig":
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: pytest.fail("Netz"))
        r = TestClient(main.app).post("/ai/realtime/token", headers={"Authorization": BEARER}, json={})
        assert r.status_code == 503
        assert "realtime_auth_unvollstaendig" in r.text


# --- Knowledge und ArTrack: Koordinaten gehen nicht an uns -----------------

KOORD = {"lat": 46.6211, "lon": 14.3055}
GEO = [
    ("knowledge_query", "_tool_knowledge_query", "KNOWLEDGE_API_URL",
     "https://knowledge-api.arkturian.com", "/api/v1/knowledge/near"),
    ("pois_near", "_tool_pois_near", "ARTRACK_API_URL",
     "https://api-artrack.arkturian.com", "/places/nearby/compact"),
    ("osm_nearby", "_tool_osm_nearby", "ARTRACK_API_URL",
     "https://api-artrack.arkturian.com", "/osm/nearby/compact"),
]


class _GeoAntwort:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"items": [], "text": "", "count": 0}


def _geo_client(ziele):
    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            ziele.append(url)
            return _GeoAntwort()

    return Client


@pytest.fixture
def rr(monkeypatch):
    from ai.routes import realtime_routes as m
    for n in ("KNOWLEDGE_API_URL", "ARTRACK_API_URL"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(m, "_dienst_fehlt_gemeldet", set())
    return m


def _nicht_eingerichtet(out):
    return out["status"] == "unavailable" and out["count"] == 0 and "NICHT" in out["hint"]


@pytest.mark.parametrize("tool,fn,env,vorgabe,pfad", GEO)
def test_eigene_instanz_ohne_dienst_schickt_nichts(monkeypatch, kein_netz, rr, tool, fn, env, vorgabe, pfad):
    setzen(monkeypatch, **LOKAL)
    assert _nicht_eingerichtet(asyncio.run(getattr(rr, fn)(dict(KOORD))))


@pytest.mark.parametrize("tool,fn,env,vorgabe,pfad", GEO)
def test_eigene_instanz_mit_betreiberadresse_schickt_nichts(monkeypatch, kein_netz, rr, tool, fn, env, vorgabe, pfad):
    setzen(monkeypatch, **{**LOKAL, env: vorgabe})
    assert _nicht_eingerichtet(asyncio.run(getattr(rr, fn)(dict(KOORD))))


@pytest.mark.parametrize("namen", HALB)
@pytest.mark.parametrize("tool,fn,env,vorgabe,pfad", GEO)
def test_halbe_konfiguration_schickt_nichts(monkeypatch, kein_netz, rr, tool, fn, env, vorgabe, pfad, namen):
    setzen(monkeypatch, **{n: LOKAL[n] for n in namen})
    assert _nicht_eingerichtet(asyncio.run(getattr(rr, fn)(dict(KOORD))))


@pytest.mark.parametrize("tool,fn,env,vorgabe,pfad", GEO)
def test_eigene_instanz_mit_eigenem_dienst(monkeypatch, rr, tool, fn, env, vorgabe, pfad):
    setzen(monkeypatch, **{**LOKAL, env: "http://127.0.0.1:9000/"})
    ziele = []
    monkeypatch.setattr(httpx, "AsyncClient", _geo_client(ziele))
    asyncio.run(getattr(rr, fn)(dict(KOORD)))
    assert ziele == [f"http://127.0.0.1:9000{pfad}"]


def test_eigener_name_unter_arkturian_bleibt_erlaubt(monkeypatch, rr):
    setzen(monkeypatch, **{**LOKAL, ak.ISSUER_ENV: "ele.arkturian.com",
                           "ARTRACK_API_URL": "https://ele.arkturian.com/artrack"})
    ziele = []
    monkeypatch.setattr(httpx, "AsyncClient", _geo_client(ziele))
    asyncio.run(rr._tool_osm_nearby(dict(KOORD)))
    assert ziele == ["https://ele.arkturian.com/artrack/osm/nearby/compact"]


@pytest.mark.parametrize("tool,fn,env,vorgabe,pfad", GEO)
def test_unser_kern_nutzt_die_vorgabe(monkeypatch, rr, tool, fn, env, vorgabe, pfad):
    ziele = []
    monkeypatch.setattr(httpx, "AsyncClient", _geo_client(ziele))
    asyncio.run(getattr(rr, fn)(dict(KOORD)))
    assert ziele == [f"{vorgabe}{pfad}"]


def test_knowledge_ohne_koordinaten_bleibt_wie_bisher(monkeypatch, kein_netz, rr):
    setzen(monkeypatch, **LOKAL)
    out = asyncio.run(rr._tool_knowledge_query({}))
    assert out["count"] == 0 and "status" not in out


def test_meldung_einmal_und_ohne_koordinaten(monkeypatch, kein_netz, rr, caplog):
    setzen(monkeypatch, **LOKAL)
    with caplog.at_level(logging.WARNING, logger=rr.logger.name):
        asyncio.run(rr._tool_osm_nearby(dict(KOORD)))
        asyncio.run(rr._tool_pois_near(dict(KOORD)))
    zeilen = [r.getMessage() for r in caplog.records if "ARTRACK_API_URL" in r.getMessage()]
    assert len(zeilen) == 1 and "46.62" not in zeilen[0]


def test_werkzeug_route_ohne_dienst(monkeypatch, kein_netz, rr):
    import main
    setzen(monkeypatch, **LOKAL)
    r = TestClient(main.app).post("/ai/realtime/tool/osm_nearby",
                                  json={"call_id": "c1", "arguments": dict(KOORD)})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["result"]["status"] == "unavailable"
