"""6586/I5: Dienstweg je Aufrufer (Client der cloud-api) und Sperre fuer Kunden-Mandanten.

- dienstweg.aufloesen haelt den Vertrag von GET /api/dienstwege/aufloesen ein
  (Dienst-JWT, 404 -> 403 kein_dienstweg, 409 -> 403 mit Grund, sonst 503).
- mandant_wege: Ein Kunden-Mandant (gepruftes JWT) rechnet auf dem Kern nie ueber
  Abos oder Schluessel des Betreibers. `beobachten` vermerkt nur, `an` weist ab.
"""
import asyncio
import json
import subprocess

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.services import dienstweg, jwt_pruefung, mandant_wege, tresor


# --- Client gegen den Vertrag der cloud-api ---------------------------------------

class _Antwort:
    def __init__(self, status, daten=None):
        self.status_code = status
        self._d = daten

    def json(self):
        if self._d is None:
            raise ValueError("kein json")
        return self._d


@pytest.fixture
def cloud(monkeypatch):
    dienstweg.leeren()
    monkeypatch.setenv("DIENSTWEG_URL", "https://cloud.test")
    gesehen = {"rufe": [], "antwort": _Antwort(200, {
        "weg": "api", "anbieter": "openai", "modell": "gpt-6-luna", "zugang": "z_1", "version": 3,
        "tresor_verweis": "zugang:ab12", "quelle": "mensch"})}

    async def token(mandant):
        return "dienst-jwt"
    monkeypatch.setattr(tresor, "_dienst_token", token)

    async def get(self, url, params=None, headers=None, **kw):
        gesehen["rufe"].append((url, dict(params or {}), dict(headers or {})))
        a = gesehen["antwort"]
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    return gesehen


def _aufloesen(*a, **kw):
    return asyncio.run(dienstweg.aufloesen(*a, **kw))


def test_vertrag_200(cloud):
    w = _aufloesen("lesser", "conversation", "Manuel@Lesser.at")
    assert (w.weg, w.anbieter, w.tresor_verweis, w.quelle) == ("api", "openai", "zugang:ab12", "mensch")
    url, params, kopf = cloud["rufe"][0]
    assert url == "https://cloud.test/api/dienstwege/aufloesen"
    assert params == {"mandant": "lesser", "dienst": "conversation", "mensch": "manuel@lesser.at"}
    assert kopf == {"Authorization": "Bearer dienst-jwt"}


def test_ohne_mensch_kein_parameter_und_zwischenspeicher(cloud):
    _aufloesen("lesser", "bild")
    _aufloesen("lesser", "bild")
    assert len(cloud["rufe"]) == 1 and "mensch" not in cloud["rufe"][0][1]


@pytest.mark.parametrize("antwort,status,fehler,grund", [
    (_Antwort(404, {"detail": {"error": "kein_dienstweg"}}), 403, "dienstweg_nicht_nutzbar", "kein_dienstweg"),
    (_Antwort(409, {"detail": {"error": "zugang_nicht_nutzbar", "grund": "inaktiv"}}), 403,
     "dienstweg_nicht_nutzbar", "inaktiv"),
    (_Antwort(409, {"detail": {"grund": "etwas_neues"}}), 403, "dienstweg_nicht_nutzbar", "nicht_nutzbar"),
    (_Antwort(403, {"detail": {"error": "aufloesen_verweigert"}}), 503, "dienstweg_nicht_lesbar",
     "aufloesen_verweigert"),
    (_Antwort(500), 503, "dienstweg_nicht_lesbar", "cloud_api_500"),
    (_Antwort(200, {"anbieter": "openai"}), 503, "dienstweg_nicht_lesbar", "antwort_ohne_weg"),
    (httpx.ConnectError("weg"), 503, "dienstweg_nicht_lesbar", "cloud_api_nicht_erreichbar"),
])
def test_vertrag_fehler(cloud, antwort, status, fehler, grund):
    cloud["antwort"] = antwort
    with pytest.raises(HTTPException) as e:
        _aufloesen("lesser", "conversation")
    assert e.value.status_code == status
    assert e.value.detail["error"] == fehler and e.value.detail["grund"] == grund


def test_ohne_dienstausweis_nicht_lesbar(cloud, monkeypatch):
    async def ohne(mandant):
        raise tresor._nicht_verfuegbar("dienstausweis_fehlt", mandant)
    monkeypatch.setattr(tresor, "_dienst_token", ohne)
    with pytest.raises(HTTPException) as e:
        _aufloesen("lesser", "conversation")
    assert e.value.status_code == 503 and e.value.detail["grund"] == "dienstausweis_fehlt"
    assert cloud["rufe"] == []


@pytest.mark.parametrize("args,fehler", [(("Lesser!", "bild"), "mandant_ungueltig"),
                                         (("lesser", "zauberei"), "dienst_ungueltig"),
                                         (("lesser", "bild", "kein-mensch"), "mensch_ungueltig")])
def test_eingaben_werden_geprueft(cloud, args, fehler):
    with pytest.raises(HTTPException) as e:
        _aufloesen(*args)
    assert e.value.status_code == 422 and e.value.detail["error"] == fehler and cloud["rufe"] == []


# --- Sperre fuer Kunden-Mandanten ------------------------------------------------

@pytest.mark.parametrize("pfad,dienst", [
    ("/ai/genimage", "bild"), ("/ai/genimage/flow", "bild"), ("/ai/genvideo/flow", "video"),
    ("/ai/genvideo/kling", "video"), ("/ai/transcribe", "stt"), ("/ai/tts/minimax", "tts"),
    ("/ai/realtime/token", "realtime"), ("/ai/gemini/vision", "conversation"), ("/ai/claude", "conversation"),
    ("/ai/openai", "conversation"), ("/ai/scene/generate", "bild"),
    ("/ai/openai-cost-shared-state", None), ("/ai/gen3d", None), ("/ai/genmusic", None), ("/ai/ttsx", None),
])
def test_dienst_je_pfad(pfad, dienst):
    assert mandant_wege.dienst_fuer(pfad) == dienst


@pytest.fixture
def kern(monkeypatch, tmp_path):
    def verboten(*a, **kw):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    for n in ("API_ACCESS_KEY", "BETREIBER_MANDANT"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("MANDANT_WEGE_DIR", str(tmp_path / "mw"))
    monkeypatch.setenv("KOSTENBUCH_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path / "auf"))
    tokens = {"Bearer edera": {"sub": "SaulGoodman", "type": "agent", "tenant": "edera"},
              "Bearer ark": {"sub": "Cloud", "type": "agent", "tenant": "arkturian"},
              "Bearer lesser-dienst": {"sub": "service:conversation-api-lesser", "type": "service",
                                       "tenant": "lesser"}}
    monkeypatch.setattr(jwt_pruefung, "claims_oder_none", lambda a: tokens.get(a))
    gerufen = {"register": []}

    async def aufloesen(mandant, dienst, mensch=None):
        gerufen["register"].append((mandant, dienst, mensch))
        if mandant == "lesser":
            return dienstweg.Weg(weg="abo", anbieter="gemini", modell=None, zugang=None, tresor_verweis=None,
                                 quelle="mandant")
        raise dienstweg.nicht_nutzbar("kein_dienstweg", mandant, dienst)
    monkeypatch.setattr(dienstweg, "aufloesen", aufloesen)
    return gerufen


# Ein Pfad unter /ai/genimage ohne Handler: Kommt der Aufruf durch die Sperre, antwortet
# das Routing mit 404 -- es laeuft also nie ein echter Anbieter, und "durch" ist sichtbar.
PROBE = "/ai/genimage/probe/i5"


def _flow(client, token, **kopf):
    return client.post(PROBE, json={"prompt": "x"}, headers={"Authorization": token, **kopf})


def _vermerke(tmp_path):
    return [json.loads(z) for p in (tmp_path / "mw").glob("*.jsonl") for z in p.read_text().splitlines()]


def test_an_kunde_ohne_dienstweg_wird_abgewiesen(kern, monkeypatch, tmp_path):
    monkeypatch.setenv("MANDANT_WEGE", "an")
    r = _flow(TestClient(main.app), "Bearer edera")
    d = r.json()["detail"]
    assert r.status_code == 403 and d["error"] == "dienstweg_nicht_nutzbar" and d["grund"] == "kein_dienstweg"
    assert kern["register"] == [("edera", "bild", None)]
    v = _vermerke(tmp_path)
    assert v[-1]["mandant"] == "edera" and v[-1]["modus"] == "an" and v[-1]["status"] == 403


def test_an_kunde_mit_abo_weg_laeuft_nicht_auf_dem_kern(kern, monkeypatch):
    monkeypatch.setenv("MANDANT_WEGE", "an")
    r = _flow(TestClient(main.app), "Bearer lesser-dienst", **{"X-Auftrag-Mensch": "manuel@lesser.at"})
    d = r.json()["detail"]
    assert r.status_code == 503 and d["error"] == "weg_nicht_auf_dem_kern" and d["weg"] == "abo"
    assert kern["register"] == [("lesser", "bild", "manuel@lesser.at")]


def test_mensch_nur_von_einem_dienst_jwt(kern, monkeypatch):
    monkeypatch.setenv("MANDANT_WEGE", "an")
    _flow(TestClient(main.app), "Bearer edera", **{"X-Auftrag-Mensch": "alex@example.org"})
    assert kern["register"] == [("edera", "bild", None)]


def test_beobachten_laesst_durch_und_vermerkt(kern, monkeypatch, tmp_path):
    monkeypatch.setenv("MANDANT_WEGE", "beobachten")
    r = _flow(TestClient(main.app), "Bearer edera")
    assert r.status_code == 404                     # durchgelassen bis zum Routing
    v = _vermerke(tmp_path)
    assert v[-1]["modus"] == "beobachten" and v[-1]["grund"] == "kein_dienstweg" and v[-1]["pfad"] == PROBE


def test_vorgabe_ist_beobachten(kern, monkeypatch):
    monkeypatch.delenv("MANDANT_WEGE", raising=False)
    assert mandant_wege.modus() == "beobachten"
    monkeypatch.setenv("MANDANT_WEGE", "scharf")          # Tippfehler -> streng
    assert mandant_wege.modus() == "an"


@pytest.mark.parametrize("token", ["Bearer ark", None])
def test_betreiber_und_ohne_jwt_unveraendert(kern, monkeypatch, token):
    monkeypatch.setenv("MANDANT_WEGE", "an")
    h = {"Authorization": token} if token else {}
    r = TestClient(main.app).post(PROBE, json={"prompt": "x"}, headers=h)
    assert r.status_code == 404 and kern["register"] == []


def test_an_pruefungsfehler_weist_ab(kern, monkeypatch):
    monkeypatch.setenv("MANDANT_WEGE", "an")

    async def kaputt(*a, **kw):
        raise RuntimeError("x")
    monkeypatch.setattr(mandant_wege, "urteil", kaputt)
    r = _flow(TestClient(main.app), "Bearer edera")
    assert r.status_code == 503 and r.json()["detail"]["grund"] == "pruefung_fehler"


def test_aus_prueft_nicht(kern, monkeypatch):
    monkeypatch.setenv("MANDANT_WEGE", "aus")
    assert _flow(TestClient(main.app), "Bearer edera").status_code == 404 and kern["register"] == []
