"""`product_details` — der Agent spricht über das offene Produkt (#1404).

Der Unterschied zu den Suchwerkzeugen: Hier kommt Produkt-TEXT in den
Modellkontext, und das ist der Zweck. Die Auswahl trifft trotzdem der
Server — das Modell nennt höchstens eine Ordnungszahl.

Der wunde Punkt sind die zwei Fachzustände. `no_focus` und
`no_such_position` kommen laut Vertrag mit HTTP 200. Behandelte ich
sie wie einen Ausfall, sagte der Sprecher „Katalog nicht erreichbar"
und der Kunde suchte den Fehler in der Technik statt auf seinem
Bildschirm.
"""

import asyncio
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


class _Antwort:
    def __init__(self, code, daten=None, text=""):
        self.status_code = code
        self._daten = daten
        self.text = text

    def json(self):
        if self._daten is None:
            raise ValueError("kein JSON")
        return self._daten


class _Client:
    gesehen = []
    antwort = None
    fehler = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, **k):
        _Client.gesehen.append({"url": url, "json": json, "headers": headers})
        if _Client.fehler:
            raise _Client.fehler
        return _Client.antwort


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "x" * 32)
    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    _Client.gesehen = []
    _Client.fehler = None
    _Client.antwort = _Antwort(200, {"status": "details", "name": "Sonus"})


def _ruf(args=None, sid="vs_pf_1"):
    return asyncio.run(rr._tool_product_details(args or {}, sid))


# ─────────────────────────── Fachzustände sind KEINE Ausfälle

@pytest.mark.parametrize("zustand", ["no_focus", "no_such_position"])
def test_fachzustaende_kommen_unveraendert_durch(zustand):
    """Der Kern. Würde ich sie zu `unavailable` machen, verdächtigte
    der Kunde die Technik statt auf seinen Bildschirm zu schauen."""
    _Client.antwort = _Antwort(200, {"status": zustand})
    assert _ruf() == {"status": zustand}


def test_detailfelder_kommen_unveraendert_durch():
    daten = {"status": "details", "name": "Sonus", "material": "Polycarbonat",
             "features": ["belüftet"], "price_eur": [89.0, 129.0]}
    _Client.antwort = _Antwort(200, daten)
    assert _ruf() == daten


# ─────────────────────────────── echte Ausfälle bleiben Ausfälle

@pytest.mark.parametrize("code", [401, 403, 422, 500, 503])
def test_jeder_nicht_200_ist_ein_ausfall(code):
    _Client.antwort = _Antwort(code, None, "kaputt")
    assert _ruf() == {"status": "unavailable"}


def test_netzfehler_ist_ein_ausfall():
    _Client.fehler = RuntimeError("Netz weg")
    assert _ruf() == {"status": "unavailable"}


def test_unlesbare_antwort_ist_ein_ausfall():
    _Client.antwort = _Antwort(200, None, "kein json")
    assert _ruf() == {"status": "unavailable"}


def test_ohne_konfiguration_kein_aufruf(monkeypatch):
    monkeypatch.delenv(rr.ONEAL_INTERNAL_KEY_ENV, raising=False)
    assert _ruf() == {"status": "unavailable"}
    assert _Client.gesehen == []


def test_ohne_sitzungskennung_kein_aufruf():
    """oneal kann ohne Sitzung weder Fokus noch Auswahl auflösen. Einen
    Fachzustand zu erfinden wäre schlimmer als der ehrliche Ausfall."""
    assert asyncio.run(rr._tool_product_details({}, None)) == {"status": "unavailable"}
    assert _Client.gesehen == []


# ─────────────────────────────────────────── Positionen, 1-basiert

def test_position_wird_weitergereicht():
    _ruf({"position": 3})
    assert _Client.gesehen[-1]["json"] == {"session_id": "vs_pf_1", "position": 3}


def test_ohne_position_kein_feld():
    """Kein Feld heißt „das fokussierte Produkt". Eine 0 hineinzu-
    schreiben hieße etwas anderes."""
    _ruf()
    assert "position" not in _Client.gesehen[-1]["json"]


@pytest.mark.parametrize("schlecht", [0, -1, -99])
def test_unbrauchbare_position_wird_verworfen_statt_zum_ausfall(schlecht):
    """oneal weist 0 und negativ mit 422 ab — richtig, aber ein 422
    landet bei mir im Ausfall-Zweig. Der Kunde hörte „Katalog nicht
    erreichbar", weil das Modell „das nullte" gesagt hat. Also
    verwerfe ich hier und frage nach dem fokussierten Produkt."""
    _ruf({"position": schlecht})
    gesendet = _Client.gesehen[-1]["json"]
    assert "position" not in gesendet
    assert gesendet == {"session_id": "vs_pf_1"}


@pytest.mark.parametrize("schlecht", ["drei", None, [], {}])
def test_unsinnige_position_reisst_nichts_mit(schlecht):
    _ruf({"position": schlecht})
    assert "position" not in _Client.gesehen[-1]["json"]


def test_position_als_ziffernstring_wird_uebernommen():
    """Das Modell schickt gelegentlich '3' statt 3."""
    _ruf({"position": "3"})
    assert _Client.gesehen[-1]["json"]["position"] == 3


# ───────────────────────────────────── Vollmacht und Adressierung

def test_interner_schluessel_geht_mit():
    _ruf()
    assert _Client.gesehen[-1]["headers"]["X-Realtime-Internal-Key"]


def test_richtiger_endpunkt():
    _ruf()
    assert _Client.gesehen[-1]["url"].endswith("/v1/realtime/selections/details")


def test_sitzung_kommt_vom_server_nicht_vom_modell():
    """Auch wenn das Modell eine session_id mitschickt, gewinnt die
    Kennung aus dem Kopf — sonst könnte ein Gespräch die Sitzung eines
    anderen auslesen."""
    _ruf({"session_id": "fremde-sitzung", "position": 1})
    assert _Client.gesehen[-1]["json"]["session_id"] == "vs_pf_1"


# ────────────────────────────── Werkzeug, Dispatch und Persona

def test_werkzeug_ist_im_schema_und_1_basiert():
    t = [x for x in rr._product_finder_tools("O'Neal")
         if x["name"] == "product_details"][0]
    p = t["parameters"]
    assert p["properties"]["position"]["minimum"] == 1
    assert p["additionalProperties"] is False
    assert not p.get("required"), "position ist optional"
    assert "ab EINS" in t["description"]


def test_werkzeug_verlangt_einen_grant():
    """Es liefert Produkttext — es gehört zu den geschützten, nicht zu
    den offenen Altwerkzeugen."""
    assert "product_details" in rr.PRODUCT_TOOL_NAMES
    assert "product_details" in rr.READ_TOOL_NAMES


def test_dispatch_kennt_das_werkzeug():
    import inspect
    quelle = inspect.getsource(rr.realtime_tool_call)
    assert 'tool_name == "product_details"' in quelle
    assert "_tool_product_details(args, x_session_id, diagnose)" in quelle


def test_persona_nennt_beide_fachzustaende_und_verbietet_die_stoerung():
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "Oeffnen Sie eins" in t
    assert "So weit reicht" in t
    assert "Sag NICHT, der Katalog" in t
