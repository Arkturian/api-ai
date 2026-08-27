"""`cart_details` — der Agent kennt den Warenkorb (#1426).

Read-only, und zwar durch ABWESENHEIT: Es gibt kein Werkzeug zum
Hinzufügen, und die Persona erwähnt keines. Ein Verbotssatz wäre eine
Erwähnung — und die lässt das Modell danach greifen (am Arcturian-
Agenten gemessen: 8/8 sauber ohne Erwähnung, mit Verbot gegriffen).
"""

import asyncio
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


class _A:
    def __init__(self, code, daten=None, text=""):
        self.status_code = code
        self._d = daten
        self.text = text

    def json(self):
        if self._d is None:
            raise ValueError("kein JSON")
        return self._d


class _C:
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
        _C.gesehen.append({"url": url, "json": json, "headers": headers})
        if _C.fehler:
            raise _C.fehler
        return _C.antwort


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "k" * 32)
    monkeypatch.setattr(rr.httpx, "AsyncClient", _C)
    _C.gesehen = []
    _C.fehler = None
    _C.antwort = _A(200, {"status": "cart", "count": 2, "total_eur": 219.98,
                          "items": [{"name": "3SRS", "size": "M",
                                     "quantity": 1, "price_eur": 129.99}]})


def _ruf(args=None, sid="vs_pf_1"):
    return asyncio.run(rr._tool_cart_details(args or {}, sid))


# ───────────────────── read-only durch Abwesenheit

def test_es_gibt_kein_werkzeug_zum_hinzufuegen():
    """Der Kern des Einwands gegen den ursprünglichen Auftrag."""
    namen = {t["name"] for t in rr._product_finder_tools("O'Neal")}
    for verboten in ("add_to_cart", "cart_add", "order", "checkout",
                     "remove_from_cart"):
        assert verboten not in namen


def test_die_persona_erwaehnt_das_bestellen_mit_keinem_wort():
    """Ein Verbot ist auch eine Erwähnung. Die Fähigkeit fehlt, also
    schweigt der Text darüber."""
    t = rr._product_finder_prompt("de", "O'Neal")
    for wort in ("bestell", "Bestell", "lege nichts", "in den Korb legen",
                 "hinzufueg", "Hinzufueg"):
        assert wort not in t, wort


# ───────────────────── Zustände

def test_voller_korb_kommt_unveraendert_durch():
    d = {"status": "cart", "count": 2, "total_eur": 219.98, "items": []}
    _C.antwort = _A(200, d)
    assert _ruf() == d


def test_leerer_korb_ist_ein_erfolg_kein_ausfall():
    """„Ihr Korb ist leer" ist eine Auskunft. Würde ich daraus
    `unavailable` machen, hörte der Kunde eine Störung."""
    _C.antwort = _A(200, {"status": "empty", "count": 0})
    assert _ruf()["status"] == "empty"


def test_no_such_position_kommt_durch():
    _C.antwort = _A(200, {"status": "no_such_position"})
    assert _ruf({"position": 99})["status"] == "no_such_position"


@pytest.mark.parametrize("code", [401, 403, 422, 500])
def test_nicht_200_ist_ein_ausfall(code):
    _C.antwort = _A(code, None, "kaputt")
    assert _ruf() == {"status": "unavailable"}


def test_netzfehler_ist_ein_ausfall():
    _C.fehler = RuntimeError("weg")
    assert _ruf() == {"status": "unavailable"}


def test_ohne_sitzung_kein_aufruf():
    assert asyncio.run(rr._tool_cart_details({}, None)) == {"status": "unavailable"}
    assert _C.gesehen == []


# ───────────────────── Wire

def test_richtiger_endpunkt_und_schluessel():
    _ruf()
    g = _C.gesehen[-1]
    assert g["url"].endswith("/v1/realtime/selections/cart")
    assert g["headers"]["X-Realtime-Internal-Key"]


def test_sitzung_kommt_vom_server_nicht_vom_modell():
    _ruf({"session_id": "fremd"})
    assert _C.gesehen[-1]["json"]["session_id"] == "vs_pf_1"


def test_position_1_basiert_und_unbrauchbares_verworfen():
    _ruf({"position": 2})
    assert _C.gesehen[-1]["json"]["position"] == 2
    _ruf({"position": 0})
    assert "position" not in _C.gesehen[-1]["json"]


def test_verworfene_position_steht_in_der_diagnose():
    diag = {}
    asyncio.run(rr._tool_cart_details({"position": 0}, "vs1", diag))
    assert diag["dropped_position"] == 0


# ───────────────────── Werkzeug und Persona

def test_werkzeug_ist_grantpflichtig():
    assert "cart_details" in rr.PRODUCT_TOOL_NAMES
    assert "cart_details" in rr.READ_TOOL_NAMES


def test_schema_ist_geschlossen_und_1_basiert():
    t = [x for x in rr._product_finder_tools(None)
         if x["name"] == "cart_details"][0]["parameters"]
    assert t["properties"]["position"]["minimum"] == 1
    assert t["additionalProperties"] is False
    assert not t.get("required")


def test_persona_verlangt_nachsehen_statt_erinnern():
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "WARENKORB" in t
    assert "nie aus dem Gedaechtnis" in t


def test_dispatch_kennt_das_werkzeug():
    import inspect
    q = inspect.getsource(rr.realtime_tool_call)
    assert 'tool_name == "cart_details"' in q
