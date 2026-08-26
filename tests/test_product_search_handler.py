"""Werkzeug-Handler für die Produktsuche (#4831, Vertrag c887a89/bad029d).

Die drei Zustände und wer sie bestimmt:
`matches` und `empty` kommen von der Gegenseite — sie weiss, ob nichts
da ist. `unavailable` ist **ausschliesslich** unsere Übersetzung eines
technischen Fehlers — wir wissen nur, ob wir sie erreicht haben.
Verwechselt man beides, sagt der Agent „das führen wir nicht", während
der Katalog bloss klemmt: eine falsche Auskunft über das Sortiment.
"""

import asyncio
import json

import pytest

from ai.routes import realtime_routes as rr
from ai.services import realtime_session_scope as sc


class _Antwort:
    def __init__(self, code, daten=None, text=""):
        self.status_code, self._d, self.text = code, daten, text
    def json(self):
        if self._d is None:
            raise ValueError("kein JSON")
        return self._d


class _Client:
    """Minimaler Ersatz für httpx.AsyncClient. `gesehen` sammelt die
    Aufrufe, damit die Nutzlast geprüft werden kann statt nur der
    Rückgabewert."""
    gesehen = []
    antwort = _Antwort(200, {"status": "matches", "count": 3})
    fehler = None
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None, headers=None):
        _Client.gesehen.append({"url": url, "json": json, "headers": headers})
        if _Client.fehler:
            raise _Client.fehler
        return _Client.antwort


@pytest.fixture(autouse=True)
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SCOPE_PATH", str(tmp_path / "scope.json"))
    monkeypatch.setenv(rr.ONEAL_SELECTION_BASE_ENV, "https://oneal.example")
    monkeypatch.setenv(rr.ONEAL_INTERNAL_KEY_ENV, "x" * 32)
    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    _Client.gesehen = []
    _Client.fehler = None
    _Client.antwort = _Antwort(200, {"status": "matches", "count": 3})
    sc.merken("vs1", "O'Neal", 2027)


def _ruf(args=None, sid="vs1", verfeinern=False):
    return asyncio.run(rr._tool_product_search(args or {}, sid, verfeinern))


# ───────────────────────────────────────────── Scope kommt vom Server

def test_marke_und_jahr_kommen_aus_dem_scope_nicht_vom_modell():
    """Auch wenn das Modell sie mitschickt, gewinnt der Sitzungs-Scope.

    Beide Felder wurden aus den Werkzeugkriterien entfernt, damit das
    Modell sie nicht setzen KANN — aber ein Werkzeugaufruf ist kein
    Ort für Vertrauen. Hier wird geprüft, dass sie auch dann nicht
    durchkommen, wenn sie doch dastehen.
    """
    _ruf({"brand": "Kini Red Bull", "collection_year": 1999, "size": "L"})
    n = _Client.gesehen[-1]["json"]
    assert n["brand"] == "O'Neal"
    assert n["collection_year"] == 2027
    assert "brand" not in n["criteria"]
    assert "collection_year" not in n["criteria"]
    assert n["criteria"]["size"] == "L"


def test_ohne_scope_kein_aufruf_und_unavailable():
    """Unbekannter Scope ist NICHT markenoffen. Lieber gar nicht suchen
    als über den ganzen Katalog."""
    assert _ruf(sid="fremd") == {"status": "unavailable"}
    assert _Client.gesehen == []


# ────────────────────────────────────── finden gegen verfeinern

def test_finden_schickt_kein_basis_token():
    _ruf({"size": "L"}, verfeinern=False)
    assert _Client.gesehen[-1]["json"]["base_selection_token"] is None


def test_verfeinern_reicht_das_token_weiter():
    _ruf({"selection_token": "st_a1", "size": "L"}, verfeinern=True)
    n = _Client.gesehen[-1]["json"]
    assert n["base_selection_token"] == "st_a1"
    # und das Token ist KEIN Suchkriterium
    assert "selection_token" not in n["criteria"]


def test_interner_schluessel_geht_mit_der_anfrage():
    _ruf({"size": "L"})
    assert _Client.gesehen[-1]["headers"]["X-Realtime-Internal-Key"]


# ─────────────────────────────────────── Ergebnis unverändert weiter

def test_app_command_wird_NICHT_entfernt():
    """Der Punkt, an dem sich zwei Lanes widersprochen hatten.

    Entfernte der Handler ihn, öffnete die Trefferfläche nie — obwohl
    die Suche erfolgreich war. Nur der Browser-Core darf ihn
    herauslösen.
    """
    _Client.antwort = _Antwort(200, {
        "status": "matches", "count": 4, "selection_token": "st_z",
        "__app_command__": {"name": "show_product_results",
                            "args": {"selection_token": "st_z"}},
    })
    erg = _ruf({"size": "L"})
    assert erg["__app_command__"]["args"]["selection_token"] == "st_z"


def test_leeres_ergebnis_bleibt_leer_und_wird_nicht_zu_unavailable():
    _Client.antwort = _Antwort(200, {"status": "empty", "count": 0})
    assert _ruf({"size": "XXL"})["status"] == "empty"


# ─────────────────────────────── technische Fehler -> unavailable

@pytest.mark.parametrize("code", [400, 403, 404, 409, 410, 422, 500, 503])
def test_jeder_http_fehler_wird_unavailable_nie_empty(code):
    """Ein abgelaufenes Token oder ein Schemafehler heisst nicht
    „es gibt nichts". Der Agent darf daraus keine Sortimentsaussage
    machen."""
    _Client.antwort = _Antwort(code, None, "Fehler")
    assert _ruf({"size": "L"}) == {"status": "unavailable"}


def test_netzwerkfehler_wird_unavailable():
    _Client.fehler = OSError("Verbindung weg")
    assert _ruf({"size": "L"}) == {"status": "unavailable"}


def test_unlesbare_antwort_wird_unavailable():
    _Client.antwort = _Antwort(200, None, "kein json")
    assert _ruf({"size": "L"}) == {"status": "unavailable"}


def test_ohne_konfiguration_kein_aufruf(monkeypatch):
    monkeypatch.delenv(rr.ONEAL_SELECTION_BASE_ENV, raising=False)
    assert _ruf({"size": "L"}) == {"status": "unavailable"}
    assert _Client.gesehen == []


# ─────────────────────────────────────────────────────── Dispatch

def test_beide_werkzeuge_sind_routbar():
    """Vorher wies der Dispatch sie mit 400 ab — deklariert, aber nicht
    geroutet, also ein Werkzeug, das ins Leere greift."""
    assert "find_products" in rr.READ_TOOL_NAMES
    assert "refine_search" in rr.READ_TOOL_NAMES


def test_dispatch_unterscheidet_finden_und_verfeinern():
    import inspect
    quelle = inspect.getsource(rr.realtime_tool_call)
    assert "_tool_product_search(args, x_session_id, False)" in quelle
    assert "_tool_product_search(args, x_session_id, True)" in quelle


# ─────────────────────────── Zugang zum Werkzeug-Dispatch (Befund 26.08.)

def test_produktwerkzeuge_verlangen_einen_grant():
    """Gemessen am laufenden Dienst: `POST /ai/realtime/tool/{name}` ist
    von aussen ohne jede Anmeldung erreichbar — HTTP 200, beide Hosts.

    Für `agent_status` fällt das nicht auf, weil der Handler intern eine
    Aufruferkennung verlangt. Die Produktsuche braucht sie nicht: Sie
    zieht Marke und Jahr aus dem Sitzungs-Scope. Wer eine `session_id`
    kennt oder rät, könnte also über einen offenen Endpunkt gegen den
    Katalog suchen.

    Deshalb Grant-Pflicht für **diese beiden** Werkzeuge. Die übrigen
    Lesewerkzeuge bleiben unverändert — nicht weil das richtig wäre,
    sondern weil der Wanderlaut-Browser sie heute ohne Grant ruft und
    eine harte Pflicht ihn sofort bräche. Neue Fähigkeit ab Tag eins
    geschützt, alter Aufrufer stufenweise.
    """
    import inspect
    quelle = inspect.getsource(rr.realtime_tool_call)
    assert "PRODUCT_TOOL_NAMES" in quelle
    assert "realtime_grant_required" in quelle
    # Die Prüfung muss VOR dem Ausführen stehen, nicht danach.
    pruefung = quelle.index("PRODUCT_TOOL_NAMES")
    ausfuehrung = quelle.index("_tool_product_search")
    assert pruefung < ausfuehrung


def test_die_uebrigen_lesewerkzeuge_bleiben_ohne_grant():
    """Sonst bricht der Wanderlaut-Guide beim Ausliefern — derselbe
    Schaden wie der beklagte, nur mit umgekehrtem Vorzeichen."""
    assert rr.PRODUCT_TOOL_NAMES == {"find_products", "refine_search"}
    assert "knowledge_query" not in rr.PRODUCT_TOOL_NAMES
    assert "agent_status" not in rr.PRODUCT_TOOL_NAMES
