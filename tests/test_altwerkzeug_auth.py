"""Altwerkzeuge: erst messen, dann durchsetzen.

Die uebrigen Lesewerkzeuge verlangen bis heute keinen Grant. Eine
harte Pflicht braeche den Wanderlaut-Browser in dem Moment, in dem sie
ausgeliefert wird. Also zwei Stufen an einem Schalter — und Tests, die
den Schalter WIRKLICH umlegen statt den Quelltext zu lesen.

Anlass fuer die Schaerfe: Mein Markenpruefer blieb gruen, als ich die
Bedingung durch `if False:` ersetzte, weil der Test die Quelle grepte
statt die Funktion aufzurufen.
"""

import pytest

from ai.routes import realtime_routes as rr


@pytest.fixture(autouse=True)
def zaehler_frisch(monkeypatch):
    monkeypatch.delenv(rr.LEGACY_TOOL_AUTH_ENV, raising=False)
    rr._legacy_auth_zaehler.clear()
    yield
    rr._legacy_auth_zaehler.clear()


# ─────────────────────────────────────────────── der Schalter

def test_vorgabe_ist_aus():
    """Ein neuer Host darf den bestehenden Aufrufer nicht abschalten,
    nur weil jemand deployt hat."""
    assert rr._legacy_auth_modus() == "off"


def test_enforce_laesst_sich_schalten(monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enforce")
    assert rr._legacy_auth_modus() == "enforce"


def test_grossschreibung_und_leerzeichen_stoeren_nicht(monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "  ENFORCE ")
    assert rr._legacy_auth_modus() == "enforce"


def test_tippfehler_faellt_auf_aus_zurueck(monkeypatch):
    """Hier ist fail-open richtig, obwohl es sonst falsch waere: Ein
    Tippfehler darf nicht den laufenden Browser abschalten. Die
    Stelle, an der fail-closed zaehlt — die Produktwerkzeuge — haengt
    nicht an diesem Schalter."""
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enfoce")
    assert rr._legacy_auth_modus() == "off"


def test_leerer_wert_ist_aus(monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "")
    assert rr._legacy_auth_modus() == "off"


# ─────────────────────────────────────────── die Kopf-Erkennung

@pytest.mark.parametrize("kopf,erwartet", [
    ("Bearer abc.def.ghi", True),
    ("bearer abc", True),
    ("Bearer   ", False),
    ("Bearer", False),
    ("Basic abc", False),
    ("abc.def.ghi", False),
    ("", False),
    (None, False),
])
def test_bearer_erkennung(kopf, erwartet):
    assert rr._hat_bearer(kopf) is erwartet


def test_erkennung_prueft_nicht_die_gueltigkeit():
    """Stufe 1 fragt „kommt ueberhaupt einer", nicht „ist er gueltig".
    Ein Austausch gegen auth-api kostete einen Roundtrip im 250-ms-
    Budget fuer eine Antwort, die diese Stufe nicht braucht."""
    assert rr._hat_bearer("Bearer offensichtlich-ungueltig") is True


# ─────────────────────────────────────────────── die Messung

def test_der_befund_zaehlt_beide_seiten():
    rr._log_auth_presence("pois_near", "Bearer x")
    rr._log_auth_presence("pois_near", None)
    rr._log_auth_presence("pois_near", None)
    befund = rr.legacy_auth_befund()
    assert befund["counts"]["pois_near:mit"] == 1
    assert befund["counts"]["pois_near:ohne"] == 2


def test_der_befund_nennt_den_modus(monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enforce")
    assert rr.legacy_auth_befund()["mode"] == "enforce"


def test_der_befund_haengt_am_statusendpunkt():
    """Ohne Ausleitung stuende die Messung nur im journalctl — und das
    liefert ohne sudo lautlos nichts. Eine Messung, die man nur mit
    dem richtigen Recht sieht, wird als „keine Aufrufe" fehlgelesen."""
    import inspect
    quelle = inspect.getsource(rr.realtime_config_health)
    assert "legacy_auth_befund()" in quelle


def test_messung_trennt_die_werkzeuge():
    rr._log_auth_presence("pois_near", None)
    rr._log_auth_presence("osm_nearby", None)
    counts = rr.legacy_auth_befund()["counts"]
    assert counts["pois_near:ohne"] == 1
    assert counts["osm_nearby:ohne"] == 1


# ───────────────────────────── die Werkzeugmengen bleiben getrennt

def test_produktwerkzeuge_haengen_nicht_am_schalter():
    """Sie sind ab Tag eins hart geschuetzt. Waeren sie in derselben
    Menge, machte ein `off` sie wieder auf — eine Ruecknahme durch
    eine Umgebungsvariable, die niemand als solche liest."""
    assert rr.PRODUCT_TOOL_NAMES <= rr.READ_TOOL_NAMES
    gemessen = rr.READ_TOOL_NAMES - rr.PRODUCT_TOOL_NAMES
    assert "find_products" not in gemessen
    assert "refine_search" not in gemessen
    assert "pois_near" in gemessen


# ═══════════════ der Endpunkt selbst — aufgerufen, nicht gelesen ═══

import pytest as _pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@_pytest.fixture
def client(monkeypatch):
    """Ein Aufruf durch den echten Endpunkt.

    Das Werkzeug-Double ersetzt nur den Fachteil. Geprueft wird der
    Torwaechter davor — genau der Teil, den ein Quelltext-Grep gruen
    durchwinken wuerde, auch wenn er nie laeuft.
    """
    async def _falsches_werkzeug(args):
        return {"ok": True, "quelle": "double"}

    monkeypatch.setattr(rr, "_tool_pois_near", _falsches_werkzeug)
    app = FastAPI()
    app.include_router(rr.router, prefix="/ai")
    return TestClient(app, raise_server_exceptions=True)


def _ruf(client, kopf=None):
    headers = {"Authorization": kopf} if kopf else {}
    return client.post("/ai/realtime/tool/pois_near",
                       json={"call_id": "c1", "arguments": {}},
                       headers=headers)


def test_ohne_schalter_geht_der_aufruf_ohne_bearer_durch(client):
    """Der heutige Wanderlaut-Browser. Bricht das, ist die Umstellung
    kein Fortschritt sondern ein Ausfall."""
    r = _ruf(client)
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_mit_schalter_wird_der_aufruf_ohne_bearer_abgewiesen(client, monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enforce")
    r = _ruf(client)
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "realtime_grant_required"
    assert r.json()["detail"]["tool"] == "pois_near"


def test_mit_schalter_wird_auch_ein_ungueltiger_bearer_abgewiesen(client, monkeypatch):
    """Ein Kopf allein reicht nicht. Sonst waere die Pflicht eine
    Formsache, die jeder Aufrufer mit dem Wort „Bearer" erfuellt."""
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enforce")
    async def _immer_nein(auth, scope):
        raise RuntimeError("ungueltig")
    monkeypatch.setattr(rr, "exchange_and_verify", _immer_nein)
    r = _ruf(client, "Bearer sieht-echt-aus")
    assert r.status_code == 401


def test_mit_schalter_und_gueltigem_bearer_geht_es_durch(client, monkeypatch):
    monkeypatch.setenv(rr.LEGACY_TOOL_AUTH_ENV, "enforce")
    async def _immer_ja(auth, scope):
        return object()
    monkeypatch.setattr(rr, "exchange_and_verify", _immer_ja)
    r = _ruf(client, "Bearer gueltig")
    assert r.status_code == 200


def test_die_messung_laeuft_auch_ohne_schalter(client):
    """Sonst haette ich am Umstelltag keine Zahl und muesste raten."""
    rr._legacy_auth_zaehler.clear()
    _ruf(client)
    assert rr.legacy_auth_befund()["counts"].get("pois_near:ohne") == 1


def test_im_aus_modus_wird_auth_api_nie_gefragt(client, monkeypatch):
    """Die Messung darf keinen Roundtrip ins 250-ms-Budget legen."""
    gerufen = []
    async def _zaehl(auth, scope):
        gerufen.append(scope)
        return object()
    monkeypatch.setattr(rr, "exchange_and_verify", _zaehl)
    _ruf(client, "Bearer x")
    assert gerufen == []


def test_produktwerkzeuge_verfaelschen_die_messung_nicht(monkeypatch):
    """Der Zaehler darf NUR Altaufrufer sehen.

    Fielen `find_products`/`refine_search` in die Altmenge, stuende im
    Befund „mit Bearer, 500x" — und ich schloesse daraus, die alten
    Aufrufer schickten laengst einen Grant. Es waeren die neuen. Am
    Umstelltag entschiede ich anhand einer Zahl, die eine andere Frage
    beantwortet als die, die ich stelle.

    Sicherheitsluecke ist das keine (der harte Torwaechter der
    Produktwerkzeuge laeuft vorher), aber eine Messfaelschung — und
    die faellt spaeter auf als ein Ausfall.
    """
    async def _ja(auth, scope):
        return object()

    async def _produkt(args, sid, verfeinern):
        return {"status": "ok"}

    monkeypatch.setattr(rr, "exchange_and_verify", _ja)
    monkeypatch.setattr(rr, "_tool_product_search", _produkt)
    app = FastAPI()
    app.include_router(rr.router, prefix="/ai")
    c = TestClient(app, raise_server_exceptions=True)

    rr._legacy_auth_zaehler.clear()
    r = c.post("/ai/realtime/tool/find_products",
               json={"call_id": "c1", "arguments": {}},
               headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert rr.legacy_auth_befund()["counts"] == {}, \
        "Produktwerkzeug im Altwerkzeug-Zaehler gelandet"
