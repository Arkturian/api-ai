"""Sitzungs-Scope: Marke und Jahr gehören dem Server (#4831, 1d).

OnealServs Suchroute erwartet `brand` und `collection_year` von uns.
Beide werden beim Mint gebunden — aber der Werkzeug-Dispatch kennt
diesen Zustand nicht. Von drei möglichen Quellen sind zwei
ausgeschlossen: Der Browser darf ihn nicht behaupten (der öffentliche
Key ist keine Sitzungsidentität), und das Modell kann ihn nicht setzen
(beide Felder wurden bewusst aus den Werkzeugkriterien entfernt).
Bleibt der Server.
"""

import json
import time

import pytest

from ai.services import realtime_session_scope as sc


@pytest.fixture(autouse=True)
def ablage(tmp_path, monkeypatch):
    ziel = str(tmp_path / "scope.json")
    monkeypatch.setattr(sc, "SCOPE_PATH", ziel)
    assert "/var/lib" not in sc.SCOPE_PATH
    return ziel


def test_merken_und_lesen():
    sc.merken("vs1", "O'Neal", 2027, {"sport_id": "moto", "category_id": None})
    e = sc.lesen("vs1")
    assert e["brand"] == "O'Neal"
    assert e["collection_year"] == 2027
    assert e["entry_selection"]["sport_id"] == "moto"


def test_markenoffen_ist_nicht_unbekannt():
    """Der Kern dieses Moduls, und dieselbe Klasse wie „Ausfall ist kein
    leeres Regal" — nur eine Ebene höher.

    `brand=None` in einem vorhandenen Scope heisst MARKENOFFEN. Gar kein
    Scope heisst UNBEKANNT. Wer beides gleich behandelt, lässt eine
    gebundene Sitzung stillschweigend über den ganzen Katalog laufen.
    """
    sc.merken("offen", None, 2027)
    e = sc.lesen("offen")
    assert e is not None and e["brand"] is None      # markenoffen
    assert sc.lesen("gibtsnicht") is None            # unbekannt


def test_abgelaufener_scope_gilt_als_unbekannt(monkeypatch):
    monkeypatch.setattr(sc, "SCOPE_TTL_SEC", -1.0)
    sc.merken("alt", "O'Neal", 2027)
    assert sc.lesen("alt") is None


def test_ohne_sitzungskennung_wird_nichts_abgelegt(ablage):
    sc.merken("", "O'Neal", 2027)
    import os
    assert not os.path.exists(ablage) or json.load(open(ablage)) == {}


def test_schreiben_kehrt_abgelaufenes_mit_aus(monkeypatch, ablage):
    """Der Speicher raeumt sich dort auf, wo er ohnehin angefasst wird —
    statt einen eigenen Aufraeum-Job zu brauchen, den jemand vergisst."""
    monkeypatch.setattr(sc, "SCOPE_TTL_SEC", -1.0)
    sc.merken("alt", "O'Neal", 2027)
    monkeypatch.setattr(sc, "SCOPE_TTL_SEC", 3600.0)
    sc.merken("neu", "ONE Industries", 2027)
    daten = json.load(open(ablage))
    assert "neu" in daten and "alt" not in daten


def test_vergessen():
    sc.merken("weg", "O'Neal", 2027)
    assert sc.vergessen("weg") is True
    assert sc.lesen("weg") is None
    assert sc.vergessen("weg") is False


def test_kaputte_datei_wirft_nicht(ablage):
    open(ablage, "w").write("{kein json")
    assert sc.lesen("egal") is None
    sc.merken("neu", "O'Neal", 2027)     # darf nicht werfen
    assert sc.lesen("neu")["brand"] == "O'Neal"


def test_mint_legt_den_scope_ab():
    import inspect
    from ai.routes import realtime_routes as rr
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert "realtime_session_scope.merken(" in quelle
    # Und er darf den Mint nicht toeten, wenn das Ablegen scheitert:
    stelle = quelle.index("realtime_session_scope.merken(")
    fenster = quelle[stelle - 200:stelle + 500]
    assert "try:" in fenster and "except Exception" in fenster
