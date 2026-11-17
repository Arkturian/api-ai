"""#1960: Kein fest eingebauter Storage-Schluessel mehr.

Bis 17.11.2026 fiel storage_api_key() ohne STORAGE_API_KEY/API_KEY auf ein
Literal zurueck. Storage sperrt dieses Literal mit Schritt 3 der Rotation.
"""
import importlib
import logging

from ai.clients import storage_client as sc

LITERAL_SHA = "43ba25cae1de"


def _sha(v):
    import hashlib
    return hashlib.sha256(v.encode()).hexdigest()[:12]


def test_ohne_env_kein_literal(monkeypatch, caplog):
    monkeypatch.delenv("STORAGE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(sc, "_warned_missing", False, raising=False)
    with caplog.at_level(logging.ERROR):
        key = sc.storage_api_key()
    assert key == ""
    assert "1960" in caplog.text


def test_env_wird_genommen(monkeypatch):
    monkeypatch.setenv("STORAGE_API_KEY", "  eigener-schluessel \n")
    monkeypatch.setenv("API_KEY", "anderer")
    assert sc.storage_api_key() == "eigener-schluessel"


def test_literal_steht_nicht_mehr_im_modul():
    import inspect
    quelle = inspect.getsource(sc)
    for wort in quelle.replace('"', " ").replace("'", " ").split():
        assert _sha(wort) != LITERAL_SHA
