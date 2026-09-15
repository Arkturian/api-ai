"""Ein angefordertes Modell, das das Abo nicht fuehrt, ist ein Fehler —
kein stiller Lauf auf dem Abo-Standard. Anlass 15.09.2026 (Automation):
?model=gpt-5.4-mini -> 200 model=codex-default; Ursache 4ce0410 (10.09.),
Nachsetzen ohne --model bei Ablehnung. Rueckfall nur mit Opt-in oder bei
Servervorgabe, und dann in der Antwort benannt."""
import asyncio
import json
import types

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t

KATALOG = {"models": [
    {"slug": "gpt-6-astra", "visibility": "list", "default_reasoning_level": "medium",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}, {"effort": "high"}]},
    {"slug": "gpt-5.6-luna", "visibility": "list", "default_reasoning_level": "medium",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}]},
    {"slug": "gpt-reserve", "visibility": "hide", "default_reasoning_level": "medium",
     "supported_reasoning_levels": [{"effort": "medium"}]},
]}

ABLEHNUNG = ('{"type":"error","message":"Model metadata for `gpt-5.4-mini` not found. Defaulting"}\n'
             '{"type":"turn.failed","error":{"message":"Model metadata for `gpt-5.4-mini` not found"}}\n')
ERFOLG = ('{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n'
          '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}\n')


# --- Katalog + Pruefung --------------------------------------------------

def test_katalog_traegt_sichtbarkeit():
    k = t._codex_katalog_parsen(KATALOG)
    assert k["gpt-6-astra"]["visibility"] == "list"
    assert k["gpt-reserve"]["visibility"] == "hide"


def test_pruefung_kennt_nur_gelistete_modelle():
    k = t._codex_katalog_parsen(KATALOG)
    assert t._codex_modell_pruefen(k, "gpt-6-astra") == (True, ["gpt-5.6-luna", "gpt-6-astra"])
    assert t._codex_modell_pruefen(k, "gpt-reserve")[0] is False       # versteckt
    assert t._codex_modell_pruefen(k, "gpt-5.4-mini")[0] is False      # gestrichen
    assert t._codex_modell_pruefen(k, "gpt-9-gibtsnicht")[0] is False  # nie existiert


def test_ohne_katalog_oder_ohne_modell_wird_nicht_geblockt():
    assert t._codex_modell_pruefen(None, "gpt-9-gibtsnicht") == (True, None)
    assert t._codex_modell_pruefen(t._codex_katalog_parsen(KATALOG), None)[0] is True


# --- Endpunkt --------------------------------------------------------------

def _verdrahten(monkeypatch, katalog=KATALOG, ablehnen_bei_model=True):
    laeufe = []

    def fake_run(cmd, env=None, timeout=None, input=None):
        laeufe.append(list(cmd))
        stdout = ABLEHNUNG if (ablehnen_bei_model and "--model" in cmd) else ERFOLG
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    class Slot:
        def release(self): pass

    async def slot(_name): return Slot()

    monkeypatch.setattr(t, "_run_cli_with_pgid", fake_run)
    monkeypatch.setattr(t, "_acquire_cli_slot", slot)
    monkeypatch.setattr(t, "_codex_katalog", lambda: t._codex_katalog_parsen(katalog))
    monkeypatch.setattr(t, "_codex_config_effort", lambda *a, **k: None)
    monkeypatch.setattr(t, "_download_storage_images", lambda _p: [])
    from ai.services import codex_cost_tracker as _cct
    monkeypatch.setattr(_cct.codex_cost_tracker, "track_usage", lambda *a, **k: None)
    monkeypatch.delenv("CODEX_VISION_MODEL", raising=False)
    return laeufe


def _aufruf(prompt, model):
    return asyncio.run(t._chatgpt_einmal(prompt, model, "x"))


def test_gestrichenes_modell_ist_422_vor_dem_aufruf(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    with pytest.raises(HTTPException) as e:
        _aufruf(t.Prompt(prompt="hi"), "gpt-5.4-mini")
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "unsupported_model"
    assert e.value.detail["model"] == "gpt-5.4-mini"
    assert e.value.detail["available"] == ["gpt-5.6-luna", "gpt-6-astra"]
    assert laeufe == []                                    # kein codex-Lauf


def test_verstecktes_modell_ist_422(monkeypatch):
    _verdrahten(monkeypatch)
    with pytest.raises(HTTPException) as e:
        _aufruf(t.Prompt(prompt="hi"), "gpt-reserve")
    assert e.value.status_code == 422


def test_laufzeitablehnung_ohne_optin_ist_422_und_setzt_nicht_nach(monkeypatch):
    """Katalog veraltet (Modell noch gelistet), codex lehnt ab: 422, kein
    zweiter Lauf ohne --model."""
    laeufe = _verdrahten(monkeypatch)
    with pytest.raises(HTTPException) as e:
        _aufruf(t.Prompt(prompt="hi"), "gpt-6-astra")
    assert e.value.status_code == 422 and e.value.detail["error"] == "unsupported_model"
    assert e.value.detail["rejected_by"] == "codex"
    assert len(laeufe) == 1 and "--model" in laeufe[0]


def test_optin_setzt_nach_und_benennt_es(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    r = _aufruf(t.Prompt(prompt="hi", model_fallback=True), "gpt-6-astra")
    assert r.response == "OK"
    assert r.model == "codex-default"
    assert r.model_requested == "gpt-6-astra"
    assert r.model_fallback is True
    assert len(laeufe) == 2 and "--model" not in laeufe[1]


def test_optin_gilt_auch_fuer_gestrichenes_modell_ohne_katalogtreffer(monkeypatch):
    laeufe = _verdrahten(monkeypatch)
    r = _aufruf(t.Prompt(prompt="hi", model_fallback=True), "gpt-5.4-mini")
    assert r.model == "codex-default" and r.model_fallback is True
    assert len(laeufe) == 2                                # Versuch + Nachsetzen


def test_ohne_modell_bleibt_alles_wie_bisher(monkeypatch):
    laeufe = _verdrahten(monkeypatch, ablehnen_bei_model=False)
    r = _aufruf(t.Prompt(prompt="hi"), None)
    assert r.model == "codex-default" and r.model_fallback is False and r.model_requested is None
    assert len(laeufe) == 1 and "--model" not in laeufe[0]


def test_servervorgabe_fuer_bilder_darf_zurueckfallen_und_sagt_es(monkeypatch, tmp_path):
    """Der Vision-Default ist unsere Wahl, nicht die des Aufrufers: wird er
    abgelehnt, laeuft der Abo-Standard — benannt, nicht still."""
    laeufe = _verdrahten(monkeypatch)
    bild = tmp_path / "b.png"
    bild.write_bytes(b"\x89PNG")
    r = _aufruf(t.Prompt(prompt="was ist das", image_paths=[str(bild)]), None)
    assert r.model == "codex-default" and r.model_fallback is True
    assert r.model_requested == "gpt-5.6-luna"
    assert len(laeufe) == 2 and "--model" in laeufe[0] and "--model" not in laeufe[1]
