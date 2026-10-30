"""/ai/models aus der CLI-Erhebung der cloud-api (Content #5740, 30.10.2026).

Vorlage ist die echte Datei von arkserver vom 30.10. (`tests/daten/`),
nicht ein erfundenes Format: cloud-api schreibt sie, api-ai liest sie
aus einem anderen Repo."""
import asyncio
import json
import time
from pathlib import Path

import pytest

from ai.routes import text_ai_routes as t
from ai.services import cli_katalog as ck

ECHT = json.loads((Path(__file__).parent / "daten" / "cli_models_arkserver_20261030.json").read_text())


@pytest.fixture
def cli(tmp_path, monkeypatch):
    def schreiben(daten=None, alter_h=1):
        d = json.loads(json.dumps(daten or ECHT))
        d["erhoben_am"] = time.time() - alter_h * 3600
        p = tmp_path / "cli-models.json"
        p.write_text(json.dumps(d))
        monkeypatch.setenv("CLI_MODELS_PATH", str(p))
        return d
    monkeypatch.setattr(ck, "codex_config", lambda *a: ("gpt-5.6-sol", "high"))
    return schreiben


@pytest.fixture
def basis(tmp_path, monkeypatch):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"_source": "automation-curated", "providers": {
        "codex": {"available": ["gpt-5"], "default": "gpt-5", "min_cli_version": "0.144.0",
                  "efforts": {"param_name": "effort", "supported_via": "-c reasoning_effort= override"}},
        "gemini": {"available": ["gemini-alt"], "default": "gemini-3.8-flash",
                   "efforts": {"default": "medium", "supported_via": "slug_suffix_composition"}},
        "grok": {"available": ["grok-4.6"], "default": "grok-4.6"}}}))
    monkeypatch.setattr(t, "_MODELS_STATE_PATH", p)
    monkeypatch.setattr(t, "_MODELS_JSON_PFAD", p)
    return p


def _prov(r):
    return r["providers_meta"]


def _ids(r, p):
    return [m["id"] for m in r["models"] if m["provider"] == p]


def test_codex_versteckte_raus_reihenfolge_nach_rang(cli, basis):
    cli()
    r = asyncio.run(t.list_text_models())
    ids = _ids(r, "codex")
    assert "gpt-reserve" not in ids and "codex-auto-review" not in ids
    assert ids[:3] == ["gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-terra"]
    assert "gpt-5" not in ids  # Basis-Liste ersetzt, nicht ergaenzt


def test_codex_vorgabe_ist_was_config_toml_wirklich_setzt(cli, basis):
    cli()
    r = asyncio.run(t.list_text_models())
    meta = _prov(r)["codex"]
    assert meta["default"] == "gpt-5.6-sol" and meta["efforts"]["default"] == "high"
    assert meta["efforts"]["per_model"]["gpt-6-luna"] == ["low", "medium", "high", "xhigh", "max"]
    assert meta["cli_version"] == "0.166.0" and meta["discovery_method"] == "cli"


def test_codex_vorgabe_ohne_config_ist_rang_eins(monkeypatch):
    p = ck._codex(ECHT["clis"]["codex"], {}, (None, None))
    assert p["default"] == "gpt-6.1-sol"
    p = ck._codex(ECHT["clis"]["codex"], {}, ("gpt-gibts-nicht", None))
    assert p["default"] == "gpt-6.1-sol"


def test_agy_familien_nur_gemini_mit_pflichtaufwand(cli, basis):
    cli()
    r = asyncio.run(t.list_text_models())
    assert _ids(r, "gemini") == ["gemini-4-argon", "gemini-3.8-flash"]
    eff = _prov(r)["gemini"]["efforts"]
    assert eff["required"] is True and eff["per_model"]["gemini-4-argon"] == ["low", "medium", "high"]
    assert _prov(r)["gemini"]["default"] == "gemini-3.8-flash"


def test_claude_aliase_grok_bleibt_api_anbieter_aus_repo(cli, basis):
    cli()
    r = asyncio.run(t.list_text_models())
    assert _ids(r, "claude") == ["fable", "opus", "sonnet", "haiku"]
    assert _ids(r, "grok") == ["grok-4.6"]
    assert "deepseek-v4-pro" in _ids(r, "deepseek") and "glm-5.2" in _ids(r, "glm")
    assert r["source"] == "cli"
    assert r["sources"]["codex"] == "cli" and r["sources"]["grok"] == "automation-curated"
    assert r["sources"]["minimax"] == "api-ai"


def test_ohne_models_json_trotzdem_katalog(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(t, "_MODELS_STATE_PATH", tmp_path / "fehlt.json")
    cli()
    r = asyncio.run(t.list_text_models())
    assert r["source"] == "cli" and "gpt-6.1-sol" in _ids(r, "codex") and _ids(r, "grok") == []


def test_alte_erhebung_wird_ignoriert(cli, basis):
    cli(alter_h=50)
    r = asyncio.run(t.list_text_models())
    assert r["source"] == "automation-curated" and _ids(r, "codex") == ["gpt-5"]


def test_unbekanntes_format_wird_ignoriert(cli, basis):
    cli({"erhoben_am": 0, "models": []})
    r = asyncio.run(t.list_text_models())
    assert r["source"] == "automation-curated"


def test_agy_fehler_letzter_guter_stand_sichtbar(cli, basis):
    d = json.loads(json.dumps(ECHT))
    d["clis"]["agy"]["fehler"] = "timeout"
    d["clis"]["agy"]["stand_von"] = 1793300000
    cli(d)
    r = asyncio.run(t.list_text_models())
    assert "gemini-4-argon" in _ids(r, "gemini")
    assert r["sources"]["gemini"] == "cli"


def test_agy_effort_vorgabe_liest_die_erhebung(cli, basis):
    d = json.loads(json.dumps(ECHT))
    d["clis"]["agy"]["modelle"] = [{"id": "gemini-9-x-low", "name": "x"}, {"id": "gemini-9-x-high", "name": "x"}]
    cli(d)
    # medium gibt es fuer gemini-9-x nicht -> nichts nachreichen statt raten
    assert t._agy_effort_vorgabe("gemini-9-x") is None
    assert t._agy_effort_vorgabe("gemini-3.8-flash") == "medium"


def test_codex_config_liest_toml(tmp_path):
    (tmp_path / "config.toml").write_text('model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\n[x]\nmodel = "nein"\n')
    assert ck.codex_config(str(tmp_path)) == ("gpt-5.6-sol", "high")
    assert ck.codex_config(str(tmp_path / "fehlt")) == (None, None)
