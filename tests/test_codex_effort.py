"""codex-Effort: fail-closed gegen stille Degradation (Automation, 02.09.).

Gemessen mit dem Binary des Dienstes, gpt-5.6-luna, Einholaufgabe:
  low   -> reasoning_output_tokens=0,  Antwort falsch
  xhigh -> reasoning_output_tokens=53, Antwort richtig
  ultra -> reasoning_output_tokens=0,  Antwort falsch, HTTP 200  <- der Fehler
`ultra` steht fuer luna nicht in `supported_reasoning_levels`; codex nimmt
es ohne Fehler an. Ab jetzt: 422 mit `available`. Und `tokens_used` ist als
Fingerabdruck blind (14k Systemprompt) — `thinking_tokens` traegt die
Reasoning-Tokens.
"""

import json

from ai.routes import text_ai_routes as t

KATALOG_JSON = {"models": [
    {"slug": "gpt-5.6-luna", "default_reasoning_level": "medium",
     "supported_reasoning_levels": [{"effort": e} for e in ("low", "medium", "high", "xhigh", "max")]},
    {"slug": "gpt-5.6-sol", "default_reasoning_level": "low",
     "supported_reasoning_levels": [{"effort": e} for e in ("low", "medium", "high", "xhigh", "max", "ultra")]},
    {"slug": "", "supported_reasoning_levels": []},
]}


def test_katalog_parsen_liest_slug_default_und_stufen():
    k = t._codex_katalog_parsen(KATALOG_JSON)
    assert k["gpt-5.6-luna"] == {"default": "medium", "levels": ["low", "medium", "high", "xhigh", "max"]}
    assert "ultra" in k["gpt-5.6-sol"]["levels"] and "ultra" not in k["gpt-5.6-luna"]["levels"]
    assert "" not in k
    assert t._codex_katalog_parsen(None) is None and t._codex_katalog_parsen("kaputt") is None


def test_unbekannte_stufe_fuer_bekanntes_modell_wird_abgewiesen():
    k = t._codex_katalog_parsen(KATALOG_JSON)
    ok, applied, available = t._codex_effort_pruefen(k, "gpt-5.6-luna", "ultra")
    assert ok is False and applied is None
    assert available == ["low", "medium", "high", "xhigh", "max"]


def test_bekannte_stufe_geht_durch_und_wird_gemeldet():
    k = t._codex_katalog_parsen(KATALOG_JSON)
    assert t._codex_effort_pruefen(k, "gpt-5.6-luna", "xhigh") == (True, "xhigh", ["low", "medium", "high", "xhigh", "max"])
    assert t._codex_effort_pruefen(k, "gpt-5.6-sol", "ULTRA ")[0:2] == (True, "ultra")


def test_ohne_effort_meldet_die_modellvorgabe_als_applied():
    """`effort_applied` war ein Meldeloch (nur agy setzte es). Ohne Angabe
    gilt das `default_reasoning_level` des Modells — und das wird gesagt."""
    k = t._codex_katalog_parsen(KATALOG_JSON)
    assert t._codex_effort_pruefen(k, "gpt-5.6-luna", None) == (True, "medium", ["low", "medium", "high", "xhigh", "max"])
    assert t._codex_effort_pruefen(k, "gpt-5.6-sol", "") == (True, "low", ["low", "medium", "high", "xhigh", "max", "ultra"])


def test_ohne_katalog_oder_unbekanntes_modell_wird_nicht_geblockt():
    """Ein fehlender Katalog darf keinen Aufruf kosten — dann keine Behauptung."""
    assert t._codex_effort_pruefen(None, "gpt-5.6-luna", "ultra") == (True, "ultra", None)
    k = t._codex_katalog_parsen(KATALOG_JSON)
    assert t._codex_effort_pruefen(k, "gpt-9-unbekannt", "ultra") == (True, "ultra", None)
    assert t._codex_effort_pruefen(k, None, "xhigh") == (True, "xhigh", None)


def test_katalog_wird_gecacht_und_bei_fehler_nicht_geblockt(monkeypatch):
    calls = []

    class _R:
        returncode = 0
        stdout = json.dumps(KATALOG_JSON)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        assert cmd == ["codex", "debug", "models"]
        return _R()

    monkeypatch.setattr(t.subprocess, "run", fake_run)
    t._codex_katalog_cache.update({"at": 0.0, "katalog": None})
    k1 = t._codex_katalog(); k2 = t._codex_katalog()
    assert k1 and k1 is k2 and len(calls) == 1          # zweiter Aufruf aus dem Cache

    def boom(cmd, **kw):
        raise RuntimeError("kein codex")
    monkeypatch.setattr(t.subprocess, "run", boom)
    t._codex_katalog_cache.update({"at": 0.0, "katalog": None})
    assert t._codex_katalog() is None                     # kein Katalog, kein Crash


def test_antwort_traegt_thinking_tokens_feld():
    r = t.AIResponse(response="OK", model="gpt-5.6-luna", tokens_used=1,
                     effort_applied="xhigh", thinking_tokens=53)
    assert r.thinking_tokens == 53 and r.effort_applied == "xhigh"
