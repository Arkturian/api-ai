"""Der Effort-Katalog (`codex debug models`) darf einen Aufruf nicht
aufhalten. Anlass CloudV2 12.11.2026 (Vorlesen 62/143/83 s): der Katalog
lief bei jeder Anfrage zweimal je 15 s in den Timeout, weil ein Fehlschlag
nicht gemerkt wurde, und das synchron in der Ereignisschleife — alle
anderen Anfragen des Workers standen solange mit.

Dazu die Phasen-Zeiten je codex-Lauf, damit ein langsamer Lauf zeigt, wo
die Zeit liegt."""
import asyncio
import json
import subprocess
import sys
import time
import types

import pytest

from ai.routes import text_ai_routes as t

KATALOG_JSON = {"models": [{"slug": "gpt-5.6-luna", "visibility": "list",
                            "default_reasoning_level": "low",
                            "supported_reasoning_levels": [{"effort": "low"}]}]}


@pytest.fixture
def katalog_lauf(monkeypatch):
    laeufe = []
    verhalten = {"modus": "ok"}

    def fake_run(cmd, **kw):
        laeufe.append(kw.get("env", {}).get("CODEX_HOME"))
        if verhalten["modus"] == "timeout":
            raise subprocess.TimeoutExpired(cmd, 15)
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(KATALOG_JSON))

    monkeypatch.setattr(t.subprocess, "run", fake_run)
    monkeypatch.setattr(t, "_codex_text_home", lambda haupt: haupt + "-text")
    monkeypatch.setenv("CODEX_HOME", "/x/.codex")
    t._codex_katalog_cache.update({"at": 0.0, "katalog": None, "fehl_at": None})
    t._codex_katalog_auffrischung["laeuft"] = False
    return laeufe, verhalten


def test_fehlschlag_wird_gemerkt(katalog_lauf):
    laeufe, verhalten = katalog_lauf
    verhalten["modus"] = "timeout"
    assert t._codex_katalog() is None
    assert t._codex_katalog() is None
    assert t._codex_katalog() is None
    assert len(laeufe) == 1                      # nicht je Aufruf neu 15 s


def test_nach_ablauf_der_fehlerfrist_neuer_versuch(katalog_lauf):
    laeufe, verhalten = katalog_lauf
    verhalten["modus"] = "timeout"
    t._codex_katalog()
    t._codex_katalog_cache["fehl_at"] -= t._CODEX_KATALOG_FEHL_TTL_S + 1
    verhalten["modus"] = "ok"
    assert t._codex_katalog()["gpt-5.6-luna"]["levels"] == ["low"]
    assert len(laeufe) == 2


def test_abgelaufener_katalog_kommt_sofort_und_frischt_im_hintergrund_auf(katalog_lauf, monkeypatch):
    laeufe, _ = katalog_lauf
    alt = {"alt": {"levels": [], "default": None, "visibility": "list"}}
    t._codex_katalog_cache.update({"at": time.monotonic() - t._CODEX_KATALOG_TTL_S - 1, "katalog": alt})
    gestartet = []
    echt = t._codex_katalog_auffrischen

    def langsam():
        gestartet.append(1)
        return echt()

    monkeypatch.setattr(t, "_codex_katalog_auffrischen", langsam)
    assert t._codex_katalog() is alt             # wartet nicht
    for th in [th for th in __import__("threading").enumerate() if th.name == "codex-katalog"]:
        th.join(timeout=5)
    assert gestartet == [1]
    assert "gpt-5.6-luna" in t._codex_katalog_cache["katalog"]


def test_katalog_liest_im_schlanken_home(katalog_lauf):
    laeufe, _ = katalog_lauf
    t._codex_katalog()
    assert laeufe == ["/x/.codex-text"]


def test_langsamer_katalog_haelt_die_ereignisschleife_nicht_an(monkeypatch):
    """Waehrend der Katalog 0,6 s braucht, muss eine zweite Koroutine
    weiterlaufen. Vorher: synchroner Aufruf, Ticker stand still."""
    fenster = []

    def langsamer_katalog():
        a = time.monotonic()
        time.sleep(0.6)
        fenster.append((a, time.monotonic()))
        return None

    class Slot:
        def release(self): pass

    async def slot(_n): return Slot()

    erfolg = ('{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n'
              '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}\n')
    monkeypatch.setattr(t, "_codex_katalog", langsamer_katalog)
    monkeypatch.setattr(t, "_run_cli_with_pgid",
                        lambda cmd, env=None, timeout=None, input=None, phasen=None:
                        types.SimpleNamespace(returncode=0, stdout=erfolg, stderr=""))
    monkeypatch.setattr(t, "_acquire_cli_slot", slot)
    monkeypatch.setattr(t, "_codex_config_effort", lambda *a, **k: None)
    monkeypatch.setattr(t, "_download_storage_images", lambda _p: [])
    from ai.services import codex_cost_tracker as _cct
    monkeypatch.setattr(_cct.codex_cost_tracker, "track_usage", lambda *a, **k: None)

    async def ablauf():
        ticks = []

        async def ticker():
            for _ in range(40):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        tk = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        r = await t._chatgpt_einmal(t.Prompt(prompt="hi", tools=False), "gpt-5.6-luna", "x")
        await tk
        return r, ticks

    r, ticks = asyncio.run(ablauf())
    assert r.response == "OK"
    (a, e), = fenster
    waehrend = [x for x in ticks if a < x < e]
    assert len(waehrend) >= 3        # Ticker lief weiter, waehrend der Katalog las


def test_jsonl_phasen():
    zeilen = [(0.4, '{"type":"thread.started"}\n'), (0.5, '{"type":"turn.started"}\n'),
              (3.9, '{"type":"item.completed","item":{"type":"error"}}\n'),
              (4.2, '{"type":"item.completed","item":{"type":"agent_message","text":"x"}}\n'),
              (4.3, '{"type":"turn.completed"}\n'), (4.3, 'kein json\n')]
    assert t._jsonl_phasen(zeilen) == {"start": 0.4, "turn": 0.5, "antwort": 4.2, "fertig": 4.3}


def _skript(code):
    return [sys.executable, "-c", code]


def test_lauf_mit_phasen_liefert_ausgabe_und_zeiten():
    code = ("import sys,time,json;d=sys.stdin.read();"
            "print(json.dumps({'type':'thread.started'}),flush=True);time.sleep(0.3);"
            "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':d.upper()}}),flush=True);"
            "print(json.dumps({'type':'turn.completed'}),flush=True);sys.stderr.write('warn')")
    phasen = {}
    r = t._run_cli_with_pgid(_skript(code), env=None, timeout=10, input="hallo", phasen=phasen)
    assert r.returncode == 0 and "HALLO" in r.stdout and r.stderr == "warn"
    # relativ, nicht absolut: der Interpreterstart allein dauert unter Last Sekunden
    assert phasen["antwort"] - phasen["start"] >= 0.25
    assert phasen["start"] <= phasen["antwort"] <= phasen["fertig"] <= phasen["gesamt"]


def test_lauf_mit_phasen_timeout_toetet_und_wirft():
    code = "import json,time;print(json.dumps({'type':'thread.started'}),flush=True);time.sleep(30)"
    phasen = {}
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        t._run_cli_with_pgid(_skript(code), env=None, timeout=0.5, input="", phasen=phasen)
    assert time.monotonic() - t0 < 5 and "start" in phasen
