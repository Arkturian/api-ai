"""`agent_status` liefert Inhalt, nicht nur Zustandsflaggen.

Alex' Entscheidung, 2026-08-10: „Statusabfrage, vielleicht machen wir es
so, per MCP holt er sich einfach die letzte Antwort des Agenten oder die
gerade aktive Antwort."

Der Grund steht in einem gemessenen Lauf: Das Werkzeug gab vorher nur
`working`/`thinking` zurueck. Darauf laesst sich kein Satz bauen, den ein
Mensch hoeren will — und genau da hat der Agent angefangen, die Luecke
mit „ich kuemmere mich darum" zu fuellen. Inhalt statt Flagge nimmt dem
Fuellsatz den Anlass.

Was hier bewacht wird:
  * `last_reply` kommt aus dem letzten `kind == "text"`-Abschnitt.
  * Faellt eine Quelle aus, wird die Antwort AERMER, nicht leer — ein
    Sprachagent darf nicht auf eine langsame Verlaufsdatei warten.
  * Ohne Bearer wird nicht anonym gelesen.
"""

import pytest

from ai.routes import realtime_routes as rr


# ---------------------------------------------------------------- condense

def test_condense_entfernt_markdown_und_codebloecke():
    roh = "**Fertig.**\n```python\nprint(1)\n```\n- Punkt eins"
    out = rr._condense_for_speech(roh)
    assert "```" not in out and "**" not in out
    assert "print(1)" not in out
    assert out.startswith("Fertig.")


def test_condense_kappt_an_satzgrenze_nicht_im_wort():
    satz = "Der Befund ist bestaetigt. " * 60
    out = rr._condense_for_speech(satz)
    assert len(out) <= rr._SPEECH_MAX_CHARS + 2
    assert out.endswith("…")
    # Vor dem Auslassungszeichen steht ein ganzer Satz, kein Wortbruch.
    assert out[: -len(" …")].rstrip().endswith(".")


def test_condense_laesst_kurzen_text_unveraendert():
    assert rr._condense_for_speech("Alles erledigt.") == "Alles erledigt."


# ------------------------------------------------------------------ gating

@pytest.mark.asyncio
async def test_ohne_bearer_wird_nicht_anonym_gelesen():
    res = await rr._tool_agent_status({"agent": "3dApi"}, None)
    assert res["ok"] is False
    assert res["error"] == "no_caller_identity"


# ------------------------------------------------------------------- inhalt

class _Antwort:
    def __init__(self, code, payload):
        self.status_code = code
        self._payload = payload

    def json(self):
        return self._payload


def _client_factory(routen: dict):
    """httpx.AsyncClient-Ersatz, der pro Pfad-Fragment antwortet.

    `routen` bildet ein Teilstueck des Pfads auf (code, payload) ab oder
    auf eine Exception-Instanz, die geworfen werden soll.
    """

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            for fragment, antwort in routen.items():
                if fragment in url:
                    if isinstance(antwort, Exception):
                        raise antwort
                    return _Antwort(*antwort)
            return _Antwort(404, {})

    return _FakeClient


_VERLAUF = {
    "turns": [
        {
            "timestamp": "2026-08-10T17:00:00Z",
            "ended_at": "2026-08-10T17:01:00Z",
            "sections": [
                {"kind": "text", "content": "Erster Absatz."},
                {"kind": "tool", "content": '{"result":"egal"}'},
                {"kind": "text", "content": "**Der Import laeuft**, 3 von 7 fertig."},
            ],
        }
    ]
}


@pytest.mark.asyncio
async def test_liefert_letzte_echte_antwort_statt_nur_zustand(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _client_factory({
        "/api/agents/": (200, {"state": "working", "summary": "Import"}),
        "/agent-state": (200, {"state": "thinking"}),
        "/history": (200, _VERLAUF),
    }))
    res = await rr._tool_agent_status({"agent": "3dApi"}, "Bearer x")
    assert res["ok"] is True
    assert res["state"] == "thinking"
    # Der LETZTE Textabschnitt, nicht der erste, und ohne Markdown.
    assert res["last_reply"] == "Der Import laeuft, 3 von 7 fertig."
    assert res["last_reply_at"] == "2026-08-10T17:01:00Z"


@pytest.mark.asyncio
async def test_ausfall_einer_quelle_macht_die_antwort_aermer_nicht_leer(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _client_factory({
        "/api/agents/": (500, {}),
        "/agent-state": (200, {"state": "ready"}),
        "/history": TimeoutError("Verlauf zu langsam"),
    }))
    res = await rr._tool_agent_status({"agent": "3dApi"}, "Bearer x")
    assert res["ok"] is True
    assert res["state"] == "ready"
    assert res["last_reply"] is None


@pytest.mark.asyncio
async def test_gar_nichts_lesbar_sagt_das_offen_statt_zu_erfinden(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _client_factory({
        "/api/agents/": (500, {}),
        "/agent-state": (500, {}),
        "/history": (500, {}),
    }))
    res = await rr._tool_agent_status({"agent": "Unbekannt"}, "Bearer x")
    assert res["ok"] is False
    assert res["error"] == "nothing_readable"
    assert "erfinde nichts" in res["hint"]


@pytest.mark.asyncio
async def test_ohne_agentnamen_bleibt_es_die_uebersicht(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _client_factory({
        "/api/agents/status": (200, {"agents": []}),
    }))
    res = await rr._tool_agent_status({}, "Bearer x")
    assert res["ok"] is True and res["agent"] == "(alle)"
    # `data`, NICHT `board`: Der Web-Client kompaktiert die Uebersicht
    # ueber `result.data.agents`. Beim Umbau auf den benannten Fall hatte
    # ich diesen Zweig mitbenannt — CloudV2-Codex fand es in der
    # Gegenpruefung, bevor es lief. Ohne den Schluessel liefe die volle
    # Agentenliste unkompaktiert in den Modellkontext.
    assert "data" in res, "Uebersicht muss `data` heissen (Web-Kompaktierung)"
    assert res["data"] == {"agents": []}
    # `scope` ist das maschinenlesbare Merkmal. `agent: "(alle)"` ist
    # Anzeigetext — wer darauf prueft, bricht bei der ersten Uebersetzung.
    assert res["scope"] == "all"


@pytest.mark.asyncio
async def test_benannter_fall_traegt_scope_one(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _client_factory({
        "/agent-state": (200, {"state": "ready"}),
        "/api/agents/": (500, {}),
        "/history": (500, {}),
    }))
    res = await rr._tool_agent_status({"agent": "3dApi"}, "Bearer x")
    assert res["scope"] == "one"


# ------------------------------------------------------------------- persona

def test_persona_verbietet_die_gemessenen_fuellsaetze():
    """REGEL 5 muss die Formulierungen nennen, die im Prueflauf fielen.

    Ein abstraktes „sei ehrlich" hat sie nicht verhindert; benannt
    wurden sie erst, nachdem der Prueflauf „ich kuemmere mich darum"
    dreimal und „die Antwort steht noch aus" fuenfmal gemessen hatte.
    """
    text = rr._companion_arcturian_prompt("de")
    assert "REGEL 5" in text
    for satz in ("kuemmere mich darum", "steht noch aus", "melde mich"):
        assert satz in text, f"REGEL 5 nennt '{satz}' nicht beim Namen"
