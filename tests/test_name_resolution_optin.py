"""Namensauflösung für gesprochene Agentennamen (#1037).

Alexanders Vorgabe woertlich: *„aber nicht ueber eine KI, sondern ueber
eine semantische Aehnlichkeitssuche vom Namen her."* Der Grund ist
Reproduzierbarkeit — eine Aufloesung, die bei gleicher Eingabe zweimal
Verschiedenes liefert, ist schlimmer als eine, die zugibt, dass sie es
nicht weiss.

Der Anlass war messbar: Sitzung `vs_CloudV2_66599` (2026-08-11 17:36),
Vertrag angenommen, Werkzeug geliefert, nie aufgerufen, kein Auftrag —
verworfen mit `authority_missing_for_target`, obwohl die wahrscheinliche
Ursache der gesprochene Name war.
"""

import pytest

from ai.routes import realtime_routes as rr


def test_werkzeug_haengt_an_eigenem_opt_in():
    """Getrennt von `read_tools` — wegen der Reihenfolge, nicht aus Stil.

    Cloud baut den Endpunkt, CloudV2 die Annahme, dann erst ich. Haengte
    das Werkzeug am bestehenden Opt-in, erschiene es in jeder Sitzung,
    sobald ich ausliefere, und riefe ins Leere. Am 2026-08-11 hat die
    umgekehrte Reihenfolge Arcturian eine Stunde stillgelegt.
    """
    assert [t["name"] for t in rr._arcturian_read_tools()] == ["agent_status"]
    mit = [t["name"] for t in rr._arcturian_read_tools(True)]
    assert mit == ["agent_status", "resolve_agent_name"]


def test_werkzeug_ist_routbar():
    assert "resolve_agent_name" in rr.READ_TOOL_NAMES


def test_beschreibung_verbietet_das_selber_waehlen():
    """`ambiguous` heisst zurueckfragen, nicht entscheiden.

    Ein stiller Fehlgriff schickt eine Nachricht an den falschen
    Agenten; eine Rueckfrage kostet zwei Sekunden.
    """
    t = [x for x in rr._arcturian_read_tools(True) if x["name"] == "resolve_agent_name"][0]
    d = t["description"]
    assert "ambiguous" in d and "FRAG" in d and "waehle NICHT" in d
    # Und der gesprochene Wortlaut wird unveraendert weitergereicht —
    # dieselbe Regel wie im Resolver: das Falten macht der Server.
    spoken = t["parameters"]["properties"]["spoken"]["description"]
    assert "Nicht normalisieren" in spoken


@pytest.mark.asyncio
async def test_ohne_bearer_wird_nicht_aufgeloest():
    res = await rr._tool_resolve_agent_name({"spoken": "drei D API"}, None)
    assert res["ok"] is False and res["error"] == "no_caller_identity"


@pytest.mark.asyncio
async def test_fehlender_endpunkt_wird_zugegeben_statt_geraten(monkeypatch):
    """404 ist der Normalfall, solange Cloud noch baut.

    Ein ehrliches „kann ich nicht" ist hier richtig; ein Fallback auf
    eigenes Raten waere genau der stille Fehlgriff, gegen den der
    Endpunkt gebaut wird.
    """
    class _R:
        status_code = 404
        text = ""
        def json(self): return {}

    class _C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _R()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _C)
    res = await rr._tool_resolve_agent_name({"spoken": "drei D API"}, "Bearer x")
    assert res["ok"] is False
    assert res["error"] == "name_resolution_unavailable"


def test_persona_regel_nennt_kein_werkzeug():
    """Gemessen am 2026-08-11: Ein Werkzeugname in der Persona bricht
    den Entscheidungs-Zug (Resolver 8/8 ohne, 1/8 mit, 0/8 mit Verbot).

    Die Regel darf deshalb nur die VERFUEGBARKEIT behaupten — „es gibt
    einen Weg" —, nie den Namen. CloudV2 hatte dieselbe Form
    vorgeschlagen; gegen P0-4, P1-1, P1-3 und P0-5 gemessen: 5/5 in
    beiden Armen, kein Rueckschritt.
    """
    text = rr._companion_arcturian_prompt("de") + rr._arcturian_resolver_addendum("de")
    assert "NAMEN, DIE DU HOERST" in text
    assert "Rate nicht" in text
    assert "resolve_agent_name" not in text
    assert "agent_status" not in text
