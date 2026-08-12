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


def test_fokusregel_trennt_name_von_taetigkeit():
    """Gemessen 2026-08-12: 4/8 auf 8/8, ohne die Gegenrichtung zu brechen.

    Anlass war Alexanders Sitzung `vs_Tschepp_154479`: Er fragte nach
    dem NAMEN des aktiven Agenten, bekam die richtige Antwort — und
    zusaetzlich einen Nachschlag samt Bericht, den er nicht wollte.
    Ein erzwungener Zug und rund zehn Cent fuer eine Frage, die aus der
    Sitzung beantwortbar war.

    | Fall | live | mit Regel |
    |---|---|---|
    | UC-G Namensfrage (darf NICHT nachschlagen) | 4/8 | 8/8 |
    | UC-H Taetigkeitsfrage (MUSS nachschlagen)  | 8/8 | 8/8 |

    Und ohne Rueckschritt auf UC-A, UC-B, P0-3, P4-1.
    """
    text = rr._companion_arcturian_prompt("de")
    assert "NAME ODER TAETIGKEIT" in text
    # Beide Richtungen muessen benannt sein — eine Regel, die nur das
    # Unterlassen kennt, verschiebt den Fehler auf die andere Seite.
    assert "WER gerade offen ist" in text
    assert "WAS dieser Agent tut" in text
    # Und kein Werkzeugname, sonst greift das Modell im Entscheidungs-Zug.
    assert "agent_status" not in text


# ------------------------------------------------------------ Faltung (#1037)

def test_faltung_trifft_die_tabelle_aus_1037():
    """Beide Seiten muessen identisch falten, sonst findet sie nichts."""
    faelle = [
        ("drei D API", "3dapi"), ("3D API", "3dapi"),
        ("Cloud V zwei", "cloudv2"), ("App Dev V zwei", "appdevv2"),
        ("Auth API", "authapi"), ("AI API", "aiapi"),
        ("S W F M E", "swfme"), ("K I T T", "kitt"),
        ("Förderungen", "foerderungen"),
    ]
    for gehoert, erwartet in faelle:
        assert rr._fold_agent_name(gehoert) == erwartet, gehoert


def test_umlaute_werden_vor_der_normalisierung_ersetzt():
    """Die in #1037 notierte Reihenfolge funktioniert nicht.

    NFKD zerlegt „ö" in „o" + kombinierendes Zeichen; danach findet
    `.replace("ö","oe")` nichts mehr, und „Förderungen" faltet auf
    `forderungen` statt `foerderungen`. Ausgerechnet der Agent, der in
    der Tabelle als Beispiel steht, waere damit unauffindbar.
    """
    assert rr._fold_agent_name("Förderungen") == "foerderungen"
    assert rr._fold_agent_name("Über-Agent") == "ueberagent"
    assert rr._fold_agent_name("Straße") == "strasse"


@pytest.mark.asyncio
async def test_toter_agent_zaehlt_nicht_als_gefunden(monkeypatch):
    """`state: "dead"` darf die Namensaufloesung nicht verhindern.

    Ein unbekannter Agent liefert von `/agent-state` **200 mit
    `{"state": "dead"}`** — nicht 404. Die erste Fassung der Bedingung
    fragte `if not any([board, state, last_reply])`, und `"dead"` ist
    wahr; die Aufloesung lief deshalb nie an.

    **Alle Unit-Tests waren gruen**, weil die Doubles dort 500 liefern
    statt „dead". Gefunden hat es erst ein echter Aufruf gegen den
    ausgelieferten Dienst: `agent="drei D API"` kam mit
    `resolved_from: None` zurueck, der Name blieb unaufgeloest.
    """
    aufgerufen = []

    class _Antwort:
        def __init__(self, code, payload):
            self.status_code, self._p = code, payload
        def json(self):
            return self._p

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None):
            aufgerufen.append(url)
            if "/agent-state" in url:
                # Der gehoerte Name ist tot, der aufgeloeste lebt —
                # sonst prueft der Fall nur die halbe Kette.
                return _Antwort(200, {"state": "dead" if "drei" in url else "ready"})
            if "/api/sessions" in url and "history" not in url:
                return _Antwort(200, {"sessions": [{"name": "3dApi"}]})
            return _Antwort(404, {})
        async def post(self, url, headers=None, json=None):
            aufgerufen.append(url)
            return _Antwort(404, {})   # Clouds Endpunkt nicht da -> Rueckfall

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    res = await rr._tool_agent_status({"agent": "drei D API"}, "Bearer x")

    assert any("/api/agents/resolve-name" in u for u in aufgerufen), (
        "Clouds Aufloesung wurde nicht einmal versucht — `state: dead` "
        "wurde als Treffer gewertet"
    )
    assert res.get("resolved_from") == "drei D API", (
        "Die Umdeutung wird verschwiegen — der Operator hoert einen "
        "Namen, den er nicht gesagt hat"
    )
    assert res.get("agent") == "3dApi"


def test_beschreibung_bindet_die_umdeutung_an_einen_wert():
    """„sag das dazu" allein erzeugte eine Erfindung.

    Gemessen 2026-08-12, P4-2/P4-4: Auf „woran arbeitet 3dApi gerade" —
    ein exakt richtiger Name — sagte das Modell in 3 von 4 bzw. 4 von 4
    Laeufen „Ich habe das als 3dApi verstanden", obwohl `resolved_from`
    im Ergebnis gar nicht vorkam. Es hat eine Umdeutung behauptet, die
    nie stattfand.

    Die Anweisung muss also die ABWESENHEIT des Feldes genauso klar
    regeln wie seine Anwesenheit — sonst fuellt das Modell die Luecke.
    Dieselbe Klasse wie REGEL 5: Ein Hinweis ohne Gegenfall wird zur
    Gewohnheit.
    """
    t = [x for x in rr._arcturian_read_tools() if x["name"] == "agent_status"][0]
    d = t["description"]
    assert "resolved_from" in d
    assert "Fehlt das Feld" in d, "der Gegenfall fehlt"
    assert "sag NICHTS ueber" in d
