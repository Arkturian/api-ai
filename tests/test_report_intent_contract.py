"""Die Absichtszeile bei gescheiterten Zuegen (#1035).

Alex' Muster: Schlaegt etwas fehl, entsteht ein Issue — Ursache,
Absicht, Kontext — damit die Liste abgearbeitet und ueber mehrere
Installationen konsolidiert werden kann. Ausdruecklich nicht
arcturian-spezifisch gedacht; der Wanderlaut-Guide ist der zweite Fall.

Die Arbeitsteilung stammt von CloudV2 und ist der Grund, warum hier so
wenig steht: Der Client baut den Bericht aus Tatsachen, die er ohnehin
hat (Fehlercode, Betriebsart, Vertrag, Build) — deterministisch, ohne
Modell, ohne Token. Denn wenn die Ausgabe des Modells gerade verworfen
wurde, ist es das Letzte, dem man den Bericht darueber anvertrauen
sollte.

Uebrig bleibt genau ein Feld, das kein Client rekonstruieren kann: die
ABSICHT. Der Fehlercode sagt, was brach — nicht, was der Operator
wollte.
"""

import pytest

from ai.routes import realtime_routes as rr


# ------------------------------------------------------------- Werkzeugform

def test_werkzeug_verlangt_genau_ein_feld():
    t = rr._arcturian_report_intent_tool()
    assert t["name"] == "report_intent"
    p = t["parameters"]
    assert p["required"] == ["intent"]
    assert list(p["properties"]) == ["intent"]
    # Kein Zusatzfeld: Jedes weitere waere eine Einladung, den
    # Fehlschlag zu ERKLAEREN statt ihn zu benennen — und das Modell
    # ist hier per Definition der unzuverlaessige Zeuge.
    assert p["additionalProperties"] is False


def test_beschreibung_grenzt_gegen_fehleranalyse_ab():
    t = rr._arcturian_report_intent_tool()
    text = t["description"] + t["parameters"]["properties"]["intent"]["description"]
    assert "EINEM Satz" in t["description"]
    for verboten in ("Entschuldigung", "Fehleranalyse", "Vermutung"):
        assert verboten in text, f"Abgrenzung gegen '{verboten}' fehlt"


# ---------------------------------------------------------------- Zug-Form

def test_erzwungener_zug_mit_genau_einem_werkzeug():
    """`required`, nicht `auto` — und nicht die benannte Funktionsform.

    Gemessen: die erzwungene Form feuert 6/6, die benannte Funktionsform
    wurde 0/6 stillschweigend ignoriert. Ein `auto` waere hier falsch,
    weil der Zug einen einzigen Zweck hat.
    """
    r = rr._arcturian_report_intent_payload()["response"]
    assert [t["name"] for t in r["tools"]] == ["report_intent"]
    assert r["tool_choice"] == "required"


def test_zug_bringt_den_resolver_nicht_zurueck():
    """Die Flaeche, auf der `tool_misrouted got=report_affect` entstand.

    Ein Override ERSETZT die Sitzungsliste — deshalb ist ein zweites
    Werkzeug in diesem Zug kein Komfort, sondern die Rueckkehr eines
    Fehlers, dessen Suche 72 Laeufe gekostet hat.
    """
    namen = [t["name"] for t in rr._arcturian_report_intent_payload()["response"]["tools"]]
    assert "resolve_arcturian_turn" not in namen
    assert "report_affect" not in namen
    assert len(namen) == 1


def test_zug_spricht_nicht():
    """Der Operator hoert die Erklaerung im regulaeren Zug, nicht hier."""
    r = rr._arcturian_report_intent_payload()["response"]
    assert r["output_modalities"] == ["text"]


def test_zug_traegt_seinen_korrelationsanker():
    r = rr._arcturian_report_intent_payload()["response"]
    assert r["metadata"][rr.ARCTURIAN_RESPONSE_KIND_FIELD] == "arcturian.report_intent"


def test_typ_steht_im_vertrag_sonst_weist_der_client_ihn_ab():
    """Fehlt der Typ in der Adapter-Menge, greift die fail-closed Pruefung.

    Er gehoert NICHT in die Server-Menge: Wir liefern die Vorlage aus,
    auf die Leitung legt sie der Client — dieselbe Rolle wie
    `arcturian.primary_audio`.
    """
    assert "arcturian.report_intent" in rr.ARCTURIAN_RESPONSE_KINDS_ADAPTER
    assert "arcturian.report_intent" not in rr.ARCTURIAN_RESPONSE_KINDS_SERVER


# ------------------------------------------------------------------ Persona

def test_regel5_benennt_den_grund_statt_zu_schonen():
    """Die alte Fassung war nicht befolgbar — gemessen, nicht vermutet.

    Sie sagte 'sag lieber nichts'. Der Client fordert nach jeder
    Entscheidung ausdruecklich eine gesprochene Antwort an; Schweigen
    ist also kein verfuegbarer Zug. Im Prueflauf vom 2026-08-10 kamen
    die woertlich verbotenen Fuellsaetze trotzdem — 5/5 in UC-A, 4/5 in
    UC-B, mit nachgewiesen geladener Regel (sha256 a4c8f66c).

    Die Regel muss dem Modell also einen WAHREN Satz geben, statt ihm
    das Sprechen zu verbieten.
    """
    text = rr._companion_arcturian_prompt("de")
    assert "REGEL 5" in text
    # Die gemessenen Fuellsaetze bleiben namentlich verboten.
    for satz in ("kuemmere mich darum", "steht noch aus", "melde mich"):
        assert satz in text
    # Neu: der Grund wird ausgesprochen, nicht verschwiegen.
    assert "nicht durch" in text, "der schonende Satz wird nicht als falsch benannt"
    assert "Dazu weiss ich gerade nichts" in text, "die wahre Ersatzantwort fehlt"
    # Und die alte, unbefolgbare Anweisung ist weg.
    assert "sag NICHTS" not in text


# --------------------------------------------------------------------- Mint

def test_mint_liefert_den_zug_nur_fuer_arcturian(monkeypatch):
    """Andere Betriebsarten duerfen die Nutzlast nicht sehen.

    Sie ist an den Arcturian-Ablauf gebunden; ein Guide-Client, der sie
    faelschlich sendet, erzeugt einen Zug ohne passenden Empfaenger.
    """
    import inspect
    quelle = inspect.getsource(rr)
    assert '"report_intent_response": (' in quelle
    # Die Bindung an die Betriebsart steht unmittelbar an der Stelle —
    # ohne sie waere die Nutzlast fuer jedes Profil im Mint.
    stelle = quelle.index('"report_intent_response": (')
    fenster = quelle[stelle:stelle + 260]
    assert 'companion_mode == "arcturian"' in fenster
    assert "else None" in fenster


# ------------------------------------------------------- erzwungener Nachschlag

def test_nachschlag_zug_erzwingt_statt_anzubieten():
    """`auto` ist die gemessene 1-von-8-Fassung — hier gilt `required`.

    24 Laeufe am 2026-08-11, Fall UC-A: ohne Erwaehnung in der Persona
    schlug das Modell 1/8 nach; mit Erwaehnung brach der Resolver-Zug
    (1/8 bzw. 0/8), weil es `agent_status` dort greift, wo nur
    `resolve_arcturian_turn` angeboten wird.
    """
    r = rr._arcturian_status_lookup_payload()["response"]
    assert r["tool_choice"] == "required"
    assert "agent_status" in [t["name"] for t in r["tools"]]
    assert r["metadata"][rr.ARCTURIAN_RESPONSE_KIND_FIELD] == "arcturian.status_lookup"


def test_nachschlag_zug_bringt_den_resolver_nicht_zurueck():
    namen = [t["name"] for t in rr._arcturian_status_lookup_payload()["response"]["tools"]]
    assert "resolve_arcturian_turn" not in namen
    assert "report_affect" not in namen


def test_persona_erwaehnt_das_werkzeug_NICHT():
    """Die teuerste Zeile dieses Tages, als Test.

    Jede Erwaehnung von `agent_status` in der Persona laesst das Modell
    es im Entscheidungs-Zug greifen — auch ein ausdrueckliches Verbot,
    denn ein Verbot ist eine Erwaehnung. Gemessen: Resolver 8/8 ohne
    Erwaehnung, 1/8 mit Aufforderung, 0/8 mit Verbot.
    """
    text = rr._companion_arcturian_prompt("de") + rr._arcturian_resolver_addendum("de")
    assert "agent_status" not in text, (
        "Der Werkzeugname steht in der Persona — gemessen bricht das den "
        "Entscheidungs-Zug. Der Nachschlag gehoert in den erzwungenen Zug."
    )


def test_nachschlag_zug_nur_mit_opt_in():
    import inspect
    quelle = inspect.getsource(rr)
    stelle = quelle.index('"status_lookup_response": (')
    fenster = quelle[stelle:stelle + 220]
    assert "request.read_tools" in fenster, (
        "Ohne Opt-in forderte die Nutzlast ein Werkzeug, das die Sitzung "
        "nicht fuehrt."
    )
