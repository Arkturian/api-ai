"""Resolver revision negotiation (#4518, Cloud-Codex review).

The defect these guard against was mine and it was structural: a single
global constant flipped to v2 hands EVERY already-shipped client an
identifier it compares for exact equality. The mint succeeds, the client
dies in `invalidTokenContract` before the resolver ever runs — a hard
break wearing a version bump's clothing.

Same "client first, then server" lesson that cost the owner his voice
sessions this morning, re-made one layer down. Hence: the revision is
negotiated, never decreed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.realtime_routes import (  # noqa: E402
    ARCTURIAN_RESOLVER_V1,
    ARCTURIAN_RESOLVER_V2,
    DEFAULT_ARCTURIAN_RESOLVER,
    RESOLVER_EXECUTABLE_KINDS,
    SUPPORTED_ARCTURIAN_RESOLVERS,
    RealtimeTokenRequest,
    _arcturian_resolver_followup_payload,
    _arcturian_resolver_tools,
)


def _schema(resolver):
    return _arcturian_resolver_tools(resolver)[0]["parameters"]


def test_default_is_v1_so_legacy_clients_never_break():
    """The whole point. A client that asks for nothing keeps working."""
    assert DEFAULT_ARCTURIAN_RESOLVER == ARCTURIAN_RESOLVER_V1


def test_omitting_the_field_means_v1():
    req = RealtimeTokenRequest(companion_mode="arcturian")
    assert req.arcturian_resolver is None
    resolved = req.arcturian_resolver or DEFAULT_ARCTURIAN_RESOLVER
    assert resolved == ARCTURIAN_RESOLVER_V1


def test_both_revisions_are_supported_simultaneously():
    """Migration needs an overlap window; one-at-a-time is the break."""
    # Enthaltensein, NICHT Gleichheit. Die Gleichheitsfassung dieses
    # Tests ging am 2026-08-11 rot, als v3 dazukam — obwohl nichts
    # kaputt war. Am selben Tag legte dieselbe Denkweise auf der
    # Client-Seite Arcturian eine Stunde still: CloudV2s Vertragspruefung
    # verglich die Adapter-Kinds auf Gleichheit und warf, als meine
    # Menge um zwei erlaubte Eintraege wuchs.
    #
    # Eine Pruefung darf nur pinnen, was schuetzenswert ist. Hier ist das
    # die fortgesetzte Unterstuetzung der ausgelieferten Fassungen — nicht
    # die Abwesenheit kuenftiger.
    assert {ARCTURIAN_RESOLVER_V1, ARCTURIAN_RESOLVER_V2} <= SUPPORTED_ARCTURIAN_RESOLVERS


def test_v1_schema_is_exactly_what_shipped_clients_decode():
    s = _schema(ARCTURIAN_RESOLVER_V1)
    assert s["required"] == ["decision", "kind", "target", "instruction"]
    assert "target_kind" not in s["properties"], (
        "a wider properties map is still a wider schema under "
        "additionalProperties:false"
    )
    assert s["properties"]["kind"]["enum"] == RESOLVER_EXECUTABLE_KINDS + [None]


def test_v1_never_offers_navigate_ui():
    """A v1 client has no branch for it — it would reach the action path."""
    assert "navigate_ui" not in _schema(ARCTURIAN_RESOLVER_V1)["properties"]["kind"]["enum"]


def test_v2_schema_has_all_five_required_fields():
    s = _schema(ARCTURIAN_RESOLVER_V2)
    assert set(s["required"]) == {
        "decision", "kind", "target", "target_kind", "instruction",
    }
    assert "navigate_ui" in s["properties"]["kind"]["enum"]


def test_both_revisions_stay_closed():
    for resolver in SUPPORTED_ARCTURIAN_RESOLVERS:
        assert _schema(resolver)["additionalProperties"] is False


def test_shared_fields_cannot_drift_between_revisions():
    """Built from one code path so v1 and v2 agree where they overlap."""
    v1 = _schema(ARCTURIAN_RESOLVER_V1)["properties"]
    v2 = _schema(ARCTURIAN_RESOLVER_V2)["properties"]
    for field in ("decision", "target", "instruction"):
        assert v1[field] == v2[field], f"{field} drifted between revisions"


def test_followup_carries_the_negotiated_schema():
    """A v1 client must never receive the five-field tool."""
    v1 = _arcturian_resolver_followup_payload(ARCTURIAN_RESOLVER_V1)
    v2 = _arcturian_resolver_followup_payload(ARCTURIAN_RESOLVER_V2)
    assert "target_kind" not in v1["response"]["tools"][0]["parameters"]["properties"]
    assert "target_kind" in v2["response"]["tools"][0]["parameters"]["properties"]


def test_followup_defaults_to_v1_as_well():
    payload = _arcturian_resolver_followup_payload()
    props = payload["response"]["tools"][0]["parameters"]["properties"]
    assert "target_kind" not in props


def test_request_model_accepts_an_explicit_revision():
    req = RealtimeTokenRequest(
        companion_mode="arcturian", arcturian_resolver=ARCTURIAN_RESOLVER_V2,
    )
    assert req.arcturian_resolver == ARCTURIAN_RESOLVER_V2


def test_unknown_revision_is_not_silently_downgraded():
    """Fail closed. A downgrade hands a v2 client a v1 schema, and the
    mismatch then surfaces three layers from its cause."""
    for bogus in ("agentos.arcturian-action.v99", "v2", "", "latest"):
        assert bogus not in SUPPORTED_ARCTURIAN_RESOLVERS


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)


# ------------------------------------------------------------------ v3

def test_v3_kennt_query_status_und_v2_bleibt_unberuehrt():
    """Eigene Fassung statt Erweiterung — sonst bricht jedes v2-Geraet.

    Der Client prueft fail-closed auf unbekannte Arten. Haette ich
    `query_status` in v2 gelegt, antwortete ein ausgeliefertes Geraet
    auf JEDE Statusfrage mit `invalid_resolver_payload_unknown_kind` —
    also mit genau dem Fehler, unter dem Alexander die Woche gelitten
    hat. Die Versionierung erzwingt die Reihenfolge, statt sie zu
    verabreden: Wer v3 nicht anfordert, sieht die Art nie.
    """
    from ai.routes.realtime_routes import (
        _arcturian_resolver_tools, ARCTURIAN_RESOLVER_V2, ARCTURIAN_RESOLVER_V3,
        ARCTURIAN_RESOLVER_V1,
    )
    v1 = _arcturian_resolver_tools(ARCTURIAN_RESOLVER_V1)[0]["parameters"]
    v2 = _arcturian_resolver_tools(ARCTURIAN_RESOLVER_V2)[0]["parameters"]
    v3 = _arcturian_resolver_tools(ARCTURIAN_RESOLVER_V3)[0]["parameters"]
    assert "query_status" not in v1["properties"]["kind"]["enum"]
    assert "query_status" not in v2["properties"]["kind"]["enum"]
    assert "query_status" in v3["properties"]["kind"]["enum"]
    # v3 erbt die v2-Pflichtfelder, damit CloudV2s Parser nicht zweimal
    # umgebaut werden muss.
    assert v3["required"] == v2["required"]


def test_v3_anweisung_nennt_das_nachschlag_werkzeug_nicht():
    """Der teuerste Befund des Tages, als Test an der v3-Anweisung.

    Gemessen: Resolver 8/8 ohne Erwaehnung, 1/8 mit Aufforderung, 0/8
    mit ausdruecklichem Verbot. Ein Verbot ist eine Erwaehnung.
    """
    from ai.routes.realtime_routes import (
        _arcturian_resolver_addendum, ARCTURIAN_RESOLVER_V2, ARCTURIAN_RESOLVER_V3,
    )
    v3 = _arcturian_resolver_addendum("de", ARCTURIAN_RESOLVER_V3)
    assert "query_status" in v3
    assert "agent_status" not in v3
    # Und der Absatz darf nicht in v2 lecken.
    assert "query_status" not in _arcturian_resolver_addendum("de", ARCTURIAN_RESOLVER_V2)


def test_v3_absatz_widerspricht_der_feldbeschreibung_nicht():
    """Zwei Stellen desselben Prompts duerfen nicht Verschiedenes sagen.

    Bis 2026-08-11 sagte der v3-Absatz „`target_kind: agent`", waehrend
    die Feldbeschreibung „Only for kind='navigate_ui', otherwise null"
    vorschreibt. Das Modell folgte der Feldbeschreibung — richtig — und
    Alexanders erste Sitzung nach der 500er-Reparatur endete an
    `invalid_resolver_payload_missing_target_kind`.

    Dieselbe Fehlerklasse wie die Zwischenformulierungen am selben Tag:
    Ein Widerspruch im Prompt sieht aus wie Ungehorsam des Modells.
    """
    from ai.routes.realtime_routes import (
        _arcturian_resolver_addendum, _arcturian_resolver_tools,
        ARCTURIAN_RESOLVER_V3,
    )
    absatz = _arcturian_resolver_addendum("de", ARCTURIAN_RESOLVER_V3)
    feld = (_arcturian_resolver_tools(ARCTURIAN_RESOLVER_V3)[0]
            ["parameters"]["properties"]["target_kind"]["description"])

    # Die Feldbeschreibung bindet target_kind an navigate_ui.
    assert "navigate_ui" in feld and "null" in feld
    # Also darf der Absatz es fuer query_status NICHT verlangen.
    assert "target_kind: agent" not in absatz, (
        "Der v3-Absatz verlangt ein Feld, das die Feldbeschreibung fuer "
        "diesen Fall auf null festlegt"
    )
