"""Contract tests for the Arcturian turn resolver (issue #837).

Contract: approved product p-5cde1ac88a89 (Content-Post #4431),
`last_revised_at=2026-08-02T23:32:59`, approved by Cloud, Cloud-Codex,
AppDev-Realtime and Tschepp-Codex2.

These tests defend the failures that would be INVISIBLE in production —
which is the whole reason the resolver exists:

  * The resolver is skipped for "harmless" turns. Then the adapter is
    deciding what counts as actionable, i.e. the text heuristic the
    contract bans, just relocated. Measured: letting the model volunteer
    a tool-call lands at 12/18, and the misses cluster exactly where the
    answer turns actionable.
  * `none` is treated as an absence instead of a verdict. Then a missing
    receipt cannot be told apart from "nothing needed doing", and the
    receipt gate silently stops gating.
  * The forced turn uses the named-function tool_choice. Realtime accepts
    it and silently returns a plain message — 0/6 measured. The code looks
    right and never fires.
  * The model is handed ids or authority. It would invent them; the server
    would then either trust a fabricated class or lose correlation, both
    without an error.
  * create_task_proposal reappears in the session. The model could then
    volunteer a proposal whose risk classification is the server's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.realtime_routes import (  # noqa: E402
    RESOLVER_ACTION_KINDS,
    RESOLVER_DECISIONS,
    SUPPORTED_ARCTURIAN_RESOLVERS,
    _affect_projection_tools,
    _arcturian_resolver_addendum,
    _arcturian_primary_audio_payload,
    _arcturian_resolver_followup_payload,
    _arcturian_resolver_tools,
    _companion_arcturian_prompt,
    _detail_level_addendum,
)

FORBIDDEN_TOOLS = {
    "relay_to_agent",
    "propose_to_agent",
    # Removed from the session by #837 — a proposal is a server decision.
    "create_task_proposal",
    # The executor is server-side; the model must never reach it directly.
    "execute_federation_action",
}


def _tool():
    tools = _arcturian_resolver_tools()
    assert len(tools) == 1, f"expected exactly one tool, got {[t['name'] for t in tools]}"
    return tools[0]


def _props():
    return _tool()["parameters"]["properties"]


def test_contract_version_is_the_frozen_one():
    assert SUPPORTED_ARCTURIAN_RESOLVERS == {"agentos.arcturian-action.v1"}


def test_tool_is_named_resolve_arcturian_turn():
    assert _tool()["name"] == "resolve_arcturian_turn"
    assert _tool()["type"] == "function"


def test_none_is_a_first_class_verdict():
    """`none` must be sendable — it is what makes the resolver unconditional.

    Without it the adapter would have to decide when to run the resolver,
    reintroducing the banned heuristic.
    """
    assert "none" in _props()["decision"]["enum"]
    assert set(_props()["decision"]["enum"]) == set(RESOLVER_DECISIONS)
    assert set(RESOLVER_DECISIONS) == {"action", "clarify", "none"}


def test_action_kinds_match_the_envelope():
    assert set(_props()["kind"]["enum"]) == set(RESOLVER_ACTION_KINDS) | {None}


def test_disposition_fields_are_nullable_for_none_and_clarify():
    """A `none` verdict must not be forced to invent a target or content."""
    for field in ("kind", "target", "instruction"):
        assert "null" in _props()[field]["type"], f"{field} must be nullable"


def test_all_fields_required_so_nothing_arrives_absent():
    """Required + nullable beats optional: an omitted field is ambiguous."""
    assert set(_tool()["parameters"]["required"]) == {
        "decision", "kind", "target", "instruction",
    }


def test_model_never_supplies_ids():
    props = _props()
    for forbidden in (
        "action_id", "task_id", "conversation_id", "correlation_id",
        "message_id", "in_reply_to", "revision", "idempotency_key",
        "principal_user_id", "tenant_id", "resolver_call_id",
    ):
        assert forbidden not in props, f"{forbidden} must not be a model argument"


def test_model_never_supplies_authority():
    """A model-claimed authority or action class is ineffective by contract.

    Accepting one as an argument would invite the server to trust it.
    """
    props = _props()
    for forbidden in (
        "authority", "grant_id", "grant_revision", "class",
        "action_class", "allow_mode_promotion", "external_effects",
        "confirmed", "authorized",
    ):
        assert forbidden not in props, f"{forbidden} must not be a model argument"


def test_schema_is_closed():
    assert _tool()["parameters"]["additionalProperties"] is False


def test_session_never_exposes_a_forbidden_tool():
    names = {t["name"] for t in _arcturian_resolver_tools()}
    leaked = names & FORBIDDEN_TOOLS
    assert not leaked, f"forbidden tool in the arcturian session: {leaked}"


def test_followup_uses_required_not_named_tool_choice():
    """Measured: "required" fires 6/6, the named form 0/6 and silently.

    This test exists so nobody "cleans up" the payload into the form that
    looks more precise and is in fact inert.
    """
    resp = _arcturian_resolver_followup_payload()["response"]
    assert resp["tool_choice"] == "required", (
        "named-function tool_choice is silently ignored by Realtime"
    )


def test_followup_carries_only_the_resolver():
    """What makes "required" safe when the session also has report_affect.

    Without its own single-element list, the forced turn could be answered
    with the affect tool instead — and the turn would never be resolved.
    """
    tools = _arcturian_resolver_followup_payload()["response"]["tools"]
    assert [t["name"] for t in tools] == ["resolve_arcturian_turn"]


def test_followup_is_silent():
    """It runs BEFORE the spoken answer — audio here is a false start."""
    assert _arcturian_resolver_followup_payload()["response"]["output_modalities"] == ["text"]


def test_resolver_and_affect_are_separate_forced_turns():
    """The two mandatory calls must not collide.

    One forced turn yields ONE call. Resolver runs pre-audio, affect
    post-drain; each carries its own single-element list.
    """
    r = _arcturian_resolver_followup_payload()["response"]["tools"]
    a = _affect_projection_tools()
    assert [t["name"] for t in r] == ["resolve_arcturian_turn"]
    assert [t["name"] for t in a] == ["report_affect"]
    assert {t["name"] for t in r}.isdisjoint({t["name"] for t in a})


def test_correlation_metadata_is_server_set_on_both_followups():
    """Wire v1 (#886): correlation anchors are server-authorised.

    The model cannot reach metadata — it lives on response.create, which
    only client and server construct. That is what makes it usable as an
    authority anchor at all (Cloud, #886).
    """
    from ai.routes.realtime_routes import (
        ARCTURIAN_RESPONSE_KIND_FIELD, _affect_followup_payload,
    )
    r = _arcturian_resolver_followup_payload()["response"]["metadata"]
    a = _affect_followup_payload()["response"]["metadata"]
    assert r == {ARCTURIAN_RESPONSE_KIND_FIELD: "arcturian.resolver"}
    assert a == {ARCTURIAN_RESPONSE_KIND_FIELD: "agentos.affect"}


def test_server_never_sets_adapter_correlation_keys():
    """turn id and source response id are adapter-owned.

    The local turn id does not exist at mint time and the source response
    id only exists once the primary response is bound. Setting either
    server-side would mint a value that cannot be correct.
    """
    from ai.routes.realtime_routes import (
        ARCTURIAN_CORRELATION_KEYS, _affect_followup_payload,
    )
    for payload in (
        _arcturian_resolver_followup_payload(),
        _affect_followup_payload(),
        _arcturian_primary_audio_payload(),
    ):
        md = payload["response"]["metadata"]
        for key in ARCTURIAN_CORRELATION_KEYS:
            assert key not in md, f"{key} must be set by the adapter, not here"


def test_primary_audio_closes_the_tool_gate_explicitly():
    """Measured: omitting tools/tool_choice yields 2/2 voluntary calls.

    A response override INHERITS the session tools, so a spoken turn
    without an explicit gate re-offers resolve_arcturian_turn and
    report_affect — and the model takes them. Omission is an open gate,
    not a neutral default. Both fields are asserted: the empty list makes
    the intent explicit and survives a change in inheritance semantics.
    """
    resp = _arcturian_primary_audio_payload()["response"]
    assert resp["tools"] == [], "empty tool list is the gate, not an omission"
    assert resp["tool_choice"] == "none"


def test_primary_audio_does_not_pin_modalities():
    """Forcing text here would mute the spoken answer."""
    assert "output_modalities" not in _arcturian_primary_audio_payload()["response"]


def test_prompt_states_the_receipt_gate():
    """No success claim without a committed receipt — the point of #837."""
    text = _arcturian_resolver_addendum("de").lower()
    for claim in ("gesendet", "gestartet", "erledigt"):
        assert claim in text, f"prompt must name the forbidden claim {claim!r}"
    assert "resolve_arcturian_turn" in _arcturian_resolver_addendum("de")


def test_prompt_forbids_paraphrasing_a_dictated_message():
    """The owner dictated 'Hallo AppDev. Mehr nicht.' — verbatim means verbatim."""
    text = _arcturian_resolver_addendum("de")
    assert "WOERTLICH" in text or "woertlich" in text.lower()


def test_arcturian_prompt_carries_no_narration_cadence():
    """Arcturian is not a narrator — detail_level must not reach him.

    Every observed session ran `detail_level=flowing`, whose text asks for
    a "durchgehender mündlicher Bericht … alle 4-10 Sekunden ein Satz".
    That is precisely the paraphrasing and self-commentary the owner
    rejected after the physical iPhone tests. A narration cadence and a
    terse operative dialogue cannot both hold, and the narration text was
    winning. This guards the two prompt pieces the mode actually uses.
    """
    used = _companion_arcturian_prompt("de") + _arcturian_resolver_addendum("de")
    flowing = _detail_level_addendum("flowing")
    # Sanity: the narrator text really does carry a cadence instruction.
    assert "Erzählfluss" in flowing or "Erzahlfluss" in flowing
    for marker in ("DETAIL-LEVEL", "Erzählfluss", "4-10 Sekunden", "5-15 Sekunden"):
        assert marker not in used, (
            f"narration cadence {marker!r} leaked into the arcturian prompt"
        )


def test_persona_does_not_deny_the_capability_its_tool_provides():
    """The persona must not contradict the resolver.

    This is the defect AppDevV2 reported on 2026-08-04: #837 swapped the
    tool and appended an addendum, but left the #751 persona in place —
    which stated "DEINE EINZIGE HANDLUNG: create_task_proposal" and "Du
    hast kein Werkzeug, um einen Agenten zu kontaktieren". Asked to send
    a message, Arcturian refused, quoting the persona.

    An absolute prohibition early in the prompt outranks a later
    paragraph granting the ability — the model has no reason to read the
    second as overriding the first. So the persona is checked for the
    denials themselves, not just for the presence of new text.
    """
    persona = _companion_arcturian_prompt("de")
    for denial in (
        "create_task_proposal",
        "kein Werkzeug",
        "EINZIGE HANDLUNG",
        "erreicht NIEMANDEN",
        "Aufgaben-Entwuerfe",
    ):
        assert denial not in persona, (
            f"persona still carries the overruled proposal-only role: {denial!r}"
        )


def test_persona_states_the_capability_positively():
    """Absence of a denial is not the same as granting the ability."""
    persona = _companion_arcturian_prompt("de")
    assert "fuehrst du unmittelbar aus" in persona or "fuehrst sie aus" in persona
    # The verbatim rule must survive: the owner dictated "Mehr nicht."
    assert "WOERTLICH" in persona


def test_persona_and_addendum_do_not_contradict():
    """Composed prompt must not both grant and forbid contacting agents."""
    composed = _companion_arcturian_prompt("de") + _arcturian_resolver_addendum("de")
    grants = "sende an" in composed.lower()
    denies = "kein werkzeug" in composed.lower() or "koenntest keinen" in composed.lower()
    assert grants, "composed prompt never states that sending is possible"
    # "Sage niemals, du koenntest keinen Agenten kontaktieren" is a rule
    # ABOUT the denial, not a denial — so check the bare prohibition form.
    assert "Du hast kein Werkzeug" not in composed


def test_arcturian_prompt_forbids_talking_about_the_mechanism():
    """'Interne Vertragsbegriffe … werden nicht vorgelesen' (owner correction)."""
    text = _arcturian_resolver_addendum("de")
    assert "Sprich nie ueber diesen Mechanismus" in text


def test_prompt_is_emitted_for_any_language():
    for lang in ("de", "en", "sl", "it", None):
        text = _arcturian_resolver_addendum(lang)
        assert "resolve_arcturian_turn" in text
        assert len(text) > 100


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
            except Exception as exc:  # NameError etc. must not truncate the run
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all green' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
