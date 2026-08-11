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
    ARCTURIAN_RESOLVER_V1,
    ARCTURIAN_RESOLVER_V2,
    DEFAULT_ARCTURIAN_RESOLVER,
    SUPPORTED_ARCTURIAN_RESOLVERS,
    _affect_projection_tools,
    _arcturian_resolver_addendum,
    _arcturian_primary_audio_payload,
    _arcturian_resolver_followup_payload,
    _arcturian_resolver_tools,
    _companion_arcturian_prompt,
    _detail_level_addendum,
    _session_tools,
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


def _tool_v2():
    """v2 must be asked for by name — the default is v1 on purpose."""
    tools = _arcturian_resolver_tools(ARCTURIAN_RESOLVER_V2)
    assert len(tools) == 1
    return tools[0]


def _props_v2():
    return _tool_v2()["parameters"]["properties"]


def test_contract_version_is_the_frozen_one():
    """v2 since Post #4518: navigate_ui + target_kind.

    Bumping this is deliberate — a widened schema is a new revision and
    needs a fresh approval, because the v1 sign-off was bound to its
    revision, not to the contract name.
    """
    # Enthaltensein statt Gleichheit. Der Zweck dieses Tests ist, eine
    # AENDERUNG an genehmigten Fassungen zu bemerken — nicht, das
    # Hinzufuegen einer neuen zu verhindern. Die Gleichheitsfassung ging
    # rot, als v3 dazukam, obwohl v1 und v2 unveraendert blieben.
    #
    # Am selben Tag legte dieselbe Denkweise auf der Client-Seite jede
    # Arcturian-Sitzung still (CloudV2, 2026-08-11 15:46): Eine
    # Vertragspruefung verglich meine Adapter-Kinds auf Gleichheit und
    # warf, als die Menge um zwei erlaubte Eintraege wuchs. Eine
    # fail-closed Pruefung darf nur pinnen, was schuetzenswert ist.
    assert {ARCTURIAN_RESOLVER_V1, ARCTURIAN_RESOLVER_V2} <= SUPPORTED_ARCTURIAN_RESOLVERS
    # Which one an un-asking client gets is the migration-critical half.
    assert DEFAULT_ARCTURIAN_RESOLVER == ARCTURIAN_RESOLVER_V1
    # Das eigentlich Schuetzenswerte: Die genehmigten Fassungen duerfen
    # sich nicht ausweiten. Eine neue Art gehoert in eine neue Fassung.
    from ai.routes.realtime_routes import _arcturian_resolver_tools
    for rev, erwartet in (
        (ARCTURIAN_RESOLVER_V1, 4), (ARCTURIAN_RESOLVER_V2, 5),
    ):
        felder = _arcturian_resolver_tools(rev)[0]["parameters"]["required"]
        assert len(felder) == erwartet, f"{rev} hat sich ausgeweitet: {felder}"
        arten = _arcturian_resolver_tools(rev)[0]["parameters"]["properties"]["kind"]["enum"]
        assert "query_status" not in arten, f"{rev} darf query_status nicht kennen"


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
    assert set(_props_v2()["kind"]["enum"]) == set(RESOLVER_ACTION_KINDS) | {None}


def test_disposition_fields_are_nullable_for_none_and_clarify():
    """A `none` verdict must not be forced to invent a target or content."""
    for field in ("kind", "target", "instruction"):
        assert "null" in _props()[field]["type"], f"{field} must be nullable"


def test_all_fields_required_so_nothing_arrives_absent():
    """Required + nullable beats optional: an omitted field is ambiguous."""
    assert set(_tool_v2()["parameters"]["required"]) == {
        "decision", "kind", "target", "target_kind", "instruction",
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


def test_primary_audio_states_its_tools_explicitly():
    """Measured: omitting tools/tool_choice yields 2/2 voluntary calls.

    A response override INHERITS the session tools, so a spoken turn
    without explicit fields re-offers resolve_arcturian_turn and
    report_affect — and the model takes them. Omission is an open gate,
    not a neutral default. Both fields must therefore always be present.

    The list is no longer empty: since 2026-08-09 the owner decided
    Arcturian gets read tools, and this is the ONLY turn where they can
    fire — the other two templates pin their single forced tool. A read
    tool that lives only in the session list can never be called.
    """
    resp = _arcturian_primary_audio_payload()["response"]
    assert "tools" in resp and "tool_choice" in resp, (
        "omitting either field re-opens the gate through inheritance"
    )
    # Default is the contract every shipped client was built against.
    assert resp["tools"] == [], "a client that does not ask must see no change"
    assert resp["tool_choice"] == "none"


def test_primary_audio_offers_only_read_tools():
    """What the gate is still for.

    The specific defect AppDevV2 reproduced on-device was the model
    reaching for report_affect in a turn that never offered it. That stays
    impossible: only read tools are on this turn.
    """
    from ai.routes.realtime_routes import _arcturian_read_tools
    offered = {t["name"] for t in
               _arcturian_primary_audio_payload(True)["response"]["tools"]}
    assert offered == {t["name"] for t in _arcturian_read_tools()}
    assert "report_affect" not in offered
    assert "resolve_arcturian_turn" not in offered


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


def test_persona_answers_the_owner_about_its_own_workings():
    """Anti-verbosity must not become refusal toward the owner.

    Observed on device (AppDevV2, 2026-08-05): asked about his own
    procedure, Arcturian answered "Diese Details zum Vorgehen kann ich
    nicht besprechen" — to the person the system belongs to. The rule
    "never talk about the mechanism" was written against chatter and hit
    a direct question instead.
    """
    persona = _companion_arcturian_prompt("de")
    assert "FRAGT er dich danach, antwortest du" in persona
    assert "nichts vor ihm zu verbergen" in persona
    # The unprompted-chatter rule must survive.
    assert "unaufgefordert nichts" in persona


def test_verbatim_rule_excludes_placeholders():
    """"Sende irgendwas" is permission to compose, not a message text.

    Observed on device: the operator said "sende irgendwas, egal was";
    Arcturian first asked back and then sent the literal word
    "Irgendwas." as the message body — the verbatim rule applied to a
    placeholder it was never meant for.
    """
    persona = _companion_arcturian_prompt("de")
    assert "KEIN Nachrichtentext" in persona
    assert "Erlaubnis, selbst zu formulieren" in persona
    assert "fragst nicht nach" in persona
    # And the real dictation case must still demand verbatim.
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


def test_persona_teaches_both_order_types():
    """Delegation must be taught, not just verbatim messaging.

    Measured before the fix: "Klär das mit AppDev" produced
    kind=send_internal_message in 3 of 3 runs — never delegate_internal.
    The schema offered the capability, the persona never mentioned it.
    Exact mirror of the #837 defect (capable but forbidden), inverted:
    capable but unmentioned. Reported by AppDevV2.
    """
    persona = _companion_arcturian_prompt("de")
    assert "DELEGATION" in persona, "persona teaches only verbatim messaging"
    assert "BOTSCHAFT" in persona
    # A delegation must not invent a wording the operator never said.
    assert "Erfinde dafuer keinen" in persona


def test_resolver_addendum_offers_wording_for_work_in_progress():
    """Without a permitted phrase for "still running", the model reaches
    for a forbidden one.

    Measured: with receipt {status: queued, delivered: false} the model
    said "Ist beauftragt" — one of the words the same prompt explicitly
    forbids, about a message that had not been delivered. The ban alone
    is not enough; there has to be something true to say instead.
    """
    text = _arcturian_resolver_addendum("de")
    assert "WAEHREND ETWAS LAEUFT" in text
    assert "die Antwort steht noch aus" in text
    assert "Warten ist" in text, "waiting must be framed as normal, not as failure"
    # The ban must still be there — the wording is an alternative, not a licence.
    for claim in ("gesendet", "beauftragt", "erledigt"):
        assert claim in text


def test_persona_permits_thinking_out_loud():
    """Cloud's condition: thinking must be allowed to be audible.

    The anti-verbosity rules ("keine Selbstbeschreibung", "im Normalfall
    ein Satz") also hit "das muss ich kurz nachsehen". Cloud caught four
    of her own mismeasurements today because she had time to re-check;
    in a spoken turn she would have asserted them. A prompt that forbids
    the pause trains fluency that deceives.
    """
    persona = _companion_arcturian_prompt("de")
    assert "Ich sehe kurz nach" in persona
    assert "das weiss ich nicht" in persona
    assert "Kurz heisst knapp, nicht vorschnell" in persona


def test_clarify_criterion_is_objective_not_knowledge_based():
    """The model has no agent list — "unknown recipient" is unanswerable.

    Replaced by the only testable criterion: did the operator name a
    recipient at all. A name the model does not recognise must be passed
    through, because resolving it is the server's job.
    """
    persona = _companion_arcturian_prompt("de")
    assert "gar keinen Empfaenger genannt" in persona
    assert "nicht kennst, fragst du NICHT nach" in persona
    # The old, knowledge-based wording must be gone.
    assert "unbekannter Empfaenger" not in persona


def test_arcturian_prompt_forbids_talking_about_the_mechanism():
    """'Interne Vertragsbegriffe … werden nicht vorgelesen' (owner correction)."""
    text = _arcturian_resolver_addendum("de")
    assert "Sprich nie ueber diesen Mechanismus" in text


def test_prompt_is_emitted_for_any_language():
    for lang in ("de", "en", "sl", "it", None):
        text = _arcturian_resolver_addendum(lang)
        assert "resolve_arcturian_turn" in text
        assert len(text) > 100


# --- session.tools = [] (AppDevV2 device reproduction, 2026-08-06) ---------
#
# The device logged `tool_misrouted got=report_affect
# expected=resolve_arcturian_turn`. 72 runs against this endpoint never
# reproduced the trigger, so the fix is structural rather than a
# mitigation: leave nothing in the session for a stray turn to reach.


def test_arcturian_session_never_carries_the_affect_tool():
    """The one rule the on-device misroute made necessary.

    Both arcturian turns ship their own single-element `tools` list on
    the response.create override, so the session list is never read on a
    healthy turn — it can only be reached by a turn that should not have
    called anything.
    """
    from ai.routes.realtime_routes import _arcturian_read_tools
    names = {t["name"] for t in _session_tools(_arcturian_read_tools(), "arcturian", "v1")}
    assert "report_affect" not in names, (
        "the exact tool AppDevV2 reproduced as tool_misrouted must stay per-turn"
    )
    assert "resolve_arcturian_turn" not in names, (
        "the resolver is a forced turn, never a session tool"
    )


def test_arcturian_stays_free_of_affect_regardless_of_version():
    """Read tools are the owner's decision (2026-08-09); the affect tool
    is not — it stays a forced turn in every contract version."""
    from ai.routes.realtime_routes import _arcturian_read_tools
    for version in ("v1", "v2", "anything-future"):
        names = {t["name"] for t in
                 _session_tools(_arcturian_read_tools(), "arcturian", version)}
        assert "report_affect" not in names


def test_affect_still_reaches_every_other_mode():
    """The exception is arcturian's alone — it must not become the rule.

    report_affect is a signal, not a capability, so zero-tool modes such
    as narrator-only and guide-ptt are deliberately armed with it too.
    """
    for mode in ("narrator-only", "guide-ptt", "talkback-enabled", None):
        names = {t["name"] for t in _session_tools([], mode, "v1")}
        assert "report_affect" in names, f"{mode} lost the affect contract"


def test_affect_tool_is_absent_when_projection_is_off():
    for mode in ("arcturian", "narrator-only", None):
        assert _session_tools([], mode, None) == []


def test_session_tools_never_mutates_its_input():
    base: list = []
    _session_tools(base, "narrator-only", "v1")
    assert base == [], "defensive copy missing — caller's list was mutated"


def test_affect_tool_is_never_added_twice():
    once = _session_tools(_affect_projection_tools(), "narrator-only", "v1")
    names = [t["name"] for t in once]
    assert names.count("report_affect") == 1, names



def test_persona_forbids_claiming_a_history_it_was_not_given():
    """Alex asked "Bist du mit einer kompletten History gestartet?" and
    Arcturian said yes. The mint attaches no transcript and the client
    starts with snapshot=nil, so the answer was invented.

    The persona said nothing about memory either way — neither claiming
    nor denying it — and the model filled the gap with the friendly
    answer. This is the failure mode that cost the owner the most time
    today: statements that sounded helpful and were not true.
    """
    persona = _companion_arcturian_prompt("de")
    assert "Wurde dir kein Verlauf mitgegeben" in persona
    assert "Ich starte ohne Verlauf" in persona


def test_persona_ties_the_answer_to_the_session_not_to_helpfulness():
    persona = _companion_arcturian_prompt("de")
    assert "nicht nach dem, was hilfreich klaenge" in persona


# --- v2: UI navigation (Post #4518, section q-3188aaa6479a) ---------------


def test_navigate_ui_is_the_fifth_kind():
    assert "navigate_ui" in _props_v2()["kind"]["enum"]
    assert set(RESOLVER_ACTION_KINDS) == {
        "send_internal_message", "delegate_internal",
        "create_collab", "start_workflow", "navigate_ui",
    }


def test_navigation_is_not_an_executable_kind():
    """Cloud's second bolt keys off this list.

    navigate_ui must never appear where a kind is checked for
    executability — it carries no authority and produces no receipt.
    """
    from ai.routes.realtime_routes import RESOLVER_EXECUTABLE_KINDS
    assert "navigate_ui" not in RESOLVER_EXECUTABLE_KINDS
    assert len(RESOLVER_EXECUTABLE_KINDS) == 4


def test_target_kind_is_a_free_string_not_an_enum():
    """The list of views belongs to the CLIENT, not to this schema.

    Cloud declined to own it and was right: views change with the app,
    not with the contract. An enum here would make every new iOS view a
    server deployment.
    """
    assert "enum" not in _props_v2()["target_kind"]
    assert "null" in _props_v2()["target_kind"]["type"]


def test_target_is_the_same_namespace_for_focus_agent():
    """AppDevV2's condition: one field must not mean two things.

    For focus_agent, `target` is an agent name resolved exactly as for
    send_internal_message — so "geh zu appdevv2" cannot fail for the same
    reason "sende an appdevv2" failed today.
    """
    desc = _props()["target"]["description"]
    assert "focus_agent" in desc
    assert "same namespace" in desc.lower()


def test_kind_description_separates_showing_from_sending():
    desc = _props()["kind"]["description"]
    assert "navigate_ui" in desc
    assert "sends nothing to anybody" in desc


def test_persona_teaches_that_showing_is_not_sending():
    """The confusion this prevents is asymmetric and irreversible.

    A wrongly opened view costs a tap. A wrongly sent message sits in
    someone else's window and cannot be taken back.
    """
    persona = _companion_arcturian_prompt("de")
    assert "ANSEHEN IST NICHT SENDEN" in persona
    assert "geht dabei nichts an " in persona


def test_persona_decides_on_the_verb_not_the_name():
    persona = _companion_arcturian_prompt("de")
    assert "am Verb, nicht am Namen" in persona


def test_persona_biases_towards_not_navigating_when_unsure():
    persona = _companion_arcturian_prompt("de")
    assert "ist es KEINE Navigation" in persona


def test_persona_leaves_unsupported_targets_to_the_device():
    """The device speaks the refusal; the model must not pre-empt it."""
    persona = _companion_arcturian_prompt("de")
    assert "sagt " in persona and "nicht zeigen" in persona


def test_navigate_ui_demands_a_target_kind():
    """Cloud's addendum to the v2 sign-off (#4518, 21:16).

    JSON Schema cannot express "if kind == navigate_ui then target_kind
    is not null" in a way the model reliably honours, so the rule lives
    where it actually bites: the schema TELLS the model, and the client
    treats a null target_kind exactly like an unknown one — no
    navigation, spoken hint, log line.

    Without this, null is the one value that is neither allowed nor
    forbidden. Three of today's incidents lived in exactly such an
    in-between state.
    """
    desc = _props_v2()["target_kind"]["description"]
    assert "MUST name a target_kind" in desc
    assert "not a neutral choice" in desc


# --- Brain-Regeln aus den Gespraechen vom 2026-08-07 (AppDevV2-Analyse) ---
#
# Einzeln benannt und einzeln getestet, ausdruecklich auf AppDevV2s Bitte:
# bei vier Regeln in einer Sammel-Umschreibung waere bei einer
# Verschlechterung nicht mehr zuzuordnen, welche es war. Genau so ist am
# 04.08. der #751-Satz stehengeblieben und hat #837 ueberstimmt.


def test_regel1_target_comes_from_the_sentence_not_the_conversation():
    """Der teuerste Fehler des Tages, mit Beleg.

    19:39 sagt der Operator "schick AppDevV2 eine Testnachricht", das
    Modell sagt "ich schicke AppDevV2" — und der Resolver liefert
    target=3dApi, weil ueber 3dApi Minuten vorher gesprochen wurde. Die
    Nachricht ging ab, an den Falschen, und beide Seiten hielten sie fuer
    zugestellt. Zwei Stunden Suche an der Zustellung, die nie kaputt war.
    """
    persona = _companion_arcturian_prompt("de")
    assert "REGEL 1 — DAS ZIEL STEHT IM SATZ" in persona
    assert "AUSSCHLIESSLICH aus der" in persona
    assert "ist Zusammenhang — kein Ziel" in persona


def test_regel2_spelling_is_never_a_reason_to_ask_back():
    """'sende an appdevv2' loeste eine Rueckfrage zur Schreibweise aus."""
    persona = _companion_arcturian_prompt("de")
    assert "REGEL 2 — SCHREIBWEISE IST KEIN GRUND" in persona
    assert "'appdevv2' ist AppDevV2" in persona
    assert "nie, weil dir die Schreibweise komisch vorkommt" in persona


def test_regel3_uses_the_server_supplied_last_target():
    """Der Client haengt das zuletzt benutzte Ziel an (f7e7f6b).

    Ohne diese Regel fragt das Modell trotzdem zurueck, und der Hinweis
    verpufft.
    """
    persona = _companion_arcturian_prompt("de")
    assert "REGEL 3 — 'NOCHMAL' IST EIN ZIEL" in persona
    assert "NIMM ES. Frag nicht erneut" in persona


def test_regel3_explicitly_reconciles_itself_with_regel1():
    """Zwei Regeln, die einander widersprechen, sind schlimmer als eine.

    Regel 1 verbietet Namen aus dem Verlauf, Regel 3 verlangt einen —
    der Unterschied (Server nennt ihn vs. Modell holt ihn sich) muss IM
    Text stehen, sonst gewinnt die absolutere. Genau so hat der
    #751-Satz die #837-Faehigkeit ueberstimmt.
    """
    persona = _companion_arcturian_prompt("de")
    assert "kein Widerspruch zu Regel 1" in persona
    assert "Ohne diesen Hinweis gilt" in persona


def test_regel4_never_acts_on_unintelligible_input():
    """Der gefaehrlichste: ein Whisper-Artefakt loeste eine Aktion aus.

    Es ist der einzige dieser Fehler, der etwas an Dritte schickt.
    """
    persona = _companion_arcturian_prompt("de")
    assert "REGEL 4 — WAS DU NICHT VERSTANDEN HAST" in persona
    assert "NIEMALS 'action'" in persona


def test_regel4_names_the_known_artefacts_verbatim():
    """Die Artefakte muessen woertlich dastehen — 'unverstaendlich' allein
    erkennt das Modell bei einem grammatisch sauberen Satz nicht."""
    persona = _companion_arcturian_prompt("de")
    for artefact in ("Untertitelung aufgrund der Audioqualitaet",
                     "Vielen Dank fuers Zuschauen"):
        assert artefact in persona, artefact


def test_all_four_rules_are_present_and_numbered_once():
    """Doppelte oder fehlende Nummern machen eine Verschlechterung
    unzuordenbar — der Grund, warum sie einzeln benannt sind."""
    persona = _companion_arcturian_prompt("de")
    for n in (1, 2, 3, 4):
        assert persona.count(f"REGEL {n} — ") == 1, f"REGEL {n}"

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
