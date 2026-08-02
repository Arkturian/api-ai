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
    _arcturian_resolver_followup_payload,
    _arcturian_resolver_tools,
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
    print(f"\n{'all green' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
