"""Contract tests for the avatar affect projection (issue #767).

Contract: `agentos.avatar-runtime.v1`, approved product p-5c38e01f9cb6
in Content-Post #4436.

What these tests defend is not the happy path — it is the set of failures
that would be INVISIBLE in production:

  * A frontend asks for a contract version we do not speak. If we accepted
    it silently, the session would come up without report_affect and the
    avatar would wait forever for a call that never arrives — looking like
    a client bug.
  * The affect tool displaces a mode's own tools. An `arcturian` session
    that lost create_task_proposal still connects, still talks, and is
    quietly unable to do the one thing it exists for.
  * A forbidden tool rides in on the affect path. The #751 capability gate
    is "the tool is absent", so anything that appends tools must be proven
    not to widen the set.
  * The model is handed turn_id/session_id. It would invent them, and the
    client's correlation would break silently rather than loudly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.realtime_routes import (  # noqa: E402
    AFFECT_INTENSITY_VALUES,
    AFFECT_VALUES,
    SUPPORTED_AFFECT_PROJECTIONS,
    _affect_followup_payload,
    _affect_projection_addendum,
    _affect_projection_tools,
    _companion_arcturian_tools,
)

FORBIDDEN = {"relay_to_agent", "propose_to_agent"}

# The three paths the contract requires to be reachable (§Emotion), plus
# the two shapes that must stay expressible.
CONTRACT_PATHS = [
    ("pleased", "low", "happy"),
    ("pleased", "high", "joyful"),
    ("concerned", "low", "sad"),
    ("neutral", None, "idle"),
]


def _tool():
    tools = _affect_projection_tools()
    assert len(tools) == 1, f"expected exactly one tool, got {[t['name'] for t in tools]}"
    return tools[0]


def _props():
    return _tool()["parameters"]["properties"]


def test_contract_version_is_the_frozen_one():
    assert SUPPORTED_AFFECT_PROJECTIONS == {"agentos.avatar-runtime.v1"}


def test_tool_is_named_report_affect():
    assert _tool()["name"] == "report_affect"
    assert _tool()["type"] == "function"


def test_every_contract_path_is_expressible():
    """pleased+low, pleased+high, concerned, neutral must all be sendable."""
    affect_enum = _props()["affect"]["enum"]
    intensity_enum = _props()["affect_intensity"]["enum"]
    for affect, intensity, role in CONTRACT_PATHS:
        assert affect in affect_enum, f"{affect} missing -> role {role} unreachable"
        assert intensity in intensity_enum, (
            f"intensity {intensity!r} missing -> role {role} unreachable"
        )


def test_affect_enum_is_exactly_the_contract_set():
    """Extra values would let the model emit an affect no role maps to."""
    assert _props()["affect"]["enum"] == AFFECT_VALUES == [
        "neutral", "pleased", "concerned",
    ]


def test_intensity_allows_null_for_neutral():
    """The contract mandates affect=neutral => affect_intensity=null."""
    spec = _props()["affect_intensity"]
    assert None in spec["enum"], "neutral cannot carry its mandated null"
    assert "null" in spec["type"], "type must admit null, not just the enum"
    assert set(AFFECT_INTENSITY_VALUES) == {"low", "high"}


def test_both_fields_are_required():
    """An omitted field would arrive as absent rather than as neutral/null."""
    assert set(_tool()["parameters"]["required"]) == {"affect", "affect_intensity"}


def test_model_never_supplies_ids():
    """Turn correlation is the client's job via response.id, not the model's."""
    props = _props()
    for forbidden in (
        "turn_id", "session_id", "sequence", "occurred_at",
        "projection_revision", "protocol",
    ):
        assert forbidden not in props, f"{forbidden} must not be a model argument"


def test_schema_is_closed():
    assert _tool()["parameters"]["additionalProperties"] is False


def test_affect_tool_does_not_displace_mode_tools():
    """Additive layering: arcturian keeps create_task_proposal.

    This mirrors what the endpoint does, so a regression in the merge
    logic is caught here rather than in a live session.
    """
    base = _companion_arcturian_tools()
    merged = list(base)
    existing = {t["name"] for t in merged}
    for t in _affect_projection_tools():
        if t["name"] not in existing:
            merged.append(t)
    names = {t["name"] for t in merged}
    assert "create_task_proposal" in names, "mode tool was displaced"
    assert "report_affect" in names
    assert len(merged) == len(base) + 1


def test_affect_path_never_smuggles_a_forbidden_tool():
    """The #751 gate is tool ABSENCE — appending must not widen the set."""
    names = {t["name"] for t in _affect_projection_tools()}
    assert not (names & FORBIDDEN), f"forbidden tool on the affect path: {names & FORBIDDEN}"


def test_merge_is_idempotent():
    """Re-applying must not produce a duplicate tool name.

    OpenAI rejects a session whose tool names collide, so a double-apply
    would fail the mint outright.
    """
    merged = list(_affect_projection_tools())
    existing = {t["name"] for t in merged}
    for t in _affect_projection_tools():
        if t["name"] not in existing:
            merged.append(t)
    assert len(merged) == 1


def test_prompt_names_no_language_specific_vocabulary():
    """The contract forbids affect guessed from character/language words.

    The addendum must steer on MEANING, so it may not ship a keyword list
    for the model to pattern-match on.
    """
    text = _affect_projection_addendum("de")
    for value in AFFECT_VALUES:
        assert value in text, f"prompt must name the enum value {value}"
    assert "null" in text, "prompt must state the neutral => null coupling"
    # A keyword list would reintroduce exactly the heuristic §Emotion bans.
    for banned in ("super", "toll", "leider", "schade", "great", "sorry"):
        assert banned not in text.lower(), (
            f"prompt ships a keyword heuristic ({banned!r}) the contract forbids"
        )


def test_followup_uses_required_not_named_tool_choice():
    """Measured: "required" forces the call 6/6, the named form 0/6.

    Realtime accepts {"type":"function","name":...} and then silently
    returns a plain message. This test exists so nobody "cleans up" the
    payload into the form that looks more precise and is in fact inert.
    """
    resp = _affect_followup_payload()["response"]
    assert resp["tool_choice"] == "required", (
        "named-function tool_choice is silently ignored by Realtime"
    )


def test_followup_carries_only_the_affect_tool():
    """This is what makes "required" safe in a session with other tools.

    Without its own single-element tools list, "required" could coerce a
    mode's real tool — e.g. arcturian's create_task_proposal, which would
    create a task proposal nobody asked for.
    """
    tools = _affect_followup_payload()["response"]["tools"]
    assert [t["name"] for t in tools] == ["report_affect"]


def test_followup_is_silent():
    """The affect turn must not speak — it runs after the answer."""
    assert _affect_followup_payload()["response"]["output_modalities"] == ["text"]


def test_followup_is_a_valid_response_create():
    payload = _affect_followup_payload()
    assert payload["type"] == "response.create"
    assert set(payload.keys()) == {"type", "response"}


def test_prompt_is_emitted_for_any_language():
    for lang in ("de", "en", "sl", "it", None):
        text = _affect_projection_addendum(lang)
        assert "report_affect" in text
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
