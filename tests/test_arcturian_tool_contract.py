"""Capability gates for the `arcturian` companion mode (issue #751).

The point of these tests is NOT that the happy path works — it is that the
two forbidden tools cannot appear, and that the authority sets cannot
degrade into a boolean. Both failure modes would be silent in production:
a session with `relay_to_agent` present looks perfectly healthy until the
model uses it, and `external_effects: false` passes JSON parsing while
hollowing out cloud-api's `requested ⊆ confirmed` authority check.

Contract source: Cloud-Codex, issue #751 comment 1147, against cloud-api
dev@d3892e1 and fixture backend/tests/fixtures/arcturian_task_v1.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.realtime_routes import (  # noqa: E402
    SUPPORTED_COMPANION_MODES,
    _companion_arcturian_tools,
    _companion_relay_tools,
    _companion_talkback_tools,
)

FORBIDDEN = {"relay_to_agent", "propose_to_agent"}


def _tool_names(defs):
    return {t["name"] for t in defs}


def test_arcturian_mode_is_registered():
    # Before #751 the mint answered HTTP 400 unsupported_companion_mode.
    assert "arcturian" in SUPPORTED_COMPANION_MODES


def test_arcturian_exposes_exactly_one_tool():
    tools = _companion_arcturian_tools()
    assert len(tools) == 1, f"expected exactly one tool, got {_tool_names(tools)}"
    assert _tool_names(tools) == {"create_task_proposal"}


def test_arcturian_never_exposes_relay_or_propose():
    """The capability gate: forbidden tools are absent, not just discouraged."""
    names = _tool_names(_companion_arcturian_tools())
    assert not (names & FORBIDDEN), f"forbidden tool leaked into arcturian: {names & FORBIDDEN}"


def test_external_effects_is_an_array_not_a_boolean():
    """A boolean would pass JSON but empty out cloud-api's authority set."""
    params = _companion_arcturian_tools()[0]["parameters"]
    ext = params["properties"]["authority"]["properties"]["external_effects"]
    assert ext["type"] == "array", f"external_effects must be an array, got {ext['type']}"
    assert ext["items"]["type"] == "string"


def test_authority_sets_are_all_arrays_of_strings():
    authority = _companion_arcturian_tools()[0]["parameters"]["properties"]["authority"]
    for field in ("targets", "systems", "data", "external_effects"):
        spec = authority["properties"][field]
        assert spec["type"] == "array", f"{field} must be an array"
        assert spec["items"]["type"] == "string", f"{field} must hold strings"
    # The promotion flag is the one genuinely boolean member.
    assert authority["properties"]["allow_mode_promotion"]["type"] == "boolean"


def test_model_never_supplies_ids_or_principal():
    """IDs/principal/tenant are adapter + caller metadata, not model args."""
    props = _companion_arcturian_tools()[0]["parameters"]["properties"]
    for forbidden_arg in (
        "task_id", "message_id", "correlation_id", "revision",
        "principal_user_id", "tenant_id",
    ):
        assert forbidden_arg not in props, f"{forbidden_arg} must not be a model argument"


def test_mode_enum_matches_the_frozen_wire():
    mode = _companion_arcturian_tools()[0]["parameters"]["properties"]["mode"]
    assert set(mode["enum"]) == {"direct_iacp", "content_collab", "swfme"}


def test_target_is_nullable_for_non_direct_modes():
    target = _companion_arcturian_tools()[0]["parameters"]["properties"]["target"]
    assert "null" in target["type"], "target must be nullable for content_collab/swfme"


def test_other_modes_keep_their_tools():
    """Regression guard: adding arcturian must not disturb existing modes."""
    assert _tool_names(_companion_relay_tools()) == {"relay_to_agent"}
    assert "propose_to_agent" in _tool_names(_companion_talkback_tools())


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
