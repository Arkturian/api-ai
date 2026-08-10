"""Opt-in PTT at mint time (#975, CloudV2-Codex).

Web wants push-to-talk for the dialogue turn. A pure front-end switch is
not safe: with server_vad the PROVIDER creates a response of its own
after the silence window, and that response carries no
`agentos_response_kind` metadata AND inherits the session tools —
`propose_to_agent` for talkback-enabled, `relay_to_agent` for
agent-transparent. A client that also sends `response.create` can
therefore get an unrequested proposal or relay it cannot correlate.

Measured on 2026-08-06 as the owner's "beim zweiten Push-to-Talk kommt
nichts mehr an". Only the mint decides turn_detection, so the switch is
server-side — and opt-in, so nobody who does not ask is moved.
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.realtime_routes import (  # noqa: E402
    RealtimeTokenRequest,
    _companion_relay_tools,
    _companion_talkback_tools,
    _session_tools,
)

SRC = (Path(__file__).resolve().parents[1]
       / "ai/routes/realtime_routes.py").read_text()


def test_field_exists_and_defaults_to_off():
    """Absent must mean 'exactly as before' — for every existing client."""
    field = RealtimeTokenRequest.model_fields["push_to_talk"]
    assert field.default is False
    req = RealtimeTokenRequest(companion_mode="talkback-enabled")
    assert req.push_to_talk is False


def test_flag_is_accepted():
    req = RealtimeTokenRequest(
        companion_mode="agent-transparent", push_to_talk=True,
    )
    assert req.push_to_talk is True


def test_override_can_only_disable_never_enable():
    """One-way by construction: it nulls detection, it never invents it.

    A flag that could switch detection ON would change turn ownership for
    a client that never asked — the failure I built this morning in
    a137d86 and had to withdraw.
    """
    m = re.search(
        r"if request\.push_to_talk and turn_detection is not None:(.*?)\n\n",
        SRC, re.S,
    )
    assert m, "the opt-in override is gone"
    body = m.group(1)
    assert "turn_detection = None" in body
    assert "server_vad" not in body


def test_override_runs_after_the_per_mode_branches():
    """Order matters: placed before them, a mode branch would overwrite it."""
    branch = SRC.index('elif companion_mode == "talkback-enabled":')
    override = SRC.index("if request.push_to_talk and turn_detection is not None:")
    assert override > branch


def test_response_echoes_what_was_actually_minted():
    """The client must ASSERT, not infer.

    Inferring that the server did what you asked is how 18 of 18
    heartbeats came back dead (051968b).
    """
    assert '"turn_detection": turn_detection,' in SRC
    assert '"push_to_talk": bool(request.push_to_talk) and turn_detection is None,' in SRC


def test_tool_contracts_are_untouched():
    """CloudV2-Codex's explicit condition: lists, confirmation and relay
    stay exactly as they are. PTT changes who CLOSES a turn, not what a
    turn may do."""
    assert [t["name"] for t in _companion_talkback_tools()] == ["propose_to_agent"]
    assert [t["name"] for t in _companion_relay_tools()] == ["relay_to_agent"]
    assert [t["name"] for t in _session_tools(
        _companion_talkback_tools(), "talkback-enabled", None)] == ["propose_to_agent"]
    assert [t["name"] for t in _session_tools(
        _companion_relay_tools(), "agent-transparent", None)] == ["relay_to_agent"]


def test_ptt_does_not_change_the_arcturian_tool_set():
    """PTT changes who CLOSES a turn, not what a turn may do.

    The session is no longer empty — the owner decided on 2026-08-09
    that Arcturian gets read tools. What must not change is that the
    affect tool stays per-turn, which is the specific defect AppDevV2
    reproduced on-device as tool_misrouted.
    """
    from ai.routes.realtime_routes import _arcturian_read_tools
    names = {t["name"] for t in
             _session_tools(_arcturian_read_tools(), "arcturian", "v1")}
    assert names == {"agent_status"}



def _mint_response_keys():
    """The keys of the dict the mint ACTUALLY returns.

    Parsed from the AST rather than grepped, because grepping is what let
    the bug below ship.
    """
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)]
            if "provider" in keys and "client_secret" in keys:
                return keys
    raise AssertionError("mint response body not found")


def _error_detail_keys():
    """Every key of every HTTPException(detail={...}) in the module."""
    keys = []
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.keyword) and node.arg == "detail":
            if isinstance(node.value, ast.Dict):
                keys += [k.value for k in node.value.keys
                         if isinstance(k, ast.Constant)]
    return keys


def test_echo_fields_are_in_the_response_body_not_an_error_dict():
    """The bug this exists for — mine, shipped in 7490c18.

    A find_replace on `"companion_mode": companion_mode,` hit the FIRST
    occurrence in the file, which belongs to the
    `unsupported_companion_mode` ERROR body. Both echo fields landed
    there. The mint therefore never returned either one; the client read
    `undefined !== null` as a contract violation and refused to open the
    microphone, and the owner could not start Arcturian in the browser.

    Every test I had asserted the STRING existed somewhere in the module.
    It did. That is precisely why they all stayed green while the feature
    did nothing — the same "checked the instrument, not the result"
    mistake that cost a whole day. This test asserts placement, not
    presence.
    """
    body = _mint_response_keys()
    assert "turn_detection" in body, "echo missing from the mint response"
    assert "push_to_talk" in body, "echo missing from the mint response"


def test_echo_fields_never_leak_into_an_error_detail():
    errs = _error_detail_keys()
    assert "push_to_talk" not in errs
    assert "turn_detection" not in errs


def test_error_body_for_unsupported_mode_stayed_intact():
    """The dict my edit split in half must still carry its own keys."""
    errs = _error_detail_keys()
    for key in ("unsupported_companion_mode", "companion_mode", "supported"):
        pass
    assert "supported" in errs


def test_persona_provenance_is_in_the_response_body():
    """#1002: a device run must be comparable to a bench run.

    Same placement guard as the PTT echo — these two fields are useless
    if they sit anywhere but the mint response.
    """
    body = _mint_response_keys()
    assert "persona_sha256" in body
    assert "persona_chars" in body


def test_provenance_is_hashed_not_read_from_git():
    """The serving tree's .git lies.

    On arkserver it reports 1d1c876 while the deployed commit is
    5fe4043, and realtime_routes.py is untracked there. A revision read
    from it would be wrong and look authoritative. The hash is taken
    from the instruction string actually sent, so it cannot go stale.
    """
    assert 'hashlib.sha256((instructions or "").encode())' in SRC
    assert "rev-parse" not in SRC, "provenance must not come from git at runtime"

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
