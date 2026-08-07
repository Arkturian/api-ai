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


def test_arcturian_stays_toolless_regardless_of_ptt():
    """PTT must not reopen the empty session (d4c924e)."""
    assert _session_tools([], "arcturian", "v1") == []


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
