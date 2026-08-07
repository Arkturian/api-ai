"""Contract tests for the arcturian conversation preface (Post #4518).

The preface exists because Arcturian claimed a history it never had.
The rules that make that claim true rather than merely quieter:

  * A conversation that cannot be read must NOT cost the owner his
    session. A session without history is degraded; no session is broken.
  * `prefaced_through_revision` is the boundary AppDevV2's live
    projection resumes from. If it were wrong, an answer would be spoken
    twice — instantly obvious in a voice session.
  * The limits do not measure the same thing: 20 turns is what binds,
    4000 chars only catches the outlier.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes import realtime_routes as rt  # noqa: E402


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Client:
    def __init__(self, resp):
        self._resp = resp
        self.seen = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None, headers=None):
        self.seen["url"] = url
        self.seen["params"] = params
        self.seen["headers"] = headers
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _run(resp):
    holder = {}

    def factory(*a, **kw):
        holder["client"] = _Client(resp)
        return holder["client"]

    original = rt.httpx.AsyncClient
    rt.httpx.AsyncClient = factory
    try:
        out = asyncio.run(rt._fetch_conversation_preface("conv_x", "Bearer tok"))
    finally:
        rt.httpx.AsyncClient = original
    return out, holder.get("client")


EVENTS = [
    {"revision": 5, "direction": "inbound", "actor": "Alexander",
     "content": "Schick das bitte an AppDevV2."},
    {"revision": 6, "direction": "outbound", "actor": "Arcturian",
     "content": "Ist unterwegs."},
    {"revision": 7, "direction": "inbound", "actor": "Alexander",
     "content": "Ist die Antwort da?"},
]


def test_limits_are_the_measured_ones():
    assert rt.PREFACE_TAIL_TURNS == 20
    assert rt.PREFACE_MAX_CHARS == 4000


def test_tail_and_cap_are_actually_requested():
    """Without tail= the endpoint paginates FORWARD — Cloud's catch.

    Asking for the first N of a long conversation would preface the
    oldest turns, i.e. exactly the wrong end.
    """
    _, client = _run(_Resp(200, EVENTS))
    assert client.seen["params"]["tail"] == 20
    assert client.seen["params"]["max_chars"] == 4000


def test_caller_jwt_is_forwarded_verbatim():
    """The events belong to the user; there is no service read path."""
    _, client = _run(_Resp(200, EVENTS))
    assert client.seen["headers"]["Authorization"] == "Bearer tok"


def test_direction_decides_the_speaker():
    (items, _), _ = _run(_Resp(200, EVENTS))
    assert [i["role"] for i in items] == ["user", "assistant", "user"]


def test_boundary_is_the_highest_included_revision():
    (_, through), _ = _run(_Resp(200, EVENTS))
    assert through == 7, "live projection would replay an already-spoken turn"


def test_empty_conversation_yields_null_boundary():
    """Null means 'nothing prefaced, project everything' — not revision 0."""
    (items, through), _ = _run(_Resp(200, []))
    assert items == [] and through is None


def test_blank_events_do_not_become_empty_turns():
    (items, through), _ = _run(_Resp(200, [
        {"revision": 1, "direction": "inbound", "content": "   "},
        {"revision": 2, "direction": "outbound", "content": "Da."},
    ]))
    assert len(items) == 1
    assert through == 2


def test_http_error_never_breaks_the_session():
    (items, through), _ = _run(_Resp(403, None, "forbidden"))
    assert items == [] and through is None


def test_transport_error_never_breaks_the_session():
    (items, through), _ = _run(RuntimeError("connection reset"))
    assert items == [] and through is None


def test_malformed_payload_never_breaks_the_session():
    (items, through), _ = _run(_Resp(200, {"unexpected": True}))
    assert items == [] and through is None


def test_missing_revisions_still_produce_items():
    """Content is usable even when the boundary cannot be established."""
    (items, through), _ = _run(_Resp(200, [
        {"direction": "inbound", "content": "ohne revision"},
    ]))
    assert len(items) == 1
    assert through is None


def test_request_model_carries_conversation_id():
    field = rt.RealtimeTokenRequest.model_fields["conversation_id"]
    assert field.default is None, "omitting it must mean an empty chat"



def test_assistant_items_use_output_text_not_text():
    """Issue #959 — this shipped broken and killed live sessions.

    Realtime takes DIFFERENT content types per role, and the assistant one
    is `output_text`. Sending `text` makes the provider reject the item
    with "Invalid value: 'text'. Value must be 'output_text'." and the
    session dies at start.

    The trap: `text` IS the correct value one field over, in
    `response.output_modalities` — same word, two meanings.
    """
    (items, _), _ = _run(_Resp(200, EVENTS))
    by_role = {i["role"]: i["content"][0]["type"] for i in items}
    assert by_role["user"] == "input_text"
    assert by_role["assistant"] == "output_text"


def test_no_preface_item_ever_uses_the_bare_text_type():
    (items, _), _ = _run(_Resp(200, EVENTS))
    for item in items:
        assert item["content"][0]["type"] != "text", (
            "bare 'text' is rejected by the provider for both roles"
        )

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
