"""Operator endpoints must not be public (found 2026-08-07, AppDevV2).

He called `/ai/realtime/cost-status` with a sandbox-account token and got
back the OWNER's numbers: 9.71 EUR spent of a 35 EUR pot, 489 requests,
token counts. His own pot was empty.

Verifying it turned out worse than reported: the endpoint never looked at
his token at all. `get_api_key()` returns the literal string
"placeholder" and checks nothing, so every endpoint "requiring" it was
open — including `POST /realtime/cost-status/reset-hard-cap`, the brake
against runaway spend, which answered 200 to an unauthenticated call
from the public internet.

The mint was never affected: it carries require_realtime_grant("mint")
on top, which does real JWKS-pinned verification.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException  # noqa: E402

from ai.routes.realtime_routes import require_operator_key  # noqa: E402

SECRET = "correct-horse-battery-staple"


def _with_secret(value):
    os.environ["REALTIME_GRANT_SERVICE_KEY"] = value


def test_missing_key_is_refused():
    _with_secret(SECRET)
    try:
        require_operator_key(None)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert exc.detail["error"] == "operator_key_required"
    else:
        raise AssertionError("an anonymous caller must not pass")


def test_wrong_key_is_refused():
    _with_secret(SECRET)
    try:
        require_operator_key("nope")
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("a wrong key must not pass")


def test_correct_key_passes():
    _with_secret(SECRET)
    assert require_operator_key(SECRET) == SECRET


def test_unset_secret_fails_closed_not_open():
    """An unset secret must never mean 'everyone welcome'.

    That is the exact shape of the placeholder this replaces: a check
    that looks present and permits everything.
    """
    os.environ.pop("REALTIME_GRANT_SERVICE_KEY", None)
    try:
        require_operator_key(SECRET)
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["error"] == "operator_key_not_configured"
    else:
        raise AssertionError("unset secret must fail closed")
    finally:
        _with_secret(SECRET)


def test_comparison_is_constant_time():
    """A shared secret compared with == leaks its prefix over time."""
    import inspect
    src = inspect.getsource(require_operator_key)
    assert "hmac.compare_digest" in src
    assert "x_api_key == expected" not in src


def test_the_placeholder_is_still_a_placeholder():
    """Guard: nobody should 'fix' get_api_key into looking real.

    It is used by other endpoints too; making it half-real would hide
    which routes are actually protected. The operator gate is separate
    ON PURPOSE.
    """
    from ai.routes.realtime_routes import get_api_key
    assert get_api_key() == "placeholder"


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
