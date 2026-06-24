"""
Realtime Budget Guard
=====================

Per-profile + per-user atomic pre-mint reservation against the
``grant.limits`` carried in the AuthAPI capability JWT (Content-Post
#1215, frozen v1 Auth-Contract). Codex' v1-blocking requirement:

  * Hard-cap, no soft-warning. Pre-mint reservation fail-closed.
  * Daily window in Europe/Vienna (IANA zone, DST-aware), reset at
    local 00:00.
  * Profile-strict: P3 spend never lands in alex-default's totals
    and vice versa.
  * Per-user max_parallel_sessions enforced atomically.
  * Reservation released on mint failure (try/finally).

State lives in ``/var/lib/api-ai/realtime_reservations.json`` and is
serialised through ``fcntl.flock`` so multiple uvicorn workers and
even the federation client-mode tracker can't drift.

Structure on disk (versioned for forward-compat):

  {
    "version": 1,
    "profiles": {
      "alex-default": {
        "day": "2026-06-24",          # Europe/Vienna local date
        "daily_total_eur": 0.0,
        "users": {
          "<user_uuid>": {
            "daily_eur": 0.0,
            "active_sessions": ["vs_..."],
          }
        }
      },
      "p3": { ... }
    }
  }

The daily window rolls over the first time a request lands after
local midnight Europe/Vienna.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# Locale for the daily window. IANA so DST shifts are handled
# automatically — no fixed UTC offset (Codex Final).
LOCAL_TZ = ZoneInfo("Europe/Vienna")

# Storage path. Lives next to the existing realtime cost tracker JSON.
RESERVATIONS_PATH = Path("/var/lib/api-ai/realtime_reservations.json")

# Conservative estimate of the smallest plausible mint cost. We charge
# this against the daily cap at reserve time so we cannot mint a
# session that has zero budget headroom for even the first turn.
# Real usage is metered via ``/realtime/usage`` posts.
MIN_SESSION_RESERVE_EUR = 0.50

# Sessions whose last activity (mint, heartbeat, or usage charge) is
# older than this window are treated as orphaned (browser tab crashed,
# network died, user walked away without an explicit stop) and reaped
# on the next reserve. The default 60 minutes mirrors OpenAI's hard
# Realtime session cap — past that, no real WebRTC can still be
# running anyway.
#
# Once CloudV2 ships the 30 s heartbeat lease (Content-Post #1215,
# Alex' device-switch gap), operators should set
# ``REALTIME_REAP_SECONDS=90`` so a phantom slot from a crashed tab
# frees within ~90 s, not 60 min.
SESSION_REAP_SEC = int(os.environ.get("REALTIME_REAP_SECONDS", str(60 * 60)))


# ── Exceptions ────────────────────────────────────────────────────────


class BudgetGuardError(Exception):
    """Base for cap/reservation failures.

    ``error_code`` becomes the public response code (Codex' closed
    enum on the wire); ``audit_detail`` is logger-only.
    """

    def __init__(self, error_code: str, audit_detail: str = "", status_code: int = 403):
        super().__init__(error_code)
        self.error_code = error_code
        self.audit_detail = audit_detail
        self.status_code = status_code


class DailyBudgetExceeded(BudgetGuardError):
    def __init__(self, profile_id: str, daily_total: float, cap: float):
        super().__init__(
            "daily_budget_exceeded",
            f"profile={profile_id} total_eur={daily_total:.2f} cap={cap:.2f}",
        )


class MaxParallelExceeded(BudgetGuardError):
    def __init__(self, profile_id: str, user_short: str, active: int, cap: int):
        super().__init__(
            "max_parallel_exceeded",
            f"profile={profile_id} user={user_short} active={active} cap={cap}",
        )


# ── Reservation handle (returned to caller, used to release/confirm) ──


@dataclass(frozen=True)
class Reservation:
    profile_id: str
    user_id: str
    voice_session_id: str
    day: str
    reserved_eur: float
    issued_at: float


# ── Storage helpers ───────────────────────────────────────────────────


def _today_str() -> str:
    """Local YYYY-MM-DD in Europe/Vienna — DST-aware."""
    return _now_local().strftime("%Y-%m-%d")


def _now_local():
    import datetime as _dt
    return _dt.datetime.now(LOCAL_TZ)


@contextlib.contextmanager
def _locked_state():
    """Open RESERVATIONS_PATH, take an exclusive flock, yield the dict,
    write it back atomically via tmpfile+replace, release flock.

    Initialises an empty schema if the file does not exist yet.
    Concurrent uvicorn workers / processes will serialise here.
    """
    RESERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Open r+ if exists else w+ to bootstrap.
    mode = "r+" if RESERVATIONS_PATH.exists() else "w+"
    with open(RESERVATIONS_PATH, mode, encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            if raw.strip():
                try:
                    state = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "realtime_reservations.json corrupt — re-init"
                    )
                    state = {"version": 1, "profiles": {}}
            else:
                state = {"version": 1, "profiles": {}}

            yield state

            f.seek(0)
            f.truncate()
            json.dump(state, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _profile_view(state: dict, profile_id: str, today: str) -> dict:
    """Get or initialise the profile's daily slot, rolling the window
    if a new local day has started."""
    profiles = state.setdefault("profiles", {})
    p = profiles.setdefault(profile_id, {
        "day": today,
        "daily_total_eur": 0.0,
        "users": {},
    })
    if p.get("day") != today:
        # Day rolled. Hard-reset the daily counters but keep active
        # sessions intact — a mint that started yesterday and is still
        # running shouldn't be billed against today's cap, but it
        # should still count against today's parallel limit.
        p["day"] = today
        p["daily_total_eur"] = 0.0
        for u in p.get("users", {}).values():
            u["daily_eur"] = 0.0
    return p


def _user_view(profile_view: dict, user_id: str) -> dict:
    users = profile_view.setdefault("users", {})
    return users.setdefault(user_id, {
        "daily_eur": 0.0,
        "active_sessions": [],
        "session_started": {},  # vid -> epoch seconds
    })


def _reap_orphans(uv: dict, now: float) -> None:
    """Drop sessions whose last activity is older than SESSION_REAP_SEC.

    Mutates the user-view in place. Called at reserve-time so a fresh
    mint always starts from an accurate active-session count even if
    the previous session ended without an explicit release."""
    started = uv.setdefault("session_started", {})
    cutoff = now - SESSION_REAP_SEC
    fresh_active = []
    for vid in (uv.get("active_sessions") or []):
        ts = started.get(vid)
        if ts is None:
            # Pre-existing session from before the reap logic. Treat as
            # fresh so we don't immediately drop it.
            started[vid] = now
            fresh_active.append(vid)
            continue
        if ts >= cutoff:
            fresh_active.append(vid)
        # else: orphan, dropped silently
    uv["active_sessions"] = fresh_active
    # Garbage-collect orphaned session_started entries too.
    uv["session_started"] = {
        vid: ts for vid, ts in started.items() if vid in fresh_active
    }


# ── Public API ───────────────────────────────────────────────────────


def reserve_mint(
    profile_id: str,
    user_id: str,
    voice_session_id: str,
    max_parallel_sessions: int,
    daily_budget_eur: float,
) -> Reservation:
    """Reserve a slot for an imminent mint.

    Raises ``MaxParallelExceeded`` or ``DailyBudgetExceeded`` if the
    grant's limits would be violated. Returns a ``Reservation`` handle
    that MUST be passed back to ``release_reservation()`` on mint
    failure or to ``confirm_usage_charge()`` to add real usage to the
    rolling totals.

    Idempotent on ``voice_session_id``: if a reservation for the same
    voice_session_id already exists in the user's active set we return
    a fresh Reservation handle but do NOT double-count the slot. That
    matches the OpenAI 60-min rollover semantics — same voice session
    keeps its slot across the re-mint.
    """
    today = _today_str()
    now = time.time()
    with _locked_state() as state:
        pv = _profile_view(state, profile_id, today)
        uv = _user_view(pv, user_id)
        _reap_orphans(uv, now)

        active = list(uv.get("active_sessions") or [])
        is_remint = voice_session_id in active

        if not is_remint and len(active) >= max_parallel_sessions:
            short = user_id[:8]
            raise MaxParallelExceeded(
                profile_id, short, len(active), max_parallel_sessions,
            )

        # Conservative daily-budget check: must still have room for at
        # least one minimum-cost session beyond what is already booked.
        booked = float(pv.get("daily_total_eur") or 0.0)
        if booked + MIN_SESSION_RESERVE_EUR > daily_budget_eur:
            raise DailyBudgetExceeded(
                profile_id, booked, daily_budget_eur,
            )

        if not is_remint:
            active.append(voice_session_id)
            uv["active_sessions"] = active
        # Refresh activity timestamp so the orphan reaper sees this
        # session as live until the next reserve or release.
        uv.setdefault("session_started", {})[voice_session_id] = now

        reservation = Reservation(
            profile_id=profile_id,
            user_id=user_id,
            voice_session_id=voice_session_id,
            day=today,
            reserved_eur=MIN_SESSION_RESERVE_EUR,
            issued_at=time.time(),
        )
    logger.info(
        "realtime_reserve ok profile=%s user=%s vid=%s active=%d/%d "
        "daily=%.2f/%.2f remint=%s",
        profile_id, user_id[:8], voice_session_id,
        len(active), max_parallel_sessions,
        booked, daily_budget_eur, is_remint,
    )
    return reservation


def release_reservation(reservation: Reservation) -> None:
    """Drop a session from the user's active set. Called when the
    underlying OpenAI mint failed so the slot frees up immediately."""
    with _locked_state() as state:
        pv = state.get("profiles", {}).get(reservation.profile_id)
        if not pv:
            return
        users = pv.get("users", {})
        uv = users.get(reservation.user_id)
        if not uv:
            return
        active = [
            s for s in (uv.get("active_sessions") or [])
            if s != reservation.voice_session_id
        ]
        uv["active_sessions"] = active
        started = uv.get("session_started") or {}
        started.pop(reservation.voice_session_id, None)
        uv["session_started"] = started
    logger.info(
        "realtime_release profile=%s user=%s vid=%s",
        reservation.profile_id, reservation.user_id[:8],
        reservation.voice_session_id,
    )


def confirm_usage_charge(
    profile_id: str,
    user_id: str,
    voice_session_id: str,
    cost_eur: float,
) -> dict:
    """Add a real usage cost (post-``/realtime/usage``) to the rolling
    daily total. Returns a status snapshot the caller may surface for
    debugging. No exceptions — over-cap is only enforced at reserve
    time per Codex' v1 ruling (audio in flight cannot be hard-cut).
    """
    today = _today_str()
    with _locked_state() as state:
        pv = _profile_view(state, profile_id, today)
        uv = _user_view(pv, user_id)
        pv["daily_total_eur"] = float(pv.get("daily_total_eur") or 0.0) + cost_eur
        uv["daily_eur"] = float(uv.get("daily_eur") or 0.0) + cost_eur
        snapshot = {
            "profile_id": profile_id,
            "day": today,
            "daily_total_eur": pv["daily_total_eur"],
            "user_daily_eur": uv["daily_eur"],
            "active_sessions": list(uv.get("active_sessions") or []),
        }
    logger.info(
        "realtime_charge profile=%s user=%s vid=%s eur=%.4f day_total=%.2f",
        profile_id, user_id[:8], voice_session_id,
        cost_eur, snapshot["daily_total_eur"],
    )
    return snapshot


def session_ended(reservation: Reservation) -> None:
    """Free the parallel slot at end-of-session. Same effect as
    ``release_reservation`` but semantically labelled — the session
    succeeded and we're closing it cleanly."""
    release_reservation(reservation)


def refresh_lease(
    profile_id: str,
    user_id: str,
    voice_session_id: str,
) -> bool:
    """Heartbeat: reset the session's last-activity timestamp to now.

    Returns True if the voice_session_id is in the owner's active set
    and was refreshed; False if it wasn't found (already reaped,
    released, or belongs to another user). Idempotent and cheap; the
    FE pings this every ~30 s so the orphan-reap window (typically
    ~90 s once REALTIME_REAP_SECONDS=90 is set) catches a crashed tab
    inside 1-2 heartbeat cycles.
    """
    now = time.time()
    with _locked_state() as state:
        pv = state.get("profiles", {}).get(profile_id)
        if not pv:
            return False
        uv = pv.get("users", {}).get(user_id)
        if not uv:
            return False
        active = uv.get("active_sessions") or []
        if voice_session_id not in active:
            return False
        started = uv.setdefault("session_started", {})
        started[voice_session_id] = now
    logger.debug(
        "realtime_heartbeat profile=%s user=%s vid=%s",
        profile_id, user_id[:8], voice_session_id,
    )
    return True


def release_by_voice_session(
    profile_id: str,
    user_id: str,
    voice_session_id: str,
) -> bool:
    """Owner-scoped release by voice_session_id (CloudV2 explicit Stop).

    Drops the session from the owner's active set so a parallel mint
    from another device can proceed immediately, rather than waiting
    for the 60-minute orphan reaper.

    Returns True if the session was found and released, False if it
    wasn't in the owner's active set (idempotent). Other users'
    sessions are never touched — caller's grant.sub + profile_id pin
    the scope structurally.
    """
    with _locked_state() as state:
        pv = state.get("profiles", {}).get(profile_id)
        if not pv:
            return False
        uv = pv.get("users", {}).get(user_id)
        if not uv:
            return False
        active = uv.get("active_sessions") or []
        if voice_session_id not in active:
            return False
        uv["active_sessions"] = [s for s in active if s != voice_session_id]
        started = uv.get("session_started") or {}
        started.pop(voice_session_id, None)
        uv["session_started"] = started
    logger.info(
        "realtime_release_explicit profile=%s user=%s vid=%s",
        profile_id, user_id[:8], voice_session_id,
    )
    return True
