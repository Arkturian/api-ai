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


# Locale for the budget window. IANA so DST shifts are handled
# automatically — no fixed UTC offset (Codex Final).
LOCAL_TZ = ZoneInfo("Europe/Vienna")

# Budget-window granularity. Default is daily (the original
# Codex-frozen v1 semantic). Operators set
# ``REALTIME_BUDGET_WINDOW=monthly`` once an owner clarifies that
# their grant.limits.daily_budget_eur is actually intended as a
# monthly cap (Content-Post #1215 follow-up, Alex' 30 EUR/month).
# In monthly mode the state file groups by YYYY-MM and resets on
# the first of the month local Europe/Vienna.
BUDGET_WINDOW = os.environ.get("REALTIME_BUDGET_WINDOW", "daily").lower()
if BUDGET_WINDOW not in {"daily", "monthly"}:
    BUDGET_WINDOW = "daily"

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

    def __init__(self, error_code: str, audit_detail: str = "", status_code: int = 403,
                 public_fields: dict | None = None):
        super().__init__(error_code)
        self.error_code = error_code
        self.audit_detail = audit_detail
        self.status_code = status_code
        # Felder, die der Client SEHEN darf. `audit_detail` bleibt
        # protokoll-intern; diese hier gehen auf die Leitung.
        self.public_fields = public_fields or {}


class BudgetExceeded(BudgetGuardError):
    """Fenster-Grenze erreicht.

    Der Code heisst `budget_exceeded` und nennt das Fenster NICHT
    (Issue #1314). Das ist AppDevs Empfehlung und die bessere: Das
    Fenster steht im Feld `window`, und zwei Stellen fuer dieselbe
    Aussage driften irgendwann gegeneinander. Ein `monthly_...`/
    `daily_...` im Namen waere eine zweite Wahrheitsquelle, die beim
    naechsten Umschalten wieder haengenbleibt — genau so ist der alte
    Name entstanden.

    Vorgeschichte: Er hiess `daily_budget_exceeded`, unabhaengig davon,
    worauf `REALTIME_BUDGET_WINDOW` stand. Beide Clients uebersetzten
    ihn woertlich, und Alexanders Stimme sagte fuenf Tage lang
    „morgen wieder verfuegbar", waehrend das Fenster erst am Monats-
    ersten aufging.

    Was ein Client lesen soll, in dieser Reihenfolge: `window` fuer das
    Fenster, `resets_at` fuer den Zeitpunkt, `used_eur`/`limit_eur` fuer
    die Zahlen. Der Code sagt NUR, DASS das Budget erschoepft ist.
    """

    def __init__(self, profile_id: str, daily_total: float, cap: float,
                 window: Optional[str] = None):
        # `window` ueberschreibt das Hauptfenster, wenn die MONATSgrenze
        # gerissen hat — sonst stuende dort "daily", waehrend der
        # Zurueckstellzeitpunkt der Monatserste ist. Genau diese
        # Verwechslung war #1314.
        fenster = window or BUDGET_WINDOW
        super().__init__(
            "budget_exceeded",
            f"profile={profile_id} total_eur={daily_total:.2f} cap={cap:.2f} "
            f"window={fenster}",
            public_fields={
                "window": fenster,
                "used_eur": round(daily_total, 2),
                "limit_eur": round(cap, 2),
                "resets_at": _window_reset_at(fenster),
            },
        )


# Alter Name als Alias. Nicht aus Bequemlichkeit: Wer `except
# DailyBudgetExceeded` schreibt, soll weiter fangen, was er meint —
# ein umbenanntes Symbol, das still nicht mehr faengt, waere ein
# Ausfall an genau der Stelle, die Geld begrenzt.
DailyBudgetExceeded = BudgetExceeded


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


def _window_reset_at(window: Optional[str] = None) -> str:
    """Wann sich das aktuelle Budgetfenster oeffnet — als ISO-Zeitpunkt.

    Anlass (2026-08-18): Alexanders Stimme startete fuenf Tage lang
    nicht, und die Meldung sagte „Tageslimit erreicht — morgen wieder
    verfuegbar". Das Fenster stand aber auf `monthly`; „morgen" kam nie.
    Der Fehlercode hiess bis 2026-08-25 `daily_budget_exceeded`, egal wie
    das Fenster konfiguriert war — der Kommentar an `_today_str` sagt
    bis heute ausdruecklich „name kept for compat". Seit #1314 heisst er
    `budget_exceeded` und behauptet ueber das Fenster gar nichts mehr.

    **Eine Beschriftung, die eine Umstellung nicht mitgemacht hat,
    altert lautlos.** Deshalb nennt der Fehler ab jetzt das echte
    Fenster UND den Zeitpunkt, statt einen Namen zu tragen, der einmal
    gestimmt hat.
    """
    import datetime as _dt   # lokal, wie in `_now_local` — kein Modul-Import
    jetzt = _now_local()
    if (window or BUDGET_WINDOW) == "monthly":
        if jetzt.month == 12:
            naechstes = jetzt.replace(year=jetzt.year + 1, month=1, day=1,
                                      hour=0, minute=0, second=0, microsecond=0)
        else:
            naechstes = jetzt.replace(month=jetzt.month + 1, day=1,
                                      hour=0, minute=0, second=0, microsecond=0)
    else:
        naechstes = (jetzt + _dt.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
    return naechstes.isoformat()


SESSION_BUDGET_ENV = "REALTIME_SESSION_BUDGET_EUR"


def _session_budget_eur(profile_id: Optional[str] = None) -> float:
    """Sitzungsdeckel — je Profil, sonst global.

    `REALTIME_SESSION_BUDGET_EUR__<profil>` schlaegt
    `REALTIME_SESSION_BUDGET_EUR`. Punkte und Bindestriche im Profilnamen
    werden zu Unterstrichen, weil Umgebungsvariablen sie nicht tragen:
    `product-finder` -> `REALTIME_SESSION_BUDGET_EUR__PRODUCT_FINDER`.

    Warum hier und nicht im Grant: Tages- und Monatsgrenze kommen von
    AuthApi, dieser Wert nicht — AuthApi fuehrt ihn heute nicht. Das ist
    bewusst als Zwischenstand vermerkt und gehoert spaeter zu den
    uebrigen Limits, damit nicht zwei Stellen dasselbe fuehren.
    """
    if profile_id:
        suffix = profile_id.strip().replace("-", "_").replace(".", "_").upper()
        roh = os.environ.get(f"{SESSION_BUDGET_ENV}__{suffix}", "").strip()
        if roh:
            try:
                wert = float(roh)
                return wert if wert > 0 else 0.0
            except ValueError:
                logger.warning(
                    "%s__%s=%r ist keine Zahl — falle auf den globalen "
                    "Wert zurueck", SESSION_BUDGET_ENV, suffix, roh,
                )
    return _session_budget_global()


def _session_budget_global() -> float:
    """Obergrenze je EINZELNER Sprachsitzung. 0/unbesetzt = keine.

    Der Monatstopf merkt erst nach dem Schaden, dass eine Sitzung teuer
    war — CloudV2s Punkt 3 in #1283. Die Zahl selbst ist Alex';
    solange sie fehlt, wird nur GEZAEHLT und nichts abgeschnitten.
    Zaehlen ohne Grenze ist die Vorbedingung, nicht der halbe Weg: Man
    kann nicht deckeln, was man nicht misst.
    """
    roh = os.environ.get(SESSION_BUDGET_ENV, "").strip()
    if not roh:
        return 0.0
    try:
        wert = float(roh)
    except ValueError:
        logger.warning(
            "%s=%r ist keine Zahl — Sitzungsgrenze bleibt AUS",
            SESSION_BUDGET_ENV, roh,
        )
        return 0.0
    return wert if wert > 0 else 0.0


def _today_str() -> str:
    """Local window key in Europe/Vienna — DST-aware.

    YYYY-MM-DD when REALTIME_BUDGET_WINDOW=daily, YYYY-MM when
    monthly. The name kept as ``_today_str`` for compat; the value
    is whatever the operator-configured budget window resets on.
    """
    if BUDGET_WINDOW == "monthly":
        return _now_local().strftime("%Y-%m")
    return _now_local().strftime("%Y-%m-%d")


def _month_str() -> str:
    """Monatsschluessel — unabhaengig vom konfigurierten Hauptfenster.

    Das ZWEITE Fenster (#4831, Vertrag vom 2026-08-26). Es laeuft
    zusaetzlich zum Hauptfenster und wird nur fuer Profile gefuehrt,
    deren Grant ein `monthly_budget_eur` traegt. Alle anderen behalten
    exakt das bisherige Verhalten — `None` heisst kein Monatsfenster,
    nicht "Monatsfenster mit Null".
    """
    return _now_local().strftime("%Y-%m")


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
    # Monatstopf: eigener Schluessel, eigenes Zuruecksetzen. Er lebt
    # neben dem Hauptfenster, nicht statt seiner — sonst wuerde eine
    # Umstellung des Hauptfensters stillschweigend den Monatsstand
    # loeschen.
    monat = _month_str()
    if p.get("month") != monat:
        p["month"] = monat
        p["monthly_total_eur"] = 0.0
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
    monthly_budget_eur: Optional[float] = None,
) -> Reservation:
    """Reserve a slot for an imminent mint.

    Raises ``MaxParallelExceeded`` or ``BudgetExceeded`` if the
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
            raise BudgetExceeded(
                profile_id, booked, daily_budget_eur,
            )

        # Zweites Fenster. `None` heisst KEIN Monatsfenster — das ist
        # der Zustand aller bestehenden Profile und aendert an ihnen
        # nichts. Ein Wert von 0 waere dagegen eine echte Null-Grenze;
        # die beiden duerfen nicht verwechselt werden.
        if monthly_budget_eur is not None:
            monat_gebucht = float(pv.get("monthly_total_eur") or 0.0)
            if monat_gebucht + MIN_SESSION_RESERVE_EUR > monthly_budget_eur:
                raise BudgetExceeded(
                    profile_id, monat_gebucht, monthly_budget_eur,
                    window="monthly",
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
        pv["monthly_total_eur"] = float(pv.get("monthly_total_eur") or 0.0) + cost_eur
        uv["daily_eur"] = float(uv.get("daily_eur") or 0.0) + cost_eur
        # Je Sitzung mitfuehren. Der Schluessel ist die voice_session_id;
        # ohne sie (Altaufrufer) wird nur der Tagestopf gefuehrt, damit
        # kein Sammeleintrag "" entsteht, der mehrere Sitzungen vermengt.
        sitzung_eur = None
        if voice_session_id:
            je_sitzung = uv.setdefault("session_eur", {})
            sitzung_eur = float(je_sitzung.get(voice_session_id) or 0.0) + cost_eur
            je_sitzung[voice_session_id] = sitzung_eur
        snapshot = {
            "profile_id": profile_id,
            "day": today,
            "daily_total_eur": pv["daily_total_eur"],
            "user_daily_eur": uv["daily_eur"],
            "session_eur": sitzung_eur,
            "session_budget_eur": _session_budget_eur(profile_id) or None,
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
        # Sitzungsgrenze — hier und nicht beim Verrechnen: Laufendes
        # Audio laesst sich nicht hart abschneiden (Codex' v1-Regel,
        # siehe confirm_usage_charge). Der Herzschlag ist die naechste
        # Stelle, an der ein Nein ohne Abbruch mitten im Satz wirkt;
        # der Client kennt `False` bereits als "Sitzung ist zu Ende".
        grenze = _session_budget_eur(profile_id)
        if grenze > 0:
            verbraucht = float(
                (uv.get("session_eur") or {}).get(voice_session_id) or 0.0
            )
            if verbraucht >= grenze:
                logger.warning(
                    "realtime_session_cap profile=%s user=%s vid=%s "
                    "eur=%.2f >= %.2f — Herzschlag verweigert",
                    profile_id, user_id[:8], voice_session_id,
                    verbraucht, grenze,
                )
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


def active_sessions_for(profile_id: str, user_id: str) -> list:
    """The voice_session_ids currently holding a slot for this owner.

    Read-only, for diagnostics. Added because a heartbeat miss used to
    be an unexplained False: knowing WHICH keys we hold turns "alive:
    false" from a dead end into a one-line diagnosis (AppDevV2's 18/18
    misses were a client heartbeating vs_<uuid> against a slot booked
    under an agent name).
    """
    with _locked_state() as state:
        pv = state.get("profiles", {}).get(profile_id) or {}
        uv = (pv.get("users") or {}).get(user_id) or {}
        return list(uv.get("active_sessions") or [])
