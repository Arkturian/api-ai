"""
OpenAI Realtime Cost Tracker
============================

Tracks usage and costs for OpenAI Realtime API calls (gpt-realtime,
gpt-4o-mini-realtime) over WebRTC/WebSocket.

Mirrors ``deepseek_cost_tracker.py`` + ``openai_cost_tracker.py`` in
shape so the operational patterns are identical:
  - Singleton in-memory state, persisted to
    ``openai_realtime_usage_<YYYY-MM>.json``
  - Master/client mode for federation-shared counting
  - Reuses ``COST_TRACKER_SHARED_SECRET`` env (one secret, five trackers
    now — Gemini, MiniMax, OpenAI Images, DeepSeek, OpenAI Realtime;
    each its own cap envelope)
  - Persistent hard-cap kill-switch survives restart
  - mtime-aware reload to dodge multi-worker singleton drift

Federation discussion (Content-Post #1196, 2026-06-22) consensus:
separate 100 EUR/month cap (Alex's go) on top of the existing four
caps. Realtime is ~6x more expensive per minute than text, and audio
billing is opaque to humans — a hard envelope is essential.

Pricing model: OpenAI bills audio + text tokens separately, with
distinct input/output prices. We track each modality so the by_model
breakdown remains accurate for monthly review even when usage shifts
between voice and text-only.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# OpenAI Realtime pricing — USD per 1M tokens. Source: openai.com/pricing.
# Realtime models bill audio and text tokens separately; we record both
# so a Knowledge-style live narration session (mostly audio) is priced
# differently from a Tool-heavy session (mostly text from function calls).
OPENAI_REALTIME_PRICING = {
    "gpt-realtime": {
        "audio_input_per_1m":  32.0,
        "audio_output_per_1m": 64.0,
        "text_input_per_1m":    5.0,
        "text_output_per_1m":  20.0,
    },
    "gpt-4o-realtime-preview": {
        "audio_input_per_1m":  40.0,
        "audio_output_per_1m": 80.0,
        "text_input_per_1m":    5.0,
        "text_output_per_1m":  20.0,
    },
    "gpt-4o-mini-realtime-preview": {
        "audio_input_per_1m":  10.0,
        "audio_output_per_1m": 20.0,
        "text_input_per_1m":  0.60,
        "text_output_per_1m": 2.40,
    },
    # Default fallback — bill at full gpt-realtime rates so we never
    # silently under-count an unknown model.
    "default": {
        "audio_input_per_1m":  32.0,
        "audio_output_per_1m": 64.0,
        "text_input_per_1m":    5.0,
        "text_output_per_1m":  20.0,
    },
}

EUR_USD_RATE = 1.05


class OpenAIRealtimeCostTracker:
    """Singleton cost tracker for OpenAI Realtime API usage."""

    _instance: Optional["OpenAIRealtimeCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OpenAIRealtimeCostTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        self.monthly_budget_eur = float(
            os.getenv("OPENAI_REALTIME_MONTHLY_BUDGET_EUR", "100.0")
        )
        self.data_dir = Path(
            os.getenv("OPENAI_REALTIME_COST_TRACKER_DATA_DIR", "/var/lib/api-ai")
        )
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.block_on_budget_exceeded = (
            os.getenv("OPENAI_REALTIME_BLOCK_ON_BUDGET_EXCEEDED", "true").lower()
            == "true"
        )
        self.alert_thresholds = [80, 95, 100]

        self.master_url = os.getenv(
            "OPENAI_REALTIME_COST_TRACKER_MASTER_URL", ""
        ).rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")

        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0

        self._load_data()

        logger.info(
            f"OpenAIRealtimeCostTracker initialized: "
            f"budget={self.monthly_budget_eur}EUR, "
            f"mode={'client' if self.master_url else 'master'}, "
            f"block_on_exceeded={self.block_on_budget_exceeded}"
        )

    @property
    def data_file(self) -> Path:
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"openai_realtime_usage_{month_key}.json"

    def _load_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                    self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
                logger.info(
                    f"Loaded openai-realtime usage: "
                    f"{self._usage_data.get('total_cost_eur', 0):.4f} EUR"
                )
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load openai-realtime usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save openai-realtime usage data: {e}")

    def _reset_monthly_data(self) -> None:
        self._usage_data = {
            "month": datetime.now().strftime("%Y-%m"),
            "total_cost_usd": 0.0,
            "total_cost_eur": 0.0,
            "request_count": 0,
            "session_count": 0,
            "by_model": {},
            "alerts_sent": [],
            "created_at": datetime.now().isoformat(),
        }
        self._alerts_sent = set()

    # ── Cost calculation ──────────────────────────────────────────────

    def _cost_for_session(
        self,
        model: str,
        audio_input_tokens: int,
        audio_output_tokens: int,
        text_input_tokens: int,
        text_output_tokens: int,
    ) -> tuple[float, float]:
        pricing = OPENAI_REALTIME_PRICING.get(
            model, OPENAI_REALTIME_PRICING["default"]
        )
        cost_usd = (
            audio_input_tokens  * pricing["audio_input_per_1m"]
            + audio_output_tokens * pricing["audio_output_per_1m"]
            + text_input_tokens   * pricing["text_input_per_1m"]
            + text_output_tokens  * pricing["text_output_per_1m"]
        ) / 1_000_000.0
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    def track_session(
        self,
        model: str,
        audio_input_tokens: int = 0,
        audio_output_tokens: int = 0,
        text_input_tokens: int = 0,
        text_output_tokens: int = 0,
        duration_sec: float = 0.0,
        voice_session_id: Optional[str] = None,
        usage_event_id: Optional[str] = None,
    ) -> dict:
        """Track a Realtime session's per-turn usage delta.

        OpenAI Realtime emits ``response.done`` events with a ``usage``
        block carrying ``input_tokens``, ``output_tokens`` and a
        ``details.audio_tokens`` / ``details.text_tokens`` split. The
        caller is expected to extract these and call us with all four
        numbers; we attribute to the bill correctly.

        Idempotency (Codex IACP, Content-Post #1215): when both
        ``voice_session_id`` and ``usage_event_id`` are passed, we dedup
        on that key before adding. Retries / sendBeacon flushes / final
        usage-pumps cannot double-count.

        Returns ``{deduped: bool, accepted: bool}``. ``deduped=true``
        means the (voice_session_id, usage_event_id) pair was already
        accounted for and no further work happened. ``accepted=true``
        means it was tracked (either locally or forwarded to the
        federation master).
        """
        if (
            audio_input_tokens <= 0
            and audio_output_tokens <= 0
            and text_input_tokens <= 0
            and text_output_tokens <= 0
        ):
            return {"deduped": False, "accepted": False}

        # Idempotency check — only applies when both ids are present.
        if voice_session_id and usage_event_id:
            dedup_key = f"{voice_session_id}::{usage_event_id}"
            # Reload from disk first so sibling-worker dedup state is
            # visible — critical for the multi-worker uvicorn setup
            # where successive retries can land on different workers.
            self._maybe_reload_from_file()
            with self._data_lock:
                seen = set(self._usage_data.get("seen_event_ids", []))
                if dedup_key in seen:
                    logger.info(
                        f"openai_realtime_cost_tracker: dedup hit for "
                        f"{dedup_key[:80]} — skipping double-count"
                    )
                    return {"deduped": True, "accepted": False}
                # Mark seen BEFORE the actual track so concurrent retries
                # in two workers race on the lock + only one wins. Save
                # immediately so a sibling worker sees the mark even on
                # the master-forwarding path (where _track_local that
                # would normally save is skipped).
                seen.add(dedup_key)
                self._usage_data["seen_event_ids"] = list(seen)[-5000:]
                self._save_data()
                # Keep the dedup-set bounded — 5000 keys is roughly
                # 20-30 sessions of turn-records, plenty for a month.
        else:
            logger.warning(
                "openai_realtime_cost_tracker: track_session called "
                "without (voice_session_id, usage_event_id) — non-idempotent "
                "path, retries may double-count (legacy caller?)"
            )

        if self.master_url and self.shared_secret:
            try:
                self._post_to_master(
                    model=model,
                    audio_input_tokens=audio_input_tokens,
                    audio_output_tokens=audio_output_tokens,
                    text_input_tokens=text_input_tokens,
                    text_output_tokens=text_output_tokens,
                    duration_sec=duration_sec,
                    voice_session_id=voice_session_id,
                    usage_event_id=usage_event_id,
                )
                return {"deduped": False, "accepted": True}
            except Exception as e:
                logger.error(
                    f"openai_realtime_cost_tracker: master post failed "
                    f"({e}); falling back to local — cap may temporarily lag"
                )
        self._track_local(
            model,
            audio_input_tokens,
            audio_output_tokens,
            text_input_tokens,
            text_output_tokens,
            duration_sec,
        )
        return {"deduped": False, "accepted": True}

    def _track_local(
        self,
        model: str,
        audio_input_tokens: int,
        audio_output_tokens: int,
        text_input_tokens: int,
        text_output_tokens: int,
        duration_sec: float,
    ) -> None:
        cost_usd, cost_eur = self._cost_for_session(
            model,
            audio_input_tokens,
            audio_output_tokens,
            text_input_tokens,
            text_output_tokens,
        )
        if cost_usd <= 0:
            return
        with self._data_lock:
            current_month = datetime.now().strftime("%Y-%m")
            if self._usage_data.get("month") != current_month:
                self._reset_monthly_data()
            self._usage_data["total_cost_usd"] += cost_usd
            self._usage_data["total_cost_eur"] += cost_eur
            self._usage_data["request_count"] += 1
            by_model = self._usage_data.setdefault("by_model", {})
            stats = by_model.setdefault(model, {
                "modality": "realtime",
                "request_count": 0,
                "audio_input_tokens": 0,
                "audio_output_tokens": 0,
                "text_input_tokens": 0,
                "text_output_tokens": 0,
                "duration_sec": 0.0,
                "cost_usd": 0.0,
                "cost_eur": 0.0,
            })
            stats["request_count"] += 1
            stats["audio_input_tokens"] += audio_input_tokens
            stats["audio_output_tokens"] += audio_output_tokens
            stats["text_input_tokens"] += text_input_tokens
            stats["text_output_tokens"] += text_output_tokens
            stats["duration_sec"] += duration_sec
            stats["cost_usd"] += cost_usd
            stats["cost_eur"] += cost_eur
            self._save_data()
            self._check_thresholds()
        logger.info(
            f"OpenAI Realtime tracked: {model} audio_in={audio_input_tokens} "
            f"audio_out={audio_output_tokens} text_in={text_input_tokens} "
            f"text_out={text_output_tokens} dur={duration_sec:.1f}s "
            f"= {cost_eur:.6f}EUR "
            f"(total: {self._usage_data['total_cost_eur']:.4f}EUR)"
        )

    def track_session_start(self) -> None:
        """Count an opened session, even if no usage rolls in yet."""
        with self._data_lock:
            self._usage_data.setdefault("session_count", 0)
            self._usage_data["session_count"] += 1
            self._save_data()

    # ── Block-decision + status ───────────────────────────────────────

    def is_budget_exceeded(self) -> bool:
        return (
            self._usage_data.get("total_cost_eur", 0) >= self.monthly_budget_eur
        )

    def should_block_request(self) -> bool:
        self._maybe_reload_from_file()
        if self._usage_data.get("openai_realtime_hard_cap_active"):
            return True
        if self.master_url and self.shared_secret:
            try:
                status = self._fetch_master_status()
                return bool(status.get("would_block", False))
            except Exception as e:
                logger.warning(
                    f"openai_realtime_cost_tracker: master query failed "
                    f"({e}); using local view for block-decision"
                )
        return self.block_on_budget_exceeded and self.is_budget_exceeded()

    def _maybe_reload_from_file(self) -> None:
        try:
            f = self.data_file
            if not f.exists():
                return
            mtime = f.stat().st_mtime
            last = float(self._usage_data.get("_file_mtime", 0))
            if mtime > last + 0.001:
                with open(f, "r") as fh:
                    fresh = json.load(fh)
                fresh["_file_mtime"] = mtime
                self._usage_data = fresh
                self._alerts_sent = set(fresh.get("alerts_sent", []))
        except Exception as e:
            logger.debug(
                f"openai_realtime_cost_tracker: reload check skipped ({e})"
            )

    def get_status(self) -> dict:
        self._maybe_reload_from_file()
        with self._data_lock:
            cost_eur = self._usage_data.get("total_cost_eur", 0)
            return {
                "provider": "openai-realtime",
                "month": self._usage_data.get("month"),
                "total_cost_eur": round(cost_eur, 4),
                "total_cost_usd": round(self._usage_data.get("total_cost_usd", 0), 4),
                "monthly_budget_eur": self.monthly_budget_eur,
                "usage_percentage": round(
                    (cost_eur / self.monthly_budget_eur) * 100, 2
                ) if self.monthly_budget_eur > 0 else 0.0,
                "remaining_eur": round(
                    max(0, self.monthly_budget_eur - cost_eur), 4
                ),
                "request_count": self._usage_data.get("request_count", 0),
                "session_count": self._usage_data.get("session_count", 0),
                "by_model": self._usage_data.get("by_model", {}),
                "budget_exceeded": self.is_budget_exceeded(),
                "requests_blocked": self.should_block_request(),
                "openai_realtime_hard_cap_active": bool(
                    self._usage_data.get("openai_realtime_hard_cap_active")
                ),
                "openai_realtime_hard_cap_reason": self._usage_data.get(
                    "openai_realtime_hard_cap_reason", ""
                ),
                "alerts_sent": list(self._alerts_sent),
                "last_updated": self._usage_data.get("last_updated"),
            }

    # ── Hard-cap kill-switch ──────────────────────────────────────────

    def trip_hard_cap(self, reason: str = "") -> None:
        with self._data_lock:
            self._usage_data["openai_realtime_hard_cap_active"] = True
            self._usage_data["openai_realtime_hard_cap_reason"] = reason
            self._usage_data["openai_realtime_hard_cap_tripped_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        logger.warning(
            f"openai_realtime_cost_tracker: hard-cap TRIPPED "
            f"({reason or 'no-reason-given'}) — Realtime token mint 429 "
            f"until clear_hard_cap()"
        )

    def clear_hard_cap(self) -> dict:
        with self._data_lock:
            was_active = bool(
                self._usage_data.get("openai_realtime_hard_cap_active")
            )
            self._usage_data["openai_realtime_hard_cap_active"] = False
            self._usage_data["openai_realtime_hard_cap_reason"] = ""
            self._usage_data["openai_realtime_hard_cap_cleared_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        return {"was_active": was_active, "cleared": True}

    # ── Threshold alerts ──────────────────────────────────────────────

    def _check_thresholds(self) -> None:
        if self.monthly_budget_eur <= 0:
            return
        cost_eur = self._usage_data.get("total_cost_eur", 0)
        pct = (cost_eur / self.monthly_budget_eur) * 100
        for th in self.alert_thresholds:
            if pct >= th and th not in self._alerts_sent:
                self._send_telegram_alert(th, cost_eur)
                self._alerts_sent.add(th)
                self._usage_data["alerts_sent"] = list(self._alerts_sent)

    def _send_telegram_alert(self, threshold: int, cost_eur: float) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        emoji = "🚨" if threshold >= 100 else ("⚠️" if threshold >= 95 else "📊")
        msg = (
            f"{emoji} <b>OpenAI Realtime Budget {threshold}% erreicht</b>\n\n"
            f"<b>Verbraucht:</b> {cost_eur:.2f} / "
            f"{self.monthly_budget_eur:.2f} EUR\n"
            f"<b>Sessions:</b> {self._usage_data.get('session_count', 0):,}\n"
            f"<b>Status:</b> "
            f"{'⛔ BLOCKED' if threshold >= 100 else '✅ Active'}"
        )
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage",
                    json={
                        "chat_id": self.telegram_chat_id,
                        "text": msg.strip(),
                        "parse_mode": "HTML",
                    },
                )
        except Exception as e:
            logger.error(f"Telegram alert send failed: {e}")

    # ── Federation-shared master/client HTTP plumbing ─────────────────

    def _post_to_master(
        self,
        model: str,
        audio_input_tokens: int,
        audio_output_tokens: int,
        text_input_tokens: int,
        text_output_tokens: int,
        duration_sec: float,
        voice_session_id: Optional[str] = None,
        usage_event_id: Optional[str] = None,
    ) -> None:
        url = f"{self.master_url}/internal/openai-realtime-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                url,
                json={
                    "model": model,
                    "audio_input_tokens": audio_input_tokens,
                    "audio_output_tokens": audio_output_tokens,
                    "text_input_tokens": text_input_tokens,
                    "text_output_tokens": text_output_tokens,
                    "duration_sec": duration_sec,
                    "voice_session_id": voice_session_id,
                    "usage_event_id": usage_event_id,
                    "source_host": os.environ.get("API_AI_HOST_KEY")
                    or os.uname().nodename.split(".")[0],
                },
                headers={"X-Internal-Auth": self.shared_secret},
            )
            r.raise_for_status()
            try:
                self._master_status_cache = r.json()
                self._master_status_cache_ts = time.time()
            except Exception:
                pass

    def _fetch_master_status(self) -> dict:
        now = time.time()
        if (
            self._master_status_cache is not None
            and now - self._master_status_cache_ts < 10.0
        ):
            return self._master_status_cache
        url = f"{self.master_url}/internal/openai-realtime-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache = data
        self._master_status_cache_ts = now
        return data


# Singleton instance for easy import
openai_realtime_cost_tracker = OpenAIRealtimeCostTracker()
