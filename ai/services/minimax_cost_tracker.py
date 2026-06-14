"""
MiniMax API Cost Tracker
========================

Tracks usage and costs for MiniMax multimodal API calls (image-01,
hailuo video, speech-02 TTS + voice-cloning, music 2.6).

Mirrors ``cost_tracker.py`` (Gemini-tracker) in shape so the operational
patterns are identical:
  - Singleton in-memory state, persisted to ``minimax_usage_<YYYY-MM>.json``
  - Master/client mode for federation-shared counting (arkserver master,
    arkturian client), see ``_post_to_master`` / ``_fetch_master_status``
  - Persistent kill-switch (``minimax_hard_cap_active``) survives restart;
    cleared only by ``clear_hard_cap()`` or month rollover
  - 80/95/100 threshold alerts via Telegram

Why a separate tracker (not just extending cost_tracker.py): MiniMax has
its own monthly budget envelope (Alex' explicit 25 EUR/month) that should
be enforced independently from the Gemini cap. Cleaner to keep usage
files, master/client endpoints and Telegram alerts segregated per
provider — generalising to provider-agnostic is a follow-up once we have
3+ billable providers and the abstraction earns its keep.

Pricing source-of-truth lives in ``MINIMAX_PRICING`` below — update when
platform.minimax.io publishes a tariff change. The PR adding a new model
must update this dict so cost_tracker can attribute the call.
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


# MiniMax pricing (USD per unit). Verify against platform.minimax.io
# before publishing a PR that adds a new model.
MINIMAX_PRICING = {
    # ── Text (per million tokens, input + output) ────────────────────
    # M3 PAYG pricing per Minimax-bot's IACP cee14357 (heute verified):
    # ~$0.20 / 1M input tokens, ~$1.10 / 1M output tokens.
    "minimax-m3":               {"input_per_1m": 0.20, "output_per_1m": 1.10},

    # ── Image (per image) ────────────────────────────────────────────
    "minimax-image-01":         {"per_image": 0.003},

    # ── Video (per second of output) ─────────────────────────────────
    "minimax-hailuo-pro":       {"per_second": 0.08},
    "minimax-hailuo-fast":      {"per_second": 0.02},

    # ── TTS (per 1k characters of input) ─────────────────────────────
    "minimax-speech-02":        {"per_1k_chars": 0.005},

    # ── Voice-Clone (per single clone job, voice-profile creation) ───
    # The cloned voice is reusable indefinitely afterwards at normal
    # TTS pricing — only the clone-job itself is metered here.
    "minimax-voice-clone":      {"per_clone": 0.10},

    # ── Music (per track, Music 2.6 includes Cover-Gen) ──────────────
    "minimax-music-2.6":        {"per_track": 0.30},

    # Default fallback so unknown-model attribution doesn't crash the
    # tracker. Uses image-01 (cheapest unit price) as a conservative
    # estimate. New models should always extend this dict explicitly.
    "default":                  {"per_image": 0.003},
}

# EUR/USD exchange rate (approximate, matches cost_tracker.py)
EUR_USD_RATE = 1.05


class MinimaxCostTracker:
    """Singleton cost tracker for MiniMax multimodal API usage."""

    _instance: Optional["MinimaxCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MinimaxCostTracker":
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

        # 25 EUR/month total cap per Alex' direct ask. Single shared
        # envelope across both hosts via the master/client setup below.
        # Override via MINIMAX_MONTHLY_BUDGET_EUR if Alex tunes later.
        self.monthly_budget_eur = float(
            os.getenv("MINIMAX_MONTHLY_BUDGET_EUR", "25.0")
        )
        self.data_dir = Path(
            os.getenv("MINIMAX_COST_TRACKER_DATA_DIR", "/var/lib/api-ai")
        )
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.block_on_budget_exceeded = (
            os.getenv("MINIMAX_BLOCK_ON_BUDGET_EXCEEDED", "true").lower() == "true"
        )
        self.alert_thresholds = [80, 95, 100]

        # Master / client mode. arkserver master = MINIMAX_COST_TRACKER_MASTER_URL
        # unset; arkturian client = set to the master's https URL.
        # Shared secret is identical to the gemini-side counter
        # (COST_TRACKER_SHARED_SECRET) since the federation trust model
        # is per-owner, not per-provider — one secret, two endpoints.
        self.master_url = os.getenv("MINIMAX_COST_TRACKER_MASTER_URL", "").rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")

        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0

        self._load_data()

        logger.info(
            f"MinimaxCostTracker initialized: budget={self.monthly_budget_eur}EUR, "
            f"mode={'client' if self.master_url else 'master'}, "
            f"block_on_exceeded={self.block_on_budget_exceeded}"
        )

    @property
    def data_file(self) -> Path:
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"minimax_usage_{month_key}.json"

    def _load_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                    self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
                logger.info(
                    f"Loaded minimax usage: {self._usage_data.get('total_cost_eur', 0):.4f} EUR"
                )
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load minimax usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save minimax usage data: {e}")

    def _reset_monthly_data(self) -> None:
        self._usage_data = {
            "month": datetime.now().strftime("%Y-%m"),
            "total_cost_usd": 0.0,
            "total_cost_eur": 0.0,
            "request_count": 0,
            "by_model": {},
            "alerts_sent": [],
            "created_at": datetime.now().isoformat(),
        }
        self._alerts_sent = set()

    # ── Cost calculation per metering unit ────────────────────────────

    def _cost_for_call(self, model: str, **units) -> tuple[float, float]:
        """Compute (usd, eur) for a single API call.

        ``units`` matches the pricing-dict key for the model:
          - per_image  → ``num_images=N``
          - per_second → ``seconds=N``
          - per_1k_chars → ``chars=N`` (rounded up to nearest 1k)
          - per_clone  → ``num_clones=1`` (single-shot)
          - per_track  → ``num_tracks=1``
        """
        pricing = MINIMAX_PRICING.get(model, MINIMAX_PRICING["default"])
        cost_usd = 0.0
        if "input_per_1m" in pricing:
            in_tok = units.get("input_tokens", 0)
            out_tok = units.get("output_tokens", 0)
            cost_usd = (
                (in_tok * pricing["input_per_1m"])
                + (out_tok * pricing["output_per_1m"])
            ) / 1_000_000.0
        elif "per_image" in pricing:
            cost_usd = pricing["per_image"] * units.get("num_images", 1)
        elif "per_second" in pricing:
            cost_usd = pricing["per_second"] * units.get("seconds", 0)
        elif "per_1k_chars" in pricing:
            # Round up: a 200-char call still costs 1k-worth on most APIs
            chars = units.get("chars", 0)
            cost_usd = pricing["per_1k_chars"] * max(1, (chars + 999) // 1000)
        elif "per_clone" in pricing:
            cost_usd = pricing["per_clone"] * units.get("num_clones", 1)
        elif "per_track" in pricing:
            cost_usd = pricing["per_track"] * units.get("num_tracks", 1)
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    # ── Public track methods (one per metering shape) ─────────────────

    def track_text(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self._track(
            "text", model,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

    def track_image(self, model: str, num_images: int = 1) -> None:
        self._track("image", model, num_images=num_images)

    def track_video(self, model: str, seconds: float) -> None:
        self._track("video", model, seconds=seconds)

    def track_tts(self, model: str, chars: int) -> None:
        self._track("tts", model, chars=chars)

    def track_voice_clone(self, model: str = "minimax-voice-clone") -> None:
        self._track("voice_clone", model, num_clones=1)

    def track_music(self, model: str = "minimax-music-2.6", num_tracks: int = 1) -> None:
        self._track("music", model, num_tracks=num_tracks)

    def _track(self, modality: str, model: str, **units) -> None:
        """Internal: dispatch local-vs-master tracking. Client mode posts
        to master with retry-on-failure fallback to local file so a
        master outage doesn't silently lose accounting.
        """
        if self.master_url and self.shared_secret:
            try:
                self._post_to_master(modality=modality, model=model, units=units)
                return
            except Exception as e:
                logger.error(
                    f"minimax_cost_tracker: master post failed ({e}); "
                    f"falling back to local — cap may temporarily lag"
                )
        self._track_local(modality, model, **units)

    def _track_local(self, modality: str, model: str, **units) -> None:
        cost_usd, cost_eur = self._cost_for_call(model, **units)
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
                "modality": modality,
                "request_count": 0,
                "cost_usd": 0.0,
                "cost_eur": 0.0,
            })
            stats["request_count"] += 1
            stats["cost_usd"] += cost_usd
            stats["cost_eur"] += cost_eur
            self._save_data()
            self._check_thresholds()
        logger.info(
            f"MiniMax tracked: {modality}/{model} = {cost_eur:.6f}EUR "
            f"(total: {self._usage_data['total_cost_eur']:.4f}EUR)"
        )

    # ── Block-decision + status ───────────────────────────────────────

    def is_budget_exceeded(self) -> bool:
        return self._usage_data.get("total_cost_eur", 0) >= self.monthly_budget_eur

    def _maybe_reload_from_file(self) -> None:
        """Re-read the on-disk usage file if a sibling uvicorn worker
        wrote it since our last load — see deepseek_cost_tracker for the
        full rationale (multi-worker singleton drift fix)."""
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
            logger.debug(f"minimax_cost_tracker: reload check skipped ({e})")

    def should_block_request(self) -> bool:
        """Three independent block conditions, any of which trips the gate:

        1. **GCP/MiniMax-side hard-cap** flag (persistent, set via
           ``trip_hard_cap()`` — currently invoked manually; future:
           hook MiniMax-platform billing webhook if Minimax exposes one).
        2. **Federation-shared master state** (client mode queries master,
           falls back to local view on master-unreachable).
        3. **Local view** of monthly cap.
        """
        self._maybe_reload_from_file()
        if self._usage_data.get("minimax_hard_cap_active"):
            return True
        if self.master_url and self.shared_secret:
            try:
                status = self._fetch_master_status()
                return bool(status.get("would_block", False))
            except Exception as e:
                logger.warning(
                    f"minimax_cost_tracker: master query failed ({e}); "
                    f"using local view for block-decision"
                )
        return self.block_on_budget_exceeded and self.is_budget_exceeded()

    def get_status(self) -> dict:
        self._maybe_reload_from_file()
        with self._data_lock:
            cost_eur = self._usage_data.get("total_cost_eur", 0)
            return {
                "provider": "minimax",
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
                "by_model": self._usage_data.get("by_model", {}),
                "budget_exceeded": self.is_budget_exceeded(),
                "requests_blocked": self.should_block_request(),
                "minimax_hard_cap_active": bool(
                    self._usage_data.get("minimax_hard_cap_active")
                ),
                "minimax_hard_cap_reason": self._usage_data.get(
                    "minimax_hard_cap_reason", ""
                ),
                "alerts_sent": list(self._alerts_sent),
                "last_updated": self._usage_data.get("last_updated"),
            }

    # ── Hard-cap kill-switch (mirrors GCP-webhook pattern) ────────────

    def trip_hard_cap(self, reason: str = "") -> None:
        with self._data_lock:
            self._usage_data["minimax_hard_cap_active"] = True
            self._usage_data["minimax_hard_cap_reason"] = reason
            self._usage_data["minimax_hard_cap_tripped_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        logger.warning(
            f"minimax_cost_tracker: hard-cap TRIPPED "
            f"({reason or 'no-reason-given'}) — "
            f"all MiniMax-API calls will return 429 until clear_hard_cap()"
        )

    def clear_hard_cap(self) -> dict:
        with self._data_lock:
            was_active = bool(self._usage_data.get("minimax_hard_cap_active"))
            self._usage_data["minimax_hard_cap_active"] = False
            self._usage_data["minimax_hard_cap_reason"] = ""
            self._usage_data["minimax_hard_cap_cleared_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        return {"was_active": was_active, "cleared": True}

    # ── Threshold alerts via Telegram ─────────────────────────────────

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
            f"{emoji} <b>MiniMax Budget {threshold}% erreicht</b>\n\n"
            f"<b>Verbraucht:</b> {cost_eur:.2f} / {self.monthly_budget_eur:.2f} EUR\n"
            f"<b>Requests:</b> {self._usage_data.get('request_count', 0):,}\n"
            f"<b>Status:</b> {'⛔ BLOCKED' if threshold >= 100 else '✅ Active'}"
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
        self, modality: str, model: str, units: dict
    ) -> None:
        url = f"{self.master_url}/internal/minimax-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                url,
                json={
                    "modality": modality,
                    "model": model,
                    "units": units,
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
        url = f"{self.master_url}/internal/minimax-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache = data
        self._master_status_cache_ts = now
        return data


# Singleton instance for easy import
minimax_cost_tracker = MinimaxCostTracker()
