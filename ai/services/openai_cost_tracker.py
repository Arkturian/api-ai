"""
OpenAI API Cost Tracker
=======================

Tracks usage and costs for OpenAI Images API calls (gpt-image-2,
gpt-image-1/1.5, dall-e-3) plus future audio/text PAYG paths.

Mirrors ``minimax_cost_tracker.py`` in shape so the operational
patterns are identical across providers:
  - Singleton in-memory state, persisted to ``openai_usage_<YYYY-MM>.json``
  - Master/client mode for federation-shared counting (arkserver master,
    arkturian client) — same ``COST_TRACKER_SHARED_SECRET`` as MiniMax
    + Gemini, federation trust is per-owner not per-provider
  - Persistent hard-cap kill-switch survives restart; cleared only by
    ``clear_hard_cap()`` or month rollover
  - 80/95/100 threshold alerts via Telegram

Pricing source-of-truth lives in ``OPENAI_PRICING`` below — update when
OpenAI publishes a tariff change. The PR adding a new model must update
this dict so the tracker can attribute the call.

Story's IACP b3244014 (2026-06-13) explicitly requested a 50 EUR/month
default cap pattern analog to Gemini (15 EUR) and MiniMax (25 EUR) —
the three caps are independent so OpenAI burn doesn't accidentally
block a MiniMax-image generation and vice versa.
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


# OpenAI pricing — USD per unit. Source: platform.openai.com/docs/pricing
# Image API: priced per generated image (size + quality bands).
# Conservative defaults used here; OpenAI's actual bill rounds per call.
OPENAI_PRICING = {
    # gpt-image-2 (current flagship, photo+illustration)
    "gpt-image-2":      {"per_image": 0.175},  # ~$0.15-0.20 high-q 2048², avg
    "gpt-image-1.5":    {"per_image": 0.10},
    "gpt-image-1":      {"per_image": 0.05},
    "gpt-image-1-mini": {"per_image": 0.02},

    # Legacy DALL-E
    "dall-e-3":         {"per_image": 0.04},
    "dall-e-2":         {"per_image": 0.02},

    # Default fallback — picks the gpt-image-2 average so unknown-model
    # attribution doesn't undercount. New models should always extend
    # this dict explicitly.
    "default":          {"per_image": 0.175},
}

# EUR/USD exchange rate (approximate, matches cost_tracker.py)
EUR_USD_RATE = 1.05


class OpenAICostTracker:
    """Singleton cost tracker for OpenAI Images API usage."""

    _instance: Optional["OpenAICostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OpenAICostTracker":
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

        # 50 EUR/month total cap per Story's IACP b3244014. Independent
        # from MiniMax (25) and Gemini (15) caps — three separate
        # envelopes so a single provider's burn doesn't block the others.
        # Override via OPENAI_MONTHLY_BUDGET_EUR if Alex tunes later.
        self.monthly_budget_eur = float(
            os.getenv("OPENAI_MONTHLY_BUDGET_EUR", "50.0")
        )
        self.data_dir = Path(
            os.getenv("OPENAI_COST_TRACKER_DATA_DIR", "/var/lib/api-ai")
        )
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.block_on_budget_exceeded = (
            os.getenv("OPENAI_BLOCK_ON_BUDGET_EXCEEDED", "true").lower() == "true"
        )
        self.alert_thresholds = [80, 95, 100]

        # Master / client mode — arkserver master, arkturian client.
        # Shared secret identical to the other two cost-tracker counters
        # (one secret, three endpoints) since the federation trust model
        # is per-owner not per-provider.
        self.master_url = os.getenv("OPENAI_COST_TRACKER_MASTER_URL", "").rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")

        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0

        self._load_data()

        logger.info(
            f"OpenAICostTracker initialized: budget={self.monthly_budget_eur}EUR, "
            f"mode={'client' if self.master_url else 'master'}, "
            f"block_on_exceeded={self.block_on_budget_exceeded}"
        )

    @property
    def data_file(self) -> Path:
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"openai_usage_{month_key}.json"

    def _load_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                    self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
                logger.info(
                    f"Loaded openai usage: {self._usage_data.get('total_cost_eur', 0):.4f} EUR"
                )
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load openai usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save openai usage data: {e}")

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

    # ── Cost calculation per call ─────────────────────────────────────

    def _cost_for_call(self, model: str, num_images: int = 1) -> tuple[float, float]:
        pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["default"])
        cost_usd = pricing.get("per_image", 0.0) * num_images
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    # ── Public track method ───────────────────────────────────────────

    def track_image(self, model: str, num_images: int = 1) -> None:
        """Track an OpenAI Image generation. Routes to master if client,
        otherwise tracks locally. See ``_track_local`` for the actual
        accounting logic.
        """
        if num_images <= 0:
            return
        if self.master_url and self.shared_secret:
            try:
                self._post_to_master(model=model, num_images=num_images)
                return
            except Exception as e:
                logger.error(
                    f"openai_cost_tracker: master post failed ({e}); "
                    f"falling back to local — cap may temporarily lag"
                )
        self._track_local(model, num_images)

    def _track_local(self, model: str, num_images: int = 1) -> None:
        cost_usd, cost_eur = self._cost_for_call(model, num_images)
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
                "modality": "image",
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
            f"OpenAI tracked: image/{model} = {cost_eur:.4f}EUR "
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
            logger.debug(f"openai_cost_tracker: reload check skipped ({e})")

    def should_block_request(self) -> bool:
        """Three independent block conditions, any of which trips the gate:
        (1) persistent hard-cap flag, (2) shared master state in client
        mode, (3) local-view cap. Mirrors ``minimax_cost_tracker`` shape.
        """
        self._maybe_reload_from_file()
        if self._usage_data.get("openai_hard_cap_active"):
            return True
        if self.master_url and self.shared_secret:
            try:
                status = self._fetch_master_status()
                return bool(status.get("would_block", False))
            except Exception as e:
                logger.warning(
                    f"openai_cost_tracker: master query failed ({e}); "
                    f"using local view for block-decision"
                )
        return self.block_on_budget_exceeded and self.is_budget_exceeded()

    def get_status(self) -> dict:
        self._maybe_reload_from_file()
        with self._data_lock:
            cost_eur = self._usage_data.get("total_cost_eur", 0)
            return {
                "provider": "openai",
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
                "openai_hard_cap_active": bool(
                    self._usage_data.get("openai_hard_cap_active")
                ),
                "openai_hard_cap_reason": self._usage_data.get(
                    "openai_hard_cap_reason", ""
                ),
                "alerts_sent": list(self._alerts_sent),
                "last_updated": self._usage_data.get("last_updated"),
            }

    # ── Hard-cap kill-switch ──────────────────────────────────────────

    def trip_hard_cap(self, reason: str = "") -> None:
        with self._data_lock:
            self._usage_data["openai_hard_cap_active"] = True
            self._usage_data["openai_hard_cap_reason"] = reason
            self._usage_data["openai_hard_cap_tripped_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        logger.warning(
            f"openai_cost_tracker: hard-cap TRIPPED "
            f"({reason or 'no-reason-given'}) — all OpenAI API calls 429 "
            f"until clear_hard_cap()"
        )

    def clear_hard_cap(self) -> dict:
        with self._data_lock:
            was_active = bool(self._usage_data.get("openai_hard_cap_active"))
            self._usage_data["openai_hard_cap_active"] = False
            self._usage_data["openai_hard_cap_reason"] = ""
            self._usage_data["openai_hard_cap_cleared_at"] = (
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
            f"{emoji} <b>OpenAI Budget {threshold}% erreicht</b>\n\n"
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

    def _post_to_master(self, model: str, num_images: int = 1) -> None:
        url = f"{self.master_url}/internal/openai-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                url,
                json={
                    "model": model,
                    "num_images": num_images,
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
        url = f"{self.master_url}/internal/openai-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache = data
        self._master_status_cache_ts = now
        return data


# Singleton instance for easy import
openai_cost_tracker = OpenAICostTracker()
