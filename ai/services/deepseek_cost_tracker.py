"""
DeepSeek API Cost Tracker
=========================

Tracks usage and costs for DeepSeek V4 chat-completion calls
(deepseek-v4-flash, deepseek-v4-pro) via the OpenAI-compatible
endpoint at ``https://api.deepseek.com/v1``.

Mirrors ``openai_cost_tracker.py`` + ``minimax_cost_tracker.py`` in
shape so the operational patterns are identical:
  - Singleton in-memory state, persisted to ``deepseek_usage_<YYYY-MM>.json``
  - Master/client mode for federation-shared counting
  - Reuses ``COST_TRACKER_SHARED_SECRET`` env (one secret, four trackers
    now — Gemini, MiniMax, OpenAI, DeepSeek; each its own cap envelope)
  - Persistent hard-cap kill-switch survives restart
  - 80/95/100 threshold alerts via Telegram

Minimax IACP 315228fe (2026-06-14) requested DeepSeek V4 alongside
MiniMax M3 with the same `/ai/m3`-style endpoint pattern. DeepSeek is
PAYG (not Subscription) — exact pricing varies by model + cache hit/miss
status (see ``DEEPSEEK_PRICING`` below). DeepSeek is known to be cheap
relative to other PAYG providers; conservative defaults used here, exact
verification against platform.deepseek.com on each new model addition.
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


# DeepSeek pricing — USD per 1M tokens. Source: platform.deepseek.com/pricing
# DeepSeek differentiates between cache hit and cache miss for input tokens
# in the live API response (``prompt_cache_hit_tokens`` /
# ``prompt_cache_miss_tokens``). We attribute conservatively to the miss
# price for total input, which slightly over-counts when the prompt was
# cache-hit. Per-call cache-aware accounting is a follow-up.
DEEPSEEK_PRICING = {
    "deepseek-v4-flash": {"input_per_1m": 0.07,  "output_per_1m": 0.27},
    "deepseek-v4-pro":   {"input_per_1m": 0.27,  "output_per_1m": 1.10},
    # Default fallback
    "default":           {"input_per_1m": 0.07,  "output_per_1m": 0.27},
}

# EUR/USD exchange rate (approximate)
EUR_USD_RATE = 1.05


class DeepSeekCostTracker:
    """Singleton cost tracker for DeepSeek V4 chat-completion usage."""

    _instance: Optional["DeepSeekCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DeepSeekCostTracker":
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

        # 25 EUR/month default — aligned with MiniMax (also PAYG text).
        # DeepSeek is the cheapest of the PAYG text providers so this is
        # generous; tune via DEEPSEEK_MONTHLY_BUDGET_EUR if Alex prefers
        # a different envelope.
        self.monthly_budget_eur = float(
            os.getenv("DEEPSEEK_MONTHLY_BUDGET_EUR", "25.0")
        )
        self.data_dir = Path(
            os.getenv("DEEPSEEK_COST_TRACKER_DATA_DIR", "/var/lib/api-ai")
        )
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.block_on_budget_exceeded = (
            os.getenv("DEEPSEEK_BLOCK_ON_BUDGET_EXCEEDED", "true").lower() == "true"
        )
        self.alert_thresholds = [80, 95, 100]

        self.master_url = os.getenv("DEEPSEEK_COST_TRACKER_MASTER_URL", "").rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")

        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0

        self._load_data()

        logger.info(
            f"DeepSeekCostTracker initialized: budget={self.monthly_budget_eur}EUR, "
            f"mode={'client' if self.master_url else 'master'}, "
            f"block_on_exceeded={self.block_on_budget_exceeded}"
        )

    @property
    def data_file(self) -> Path:
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"deepseek_usage_{month_key}.json"

    def _load_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                    self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
                logger.info(
                    f"Loaded deepseek usage: "
                    f"{self._usage_data.get('total_cost_eur', 0):.4f} EUR"
                )
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load deepseek usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save deepseek usage data: {e}")

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

    def _cost_for_call(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float]:
        pricing = DEEPSEEK_PRICING.get(model, DEEPSEEK_PRICING["default"])
        cost_usd = (
            (input_tokens * pricing["input_per_1m"])
            + (output_tokens * pricing["output_per_1m"])
        ) / 1_000_000.0
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    # ── Public track method ───────────────────────────────────────────

    def track_text(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Track a DeepSeek V4 chat-completion call. Routes to master if
        client mode, otherwise tracks locally.
        """
        if input_tokens <= 0 and output_tokens <= 0:
            return
        if self.master_url and self.shared_secret:
            try:
                self._post_to_master(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return
            except Exception as e:
                logger.error(
                    f"deepseek_cost_tracker: master post failed ({e}); "
                    f"falling back to local — cap may temporarily lag"
                )
        self._track_local(model, input_tokens, output_tokens)

    def _track_local(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        cost_usd, cost_eur = self._cost_for_call(model, input_tokens, output_tokens)
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
                "modality": "text",
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "cost_eur": 0.0,
            })
            stats["request_count"] += 1
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["cost_usd"] += cost_usd
            stats["cost_eur"] += cost_eur
            self._save_data()
            self._check_thresholds()
        logger.info(
            f"DeepSeek tracked: text/{model} in={input_tokens} out={output_tokens} "
            f"= {cost_eur:.6f}EUR (total: {self._usage_data['total_cost_eur']:.4f}EUR)"
        )

    # ── Block-decision + status ───────────────────────────────────────

    def is_budget_exceeded(self) -> bool:
        return self._usage_data.get("total_cost_eur", 0) >= self.monthly_budget_eur

    def should_block_request(self) -> bool:
        if self._usage_data.get("deepseek_hard_cap_active"):
            return True
        if self.master_url and self.shared_secret:
            try:
                status = self._fetch_master_status()
                return bool(status.get("would_block", False))
            except Exception as e:
                logger.warning(
                    f"deepseek_cost_tracker: master query failed ({e}); "
                    f"using local view for block-decision"
                )
        return self.block_on_budget_exceeded and self.is_budget_exceeded()

    def get_status(self) -> dict:
        with self._data_lock:
            cost_eur = self._usage_data.get("total_cost_eur", 0)
            return {
                "provider": "deepseek",
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
                "deepseek_hard_cap_active": bool(
                    self._usage_data.get("deepseek_hard_cap_active")
                ),
                "deepseek_hard_cap_reason": self._usage_data.get(
                    "deepseek_hard_cap_reason", ""
                ),
                "alerts_sent": list(self._alerts_sent),
                "last_updated": self._usage_data.get("last_updated"),
            }

    # ── Hard-cap kill-switch ──────────────────────────────────────────

    def trip_hard_cap(self, reason: str = "") -> None:
        with self._data_lock:
            self._usage_data["deepseek_hard_cap_active"] = True
            self._usage_data["deepseek_hard_cap_reason"] = reason
            self._usage_data["deepseek_hard_cap_tripped_at"] = (
                datetime.now().isoformat()
            )
            self._save_data()
        logger.warning(
            f"deepseek_cost_tracker: hard-cap TRIPPED "
            f"({reason or 'no-reason-given'}) — DeepSeek calls 429 "
            f"until clear_hard_cap()"
        )

    def clear_hard_cap(self) -> dict:
        with self._data_lock:
            was_active = bool(self._usage_data.get("deepseek_hard_cap_active"))
            self._usage_data["deepseek_hard_cap_active"] = False
            self._usage_data["deepseek_hard_cap_reason"] = ""
            self._usage_data["deepseek_hard_cap_cleared_at"] = (
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
            f"{emoji} <b>DeepSeek Budget {threshold}% erreicht</b>\n\n"
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
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        url = f"{self.master_url}/internal/deepseek-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                url,
                json={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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
        url = f"{self.master_url}/internal/deepseek-cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache = data
        self._master_status_cache_ts = now
        return data


# Singleton instance for easy import
deepseek_cost_tracker = DeepSeekCostTracker()
