"""
Gemini API Cost Tracker Service
===============================

Tracks token usage and costs for Gemini API calls.
Sends Telegram alerts at configurable thresholds.
Blocks requests when monthly budget is exceeded.

Pricing (Gemini 2.0 Flash - as of Dec 2024):
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# Gemini pricing (USD per 1M tokens) - Updated Dec 2024
# See: https://ai.google.dev/pricing
PRICING = {
    # Text Models
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},

    # Image Generation Models (per image, not per token)
    # Pricing: ~$0.02-0.04 per image for Imagen, estimate for Gemini image
    "gemini-2.5-flash-image": {"per_image": 0.02},
    "gemini-3-pro-image-preview": {"per_image": 0.04},
    "imagen-4.0-generate-001": {"per_image": 0.03},

    # Audio/STT - same token pricing as text models
    # Note: Audio input is converted to tokens (~25 tokens/second of audio)
    "gemini-1.5-flash-audio": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro-audio": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp-audio": {"input": 0.075, "output": 0.30},

    # Default fallback
    "default": {"input": 0.075, "output": 0.30},
}

# EUR/USD exchange rate (approximate)
EUR_USD_RATE = 1.05


class CostTracker:
    """
    Singleton cost tracker for Gemini API usage.

    Features:
    - Tracks input/output tokens per request
    - Calculates costs in EUR
    - Persists monthly usage to JSON file
    - Sends Telegram alerts at thresholds (50%, 80%, 95%, 100%)
    - Can block requests when budget exceeded
    """

    _instance: Optional["CostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CostTracker":
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

        # Configuration from environment.
        #
        # Default monthly cap lowered 2026-05-30 from 30 → 15 EUR following
        # the Mai 2026 cost incident (209.56 EUR for /ai/gemini API calls).
        # Alex' direct ask: 15 EUR/month TOTAL across both hosts, enforced
        # as a *shared* counter via the master/client setup below.
        #
        # Per-host fallback: if COST_TRACKER_MASTER_URL is unset on a
        # non-master host, that host falls back to its own local counter
        # and the cap effectively becomes per-host (caller should still
        # see correct gate behaviour, just without cross-host coherence).
        # Set GEMINI_MONTHLY_BUDGET_EUR to override the 15 EUR default.
        self.monthly_budget_eur = float(os.getenv("GEMINI_MONTHLY_BUDGET_EUR", "15.0"))

        # Master/client mode — see _track_usage_local + the corresponding
        # /internal/cost-shared-state endpoints in internal_routes.py.
        # When COST_TRACKER_MASTER_URL is set, this instance becomes a
        # CLIENT and forwards track/should_block calls to the master.
        # When unset, this instance is either the master (canonical) or a
        # standalone host in fallback mode.
        self.master_url = os.getenv("COST_TRACKER_MASTER_URL", "").rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")
        # Cache for the latest known master status — used as fallback if
        # the master is briefly unreachable so a transient network hiccup
        # doesn't fail-open all API-billed calls. TTL is short (10s) so
        # the cap reacts quickly to real spending.
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0
        self.data_dir = Path(os.getenv("COST_TRACKER_DATA_DIR", "/var/lib/api-ai"))
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        # Hard-block changed from optional → on-by-default. Disabling it
        # requires explicit BLOCK_ON_BUDGET_EXCEEDED=false, not the previous
        # silent-default fallback. The cost incident was partly enabled by a
        # cap that existed but didn't fire.
        self.block_on_budget_exceeded = os.getenv("BLOCK_ON_BUDGET_EXCEEDED", "true").lower() == "true"

        # Alert thresholds (percentage of budget). 50% removed (noise at low
        # caps), 80/95/100 kept — single Telegram per threshold per month
        # via _alerts_sent dedup below.
        self.alert_thresholds = [80, 95, 100]

        # In-memory state
        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()

        # Load existing data
        self._load_data()

        logger.info(
            f"CostTracker initialized: budget={self.monthly_budget_eur}EUR, "
            f"block_on_exceeded={self.block_on_budget_exceeded}"
        )

    @property
    def data_file(self) -> Path:
        """Get path to current month's data file."""
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"gemini_usage_{month_key}.json"

    def _load_data(self) -> None:
        """Load usage data from file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                    self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
                logger.info(f"Loaded usage data: {self._usage_data.get('total_cost_eur', 0):.4f} EUR")
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        """Save usage data to file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")

    def _reset_monthly_data(self) -> None:
        """Reset data for new month."""
        self._usage_data = {
            "month": datetime.now().strftime("%Y-%m"),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "total_cost_eur": 0.0,
            "request_count": 0,
            "by_model": {},
            "alerts_sent": [],
            "created_at": datetime.now().isoformat(),
        }
        self._alerts_sent = set()

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        """
        Calculate cost in USD and EUR.

        Returns:
            Tuple of (cost_usd, cost_eur)
        """
        pricing = PRICING.get(model, PRICING["default"])
        cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    def track_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Track token usage. Routes to the master if this host is a client,
        otherwise tracks locally. See ``_track_usage_local`` for the
        actual accounting logic.
        """
        if input_tokens == 0 and output_tokens == 0:
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
                # Don't fail the underlying request — but DO log loudly so
                # we notice the master is unreachable and a cap drift can
                # be reconciled. Local fallback also tracks, so the host
                # at least has its own slice of state until master recovers.
                logger.error(
                    f"cost_tracker: master post failed ({e}); falling back "
                    f"to local tracking — cap may temporarily lag"
                )

        self._track_usage_local(model, input_tokens, output_tokens)

    def _track_usage_local(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """The legacy local-file tracking path. Master mode + client
        fallback both end up here.
        """
        if input_tokens == 0 and output_tokens == 0:
            return

        cost_usd, cost_eur = self._calculate_cost(model, input_tokens, output_tokens)

        with self._data_lock:
            # Check if we need to reset for new month
            current_month = datetime.now().strftime("%Y-%m")
            if self._usage_data.get("month") != current_month:
                self._reset_monthly_data()

            # Update totals
            self._usage_data["total_input_tokens"] += input_tokens
            self._usage_data["total_output_tokens"] += output_tokens
            self._usage_data["total_cost_usd"] += cost_usd
            self._usage_data["total_cost_eur"] += cost_eur
            self._usage_data["request_count"] += 1

            # Update by-model stats
            if model not in self._usage_data["by_model"]:
                self._usage_data["by_model"][model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_eur": 0.0,
                    "request_count": 0,
                }
            self._usage_data["by_model"][model]["input_tokens"] += input_tokens
            self._usage_data["by_model"][model]["output_tokens"] += output_tokens
            self._usage_data["by_model"][model]["cost_usd"] += cost_usd
            self._usage_data["by_model"][model]["cost_eur"] += cost_eur
            self._usage_data["by_model"][model]["request_count"] += 1

            # Save to disk
            self._save_data()

            # Check thresholds and send alerts
            self._check_thresholds()

        logger.info(
            f"Tracked: {model} - {input_tokens}in/{output_tokens}out = "
            f"{cost_eur:.6f}EUR (total: {self._usage_data['total_cost_eur']:.4f}EUR)"
        )

    def track_image_generation(self, model: str, num_images: int = 1) -> None:
        """
        Track image generation costs.

        Args:
            model: Model name (e.g., "gemini-2.5-flash-image", "imagen-4.0-generate-001")
            num_images: Number of images generated
        """
        pricing = PRICING.get(model, {})
        per_image_cost = pricing.get("per_image", 0.03)  # Default $0.03 per image
        cost_usd = per_image_cost * num_images
        cost_eur = cost_usd / EUR_USD_RATE

        with self._data_lock:
            current_month = datetime.now().strftime("%Y-%m")
            if self._usage_data.get("month") != current_month:
                self._reset_monthly_data()

            self._usage_data["total_cost_usd"] += cost_usd
            self._usage_data["total_cost_eur"] += cost_eur
            self._usage_data["request_count"] += 1

            # Track images separately
            if "total_images" not in self._usage_data:
                self._usage_data["total_images"] = 0
            self._usage_data["total_images"] += num_images

            if model not in self._usage_data["by_model"]:
                self._usage_data["by_model"][model] = {
                    "images_generated": 0,
                    "cost_usd": 0.0,
                    "cost_eur": 0.0,
                    "request_count": 0,
                }
            self._usage_data["by_model"][model]["images_generated"] = \
                self._usage_data["by_model"][model].get("images_generated", 0) + num_images
            self._usage_data["by_model"][model]["cost_usd"] += cost_usd
            self._usage_data["by_model"][model]["cost_eur"] += cost_eur
            self._usage_data["by_model"][model]["request_count"] += 1

            self._save_data()
            self._check_thresholds()

        logger.info(
            f"Tracked image: {model} - {num_images} images = "
            f"{cost_eur:.4f}EUR (total: {self._usage_data['total_cost_eur']:.4f}EUR)"
        )

    def _check_thresholds(self) -> None:
        """Check if any alert thresholds have been crossed."""
        current_cost = self._usage_data["total_cost_eur"]
        percentage = (current_cost / self.monthly_budget_eur) * 100

        for threshold in self.alert_thresholds:
            if percentage >= threshold and threshold not in self._alerts_sent:
                self._alerts_sent.add(threshold)
                self._send_alert(threshold, current_cost, percentage)

    def _send_alert(self, threshold: int, current_cost: float, percentage: float) -> None:
        """Send Telegram alert for threshold."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning(f"Telegram not configured, skipping alert for {threshold}%")
            return

        # Determine emoji and urgency
        if threshold >= 100:
            emoji = "🚨"
            status = "BUDGET EXCEEDED"
        elif threshold >= 95:
            emoji = "⚠️"
            status = "CRITICAL"
        elif threshold >= 80:
            emoji = "🔶"
            status = "WARNING"
        else:
            emoji = "📊"
            status = "INFO"

        message = f"""
{emoji} <b>Gemini API Cost Alert - {status}</b>

<b>Threshold:</b> {threshold}% reached
<b>Current Cost:</b> {current_cost:.2f} EUR
<b>Monthly Budget:</b> {self.monthly_budget_eur:.2f} EUR
<b>Usage:</b> {percentage:.1f}%

<b>Details:</b>
- Requests: {self._usage_data['request_count']}
- Input Tokens: {self._usage_data['total_input_tokens']:,}
- Output Tokens: {self._usage_data['total_output_tokens']:,}

<b>Remaining:</b> {max(0, self.monthly_budget_eur - current_cost):.2f} EUR
"""

        if threshold >= 100 and self.block_on_budget_exceeded:
            message += "\n<b>⛔ New requests will be BLOCKED until next month!</b>"

        self._send_telegram_message(message.strip())
        logger.warning(f"Alert sent: {threshold}% threshold - {current_cost:.2f}EUR")

    def _send_telegram_message(self, message: str) -> None:
        """Send message via Telegram Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    logger.error(f"Telegram API error: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def is_budget_exceeded(self) -> bool:
        """Check if monthly budget is exceeded."""
        return self._usage_data.get("total_cost_eur", 0) >= self.monthly_budget_eur

    def should_block_request(self) -> bool:
        """Check if request should be blocked due to budget.

        Client mode: query master, fall back to local view if master is
        unreachable for >10s (short cache TTL means a real overage still
        triggers within seconds of the master observing it).

        Master / standalone: use local state directly.
        """
        if self.master_url and self.shared_secret:
            try:
                status = self._fetch_master_status()
                return bool(status.get("would_block", False))
            except Exception as e:
                logger.warning(
                    f"cost_tracker: master query failed ({e}); using local "
                    f"view for block-decision (may under-count cross-host)"
                )
                # Fall through to local view as graceful degradation
        return self.block_on_budget_exceeded and self.is_budget_exceeded()

    def _post_to_master(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Client → master: log a call. Raises on transport / auth errors so
        the caller can fall back to local tracking.
        """
        url = f"{self.master_url}/internal/cost-shared-state"
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
            # Refresh status cache from the master's response so the next
            # should_block_request() in this process sees the updated total
            # immediately, not after the 10s cache TTL.
            try:
                self._master_status_cache = r.json()
                import time as _t
                self._master_status_cache_ts = _t.time()
            except Exception:
                pass

    def _fetch_master_status(self) -> dict:
        """Client → master: read shared counter. 10s in-process cache so
        bursts of calls don't hammer the master.
        """
        import time as _t
        now = _t.time()
        if (
            self._master_status_cache is not None
            and now - self._master_status_cache_ts < 10.0
        ):
            return self._master_status_cache
        url = f"{self.master_url}/internal/cost-shared-state"
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache = data
        self._master_status_cache_ts = now
        return data

    def get_status(self) -> dict:
        """Get current usage status."""
        with self._data_lock:
            current_cost = self._usage_data.get("total_cost_eur", 0)
            return {
                "month": self._usage_data.get("month"),
                "total_cost_eur": round(current_cost, 4),
                "total_cost_usd": round(self._usage_data.get("total_cost_usd", 0), 4),
                "monthly_budget_eur": self.monthly_budget_eur,
                "usage_percentage": round((current_cost / self.monthly_budget_eur) * 100, 2),
                "remaining_eur": round(max(0, self.monthly_budget_eur - current_cost), 4),
                "request_count": self._usage_data.get("request_count", 0),
                "total_input_tokens": self._usage_data.get("total_input_tokens", 0),
                "total_output_tokens": self._usage_data.get("total_output_tokens", 0),
                "budget_exceeded": self.is_budget_exceeded(),
                "requests_blocked": self.should_block_request(),
                "by_model": self._usage_data.get("by_model", {}),
                "alerts_sent": list(self._alerts_sent),
                "last_updated": self._usage_data.get("last_updated"),
            }

    def send_daily_report(self) -> None:
        """Send daily usage report via Telegram."""
        status = self.get_status()

        message = f"""
📊 <b>Gemini API Daily Report</b>

<b>Month:</b> {status['month']}
<b>Cost:</b> {status['total_cost_eur']:.2f} EUR / {status['monthly_budget_eur']:.2f} EUR
<b>Usage:</b> {status['usage_percentage']:.1f}%
<b>Remaining:</b> {status['remaining_eur']:.2f} EUR

<b>Statistics:</b>
- Requests: {status['request_count']:,}
- Input Tokens: {status['total_input_tokens']:,}
- Output Tokens: {status['total_output_tokens']:,}

<b>Status:</b> {'⛔ BLOCKED' if status['requests_blocked'] else '✅ Active'}
"""
        self._send_telegram_message(message.strip())


# Global singleton instance
cost_tracker = CostTracker()
