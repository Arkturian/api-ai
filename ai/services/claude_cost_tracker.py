"""
Claude CLI Cost Tracker Service
================================

Tracks usage and costs for Claude CLI (claude -p) calls.
Similar to cost_tracker.py but for Claude Code CLI usage.

Note: Claude CLI uses your subscription (Pro/Team), costs shown are for reference only.
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

# EUR/USD exchange rate (approximate)
EUR_USD_RATE = 1.05


class ClaudeCostTracker:
    """
    Singleton cost tracker for Claude CLI usage.

    Features:
    - Tracks requests, tokens, and estimated costs
    - Persists monthly usage to JSON file
    - Provides status endpoint for monitoring
    - Sends Telegram alerts at thresholds (optional)

    Note: Claude CLI uses subscription, so costs are informational only.
    """

    _instance: Optional["ClaudeCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ClaudeCostTracker":
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

        # Configuration from environment
        self.data_dir = Path(os.getenv("COST_TRACKER_DATA_DIR", "/var/lib/api-ai"))
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

        # In-memory state
        self._usage_data: dict = {}
        self._data_lock = threading.Lock()

        # Load existing data
        self._load_data()

        logger.info("ClaudeCostTracker initialized")

    @property
    def data_file(self) -> Path:
        """Get path to current month's data file."""
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"claude_usage_{month_key}.json"

    def _load_data(self) -> None:
        """Load usage data from file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                logger.info(f"Loaded Claude usage data: {self._usage_data.get('total_cost_usd', 0):.4f} USD")
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load Claude usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        """Save usage data to file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Claude usage data: {e}")

    def _reset_monthly_data(self) -> None:
        """Reset data for new month."""
        self._usage_data = {
            "month": datetime.now().strftime("%Y-%m"),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0,
            "total_cost_eur": 0.0,
            "request_count": 0,
            "total_duration_ms": 0,
            "total_api_duration_ms": 0,
            "by_model": {},
            "created_at": datetime.now().isoformat(),
        }

    def track_usage(self, cli_response: dict) -> None:
        """
        Track usage from Claude CLI JSON response.

        Args:
            cli_response: Parsed JSON from claude -p --output-format json
        """
        try:
            with self._data_lock:
                # Check if we need to reset for new month
                current_month = datetime.now().strftime("%Y-%m")
                if self._usage_data.get("month") != current_month:
                    self._reset_monthly_data()

                # Extract usage data
                usage = cli_response.get("usage", {})
                model_usage = cli_response.get("modelUsage", {})

                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_creation = usage.get("cache_creation_input_tokens", 0)
                cost_usd = cli_response.get("total_cost_usd", 0.0)
                duration_ms = cli_response.get("duration_ms", 0)
                api_duration_ms = cli_response.get("duration_api_ms", 0)

                # Update totals
                self._usage_data["total_input_tokens"] += input_tokens
                self._usage_data["total_output_tokens"] += output_tokens
                self._usage_data["total_cache_read_tokens"] += cache_read
                self._usage_data["total_cache_creation_tokens"] += cache_creation
                self._usage_data["total_cost_usd"] += cost_usd
                self._usage_data["total_cost_eur"] = self._usage_data["total_cost_usd"] / EUR_USD_RATE
                self._usage_data["request_count"] += 1
                self._usage_data["total_duration_ms"] += duration_ms
                self._usage_data["total_api_duration_ms"] += api_duration_ms

                # Track per-model usage
                for model_name, model_data in model_usage.items():
                    if model_name not in self._usage_data["by_model"]:
                        self._usage_data["by_model"][model_name] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_read_tokens": 0,
                            "cache_creation_tokens": 0,
                            "cost_usd": 0.0,
                            "request_count": 0,
                        }

                    self._usage_data["by_model"][model_name]["input_tokens"] += model_data.get("inputTokens", 0)
                    self._usage_data["by_model"][model_name]["output_tokens"] += model_data.get("outputTokens", 0)
                    self._usage_data["by_model"][model_name]["cache_read_tokens"] += model_data.get("cacheReadInputTokens", 0)
                    self._usage_data["by_model"][model_name]["cache_creation_tokens"] += model_data.get("cacheCreationInputTokens", 0)
                    self._usage_data["by_model"][model_name]["cost_usd"] += model_data.get("costUSD", 0.0)
                    self._usage_data["by_model"][model_name]["request_count"] += 1

                # Save to disk
                self._save_data()
                from ai.services import kostenbuch as _kb   # #4127
                _kb.buchen(anbieter="claude_cli", art="abo", modell=next(iter(model_usage), None),
                           tokens_ein=input_tokens + cache_read + cache_creation,
                           tokens_aus=output_tokens, gegenwert_usd=cost_usd)

                logger.info(
                    f"Tracked Claude: {input_tokens}in/{output_tokens}out = "
                    f"${cost_usd:.4f} (total: ${self._usage_data['total_cost_usd']:.4f})"
                )

        except Exception as e:
            logger.error(f"Failed to track Claude usage: {e}")

    def get_status(self) -> dict:
        """Get current usage status."""
        with self._data_lock:
            return {
                "month": self._usage_data.get("month"),
                "total_cost_usd": round(self._usage_data.get("total_cost_usd", 0), 4),
                "total_cost_eur": round(self._usage_data.get("total_cost_eur", 0), 4),
                "request_count": self._usage_data.get("request_count", 0),
                "total_input_tokens": self._usage_data.get("total_input_tokens", 0),
                "total_output_tokens": self._usage_data.get("total_output_tokens", 0),
                "total_cache_read_tokens": self._usage_data.get("total_cache_read_tokens", 0),
                "total_cache_creation_tokens": self._usage_data.get("total_cache_creation_tokens", 0),
                "total_duration_ms": self._usage_data.get("total_duration_ms", 0),
                "total_api_duration_ms": self._usage_data.get("total_api_duration_ms", 0),
                "avg_duration_ms": round(
                    self._usage_data.get("total_duration_ms", 0) / max(1, self._usage_data.get("request_count", 1)),
                    0
                ),
                "by_model": self._usage_data.get("by_model", {}),
                "last_updated": self._usage_data.get("last_updated"),
                "note": "Costs are informational - Claude CLI uses your subscription",
            }

    def _send_telegram_message(self, message: str) -> None:
        """Send message via Telegram Bot API."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
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


# Global singleton instance
claude_cost_tracker = ClaudeCostTracker()
