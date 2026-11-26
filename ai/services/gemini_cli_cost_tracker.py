"""
Gemini CLI Cost Tracker Service
================================

Tracks usage and costs for Google Gemini CLI calls.
Similar to codex_cost_tracker.py but for Gemini CLI usage.

Note: Gemini CLI uses free tier (60 req/min, 1000 req/day), costs shown are for reference only.
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

# Google Gemini pricing (USD per 1M tokens) - for reference only
# Gemini CLI uses free tier, but we track for informational purposes
PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.01, "output": 0.04},
    "default": {"input": 0.01, "output": 0.04},  # Use flash-lite pricing as default
}


class GeminiCliCostTracker:
    """
    Singleton cost tracker for Gemini CLI usage.

    Features:
    - Tracks requests, tokens, and estimated costs
    - Persists monthly usage to JSON file
    - Provides status endpoint for monitoring

    Note: Gemini CLI uses free tier, so costs are informational only.
    """

    _instance: Optional["GeminiCliCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GeminiCliCostTracker":
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

        logger.info("GeminiCliCostTracker initialized")

    @property
    def data_file(self) -> Path:
        """Get path to current month's data file."""
        month_key = datetime.now().strftime("%Y-%m")
        return self.data_dir / f"gemini_cli_usage_{month_key}.json"

    def _load_data(self) -> None:
        """Load usage data from file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                logger.info(f"Loaded Gemini CLI usage data: {self._usage_data.get('total_cost_usd', 0):.4f} USD")
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error(f"Failed to load Gemini CLI usage data: {e}")
            self._reset_monthly_data()

    def _save_data(self) -> None:
        """Save usage data to file."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Gemini CLI usage data: {e}")

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
            "created_at": datetime.now().isoformat(),
        }

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> tuple:
        """Calculate estimated cost in USD and EUR."""
        pricing = PRICING.get(model, PRICING["default"])
        cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        cost_eur = cost_usd / EUR_USD_RATE
        return cost_usd, cost_eur

    def track_usage(self, usage_data: dict) -> None:
        """
        Track usage from Gemini CLI response.

        Args:
            usage_data: Dict with input_tokens, output_tokens, model
        """
        try:
            with self._data_lock:
                # Check if we need to reset for new month
                current_month = datetime.now().strftime("%Y-%m")
                if self._usage_data.get("month") != current_month:
                    self._reset_monthly_data()

                # Extract usage data
                input_tokens = usage_data.get("input_tokens", 0)
                output_tokens = usage_data.get("output_tokens", 0)
                model = usage_data.get("model", "gemini-2.5-flash-lite")

                # Calculate cost
                cost_usd, cost_eur = self._calculate_cost(model, input_tokens, output_tokens)

                # Update totals
                self._usage_data["total_input_tokens"] += input_tokens
                self._usage_data["total_output_tokens"] += output_tokens
                self._usage_data["total_cost_usd"] += cost_usd
                self._usage_data["total_cost_eur"] += cost_eur
                self._usage_data["request_count"] += 1

                # Track per-model usage
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
                from ai.services import kostenbuch as _kb   # #4127
                _kb.buchen(anbieter="agy", art="abo", modell=model, tokens_ein=input_tokens,
                           tokens_aus=output_tokens, gegenwert_usd=cost_usd)

                logger.info(
                    f"Tracked Gemini CLI: {input_tokens}in/{output_tokens}out = "
                    f"${cost_usd:.4f} (total: ${self._usage_data['total_cost_usd']:.4f})"
                )

        except Exception as e:
            logger.error(f"Failed to track Gemini CLI usage: {e}")

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
                "by_model": self._usage_data.get("by_model", {}),
                "last_updated": self._usage_data.get("last_updated"),
                "note": "Costs are informational - Gemini CLI uses free tier (60 req/min, 1000 req/day)",
            }


# Global singleton instance
gemini_cli_cost_tracker = GeminiCliCostTracker()
