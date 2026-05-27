"""
Internal Routes
===============

Endpoints intended for federation-internal callers (Automation,
maintenance crons, sibling agents). Do **NOT** expose these via a
user-facing frontend — they trigger expensive background work.

Currently:
  • POST /internal/notify-cli-update
      Called by Automation's CLI-update orchestrator after a successful
      `npm install -g @openai/codex@latest` (or sibling) run. Triggers
      an out-of-band re-discovery so the cached models.json reflects
      the new CLI versions within seconds instead of waiting for the
      next 05:00 timer.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class CliUpdateNotification(BaseModel):
    """Payload posted by Automation after a CLI-update run."""

    host: str
    users: Optional[list[str]] = None
    results: Optional[dict[str, Any]] = None


@router.post("/notify-cli-update")
async def notify_cli_update(payload: CliUpdateNotification):
    """Fire-and-forget hook: re-run the discovery script in the background.

    Returns 200 immediately so Automation's orchestrator never blocks.
    The discovery rewrites /var/lib/api-ai/models.json which the
    /ai/models endpoint reads on every request. A subsequent Telegram
    alert fires automatically if the smoke-test produces a diff.
    """
    logger.info(
        f"notify-cli-update received from host={payload.host} "
        f"users={payload.users or []}"
    )

    script = "/usr/local/bin/api-ai-models-discovery.py"
    diff_alert = "/usr/local/bin/api-ai-models-diff-alert.py"
    venv_python = "/var/www/api-ai.arkturian.com/venv/bin/python"

    if not os.path.exists(script):
        return {
            "accepted": False,
            "reason": "discovery script not deployed on this host",
            "expected_path": script,
        }

    # Build a tiny wrapper that runs discovery then diff-alert. We
    # detach so Automation's HTTP call returns within ms.
    cmd = (
        f"{venv_python} {script} && "
        f"{venv_python} {diff_alert}"
    )
    try:
        subprocess.Popen(
            ["bash", "-c", cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        logger.error(f"failed to spawn discovery: {e}")
        return {"accepted": False, "reason": f"spawn failed: {e}"}

    return {
        "accepted": True,
        "triggered": ["discovery", "diff-alert"],
        "note": "out-of-band; check /var/log/api-ai-maintenance.log and /ai/models after ~30s",
    }
