"""
Internal Routes
===============

Endpoints intended for federation-internal callers (Automation,
maintenance crons, sibling agents). Do **NOT** expose these via a
user-facing frontend — they trigger expensive background work.

Currently:
  • POST /internal/notify-cli-update
      Called by Automation's CLI-update orchestrator after a successful
      CLI update run.

      Two modes:
      (1) Curated push (preferred): payload includes `models` — the
          handler writes /var/lib/api-ai/models.json directly with
          Automation's KI-curated provider/model lists. Only the
          diff-alert is then spawned (no rediscovery — Automation
          already did the smart work via web-research). Reflects in
          /ai/models immediately.
      (2) Legacy trigger: payload omits `models` — handler runs the
          local discovery script which writes models.json, then runs
          diff-alert. Reflects in /ai/models after ~30s. Kept for
          backward compatibility and as a fallback path.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path as _Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

_MODELS_JSON = _Path("/var/lib/api-ai/models.json")
_MODELS_PREV = _Path("/var/lib/api-ai/models.prev.json")


class CliUpdateNotification(BaseModel):
    """Payload posted by Automation after a CLI-update run."""

    host: str
    users: Optional[list[str]] = None
    results: Optional[dict[str, Any]] = None
    # Curated provider→models map (KI-validated). When present the
    # handler writes this directly to models.json instead of running
    # local discovery. Shape mirrors what the discovery script writes:
    #   {"updated_at": "...", "providers": {"claude": {...}, ...}}
    models: Optional[dict[str, Any]] = None


def _write_curated_models(host: str, models: dict[str, Any]) -> None:
    """Atomically write Automation's curated payload to models.json."""
    disk = {
        "host": host,
        "updated_at": models.get("updated_at"),
        "providers": models.get("providers", {}),
        "_source": "automation-curated",
    }
    # Preserve previous snapshot so diff-alert can compare.
    if _MODELS_JSON.exists():
        _MODELS_PREV.write_text(_MODELS_JSON.read_text())
    tmp = _MODELS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(disk, indent=2))
    tmp.replace(_MODELS_JSON)


def _spawn(cmd: str) -> None:
    """Fire-and-forget shell command, fully detached."""
    subprocess.Popen(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


@router.post("/notify-cli-update")
async def notify_cli_update(payload: CliUpdateNotification):
    """Fire-and-forget hook: write curated models or trigger rediscovery.

    Returns 200 immediately so Automation's orchestrator never blocks.
    """
    logger.info(
        f"notify-cli-update from host={payload.host} "
        f"users={payload.users or []} "
        f"models_present={payload.models is not None}"
    )

    venv_python = "/var/www/api-ai.arkturian.com/venv/bin/python"
    discover = "/usr/local/bin/api-ai-models-discovery.py"
    diff_alert = "/usr/local/bin/api-ai-models-diff-alert.py"

    # ── Mode 1: curated push ───────────────────────────────────────────
    if payload.models is not None:
        try:
            _write_curated_models(payload.host, payload.models)
            logger.info(
                f"notify-cli-update: wrote curated models.json "
                f"(host={payload.host})"
            )
        except Exception as e:
            logger.error(f"notify-cli-update: curated write failed: {e}")
            return {"accepted": False, "reason": f"write failed: {e}"}

        if os.path.exists(diff_alert):
            try:
                _spawn(f"{venv_python} {diff_alert}")
            except Exception as e:
                logger.error(f"notify-cli-update: diff-alert spawn failed: {e}")

        return {
            "accepted": True,
            "triggered": ["direct-write", "diff-alert"],
            "note": "curated models persisted; /ai/models reflects immediately",
        }

    # ── Mode 2: legacy trigger (no curated payload) ────────────────────
    if not os.path.exists(discover):
        return {
            "accepted": False,
            "reason": "discovery script not deployed on this host",
            "expected_path": discover,
        }
    try:
        _spawn(f"{venv_python} {discover} && {venv_python} {diff_alert}")
    except Exception as e:
        logger.error(f"failed to spawn discovery: {e}")
        return {"accepted": False, "reason": f"spawn failed: {e}"}

    return {
        "accepted": True,
        "triggered": ["discovery", "diff-alert"],
        "note": "out-of-band; check /var/log/api-ai-maintenance.log and /ai/models after ~30s",
    }
