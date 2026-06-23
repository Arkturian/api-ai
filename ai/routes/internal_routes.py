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

from fastapi import APIRouter, Header, HTTPException
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


def _verify_shared_counter_auth(provided: Optional[str]) -> None:
    """Compare provided header against ``COST_TRACKER_SHARED_SECRET`` env."""
    expected = os.environ.get("COST_TRACKER_SHARED_SECRET", "")
    if not expected:
        # Master mode is opt-in. If the secret isn't configured, decline
        # cleanly instead of letting unauthenticated callers through.
        raise HTTPException(
            status_code=503,
            detail="shared-counter master is not configured on this host",
        )
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="invalid shared counter auth")


class _SharedTrackPayload(BaseModel):
    """Payload posted by a sibling host to log a Gemini API call into the
    shared monthly counter. Field names align with the existing local
    ``cost_tracker._usage_data`` shape so the master can just delegate."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "default"
    # Caller's identification — purely informational, ends up in
    # by_host breakdown for postmortem-debug. NOT used for auth.
    source_host: Optional[str] = None


class _DeepSeekSharedTrackPayload(BaseModel):
    """Payload posted by a sibling host to log a DeepSeek V4 chat-completion
    call into the shared monthly counter. Mirror of ``_MinimaxSharedTrackPayload``
    with the M3-style (input_tokens, output_tokens) shape."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    source_host: Optional[str] = None


class _OpenAISharedTrackPayload(BaseModel):
    """Payload posted by a sibling host to log an OpenAI Images API call
    into the shared monthly counter. Mirror of ``_MinimaxSharedTrackPayload``
    with the per-image counter shape."""

    model: str
    num_images: int = 1
    source_host: Optional[str] = None


class _OpenAIRealtimeSharedTrackPayload(BaseModel):
    """Payload posted by a sibling host to log an OpenAI Realtime session's
    token usage into the shared monthly counter. Distinct audio + text
    input/output counts so the per-modality billing model is preserved.

    Idempotency keys (Content-Post #1215 Codex contract): the originating
    api-ai host forwards ``(voice_session_id, usage_event_id)`` along so
    the master dedupes there too. Without dedup at the master, retries
    from the client host could still double-count federation-wide."""

    model: str
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    duration_sec: float = 0.0
    source_host: Optional[str] = None
    voice_session_id: Optional[str] = None
    usage_event_id: Optional[str] = None


class _MinimaxSharedTrackPayload(BaseModel):
    """Payload posted by a sibling host to log a MiniMax API call into the
    shared monthly counter. Mirrors ``_SharedTrackPayload`` but with the
    MiniMax-specific per-modality units (per_image / per_second / etc.)."""

    modality: str  # "image" | "video" | "tts" | "voice_clone" | "music"
    model: str
    units: dict = {}  # e.g. {"num_images": 1} or {"seconds": 5}
    source_host: Optional[str] = None


@router.get("/cost-shared-state")
async def cost_shared_state_get(
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side state read. Federation-internal: sibling api-ai hosts
    query this before serving an API-billed Gemini call so the cap is
    enforced against a single shared counter.

    Auth: caller must send the configured shared secret in
    ``X-Internal-Auth`` header. See cost_tracker.CostTracker for client side.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.cost_tracker import cost_tracker
    status = cost_tracker.get_status()
    # Reuse the existing block-decision so the client doesn't reimplement it.
    status["would_block"] = cost_tracker.should_block_request()
    return status


@router.post("/cost-shared-state")
async def cost_shared_state_track(
    payload: _SharedTrackPayload,
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side increment. Sibling host calls this *before* serving an
    API-billed Gemini call so the cap is enforced against a single counter.

    We log into the same ``cost_tracker`` the master itself uses, so
    arkserver's own calls and arkturian's reported calls accumulate
    together into ``gemini_usage_<YYYY-MM>.json``.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.cost_tracker import cost_tracker
    # Force local-mode write on the master: even if this process has
    # COST_TRACKER_MASTER_URL set (shouldn't, but defensive), the master
    # IS the source of truth and must not recurse to itself.
    cost_tracker._track_usage_local(
        model=payload.model,
        input_tokens=payload.input_tokens,
        output_tokens=payload.output_tokens,
    )
    status = cost_tracker.get_status()
    status["would_block"] = cost_tracker.should_block_request()
    return status


@router.get("/minimax-cost-shared-state")
async def minimax_cost_shared_state_get(
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side state read for the MiniMax shared counter.

    Federation-internal twin of ``/internal/cost-shared-state`` (Gemini).
    Sibling api-ai hosts query this before serving an API-billed MiniMax
    call so the 25 EUR monthly cap is enforced against a single shared
    counter instead of N independent per-host counters.

    Auth contract identical: caller sends the configured shared secret in
    the ``X-Internal-Auth`` header. The secret env (``COST_TRACKER_SHARED_SECRET``)
    is shared with the Gemini-side counter — one secret, two endpoints,
    same trust model (same-owner federation).
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.minimax_cost_tracker import minimax_cost_tracker
    status = minimax_cost_tracker.get_status()
    status["would_block"] = minimax_cost_tracker.should_block_request()
    return status


@router.post("/minimax-cost-shared-state")
async def minimax_cost_shared_state_track(
    payload: _MinimaxSharedTrackPayload,
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side increment for the MiniMax shared counter.

    Sibling host calls this *before* serving an API-billed MiniMax call.
    The master attributes the call to the matching modality method on
    its local ``minimax_cost_tracker`` so arkserver-direct calls and
    arkturian-reported calls accumulate into the same monthly file.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.minimax_cost_tracker import minimax_cost_tracker
    modality = payload.modality
    model = payload.model
    units = payload.units or {}

    # Dispatch to the matching local-track method based on modality.
    # ``_track_local`` is used because the master IS the source of
    # truth — recursing through ``_track`` (which would post-to-master
    # if it had a master_url set) would loop infinitely.
    if modality == "text":
        minimax_cost_tracker._track_local(
            "text", model,
            input_tokens=units.get("input_tokens", 0),
            output_tokens=units.get("output_tokens", 0),
        )
    elif modality == "image":
        minimax_cost_tracker._track_local(
            "image", model, num_images=units.get("num_images", 1)
        )
    elif modality == "video":
        minimax_cost_tracker._track_local(
            "video", model, seconds=units.get("seconds", 0)
        )
    elif modality == "tts":
        minimax_cost_tracker._track_local(
            "tts", model, chars=units.get("chars", 0)
        )
    elif modality == "voice_clone":
        minimax_cost_tracker._track_local(
            "voice_clone", model, num_clones=units.get("num_clones", 1)
        )
    elif modality == "music":
        minimax_cost_tracker._track_local(
            "music", model, num_tracks=units.get("num_tracks", 1)
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"unknown modality '{modality}' — "
                   f"valid: text|image|video|tts|voice_clone|music",
        )

    status = minimax_cost_tracker.get_status()
    status["would_block"] = minimax_cost_tracker.should_block_request()
    return status


@router.get("/deepseek-cost-shared-state")
async def deepseek_cost_shared_state_get(
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side state read for the DeepSeek shared counter.

    Federation-internal twin of the Gemini / MiniMax / OpenAI shared-state
    endpoints. Same ``X-Internal-Auth`` header + same
    ``COST_TRACKER_SHARED_SECRET`` env — one secret, four endpoints,
    same trust model.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.deepseek_cost_tracker import deepseek_cost_tracker
    status = deepseek_cost_tracker.get_status()
    status["would_block"] = deepseek_cost_tracker.should_block_request()
    return status


@router.post("/deepseek-cost-shared-state")
async def deepseek_cost_shared_state_track(
    payload: _DeepSeekSharedTrackPayload,
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side increment for the DeepSeek shared counter."""
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.deepseek_cost_tracker import deepseek_cost_tracker
    deepseek_cost_tracker._track_local(
        model=payload.model,
        input_tokens=payload.input_tokens,
        output_tokens=payload.output_tokens,
    )
    status = deepseek_cost_tracker.get_status()
    status["would_block"] = deepseek_cost_tracker.should_block_request()
    return status


@router.get("/openai-cost-shared-state")
async def openai_cost_shared_state_get(
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side state read for the OpenAI shared counter.

    Federation-internal twin of ``/internal/cost-shared-state`` (Gemini)
    and ``/internal/minimax-cost-shared-state`` (MiniMax). Sibling api-ai
    hosts query this before serving an OpenAI Images API call so the
    50 EUR monthly cap is enforced against a single shared counter
    instead of N independent per-host counters.

    Auth: same ``X-Internal-Auth`` header + same ``COST_TRACKER_SHARED_SECRET``
    env as the other two — one secret, three endpoints, same trust model.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.openai_cost_tracker import openai_cost_tracker
    status = openai_cost_tracker.get_status()
    status["would_block"] = openai_cost_tracker.should_block_request()
    return status


@router.post("/openai-cost-shared-state")
async def openai_cost_shared_state_track(
    payload: _OpenAISharedTrackPayload,
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side increment for the OpenAI shared counter.

    Sibling host calls this *before* (or rather: after) serving an
    OpenAI Images API call. The master delegates to its local
    ``openai_cost_tracker._track_local`` so arkserver-direct calls and
    arkturian-reported calls accumulate into the same monthly file.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.openai_cost_tracker import openai_cost_tracker
    openai_cost_tracker._track_local(
        model=payload.model, num_images=payload.num_images
    )
    status = openai_cost_tracker.get_status()
    status["would_block"] = openai_cost_tracker.should_block_request()
    return status


@router.get("/openai-realtime-cost-shared-state")
async def openai_realtime_cost_shared_state_get(
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side state read for the OpenAI Realtime shared counter.

    Federation-internal twin of the four older shared-state endpoints.
    Sibling api-ai hosts query this before minting an ephemeral Realtime
    token so the 100 EUR monthly cap is enforced against a single shared
    counter instead of N independent per-host counters.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    status = openai_realtime_cost_tracker.get_status()
    status["would_block"] = openai_realtime_cost_tracker.should_block_request()
    return status


@router.post("/openai-realtime-cost-shared-state")
async def openai_realtime_cost_shared_state_track(
    payload: _OpenAIRealtimeSharedTrackPayload,
    x_internal_auth: Optional[str] = Header(default=None, alias="X-Internal-Auth"),
):
    """Master-side increment for the OpenAI Realtime shared counter.

    Dedup on ``(voice_session_id, usage_event_id)`` (Post #1215 Codex
    contract). The master uses ``track_session`` (not ``_track_local``)
    so the dedup path runs at the master too — without it, retries from
    a sibling host would still double-count federation-wide.
    """
    _verify_shared_counter_auth(x_internal_auth)
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    # Avoid recursion: temporarily clear master_url so this call writes
    # locally (we ARE the master). _track_local doesn't carry the dedup
    # key, so we go through the dedup-aware ``track_session`` path with
    # master_url stubbed out for the duration of this call.
    saved = openai_realtime_cost_tracker.master_url
    openai_realtime_cost_tracker.master_url = ""
    try:
        result = openai_realtime_cost_tracker.track_session(
            model=payload.model,
            audio_input_tokens=payload.audio_input_tokens,
            audio_output_tokens=payload.audio_output_tokens,
            text_input_tokens=payload.text_input_tokens,
            text_output_tokens=payload.text_output_tokens,
            duration_sec=payload.duration_sec,
            voice_session_id=payload.voice_session_id,
            usage_event_id=payload.usage_event_id,
        )
    finally:
        openai_realtime_cost_tracker.master_url = saved
    status = openai_realtime_cost_tracker.get_status()
    status["would_block"] = openai_realtime_cost_tracker.should_block_request()
    status["deduped"] = bool(result and result.get("deduped"))
    return status


@router.get("/cli-pressure")
async def cli_pressure():
    """Live snapshot of per-provider CLI semaphore saturation.

    Returns counters of how many slots are in use vs configured cap, so
    monitoring or status dashboards can spot when /ai/<p> traffic is
    queueing on the host-protection caps. Cheap, side-effect-free.
    """
    # Lazy import to avoid touching the loop-local sems at module load
    from ai.routes.text_ai_routes import _CLI_SEM, _CLI_MAX
    out = {}
    for provider, cap in _CLI_MAX.items():
        sem = _CLI_SEM.get(provider)
        # ._value counts AVAILABLE slots; in-flight = cap - available
        available = sem._value if sem is not None else cap
        in_flight = max(0, cap - available)
        out[provider] = {
            "max_concurrent": cap,
            "in_flight": in_flight,
            "available": available,
            "saturated": in_flight >= cap,
        }
    return out
