"""
MiniMax API client
==================

Thin httpx-wrapper around the MiniMax REST API
(``https://api.minimax.io/v1/*``). Used by image / video / TTS / voice-clone /
music endpoints in ``ai/routes/*``.

Design notes
------------
- **Single auth source:** ``MINIMAX_MULTIMODAL_API_KEY`` env-var, loaded
  via dotenv from ``/var/www/api-ai.arkturian.com/.env``. Distinct from the
  hypothetical ``MINIMAX_API_KEY`` for the M3 text path (Anthropic-compatible
  endpoint) — see plan post #1046 "Vorbedingungen".

- **No SDK dependency.** MiniMax does offer an official SDK + MCP server
  but we deliberately stay on raw httpx to avoid (a) extra deps that ship
  with their own auth assumptions, (b) MCP-server-as-dep which would add
  an out-of-process bridge for what is fundamentally a REST library.

- **Async only.** All endpoints are async — long-running multimodal
  generations (video, music) MUST not block the FastAPI worker thread.

- **Errors are HTTPException-shaped.** Network errors / non-2xx responses
  get mapped to clean ``HTTPException(status_code, detail=...)`` so the
  caller in ``ai/routes/*`` doesn't have to re-wrap.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")


def _api_key() -> str:
    """Read the pay-as-you-go multimodal key at call time so a rotation
    via ``.env`` + service restart picks up without a process-wide cache.
    """
    key = os.getenv("MINIMAX_MULTIMODAL_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "minimax_api_key_missing",
                "hint": ("MINIMAX_MULTIMODAL_API_KEY not configured. "
                         "Set it in /var/www/api-ai.arkturian.com/.env "
                         "and restart api-ai.service."),
            },
        )
    return key


async def post_json(
    path: str,
    payload: dict,
    timeout: float = 120.0,
) -> dict:
    """POST a JSON body to ``{MINIMAX_BASE_URL}/{path}`` with bearer auth.

    Returns parsed JSON dict on success. Raises HTTPException for any
    non-2xx response — the upstream MiniMax error body is preserved in
    the ``detail`` so the API caller can reason about it (auth, model
    not available, quota, content-policy, etc.).
    """
    url = f"{MINIMAX_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            # httpx.ReadTimeout / ConnectTimeout / PoolTimeout all carry
            # an empty str(e) by default — surfacing only an empty
            # transport_error has burned callers (MusicGenie IACP
            # ad948e77). Always include the exception class name + the
            # configured timeout in the detail so the consumer can
            # distinguish a timeout from a real connection failure.
            error_class = type(e).__name__
            logger.error(
                f"MiniMax POST {path} {error_class} after {timeout}s "
                f"timeout: {str(e) or '(empty)'}"
            )
            is_timeout = isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout))
            raise HTTPException(
                status_code=504 if is_timeout else 502,
                detail={
                    "error": "minimax_upstream_timeout" if is_timeout else "minimax_upstream_unreachable",
                    "path": path,
                    "transport_error": str(e) or "(no message)",
                    "exception_class": error_class,
                    "timeout_sec": timeout,
                    "hint": (
                        "MiniMax took longer than the configured timeout. "
                        "For music_generation the empirical floor is ~300s "
                        "for 60-90s tracks; raise the caller-side timeout."
                    ) if is_timeout else None,
                },
            )
    if r.status_code >= 400:
        try:
            upstream = r.json()
        except Exception:
            upstream = {"raw": r.text[:500]}
        logger.error(f"MiniMax POST {path} → {r.status_code}: {upstream}")
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail={
                "error": "minimax_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
                "path": path,
            },
        )
    try:
        return r.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "minimax_invalid_json_response",
                "path": path,
                "raw": r.text[:500],
            },
        )


async def get_json(path: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
    """GET helper (status polling for video / music generations)."""
    url = f"{MINIMAX_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.get(url, headers=headers, params=params or {})
        except httpx.HTTPError as e:
            error_class = type(e).__name__
            is_timeout = isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout))
            raise HTTPException(
                status_code=504 if is_timeout else 502,
                detail={
                    "error": "minimax_upstream_timeout" if is_timeout else "minimax_upstream_unreachable",
                    "path": path,
                    "transport_error": str(e) or "(no message)",
                    "exception_class": error_class,
                    "timeout_sec": timeout,
                },
            )
    if r.status_code >= 400:
        try:
            upstream = r.json()
        except Exception:
            upstream = {"raw": r.text[:500]}
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail={
                "error": "minimax_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
                "path": path,
            },
        )
    return r.json()


async def post_multipart(
    path: str,
    fields: dict,
    files: dict,
    timeout: float = 120.0,
) -> dict:
    """POST multipart/form-data — used by voice-cloning (audio upload).

    ``files`` follows httpx-multipart convention: ``{name: (filename, bytes, content_type)}``.
    """
    url = f"{MINIMAX_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, data=fields, files=files, headers=headers)
        except httpx.HTTPError as e:
            error_class = type(e).__name__
            is_timeout = isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout))
            raise HTTPException(
                status_code=504 if is_timeout else 502,
                detail={
                    "error": "minimax_upstream_timeout" if is_timeout else "minimax_upstream_unreachable",
                    "path": path,
                    "transport_error": str(e) or "(no message)",
                    "exception_class": error_class,
                    "timeout_sec": timeout,
                },
            )
    if r.status_code >= 400:
        try:
            upstream = r.json()
        except Exception:
            upstream = {"raw": r.text[:500]}
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail={
                "error": "minimax_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
                "path": path,
            },
        )
    return r.json()


def base_resp_failed(body: dict) -> Optional[str]:
    """MiniMax wraps every response in ``base_resp: {status_code, status_msg}``.
    ``status_code: 0`` means success, anything else is a per-request error
    that still came back as HTTP 200.

    Returns the error message if the call failed, ``None`` if it succeeded.
    """
    base = body.get("base_resp") or {}
    code = base.get("status_code", 0)
    if code != 0:
        return base.get("status_msg") or f"minimax base_resp status_code={code}"
    return None
