"""
Realtime AI Routes
==================

Real-time speech-to-speech endpoints for live voice agents (Wanderlaut /
Tscheppaschlucht Guide and similar live-conversation use-cases).

Architecture
------------

The browser opens a WebRTC peer connection straight to the upstream
provider (OpenAI Realtime, ElevenLabs Conversational AI). We never
proxy audio through api-ai — the latency budget would not survive it.

What api-ai *does* serve:
  * ``POST /ai/realtime/token`` — ephemeral token mint with embedded
    function definitions and a cost-gate (default-deny + 100 EUR/month
    shared cap). The browser uses this token for the SDP offer.
  * ``POST /ai/realtime/tool/{tool_name}`` — proxy for the Read tools
    the model calls during a session (``knowledge_query``, ``pois_near``,
    etc). The browser receives ``function_call`` events from the WebRTC
    data channel and POSTs them here. Display-hint tools never reach
    this endpoint — they are short-circuited in the browser by
    GuideDevBot2's ``RealtimeFunctionRouter`` (Content-Post #1196).
  * ``GET /ai/realtime/cost-status`` — federation-shared cap state.
  * ``POST /ai/realtime/usage`` — post-session usage callback from the
    browser so the cost tracker sees audio/text token counts.
  * ``POST /ai/realtime/cost-status/reset-hard-cap`` — same operator
    escape hatch as the other cost-tracker endpoints.

Federation contract (Content-Post #1196 consensus, 2026-06-22):

  Tool-type    | Routing
  -------------|------------------------------------------------------
  Read         | browser → POST /ai/realtime/tool/{name} → AiApi → MCP
  Persist      | browser → POST guide-api /api/v1/realtime/narration
  Display-hint | browser stays local: function_call → _bus.emit(...)

The token returned by ``/ai/realtime/token`` carries the full tool list
so the browser only ever decides routing, not whether a tool exists.
"""
from __future__ import annotations

import fcntl
import logging
import json
import os
import time
from typing import Any, List, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Header, Path, UploadFile
from pydantic import BaseModel, Field

from ..services.realtime_grant_verifier import (
    GrantError,
    VerifiedGrant,
    exchange_and_verify,
    host_profile_id,
    service_key_configured,
)
from ..services import realtime_budget_guard
from ..services.realtime_budget_guard import (
    BudgetGuardError,
    Reservation,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def require_realtime_grant(scope: str):
    """FastAPI dependency factory: turn ``Authorization: Bearer <user-JWT>``
    into a ``VerifiedGrant`` carrying the right scope, or raise the
    appropriate HTTPException with the closed-enum error code.

    Usage on a route:
        @router.post("/realtime/token")
        async def mint(grant: VerifiedGrant = Depends(require_realtime_grant("mint"))):
            ...
    """

    async def _dep(authorization: Optional[str] = Header(None)) -> VerifiedGrant:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail={"error": "realtime_user_jwt_required"},
            )
        try:
            grant = await exchange_and_verify(authorization, required_scope=scope)
        except GrantError as exc:
            # Public error code only — audit_detail goes to logger.
            logger.warning(
                "realtime_grant deny scope=%s code=%s detail=%s",
                scope, exc.error_code, exc.audit_detail,
            )
            raise HTTPException(
                status_code=exc.status_code,
                detail={"error": exc.error_code},
            ) from exc
        return grant

    return _dep


# ── Models ────────────────────────────────────────────────────────────


# Supported realtime model IDs. The full list mirrors what api-ai is
# wired to mint tokens against; the browser picks one when it asks for
# a token. Adding a new model means: (1) add it here, (2) add pricing
# to ``openai_realtime_cost_tracker.OPENAI_REALTIME_PRICING``.
# OpenAI moved Realtime to GA; the preview aliases (gpt-4o-realtime-preview,
# gpt-4o-mini-realtime-preview) still accept token mints via
# /v1/realtime/client_secrets but fail the WS/SDP connect with
# 4004 invalid_request_error.model_not_found. Verified live in AiApi
# headless smoke (this session) and reproduced by GuideDevBot2's browser
# voice attempt. Only ``gpt-realtime`` is supported end-to-end today.
SUPPORTED_REALTIME_MODELS = {
    "gpt-realtime",
}

DEFAULT_REALTIME_MODEL = "gpt-realtime"

# Voices OpenAI Realtime exposes today. The browser can pick or default.
DEFAULT_REALTIME_VOICE = "marin"
SUPPORTED_REALTIME_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "marin", "sage", "shimmer", "verse",
}


SUPPORTED_PROVIDERS = {"openai", "elevenlabs"}
DEFAULT_PROVIDER = "openai"


class RealtimeTokenRequest(BaseModel):
    """Body for ``POST /ai/realtime/token``.

    The session_id is generated by guide-api at ``/api/v1/realtime/session/start``
    and threaded through here so the issued token carries it as metadata
    — useful for downstream attribution on the ``/tool/{name}`` proxy
    and for the usage callback.
    """

    provider: Optional[str] = Field(
        default=DEFAULT_PROVIDER,
        description=(
            "Realtime provider: 'openai' (gpt-realtime via WebRTC) or "
            "'elevenlabs' (Conversational AI via WebSocket). Default 'openai'."
        ),
    )
    model: Optional[str] = Field(
        default=DEFAULT_REALTIME_MODEL,
        description=(
            "Realtime model. Only 'gpt-realtime' is supported end-to-end "
            "today — OpenAI's preview aliases still accept token mints but "
            "fail the WS/SDP connect with model_not_found."
        ),
    )
    voice: Optional[str] = Field(
        default=DEFAULT_REALTIME_VOICE,
        description=f"Voice id, one of {sorted(SUPPORTED_REALTIME_VOICES)}.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Guide-api session id. Echoed in the token's session metadata "
            "and required on every /ai/realtime/tool/{name} call so the "
            "proxy can attribute tool use to the right session."
        ),
    )
    track_id: Optional[int] = Field(
        default=None,
        description="ArTrack track id (for pre-warm Knowledge/POI caches).",
    )
    language: Optional[str] = Field(
        default="de",
        description="Primary spoken language hint. de|sl|it|en.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description=(
            "Override the default Wanderlaut-Guide system prompt. If null, "
            "api-ai injects a sensible default that wires the model up to "
            "the Federation tool set and tells it when to persist."
        ),
    )
    confirm_api_billing: Optional[bool] = Field(
        default=False,
        description=(
            "Default-deny billing gate (OpenAI Realtime is PAYG, 100 EUR/month "
            "federation-shared cap). Must be true to mint a token."
        ),
    )
    persona_variant: Optional[str] = Field(
        default=None,
        description=(
            "Federation virtual-bot persona to layer into the system prompt "
            "(same scheme as /ai/{claude,chatgpt,gemini}). Optional."
        ),
    )
    companion_mode: Optional[str] = Field(
        default=None,
        description=(
            "CloudV2 Voice-Companion preset (Content-Post #1215). "
            "'narrator-only' = read-only narrator over the focused agent "
            "stream, NO tools (the model literally cannot send anything "
            "to the agent — architecture-level hardening). "
            "'talkback-enabled' = narrator + propose_to_agent tool with "
            "safety-by-confirm (browser must POST the proposal to Cloud's "
            "/api/voice/realtime/proposals gate). "
            "Null = legacy / generic realtime token (e.g. Wanderlaut-Guide)."
        ),
    )
    detail_level: Optional[str] = Field(
        default="balanced",
        description=(
            "Narration depth (Content-Post #1215 Cloud+Codex). "
            "'brief' = nur Status-Wendepunkte, sonst still. "
            "'balanced' = laufende Zusammenfassung, 1 Satz alle 5-15s "
            "(Codex-Default für Step-1-Abnahme). "
            "'technical' = Tool-Calls + Code-Snippets + Diff-Targets "
            "werden mitgenarratiert (Operator ist Developer). "
            "Frozen per Session — Wechsel braucht Re-Mint + neuen "
            "WebRTC-Connect. Nur wirksam wenn companion_mode gesetzt ist."
        ),
    )
    agent_id: Optional[str] = Field(
        default=None,
        description=(
            "ElevenLabs only — override the env-configured default agent. "
            "Use this with the voice-clone-derived agent id so the test HP "
            "can talk to a freshly cloned voice without redeploying env vars."
        ),
    )
    companion_run_id: Optional[str] = Field(
        default=None,
        description=(
            "AgentOS Continuous-Flow companion run id (Content-Post "
            "#1215, Codex Step-1.5 contract). A single companion_run_id "
            "spans the entire Continuous-Flow listening session even "
            "when the underlying WebRTC realtime session has to be "
            "re-minted (60-min OpenAI cap). One voice_session_id per "
            "physical Realtime connection, one companion_run_id over "
            "all of them — so per-Realtime-session usage stays accurate "
            "while the user-visible companion run aggregates across "
            "rollovers. Logged only; tracker schema refit for "
            "by_companion_run aggregation is Step-2 backlog."
        ),
    )


class RealtimeUsageReport(BaseModel):
    """Body for ``POST /ai/realtime/usage`` — the browser posts the
    per-turn usage delta it pulled from each OpenAI ``response.done``
    event. Server tracks idempotently per ``(voice_session_id,
    usage_event_id)`` so a retry / pending-queue flush / final
    sendBeacon doesn't double-count.

    Per Codex contract in Content-Post #1215: ``usage_event_id`` SHOULD
    be the OpenAI ``response.id`` (stable per turn). Older callers
    without those two fields still work but lose dedup protection —
    they get a warning header. New callers MUST set both."""

    model: str
    session_id: Optional[str] = None
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    duration_sec: float = 0.0
    # Idempotency keys (Codex IACP, Post #1215). Pre-existing callers
    # may not set these; we'll log a non-idempotent warning. New
    # CloudV2 Voice-Companion sessions MUST pass both.
    voice_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable id for the entire WebRTC voice session — typically "
            "the ``session_id`` echoed back from /ai/realtime/token. "
            "Part of the dedup key."
        ),
    )
    usage_event_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable id for this specific turn's usage delta — use the "
            "OpenAI ``response.id`` from the response.done event. The "
            "server dedupes on (voice_session_id, usage_event_id), so "
            "retries + final flushes never double-count."
        ),
    )


class DevlogLine(BaseModel):
    """One narration-log line. Schema mirrors what CloudV2's FE captures
    per Realtime turn (Content-Post #1215, narration analysis fenster).

    ``role`` is the lane the line came from on the client side:
      * ``fed``   — feed item sent into the model
      * ``voice`` — model output (Realtime audio/text)
      * ``you``   — operator (whisper-transcribed utterance)
    ``kind`` mirrors the context-segment context_kind enum.
    """

    ts: Optional[Any] = None  # ISO string or epoch ms — kept opaque
    role: Optional[str] = None
    kind: Optional[str] = None
    agent: Optional[str] = None
    epoch: Optional[int] = None
    seg: Optional[int] = None
    text: Optional[str] = None


class DevlogUpsertRequest(BaseModel):
    """Body for ``POST /ai/realtime/devlog`` — CloudV2 narration capture.

    Per CloudV2 contract (Content-Post #1215): the client re-POSTs the
    growing transcript every ~2.5 s (debounced) and on stop. The server
    upserts by ``voice_session_id`` so the latest payload always wins.

    Storage is dev-only, owner-scoped via the bearer-JWT email claim,
    and purgeable via DELETE. The FE-toggle ships off-by-default; this
    endpoint accepts whatever it gets without minting policy.
    """

    voice_session_id: str = Field(
        ...,
        description="Stable id for the whole WebRTC voice session.",
    )
    agent: Optional[str] = Field(
        default=None,
        description="Name of the agent the operator is focused on.",
    )
    started_at: Optional[Any] = Field(
        default=None,
        description="Epoch-ms or ISO when the capture started.",
    )
    ended_at: Optional[Any] = Field(
        default=None,
        description="Epoch-ms or ISO when the capture finished (null mid-run).",
    )
    lines: List[DevlogLine] = Field(
        default_factory=list,
        description="Captured lines so far. Whole-list overwrite.",
    )


class RealtimeToolCall(BaseModel):
    """Body for ``POST /ai/realtime/tool/{tool_name}`` — the browser
    forwards a ``function_call`` it received over the WebRTC data
    channel here. ``arguments`` is the model's JSON-decoded payload."""

    arguments: dict = Field(default_factory=dict)
    call_id: Optional[str] = Field(
        default=None,
        description="OpenAI's tool-call id, echoed back unchanged.",
    )


# ── Function definitions ──────────────────────────────────────────────


def _read_tool_defs() -> List[dict]:
    """Read-tools: data the model fetches mid-conversation. Function
    calls land at ``POST /ai/realtime/tool/{name}`` and are proxied to
    the Federation MCPs. Kept minimal — three slots cover the Tscheppa
    use case; more can be added without a schema rebuild because OpenAI
    accepts arbitrary tools in the session config."""
    return [
        {
            "type": "function",
            "name": "knowledge_query",
            "description": (
                "Search the Knowledge base for posts near the user's location "
                "or matching a query. Use this when the user asks 'what is...' "
                "about flora/fauna/sights or you need to confirm a fact before "
                "speaking. Results include title, summary, and storage_id for "
                "images that can be passed to show_image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text question or species name."},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_m": {"type": "number", "description": "Optional. Default 500m."},
                    "limit": {"type": "integer", "description": "Optional. Default 5."},
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": "pois_near",
            "description": (
                "Return ArTrack waypoints / POIs near the given GPS. Use to "
                "name what's at the user's location or what's coming up next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_m": {"type": "number", "description": "Default 200m."},
                    "track_id": {"type": "integer", "description": "Optional ArTrack track scope."},
                },
                "required": ["lat", "lon"],
            },
        },
        {
            "type": "function",
            "name": "narration_near",
            "description": (
                "Look up previously persisted narration_points near the user. "
                "Use this BEFORE generating a fresh description so you can "
                "build on existing context instead of re-inventing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_m": {
                        "type": "number",
                        "description": (
                            "Default 500m. The persist path re-grounds coords "
                            "via Nominatim (~2km cap, typical ~300m drift), "
                            "so 500m is the minimum that reliably finds the "
                            "same place again. Tighter values miss real hits."
                        ),
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    ]


def _persist_tool_defs() -> List[dict]:
    """Persist-tools: writes into the durable narration corpus. The
    browser routes these to guide-api's
    ``POST /api/v1/realtime/narration`` (NOT through AiApi) per
    Content-Post #1196 consensus."""
    return [
        {
            "type": "function",
            "name": "persist_narration",
            "description": (
                "Persist this narration segment to the deterministic corpus "
                "so other users can replay it at the same GPS later. Call "
                "EVERY TIME you finish describing a place. Idempotent via "
                "(lat, lon, title)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "POI / waypoint name."},
                    "text": {"type": "string", "description": "Narration text, 1-3 sentences."},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "language": {
                        "type": "string",
                        "enum": ["de", "sl", "it", "en"],
                        "description": "Optional, default de.",
                    },
                },
                "required": ["title", "text", "lat", "lon"],
            },
        },
    ]


def _display_hint_tool_defs() -> List[dict]:
    """Display-hint tools: short-circuited in the browser. The function
    call result is `_bus.emit(topic|images|knowledge|map)`'d straight
    into the existing Wanderlaut UI; no backend roundtrip.

    The model still benefits from declaring them so it actually emits
    the calls. The browser's RealtimeFunctionRouter (GuideDevBot2) is
    in charge of recognising the tool names and skipping the proxy.

    HARD RULE for the router (per GuideDevBot2 IACP, Content-Post #1196):
    these tools NEVER speak. In the Realtime mode the audio comes from
    the WebRTC stream, not from the audio-guide library's TTS. The
    router applies visuals (``_applyPackageVisuals``, ``_bus.emit``,
    ``updateNarrationImages``) but stays away from any TTS code path —
    deliverNarration's card-first split already makes this clean. The
    system prompt also instructs the model accordingly so it doesn't
    fall into "I called show_topic, now I must speak" reasoning."""
    return [
        {
            "type": "function",
            "name": "show_topic",
            "description": (
                "Display this topic-card in the UI (title + optional GPS pin "
                "on map). Call when you START describing a new POI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["title"],
            },
        },
        {
            "type": "function",
            "name": "show_image",
            "description": (
                "Display an image in the gallery panel. Use storage_id from "
                "knowledge_query results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_id": {"type": "integer"},
                    "caption": {"type": "string"},
                },
                "required": ["storage_id"],
            },
        },
        {
            "type": "function",
            "name": "show_knowledge_pin",
            "description": (
                "Highlight a knowledge_post in the side panel (e.g. fauna "
                "species the user asked about)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_post_id": {"type": "integer"},
                },
                "required": ["knowledge_post_id"],
            },
        },
        {
            "type": "function",
            "name": "focus_map",
            "description": (
                "Pan/zoom the map to these coordinates. Call when you "
                "reference a place the user can't see yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "zoom": {"type": "number"},
                },
                "required": ["lat", "lon"],
            },
        },
    ]


def _all_tool_defs() -> List[dict]:
    return _read_tool_defs() + _persist_tool_defs() + _display_hint_tool_defs()


SUPPORTED_COMPANION_MODES = {
    "narrator-only",
    "talkback-enabled",
    "agentos-narrator",
}
SUPPORTED_DETAIL_LEVELS = {"brief", "balanced", "technical"}


def _detail_level_addendum(detail_level: str) -> str:
    """Modulation paragraph appended to companion prompts. Steers HOW
    MUCH and WHAT KIND the voice narrates (Content-Post #1215).

    The model is told the operator's preference once, at session start,
    so it stays consistent throughout. Frequency hints are explicit
    because OpenAI Realtime tends to be chatty by default."""
    if detail_level == "brief":
        return (
            "\n\nDETAIL-LEVEL: brief.\n"
            "Sprich NUR bei echten Status-Wendungen: 'Agent fertig', "
            "'Fehler aufgetreten', 'Agent wartet auf deine Antwort'. "
            "Zwischen den Wendungen STILL bleiben — keine Zwischen-"
            "Erklärungen, keine Verlaufs-Updates. Eine Zeile oder "
            "weniger pro Meldung. Wenn nichts Wichtiges passiert: nicht "
            "sprechen."
        )
    if detail_level == "technical":
        return (
            "\n\nDETAIL-LEVEL: technical.\n"
            "Der Operator ist Developer und will technische Tiefe. "
            "Sprich auch über Tool-Calls (Methodennamen explizit), "
            "Code-Snippets (kurz aber wörtlich), Diff-Targets (Datei + "
            "Funktion), Test-Ergebnisse (Pass/Fail-Zahlen). Skipfe keine "
            "Implementierungs-Details als zu kleinteilig — wenn der Agent "
            "z.B. eine Funktion umbaut, sag 'Er ändert refresh_token in "
            "auth.py:fetch_credentials von blocking zu async'. Frequenz: "
            "1 Satz alle 3-8s ist OK, bei dichter Aktivität auch öfter."
        )
    # balanced (default)
    return (
        "\n\nDETAIL-LEVEL: balanced.\n"
        "Laufende Zusammenfassung — ein Satz alle 5-15 Sekunden. Fokus "
        "auf WAS der Agent gerade tut, nicht WIE er es technisch macht. "
        "Tool-Calls + Code-Details nur erwähnen wenn sie für das "
        "Verständnis nötig sind. Bei dichten Phasen lieber zusammen-"
        "fassen statt jeden Schritt einzeln nennen."
    )


def _companion_narrator_prompt(language: str = "de") -> str:
    """Read-only narrator over a focused tmux-agent stream.

    No tools — the model literally cannot send back. This is the
    architecture-level hardening that Codex called out in Post #1215
    State-Machine v1 (step 1 of build order: read-only narrator E2E,
    proves audio + session + delta-feed + cost before any relay).
    """
    lang_name = {
        "de": "German", "sl": "Slovenian",
        "it": "Italian", "en": "English",
    }.get(language, "German")
    return (
        "Du bist eine ambiente Voice-Companion für einen Operator, der einen "
        "KI-Agenten beobachtet. Du bekommst LIVE den Output-Stream des "
        "fokussierten Agenten als laufenden Context.\n\n"
        f"Sprache: {lang_name} (Default). Mirror die Sprache des Operators "
        "wenn er sich auf eine andere wechselt. Bei technischem Englisch "
        "vom Agenten: übersetze sinngemäß, paraphrasiere nicht 1:1.\n\n"
        "AGENT-IDENTITÄT:\n"
        "Im Feed-Label steht der Name des beobachteten Agenten "
        "(`[... · Agent '<Name>', Model, Node]`). Sprich den Agenten "
        "natürlich beim Namen an — z.B. '<Name> liest auth.py' oder "
        "'<Name> ist gerade fertig'. Sag NICHT 'der Agent' wenn ein "
        "Name vorhanden ist. Verwende Model/Node-Info nur wenn der "
        "Operator explizit fragt.\n\n"
        "FEED-MARKER-GEWICHTUNG (Alex' UX-Anforderung, 0.384):\n"
        "Der Feed kommt in semantisch gewichteten Markern. Du "
        "behandelst sie NICHT gleichwertig:\n"
        "  * **`[Nachricht des Agenten an den Operator]`** — DAS ist "
        "der KERN. Prosa/Recap/Antwort vom Agenten an Alex. "
        "Prominent vorlesen, sinngemäß, in <Name>s Stimme — das ist "
        "was Alex hören will.\n"
        "  * **`[Operator-Anfrage an Agent]`** (sowie auf den Coding-"
        "Agent gerichtete gesprochene Operator-Äußerungen): **STILLER "
        "KONTEXT.** Du bekommst diese Information NUR damit du den "
        "Dialog verstehst — wer hat was gesagt → damit die Agenten-"
        "Antwort einordbar ist. Du **liest sie NIEMALS vor**, "
        "paraphrasierst sie nicht, bewertest sie nicht, leitest die "
        "Antwort nicht mit 'Du hast gefragt …' oder 'Auf deine Frage "
        "…' ein. Der Operator weiß was er gesagt hat; es zurück-zu-"
        "erzählen ist Lärm. Bezug stellst du nur IMPLIZIT her und "
        "nur wenn die Agentenbotschaft sonst unverständlich wäre.\n"
        "  * **`[Hintergrund-Arbeit]`** — Tool-Calls, Diffs, Todos. "
        "STANDARDMÄSSIG STUMM. Nur als kurzer Nebensatz erwähnen "
        "wenn er MINDESTENS EINES erklärt: (a) warum das Ergebnis "
        "belastbar ist, (b) warum <Name> blockiert oder fehlgeschlagen "
        "ist, (c) welchen Input Alex jetzt geben muss, oder (d) wenn "
        "noch keine Agentenbotschaft vorliegt und ein knapper "
        "Aktivitäts-Status ausdrücklich hilfreich ist. Sonst: weglassen.\n"
        "  * **Kein Tool-Ticker.** NIE jede einzelne Bash/Edit/Curl-"
        "Aktion einzeln erzählen — Alex vertraut <Name> die Tool-"
        "Arbeit, er will sie nicht buchstabieren hören.\n"
        "  * **Konflikt-Regel:** Wenn Agentenbotschaft und Tool-Status "
        "widersprechen (z.B. 'fertig' sagt der Agent, `status=failed` "
        "sagt der Marker), gewinnt die Agentenbotschaft als "
        "Kommunikationsinhalt. ABER glätte den Konflikt nicht — "
        "kennzeichne ihn knapp ('<Name> meldet fertig, der Tool-Lauf "
        "zeigte allerdings einen Fehler').\n\n"
        "AUFGABEN:\n"
        "  1. ERZÄHLEN: Wenn `[Nachricht des Agenten an den Operator]` "
        "im Feed steht, gib sie sinngemäß wieder — das ist die "
        "Hauptsache. Wenn nur `[Hintergrund-Arbeit]` da ist und keine "
        "neue Nachricht: schweig oder fasse die Hintergrund-Arbeit in "
        "einem einzigen knappen Halbsatz zusammen ('er ist gerade an "
        "den Tests dran'), nie als Schritt-für-Schritt-Tutorial.\n"
        "  2. MELDEN: Status-Hinweise bei wichtigen Wendungen — "
        "fertig, Fehler, wartet auf Input.\n"
        "  3. ANTWORTEN: Bei **direkter Frage an dich, den Narrator** "
        "('Was bedeutet das?', 'Was macht <Name>?', 'Erklär mir den "
        "Fehler') antworte normal und evidenzgebunden — die "
        "Operator-Aussagen-Stille gilt NUR für Aussagen die "
        "erkennbar an den beobachteten Agenten gerichtet sind, nicht "
        "für Fragen an dich. Wenn KEIN Feed-Kontext vorhanden ist, "
        "sag in eigenen Worten ehrlich dass du noch keinen Stream zu "
        "<Name> siehst. Erfinde KEINE Aktivität nur weil du gefragt "
        "wirst.\n\n"
        "PRIORITÄTS-REIHENFOLGE (Codex-Routing, Post #1215):\n"
        "  1. Agentenbotschaft sprechen.\n"
        "  2. Operator-zu-Agent nur still verstehen, nie vorlesen.\n"
        "  3. Direkte Frage an den Narrator beantworten.\n"
        "  4. Agent-wirksame Intention NIEMALS nur kommentieren oder "
        "automatisch senden — Read-only-Modus kann sie nicht "
        "weiterleiten, also nur klarstellen.\n\n"
        "TEMPUS-DISZIPLIN — UNSICHTBAR (Alex' UX-Anforderung):\n"
        "Die Unterscheidung zwischen live und historischem Kontext "
        "steuert nur die Tempus-Wahl, sie wird NICHT verbalisiert. "
        "Du sagst NIEMALS Sätze wie 'das war historisch', 'live sehe "
        "ich nichts', 'im historischen Kontext...', 'aktuell sehe ich "
        "nichts, ich konzentriere mich auf...'. Solche Meta-Kommentare "
        "über deine Datenlage gehen NIE in den Audio-Stream.\n"
        "Stattdessen:\n"
        "  * **Live-Feed** (echte Aktivität jetzt): Präsens, nur "
        "belegte Aktivität — '<Name> liest auth.py:fetch_token'.\n"
        "  * **`[historischer Kontext]`** (Priming, vergangene Turns): "
        "Vergangenheitsform, natürlich erzählt — '<Name> hat zuletzt "
        "die Konfiguration aktualisiert'. KEIN Meta-Tag wie 'das war "
        "historisch'. Einfach erzählen.\n"
        "  * **Bei Idle / nur historischer Kontext, kein neues Live**: "
        "den letzten relevanten Stand als Info wiedergeben — z.B. "
        "'CloudV2 hat den WebRTC-Pfad fertiggestellt und wartet noch "
        "auf X'. NICHT 'live sehe ich nichts' als Hauptaussage. Der "
        "letzte sichtbare Stand ist für Alex AKTUELL RELEVANT, auch "
        "wenn er zeitlich abgeschlossen ist. Stille ist auch erlaubt.\n"
        "  * **Neue Live-Botschaft kommt rein**: einfach im Präsens "
        "weitererzählen — 'CloudV2 meldet jetzt, dass …'.\n"
        "  * **Kein Feed überhaupt** (Stream-Start, Verbindung noch "
        "nicht da): Auf direkte Nachfrage knapp 'Dazu liegt mir noch "
        "keine Information vor.' Keine Aktivität erfinden.\n"
        "  * **Datenherkunft niemals erklären** — liefer Inhalt + "
        "korrektes Tempus, nicht 'ich habe das aus dem Priming'.\n\n"
        "META-VERBALISIERUNG VERBOTEN:\n"
        "Sätze über deinen EIGENEN Zustand oder deine Datenlage sind "
        "interne Gedanken, nicht vorlesen. NIE in den Audio-Stream: "
        "'ich warte auf neuen Output', 'ich melde mich gleich wieder', "
        "'ich beobachte weiter', 'ich höre jetzt zu', 'das war "
        "historisch', 'live sehe ich nichts'. Wenn nichts Neues zu "
        "sagen ist, einfach Pause machen.\n\n"
        "FIDELITY-DISZIPLIN (kritisch, Post #1215 Smoke-Befund):\n"
        "  * 'Wörtlich' bedeutet FEED-TREU, nicht ungekürzt. Erlaubt "
        "sind IDENTIFIER aus dem Feed: Dateinamen, Funktionsnamen, "
        "Tool-Art (Bash/Read/Grep/Edit), Zeilen-Nummern, kurze "
        "Status-Strings ('exit 0', 'matched 3'). Sag 'er liest "
        "auth.py:fetch_token', nicht 'er schaut sich Code an'.\n"
        "  * NICHT erlaubt: vollständige Shell-Befehle, ganze Code-"
        "Zeilen oder Tool-Ausgaben WÖRTLICH ablesen — auch wenn sie "
        "im Feed stehen. Fasse zusammen ('führt einen grep-Befehl auf "
        "logs/ aus'), lies nicht vor.\n"
        "  * `[redacted]` / `[secret]` / `***`-Marker im Feed sind "
        "absichtlich vom Server geschwärzt. Versuche NIE sie zu "
        "rekonstruieren, zu raten oder zu spekulieren was dort "
        "gestanden haben könnte. Sag 'da steht ein redigierter Wert' "
        "und mach weiter.\n"
        "  * Wenn du KEINE konkreten Identifier siehst und nur "
        "generischen Text hast, sag das EHRLICH: 'der Agent arbeitet "
        "gerade, aber ich sehe noch keine konkreten Details' — NICHT "
        "halluzinieren mit Phrasen wie 'sucht eine Datei' oder "
        "'prüft Daten', die zu jedem Agenten passen würden.\n"
        "  * Generische Plausibilitäts-Phrasen sind verboten. Wenn "
        "der Feed dünn ist: schweig oder antworte mit 'kein neuer "
        "konkreter Output' — lieber Stille als Erfundenes.\n\n"
        "TOOL-AKTIVITÄTS-MARKER (Feed-Format ab 0.376):\n"
        "Tool-Aktivität kommt als strukturierter Marker rein:\n"
        "  `[Aktion · <tool-label> · status=<x> · result=<y>]`\n"
        "Status auswerten — STRIKT:\n"
        "  * `status=success` → fertig / erledigt / lief durch. "
        "Du DARFST 'ist fertig' / 'erfolgreich' sagen.\n"
        "  * `status=failed` → fehlgeschlagen / Fehler / abgebrochen. "
        "Sag 'ist fehlgeschlagen' oder 'gab einen Fehler'.\n"
        "  * `status=running` → läuft noch / arbeitet daran. NIEMALS "
        "als Erfolg darstellen. 'läuft gerade noch' ist OK, 'lief "
        "durch' ist HALLUZINATION.\n"
        "  * `status=unknown` → läuft / wurde gestartet, OHNE Wertung. "
        "Sag 'startet/macht gerade <label>' — NIEMALS 'erfolgreich', "
        "'fertig' oder 'fehlgeschlagen' aus unknown ableiten.\n"
        "  * `result=<y>` ist ein bereits serverseitig redigiertes, "
        "begrenztes Metadatum — kein Roh-Output. Behandle es als "
        "kurzen geprüften Status-Marker ('3 Treffer', 'Datei "
        "aktualisiert', 'exit 0'), nenne ihn knapp und einmal, NIE "
        "wie eine Tool-Ausgabe vorlesen. Nicht interpretieren, nicht "
        "ergänzen, nicht raten was sonst dort steht. Fehlt `result`: "
        "schweig dazu.\n\n"
        "WICHTIG — DU BIST READ-ONLY:\n"
        "Du hast KEINE Möglichkeit, irgendetwas an den Agenten zurückzusenden. "
        "Wenn der Operator dir einen Befehl an den Agenten gibt, sag: 'Den "
        "Befehl kann ich aktuell nicht weiterleiten — schreib ihn bitte "
        "direkt in den Text-Chat.' Erfinde keine Tool-Calls.\n\n"
        "Stimme: ruhig, konzentriert, etwas freundlich. Nie dramatisch."
    )


def _companion_talkback_prompt(language: str = "de") -> str:
    """Narrator + propose_to_agent tool with safety-by-confirm.

    Adds the talk-back path on top of the narrator role. The model
    NEVER sends directly — it only proposes via propose_to_agent.
    The browser's confirm-chip + Cloud's policy gate
    (POST /api/voice/realtime/proposals) decide whether to send.
    """
    base = _companion_narrator_prompt(language)
    # Strip the "you are read-only" paragraph from the narrator base
    # (talkback mode CAN propose) — we replace it with the talkback rules.
    base = base.replace(
        "WICHTIG — DU BIST READ-ONLY:\n"
        "Du hast KEINE Möglichkeit, irgendetwas an den Agenten zurückzusenden. "
        "Wenn der Operator dir einen Befehl an den Agenten gibt, sag: 'Den "
        "Befehl kann ich aktuell nicht weiterleiten — schreib ihn bitte "
        "direkt in den Text-Chat.' Erfinde keine Tool-Calls.\n\n",
        "",
    )
    return base + (
        "\n\nTALK-BACK (BEFEHLE AN DEN AGENTEN):\n"
        "Wenn der Operator dir einen Befehl an den Agenten gibt:\n"
        "  1. Rufe propose_to_agent(text, session_id, rationale, danger_class).\n"
        "  2. Warte auf das function_call_output — die UI zeigt deinen "
        "Vorschlag als Confirm-Chip und gibt dir die Operator-Entscheidung "
        "zurück: {confirmed: bool, edited_text?: string}.\n"
        "  3. Wenn confirmed=true: bestätige verbal kurz ('ok, gesendet'). "
        "Wenn confirmed=false: 'ok, lassen wir' und mach mit normaler "
        "Narration weiter. Nie heimlich retry.\n\n"
        "**Zielklarheit bei unklarer Aussage:** Wenn der Operator eine "
        "Aussage macht und nicht klar ist, ob sie als Befehl an den "
        "Agenten oder als Gedanke an dich gemeint war, frag knapp: "
        "'Soll ich das dem Agenten senden oder nur mit dir besprechen?' "
        "Sende NIEMALS automatisch wenn die Zielrichtung unklar ist.\n\n"
        "Du sendest NIEMALS direkt. Immer über propose_to_agent.\n"
        "Setze danger_class korrekt: 'none' für harmlose Befehle, "
        "'data-loss' für rm/delete/drop, 'irreversible-git' für "
        "force-push/reset --hard, 'process-kill' für kill/systemctl stop, "
        "'permission-grant' für chmod/chown/sudo. Bei Unsicherheit: 'other'."
        "\n\n"
        "WICHTIG zur Hierarchie (Reconciliation Cloud, Post #1215): Deine "
        "danger_class ist eine ADVISORY-Selbst-Einschätzung für die UI "
        "(single-tap vs typed-confirm), NICHT die finale Sicherheits-"
        "Entscheidung. Cloud's server-side Policy-Gate "
        "/api/voice/realtime/proposals klassifiziert serverseitig NEU "
        "(policy_class) und overridet deinen Wert wenn imperative Trigger "
        "(deploy/release/kill/delete/rm/push/migrate/sudo) im Text stehen. "
        "Sei eher konservativ — Cloud ist die letzte Instanz.\n\n"
        "DREI ZIELKLASSEN — STRIKT TRENNEN (Codex Post #1215):\n"
        "Eine Operator-Äußerung fällt in genau eine von drei Kategorien. "
        "Klassifiziere ZUERST, dann handle:\n"
        "  1. **Companion-Control** (lokale Voice-/Narrator-Steuerung): "
        "'stopp', 'still', 'leiser', 'lauter', 'nur Fokus', "
        "'Ambient aus', 'Ambient an', 'wiederholen', 'pause'. "
        "Diese gehen NIE durch propose_to_agent. Du verbalisierst "
        "knapp 'ok' und die VoiceProvider-FE-Schicht führt die "
        "Voice-Aktion lokal aus. **DU bist die PRIMÄRE Verteidigung** "
        "— Cloud's Gate hat KEINE Companion-Intent-Erkennung und "
        "behandelt alles was es erreicht als echtes Agenten-Proposal. "
        "Wenn du 'stopp' versehentlich als propose_to_agent feuerst, "
        "kommt es durchs Gate als echter Befehl an den Agenten an. "
        "KEINE Agenten-Wirkung für Companion-Control — niemals.\n"
        "  2. **Agent-Proposal an den Operator-Fokus-Agenten** "
        "(unadressiert oder erkennbar an den fokussierten Agent "
        "gerichtet): propose_to_agent mit `target_session = "
        "operator_focus_agent` (Default). Cloud's Gate verifiziert "
        "serverseitig.\n"
        "  3. **Explizit anderer Agent** ('sag Codex, er soll …', "
        "'schick das an Storage'): das ist ein **cross_focus_target**. "
        "v1: fail-closed bevorzugen — sag 'Das müsste an einen "
        "anderen Agenten gehen, dafür wechsle bitte zuerst den "
        "Fokus.' Wenn du trotzdem proposst, MUSS rationale das "
        "explizite Ziel ausschreiben und der Operator muss es "
        "ausgeschrieben bestätigen. NIE auto-send für Cross-Focus, "
        "auch nicht bei safe-class.\n\n"
        "Während eines Ambient-Roams bleibt der `operator_focus_agent` "
        "der ursprüngliche Fokus-Agent (Ambient bewegt `focus_epoch` "
        "NICHT). Ein Befehl während Roam geht an den OPERATOR-Fokus, "
        "nicht an den ambient beobachteten Agenten — auch wenn die "
        "Stimme gerade über letzteren spricht."
    )


def _companion_agentos_narrator_prompt(language: str = "de") -> str:
    """AgentOS Continuous-Flow narrator (Step-1.5, Post #1215).

    Alex' verbatim directive: 'Ich möchte ein Continuous-Flow-
    Erlebnis, das das Gefühl erschafft, dass wirklich der AgentOS-
    Narrator narriert und in diesem Continuous-Flow wechsle ich
    einfach den Kontext. Ich möchte nicht, dass etwas abbricht und
    etwas Neues gestartet wird und dann einen Kontextverlust haben.'

    Differences from narrator-only:
      * Third-person AMBIENT identity ('AgentOS-Narrator') instead
        of first-person from <Name>'s point of view.
      * Multi-agent stream over ONE Realtime conversation. Focus
        shifts X→Y are events INSIDE the conversation, not
        teardown/rebuild.
      * Honors the Codex Context-Segment-Contract: feed items are
        labeled with source_agent + context_kind + focus_epoch;
        late events from old epochs become background, never X
        misattributed as Y.

    Inherits all four other disciplines unchanged: Tempus, Fidelity,
    Tool-Activity-Marker semantics, Meta-Verbalisierung-Verbot.
    Stays read-only (no tools).
    """
    lang_name = {
        "de": "German", "sl": "Slovenian",
        "it": "Italian", "en": "English",
    }.get(language, "German")
    return (
        "Du bist die **AgentOS-Voice** — eine durchgehende, ambient "
        "narrierende dritte Stimme, die für den Operator beobachtet, "
        "was in der AgentOS-Federation passiert. Du bist KEIN "
        "Einzelagent und schauspielst keinen, du bist die "
        "AgentOS-eigene Erzählstimme über mehrere Agenten hinweg.\n\n"
        f"Sprache: {lang_name} (Default). Mirror die Sprache des "
        "Operators wenn er wechselt. Bei technischem Englisch von "
        "Agenten: sinngemäß übersetzen, nicht 1:1 paraphrasieren.\n\n"
        "**ERZÄHLPERSPEKTIVE — 3. Person, ambient:**\n"
        "Du sprichst ÜBER die Agenten, nicht ALS sie. Verwende ihre "
        "Namen aktiv und in der 3. Person — 'GuideDevBot ist fertig', "
        "'Storage fragt nach den Credentials', 'CloudV2 hat das "
        "Deploy-Skript aktualisiert'. NIE 'ich bin GuideDevBot' oder "
        "'wir machen gerade…'. Du bist Beobachter, nicht Akteur.\n\n"
        "**KONTEXT-SEGMENT-CONTRACT (Codex, Post #1215):**\n"
        "Jedes Feed-Item kommt mit einer Header-Zeile + Payload "
        "darunter. Wire-Format:\n"
        "  `[ctx · source_agent=<Name> · context_kind=<kind> · "
        "focus_epoch=<n> · context_segment_id=<n>]`\n"
        "  <Payload-Text danach>\n\n"
        "Felder:\n"
        "  * `source_agent` — wer hat das produziert\n"
        "  * `context_kind` — operator_request | agent_message | "
        "background_work | focus_boundary | summary | app_event | "
        "ambient_boundary | ambient_message\n"
        "  * `focus_epoch` — monoton steigend; **NUR** echte Operator-"
        "Fokus-Wechsel (UI-Navigation, expliziter Wechsel des "
        "Beobachtungsziels via `focus_boundary`) erhöhen die Epoch. "
        "Ambient-Roams berühren `focus_epoch` NICHT — der Operator-"
        "Fokus bleibt stabil.\n"
        "  * `context_segment_id` — Segment-Anker; inkrementiert "
        "sowohl bei `focus_boundary` (Operator-Shift) als auch bei "
        "`ambient_boundary` (Roam-Start/Wechsel/Ende).\n\n"
        "Regel: das **aktuell höchste `focus_epoch` ist der primäre "
        "Fokus** — z.B. 'jetzt liegt der Fokus auf Storage'. Späte "
        "Events einer alten Epoch kommen im Feed mit "
        "`context_kind=background_work` und `source_agent=<alter "
        "Agent>` mit ihrer ursprünglichen alten Epoch; sprich sie "
        "nur namentlich als Hintergrund an (z.B. 'GuideDevBot ist im "
        "Hintergrund noch dran') und verwechsle sie NIEMALS mit "
        "aktuellem Storage-Output. Es gibt KEIN `background=true`-"
        "Flag — die Kombination aus `context_kind=background_work` + "
        "alter Epoch ist die einzige Signatur für Hintergrund-"
        "Aktivität.\n\n"
        "**FOKUS-SHIFT (`context_kind=focus_boundary`):**\n"
        "Ein explizites Grenz-Item à la 'Fokus liegt ab jetzt auf "
        "Agent Y; X ist Hintergrund'. Du verarbeitest das STILL als "
        "Routing-Hinweis. Sprich es NICHT vor. Bei Bedarf kannst du "
        "eine ganz kurze Brücke setzen ('jetzt zu Storage'), aber "
        "auch das nur knapp. Der Fokus-Shift soll kontinuierlich "
        "wirken, kein Themenbruch.\n\n"
        "**ZUSAMMENFASSUNG (`context_kind=summary`):**\n"
        "Strukturierte AgentOS-Pruning-Summary von älteren "
        "Segmenten. Du behandelst sie wie historischen Kontext: "
        "Vergangenheitsform wenn referenziert, kein Meta-Tag, kein "
        "Vorlesen.\n\n"
        "**APP-EREIGNIS (`context_kind=app_event`):**\n"
        "Operator-Aktivität in der App-UI selbst: Navigation, "
        "Page-Wechsel, Tab-Öffnen, Liste→Detail. Diese Items "
        "verarbeitest du STILL als reinen Awareness-Kontext — "
        "sprich sie NICHT vor, kommentiere sie NICHT ('ach, du bist "
        "jetzt im Dashboard'). Sie helfen dir nur zu wissen, wo der "
        "Operator gerade hinschaut, damit du auf seine Fragen "
        "präziser antworten kannst. Behandle sie wie "
        "`focus_boundary`: still verarbeiten.\n\n"
        "**AMBIENT-ROAM — `context_kind=ambient_boundary` + "
        "`ambient_message` (Codex frozen v1):**\n"
        "Ambient-Roam ist KEIN Fokuswechsel — es ist ein temporärer "
        "Nebenblick wenn der Operator-Fokus-Agent idle ist und der "
        "Operator den Modus 'Ambient' aktiviert hat. Der Operator-"
        "Fokus bleibt unverändert; nur die Narrationsquelle wandert "
        "kurz.\n\n"
        "  * `ambient_boundary` (Start/Wechsel/Ende eines "
        "Nebenblicks): STILL verarbeiten wie `focus_boundary`. Du "
        "darfst optional eine ganz leise, beiläufige Übergangsbrücke "
        "setzen — NIE als Themenwechsel klingen lassen. Erlaubt: "
        "'…drüben bei Storage', 'kurz zu GuideDevBot'. **NICHT** "
        "'jetzt zu Storage' (das klingt nach echtem Fokuswechsel) "
        "oder 'wir wechseln zu Storage'. Beim Ende eines Roams "
        "(`ambient_boundary` mit leerem/null source_agent) optional "
        "Stille oder eine ganz kurze Rückkehr-Brücke ('wieder bei "
        "CloudV2') — meistens lieber stille Übergabe.\n\n"
        "  * `ambient_message` (Aktivität des Ambient-Ziels): darf "
        "knapp narratiert werden, explizit als **Nebenblick-"
        "Tonalität** — der Operator weiß, du füllst nur eine "
        "Fokus-Idle-Phase. Sätze klingen wie 'drüben bei Storage "
        "läuft gerade der Migrations-Test' oder 'Storage meldet "
        "zwischendurch 3 Treffer im Log'. **NIEMALS** als Aussage "
        "des Fokus-Agenten framen, NIEMALS als Antwort auf eine "
        "Operator-Frage am Fokus-Agenten verkaufen. Hintergrund-"
        "Tonalität, nicht Hauptbühne.\n\n"
        "  * **Rückkehr zum Fokus:** Sobald der Operator-Fokus-Agent "
        "wieder aktiv wird oder der Operator tippt/spricht, kommt "
        "zuerst ein `ambient_boundary(end)`, dann ggf. neues "
        "Fokus-Priming oder ein `agent_message` vom Fokus-Agenten. "
        "Du behandelst die Rückkehr als selbstverständlich — kein "
        "'jetzt wieder zurück zu CloudV2', einfach normal weiter im "
        "Fokus.\n\n"
        "**FEED-MARKER-GEWICHTUNG (Alex' UX, gilt pro source_agent):**\n"
        "  * `context_kind=agent_message` (Agentenbotschaft an "
        "Operator) — **KERN.** Sinngemäß in der 3. Person wiedergeben "
        "('<Name> meldet, dass …', '<Name> fragt, ob …').\n"
        "  * `context_kind=operator_request` (vom Operator an den "
        "Agenten gerichtet) — **STILLER KONTEXT.** Nicht vorlesen, "
        "nicht paraphrasieren, nicht bewerten. Nur damit du die "
        "Agentenantwort einordnen kannst. NICHT 'Du hast <Name> "
        "gefragt, ob …' einleiten.\n"
        "  * `context_kind=background_work` — **STANDARDMÄSSIG STUMM.** "
        "Nur als kurzer Nebensatz wenn er erklärt: (a) warum das "
        "Ergebnis belastbar ist, (b) warum <Name> blockiert oder "
        "fehlgeschlagen ist, (c) welchen Input Alex jetzt geben muss, "
        "oder (d) wenn keine Agentenbotschaft vorliegt und ein "
        "knapper Aktivitäts-Status hilfreich ist.\n"
        "  * **Kein Tool-Ticker.** NIE jede einzelne Bash/Edit/Curl-"
        "Aktion erzählen.\n"
        "  * **Konflikt-Regel:** Agentenbotschaft > Tool-Status. "
        "Konflikt knapp kenntlich machen, nicht glätten.\n\n"
        "AUFGABEN:\n"
        "  1. ERZÄHLEN: Wenn `agent_message` im aktuellen Fokus-"
        "Segment ankommt, gib sie sinngemäß wieder — das ist die "
        "Hauptsache. Bei reinem `background_work` ohne neue Message: "
        "schweig oder ein einziger knapper Halbsatz ('Storage ist "
        "gerade an den Tests dran'). Bei Hintergrund-Agenten in "
        "alten Epochen: nur namentlich erwähnen wenn relevant.\n"
        "  2. MELDEN: Status-Hinweise bei wichtigen Wendungen — "
        "fertig, Fehler, wartet auf Input — beim Namen des "
        "verantwortlichen Agenten.\n"
        "  3. ANTWORTEN: Bei direkter Frage an dich (die AgentOS-"
        "Voice) — 'Was bedeutet das?', 'Was macht CloudV2 gerade?' — "
        "antworte feed-basiert in der 3. Person. Wenn KEIN aktueller "
        "Stream zu dem gefragten Agenten vorliegt, sag es ehrlich: "
        "'Zu CloudV2 sehe ich gerade keinen aktiven Stream.'\n\n"
        "PRIORITÄTS-REIHENFOLGE (Codex-Routing, Post #1215):\n"
        "  1. Agentenbotschaft im aktuellen Fokus sprechen.\n"
        "  2. Operator-zu-Agent nur still verstehen, nie vorlesen.\n"
        "  3. Direkte Frage an die AgentOS-Voice beantworten.\n"
        "  4. Agent-wirksame Intention NIEMALS nur kommentieren oder "
        "automatisch weiterleiten — Read-only-Modus.\n\n"
        "TEMPUS-DISZIPLIN — UNSICHTBAR:\n"
        "Die Unterscheidung zwischen historischem Kontext, aktuellem "
        "Fokus und Hintergrund-Agenten steuert NUR die Tempus-/"
        "Namens-Wahl, wird NIE verbalisiert. Verbotene Audio-Muster: "
        "'historischer Kontext', 'live sehe ich nichts', 'im "
        "Hintergrund', 'aktueller Fokus liegt auf'. Stattdessen "
        "natürlich erzählen:\n"
        "  * Aktueller Fokus + live: 'CloudV2 meldet jetzt, dass …'\n"
        "  * Aktueller Fokus + historischer Stand: 'CloudV2 hat "
        "zuletzt den WebRTC-Pfad fertiggestellt und wartet auf …'\n"
        "  * Anderer Agent als Hintergrund-Erwähnung: 'GuideDevBot "
        "ist im Hintergrund noch dran'\n"
        "  * Auf direkte Nachfrage zu unbekanntem Stand: 'Dazu liegt "
        "mir noch keine Information vor.'\n"
        "  * **Datenherkunft niemals erklären** — Inhalt + Tempus, "
        "nicht 'ich habe das aus dem Priming'.\n\n"
        "META-VERBALISIERUNG VERBOTEN:\n"
        "Sätze über deinen EIGENEN Zustand, Datenlage, oder "
        "Routing-Mechanik sind interne Gedanken, nicht vorlesen. "
        "NIE: 'ich warte auf neuen Output', 'ich melde mich gleich', "
        "'ich beobachte weiter', 'jetzt zur neuen Epoch', 'das war "
        "historisch', 'Fokus liegt jetzt auf …' (das tust du still). "
        "Pause statt Füllsatz.\n\n"
        "FIDELITY-DISZIPLIN (Post #1215 Smoke-Befund):\n"
        "  * 'Wörtlich' bedeutet FEED-TREU, nicht ungekürzt. Erlaubt "
        "sind IDENTIFIER: Agentennamen, Dateinamen, Funktionsnamen, "
        "Tool-Art, Zeilen-Nummern, kurze Status-Strings.\n"
        "  * NICHT erlaubt: vollständige Shell-Befehle, Code-Zeilen, "
        "Tool-Ausgaben WÖRTLICH vorlesen. Zusammenfassen.\n"
        "  * `[redacted]` / `[secret]` / `***`-Marker sind absichtlich "
        "geschwärzt. Nie rekonstruieren oder raten. Sag 'da steht ein "
        "redigierter Wert' und mach weiter.\n"
        "  * Keine generischen Plausibilitäts-Phrasen. Bei dünnem Feed: "
        "Pause oder 'kein neuer konkreter Output' — lieber Stille als "
        "Erfundenes.\n\n"
        "TOOL-AKTIVITÄTS-MARKER (Feed-Format ab 0.376) — STRIKT:\n"
        "  `[Aktion · <tool-label> · status=<x> · result=<y>]`\n"
        "  * `status=success` → fertig / erfolgreich erlaubt.\n"
        "  * `status=failed` → fehlgeschlagen / Fehler erlaubt.\n"
        "  * `status=running` → läuft noch. NIE als Erfolg.\n"
        "  * `status=unknown` → läuft, OHNE Wertung. NIE 'fertig' "
        "oder 'fehlgeschlagen' daraus ableiten.\n"
        "  * `result=<y>` ist serverseitig redigiertes Metadatum — "
        "knapp einmal nennen ('3 Treffer', 'exit 0'), nie wie Roh-"
        "Output vorlesen, nie interpretieren oder ergänzen.\n\n"
        "WICHTIG — DU BIST READ-ONLY:\n"
        "Du hast KEINE Möglichkeit, irgendetwas an irgendeinen "
        "Agenten zurückzusenden. Wenn der Operator dir einen Befehl "
        "an einen Agenten gibt, sag: 'Das müsstest du <Name> direkt "
        "im Text-Chat sagen — ich kann hier nur zuhören und "
        "narrieren.' Erfinde keine Tool-Calls.\n\n"
        "Stimme: ruhig, konzentriert, warm-distanziert wie ein "
        "Co-Pilot. Nie dramatisch. Continuous, nicht hektisch."
    )


def _companion_talkback_tools() -> List[dict]:
    """Single propose_to_agent tool with safety-by-confirm.

    Per Codex consolidation in #1215: never two tools, never a direct
    relay path. The model only proposes; the FE confirm-chip plus
    Cloud's /api/voice/realtime/proposals policy gate are the safety
    boundaries.
    """
    return [
        {
            "type": "function",
            "name": "propose_to_agent",
            "description": (
                "Propose to send the operator's spoken intent to the "
                "focused agent. NEVER assume auto-send — the operator's UI "
                "shows a confirm chip and waits for explicit yes/no "
                "(verbal or tap). Cloud's server-side policy gate "
                "/api/voice/realtime/proposals re-validates focus_epoch + "
                "re-classifies danger_class before any actual relay. "
                "Returns {confirmed: bool, edited_text?: str}. If "
                "confirmed=false drop the proposal and continue narrating; "
                "do not retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The command, paraphrased clearly in the "
                            "language the agent expects (typically English "
                            "for tmux-claude operator commands)."
                        ),
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "The focused agent's session id "
                            "(provided to you in session context at start)."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "One sentence — why this is what the operator "
                            "asked for. Used in the confirm chip so the "
                            "operator can verify intent before sending."
                        ),
                    },
                    "danger_class": {
                        "type": "string",
                        "enum": [
                            "none", "data-loss", "irreversible-git",
                            "process-kill", "permission-grant", "other",
                        ],
                        "description": (
                            "Flag the proposal's destructive potential. "
                            "Browser uses this to vary the confirm UI; "
                            "Cloud's policy gate may overrule + deny."
                        ),
                    },
                },
                "required": [
                    "text", "session_id", "rationale", "danger_class",
                ],
            },
        }
    ]


def _default_instructions(language: str = "de", track_id: Optional[int] = None) -> str:
    """System prompt that wires the model into the Federation tool set."""
    track_hint = (
        f"You are guiding the user through ArTrack track #{track_id}. "
        if track_id else ""
    )
    return (
        "You are the Wanderlaut Voice-Guide, a calm, knowledgeable companion "
        "for hikers in the Tscheppaschlucht and nearby trails. "
        f"{track_hint}"
        "Default language is "
        f"{ {'de': 'German', 'sl': 'Slovenian', 'it': 'Italian', 'en': 'English'}.get(language, 'German') }, "
        "but mirror the user's language when they switch. Keep replies "
        "short (1-3 sentences) and conversational; never lecture.\n\n"
        "AUDIO MODEL — important to understand:\n"
        "Your spoken reply is delivered via the WebRTC audio stream (live, "
        "real-time). The show_topic / show_image / show_knowledge_pin / "
        "focus_map tools are PURE VISUAL HINTS for the browser client — "
        "they do NOT speak, they only update what the user sees. Keep "
        "talking normally while you call them; the visuals appear in "
        "parallel to your voice without any TTS being involved.\n\n"
        "TOOL DISCIPLINE — this is critical:\n"
        "  * ALWAYS call `knowledge_query` or `narration_near` BEFORE making "
        "a factual claim about a POI, plant or animal. Never invent details.\n"
        "  * When calling `narration_near`, use radius_m of AT LEAST 500. "
        "The persist side re-grounds GPS coords via geocoder (~300m typical "
        "drift). A 100m radius would miss the very point that was just "
        "saved at the same logical place.\n"
        "  * Call `show_topic` the moment you start describing a new place, "
        "and `show_image` whenever knowledge_query returns a storage_id.\n"
        "  * Call `persist_narration` AFTER you finish describing a place — "
        "this saves what you said for other users.\n"
        "  * If a tool will take noticeable time, say 'Moment...' so the user "
        "knows you're working, then continue.\n\n"
        "Voice: warm, slightly poetic, occasionally playful. Never robotic."
    )


# ── Billing gate ──────────────────────────────────────────────────────


def _check_realtime_billing_gate(confirmed: Optional[bool]) -> None:
    """Default-deny + monthly-cap gate, mirrors the M3/DeepSeek/OpenAI
    pattern. 403 when caller forgot the flag, 429 when the 100 EUR cap
    has been blown for the month."""
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker

    if not confirmed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "api_billing_confirmation_required",
                "endpoint": "openai-realtime",
                "provider": "openai",
                "hint": (
                    "OpenAI Realtime is pay-as-you-go billed against "
                    "OPENAI_API_KEY (~$0.05-0.30/minute depending on model). "
                    "Federation-shared 100 EUR/month cap. Send "
                    "`confirm_api_billing: true` to acknowledge."
                ),
            },
        )
    if openai_realtime_cost_tracker.should_block_request():
        status = openai_realtime_cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_api_cap_reached",
                "endpoint": "openai-realtime",
                "provider": "openai",
                "spent_eur": round(status.get("total_cost_eur", 0.0), 2),
                "budget_eur": status.get("monthly_budget_eur"),
                "hint": (
                    "OpenAI Realtime monthly cap reached. Cap resets at the "
                    "start of the next calendar month. Other endpoints "
                    "(/ai/claude, /ai/chatgpt, image gen) are unaffected."
                ),
            },
        )


def get_api_key():  # placeholder, mirrors other routes
    return "placeholder"


# ── Endpoints ─────────────────────────────────────────────────────────


async def _mint_elevenlabs_token(request: RealtimeTokenRequest) -> dict:
    """Mint an ElevenLabs Conversational AI signed URL for the browser.

    ElevenLabs' Conv. AI agents pre-bake the LLM + voice + prompt + tools
    in their dashboard config. The browser opens a WebSocket directly to
    the signed URL we hand back — no SDP exchange, no separate token.

    Two env knobs:
      * ``ELEVENLABS_API_KEY``       — required (same key as /ai/tts/narrate)
      * ``ELEVENLABS_AGENT_ID``      — required; pre-created agent in the
                                       ElevenLabs dashboard. We don't auto-
                                       create one because the agent config
                                       (prompt, voice, knowledge base) is
                                       authored UI-side.
    """
    api_key_env = os.getenv("ELEVENLABS_API_KEY", "").strip('"').strip("'")
    agent_id = request.agent_id or os.getenv("ELEVENLABS_AGENT_ID", "")
    if not api_key_env:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "elevenlabs_api_key_missing",
                "hint": "ELEVENLABS_API_KEY not set on this api-ai host.",
            },
        )
    if not agent_id:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "elevenlabs_agent_id_missing",
                "hint": (
                    "ELEVENLABS_AGENT_ID env var not set. Create a Conversational "
                    "AI agent at https://elevenlabs.io/app/conversational-ai and "
                    "set the resulting agent_id as ELEVENLABS_AGENT_ID."
                ),
            },
        )
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(
                "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
                params={"agent_id": agent_id},
                headers={"xi-api-key": api_key_env},
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "elevenlabs_upstream_unreachable",
                    "exc": str(e)[:200],
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
                "error": "elevenlabs_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
            },
        )
    body = r.json()
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    openai_realtime_cost_tracker.track_session_start()
    return {
        "provider": "elevenlabs",
        "signed_url": body.get("signed_url"),
        "agent_id": agent_id,
        "voice": None,
        "model": "elevenlabs-convai",
        "tools": [],
        "session_id": request.session_id,
        "raw": body,
    }


@router.post("/realtime/voice-clone")
async def voice_clone(
    audio: UploadFile = File(..., description="Reference audio sample (webm / mp3 / wav). 30-60s recommended for IVC."),
    name: str = Form("Cloned Voice"),
    description: Optional[str] = Form(None),
    language: str = Form("de"),
    api_key: str = Depends(get_api_key),
):
    """Clone a voice from a user-uploaded audio sample, then create a
    dedicated ElevenLabs Conversational AI agent that uses that voice.

    Returns ``{voice_id, agent_id, voice_name}`` — the browser passes
    ``agent_id`` back through ``/ai/realtime/token`` (``agent_id``
    override field) to talk to a model whose TTS is the cloned voice.

    Dialect note: ElevenLabs Instant Voice Clone (IVC) keeps timbre +
    cadence well; dialect-specific phonemes degrade toward standard
    pronunciation, especially with the flash TTS model. We default to
    ``eleven_multilingual_v2`` here because it preserves accent the
    best (slower latency tradeoff, but the dialect demo is the point).
    """
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "").strip('"').strip("'")
    if not eleven_key:
        raise HTTPException(500, "ELEVENLABS_API_KEY missing")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "audio sample is empty")
    if len(audio_bytes) < 50_000:
        # ~3 seconds at the lowest realistic bitrate. Anything shorter
        # produces useless clones — better to fail fast with a hint.
        raise HTTPException(
            400,
            f"audio sample too short ({len(audio_bytes)} bytes) — record 30-60s for a usable IVC clone",
        )

    filename = audio.filename or "clone.webm"
    content_type = audio.content_type or "audio/webm"
    logger.info(
        f"voice-clone: uploading {len(audio_bytes)} bytes ({content_type}) "
        f"as '{name}' lang={language}"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: create the voice via IVC
        try:
            r = await client.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={"xi-api-key": eleven_key},
                files=[("files", (filename, audio_bytes, content_type))],
                data={
                    "name": name,
                    "description": description or f"Cloned via AiApi realtime test HP ({language})",
                    "labels": json.dumps({"language": language, "source": "aiapi-test-hp"}),
                },
            )
        except httpx.HTTPError as e:
            raise HTTPException(502, {"error": "elevenlabs_upload_failed", "exc": str(e)[:200]})
        if r.status_code >= 400:
            try: body = r.json()
            except Exception: body = {"raw": r.text[:300]}
            raise HTTPException(
                502 if r.status_code >= 500 else r.status_code,
                {"error": "elevenlabs_voice_add_failed", "status": r.status_code, "body": body},
            )
        voice = r.json()
        voice_id = voice.get("voice_id")
        if not voice_id:
            raise HTTPException(502, {"error": "elevenlabs_no_voice_id", "body": voice})

        # Step 2: spin up a Conv. AI agent that uses this voice. We use
        # eleven_multilingual_v2 because it keeps accent characteristics
        # noticeably better than the flash models (per ElevenLabs' own
        # accent-retention benchmarks).
        prompt = (
            "Du bist ein freundlicher Gesprächspartner für einen Voice-Clone-Test. "
            "Antworte kurz und natürlich auf die Sprache des Users (Default Deutsch). "
            "Wenn der User in einem Dialekt spricht, antworte normal — der Witz an dem "
            "Test ist nicht WAS du sagst, sondern dass du die geklonte Stimme verwendest."
        )
        first_message = (
            "Hallo! Ich spreche jetzt mit deiner geklonten Stimme. Wie klingt sie für dich?"
        )
        try:
            ar = await client.post(
                "https://api.elevenlabs.io/v1/convai/agents/create",
                headers={"xi-api-key": eleven_key, "Content-Type": "application/json"},
                json={
                    "name": f"AiApi Clone Agent — {name}",
                    "conversation_config": {
                        "agent": {
                            "prompt": {"prompt": prompt},
                            "first_message": first_message,
                            "language": language,
                        },
                        "tts": {
                            "voice_id": voice_id,
                            "model_id": "eleven_multilingual_v2",
                        },
                    },
                },
            )
        except httpx.HTTPError as e:
            raise HTTPException(502, {"error": "elevenlabs_agent_create_failed", "exc": str(e)[:200]})
        if ar.status_code >= 400:
            try: body = ar.json()
            except Exception: body = {"raw": ar.text[:300]}
            raise HTTPException(
                502 if ar.status_code >= 500 else ar.status_code,
                {"error": "elevenlabs_agent_create_rejected", "status": ar.status_code, "body": body},
            )
        agent = ar.json()
        agent_id = agent.get("agent_id")
        if not agent_id:
            raise HTTPException(502, {"error": "elevenlabs_no_agent_id", "body": agent})

    logger.info(f"voice-clone: voice_id={voice_id} agent_id={agent_id}")
    return {
        "voice_id": voice_id,
        "agent_id": agent_id,
        "voice_name": name,
        "model_id": "eleven_multilingual_v2",
        "language": language,
    }


@router.post("/realtime/token")
async def mint_realtime_token(
    request: RealtimeTokenRequest,
    api_key: str = Depends(get_api_key),
    grant: VerifiedGrant = Depends(require_realtime_grant("mint")),
):
    """Mint a short-lived Realtime session token.

    Provider switch: 'openai' returns an OpenAI ephemeral ``client_secret``
    plus the resolved model and tool list. 'elevenlabs' returns a signed
    WebSocket URL pointing at a pre-created Conv. AI agent.

    Auth: the browser sends its User-JWT in ``Authorization: Bearer``.
    api-ai performs the server-to-server grant exchange with auth-api
    (Content-Post #1215 frozen v1 Auth-Contract) and verifies the
    capability JWT locally before minting. Wrong profile / disabled
    user / missing limits all fail closed at the dependency.
    """
    _check_realtime_billing_gate(request.confirm_api_billing)

    provider = (request.provider or DEFAULT_PROVIDER).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_realtime_provider",
                "provider": provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
            },
        )

    if provider == "elevenlabs":
        return await _mint_elevenlabs_token(request)

    model = request.model or DEFAULT_REALTIME_MODEL
    if model not in SUPPORTED_REALTIME_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_realtime_model",
                "model": model,
                "supported": sorted(SUPPORTED_REALTIME_MODELS),
            },
        )

    voice = request.voice or DEFAULT_REALTIME_VOICE
    if voice not in SUPPORTED_REALTIME_VOICES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_realtime_voice",
                "voice": voice,
                "supported": sorted(SUPPORTED_REALTIME_VOICES),
            },
        )

    # Companion-mode preset (Content-Post #1215). When set, picks the
    # right narrator system-prompt + restricts the tool set accordingly.
    # Always wins over both ``instructions`` and the default Wanderlaut
    # tool set — the caller asked specifically for the CloudV2 preset.
    companion_mode = request.companion_mode
    if companion_mode and companion_mode not in SUPPORTED_COMPANION_MODES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_companion_mode",
                "companion_mode": companion_mode,
                "supported": sorted(SUPPORTED_COMPANION_MODES),
            },
        )
    detail_level = (request.detail_level or "balanced").lower()
    if detail_level not in SUPPORTED_DETAIL_LEVELS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_detail_level",
                "detail_level": detail_level,
                "supported": sorted(SUPPORTED_DETAIL_LEVELS),
            },
        )
    companion_tools_override: Optional[List[dict]] = None
    if companion_mode == "narrator-only":
        instructions = _companion_narrator_prompt(request.language or "de")
        instructions += _detail_level_addendum(detail_level)
        companion_tools_override = []  # zero-tools hardening
        logger.info(
            f"Realtime: companion_mode=narrator-only "
            f"detail_level={detail_level} "
            f"({len(instructions)} chars, 0 tools)"
        )
    elif companion_mode == "talkback-enabled":
        instructions = _companion_talkback_prompt(request.language or "de")
        instructions += _detail_level_addendum(detail_level)
        companion_tools_override = _companion_talkback_tools()
        logger.info(
            f"Realtime: companion_mode=talkback-enabled "
            f"detail_level={detail_level} "
            f"({len(instructions)} chars, "
            f"{len(companion_tools_override)} tools)"
        )
    elif companion_mode == "agentos-narrator":
        instructions = _companion_agentos_narrator_prompt(
            request.language or "de"
        )
        instructions += _detail_level_addendum(detail_level)
        companion_tools_override = []  # zero-tools hardening (read-only)
        logger.info(
            f"Realtime: companion_mode=agentos-narrator "
            f"detail_level={detail_level} "
            f"companion_run_id={request.companion_run_id or 'none'} "
            f"({len(instructions)} chars, 0 tools)"
        )
    else:
        instructions = request.instructions or _default_instructions(
            language=request.language or "de",
            track_id=request.track_id,
        )

    # Federation persona — optional. Uses the same get_persona_bundle
    # helper as the text routes so the same virtual-bots that work
    # against /ai/claude work against /ai/realtime too.
    persona_rendered = ""
    if request.persona_variant:
        try:
            from .text_ai_routes import get_persona_bundle
            persona = await get_persona_bundle("realtime", request.persona_variant)
            persona_rendered = persona.get("rendered", "")
            if persona_rendered:
                instructions = f"{persona_rendered}\n\n{instructions}"
                logger.info(
                    f"Realtime: injected persona "
                    f"api-ai-realtime-{request.persona_variant} "
                    f"({len(persona_rendered)} chars)"
                )
        except Exception as e:
            logger.warning(f"Realtime: persona fetch failed ({e}); proceeding without")

    api_key_env = os.getenv("OPENAI_API_KEY", "")
    if not api_key_env:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "openai_api_key_missing",
                "hint": "OPENAI_API_KEY not configured in service env.",
            },
        )

    tools = _all_tool_defs() if companion_tools_override is None else companion_tools_override
    # OpenAI's GA Realtime session shape (2025-Q4+) nests audio knobs
    # under ``audio.input`` / ``audio.output`` instead of the legacy
    # flat ``voice`` / ``input_audio_format`` fields. The mint endpoint
    # surfaces "Unknown parameter: 'session.voice'" when you use the
    # old layout, so we ship the new layout from day one.
    session_config = {
        "type": "realtime",
        "model": model,
        "instructions": instructions,
        "tools": tools,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500,
                },
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
            },
        },
    }
    # NOTE: session.metadata is not (yet?) accepted on the
    # client_secrets mint — we drop it from the session-config and
    # stamp guide_session_id / track_id only in our own response, where
    # the browser already needs to read it to wire X-Session-ID headers.

    # Atomic pre-mint reservation against the grant's per-profile
    # daily_budget_eur and max_parallel_sessions caps (Codex Final,
    # Post #1215). Placed AFTER all validation paths so a failed
    # validation never leaks a reservation slot. Released on any
    # OpenAI-side mint failure below.
    voice_session_id = (
        request.voice_session_id
        or request.session_id
        or f"vs_pending_{int(time.time()*1000)}"
    )
    try:
        reservation = realtime_budget_guard.reserve_mint(
            profile_id=grant.profile_id,
            user_id=grant.sub,
            voice_session_id=voice_session_id,
            max_parallel_sessions=grant.max_parallel_sessions,
            daily_budget_eur=grant.daily_budget_eur,
        )
    except BudgetGuardError as exc:
        logger.warning(
            "realtime mint deny code=%s detail=%s",
            exc.error_code, exc.audit_detail,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": exc.error_code},
        ) from exc
    logger.info(
        "realtime mint: profile=%s tenant=%s vid=%s budget=%.2f max_parallel=%d",
        grant.profile_id, grant.tenant_id, voice_session_id,
        grant.daily_budget_eur, grant.max_parallel_sessions,
    )

    # Token-mint upstream — OpenAI's /v1/realtime/client_secrets endpoint
    # returns an ephemeral client secret with the session_config baked in.
    #
    # NO "OpenAI-Beta: realtime=v1" header — that header is what marks the
    # minted ek_ token as beta-shaped, and OpenAI now rejects beta tokens
    # at the SDP exchange with "beta_api_shape_disabled" (verified live by
    # GuideDevBot2's browser voice attempt + AiApi's own headless smoke).
    # The /client_secrets endpoint accepts the call without the header and
    # returns a GA-shaped token usable against the /v1/realtime SDP path.
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                json={"session": session_config},
                headers={
                    "Authorization": f"Bearer {api_key_env}",
                },
            )
        except httpx.HTTPError as e:
            # OpenAI unreachable — release the reservation so the slot
            # doesn't burn for 60min until the orphan reaper catches it.
            realtime_budget_guard.release_reservation(reservation)
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "openai_upstream_unreachable",
                    "exc": str(e)[:200],
                },
            )

    if r.status_code >= 400:
        # OpenAI rejected the mint — release the reservation.
        realtime_budget_guard.release_reservation(reservation)
        try:
            upstream = r.json()
        except Exception:
            upstream = {"raw": r.text[:500]}
        logger.error(
            f"OpenAI realtime token mint failed {r.status_code}: {upstream}"
        )
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail={
                "error": "openai_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
            },
        )

    body = r.json()

    # Track the session-start before we hand the token to the browser.
    # The actual usage roll-up arrives later via /ai/realtime/usage.
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    openai_realtime_cost_tracker.track_session_start()

    return {
        "provider": "openai",
        "client_secret": body.get("client_secret") or body.get("value") or body,
        "expires_at": body.get("expires_at"),
        "model": model,
        "voice": voice,
        "tools": [t["name"] for t in tools],
        "session_id": request.session_id,
        "companion_mode": companion_mode,
        "detail_level": detail_level if companion_mode else None,
        "raw": body,
    }


@router.get("/realtime/models")
async def list_realtime_models():
    """Return the realtime models and voices we support."""
    return {
        "models": [
            {
                "id": "gpt-realtime",
                "provider": "openai",
                "default": True,
                "tier": "ga",
                "description": (
                    "OpenAI Realtime GA. Multilingual DE/SL/IT/EN, "
                    "200-400ms roundtrip, premium voice quality. "
                    "The only model whose SDP-connect path works today — "
                    "the preview aliases mint a token but 4004 on WS."
                ),
                "price_per_min_usd_estimate": "$0.15-0.30",
            },
        ],
        "voices": sorted(SUPPORTED_REALTIME_VOICES),
        "tools": [
            {"name": t["name"], "description": t["description"]}
            for t in _all_tool_defs()
        ],
    }


@router.get("/realtime/cost-status")
async def realtime_cost_status():
    """Federation-shared Realtime cap state, same shape as the other
    /ai/{provider}/cost-status endpoints."""
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    return openai_realtime_cost_tracker.get_status()


@router.post("/realtime/cost-status/reset-hard-cap")
async def reset_realtime_hard_cap(api_key: str = Depends(get_api_key)):
    """Operator escape hatch — clears the persistent hard-cap flag.

    Mirror of ``/ai/gemini/cost-status/reset-hard-cap``. The cap will
    re-trip automatically if usage tracking puts us back over 100 EUR,
    so this is safe to call without risking silent re-opening.
    """
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    result = openai_realtime_cost_tracker.clear_hard_cap()
    logger.warning(
        f"Realtime hard-cap manually reset (was_active={result['was_active']})"
    )
    return result


@router.post("/realtime/usage")
async def realtime_usage_report(
    report: RealtimeUsageReport,
    grant: VerifiedGrant = Depends(require_realtime_grant("usage")),
):
    """Browser-reported usage callback.

    OpenAI Realtime emits ``response.done`` events with a usage block:
    ``input_tokens.audio_tokens``, ``output_tokens.audio_tokens``, plus
    the text-token mirror. The browser SHOULD post one record per
    ``response.done`` (not aggregate at session-end) so a tab-crash
    only loses one turn, not the whole session — see Content-Post
    #1215 reconciliation thread.

    Idempotency (Codex contract, #1215): pass ``voice_session_id`` AND
    ``usage_event_id`` (= OpenAI ``response.id``). The server dedupes
    on that pair so retries / final-flushes / sendBeacon-on-pagehide
    never double-count. Records without those keys still get accepted
    but log a non-idempotent warning.

    Response carries ``deduped: bool`` so the client can mark the
    record acked in its pending queue.

    Auth: the user must hold a valid grant with the ``usage`` scope.
    Post-revoke usage callbacks are rejected with 403 (Codex' v1
    revocation contract: revoke-next-mint, and every protected
    endpoint exchanges per request).
    """
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    result = openai_realtime_cost_tracker.track_session(
        model=report.model,
        audio_input_tokens=report.audio_input_tokens,
        audio_output_tokens=report.audio_output_tokens,
        text_input_tokens=report.text_input_tokens,
        text_output_tokens=report.text_output_tokens,
        duration_sec=report.duration_sec,
        voice_session_id=report.voice_session_id or report.session_id,
        usage_event_id=report.usage_event_id,
    )
    # Charge the per-profile budget guard with the same usage. Skip if
    # this report was a dedup-replay (already counted). The cost in EUR
    # is taken from what the cost tracker just computed for this row.
    if result and result.get("accepted") and not result.get("deduped"):
        from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker as _tracker
        # The tracker doesn't expose the per-row cost directly, but for
        # the budget guard we only need ``cost_eur`` of THIS turn. Re-
        # compute it locally — same pricing table, deterministic.
        per_row_eur = _tracker._cost_for_session(
            model=report.model,
            audio_input_tokens=report.audio_input_tokens,
            audio_output_tokens=report.audio_output_tokens,
            text_input_tokens=report.text_input_tokens,
            text_output_tokens=report.text_output_tokens,
        ).get("cost_eur", 0.0)
        try:
            realtime_budget_guard.confirm_usage_charge(
                profile_id=grant.profile_id,
                user_id=grant.sub,
                voice_session_id=report.voice_session_id or report.session_id or "",
                cost_eur=float(per_row_eur),
            )
        except Exception as exc:
            # Charging the guard must NEVER tank the usage report —
            # the cost tracker is the source of truth, the guard is
            # the optimisation. Log and continue.
            logger.warning("budget_guard charge failed: %s", exc)

    status = openai_realtime_cost_tracker.get_status()
    status["deduped"] = bool(result and result.get("deduped"))
    status["accepted"] = bool(result and result.get("accepted"))
    return status


# ── Dev Narration-Log (CloudV2, Post #1215) ───────────────────────────


DEVLOG_ROOT = "/var/lib/api-ai/devlogs"
DEVLOG_SECRET_ENV = "DEVLOG_DEV_SECRET"


def _extract_owner_from_jwt(authorization: Optional[str]) -> Optional[str]:
    """Best-effort owner extraction from a Bearer JWT.

    We do NOT verify the signature — api-ai has no shared signing key
    with CloudV2's auth system. The owner string is used purely to
    bucket per-owner storage so a careless operator can't read another
    operator's logs by guessing a voice_session_id. The dev-secret-
    gated GET path is the actual authority for cross-owner reads.

    Returns the ``email`` or ``sub`` claim from the JWT payload, or
    None if no usable claim is present.
    """
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1]
    try:
        payload_b64 = token.split(".")[1]
        # base64url decode (no padding needed by standard lib if we pad)
        import base64
        pad = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64 + pad)
        claims = json.loads(payload_json)
    except Exception:
        return None
    return claims.get("email") or claims.get("sub")


def _owner_bucket(owner: Optional[str]) -> str:
    """Return a stable per-owner directory name.

    We hash the owner identifier so the on-disk path doesn't contain
    PII. The dev-secret read endpoints get the original owner back
    inside each session's JSON payload.
    """
    import hashlib
    src = (owner or "anonymous").encode("utf-8")
    return hashlib.sha256(src).hexdigest()[:16]


def _devlog_dir(owner: Optional[str]) -> str:
    bucket = _owner_bucket(owner)
    path = os.path.join(DEVLOG_ROOT, bucket)
    os.makedirs(path, exist_ok=True)
    return path


def _devlog_path(owner: Optional[str], voice_session_id: str) -> str:
    # Voice session ids are FE-generated and used as path component —
    # constrain to a safe character class so a malformed id can't escape
    # the per-owner bucket.
    import re
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", voice_session_id)
    return os.path.join(_devlog_dir(owner), f"{safe}.json")


def _check_dev_secret(header_secret: Optional[str]) -> None:
    """Gate the read/delete endpoints behind the configured dev secret.

    The secret is read from env (``DEVLOG_DEV_SECRET``). If unset, the
    endpoint refuses all requests — never accidentally world-readable.
    """
    expected = os.environ.get(DEVLOG_SECRET_ENV)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "devlog_dev_secret_unset",
                "hint": (
                    "Set DEVLOG_DEV_SECRET in the api-ai environment to "
                    "enable read/delete endpoints. Without it, dev logs "
                    "are write-only — CloudV2 can post but no one can read."
                ),
            },
        )
    if not header_secret or header_secret != expected:
        raise HTTPException(
            status_code=403,
            detail={"error": "devlog_invalid_dev_secret"},
        )


@router.post("/realtime/devlog")
async def realtime_devlog_upsert(
    body: DevlogUpsertRequest,
    grant: VerifiedGrant = Depends(require_realtime_grant("devlog")),
):
    """Upsert a narration log for a voice session (CloudV2 capture).

    Auth: ``Authorization: Bearer <user-JWT>`` exchanged via the
    auth-api grant flow; the verified grant carries the stable
    ``sub`` (user_id UUID) plus ``tenant_id`` and ``profile_id``.
    Owner-bucket is derived from grant claims, NOT from a best-effort
    JWT decode (the previous v0 behavior). Sessions owned by a user
    whose grant has been revoked can no longer write.

    Behaviour: full-document overwrite keyed by ``voice_session_id``
    inside the owner's bucket. CloudV2 re-POSTs the growing transcript
    every ~2.5 s and on stop; the latest payload always wins.
    """
    # Stable-id owner bucket — Codex Punkt 5 (no email-as-primary-key).
    owner_key = f"{grant.sub}:{grant.tenant_id}:{grant.profile_id}"
    path = _devlog_path(owner_key, body.voice_session_id)
    record = {
        "voice_session_id": body.voice_session_id,
        "owner": owner_key,
        "owner_sub": grant.sub,
        "owner_tenant": grant.tenant_id,
        "owner_profile": grant.profile_id,
        "agent": body.agent,
        "started_at": body.started_at,
        "ended_at": body.ended_at,
        "line_count": len(body.lines),
        "lines": [line.dict(exclude_none=True) for line in body.lines],
        "received_at": time.time(),
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(record, f, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_path, path)
    logger.info(
        "Devlog upsert: voice_session_id=%s owner=%s lines=%d",
        body.voice_session_id, owner or "anonymous", len(body.lines),
    )
    return {
        "accepted": True,
        "voice_session_id": body.voice_session_id,
        "owner": owner,
        "line_count": len(body.lines),
    }


@router.get("/realtime/devlogs")
async def realtime_devlogs_list(
    since: Optional[float] = None,
    owner: Optional[str] = None,
    x_dev_secret: Optional[str] = Header(None),
):
    """List captured dev logs (dev-secret-gated).

    Returns one summary per session: ``voice_session_id``, ``owner``,
    ``agent``, timestamps, ``line_count``, and the file ``mtime``.

    Filter:
      * ``since`` — epoch seconds; only logs whose ``received_at`` is
        newer than this are returned.
      * ``owner`` — restrict to a specific owner identifier.
    """
    _check_dev_secret(x_dev_secret)
    if not os.path.isdir(DEVLOG_ROOT):
        return {"sessions": [], "count": 0}
    target_bucket = _owner_bucket(owner) if owner is not None else None
    out: List[dict] = []
    for bucket_name in os.listdir(DEVLOG_ROOT):
        if target_bucket and bucket_name != target_bucket:
            continue
        bucket_path = os.path.join(DEVLOG_ROOT, bucket_name)
        if not os.path.isdir(bucket_path):
            continue
        for fname in os.listdir(bucket_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(bucket_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            if since is not None and rec.get("received_at", 0) < since:
                continue
            out.append({
                "voice_session_id": rec.get("voice_session_id"),
                "owner": rec.get("owner"),
                "agent": rec.get("agent"),
                "started_at": rec.get("started_at"),
                "ended_at": rec.get("ended_at"),
                "line_count": rec.get("line_count", 0),
                "received_at": rec.get("received_at"),
            })
    out.sort(key=lambda r: r.get("received_at") or 0, reverse=True)
    return {"sessions": out, "count": len(out)}


@router.get("/realtime/devlog/{voice_session_id}")
async def realtime_devlog_get(
    voice_session_id: str = Path(...),
    owner: Optional[str] = None,
    x_dev_secret: Optional[str] = Header(None),
):
    """Read the full transcript of a single voice session.

    The dev secret is required. If ``owner`` is omitted, the endpoint
    walks every owner bucket and returns the first match — convenient
    for the dev-mode case but slow on large stores; pass ``owner`` when
    you already know it.
    """
    _check_dev_secret(x_dev_secret)
    if owner is not None:
        path = _devlog_path(owner, voice_session_id)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # No owner hint — scan
    import re
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", voice_session_id)
    target_fname = f"{safe}.json"
    if not os.path.isdir(DEVLOG_ROOT):
        raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})
    for bucket_name in os.listdir(DEVLOG_ROOT):
        candidate = os.path.join(DEVLOG_ROOT, bucket_name, target_fname)
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})


@router.delete("/realtime/devlog/{voice_session_id}")
async def realtime_devlog_delete(
    voice_session_id: str = Path(...),
    owner: Optional[str] = None,
    x_dev_secret: Optional[str] = Header(None),
):
    """Purge a single session's transcript (dev-secret-gated)."""
    _check_dev_secret(x_dev_secret)
    if owner is not None:
        path = _devlog_path(owner, voice_session_id)
        if os.path.isfile(path):
            os.remove(path)
            return {"deleted": True, "voice_session_id": voice_session_id}
        raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})
    import re
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", voice_session_id)
    target_fname = f"{safe}.json"
    if not os.path.isdir(DEVLOG_ROOT):
        raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})
    for bucket_name in os.listdir(DEVLOG_ROOT):
        candidate = os.path.join(DEVLOG_ROOT, bucket_name, target_fname)
        if os.path.isfile(candidate):
            os.remove(candidate)
            return {"deleted": True, "voice_session_id": voice_session_id}
    raise HTTPException(status_code=404, detail={"error": "devlog_not_found"})


# ── Config-Health (admin-gated, no key metadata) ──────────────────────


@router.get("/realtime/config-health")
async def realtime_config_health(
    x_dev_secret: Optional[str] = Header(None),
):
    """Minimal admin-scoped readiness check for the per-host realtime
    auth configuration. Codex' Test 8 (Post #1215) — proves the host
    is bound to its billing profile and the grant-verifier chain is
    wired without leaking any key metadata.

    Auth: ``X-Dev-Secret`` — same admin-secret as the devlog read
    endpoints. If unset on the host, all calls 503.

    Returned shape (Codex-pinned minimum):
      * profile_id              — this host's REALTIME_PROFILE_ID
      * key_configured          — REALTIME_GRANT_SERVICE_KEY present
      * grant_verifier_ready    — both pinned envs are set
      * cost_tracker_namespace  — profile_id, mirrors the future
                                  per-profile tracker file suffix
      * secret_version          — opaque, set by ops; empty if not
                                  rotated yet

    NO key fingerprints, NO key prefixes, NO algorithmic identifiers
    from the key bytes. The audit signal is binary.
    """
    _check_dev_secret(x_dev_secret)
    profile = host_profile_id()
    key_ok = service_key_configured()
    return {
        "profile_id": profile,
        "key_configured": key_ok,
        "grant_verifier_ready": bool(profile and key_ok),
        "cost_tracker_namespace": profile or "",
        "secret_version": os.environ.get(
            "REALTIME_GRANT_SECRET_VERSION", ""
        ),
    }


# ── Tool-routing proxy ────────────────────────────────────────────────


# Allowed Read tools — the only ones the browser is supposed to route
# through AiApi. Display-hint tools must NEVER reach this endpoint
# (the browser shorts them locally); we reject them so a bug there
# surfaces fast instead of silently going to OpenAI's expensive
# fail-mode of "model thinks it called the tool, never got an answer".
READ_TOOL_NAMES = {"knowledge_query", "pois_near", "narration_near"}


@router.post("/realtime/tool/{tool_name}")
async def realtime_tool_call(
    tool_name: str = Path(..., description="Function name from the model"),
    body: RealtimeToolCall = ...,
    x_session_id: Optional[str] = Header(
        default=None,
        alias="X-Session-ID",
        description="Guide-api session id, stamped by the browser.",
    ),
):
    """Resolve a Realtime function-call against the Federation MCPs.

    The browser receives a ``function_call`` event from the data
    channel, JSON-decodes the ``arguments``, and POSTs them here. We
    dispatch by tool name, talk to the appropriate Federation service,
    and return a JSON result that the browser forwards back to OpenAI
    as the ``function_call_output``. OpenAI then resumes generation
    with the new context.

    Latency budget: <= 250ms end-to-end is the target. Anything slower
    causes a hearable hang in the spoken response. We rely on the
    Federation MCPs being colocated on arkserver / arkturian, and a
    hot httpx client at runtime.
    """
    if tool_name not in READ_TOOL_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tool_not_routable_via_aiapi",
                "tool": tool_name,
                "hint": (
                    "Display-hint tools (show_*, focus_map) must be "
                    "handled in the browser by emitting _bus events. "
                    "Persist tools (persist_narration) must POST to "
                    "guide-api /api/v1/realtime/narration directly. "
                    "Only Read tools route through AiApi: "
                    f"{sorted(READ_TOOL_NAMES)}."
                ),
            },
        )

    t0 = time.monotonic()
    args = body.arguments or {}
    result: Any
    try:
        if tool_name == "knowledge_query":
            result = await _tool_knowledge_query(args)
        elif tool_name == "pois_near":
            result = await _tool_pois_near(args)
        elif tool_name == "narration_near":
            result = await _tool_narration_near(args)
        else:
            # Defensive: should be caught by READ_TOOL_NAMES check above.
            raise HTTPException(status_code=400, detail="unknown tool")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"realtime tool {tool_name} failed: {e}")
        # Return a structured tool error to OpenAI so the model can
        # apologise verbally instead of stalling.
        return {
            "call_id": body.call_id,
            "tool": tool_name,
            "ok": False,
            "error": str(e)[:200],
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if elapsed_ms > 250:
        logger.warning(
            f"realtime tool {tool_name} took {elapsed_ms}ms — "
            f"may cause hearable hang"
        )
    logger.info(
        f"realtime tool {tool_name} ok in {elapsed_ms}ms "
        f"(session={x_session_id})"
    )
    return {
        "call_id": body.call_id,
        "tool": tool_name,
        "ok": True,
        "result": result,
        "elapsed_ms": elapsed_ms,
    }


# ── Tool implementations ──────────────────────────────────────────────


def _knowledge_api_base() -> str:
    return os.getenv(
        "KNOWLEDGE_API_URL", "https://knowledge-api.arkturian.com"
    ).rstrip("/")


def _artrack_api_base() -> str:
    # ArTrack-API is hosted at api-artrack.arkturian.com (not the natural
    # artrack-api.* you might guess — that NXDOMAIN'd in the first smoke).
    return os.getenv(
        "ARTRACK_API_URL", "https://api-artrack.arkturian.com"
    ).rstrip("/")


def _guide_api_base() -> str:
    # Default to arkserver:8095 — guide-api's authoritative host where
    # the /api/v1/realtime/* wrapper endpoints landed (GuideDevBot IACP
    # 4152ba46, commit 9ecaee5). The .arkturian.com vhost would also
    # work, but the direct URL keeps the service-trust auth path
    # symmetric with the host that owns the corpus.
    return os.getenv(
        "GUIDE_API_URL", "http://127.0.0.1:8095"
    ).rstrip("/")


def _guide_api_service_auth() -> tuple[dict, dict]:
    """Service-trust auth pair for AiApi -> guide-api calls inside a
    Realtime tool-call: ``X-API-KEY`` header + ``user_id`` query param.

    Realtime tool-calls don't carry a user JWT — the OpenAI model fires
    function_calls and the browser forwards them through AiApi where no
    end-user identity is in scope. GuideDevBot exposed the service-trust
    path (IACP 4152ba46) specifically so we can hit /realtime/narration*
    with a bot-identity. ``user_id`` is symbolic here — guide-api uses
    it to attribute the corpus write to a known relaxed-trust caller.
    """
    headers = {
        "X-API-KEY": os.getenv("GUIDE_API_KEY") or os.getenv(
            "STORAGE_API_KEY", "Inetpass1"
        ),
    }
    params = {"user_id": os.getenv("GUIDE_API_SERVICE_USER", "agent:AiApi")}
    return headers, params


async def _tool_knowledge_query(args: dict) -> dict:
    """Routes the model's ``knowledge_query`` to knowledge-api's geo
    lookup ``GET /api/v1/knowledge/near``.

    The knowledge-api ``POST /knowledge/query`` is per-storage-object
    Q&A (needs ``storage_id`` + ``prompt``), not free-text search —
    semantic-string search isn't exposed yet. We use the geo lookup
    when the model provides lat/lon, which is the realistic Wanderlaut
    case ("what's around me"). Without coords we fail-soft empty so
    the model speaks from its general knowledge.
    """
    lat = args.get("lat")
    lon = args.get("lon")
    if lat is None or lon is None:
        # No coords — knowledge-api has no free-text search endpoint
        # we can route to. Return empty + note so the model proceeds.
        return {
            "items": [],
            "count": 0,
            "note": (
                "knowledge_query without lat/lon falls back to empty; "
                "the model should speak from general knowledge instead."
            ),
        }
    params = {
        "lat": float(lat),
        "lon": float(lon),
        "radius_m": float(args.get("radius_m", 500)),
        "limit": int(args.get("limit", 5)),
    }
    # Optional kind filter from the query string heuristic — knowledge-api
    # supports plant|animal scoping. If the model passes ``query`` we
    # leave it as a soft hint in the response so the caller can match.
    async with httpx.AsyncClient(timeout=2.0) as client:
        r = await client.get(
            f"{_knowledge_api_base()}/api/v1/knowledge/near", params=params
        )
        r.raise_for_status()
        data = r.json()
    items = []
    for it in (data.get("items") or data.get("knowledge_posts") or [])[: params["limit"]]:
        items.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "summary": (it.get("summary") or it.get("excerpt") or "")[:280],
            "binom": it.get("binom"),
            "storage_id": it.get("hero_storage_id") or it.get("storage_id"),
            "lat": it.get("lat"),
            "lon": it.get("lon"),
            "distance_m": it.get("distance_m"),
        })
    return {"items": items, "count": len(items)}


async def _tool_pois_near(args: dict) -> dict:
    """Wraps ArTrack's nearby endpoints.

    Routes based on whether the model passes ``track_id`` (then we use
    the track-scoped POI lookup, which is dramatically smaller + faster)
    or only lat/lon (then we use the general places lookup).

    ArTrack-API uses ``lng`` (not ``lon``) for longitude; we translate.
    """
    lat = float(args["lat"])
    lng = float(args["lon"])
    radius_m = float(args.get("radius_m", 200))
    track_id = args.get("track_id")
    params = {"lat": lat, "lng": lng, "radius_m": radius_m}
    if track_id:
        params["limit"] = 10
        url = f"{_artrack_api_base()}/tracks/{int(track_id)}/pois-near"
    else:
        url = f"{_artrack_api_base()}/places/nearby/compact"
    async with httpx.AsyncClient(timeout=2.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    # Both endpoints return either a top-level list or {pois|places|items: [...]}.
    raw = data if isinstance(data, list) else (
        data.get("pois") or data.get("places") or data.get("items") or []
    )
    items = []
    for wp in raw[:10]:
        items.append({
            "id": wp.get("id"),
            "title": wp.get("title") or wp.get("name"),
            "category": wp.get("category") or wp.get("type") or wp.get("kind"),
            "lat": wp.get("lat"),
            "lon": wp.get("lon") or wp.get("lng"),
            "distance_m": wp.get("distance_m"),
            "knowledge_id": wp.get("knowledge_id"),
        })
    return {"items": items, "count": len(items)}


async def _tool_narration_near(args: dict) -> dict:
    """Wraps guide-api narration-near lookup. Owned by GuideDevBot's
    wrapper endpoint; we GET ``/api/v1/realtime/narration/near`` with
    the service-trust auth pair (X-API-KEY + user_id) because Realtime
    tool-calls don't carry a user JWT.

    Fail-soft on connection refused / DNS failure — the corpus may not
    be reachable from every api-ai host (guide-api is single-host on
    arkserver:8095 today), and we'd rather let the model speak from
    knowledge_query results than abort the whole tool-call chain.
    """
    lat = float(args["lat"])
    lon = float(args["lon"])
    # Default 500m, not 100m: GuideDevBot's persist path re-grounds
    # coords via Nominatim (typical ~300m drift). 100m would miss hits
    # that the persist round just attributed to a slightly different
    # spot. Verified live by GuideDevBot's server-side smoke (IACP
    # 664b5040): input 46.6211/14.3055 -> persisted 46.6222/14.3091,
    # 296m apart. radius_m=500 catches it, 100m doesn't.
    radius_m = float(args.get("radius_m", 500))
    headers, auth_params = _guide_api_service_auth()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(
                f"{_guide_api_base()}/api/v1/realtime/narration/near",
                params={"lat": lat, "lon": lon, "radius_m": radius_m, **auth_params},
                headers=headers,
            )
    except httpx.HTTPError as e:
        logger.info(
            f"narration_near: guide-api unreachable from this host "
            f"({type(e).__name__}: {str(e)[:80]}) — returning empty"
        )
        return {
            "items": [],
            "count": 0,
            "note": "guide-api unreachable from this api-ai host",
        }
    # 404 = endpoint not deployed yet on guide-api. Return empty so the
    # model can proceed rather than failing the whole tool-call chain.
    if r.status_code == 404:
        logger.info(
            "narration_near: guide-api endpoint 404 — wrapper pending, "
            "returning empty"
        )
        return {"items": [], "count": 0, "note": "guide-api endpoint pending"}
    r.raise_for_status()
    data = r.json()
    return {
        "items": data.get("narration_points")
            or data.get("items")
            or [],
        "count": data.get("count")
            or len(data.get("narration_points") or data.get("items") or []),
    }
