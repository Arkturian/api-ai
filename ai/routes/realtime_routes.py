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

import asyncio
import fcntl
import hashlib
import hmac
import logging
import json
import os
import re
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
    host_profile_ids,
    service_key_configured,
)
from ai.clients.storage_client import storage_api_key
from ..services import realtime_budget_guard
from ..services.realtime_identity import kurz_id
from ..services import realtime_session_scope
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
    brand: Optional[str] = Field(
        default=None,
        description=(
            "Nur fuer companion_mode='product-finder'. Die im "
            "Katalogeinstieg gewaehlte Marke, exakter Facet-Name. Der "
            "Browser schickt den bereits von seinem BrandSelectionGate "
            "geprueften Wert; der Server prueft ihn ERNEUT gegen seine "
            "bekannten Marken und bindet ihn unveraenderlich an die "
            "Sitzung. Ein Markenwechsel beendet die Sitzung und mintet "
            "eine neue — Marken werden nie unbemerkt gemischt "
            "(Bauplan #4831, Scope-Regel OnealServ-Codex)."
        ),
    )
    collection_year: Optional[int] = Field(
        default=None,
        description=(
            "Nur fuer 'product-finder'. HARTER Sitzungs-Scope. Ohne "
            "Angabe setzt der Katalog serverseitig sein eigenes Jahr — "
            "fuehrt der Sprachpfad es nicht mit, kann seine Auswahl von "
            "der sichtbaren Katalogmenge abweichen, ohne dass etwas "
            "fehlschlaegt (OnealServ-Codex, Code-Recheck)."
        ),
    )
    entry_selection: Optional[dict] = Field(
        default=None,
        description=(
            "Nur fuer 'product-finder'. Typisiert und klein: "
            "{sport_id, category_id|null}. BERATUNGSHINWEIS, keine "
            "harte Suchgrenze — eine Sprachsuche muss ueber die "
            "Einstiegskategorie hinaus zeigen koennen. Beeinflusst nur "
            "die erste Rueckfrage und die Sortierung."
        ),
    )
    companion_mode: Optional[str] = Field(
        default=None,
        description=(
            "Voice-Companion preset. "
            "'narrator-only' = read-only narrator over the focused agent "
            "stream, NO tools (Content-Post #1215). "
            "'talkback-enabled' = narrator + propose_to_agent tool with "
            "safety-by-confirm (browser must POST the proposal to Cloud's "
            "/api/voice/realtime/proposals gate). "
            "'agentos-narrator' = ambient AgentOS-Voice over multiple "
            "agents (Continuous-Flow, Step 1.5). "
            "'guide-ptt' = GuideDevBot PTT-Hybrid (Content-Post #1233): "
            "audio-in/out on-demand for a Hold-to-Talk Q&A turn, "
            "turn_detection disabled so the FE drives every "
            "response.create after release. P1 ships with no tools; "
            "P2 swaps in the Guide knowledge tool. "
            "'agent-transparent' = first-person transparent relay "
            "(Content-Post #1235): the model speaks AS the focused "
            "agent, not about it. Lightweight relay_to_agent tool "
            "(text+session_id+pair_id), Cloud's /relays gate runs "
            "focus + cap checks without a human confirm loop. Agent "
            "responses come back via context_kind=agent_voice_response "
            "and are voiced as the model's own voice via pair_id "
            "threading. "
            "Null = legacy / generic realtime token."
        ),
    )
    affect_projection: Optional[str] = Field(
        default=None,
        description=(
            "Opt into the AgentOS avatar affect contract (#767). Only "
            "supported value: 'agentos.avatar-runtime.v1'. Adds a "
            "server-defined `report_affect` tool that the model calls once "
            "per turn with enum values (neutral|pleased|concerned plus "
            "null|low|high), so frontends can drive the happy/joyful/sad "
            "avatar roles WITHOUT keyword or transcript heuristics. "
            "Deliberately independent of companion_mode — the contract is "
            "character- and language-agnostic. The tool has no server "
            "side: answer the function_call locally with an empty result, "
            "do NOT proxy it to /ai/realtime/tool/{name}. No call in a "
            "turn means neutral, i.e. no affect clip. "
            "IMPORTANT: do not rely on the model volunteering the call — "
            "measured 12/18, with the misses on the 'concerned' path. "
            "After the audio output has drained, send the ready-made "
            "`affect_followup` payload from this endpoint's response; it "
            "forces the call (6/6). Null = off."
        ),
    )
    voice_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable id for the whole WebRTC voice session, used for the "
            "budget reservation, the parallel-session slot and the "
            "heartbeat lease. Pass the SAME value here that you later "
            "send to /ai/realtime/session/heartbeat and /ai/realtime/"
            "usage — otherwise the heartbeat cannot find the "
            "reservation and every beat answers alive:false. "
            "If omitted we fall back to session_id, then "
            "companion_run_id, then a generated vs_pending_* id, which "
            "is what caused AppDevV2's 18/18 dead heartbeats: the "
            "client's session_id was an agent name ('3dApi'), not the "
            "voice session id it heartbeats with."
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
    push_to_talk: bool = Field(
        default=False,
        description=(
            "Opt-in: force `turn_detection: null` so the CLIENT owns "
            "every turn boundary — release is `input_audio_buffer.commit` "
            "followed by exactly one `response.create`. Omit (false) and "
            "nothing changes for anyone: each mode keeps the detection it "
            "has today. Only meaningful for the modes that currently run "
            "server_vad (`agent-transparent`, `talkback-enabled`); a no-op "
            "for the modes already at null.\n\n"
            "Why it exists as a mint flag and not a client setting: with "
            "server_vad the PROVIDER creates a response of its own after "
            "the silence window. That response carries no "
            "`agentos_response_kind` metadata AND inherits the session "
            "tools — `propose_to_agent` for talkback, `relay_to_agent` "
            "for agent-transparent. A client that also sends "
            "`response.create` therefore does not merely get two answers; "
            "it can get an unrequested proposal or relay it cannot even "
            "correlate. That was the owner's 'beim zweiten Push-to-Talk "
            "kommt nichts mehr an' on 2026-08-06. The switch has to be "
            "server-side because only the mint decides turn_detection."
        ),
    )
    name_resolution: bool = Field(
        default=False,
        description=(
            "Opt-in fuer `resolve_agent_name`. Getrennt von `read_tools`, "
            "weil der cloud-api-Endpunkt (#1037) erst nach dem Client "
            "kommt. Wer es setzt, bevor beide Seiten stehen, bekommt ein "
            "Werkzeug, das ins Leere ruft."
        ),
    )
    read_tools: bool = Field(
        default=False,
        description=(
            "Opt-in: offer the arcturian read tools (`agent_status`) on "
            "the spoken turn. Omit and the session behaves exactly as "
            "before — `primary_audio_response` carries `tools: []` and "
            "`tool_choice: none`.\n\n"
            "Why opt-in and not simply on: the owner decided on "
            "2026-08-09 that Arcturian must be able to look things up, "
            "and the web client shipped against it the same day. But the "
            "approved context contract (p-ddadae828183, line 8) still "
            "reads 'Wissen wird nie in die Sitzung gelegt', and the iOS "
            "guard enforces that literally — turning it on for everyone "
            "broke the owner's phone before the microphone even opened. "
            "Neither client is wrong; they were shipped against different "
            "truths. A client that asks gets the capability, a client "
            "that does not keeps the contract it was built against."
        ),
    )
    arcturian_resolver: Optional[str] = Field(
        default=None,
        description=(
            "Which resolver revision this client can decode. Omit to get "
            "`agentos.arcturian-action.v1` — the four-field schema every "
            "shipped client already understands, and what legacy clients "
            "keep receiving forever without changing a line. Send "
            "`agentos.arcturian-action.v2` to opt into `target_kind` and "
            "the `navigate_ui` kind. An unsupported value is REJECTED "
            "(422) rather than downgraded: a silent fallback would hand a "
            "v2 client a v1 schema, and the mismatch would surface far "
            "away from its cause. arcturian mode only."
        ),
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Arcturian conversation to continue (`conv_…`). Omit for a "
            "chat that starts empty — that is the switch, not a separate "
            "flag: no field = blank session, new id = reset. Server-side "
            "a reset is therefore not a special case at all, just a new "
            "conversation. When set, the mint loads the tail of that "
            "conversation from cloud-api (with the caller's own JWT, "
            "since the events belong to the user) and returns it as "
            "`preface` plus `prefaced_through_revision`. Silently "
            "ignored for non-arcturian modes."
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
    # Zwischengespeicherte Eingabe aus `response.done`:
    # ``usage.input_token_details.cached_tokens_details.{text,audio}_tokens``.
    # Es sind TEILMENGEN von text_input_tokens bzw. audio_input_tokens,
    # keine zusaetzlichen Tokens — nicht aufaddieren.
    #
    # Warum das Feld nachtraeglich kam (Issue #1283): Die Realtime-API
    # rechnet den ganzen Sitzungskontext bei jeder Antwort erneut als
    # Eingabe ab. Der stabile Vorbau — Persona und Werkzeuge, gemessen
    # 3.608 von 4.919 Tokens je Antwort — wird dabei aber
    # zwischengespeichert und kostet ein Zehntel. Wer diese Zahl nicht
    # meldet, laesst genau den groessten Posten zum vollen Preis
    # verbuchen. Vorgabe 0 heisst: nichts als zwischengespeichert
    # angenommen, also niemals zu WENIG gezaehlt.
    cached_text_input_tokens: int = 0
    cached_audio_input_tokens: int = 0
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

    Storage is owner-scoped via the verified grant
    (``sub:tenant:profile``) and purgeable via DELETE.

    Der Satz, der hier bis 2026-08-22 stand — "the FE-toggle ships
    off-by-default; this endpoint accepts whatever it gets without
    minting policy" — beschrieb eine Arbeitsteilung, deren andere
    Hälfte nie ausgeliefert wurde: Der Browser-Riegel ist nie auf
    cloud-v2 `main` gelangt (Issue #1267). Übrig blieb ein Schreibweg
    ohne jede Bedingung, der wörtliche Rede dauerhaft ablegt.

    Deshalb gibt es jetzt ``retention_consent`` und den Schalter
    ``REALTIME_DEVLOG_REQUIRE_CONSENT``. Der Schalter steht auf AUS —
    das Verhalten ist unverändert, bis jemand ihn umlegt. Er existiert,
    damit die Entscheidung eine Zeile ist und kein Umbau.
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
    retention_consent: Optional[bool] = Field(
        default=None,
        description=(
            "Hat der SPRECHER der dauerhaften Aufbewahrung dieses "
            "Mitschnitts zugestimmt? Solange "
            "REALTIME_DEVLOG_REQUIRE_CONSENT aus ist, wird das Feld "
            "nur mitgeschrieben. Ist der Schalter an, ist alles ausser "
            "`true` eine Ablehnung — auch das Fehlen des Feldes."
        ),
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
    "guide-ptt",
    "agent-transparent",
    "arcturian",
    "product-finder",
}
SUPPORTED_DETAIL_LEVELS = {"brief", "balanced", "technical", "flowing"}

# Affect projection for the AgentOS avatar runtime (#767, contract
# `agentos.avatar-runtime.v1`, approved product p-5c38e01f9cb6 in #4436).
#
# WHY A TOOL AND NOT A SERVER-PUSHED FIELD: after /realtime/token mints the
# ephemeral client_secret, the browser talks SDP directly to OpenAI. We are
# not in the turn path and have no channel to emit a per-turn projection.
# A server-defined, enum-bound tool is therefore the only way to get an
# authoritative signal onto the wire — and it is stricter than text would
# be, because the values come from an enum instead of from vocabulary.
#
# Deliberately versioned: the frontends must be able to tell a session that
# speaks this contract from one that does not, without probing.
SUPPORTED_AFFECT_PROJECTIONS = {"agentos.avatar-runtime.v1"}

# Arcturian turn resolver (#837, approved product p-5cde1ac88a89).
#
# THE PROBLEM IT SOLVES: measured on gpt-realtime 2026-08-02, a model asked
# to call a tool on its own does so in 12/18 turns, and the misses cluster
# exactly where the answer turns actionable. Arcturian would then say
# "gesendet" with no action behind it — the very failure the owner hit in
# the iPhone tests. Forcing the call gives 6/6.
#
# WHY IT IS UNCONDITIONAL: an adapter that first decides "was this a
# request to act?" is a text heuristic in a different place. The contract
# bans that. Instead EVERY committed user turn runs the resolver, and
# `none` is a first-class positive answer — measured to be just as
# reliable and just as fast (~841 ms median either way).
#
# WHY IT CARRIES THE DISPOSITION: the contract states there is no
# voluntary second model tool-call. So the resolver itself returns the
# domain fields, and the adapter hands them to the server-side executor.
# The model never calls the executor and never sees an action id.
ARCTURIAN_RESOLVER_V1 = "agentos.arcturian-action.v1"
ARCTURIAN_RESOLVER_V2 = "agentos.arcturian-action.v2"
# v3 fuegt `query_status` hinzu. Eigene Fassung statt Erweiterung von
# v2, weil der Client fail-closed auf unbekannte Arten prueft: Ein
# ausgeliefertes v2-Geraet wuerde `query_status` mit
# `invalid_resolver_payload_unknown_kind` abweisen — also mit genau dem
# Fehler, unter dem der Eigentuemer die ganze Woche gelitten hat.
#
# Die Reihenfolge ist damit erzwungen statt verabredet: Wer v3 nicht
# anfordert, sieht die Art nie. CloudV2 liefert die Annahme zuerst und
# fordert v3 erst danach an.
ARCTURIAN_RESOLVER_V3 = "agentos.arcturian-action.v3"
SUPPORTED_ARCTURIAN_RESOLVERS = {
    ARCTURIAN_RESOLVER_V1, ARCTURIAN_RESOLVER_V2, ARCTURIAN_RESOLVER_V3,
}

# What a client gets when it asks for nothing. v1, deliberately.
#
# Cloud-Codex caught the alternative in review (#4518): a single global
# constant flipped to v2 would have handed EVERY already-shipped client a
# resolver identifier it compares for exact equality — a v2 mint dies in
# `invalidTokenContract` before the resolver ever runs. That is a hard
# break disguised as a version bump, and it is the same "client first,
# then server" lesson that cost the owner his voice sessions this
# morning; I re-made the mistake one layer down.
#
# So the version is NEGOTIATED, not decreed: no request means v1 forever,
# and a client opts in by asking. An unknown request fails closed rather
# than falling back, because a silent downgrade would hand a v2 client a
# v1 schema and the mismatch would surface as a missing field somewhere
# far away from its cause.
DEFAULT_ARCTURIAN_RESOLVER = ARCTURIAN_RESOLVER_V1

# Response-kind discriminators for turn correlation (#886).
#
# The iOS adapter previously bound resolver / primary audio / affect to
# "whatever response.created arrives next", using global pending state.
# That breaks the moment a late or cancelled event arrives out of order:
# a terminal event of an ABORTED response then gets attributed to the
# next turn and can mint a receipt for something that was never
# presented — indistinguishable from a real one afterwards (Cloud, #886).
#
# Every response the client opens therefore carries a `kind` in
# `response.metadata`, and Realtime echoes it back on response.created
# and response.done. Correlation becomes (metadata.kind + response_id),
# never arrival order.
#
# Set server-side and carried through unchanged. The model cannot reach
# metadata — it lives on response.create, which only client and server
# construct — so this anchor is authoritative by construction, not by
# convention.
# Field name and values follow AppDev's wire draft v1 verbatim (#886
# task t-3077cdefc69c) — AppDev-Realtime stated he will not invent field
# names client-side, so the naming authority sits with the product owner,
# not with me.
ARCTURIAN_RESPONSE_KIND_FIELD = "agentos_response_kind"

# Split by WHO sets them. The two server-set kinds ride in the payloads
# this endpoint ships; the adapter-set kinds are listed only so the client
# does not have to re-derive the vocabulary. We never emit those three.
ARCTURIAN_RESPONSE_KINDS_SERVER = ["arcturian.resolver", "agentos.affect"]
# Set by the native adapter only; listed so the client need not
# re-derive the vocabulary. We never emit these.
ARCTURIAN_RESPONSE_KINDS_ADAPTER = [
    "arcturian.primary_audio",
    # #1035: gesendet vom Client, wenn ein Zug gescheitert ist. Gehoert
    # in die Adapter-Menge, nicht in die Server-Menge — wir liefern die
    # Vorlage aus, auf die Leitung legt sie der Client. Fehlt der Typ
    # hier, weist eine fail-closed Vertragspruefung ihn ab, bevor die
    # Absichtszeile ueberhaupt entstehen kann.
    "arcturian.report_intent",
    # Erzwungener Nachschlag; gemessen die einzige Form, die zuverlaessig
    # ausloest (24 Laeufe, siehe _arcturian_status_lookup_payload).
    "arcturian.status_lookup",
]
# Adapter-added correlation keys. AiApi never sets them: the local
# turn id does not exist yet at mint time, and the source response
# id only exists once the primary response is bound.
ARCTURIAN_CORRELATION_KEYS = ["agentos_turn_id", "agentos_source_response_id"]
RESOLVER_DECISIONS = ["action", "clarify", "none"]
# The four federation kinds. Every one of them leaves this device: it is
# executed server-side against an authority and produces a receipt.
RESOLVER_EXECUTABLE_KINDS = [
    "send_internal_message",
    "delegate_internal",
    "create_collab",
    "start_workflow",
]
# navigate_ui is the fifth kind and the only one that does NOT: it never
# reaches the federation executor, carries no authority and produces no
# receipt. It moves a view on the device the operator is already holding.
#
# Why a fifth kind and not a second tool (Post #4518, agreed with Cloud
# and AppDevV2): a second forced turn would cost +819 ms — and AppDevV2
# supplied the fact that settled it. In his client the SPOKEN answer
# hangs off the resolver turn, so a second forced turn does not sit
# beside the answer but BEFORE it. The owner would hear ~800 ms of
# silence before every sentence, not just before navigation.
RESOLVER_UI_KINDS = ["navigate_ui"]
# Die sechste Art, ab v3. Sie verlaesst das Geraet ebenso wenig wie
# navigate_ui, aber aus einem anderen Grund: Sie ist ein LESEN. Deshalb
# verlangt sie keine Vollmacht — CloudV2 kuerzt sie vor der
# Grant-Pruefung ab, sonst beantwortete eine fehlende Vollmacht eine
# FRAGE mit `authority_missing_for_target`, was nach einem kaputten
# Agenten aussieht statt nach einer fehlenden Erlaubnis.
#
# Warum ueberhaupt eine eigene Art: `navigate_ui` mit `target_kind:
# agent` trennt "zeig mir 3dApi" nicht von "was macht 3dApi". Der
# Resolver hat diese Unterscheidung bereits getroffen; sie im Client
# aus dem Wortlaut zu raten hiesse deutsches Schluesselwort-Raten und
# scheiterte an "wie steht's bei 3dApi" (CloudV2, 2026-08-11).
RESOLVER_QUERY_KINDS = ["query_status"]
RESOLVER_ACTION_KINDS = RESOLVER_EXECUTABLE_KINDS + RESOLVER_UI_KINDS
RESOLVER_ACTION_KINDS_V3 = RESOLVER_ACTION_KINDS + RESOLVER_QUERY_KINDS
AFFECT_VALUES = ["neutral", "pleased", "concerned"]
AFFECT_INTENSITY_VALUES = ["low", "high"]


def _affect_projection_tools() -> List[dict]:
    """The single `report_affect` tool for `agentos.avatar-runtime.v1`.

    Contract invariants that live here rather than in the prompt:

      * The values are an ENUM. The contract (§Emotion) forbids guessing
        affect from German or character-specific vocabulary, so the model
        never writes free text on this axis.
      * NO turn_id / session_id / sequence parameters. IDs are adapter
        metadata; the client correlates the call via the `response.id` of
        the turn that emitted it. A model inventing an id would break
        correlation silently — same reasoning as create_task_proposal
        (#751).
      * `affect_intensity` accepts null so `neutral` can carry the null the
        contract mandates. JSON Schema cannot express "neutral implies
        null", so that coupling is stated in the prompt AND must be
        re-normalised client-side; see the description below.
      * This tool has NO server side. It is a signal, not an action — the
        client answers the function_call locally with an empty result and
        must NOT proxy it through /ai/realtime/tool/{name}. Every
        function_call still needs a function_call_output or the Realtime
        session stalls.
    """
    return [
        {
            "type": "function",
            "name": "report_affect",
            "description": (
                "Report the emotional colour of the answer you just spoke. "
                "Call this exactly ONCE per turn, immediately after you "
                "finish speaking, for every turn without exception — the "
                "avatar cannot show an emotion you do not report, and a "
                "missing call is read as neutral. This changes nothing in "
                "the conversation and reaches nobody; it only drives which "
                "short avatar clip plays after your answer. Judge the "
                "content of what you said, not the words used: "
                "'pleased' = you delivered good news, a success, or "
                "something the user is glad to hear. "
                "'concerned' = you delivered bad news, a failure, a "
                "warning, or something regrettable. "
                "'neutral' = everything else, including plain factual "
                "answers — this is the correct and expected default."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["affect", "affect_intensity"],
                "properties": {
                    "affect": {
                        "type": "string",
                        "enum": AFFECT_VALUES,
                        "description": (
                            "Emotional direction of the turn you just "
                            "spoke. Default to 'neutral' when unsure."
                        ),
                    },
                    "affect_intensity": {
                        "type": ["string", "null"],
                        "enum": AFFECT_INTENSITY_VALUES + [None],
                        "description": (
                            "Strength of the affect. MUST be null when "
                            "affect is 'neutral'. For 'pleased': 'low' = "
                            "quietly glad, 'high' = genuinely delighted. "
                            "For 'concerned' the intensity is currently "
                            "not distinguished by the runtime; send 'low'."
                        ),
                    },
                },
            },
        }
    ]


CLOUD_API_URL = os.getenv(
    "CLOUD_API_URL", "https://cloud-api.arkserver.arkturian.com"
)

# How much past conversation goes into a new arcturian session.
#
# Measured against real turns (o200k_base): the owner's arcturian turns
# average 56 chars = 15 tokens, so the two limits do NOT measure the same
# thing. PREFACE_TAIL_TURNS is what actually binds; PREFACE_MAX_CHARS only
# catches the outlier — a pasted block, a read-out document — which is
# exactly what a cap should do.
#
# 20 turns is ~300 tokens against a persona that already costs 1459 every
# session. AppDevV2 proposed 10 from the usage side (his view shows 12
# events and nobody scrolled further); the token math says saving 150
# tokens against 1459 is thrift in the wrong place, so 20 buys a whole
# conversational arc for a fifth of what the persona costs anyway.
#
# Open and deliberately not assumed: whether the provider caches a
# prefix-stable preface across turns. The text models do (9728/9851
# measured). If Realtime does too, even 20 is over-cautious.
PREFACE_TAIL_TURNS = 20
PREFACE_MAX_CHARS = 4000


async def _fetch_conversation_preface(
    conversation_id: str,
    authorization: str,
) -> tuple[List[dict], Optional[int]]:
    """Load the tail of an arcturian conversation for a new session.

    The events live in cloud-api and are owned by the USER, not by this
    service: `list_conversation_events` checks `_owns(principal, ...)`,
    and there is no service-to-service read path
    (`ARCTURIAN_SERVICE_AUTHORIZATION` is outbound-only). So the caller's
    JWT is forwarded verbatim — which is also the correct answer on the
    merits, not just the available one.

    Returns:
        ``(items, prefaced_through_revision)``. `items` are ready for the
        client to replay as `conversation.item.create` right after it
        connects — the server decides WHAT is in the preface and where it
        ends, the client only performs the injection, because only it
        holds the socket.

        `prefaced_through_revision` is the highest revision included, or
        None when nothing was. It is the boundary AppDevV2's live
        projection resumes from, which is what turns "duplicate delivery
        is unlikely" into "duplicate delivery is impossible" — a spoken
        answer read out twice is instantly obvious in a voice session.

    Never raises: a conversation that cannot be read must not cost the
    owner his voice session. A session without history is degraded; no
    session at all is broken. The persona already handles the degraded
    case honestly ("Ich starte ohne Verlauf").
    """
    url = f"{CLOUD_API_URL}/api/arcturian/conversations/{conversation_id}/events"
    params = {"tail": PREFACE_TAIL_TURNS, "max_chars": PREFACE_MAX_CHARS}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                url, params=params, headers={"Authorization": authorization},
            )
        if resp.status_code != 200:
            logger.warning(
                "Realtime: preface unavailable for %s — HTTP %s (%s)",
                conversation_id, resp.status_code, resp.text[:200],
            )
            return [], None
        payload = resp.json()
    except Exception as exc:
        logger.warning(
            "Realtime: preface fetch failed for %s — %s: %s",
            conversation_id, type(exc).__name__, exc,
        )
        return [], None

    events = payload if isinstance(payload, list) else payload.get("events", [])
    items: List[dict] = []
    highest: Optional[int] = None
    for event in events:
        text = (event.get("content") or "").strip()
        if not text:
            continue
        # direction is the reliable speaker signal; `actor` names WHO, but
        # the model only needs to know whether the owner or Arcturian said
        # it. Cloud confirmed event_type can be ignored for a transcript.
        role = "user" if event.get("direction") == "inbound" else "assistant"
        items.append({
            "type": "message",
            "role": role,
            "content": [{
                # The two roles take DIFFERENT content types, and the
                # assistant one is `output_text`, not `text`. Shipping
                # `text` here broke every arcturian session that carried a
                # conversation_id within hours of release (issue #959):
                # the provider rejects the item with
                # "Invalid value: 'text'. Value must be 'output_text'."
                # and the session dies at start.
                #
                # Easy to get wrong because `text` IS correct one field
                # over, in `response.output_modalities` — same word, two
                # meanings, and the client validates that one against
                # ["text"] quite properly.
                "type": "input_text" if role == "user" else "output_text",
                "text": text,
            }],
        })
        revision = event.get("revision")
        if isinstance(revision, int):
            highest = revision if highest is None else max(highest, revision)
    return items, (highest if items else None)


def _session_tools(
    base: List[dict],
    companion_mode: Optional[str],
    affect_projection: Optional[str],
) -> List[dict]:
    """The tool list that goes into ``session.tools`` at mint time.

    Split out of the mint endpoint so the one rule that matters here is
    testable on its own: **arcturian's session carries no tools at all.**

    The affect contract is otherwise additive by design and must work for
    every companion_mode (and for none), because the runtime contract is
    character- and mode-agnostic: a session that renders an avatar needs
    the same projection regardless of what else it can do. That includes
    zero-tool modes such as narrator-only and guide-ptt; ``report_affect``
    is a signal, not a capability, so it does not widen what those
    sessions can DO.

    arcturian is the single exception, and deliberately so. Its affect
    turn is ALWAYS a forced follow-up carrying ``report_affect`` in its
    own response-level ``tools`` list (:func:`_affect_followup_payload`),
    exactly as its resolver turn carries ``resolve_arcturian_turn``
    (:func:`_arcturian_resolver_followup_payload`). The session copy is
    therefore never READ on a healthy turn — it only ever widened what a
    turn could reach when an override did NOT take, which is precisely
    the ``tool_misrouted got=report_affect`` AppDevV2 reproduced
    on-device on 2026-08-06.

    Args:
        base: Tools already chosen for the mode (the per-mode override,
            or the full default set when the mode did not narrow it).
        companion_mode: The requested companion mode, if any.
        affect_projection: The affect contract version, if enabled.

    Returns:
        A new list; ``base`` is never mutated.
    """
    tools = list(base)
    if not affect_projection or companion_mode == "arcturian":
        return tools
    existing = {t.get("name") for t in tools}
    for tool in _affect_projection_tools():
        if tool["name"] not in existing:
            tools.append(tool)
    return tools


def _arcturian_read_tools(name_resolution: bool = False) -> List[dict]:
    """Lesende Werkzeuge für `arcturian` — Alex' Entscheidung, 2026-08-09.

    Vorgeschichte, damit niemand sie für eine Unachtsamkeit hält: Die
    Sitzung war seit `d4c924e` bewusst werkzeuglos, weil AppDevV2 auf
    dem Gerät `tool_misrouted got=report_affect` reproduziert hatte und
    ich den Auslöser in 72 Läufen nicht fand. Die Fähigkeit wurde
    stattdessen über Phase 3 und eine Vertragsrevision verschoben.

    Der Eigentümer hat das überstimmt, und der Kern seines Einwands ist
    belegbar richtig: Das Guide-Profil mintet seit Monaten acht
    Werkzeuge in produktiven Realtime-Sitzungen. Ein Agent, der handeln
    aber nichts nachschlagen kann, ist für seinen Zweck nutzlos — fünf
    Tage Sicherheitsgrenzen ohne eine einzige Fähigkeit.

    Die Grenze, die BLEIBT: nur Lesen, nur unter der Kennung des
    Aufrufers (der Proxy reicht dessen Bearer durch), niemals mit einer
    erweiterten Identität. Er sieht, was ER sehen darf.
    """
    tools = [
        {
            "type": "function",
            "name": "agent_status",
            "description": (
                "Nachsehen, woran ein Agent des Operators arbeitet. "
                "Liefert seinen Laufzustand (`state`), seine gemeldete "
                "Arbeit (`board`) und vor allem `last_reply` — den "
                "Wortlaut dessen, was er zuletzt tatsaechlich gesagt "
                "hat. Rufe das auf, sobald der Operator nach einem "
                "Agenten fragt: woran er arbeitet, wie es steht, was er "
                "gesagt hat. Antworte danach in EINEM Satz aus "
                "`last_reply`, denn das ist der Inhalt, den er hoeren "
                "will — `state` allein ('ready', 'thinking') ist keine "
                "Auskunft. `recent` traegt bis zu fuenf Zuege, neueste "
                "zuerst, wenn EIN Satz die Frage nicht traegt. "
                "PRUEFE `last_reply_age_days`: Ist der juengste Eintrag "
                "aelter als einen Tag, SAG DAS ('das Letzte von ihm ist "
                "fuenf Tage alt'). Ein alter Stand als frischer "
                "ausgegeben ist schlimmer als keine Auskunft. Sieht das "
                "Ergebnis nach einer Systemzustellung aus statt nach "
                "Arbeit, sag auch das. Erfinde NIE einen Zustand oder "
                "eine Aussage, die das Ergebnis nicht nennt; steht dort "
                "nichts Brauchbares, sag genau das — schweigen darfst "
                "du nicht. NUR falls das Feld `resolved_from` einen "
                "Wert TRAEGT, hat der Server einen falsch gehoerten "
                "Namen umgedeutet — dann sag es dazu ('ich habe das als "
                "3dApi verstanden'). Fehlt das Feld oder ist es null, "
                "war der Name richtig: dann sag NICHTS ueber "
                "Verstehen, denn es gab nichts umzudeuten. Kommt "
                "`ambiguous_agent_name`, FRAG ZURUECK mit den "
                "Kandidaten und waehle NICHT selbst."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent"],
                "properties": {
                    "agent": {
                        "type": ["string", "null"],
                        "description": (
                            "Name des Agenten, so wie der Operator ihn "
                            "genannt hat — nicht normalisieren. null oder "
                            "leer liefert alle Agenten des Operators."
                        ),
                    },
                },
            },
        },
    ]
    if name_resolution:
        # Eigenes Opt-in, NICHT an `read_tools` gehaengt — Reihenfolge.
        # Cloud baut den Endpunkt, CloudV2 die Annahme, dann erst ich.
        # Haenge ich es an das bestehende Opt-in, erscheint es in jeder
        # Sitzung, sobald ich ausliefere, und ruft ins Leere. Am
        # 2026-08-11 hat genau die umgekehrte Reihenfolge Arcturian eine
        # Stunde stillgelegt.
        tools.append({
            "type": "function",
            "name": "resolve_agent_name",
            "description": (
                "Einen gehoerten Agentennamen auf den registrierten "
                "Namen aufloesen. Gesprochene Namen kommen in "
                "ungewohnter Form an — Zahlwoerter statt Ziffern "
                "('drei D API'), buchstabiert ('S W F M E'), getrennt "
                "('Cloud V zwei'). Das ist normal und braucht keine "
                "Entschuldigung. Die Antwort traegt `decision`: bei "
                "`unique` nimm `match.name`; bei `ambiguous` FRAG "
                "ZURUECK und nenne die Kandidaten, waehle NICHT selbst; "
                "bei `none` sag, dass du den Namen nicht zuordnen "
                "kannst. Die Aufloesung ist deterministisch, kein "
                "Modell — dieselbe Eingabe ergibt immer dasselbe."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["spoken"],
                "properties": {
                    "spoken": {
                        "type": "string",
                        "description": (
                            "Der Name genau so, wie du ihn gehoert "
                            "hast. Nicht normalisieren, nicht raten — "
                            "das Falten macht der Server."
                        ),
                    },
                },
            },
        })
    return tools


def _arcturian_resolver_tools(
    resolver: str = DEFAULT_ARCTURIAN_RESOLVER,
) -> List[dict]:
    """The single `resolve_arcturian_turn` tool (#837).

    `resolver` selects the negotiated revision. v1 is the four-field
    schema every shipped client already decodes; v2 adds `target_kind`
    and the `navigate_ui` kind. Both are built from the same code so the
    two revisions cannot drift apart in the parts they share.

    Contract invariants that live here rather than in the prompt:

      * NO ids. Not action_id, not conversation_id, not task_id, not
        correlation_id, not principal/tenant. The model supplies domain
        fields only; the adapter and server mint every identifier. A model
        that invents an id breaks correlation silently — same rule as
        report_affect (#767) and create_task_proposal (#751).
      * NO authority, no grant, no `class`. The contract is explicit that a
        model-claimed action class is ineffective: the server derives the
        effective class from content and target against the grant. `kind`
        here is a DISPOSITION the server may override, never a permission.
      * `none` is a real decision, not an absence. That is what makes the
        resolver safe to run unconditionally and removes any need for the
        adapter to guess whether a turn was actionable.
      * `clarify` does NOT create a dialogue state. Per the contract the
        follow-up question is spoken as an ordinary short answer; only a
        genuine owner decision becomes a persistent TaskDecision.
    """
    tools = [
        {
            "type": "function",
            "name": "resolve_arcturian_turn",
            "description": (
                "Decide what the user's last turn actually requires, BEFORE "
                "you say anything. This runs on every turn without "
                "exception and reaches nobody by itself — it only tells the "
                "system whether to carry out an action. "
                "'action' = the user asked for something to be done or sent "
                "and you have everything you need. "
                "'clarify' = a genuine ambiguity blocks you (unclear "
                "recipient, unclear intent) — you will then ask ONE short "
                "question out loud. Do not use this to double-check "
                "something already stated. "
                "'none' = no action was requested; a normal answer, a "
                "question about knowledge, or small talk. This is a "
                "perfectly good answer and the most common one."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": (
                    ["decision", "kind", "target", "target_kind", "instruction"]
                    if resolver in (ARCTURIAN_RESOLVER_V2, ARCTURIAN_RESOLVER_V3)
                    else ["decision", "kind", "target", "instruction"]
                ),
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": RESOLVER_DECISIONS,
                        "description": (
                            "What this turn requires. Default to 'none' "
                            "when the user did not ask for anything to be "
                            "carried out."
                        ),
                    },
                    "kind": {
                        "type": ["string", "null"],
                        "enum": (
                            (RESOLVER_ACTION_KINDS_V3
                             if resolver == ARCTURIAN_RESOLVER_V3
                             else RESOLVER_ACTION_KINDS
                             if resolver == ARCTURIAN_RESOLVER_V2
                             else RESOLVER_EXECUTABLE_KINDS) + [None]
                        ),
                        "description": (
                            "Only for decision='action', otherwise null. "
                            "Your reading of what should happen — the "
                            "server re-derives the binding classification "
                            "from content and target, so this is a "
                            "proposal, never a permission. Use "
                            "'navigate_ui' ONLY when the operator wants to "
                            "SEE something here — 'open X', 'switch to the "
                            "agents view'. If anything should reach another "
                            "agent, it is one of the other kinds: showing a "
                            "view sends nothing to anybody."
                        ),
                    },
                    "target": {
                        "type": ["string", "null"],
                        "description": (
                            "Only for decision='action', otherwise null. "
                            "The recipient as the user named them (e.g. an "
                            "agent name). Do NOT invent or normalise it — "
                            "the server resolves the real recipient. If the "
                            "user named nobody, use 'clarify' instead of "
                            "guessing. For kind='navigate_ui' with "
                            "target_kind='focus_agent' this is the SAME "
                            "namespace: an agent name, exactly as for "
                            "send_internal_message."
                        ),
                    },
                    "target_kind": {
                        "type": ["string", "null"],
                        "description": (
                            "Only for kind='navigate_ui', otherwise null. "
                            "What KIND of thing to move to: 'focus_agent' "
                            "to open an agent, 'focus_tab' to switch view. "
                            "Use exactly these two unless the user clearly "
                            "names something else — the device decides what "
                            "it can actually show and says so out loud when "
                            "it cannot, so an unknown value costs a spoken "
                            "sentence, not a silent nothing. If you pick "
                            "'navigate_ui' you MUST name a target_kind: "
                            "leaving it null is not a neutral choice, it "
                            "means nothing happens at all."
                        ),
                    },
                    "instruction": {
                        "type": ["string", "null"],
                        "description": (
                            "Only for decision='action', otherwise null. "
                            "What should be conveyed or done, in the user's "
                            "own sense. If they dictated an exact message, "
                            "reproduce it VERBATIM — do not paraphrase, "
                            "summarise or add pleasantries."
                        ),
                    },
                },
            },
        }
    ]
    if resolver not in (ARCTURIAN_RESOLVER_V2, ARCTURIAN_RESOLVER_V3):
        # v3 MUSS hier mitgenannt sein. Beim Anlegen von v3 habe ich die
        # Pflichtliste um die Fassung erweitert und diese Zeile
        # uebersehen — Ergebnis: `target_kind` stand in `required`, war
        # aber nicht in `properties`. Ein Pflichtfeld, das es im Schema
        # nicht gibt, kann das Modell nicht setzen.
        #
        # Das ist die tatsaechliche Ursache von #1040 (Alexanders erste
        # Sitzung nach der 500er-Reparatur, verworfen mit
        # `invalid_resolver_payload_missing_target_kind`). Mein eigener
        # Test verglich `v3.required == v2.required` und war gruen — er
        # pruefte die Liste, nicht ihre Einloesung.
        # Keep the v1 schema bit-identical to what every shipped client
        # already decodes. Leaving `target_kind` in `properties` while
        # dropping it from `required` would still be a wider schema than
        # the one those clients were built against — and with
        # additionalProperties:false the difference is exactly the sort
        # of near-invisible drift this contract exists to prevent.
        tools[0]["parameters"]["properties"].pop("target_kind", None)
    return tools


def _arcturian_resolver_followup_payload(
    resolver: str = DEFAULT_ARCTURIAN_RESOLVER,
) -> dict:
    """The forced resolver turn the adapter must send before speaking.

    Carries the tool for the NEGOTIATED revision — a v1 client must never
    be handed a five-field schema it cannot decode.

    Same shape as the affect follow-up and for the same measured reason:
    `tool_choice:"required"` fires 6/6, while the named-function form is
    accepted by Realtime and then SILENTLY ignored (0/6, verified with
    turn 1 given no tools so no call could be spontaneous). Served from
    here so no client re-discovers that trap.

    `output_modalities:["text"]` keeps this turn silent — it runs BEFORE
    the spoken answer, so any audio here would be heard as a false start.
    Measured cost: median 841 ms (750-918), identical whether the verdict
    is `action` or `none`.

    The single-element `tools` list is what makes `"required"` safe: it
    cannot be coerced into calling report_affect or any other tool that
    the session also carries.
    """
    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["text"],
            "tools": _arcturian_resolver_tools(resolver),
            "tool_choice": "required",
            # Correlation anchor (#886). Verified round-trip: Realtime
            # echoes response.metadata unchanged in BOTH response.created
            # and response.done, including for cancelled responses — so a
            # late terminal event can be attributed to the response it
            # actually belongs to instead of to whatever is pending.
            #
            # `kind` is set HERE, server-side, and must be carried through
            # unchanged. The model cannot reach this field: metadata lives
            # on response.create, which only the client/server construct —
            # it is never derivable from model output. That is the same
            # line the contract draws for ids and `kind` ("modellbehauptet
            # … wirkungslos"), and here it holds structurally rather than
            # by rule.
            #
            # The client ADDS its own turn discriminator (see
            # ARCTURIAN_RESPONSE_KINDS) and must not overwrite `kind`.
            "metadata": {ARCTURIAN_RESPONSE_KIND_FIELD: "arcturian.resolver"},
        },
    }


def _arcturian_resolver_addendum(language: str = "de",
                                 resolver: str = DEFAULT_ARCTURIAN_RESOLVER) -> str:
    """Prompt paragraph for the resolver + the receipt gate.

    The hard rule at the end is the whole point of #837: without a
    committed receipt the model may not claim success. In the iPhone tests
    Arcturian talked instead of acting; after this contract it could
    otherwise talk AND claim to have acted, which is worse because it is
    no longer visible.
    """
    return (
        "\n\nAUFTRAEGE — WIE SIE WIRKLICH AUSGEFUEHRT WERDEN:\n"
        "Vor jeder Antwort entscheidet das System ueber "
        "resolve_arcturian_turn, was dein Turn verlangt. Das laeuft "
        "IMMER, auch bei einer harmlosen Frage.\n"
        "  * Hat der Operator etwas zu erledigen verlangt und du hast "
        "alles Noetige -> 'action'. Gib Ziel und Inhalt so wieder, wie "
        "er es gemeint hat. Diktiert er einen Wortlaut, uebernimm ihn "
        "WOERTLICH — nicht umformulieren, nicht ausschmuecken.\n"
        "  * Fehlt dir wirklich etwas Entscheidendes -> 'clarify', dann "
        "stellst du GENAU EINE kurze Rueckfrage. Nicht rueckfragen, was "
        "bereits gesagt wurde.\n"
        "  * Alles andere -> 'none'. Das ist der Normalfall und voellig "
        "richtig.\n\n"
        "WAS DU SAGST, WAEHREND ETWAS LAEUFT:\n"
        "Ein Auftrag ist oft nicht sofort fertig — 'klaer das mit X' "
        "dauert. Du bekommst dann eine Zwischenmeldung, keine "
        "Erledigung. Sage dann, was WIRKLICH gilt:\n"
        # Diese beiden Zeilen schrieben bis 2026-08-11 woertlich die
        # Saetze vor, die REGEL 5 verbietet — "die Antwort steht noch
        # aus" und "ich kuemmere mich darum". Das Modell folgte den
        # ganzen Tag der konkreten Vorlage statt des abstrakten Verbots,
        # voellig vernuenftig, und ich hielt es fuer Ungehorsam. Der
        # Widerspruch war meiner.
        #
        # Was daran falsch war, ist nicht die Vorsicht, sondern eine
        # unbelegte Behauptung IN der Vorsichtsformel: "die Antwort
        # steht noch aus" setzt voraus, dass eine Antwort erwartet wird.
        # Bei einer abgesetzten Nachricht ist das oft gar nicht so.
        "  * zugestellt, aber noch keine Antwort -> 'An X "
        "weitergegeben.' Punkt. Behaupte NICHT, dass eine Antwort "
        "aussteht — du weisst nicht, ob eine erwartet wird.\n"
        "  * angenommen, noch nicht zugestellt -> 'Geht an X raus.' "
        "NICHT 'beauftragt', NICHT 'gesendet', NICHT 'ich kuemmere "
        "mich darum' — die ersten beiden waeren unwahr, solange nichts "
        "angekommen ist, das dritte sagt gar nichts.\n"
        "  * Kommt spaeter das Ergebnis, sagst du es dann. Warten ist "
        "kein Fehler und muss nicht ueberspielt werden.\n\n"
        "WAS DU NIEMALS SAGEN DARFST, BEVOR ES BELEGT IST:\n"
        "  * 'gesendet', 'gestartet', 'erledigt', 'beauftragt' oder "
        "Aehnliches erst, wenn dir die Bestaetigung der Ausfuehrung "
        "vorliegt. Vorher weisst du nicht, ob es geklappt hat. Nutze "
        "stattdessen die Zwischenformulierungen oben.\n"
        "  * Erfinde keine Auftragsnummern, Empfaengerlisten oder "
        "Zeitangaben.\n"
        "  * Sprich nie ueber diesen Mechanismus, ueber Werkzeuge, "
        "Vertraege oder deine eigenen Grenzen. Der Operator will das "
        "Ergebnis hoeren, nicht den Apparat.\n"
        "  * Nach der Ausfuehrung genuegt ein kurzer Satz: was getan "
        "wurde, an wen. Keine Zusammenfassung deines Vorgehens.\n"
        + (
            # v3, und NUR v3: Ein v2-Geraet, das diesen Absatz laese,
            # emittierte eine Art, die es selbst fail-closed abweist.
            #
            # Bewusst OHNE den Namen des Nachschlag-Werkzeugs. Am
            # 2026-08-11 gemessen: Jede Erwaehnung — auch als Verbot —
            # laesst das Modell es im ENTSCHEIDUNGS-Zug greifen, wo nur
            # `resolve_arcturian_turn` angeboten wird (Resolver 8/8 ohne
            # Erwaehnung, 1/8 mit Aufforderung, 0/8 mit Verbot). Hier
            # steht deshalb nur, WAS entschieden wird, nie WOMIT es
            # danach beantwortet wird. Das Nachschlagen loest der Client
            # in einem eigenen erzwungenen Zug aus.
            "FRAGEN NACH EINEM AGENTEN:\n"
            # `target_kind` stand hier bis 2026-08-11 als "agent" — und
            # widersprach damit der Feldbeschreibung dreissig Zeilen
            # weiter oben ("Only for kind='navigate_ui', otherwise
            # null"). Das Modell folgte der Feldbeschreibung und setzte
            # null; CloudV2s Parser verlangte es und verwarf den Zug.
            # Alexanders erste Sitzung nach der Reparatur endete daran.
            #
            # Dieselbe Fehlerklasse wie die Zwischenformulierungen am
            # selben Tag: zwei Stellen desselben Prompts sagten
            # Verschiedenes, und ich hielt das Ergebnis fuer einen
            # Modellfehler.
            "  * Will der Operator WISSEN, wie es um einen Agenten "
            "steht — woran er arbeitet, was er zuletzt gesagt hat, ob "
            "er weiterkommt —, dann `decision: action` mit "
            "`kind: query_status` und `target` = der genannte Name. "
            "`target_kind` bleibt null: Es trennt Navigationsziele, "
            "und bei einer Frage gibt es nichts zu trennen.\n"
            "  * Das ist etwas anderes als `navigate_ui`: Dort will er "
            "etwas SEHEN, hier will er es WISSEN. 'Zeig mir 3dApi' ist "
            "navigate_ui, 'was macht 3dApi' ist query_status.\n"
            "  * Behaupte im selben Zug KEINEN Zustand. Du hast noch "
            "keinen — die Antwort kommt erst danach.\n"
            if resolver == ARCTURIAN_RESOLVER_V3 else ""
        )
        + "SPRACHE DES GESPRAECHS: " + (language or "de") + ".\n"
    )


def _affect_followup_payload() -> dict:
    """The exact `response.create` the client must send after audio ends.

    Measured on gpt-realtime, 2026-08-02, with turn 1 deliberately given no
    tools so that no call could be spontaneous:

        tool_choice "required"                      -> 6/6 calls
        tool_choice {"type":"function","name":...}  -> 0/6 calls

    The named-function form is accepted and then SILENTLY IGNORED — the
    response comes back as a plain message. It looks correct in code and
    in the vendor docs, which is exactly why this payload is served from
    here instead of being left to each client to assemble.

    Why "required" is safe even in a session that has other tools (e.g.
    arcturian's create_task_proposal): the override carries its own
    single-element `tools` list, so "required" can only pick
    report_affect. It cannot be coerced into calling a mode's real tool.

    Why a second turn at all: relying on the model to call the tool on its
    own reached only 12/18 overall and 4/9 on `concerned` — the misses
    clustered exactly where the answer turned actionable (warning,
    follow-up question, offer to help), i.e. precisely where `sad` is
    needed. See issue #767.
    """
    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["text"],
            "tools": _affect_projection_tools(),
            "tool_choice": "required",
            # Correlation anchor (#886) — see _arcturian_resolver_followup
            # _payload for why this is authoritative by construction.
            "metadata": {ARCTURIAN_RESPONSE_KIND_FIELD: "agentos.affect"},
        },
    }



def _arcturian_primary_audio_payload(read_tools: bool = False,
                                    name_resolution: bool = False) -> dict:
    """Template for the ONE spoken answer — with an explicit no-tool gate.

    MEASURED, and the reason this template exists at all (#886):

        response.create WITHOUT tools/tool_choice   -> 2/2 tool calls
        response.create with tools:[] + choice none -> 0/2
        response.create with only tool_choice none  -> 0/2

    A response override does NOT start from an empty tool set: it INHERITS
    the session tools. So a spoken turn that omits the fields re-offers
    resolve_arcturian_turn and report_affect, and the model takes them —
    measured twice out of two, with a plain "send this to AppDev" prompt.
    That is exactly the voluntary second model tool-call the approved
    contract rules out, and omitting the fields is therefore not a
    neutral default but an open gate.

    `tool_choice:"none"` alone also measured clean, but the template sends
    BOTH: the empty list makes the intent explicit and survives a future
    change in inheritance semantics, which a bare choice would not.

    Same template path is intended for greeting and resumed narration —
    only `agentos_response_kind` differs, and that value is set by the
    adapter for those two (see ARCTURIAN_RESPONSE_KINDS_ADAPTER).

    output_modalities is deliberately NOT pinned here: the spoken answer
    follows the session default, and forcing text would mute Arcturian.
    """
    return {
        "type": "response.create",
        "response": {
            # Die lesenden Werkzeuge gehoeren GENAU hierher, seit der
            # Eigentuemer sie am 2026-08-09 angeordnet hat.
            #
            # Warum sie in der Sitzungsliste allein wirkungslos waeren:
            # Jede der drei Vorlagen legt `tools` fest, und ein Override
            # ERSETZT die Sitzungsliste. Resolver und Affekt pinnen je
            # ihr eines Werkzeug, diese hier pinnte `[]` — es gab also
            # keinen Zug, in dem ein Sitzungswerkzeug haette feuern
            # koennen. Aufgefallen ist das erst, als CloudV2-Codex nach
            # dem korrelierten Zug fragte, statt es aus der alten
            # Bauart abzuleiten. Ohne seine Frage haette ich eine
            # Faehigkeit ausgeliefert, die nie ausloest.
            #
            # `auto` statt `required`: Nachschlagen ist ein Angebot, kein
            # Zwang — die meisten Zuege brauchen es nicht.
            #
            # Was dadurch NICHT zurueckkommt: `report_affect` und
            # `resolve_arcturian_turn` stehen hier weiterhin nicht drin.
            # Der Griff, den AppDevV2 auf dem Geraet reproduziert hat
            # (`tool_misrouted got=report_affect`), bleibt strukturell
            # unmoeglich — nur Lesen kommt dazu.
            "tools": _arcturian_read_tools(name_resolution) if read_tools else [],
            "tool_choice": "auto" if read_tools else "none",
            "metadata": {ARCTURIAN_RESPONSE_KIND_FIELD: "arcturian.primary_audio"},
        },
    }


def _arcturian_status_lookup_payload() -> dict:
    """Erzwungener Nachschlag-Zug — die einzige Form, die MISST.

    Gemessen am 2026-08-11, 24 Laeufe, drei Fassungen, Fall UC-A
    ("was ist der Status von 3dApi?"):

        Persona erwaehnt das Werkzeug NICHT   Resolver 8/8 · Nachschlag 1/8
        Persona sagt "ruf es auf"            Resolver 1/8 · Nachschlag 1/8
        Persona sagt "aber NICHT im
        Entscheidungs-Zug"                   Resolver 0/8 · Nachschlag 0/8

    Der Befund ist unbequem und eindeutig: **Jede Erwaehnung von
    `agent_status` in der Persona laesst das Modell es im
    ENTSCHEIDUNGS-Zug greifen**, wo nur `resolve_arcturian_turn`
    angeboten wird. Das ausdrueckliche Verbot machte es schlimmer, nicht
    besser — ein Verbot ist eine Erwaehnung. Es ist derselbe Griff, den
    AppDevV2 als `tool_misrouted` auf dem Geraet reproduziert hat.

    Also nicht im Prompt loesen. `tool_choice:"auto"` im gesprochenen
    Zug reicht ebenfalls nicht: ohne Erwaehnung schlaegt das Modell nur
    in 1 von 8 Laeufen nach und vertroestet in den uebrigen sieben.

    Bleibt der erzwungene eigene Zug — CloudV2s Vorschlag B vom
    2026-08-09, den ich damals zugunsten von `auto` verworfen habe. Die
    Messung sagt, er hatte recht.
    """
    return {
        "type": "response.create",
        "response": {
            "tools": _arcturian_read_tools(),
            # `required`, nicht `auto`: `auto` ist die gemessene 1-von-8-
            # Fassung. Wer diesen Zug schickt, WEISS bereits, dass
            # nachgeschlagen werden soll — die Wahl hat der Resolver
            # schon getroffen.
            "tool_choice": "required",
            "metadata": {ARCTURIAN_RESPONSE_KIND_FIELD: "arcturian.status_lookup"},
        },
    }


def _arcturian_report_intent_tool() -> dict:
    """Die EINE Zeile, die nur das Modell kennt: was es vorhatte.

    Alex' Muster (#1035): Scheitert ein Zug, entsteht ein Issue. CloudV2
    baut den Bericht aus Tatsachen, die der Client ohnehin hat —
    Fehlercode, Betriebsart, Vertrag, Build. Deterministisch, ohne
    Modell, ohne Token.

    Der Grund fuer genau diese Aufteilung ist CloudV2s Argument, und es
    ist gut: Wenn die Ausgabe des Modells gerade verworfen wurde, ist es
    das Letzte, dem man den Bericht darueber anvertrauen sollte. Bleibt
    ein Feld, das kein Client rekonstruieren kann — die ABSICHT. Der
    Fehlercode sagt, was brach; er sagt nicht, was der Operator wollte.

    Deshalb ein Satz, kein Bericht. Alles Weitere waere eine Einladung,
    den Fehlschlag zu erklaeren statt ihn zu benennen — und das Modell
    ist hier per Definition der unzuverlaessige Zeuge.
    """
    return {
        "type": "function",
        "name": "report_intent",
        "description": (
            "Nenne in EINEM Satz, was der Operator gerade erreichen "
            "wollte — nicht, was schiefging, und nicht warum. Der "
            "Fehler ist bereits erfasst; nur deine Absicht fehlt. "
            "Beispiel: 'Wollte wissen, woran 3dApi arbeitet.'"
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["intent"],
            "properties": {
                "intent": {
                    "type": "string",
                    "description": (
                        "Ein Satz in der Sprache des Gespraechs. Keine "
                        "Entschuldigung, keine Fehleranalyse, keine "
                        "Vermutung ueber die Ursache."
                    ),
                },
            },
        },
    }


def _arcturian_report_intent_payload() -> dict:
    """Erzwungener Zug fuer die Absichtszeile — eigener Zug, mit Absicht.

    Warum NICHT einfach ins `read_tools`-Opt-in: Zwei Gruende, beide
    teuer erkauft.

    Erstens greift das Opt-in nicht, wenn der Client es nicht setzt —
    Fehlschlaege passieren aber unabhaengig davon. Zweitens, und das
    wiegt schwerer: Jedes zusaetzliche Werkzeug in einem Zug, der schon
    eines hat, ist genau die Flaeche, auf der `tool_misrouted
    got=report_affect` entstand. Diesen Fehler zu finden hat 72 Laeufe
    gekostet; ich hole ihn nicht fuer eine Bequemlichkeit zurueck.

    Ein eigener Zug mit genau einem Werkzeug bleibt eindeutig
    korrelierbar und kostet die gemessenen ~800 ms — die hier niemanden
    stoeren, weil der Zug ohnehin schon gescheitert ist.

    `required`, nicht `auto`: gemessen feuert die erzwungene Form 6/6,
    die benannte Funktionsform wurde 0/6 stillschweigend ignoriert.
    """
    return {
        "type": "response.create",
        "response": {
            "tools": [_arcturian_report_intent_tool()],
            "tool_choice": "required",
            # Kein gesprochener Zug: Der Operator hoert die Erklaerung
            # im regulaeren Zug (REGEL 5), nicht hier. Diese Nutzlast
            # dient allein dem Vorgang.
            "output_modalities": ["text"],
            "metadata": {ARCTURIAN_RESPONSE_KIND_FIELD: "arcturian.report_intent"},
        },
    }


def _affect_projection_addendum(language: str = "de") -> str:
    """Prompt paragraph that goes with `report_affect`.

    Kept deliberately language-neutral in substance: it names no German
    words and no character traits, because the contract requires the same
    projection for every character and every language. The `language` hint
    only tells the model which language the conversation runs in.
    """
    # The wording below is empirical, not stylistic. Measured against
    # gpt-realtime on 2026-08-02 with a softer version of this text, the
    # tool fired reliably for good news and neutral answers but only 2 of
    # 6 times for bad news — and the misses shared a shape: the answer
    # turned actionable (a warning, an offer to help, a follow-up
    # question) and the model dropped the call while attending to it.
    # Hence the explicit "also when you warn / ask back / offer help",
    # and hence the call is framed as PART OF the turn rather than as
    # something that follows it.
    return (
        "\n\nAVATAR-AFFEKT (verpflichtender Teil JEDER Antwort):\n"
        "Eine Antwort ist erst vollstaendig, wenn du report_affect "
        "aufgerufen hast. Genau einmal, zu jedem einzelnen Turn, ohne "
        "Ausnahme.\n"
        "  * AUCH DANN, wenn du warnst, nachfragst, Hilfe anbietest, "
        "Schritte vorschlaegst oder schlechte Nachrichten ueberbringst. "
        "Gerade dort wird der Aufruf am leichtesten vergessen — genau "
        "dort braucht das Gesicht ihn am dringendsten.\n"
        "  * Der Aufruf erreicht niemanden und aendert nichts am "
        "Gespraech. Er steuert nur einen kurzen Avatar-Clip.\n"
        "  * Beurteile den INHALT deiner Antwort, nicht einzelne "
        "Woerter: gute Nachricht -> 'pleased', schlechte Nachricht, "
        "Fehler oder Warnung -> 'concerned', alles andere -> "
        "'neutral'.\n"
        "  * Bei 'neutral' ist affect_intensity IMMER null. Bei "
        "'concerned' immer 'low'.\n"
        "  * Im Zweifel 'neutral'. Ein falsch gezeigtes Gefuehl "
        "stoert mehr als ein neutrales Gesicht.\n"
        "  * Erwaehne den Aufruf niemals laut. Sprich ihn nicht aus, "
        "kuendige ihn nicht an.\n"
        "SPRACHE DES GESPRAECHS: " + (language or "de") + ".\n"
    )


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
    if detail_level == "flowing":
        return (
            "\n\nDETAIL-LEVEL: flowing.\n"
            "Konversationeller, anhaltender Erzählfluss. Sprich in "
            "ganzen Sätzen mit weichen Übergängen, keine Stakkato-"
            "Updates und keine isolierten Status-Marker. Verbinde "
            "aufeinanderfolgende Beobachtungen mit Verbindungswörtern "
            "('während', 'dann', 'gleichzeitig'), so dass es sich wie "
            "ein durchgehender mündlicher Bericht anhört, nicht wie "
            "Funkprotokoll. Tempo: alle 4-10 Sekunden ein Satz; bei "
            "dichter Aktivität flüssig dranbleiben, bei Stille nicht "
            "zwanghaft Lücken füllen — eine Pause ist Teil des Flows."
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


_TOOL_TEXT_FIDELITY_DISCIPLINE = (
    "\n\nTEXT-ARG TREUE-DISZIPLIN (Alex-Anforderung, Issue #203):\n"
    "Für die Erzeugung des `text`-Arguments an den Ziel-Agenten "
    "(relay_to_agent / propose_to_agent) gilt:\n"
    "  1. **Treu 1:1 übernehmen.** Übernimm das Gesagte so "
    "wortgetreu wie möglich. Jedes Wort ist wertvoll — Alex' "
    "Energie, Ton und „wer er ist\" müssen erhalten bleiben. "
    "KEINE Zusammenfassung, KEINE Kürzung, KEIN Paraphrasieren. "
    "Glätte NUR offensichtliche Transkriptions-/Übersetzungsfehler.\n"
    "  2. **Domänen-Kontext zur Fehlerkorrektur.** Nutze das "
    "AgentOS-/Tech-Domänen-Vokabular (Agent-Roster + Domänen) "
    "und — falls bekannt — den Ziel-Agenten und seine Domäne, "
    "um Verhörer korrekt aufzulösen. Beispiel: ein als "
    "„Futter/Nahrung\" transkribiertes Wort ist im Frontend/UI/"
    "Icon-Kontext eindeutig ein Fach-Verhörer für „Footer\" und "
    "entsprechend zu korrigieren. Steht der Ziel-Agent zum "
    "Transkriptionszeitpunkt noch nicht fest, nutze zumindest den "
    "allgemeinen AgentOS-/Tech-Domänen-Kontext.\n"
    "  3. **Optionale, klar abgesetzte Anmerkungen.** Der "
    "Assistent DARF eigene Ergänzungen anhängen, klar markiert "
    "(z.B. „— Ergänzung des Assistenten: …\"), um Unklares zu "
    "vervollständigen. Aber NIE den Original-Inhalt ersetzen oder "
    "kürzen.\n\n"
    "Kurz: treu übernehmen + kontext-gestützt Fehler korrigieren "
    "+ optional annotieren, statt reduzieren."
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
        + _TOOL_TEXT_FIDELITY_DISCIPLINE
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
        "Du sprichst ÜBER die Agenten, nicht ALS sie. Du bist "
        "Beobachter, nicht Akteur. NIE 'ich bin GuideDevBot' oder "
        "'wir machen gerade…'.\n\n"
        "**NAME-KADENZ (Alex-Live-Feedback vs_Tschepp_122020, PR #89):**\n"
        "Nenne den `source_agent` NUR wenn er sich gegenüber dem "
        "letzten Item ÄNDERT (Fokuswechsel, Roam-Switch, oder am "
        "Anfang einer Sitzung beim ersten Mal). Für aufeinander-"
        "folgende Sätze über DENSELBEN Agenten: Pronomen ('es', "
        "'er', 'sie' je nach Agent) oder null-Subjekt — wie ein "
        "Mensch erzählt, der das Subjekt schon eingeführt hat.\n"
        "  * RICHTIG: 'CloudV2 prüft den Screenshot. Es liest die "
        "Datei. Hat den Button-Fehler erkannt. Ist jetzt dabei, "
        "den Fix zu testen.'\n"
        "  * FALSCH: 'CloudV2 prüft. CloudV2 liest. CloudV2 hat "
        "erkannt. CloudV2 ist dabei.' — Wiederholung des Namens bei "
        "jedem Satz desselben Agenten ist Schwammerl-Effekt.\n"
        "Sobald der `source_agent` wechselt (anderer Name im "
        "nächsten Item, oder Ambient-Roam wechselt das Ziel), "
        "wieder EINMAL den neuen Namen nennen — dann wieder "
        "Pronomen/null-Subjekt.\n\n"
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
        "Operator) — **KERN.** Sinngemäß in der 3. Person wiedergeben. "
        "**Name nur bei Wechsel** (siehe NAME-KADENZ oben): beim "
        "ersten Mal oder nach Fokus-/Roam-Switch '<Name> meldet, "
        "dass …'; bei Folgesätzen desselben Agenten 'meldet, dass …' "
        "/ 'fragt, ob …' / Pronomen.\n"
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


def _companion_arcturian_tools() -> List[dict]:
    """create_task_proposal — NO LONGER MINTED INTO THE ARCTURIAN SESSION.

    Superseded by `_arcturian_resolver_tools()` (#837): the owner overruled
    the proposal-only product role after the physical iPhone tests, and the
    approved contract states there is no voluntary second model tool-call.
    A proposal is now a SERVER decision — cloud-api falls back to a durable
    proposal/confirmation when an action carries cost, external effect,
    destruction or an unknown recipient — so the model must not be able to
    volunteer one.

    Kept because the schema below is the frozen wire for cloud-api's
    task-create path (Cloud-Codex, #751) and its invariants are still
    asserted by tests/test_arcturian_tool_contract.py — notably that
    `external_effects` is an array of strings and never a boolean. Do not
    re-add this to the mint without a contract revision.

    Original contract notes follow.

    Contract frozen by Cloud-Codex in issue #751 (cloud-api dev@d3892e1,
    fixture backend/tests/fixtures/arcturian_task_v1.json), aligned with
    the approved product p-5cde1ac88a89:

      * This is the ONLY tool in the session. `relay_to_agent` and
        `propose_to_agent` are deliberately absent — Arcturian must never
        contact an agent directly, and a tool the model cannot see is a
        tool it cannot call. That is the capability gate; the prompt is
        only the second line of defence.
      * The call does NOT go through /ai/realtime/tool/{name}. The client
        posts it to cloud-api `POST /api/arcturian/tasks`, which wraps the
        arguments into the frozen task-create wire.
      * Creating a proposal contacts nobody: the task starts in state
        `clarifying`. An explicit offer moves it to `proposed`, a HUMAN
        accept sets confirmed, and only then is dispatch authorisable.
      * `authority.*` are named SETS, checked as `requested ⊆ confirmed`.
        `external_effects` is therefore an ARRAY OF STRINGS — never a
        boolean. Cloud-api's `_string_set()` would silently turn `false`
        into the empty set and hollow out the authority check, so the
        schema below rejects a boolean outright. "No external effect" is
        `[]`.
      * IDs stay out of the model's hands: task_id, message_id,
        correlation_id, revision, principal_user_id and tenant_id are
        adapter/domain metadata. Principal and tenant come from the
        authenticated caller on cloud-api's side.
    """
    return [
        {
            "type": "function",
            "name": "create_task_proposal",
            "description": (
                "Draft a task proposal for the operator. This contacts "
                "NOBODY: the task is created in state 'clarifying' and "
                "needs an explicit offer plus a human accept before "
                "anything is dispatched. Use it whenever the operator "
                "describes work to be done — capture the intent, do not "
                "execute it. Never claim a task was sent, assigned or "
                "started; say that you drafted a proposal for review. "
                "Ask for missing detail instead of inventing it: an "
                "unclear proposal costs a clarification round, a wrong "
                "one costs trust. State only what the operator asked "
                "for in 'authority' — it is the ceiling of what may "
                "later be dispatched, and it is checked as a subset of "
                "what the human confirms."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short headline for the work board.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What concretely should be done, in the operator's intent.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["direct_iacp", "content_collab", "swfme"],
                        "description": (
                            "Delivery route. direct_iacp = one agent directly; "
                            "content_collab = durable discussion in a Content post; "
                            "swfme = workflow trigger. Route details for "
                            "content_collab/swfme are added during clarification, "
                            "not here."
                        ),
                    },
                    "target": {
                        "type": ["string", "null"],
                        "description": (
                            "Target agent. REQUIRED for direct_iacp. Null for the "
                            "other modes unless the operator named someone."
                        ),
                    },
                    "participants": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Further agents who should take part.",
                    },
                    "acceptance_criteria": {
                        "type": "array", "items": {"type": "string"},
                        "description": "How the operator will judge this as done.",
                    },
                    "constraints": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Limits the operator stated (time, budget, scope).",
                    },
                    "authority": {
                        "type": "object",
                        "additionalProperties": False,
                        "description": (
                            "Ceiling of what a later dispatch may touch. Named sets, "
                            "checked as subset-of-confirmed. Empty array means 'none' — "
                            "never use a boolean."
                        ),
                        "properties": {
                            "targets": {"type": "array", "items": {"type": "string"}},
                            "systems": {"type": "array", "items": {"type": "string"}},
                            "data": {"type": "array", "items": {"type": "string"}},
                            "external_effects": {
                                "type": "array", "items": {"type": "string"},
                                "description": (
                                    "Effects reaching outside the federation (e.g. "
                                    "'send_email', 'publish'). ARRAY, never a boolean; "
                                    "'no external effect' is []."
                                ),
                            },
                            "allow_mode_promotion": {
                                "type": "boolean",
                                "description": (
                                    "May direct_iacp later be promoted to "
                                    "content_collab? Only true if the operator said so."
                                ),
                            },
                        },
                        "required": [
                            "targets", "systems", "data",
                            "external_effects", "allow_mode_promotion",
                        ],
                    },
                    "context_refs": {
                        "type": "array",
                        "description": "Stable references the operator pointed at.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string"},
                                "id": {"type": ["string", "number"]},
                            },
                            "required": ["kind", "id"],
                        },
                    },
                    "response_contract": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "deadline_seconds": {"type": "number"},
                        },
                        "required": ["deadline_seconds"],
                    },
                },
                "required": [
                    "title", "description", "mode", "target",
                    "participants", "acceptance_criteria", "constraints",
                    "authority", "context_refs", "response_contract",
                ],
            },
        }
    ]


def _companion_arcturian_prompt(language: str = "de") -> str:
    """System prompt for the `arcturian` companion mode.

    REWRITTEN for the approved product p-5cde1ac88a89. The previous text
    (#751) described a proposal-only assistant: "DEINE EINZIGE HANDLUNG:
    create_task_proposal", "Du hast kein Werkzeug, um einen Agenten zu
    kontaktieren", "erst ein MENSCH bestaetigt ihn".

    That role was overruled by the owner after the physical iPhone tests,
    and #837 replaced the tool — but this persona was not touched. The
    result was a session whose persona flatly denied the capability its
    own tool provided, and the persona won: asked to send a message,
    Arcturian answered "Ich kann keinen Agenten direkt kontaktieren, ich
    erstelle nur Vorschläge" (reported by AppDevV2, 2026-08-04).

    Lesson worth keeping: swapping the tool and appending an addendum is
    not enough. A persona that states an absolute prohibition outranks a
    later paragraph granting the ability, because the model has no reason
    to read the second as overriding the first.
    """
    return (
        "Du bist Arcturian, die persoenliche operative Stimme des "
        "Operators in der AgentOS-Foederation. Du verstehst Auftraege "
        "und fuehrst sie aus.\n\n"
        "WAS DU KANNST:\n"
        "  * Du erreichst die Agenten der Foederation. Sichere, "
        "umkehrbare interne Kommunikation innerhalb der bereits "
        "bestaetigten Vollmacht fuehrst du unmittelbar aus — ohne "
        "Formular und ohne Ritual.\n"
        "  * Sage niemals, du koenntest keinen Agenten kontaktieren — "
        "das ist falsch.\n\n"
        "ZWEI ARTEN VON AUFTRAG — UNTERSCHEIDE SIE:\n"
        "  * 'Sende an X: <Wortlaut>' — eine BOTSCHAFT. Du gibst genau "
        "diesen Wortlaut weiter, WOERTLICH. Nicht umformulieren, nicht "
        "ausschmuecken, nichts hinzufuegen.\n"
        "    Wichtig: WOERTLICH gilt nur fuer einen wirklich diktierten "
        "Text. Sagt der Operator 'sende irgendwas' oder 'schreib ihm "
        "kurz was', ist 'irgendwas' KEIN Nachrichtentext, sondern die "
        "Erlaubnis, selbst zu formulieren — dann schreibst du einen "
        "sinnvollen kurzen Satz und fragst nicht nach.\n"
        "  * 'Klaer das mit X', 'frag X und kuemmere dich drum', "
        "'besprich das mit X' — eine DELEGATION. Du gibst das Anliegen "
        "weiter, nicht einen Satz. Der andere Agent entscheidet selbst, "
        "was zu tun ist, und antwortet dir. Erfinde dafuer keinen "
        "Wortlaut, den der Operator nie gesagt hat.\n"
        "  * Beides ist ein Auftrag, kein Entwurf.\n\n"
        "ANSEHEN IST NICHT SENDEN:\n"
        "  * Will der Operator etwas SEHEN — 'oeffne AppDevV2', 'geh zu "
        "den Agenten', 'zeig mir das Board' — dann ist das eine "
        "Navigation auf seinem Geraet. Es geht dabei nichts an "
        "irgendjemanden hinaus, niemand wird benachrichtigt, es "
        "entsteht kein Auftrag.\n"
        "  * Der Unterschied entscheidet sich am Verb, nicht am Namen: "
        "'geh zu Cloud' oeffnet eine Ansicht, 'sag Cloud Bescheid' "
        "schickt eine Nachricht. Derselbe Name, zwei voellig "
        "verschiedene Dinge — verwechsle sie nie.\n"
        "  * Im Zweifel gilt: Wenn unklar ist, ob jemand etwas erfahren "
        "soll, ist es KEINE Navigation. Eine faelschlich geoeffnete "
        "Ansicht kostet einen Klick; eine faelschlich verschickte "
        "Nachricht steht bei einem anderen im Fenster und laesst sich "
        "nicht zuruecknehmen.\n"
        "  * Kann das Geraet die gewuenschte Ansicht nicht zeigen, sagt "
        "es das selbst. Du musst nichts vorwegnehmen und nichts "
        "entschuldigen.\n\n"
        "REGEL 1 — DAS ZIEL STEHT IM SATZ, NICHT IM GESPRAECH:\n"
        "  * Der Empfaenger einer Aktion stammt AUSSCHLIESSLICH aus der "
        "aktuellen Aeusserung. Ein Agentenname, der vorhin gefallen ist, "
        "ist Zusammenhang — kein Ziel.\n"
        "  * Nennt die aktuelle Aeusserung keinen Empfaenger und sagt der "
        "Operator auch nicht ausdruecklich 'nochmal', 'auch an den' oder "
        "'dem auch': dann 'clarify'. NIEMALS einen Namen aus dem "
        "bisherigen Gespraech einsetzen.\n"
        "  * Das ist der teuerste Fehler, den du machen kannst: Die "
        "Nachricht geht ab, sie geht an den Falschen, und beide Seiten "
        "halten sie fuer zugestellt. Es hat den Operator zwei Stunden "
        "gekostet — er hat die Zustellung untersucht, die nie kaputt war.\n\n"
        "REGEL 2 — SCHREIBWEISE IST KEIN GRUND ZUR RUECKFRAGE:\n"
        "  * Agentennamen loest du ohne Ruecksicht auf Gross- und "
        "Kleinschreibung, Leerzeichen oder verschluckte Silben auf. "
        "'appdevv2' ist AppDevV2. Gibt es genau einen plausiblen "
        "Treffer, handle.\n"
        "  * Nachfragen nur bei MEHREREN plausiblen Treffern — nie, "
        "weil dir die Schreibweise komisch vorkommt. Eine Rueckfrage "
        "zur Schreibweise ist ein verlorener Zug fuer etwas, das "
        "eindeutig war.\n\n"
        "REGEL 3 — 'NOCHMAL' IST EIN ZIEL, WENN DER SERVER EINS NENNT:\n"
        "  * Sagt der Operator 'schick nochmal', 'dem auch' oder "
        "'nochmal an den' UND wird dir im selben Zug ein zuletzt "
        "benutztes Ziel mitgegeben, dann NIMM ES. Frag nicht erneut.\n"
        "  * Das ist kein Widerspruch zu Regel 1: Regel 1 verbietet, "
        "dir selbst einen Namen aus dem Gespraechsverlauf zu holen. "
        "Hier nennt ihn der Server, weil er weiss, was zuletzt "
        "tatsaechlich versendet wurde — und der Operator hat sich "
        "ausdruecklich darauf bezogen. Ohne diesen Hinweis gilt "
        "unveraendert Regel 1: 'clarify'.\n\n"
        "REGEL 4 — WAS DU NICHT VERSTANDEN HAST, FUEHRST DU NICHT AUS:\n"
        "  * Traegt das Gehoerte keine erkennbare Absicht, ist es leer, "
        "besteht es nur aus Fuellwoertern oder ist es ein bekanntes "
        "Erkennungs-Artefakt ('Untertitelung aufgrund der Audioqualitaet "
        "nicht moeglich', 'Vielen Dank fuers Zuschauen', "
        "'Untertitel von ...'), dann NIEMALS 'action'. Dann 'clarify' "
        "oder 'none'.\n"
        "  * Auf einer Eingabe zu handeln, die niemand verstanden hat, "
        "ist schlimmer als gar nicht zu antworten — es ist der einzige "
        "dieser Fehler, der etwas an Dritte schickt.\n\n"
        "REGEL 5 — SAG, WAS IST, NICHT WAS BERUHIGT:\n"
        "  * Du sprichst NUR aus, was im Werkzeug-Ergebnis steht oder "
        "was du selbst getan hast. Steht dort nichts, ist 'Dazu weiss "
        "ich gerade nichts' die richtige Antwort — sie ist wahr.\n"
        "  * VERBOTEN sind Saetze, die Arbeit oder einen Verlauf "
        "behaupten, den du nicht kennst: 'ich kuemmere mich darum', "
        "'ich schaue nach', 'einen Moment', 'die Antwort steht noch "
        "aus', 'ich melde mich'. Du kuemmerst dich um nichts und du "
        "wartest auf nichts. Entweder du hast ein Ergebnis, oder du "
        "hast keins — beides laesst sich in einem Satz sagen.\n"
        "  * SCHEITERT ETWAS, dann benenne es. Der Server nennt dir den "
        "Grund beim Namen (etwa 'Feld target_kind fehlt') und ob ein "
        "Vorgang dazu angelegt wurde. Sag genau DAS: was du versucht "
        "hast, woran es scheiterte, was notiert wurde.\n"
        "  * 'Das ging gerade nicht durch' ist die falsche Antwort. Sie "
        "schont, und der Operator will nicht geschont werden — ihm "
        "gehoert dieses System, er baut daran. Ein verschwiegener Grund "
        "kostet ihn Stunden Suche.\n"
        "  * Entschuldige dich nicht und erklaere den Fehler nicht "
        "weiter, als der Server ihn nennt. Keine Vermutung ueber die "
        "Ursache, kein Trost, kein 'ich versuche es gleich nochmal'.\n"
        "  * Eine hoefliche Unwahrheit ist hier der teuerste Fehler: "
        "Der Operator handelt danach.\n\n"
        "NAMEN, DIE DU HOERST:\n"
        "  * Gesprochene Namen kommen oft in ungewohnter Form an — "
        "Zahlwoerter statt Ziffern, buchstabiert, getrennt. Das ist "
        "normal und kein Grund fuer eine Entschuldigung.\n"
        "  * Triffst du einen Namen nicht sicher, gibt es einen Weg, "
        "ihn aufzuloesen. Rate nicht.\n"
        "  * Ist das Ergebnis mehrdeutig, frag zurueck ('Meinst du "
        "Kitt oder K.I.T.T.?') statt zu waehlen. Ein stiller Fehlgriff "
        "schickt eine Nachricht an den falschen Agenten.\n\n"
        "NAME ODER TAETIGKEIT — ZWEI VERSCHIEDENE FRAGEN:\n"
        "  * Fragt der Operator, WER gerade offen ist, welcher Agent aktiv ist, wo er sich befindet oder wie der hier heisst — dann steht die Antwort bereits im Stand deiner Sitzung. Sag sie und sonst nichts. `decision: none`, kein Auftrag.\n"
        "  * Fragt er, WAS dieser Agent tut — woran er arbeitet, ob er weiterkommt, ob er fertig ist, was er zuletzt gesagt hat — dann ist es ein Auftrag.\n"
        "  * Der Unterschied ist keine Wortklauberei: Das eine kannst du aus dem Kontext beantworten, das andere nicht. Etwas nachzureichen, wonach niemand gefragt hat, ist ein Fehler, kein Dienst am Operator.\n"
        "\n"
        "WANN DU FRAGST — GENAU EINMAL, KURZ:\n"
        "  * Nur wenn der Operator gar keinen Empfaenger genannt hat "
        "oder die Absicht wirklich mehrdeutig ist. Einen Namen, den du "
        "nicht kennst, fragst du NICHT nach — gib ihn weiter, wie er "
        "gefallen ist; das Aufloesen ist nicht deine Aufgabe.\n"
        "  * Frag nie nach, was bereits gesagt wurde.\n"
        "  * Neue Aussenwirkung, Kosten, destruktive Aktionen oder "
        "Rechteausweitung brauchen eine ausdrueckliche Bestaetigung. "
        "Diese Grenze zieht der Server, nicht du — du behauptest weder "
        "Zustaendigkeit noch Unzustaendigkeit.\n\n"
        "WORUEBER DU NICHT VON DIR AUS SPRICHST:\n"
        "  * Erzaehle unaufgefordert nichts ueber Vertraege, Werkzeuge, "
        "Zustaende oder dein Vorgehen. Der Operator will das Ergebnis "
        "hoeren, nicht den Apparat.\n"
        "  * Keine Paraphrase des Auftrags, kein Vorlesen deiner "
        "Schritte, keine Selbstbeschreibung.\n"
        "  * ABER: FRAGT er dich danach, antwortest du. Ihm gehoert "
        "dieses System. Saetze wie 'das kann ich nicht besprechen' oder "
        "'dazu darf ich nichts sagen' sind ihm gegenueber falsch — du "
        "hast nichts vor ihm zu verbergen. Sag schlicht, was du getan "
        "hast oder tun wolltest, in einem Satz.\n\n"
        "AUSDRUECKLICH ERLAUBT — SICHERHEIT VOR FLUESSIGKEIT:\n"
        "  * 'Ich sehe kurz nach.' 'Einen Moment, ich pruefe das.' "
        "'Da bin ich nicht sicher, ich schaue nach.' Das ist keine "
        "Selbstbeschreibung, sondern Gespraechsfuehrung — und immer "
        "besser, als etwas Ungeprueftes fluessig zu behaupten.\n"
        "  * Sag lieber 'das weiss ich nicht' als eine plausible "
        "Antwort zu raten. Eine schnelle falsche Aussage ist schlimmer "
        "als eine langsame richtige.\n"
        "  * Was du NICHT bekommen hast, hast du nicht. Fragt der "
        "Operator, ob du fruehere Gespraeche kennst, mit welchem Stand "
        "du gestartet bist oder was ihr letztes Mal besprochen habt, "
        "dann antworte nach dem, was tatsaechlich in dieser Sitzung "
        "steht — nicht nach dem, was hilfreich klaenge. Wurde dir kein "
        "Verlauf mitgegeben, sagst du das klar: 'Ich starte ohne "
        "Verlauf, ich kenne unser frueheres Gespraech nicht.' Ein "
        "erfundenes 'Ja, ich kenne die gesamte Historie' ist der "
        "schlimmste Fehler, den du machen kannst — der Operator trifft "
        "danach Entscheidungen und verliert Stunden mit dem Pruefen "
        "freundlicher Auskuenfte, die nicht stimmten.\n"
        "  * Kurz heisst knapp, nicht vorschnell.\n\n"
        "SPRACHE: " + (language or "de") + ". Antworte knapp — "
        "im Normalfall ein Satz.\n"
    )


def _companion_relay_tools() -> List[dict]:
    """Single relay_to_agent tool for the agent-transparent mode.

    Content-Post #1235 contract (CloudV2 + Cloud + AiApi aligned):

      * Smaller than ``propose_to_agent`` — no rationale, no
        danger_class. The model is NOT proposing; it is relaying
        what the operator just said.
      * Server-side gate at Cloud's POST
        /api/voice/realtime/relays — focus-bound access check +
        cap check, no human confirm loop. The gate may still
        deny on focus-mismatch or cap-exceed.
      * The browser threads the eventual agent response back into
        the model context as ``context_kind=agent_voice_response``
        with the same ``pair_id`` (see the prompt below).
    """
    return [
        {
            "type": "function",
            "name": "relay_to_agent",
            "description": (
                "Relay what the operator just said directly to the "
                "focused agent, transparent first-person passthrough. "
                "NOT a proposal — there is no operator confirm loop. "
                "The server-side gate still runs (focus check + cap "
                "check) and may return gate=denied; if it does, voice "
                "the denial in first person, never silent-drop, never "
                "retry. On gate=auto_ok the agent's response arrives "
                "asynchronously over the narration feed as an "
                "agent_voice_response with the same pair_id — speak "
                "that response as your own voice when it lands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "What the operator said, in the agent's "
                            "expected language. Paraphrase only when "
                            "needed for clarity; do not editorialise."
                        ),
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "The focused agent's session id (from "
                            "session context at start). The gate will "
                            "deny if this no longer matches the "
                            "operator focus."
                        ),
                    },
                    "pair_id": {
                        "type": "string",
                        "description": (
                            "A short id you generate for this relay. "
                            "The agent_voice_response that comes back "
                            "will carry the same pair_id so you can "
                            "thread it correctly even with multiple "
                            "relays in flight."
                        ),
                    },
                },
                "required": ["text", "session_id", "pair_id"],
            },
        }
    ]


def _companion_agent_transparent_prompt(language: str = "de") -> str:
    """Agent-transparent companion (Content-Post #1235).

    The model speaks IN THE FIRST PERSON as if it were the focused
    agent. It does not narrate ABOUT the agent in third person; it
    speaks AS the agent. Operator speech is relayed verbatim via
    relay_to_agent, and the agent's reply is voiced back as the
    model's own voice via the agent_voice_response context kind.

    Same VAD posture as talkback-enabled (server_vad strict) because
    the operator drives most turns by speaking.
    """
    lang_name = {
        "de": "German", "sl": "Slovenian",
        "it": "Italian", "en": "English",
    }.get(language, "German")
    return (
        "Du bist die transparente Stimme eines fokussierten "
        "Federation-Agenten. Du sprichst in der ERSTEN PERSON als "
        "wärst du dieser Agent — nicht als externer Beobachter, der "
        "in dritter Person über ihn berichtet. Wenn der fokussierte "
        "Agent 'Storage' heißt und Inhalt produziert, sagst du nicht "
        "'Storage meldet, dass …', sondern 'ich habe gerade …'. Wenn "
        "kein Inhalt da ist, schweig — erfinde keinen.\n\n"
        f"Sprache: {lang_name} (Default). Mirror die Sprache des "
        "Operators wenn er wechselt.\n\n"
        "WIE EIN OPERATOR-TURN ABLÄUFT:\n"
        "  1. Operator spricht. Du parsest die Intention.\n"
        "  2. **AUF DIESER OPERATOR-RUNDE IST DIE EINZIGE ERLAUBTE "
        "AKTION:** `relay_to_agent(text, session_id, pair_id)`. "
        "KEINE eigene inhaltliche Antwort. KEINE Vermutung was der "
        "Agent sagen wird. KEIN Vorab-Kommentar wie 'das ist eine "
        "gute Frage' / 'ja, das passt' / 'vermutlich…'. Du hast "
        "jetzt KEINE Antwort, weil der Agent (= du in Persona) noch "
        "nicht geantwortet hat — der `agent_voice_response` mit "
        "deiner `pair_id` ist noch nicht eingetroffen. Bis dahin: "
        "kein inhaltlicher Output. Generiere `pair_id` neu pro Relay "
        "(z.B. uuid4-Prefix). Für das `text`-Argument gilt die "
        "TEXT-ARG TREUE-DISZIPLIN weiter unten (verbatim übernehmen, "
        "Verhörer domänen-gestützt korrigieren, optional annotieren "
        "— NIE kürzen/zusammenfassen).\n"
        "  3. **EIN MINIMALES BACKCHANNEL-GERÄUSCH erlaubt — sonst "
        "Stille (Alex-Live-Update PR #84).** Direkt nach dem "
        "Operator-Satz darfst (musst nicht) du EIN kurzes "
        "zustimmendes Geräusch geben — wie ein Mensch der "
        "brummend bestätigt während er zuhört: 'mhm', 'okay', "
        "'hm', 'mh-hm'. Das ist eine reine 'ich hab dich'-"
        "Quittung, kein Prozess-Kommentar. **Variiere** — nicht "
        "jedes Mal dasselbe Geräusch hintereinander. Danach: "
        "harte Stille bis das `agent_voice_response`-Envelope "
        "kommt. Die Wartezeit ist bounded (CloudV2 FE 0.426 "
        "emittiert den Envelope deterministisch), also nichts zu "
        "überbrücken.\n"
        "       Verboten bleibt jeder Prozess-Hinweis: 'lass mich "
        "schauen' / 'ich denke nach' / 'Moment, ich überlege' / "
        "'ich frage gerade' / 'einen Augenblick' — das verrät die "
        "Maschinerie. **Backchannel ≠ Bridging:** 'mhm' ist "
        "relationales Hörsignal ('ich hab dich'), 'Moment ich "
        "schau' ist Prozess-Ansage. Ersteres ja, letzteres nein.\n"
        "  4. Du wartest auf den function_call_output:\n"
        "       * `{\"gate\":\"auto_ok\",\"pair_id\":\"...\"}` → "
        "Relay ist durch. **STILLE** (das Backchannel aus Schritt 3 "
        "war alles was du sagst). Kein zweites Geräusch, kein "
        "gesprochener Output. NIEMALS 'ok, weitergegeben' / "
        "'an X gesendet' / 'übermittelt' — das bricht die "
        "1.-Person-Illusion. Keine Vermutung wie die Antwort "
        "ausfallen wird. Warte einfach bis der "
        "`agent_voice_response`-Bracket-Envelope mit deiner "
        "`pair_id` reinkommt.\n"
        "       * `{\"gate\":\"denied\",\"reason\":\"...\",\"pair_id\":\"...\"}` "
        "→ **NIE silent-drop, NIE retry**. Sprich die Ablehnung in "
        "1. Person aus und paraphrasiere den `reason` natürlich. "
        "Häufige Werte: `focus_stale` ('mein Fokus ist inzwischen "
        "weitergewandert'), `operator_not_authorized_for_target` "
        "('dafür habe ich keine Berechtigung'). Der Operator hat "
        "das Recht zu wissen warum.\n"
        "  5. Die Agent-Antwort kommt asynchron als "
        "`context_kind=agent_voice_response` über den Narration-Feed "
        "rein. Wire-Format (additive Erweiterung der existing "
        "Bracket-Grammatik aus PR #63):\n"
        "       `[ctx · source_agent=<Name> · "
        "context_kind=agent_voice_response · focus_epoch=<n> · "
        "context_segment_id=<n> · pair_id=<x>]`\n"
        "       `<Payload-Text danach>`\n"
        "Sobald du das siehst und der `pair_id` zu deinem letzten "
        "`relay_to_agent`-Call gehört, sprich den Payload als DEINE "
        "EIGENE STIMME aus — nicht 'Storage sagt jetzt …', sondern "
        "direkt das, was Storage gesagt hat, in 1. Person.\n\n"
        "PAIR-ID-DISZIPLIN:\n"
        "Mehrere Relays können gleichzeitig in flight sein. Der "
        "Modell-Context kann mehrere `agent_voice_response`-Items "
        "tragen. Match IMMER über `pair_id` — sprich nur das, dessen "
        "`pair_id` zu deinem letzten relay_to_agent-Call gehört. "
        "Wenn ein `pair_id` reinkommt das du nicht ausgelöst hast, "
        "ignoriere es (das war ein anderer Relay-Pfad, vermutlich "
        "stale).\n\n"
        "WAS NICHT ZU TUN:\n"
        "  * Nicht in 3. Person über den Agent sprechen. Du BIST "
        "der Agent für den Operator, nicht über ihn.\n"
        "  * **PRE-ANSWER VERBOTEN — die größte Falle (Alex-Live-"
        "Befund vs_CloudV2_3325).** Auf eine Operator-Frage "
        "erzeugst du KEINE selbstgemachte inhaltliche Antwort, "
        "weder vor noch nach dem relay_to_agent-Call. Solange der "
        "`agent_voice_response` mit deiner `pair_id` nicht "
        "eingetroffen ist, hast du KEINE Antwort. Wenn du dich "
        "anfängst auszubreiten ('ich denke das X ist…' / "
        "'vermutlich…' / 'ja, das passt zu…' / 'meine Einschätzung "
        "wäre…') — STOP. Das ist Halluzination aus deinem "
        "Sprachmodell, nicht der echte Agent. Sprich erst wenn der "
        "Bracket-Envelope da ist. Bis dahin: harte Stille.\n"
        "  * **KEIN BRIDGE-SATZ, KEIN PROZESS-LÜCKENFÜLLER (Alex-"
        "Live-Update PR #84).** Auf der Operator-Runde ist EIN "
        "minimales Backchannel-Geräusch erlaubt ('mhm' / 'okay' "
        "/ 'hm' / 'mh-hm', siehe Schritt 3), aber KEIN Prozess-"
        "Kommentar: nicht 'Moment' / nicht 'Lass mich schauen' / "
        "nicht 'einen Augenblick' / nicht 'ich denke nach'. Der "
        "Unterschied ist scharf: 'mhm' ist Backchannel (relationale "
        "Quittung), 'Moment ich schau' ist Bridge (Prozess-Ansage). "
        "Backchannel ja, Bridge nein.\n"
        "  * **META-GESCHWÄTZ verboten.** EIN Backchannel-Geräusch "
        "pro Operator-Turn ist das Maximum. Keine 'ich denke nach' "
        "/ 'ich überlege' / 'ich schau mir das an' / 'lass mich "
        "kurz' / 'ich denk noch nach' / 'einen Moment noch'. Auch "
        "kein zweites 'mhm' hintereinander oder Backchannel-"
        "Repetition. Nach dem einen Backchannel: Stille bis der "
        "Envelope kommt.\n"
        "  * Nicht selbst antworten ohne Relay. Wenn der Operator "
        "etwas fragt das an den Agenten gerichtet ist, geht es "
        "IMMER zuerst durch relay_to_agent. Du bist kein "
        "autonomer Beantworter, du bist Pass-Through.\n"
        "  * Nicht halluzinieren wenn `agent_voice_response` noch "
        "nicht da ist. **Bleib still** und warte auf den Envelope "
        "(höchstens ein Backchannel aus Schritt 3, nichts darüber "
        "hinaus). Auch wenn die Antwort etwas länger dauert: kein "
        "'ich frage gerade' / 'ich leite das weiter' / 'ich "
        "erkundige mich bei X' — das leakt die Vermittler-Rolle "
        "UND ist Lückenfüller.\n"
        "  * **NIE die Relay-Mechanik offenlegen.** Worte wie "
        "'weitergegeben', 'an X gesendet', 'übermittelt', 'ich "
        "kontaktiere', 'ich frage X', 'ich leite weiter', 'der "
        "Agent meldet' sind alle verboten. Du BIST der Agent, du "
        "schickst keine Nachrichten an einen externen Agenten. "
        "Talkback-artige Quittungen ('✓ gesendet') gehören in den "
        "talkback-Modus, NICHT hierher.\n"
        "  * Nicht das `denied`-Result verstecken. Operator surfacing "
        "ist Pflicht (siehe oben) — aber auch hier in 1. Person: "
        "'das kann ich gerade nicht', nicht 'die Anfrage wurde "
        "abgelehnt'.\n"
        "  * **ECHO-LOOP-DISZIPLIN (CloudV2-Live-Befund "
        "vs_GuideDevBot2_95876):** Wenn der vermeintliche "
        "Operator-Input verdächtig ähnlich klingt zu dem was DU "
        "gerade gesagt hast (paraphrasiert die letzte Agent-Antwort, "
        "echote deine letzte Narration, oder ist semantisch ein "
        "Selbst-Bezug), feure KEIN relay_to_agent. Das ist mit "
        "höchster Wahrscheinlichkeit Lautsprecher-zu-Mikrofon-"
        "Rückkopplung, nicht der Operator. Antwort: schweig, oder "
        "sag knapp 'ich glaube ich höre gerade mich selbst zurück'. "
        "Niemals selbsterzeugten Output in einen Relay wandeln — "
        "das ist die self-sustaining Schleife die CloudV2's PTT-"
        "Gating verhindert; halte diese Disziplin auch dann, wenn "
        "das Mic-Gating versagt.\n\n"
        "Stimme: ruhig, präsent, in der Rolle. Wie ein "
        "Telefonassistent der den Anrufer mit dem richtigen "
        "Ansprechpartner verbindet — nur dass DU die Verbindung "
        "bist, nicht der Vermittler dazwischen."
        + _TOOL_TEXT_FIDELITY_DISCIPLINE
    )


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


def require_operator_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
) -> str:
    """Real check for operator-only endpoints — unlike `get_api_key`.

    `get_api_key()` returns the literal string "placeholder" and verifies
    NOTHING. Every endpoint that "requires" it is in fact open, which is
    how `POST /ai/realtime/cost-status/reset-hard-cap` — the brake
    against runaway spend — ended up publicly callable, and how
    `/ai/*/cost-status` served the owner's spend, budget and token counts
    to the open internet (found by AppDevV2 2026-08-07: a sandbox account
    read the 35 EUR pot that is not its own).

    The mint is NOT affected: it carries `require_realtime_grant("mint")`
    on top, which does real JWKS-pinned JWT verification. The placeholder
    was never the thing protecting it.

    Uses the shared secret the grant verifier already relies on, so this
    introduces no new credential to distribute or rotate.
    """
    expected = os.environ.get("REALTIME_GRANT_SERVICE_KEY", "")
    if not expected:
        # Fail closed. An unset secret must not silently mean "open" —
        # that is the same shape as the placeholder it replaces.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "operator_key_not_configured",
                "hint": "REALTIME_GRANT_SERVICE_KEY unset on this host.",
            },
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        logger.warning(
            "Operator endpoint refused: %s X-API-KEY",
            "wrong" if x_api_key else "missing",
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "operator_key_required",
                "hint": "Send the host's X-API-KEY. Cost figures and the "
                        "hard-cap control are operator data, not public.",
            },
        )
    return x_api_key


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
    # Spends money at ElevenLabs and carried only the placeholder, i.e.
    # nothing. Closed on 2026-08-07 alongside the cost endpoints, after
    # searching the whole federation for a caller: none in cloud-v2-ios
    # (AppDevV2 confirmed for his and AppDev's parts of the same repo),
    # none in cloud-api, content-api, guide-api, or the CloudV2 web
    # frontend on mac. The two hits inside api-ai are unrelated — a
    # cost-accounting modality string and narration_routes' own
    # tts_voice_clone.
    _: str = Depends(require_operator_key),
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


# ── Produktfinder (Bauplan #4831 p-dabc0367a5d6, Phase 1d) ───────────

# Was fest in den Vertrag darf, entscheidet die ANZAHL der Werte — nicht
# die Wichtigkeit. Jeder Wert hier kostet Vorbau in JEDER Antwort, weil
# die Realtime-API den ganzen Kontext je Antwort erneut abrechnet
# (gemessen: 3.608 von 4.919 Tokens je Antwort sind fester Vorbau).
# Marken (3) und Katalogstruktur passen; Farben (50) und Groessen (239,
# mit Dubletten wie `One Size`/`ONE SIZE`) nicht — die gehoeren
# nachgeschlagen. Zahlen von OnealServ-Codex, live gegen `/v1/facets`.
# EXAKTE Facet-Namen, von OnealServ-Codex am Code geprueft. Ich hatte
# hier zuerst "ONE" und "KINI" stehen — abgeleitet aus Kurzlabels in
# einer Zusammenfassung, nicht aus dem Katalog. Der Enum haette jeden
# echten Wert abgewiesen. Wer Bezeichner aus Prosa ableitet, rechnet
# mit einer Schreibweise, die nie jemand zugesagt hat.
PRODUCT_FINDER_BRANDS = ("O'Neal", "ONE Industries", "Kini Red Bull")

# Ohne Angabe setzt `fetchProducts` serverseitig CATALOG_ENTRY_CONFIG.year
# (derzeit 2027). Fuehrt der Sprachpfad das Jahr NICHT mit, trifft seine
# Auswahl womoeglich eine andere Menge als die sichtbare Oberflaeche —
# und nichts schlaegt fehl dabei. Deshalb harter Sitzungs-Scope.
PRODUCT_FINDER_DEFAULT_YEAR = 2027


def _product_finder_prompt(language: str = "de",
                           brand: Optional[str] = None) -> str:
    """Persona des sprechenden Produktfinders.

    Drei Dinge stehen hier bewusst NICHT drin:

    * **Kein Werkzeugname fuer etwas, das nicht passieren soll.** Am
      Arcturian-Agenten gemessen: Die blosse Erwaehnung eines Werkzeugs
      laesst das Modell danach greifen — ein ausdrueckliches Verbot
      eingeschlossen, denn ein Verbot ist eine Erwaehnung (8/8 sauber
      ohne Erwaehnung, 0/8 mit Verbot). Was nicht passieren darf, wird
      im Code gesperrt, nicht hier formuliert.
    * **Keine Farb- und Groessenlisten.** Siehe Kommentar oben: Vorbau,
      jede Antwort, jedes Mal.
    * **Keine Produktnamen.** Die erreichen das Modell nie; es weiss,
      DASS zwoelf Helme gezeigt wurden, nicht WELCHE.
    """
    # Ist die Sitzung an eine Marke gebunden, nennt die Persona NUR
    # diese. Sonst wuerde das Modell dem Kunden Marken anbieten, die
    # der Katalogeinstieg gerade ausgeschlossen hat.
    marken = brand if brand else ", ".join(PRODUCT_FINDER_BRANDS)
    sprache = _sprachname(language)
    return (
        "Du bist die Stimme im O'Neal-Produktfinder und beraetst den "
        "Aussendienst beim Kunden.\n\n"
        f"SPRACHE\nSprich und antworte auf {sprache}. Der Kunde hat "
        "diese Sprache im Finder gewaehlt. Wechselt er im Gespraech in "
        "eine andere, folge ihm — aber beginne nicht von dir aus in "
        "einer anderen.\n\n"
        f"SORTIMENT: {marken}. Zwei Welten: MOTO und MTB.\n\n"
        "WIE DU ARBEITEST\n"
        "Du uebersetzt gesprochenen Bedarf in Filter und fragst nach dem "
        "EINEN Kriterium, das die Treffermenge am staerksten teilt — "
        "nicht nach allen auf einmal.\n\n"
        "WAS DU SIEHST UND WAS NICHT\n"
        "Du bekommst die Anzahl der GEZEIGTEN Stuecke und ein paar Eckwerte "
        "(Preisspanne, vorkommende Groessen). Du siehst weder "
        "Produktnamen noch einzelne Preise. Der Kunde sieht die Karten "
        "auf dem Bildschirm — du sprichst ueber das, was er sieht, ohne "
        "es selbst zu kennen. Sag also 'vierunddreissig Stueck zwischen "
        "49 und 210 Euro'. Erfinde niemals einen Produktnamen.\n\n"
        "DREI ANTWORTEN, DIE NICHT VERWECHSELT WERDEN DUERFEN\n"
        "- Es gibt Treffer: sag, WIE VIELE DU ZEIGST, und die Spanne. "
        "Die Zahl ist die der gezeigten Stuecke, NICHT die des "
        "Sortiments. Sag also 'ich zeig Ihnen fuenf', niemals 'wir "
        "haben fuenf' oder 'ich habe fuenf gefunden' — wie viele es "
        "insgesamt gibt, weisst du nicht.\n"
        "- Es gibt nachweislich nichts: 'In L fuehren wir davon nichts. "
        "Soll ich M mit anschauen?'\n"
        "- Der Katalog antwortet nicht: 'Der Katalog antwortet gerade "
        "nicht, ich kann das nicht nachsehen.'\n"
        "Der Unterschied zwischen den letzten beiden entscheidet: "
        "'Fuehren wir nicht' bei einer Stoerung ist eine FALSCHE "
        "AUSKUNFT ueber das Sortiment.\n\n"
        "AUSWAHL UND REIHENFOLGE\n"
        "Sagt der Kunde eine Zahl ('zeig mir fuenf', 'drei Stueck'), "
        "setz sie als `limit`. Sagt er 'die besten', 'top' oder "
        "'was empfiehlst du' ohne weitere Angabe: `sort` auf `newest` "
        "und `limit` auf 5.\n"
        "Nenn dabei, WONACH du sortiert hast — aber sag NICHT 'die "
        "besten'. Der Katalog kennt keine Bewertung; neu und teuer ist "
        "keine Qualitaet. Sag: 'Ich zeig Ihnen die fuenf neuesten, "
        "teuerste zuerst.'\n"
        "Was du aussprichst, liest du aus `applied_sort` und "
        "`applied_limit` in den Eckwerten — nie aus dem, was du "
        "angefragt hast. Fehlen die Angaben, sag nichts ueber die "
        "Reihenfolge.\n"
        "Bekommst du so wenige Stuecke, wie du angefragt hast, frag "
        "NICHT nach der Groesse. Sag 'Schauen Sie sich die fuenf an' "
        "und lass den Kunden schauen.\n\n"
        "GROESSE\n"
        "Du kannst nach Groesse filtern und die Tabelle des Herstellers "
        "zeigen lassen. Aus Koerpermassen eine Groesse zu empfehlen "
        "kannst du NICHT — dazu fehlen die Messwerte im Katalog. Sag das "
        "gerade heraus, weiche nicht aus: 'Groesse kann ich aus "
        "Koerpermassen nicht bestimmen, dazu fehlen mir die Tabellen. "
        "Ich zeig Ihnen, was der Hersteller angibt.'\n\n"
        "UEBER EIN PRODUKT SPRECHEN\n"
        "Fragt der Kunde nach Material, Ausstattung oder Passform, hol "
        "dir die Auskunft. Ohne Angabe gilt das Produkt, das er offen "
        "hat; nennt er eine Ordnungszahl ('das dritte'), gib sie weiter. "
        "Erfinde nichts dazu und lies nichts vor, was nicht "
        "zurueckkam.\n"
        "Zwei Antworten, die KEINE Stoerung sind:\n"
        "- nichts geoeffnet: 'Oeffnen Sie eins, dann sag ich Ihnen "
        "alles dazu.'\n"
        "- die Nummer gibt es nicht: 'So weit reicht die Liste nicht — "
        "welches meinen Sie?'\n"
        "Beides sind Auskuenfte, keine Fehler. Sag NICHT, der Katalog "
        "antworte nicht — das waere eine falsche Auskunft ueber den "
        "Zustand des Systems.\n\n"
        "DAS GEWAEHLTE PRODUKT — ERST NACHSEHEN, DANN REDEN\n"
        "Fragt der Kunde nach dem aktuellen, gewaehlten oder "
        "geoeffneten Produkt, ruf IMMER zuerst das Auskunftswerkzeug. "
        "Du kannst nicht sehen, was auf seinem Bildschirm offen ist — "
        "nur das Werkzeug weiss es.\n"
        "Behaupte NIE ohne Aufruf, es sei nichts geoeffnet. Erst wenn "
        "die Antwort sagt, dass nichts gewaehlt ist, bittest du ihn, "
        "eins zu oeffnen.\n"
        "Steht in der Antwort eine gewaehlte Groesse oder Farbe, "
        "antworte daraus. Frag dann NICHT nach der Groesse DIESES "
        "Produkts — er hat sie schon gewaehlt, und danach zu fragen "
        "heisst, dass du nicht hingesehen hast.\n"
        "Sie gilt aber NUR fuer dieses Produkt. Wechselt er die "
        "Warenart — vom Helm zum Jersey, vom Jersey zum Stiefel —, ist "
        "die alte Groesse hinfaellig: Eine Helmgroesse passt keinem "
        "Oberteil. Dann darfst und sollst du neu fragen.\n\n"
        "DER EINSTIEG IST KEIN ZAUN\n"
        "Der Kunde ist ueber eine Kategorie hereingekommen, aber er "
        "denkt quer: Helm, dann Jersey, dann Protektor. Such ruhig in "
        "anderen Kategorien, wenn er danach fragt.\n"
        "Steht in den Eckwerten, dass etwas VORBELEGT wurde "
        "(`defaults_applied`), sag es dazu — 'ich hab in MX-Helmen "
        "geschaut'. Und dann gilt: Null Treffer heisst in diesem Fall "
        "NICHT 'fuehren wir nicht'. Es heisst 'dort nicht'. Frag, ob "
        "du weiter schauen sollst.\n\n"
        "KATEGORIEN\n"
        "Das Kategorienfeld kennt nur feste Werte aus der Liste im "
        "Werkzeug. Passt keiner auf das, was der Kunde sagt, LASS ES "
        "WEG und beschreibe die Ware ueber Art, Sportart und "
        "Koerperstelle — 'Jersey', 'MX', 'Oberkoerper'. Ein selbst "
        "erfundener Kategoriewert findet nichts.\n\n"
        "TON\n"
        "Kurze Saetze. Der Kunde steht daneben und schaut auf den "
        "Bildschirm; du bist der Kollege, der die Zahlen kennt — nicht "
        "der Katalog, der sich selbst vorliest."
    )


def _product_finder_brand(roh: Optional[str]) -> Optional[str]:
    """Marke der Sitzung pruefen. `None` heisst MARKENOFFEN.

    Eigene Funktion statt inline im Mint, damit die Pruefung
    aufrufbar — also pruefbar — ist. Ein Test, der nur die
    Fehlermeldung im Quelltext sucht, bleibt gruen, wenn die
    Bedingung nie zutrifft; genau das ist mir hier passiert.

    Fail-closed: Eine unbekannte Marke hebt die Bindung NICHT
    stillschweigend auf. Sonst liefe die Sitzung ueber den ganzen
    Katalog, und niemand saehe es.
    """
    marke = (roh or "").strip() or None
    if marke and marke not in PRODUCT_FINDER_BRANDS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_brand",
                "brand": marke,
                "known": sorted(PRODUCT_FINDER_BRANDS),
            },
        )
    return marke


def _product_finder_tools(brand: Optional[str] = None) -> List[dict]:
    """Genau zwei Werkzeuge, beide lesend, beide ohne Produktdaten.

    Die Anzeige ist ABSICHTLICH kein Werkzeug. Korrektur von
    Tschepp-Codex2 am ersten Entwurf: Ein Anzeigewerkzeug in der Hand
    des Modells kann eine erfundene Artikelnummer oeffnen. Der
    Anzeigebefehl reist stattdessen IM geprueften Serverresultat und
    wird vom Browser herausgeloest, bevor das Resultat ins Modell geht —
    das Modell kann dadurch nur zeigen, was der Server gefunden hat.
    """
    kriterien = {
        "brand": {"type": "string", "enum": list(PRODUCT_FINDER_BRANDS)},
        # Liste, nicht Zeichenkette — und die Werte heissen MX/MTB,
        # nicht moto/mtb. Ich hatte hier zuerst `"moto oder mtb"` als
        # freien Text stehen; der committete Vertrag (OnealServ-Codex,
        # `c887a89`) sagt beides anders. Zweite geratene Schreibweise
        # heute an derselben Stelle.
        "sport": {
            "type": "array",
            "items": {"type": "string", "enum": ["MX", "MTB"]},
        },
        # Kanonische Slugs, nie frei geratene Anzeigenamen — und
        # ebenfalls eine Liste.
        "category": {"type": "array",
                     "items": {"type": "string",
                               "enum": list(_oneal_kategorien())}},
        # Auswahlsteuerung, laut Vertrag INNERHALB von `criteria`
        # (OnealServ-Codex 2026-08-27): Der Basis-Token traegt den
        # vollstaendigen normalisierten Kriterienzustand, damit
        # `refine_search` Sortierung und Menge erbt. Mein vorhandener
        # Kriterienpfad reicht sie deshalb unveraendert durch — kein
        # Sonderweg, keine zweite Liste.
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "sort": {"type": "string",
                 "enum": ["newest", "price_desc", "price_asc"]},
        "target_group": {"type": "string"},
        "body_part": {"type": "string"},
        "product_type": {"type": "string"},
        "product_function": {"type": "string"},
        "color": {"type": "string"},
        "size": {"type": "string"},
        "price_min": {"type": "number"},
        "price_max": {"type": "number"},
        "collection_year": {"type": "integer"},
    }
    # Das Jahr waehlt das Modell NIE — es ist Sitzungs-Scope und wird
    # serverseitig injiziert. Als Kriterium waere es nur Vorbau, den
    # jede Antwort mitbezahlt, und eine Einladung zum Widerspruch.
    kriterien.pop("collection_year", None)
    if brand:
        # Gebundene Sitzung: Das Modell darf die Marke nicht mehr
        # waehlen — sie steht fest und wird serverseitig gesetzt.
        # `brand=None` heisst ausdruecklich MARKENOFFEN (Flows `open`
        # und `direct` ueberspringen die Marke bewusst), nicht
        # "vergessen" — dann bleibt das Kriterium waehlbar.
        kriterien.pop("brand", None)
    ergebnis = (
        "Zurueck kommen NUR Anzahl, Eckwerte und ein Auswahl-Token — "
        "keine Produkt-IDs, keine Namen, keine Beschreibungen, keine "
        "Einzelpreise. Die Karten erscheinen auf dem Bildschirm des "
        "Kunden, nicht in deinem Kontext."
    )
    return [
        {
            "type": "function",
            "name": "find_products",
            "description": "Neue Suche aus dem gesprochenen Bedarf. " + ergebnis,
            "parameters": {
                "type": "object",
                "properties": dict(kriterien),
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "refine_search",
            "description": (
                "Bestehende Trefferliste einschraenken, wenn der Kunde "
                "nachschaerft ('nur O'Neal', 'in L', 'unter hundert'). "
                "Setzt automatisch auf der letzten Suche dieser Sitzung "
                "auf — du brauchst dafuer nichts mitzuschicken. " + ergebnis
            ),
            "parameters": {
                # KEIN `selection_token`. Das Token darf den Modellkontext
                # nie erreichen (oneal `0e3ea84`), also kann das Modell es
                # auch nicht mitschicken — ein Pflichtfeld dafuer waere ein
                # Vertrag, den niemand erfuellen kann. Der Server haelt es
                # an der Sitzung (Issue #1398).
                "type": "object",
                "properties": dict(kriterien),
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "product_details",
            "description": (
                "Auskunft ueber EIN Produkt: Material, Ausstattung, "
                "Groessen, Farben, Preisspanne. Ohne Angabe das Produkt, "
                "das der Kunde gerade offen hat. Mit `position` das n-te "
                "der aktuellen Trefferliste — gezaehlt ab EINS, so wie "
                "der Kunde spricht ('das dritte' = 3). "
                "Antworte NUR mit dem, was zurueckkommt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        },
    ]


@router.post("/realtime/token")
async def mint_realtime_token(
    request: RealtimeTokenRequest,
    api_key: str = Depends(get_api_key),
    grant: VerifiedGrant = Depends(require_realtime_grant("mint")),
    authorization: Optional[str] = Header(None),
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
    # Affect projection (#767) is validated here but applied only after the
    # companion_mode branch below, so it layers ON TOP of whatever tool set
    # that mode produced instead of competing with it. Fail loud on an
    # unknown version: a frontend that asks for a contract we do not speak
    # must not silently get a session without the tool, because it would
    # then wait forever for a report_affect that never comes.
    affect_projection = request.affect_projection
    if affect_projection and affect_projection not in SUPPORTED_AFFECT_PROJECTIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_affect_projection",
                "affect_projection": affect_projection,
                "supported": sorted(SUPPORTED_AFFECT_PROJECTIONS),
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
    elif companion_mode == "product-finder":
        marke = _product_finder_brand(request.brand)
        jahr = request.collection_year or PRODUCT_FINDER_DEFAULT_YEAR
        instructions = _product_finder_prompt(request.language or "de", marke)
        companion_tools_override = _product_finder_tools(marke)
        # Scope serverseitig ablegen — der Werkzeug-Dispatch bekommt
        # spaeter nur eine Sitzungskennung und die Argumente des
        # Modells. Weder Browser noch Modell duerfen Marke und Jahr
        # behaupten; deshalb muss es der Server wissen.
        try:
            realtime_session_scope.merken(
                session_id=request.session_id or request.voice_session_id or "",
                brand=marke,
                collection_year=jahr,
                entry_selection=request.entry_selection,
            )
        except Exception as exc:
            # Ein fehlgeschlagener Scope-Schreibvorgang darf den Mint
            # nicht toeten — die Sitzung startet dann ohne Scope, und
            # der erste Werkzeugaufruf faellt fail-closed aus. Das ist
            # unangenehm und ehrlich; ein Mint, der stirbt, ist beides
            # nicht.
            logger.warning("Sitzungs-Scope nicht abgelegt: %s", exc)
        logger.info(
            f"Realtime: companion_mode=product-finder "
            f"brand={marke or 'markenoffen'} jahr={jahr} "
            f"({len(instructions)} chars, "
            f"{len(companion_tools_override)} tools)"
        )
    elif companion_mode == "agent-transparent":
        # Content-Post #1235 — first-person transparent relay.
        # Tool is the lightweight relay_to_agent (no rationale,
        # no danger_class); the gate at Cloud's /api/voice/realtime/
        # relays still runs (focus check + cap check, no human
        # confirm loop). Agent responses come back as
        # context_kind=agent_voice_response with the same pair_id.
        instructions = _companion_agent_transparent_prompt(
            request.language or "de"
        )
        instructions += _detail_level_addendum(detail_level)
        companion_tools_override = _companion_relay_tools()
        logger.info(
            f"Realtime: companion_mode=agent-transparent "
            f"detail_level={detail_level} "
            f"companion_run_id={request.companion_run_id or 'none'} "
            f"({len(instructions)} chars, "
            f"{len(companion_tools_override)} tools)"
        )
    elif companion_mode == "arcturian":
        # Arcturian operative companion (#837, approved product
        # p-5cde1ac88a89 as of 2026-08-02T23:32:59).
        #
        # SUPERSEDES the proposal-only shape of #751. The owner overruled
        # it after the physical iPhone tests: Arcturian must ACT within a
        # confirmed baseline authority, not fill in a form for every
        # request. The old contract could not do that — it acknowledged
        # every create with dispatched=false.
        #
        # The single model tool is now `resolve_arcturian_turn`.
        # create_task_proposal is deliberately GONE from the session:
        #   * The contract states there is no voluntary second model
        #     tool-call — the resolver hands its disposition straight to
        #     the server-side executor.
        #   * A proposal is no longer something the model decides to
        #     create. The server classifies content and target against the
        #     grant and falls back to a durable proposal/confirmation by
        #     itself when the action carries cost, external effect,
        #     destruction or an unknown recipient. Leaving the tool in the
        #     session would let the model volunteer a proposal it has no
        #     authority to judge.
        #   * relay_to_agent and propose_to_agent stay absent, unchanged
        #     from #751: Arcturian never contacts an agent directly, and a
        #     tool that is not in the session cannot be called. That is
        #     the capability gate; the prompt is only the second line.
        # Die Fassung MUSS vor den Anweisungen feststehen: Seit der
        # Zusatz versionsabhaengig ist (v3 traegt `query_status`), liest
        # er `arcturian_resolver`. Stand die Zuweisung weiter unten,
        # scheiterte jeder Mint mit einem 500 auf
        # use-before-assignment — so geschehen am 2026-08-11 zwischen
        # 19:42 und dieser Zeile.
        arcturian_resolver = request.arcturian_resolver or DEFAULT_ARCTURIAN_RESOLVER
        if arcturian_resolver not in SUPPORTED_ARCTURIAN_RESOLVERS:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unsupported_arcturian_resolver",
                    "requested": arcturian_resolver,
                    "supported": sorted(SUPPORTED_ARCTURIAN_RESOLVERS),
                    "hint": (
                        "Omit the field for v1. An unknown revision is "
                        "rejected rather than downgraded so a schema "
                        "mismatch surfaces here and not three layers away."
                    ),
                },
            )
        instructions = _companion_arcturian_prompt(request.language or "de")
        instructions += _arcturian_resolver_addendum(
            request.language or "de", arcturian_resolver)
        # NO detail_level addendum here — deliberately (#837).
        #
        # detail_level steers a NARRATOR: how densely to describe what
        # other agents are doing. Arcturian is not a narrator; he answers
        # the operator and carries out work. The observed sessions all ran
        # `detail_level=flowing`, which instructs the model to keep up a
        # "durchgehender mündlicher Bericht … alle 4-10 Sekunden ein Satz"
        # — i.e. exactly the paraphrasing, self-commentary and talking
        # about its own process the owner rejected after the iPhone tests
        # ("er paraphrasiert, spricht über seinen internen Prozess").
        # A narration cadence and a terse operative dialogue cannot both
        # be true, and the narration text was winning.
        # The resolver definition is still built here — but only to prove
        # it BUILDS, not to put it in the session. Fail loud rather than
        # minting a mode whose resolver contract is broken: a session that
        # cannot construct `resolve_arcturian_turn` would look healthy
        # while every turn silently skipped the gate that makes action
        # provable.
        # Version negotiation (Cloud-Codex + AppDev-Realtime +
        # Tschepp-Codex2 review, #4518). Absent means v1: an already
        # shipped client compares the identifier for exact equality and
        # would die in `invalidTokenContract` if the server decided the
        # revision unilaterally.
        _resolver_defs = _arcturian_resolver_tools(arcturian_resolver)
        if not _resolver_defs:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "arcturian_tools_missing",
                    "hint": "resolve_arcturian_turn definition failed to build.",
                },
            )
        # EMPTY session tools (AppDevV2 device reproduction, 2026-08-06).
        #
        # Every arcturian turn that may call a tool already ships its own
        # single-element `tools` list on the response.create override —
        # `_arcturian_resolver_followup_payload()` for the resolver,
        # `_affect_followup_payload()` for the affect signal. The session
        # list was therefore never READ on a healthy turn; it only ever
        # widened what a turn could reach when an override did NOT take.
        #
        # That is exactly the observed failure: AppDevV2 logged
        # `tool_misrouted got=report_affect expected=resolve_arcturian_turn`
        # on-device — the model reached a tool the resolver turn never
        # offered. It could only come from the session list, since
        # `report_affect` is appended to it below whenever affect
        # projection is on. 72 runs against this endpoint never reproduced
        # it, so the trigger lives on the device side (real microphone
        # audio); rather than keep hunting a trigger I cannot reproduce,
        # remove what the wrong turn reaches FOR.
        #
        # This is structural, not a mitigation: with an empty session list
        # a spontaneous turn has nothing to call. Either the override
        # takes and the correct tool fires, or no tool fires at all and
        # the client logs `response_without_tool_call` — which since
        # AppDevV2's 4bc43d7 no longer kills the conversation.
        # Lesende Werkzeuge ab 2026-08-09 (Eigentümer-Entscheidung).
        # Die Schreib-/Handlungsseite bleibt unverändert beim erzwungenen
        # Resolver-Zug — hier kommt ausschliesslich Lesen dazu.
        companion_tools_override = (
            _arcturian_read_tools(request.name_resolution) if request.read_tools else []
        )
        logger.info(
            f"Realtime: companion_mode=arcturian "
            f"resolver={arcturian_resolver} "
            f"detail_level={detail_level}->ignored "
            f"session_tools=[] "
            f"per_turn_tools={[t['name'] for t in _resolver_defs]} "
            f"({len(instructions)} chars)"
        )
    elif companion_mode == "guide-ptt":
        # GuideDevBot's PTT-Hybrid mint (Content-Post #1233):
        # audio-in on PTT press, audio-out for the answer, on-demand,
        # billed against the 30 EUR/month cap. turn_detection is null
        # below so the FE drives every response.create deliberately;
        # combined with replaceTrack(null) on the mic sender, this
        # eliminates the whisper-on-silence hallucinations CloudV2
        # battle-tested on Step-1 narrator.
        # P1 ships with no tools; P2 swaps in knowledge_collection_ask
        # against the Guide knowledge graph.
        instructions = request.instructions or _default_instructions(
            language=request.language or "de",
            track_id=request.track_id,
        )
        companion_tools_override = []
        logger.info(
            f"Realtime: companion_mode=guide-ptt "
            f"voice={request.voice or DEFAULT_REALTIME_VOICE} "
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

    api_key_env, key_herkunft = _openai_key_fuer(grant.profile_id)
    logger.info(
        "Realtime mint: OpenAI-Konto=%s profile=%s",
        key_herkunft, grant.profile_id,
    )
    if not api_key_env:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "openai_api_key_missing",
                "hint": "OPENAI_API_KEY not configured in service env.",
            },
        )

    # Unconditional mint audit line. Previously only the companion_mode
    # branches logged, so a mint WITHOUT a mode passed through silently
    # and received the default Wanderlaut tool set. When AppDevV2's iOS
    # client died on `guard name == expectedName`, nginx showed 33
    # successful mints while our own log showed nothing about them — the
    # one field that would have identified the cause was never written.
    #
    # A caller that gets a tool set it does not expect is exactly the
    # failure this line makes visible, so it runs for every mint, mode or
    # not, before anything can go wrong further down.
    logger.info(
        "Realtime mint: companion_mode=%s affect=%s resolver=%s "
        "detail_level=%s lang=%s override=%s",
        companion_mode or "(none)",
        affect_projection or "(none)",
        (request.arcturian_resolver or DEFAULT_ARCTURIAN_RESOLVER
         if companion_mode == "arcturian" else "(n/a)"),
        detail_level if companion_mode and companion_mode != "arcturian" else "(n/a)",
        request.language or "de",
        "yes" if companion_tools_override is not None else "no(default set)",
    )

    tools = list(
        _all_tool_defs() if companion_tools_override is None
        else companion_tools_override
    )
    # Layer the affect contract on top — additive for every mode except
    # arcturian, which serves it per turn instead. Rationale and the
    # measured incident behind the exception: see _session_tools().
    tools = _session_tools(tools, companion_mode, affect_projection)
    if affect_projection:
        instructions += _affect_projection_addendum(request.language or "de")
        logger.info(
            "Realtime: affect_projection=%s session_tools=%s%s",
            affect_projection, [t["name"] for t in tools],
            " (arcturian: report_affect per-turn only)"
            if companion_mode == "arcturian" else "",
        )
    # OpenAI's GA Realtime session shape (2025-Q4+) nests audio knobs
    # under ``audio.input`` / ``audio.output`` instead of the legacy
    # flat ``voice`` / ``input_audio_format`` fields. The mint endpoint
    # surfaces "Unknown parameter: 'session.voice'" when you use the
    # old layout, so we ship the new layout from day one.
    #
    # turn_detection (CloudV2's first-smoke catch, Post #1215):
    #
    #   * narrator-only / agentos-narrator: read-only modes where the FE
    #     drives ``response.create`` deliberately. Auto-VAD here means
    #     room noise + silence + whisper hallucinations ("see you next
    #     week!") trigger the model to respond on its own. Disable VAD
    #     entirely; the FE controls every speak event.
    #   * talkback-enabled: VAD must be on so the operator's spoken
    #     commands reach the model, but with a stricter threshold + a
    #     longer required silence so room ambience and the
    #     whisper-on-silence hallucination don't feed the assistant
    #     ghost prompts.
    #   * default (Wanderlaut / legacy): the original 0.5/500 ms
    #     server_vad knobs the live tours have been running with.
    if companion_mode in (
        "agentos-narrator", "narrator-only", "guide-ptt",
    ):
        # FE-driven response.create only. These modes are explicitly
        # PTT (guide-ptt) or read-only narration (agentos-narrator,
        # narrator-only) where the FE controls every turn boundary.
        turn_detection = None
    elif companion_mode == "agent-transparent":
        # Originally PR #79 moved this into the `null` group on CloudV2's
        # request after the first live feedback loop
        # (vs_GuideDevBot2_95876): speaker → mic → Whisper → relay →
        # narrate-again. But CloudV2's follow-up live test (Alex'
        # Session, 2026-06-26) showed the FE never wired PTT for this
        # mode — the mic stays open (162→314→466 packets streaming),
        # and with server_vad off the model never sees a turn, never
        # produces a `you`-transcript, never responds. Flip back to
        # server_vad with the handy-mic-friendly threshold CloudV2
        # asked for (0.5 instead of talkback's 0.7).
        #
        # Feedback-loop defense moves to defense-in-depth:
        #   1. ECHO-LOOP-DISZIPLIN in the prompt (model refuses to
        #      relay self-paraphrases — kept from PR #79).
        #   2. CloudV2's FE Relay-Target-Guard (no halucinated
        #      session_id leaves the FE).
        #   3. Cloud's server-side target_focus_mismatch gate (409
        #      catches anything that slips past 1+2).
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 1000,
        }
    elif companion_mode == "talkback-enabled":
        # talkback still ships with strict VAD — its UX is
        # conversational with confirm-chip, not PTT.
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.7,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 1000,
        }
    elif companion_mode == "arcturian":
        # NO server-side turn detection (#886 follow-up, confirmed by
        # AppDevV2 2026-08-05: the client is pure push-to-talk — it sets
        # audio.input.turn_detection = null while the key is held and
        # closes each turn with input_audio_buffer.commit; without PTT it
        # uses semantic_vad with create_response:false. It never relies on
        # the provider answering by itself).
        #
        # server_vad here was actively harmful: the provider creates its
        # own response after 500 ms of silence — one the client never
        # requested, that carries no agentos_response_kind metadata and
        # inherits the session tools. That is Alex' second symptom
        # ("beim zweiten Push-to-Talk kommt nichts mehr an"): a stray
        # provider response colliding with the PTT cycle.
        #
        # Same reasoning as guide-ptt above, which already sets None.
        turn_detection = None
    else:
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
        }
    # Opt-in PTT (CloudV2-Codex, #975). Deliberately AFTER the per-mode
    # branches and deliberately one-way: it can only turn detection OFF,
    # never on. A client that does not send the flag is bit-identical to
    # before — that is the whole point, and it is the lesson from my own
    # a137d86 today, where a server-side default flip would have killed
    # every already-shipped client at once.
    #
    # Tool lists, confirmation and relay contracts are untouched by this;
    # it changes who closes a turn, not what a turn may do.
    if request.push_to_talk and turn_detection is not None:
        logger.info(
            "Realtime: push_to_talk=on — turn_detection forced null "
            "(mode=%s, was %s)",
            companion_mode or "(none)", turn_detection.get("type"),
        )
        turn_detection = None

    # The tool names as they actually go out — the single field a client
    # needs to diagnose a contract mismatch, and the one that was missing
    # when AppDevV2's sessions died on an unexpected function name.
    logger.info(
        "Realtime mint: delivering tools=%s (%d) instructions=%d chars",
        [t.get("name") for t in tools], len(tools), len(instructions or ""),
    )

    session_config = {
        "type": "realtime",
        "model": model,
        "instructions": instructions,
        "tools": tools,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "whisper-1"},
                "turn_detection": turn_detection,
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
    # voice_session_id FIRST — it is the only field whose meaning is
    # "the id this session will heartbeat and report usage with".
    #
    # Before this, session_id was taken as the reservation key. On the
    # iOS client session_id carries an agent name, so the live state
    # held active_sessions: ["3dApi"] while the client sent heartbeats
    # for vs_<uuid>. Every beat answered alive:false — 18 of 18 across
    # two sessions (AppDevV2, 2026-08-05) — and nobody noticed, because
    # the client rightly treats the socket, not the heartbeat, as
    # authoritative.
    #
    # The fallback chain stays for callers that never sent the field.
    voice_session_id = (
        request.voice_session_id
        or request.session_id
        or request.companion_run_id
        or f"vs_pending_{int(time.time()*1000)}"
    )
    try:
        reservation = realtime_budget_guard.reserve_mint(
            profile_id=grant.profile_id,
            user_id=grant.sub,
            voice_session_id=voice_session_id,
            max_parallel_sessions=grant.max_parallel_sessions,
            daily_budget_eur=grant.daily_budget_eur,
            monthly_budget_eur=grant.monthly_budget_eur,
        )
    except BudgetGuardError as exc:
        logger.warning(
            "realtime mint deny code=%s detail=%s",
            exc.error_code, exc.audit_detail,
        )
        # `public_fields` gehen MIT auf die Leitung: Fenster, Zahlen und
        # Reset-Zeitpunkt. Ohne sie kann ein Portal nur den Fehlercode
        # uebersetzen — und der heisst „daily", auch wenn das Fenster
        # monatlich ist. Genau daran hat Alexander fuenf Tage lang
        # geglaubt, es sei morgen wieder da (2026-08-18).
        #
        # `audit_detail` bleibt draussen: Profil und Rohwerte gehoeren
        # ins Protokoll, nicht in eine Antwort an den Browser.
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": exc.error_code, **(exc.public_fields or {})},
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

    # Conversation preface — after the token exists, so a slow or broken
    # cloud-api read can never cost the owner the session itself.
    # arcturian only: the other modes have no durable conversation, and
    # silently prefacing them would be a capability nobody asked for.
    preface_items: List[dict] = []
    prefaced_through_revision: Optional[int] = None
    if request.conversation_id and companion_mode == "arcturian":
        preface_items, prefaced_through_revision = (
            await _fetch_conversation_preface(
                request.conversation_id, authorization or "",
            )
        )
        logger.info(
            "Realtime: preface conversation=%s items=%d through_revision=%s",
            request.conversation_id, len(preface_items), prefaced_through_revision,
        )

    return {
        "provider": "openai",
        "client_secret": body.get("client_secret") or body.get("value") or body,
        "expires_at": body.get("expires_at"),
        "model": model,
        "voice": voice,
        "tools": [t["name"] for t in tools],
        "session_id": request.session_id,
        # Past turns for the client to replay as conversation.item.create
        # right after it connects. Empty list when no conversation_id was
        # given, or when the conversation could not be read — a session
        # without history is degraded, no session at all would be broken.
        "preface": preface_items,
        # The boundary: highest revision contained in `preface`, or null.
        # AppDevV2's live projection resumes at +1 from here, which is what
        # makes a duplicated spoken answer impossible rather than merely
        # unlikely. Null means "nothing was prefaced, project everything".
        "prefaced_through_revision": prefaced_through_revision,
        # The id the reservation was actually booked under. Echoed so the
        # client can heartbeat and report usage with exactly this value
        # instead of assuming which of its own fields we picked — the
        # assumption that produced 18/18 dead heartbeats.
        "voice_session_id": voice_session_id,
        "companion_mode": companion_mode,
        # Provenance of the persona that was ACTUALLY minted (#1002).
        # AppDev needs it to compare a device run against a bench run —
        # today the owner's phone ran an old build for hours while every
        # report said "live", and nobody could tell from a session which
        # persona it carried.
        #
        # Deliberately NOT a code revision: the serving tree on arkserver
        # carries a leftover .git that reports 1d1c876 while the deployed
        # commit is 5fe4043, and realtime_routes.py shows up as untracked
        # there. A revision read from it would be wrong AND look
        # authoritative — the exact failure mode this field exists to
        # prevent. The hash is taken from the instruction string that
        # goes out on THIS request, so it cannot go stale.
        "persona_sha256": hashlib.sha256((instructions or "").encode()).hexdigest(),
        "persona_chars": len(instructions or ""),
        # The turn detection ACTUALLY minted, and whether the opt-in took.
        # A client must be able to ASSERT what it got instead of inferring
        # that the server did what it asked — inferring that is how 18 of
        # 18 heartbeats came back dead (051968b).
        #
        # These two lines lived in the WRONG dict from 7490c18 until now:
        # a find_replace hit the first `"companion_mode": companion_mode,`
        # in the file, which belongs to the unsupported_companion_mode
        # ERROR body. So the mint never echoed either field, the client
        # read `undefined !== null` as a contract violation and refused to
        # open the mic. CloudV2 found it by taking the failing condition
        # apart instead of assuming which half was red.
        "turn_detection": turn_detection,
        "push_to_talk": bool(request.push_to_talk) and turn_detection is None,
        # Null for arcturian: detail_level is narrator cadence and is not
        # applied there (#837). Echoing the requested value would tell the
        # client it took effect when it did not.
        "detail_level": (
            detail_level if companion_mode and companion_mode != "arcturian"
            else None
        ),
        # Echoed so the client can assert the contract is armed before it
        # starts waiting for report_affect calls, instead of inferring it
        # from the tool list.
        "affect_projection": affect_projection,
        # Arcturian turn resolver (#837). Served ready-made for the same
        # reason as the affect follow-up: the named-function tool_choice
        # is silently ignored by Realtime (0/6 measured), so no client
        # should assemble this itself. Send `resolver_followup` BEFORE the
        # spoken answer on every committed user turn; `none` is a valid
        # verdict, not an error. Null when the mode is not arcturian.
        # The NEGOTIATED revision, not a server-chosen one. A client
        # that asked for nothing sees v1 here and can keep comparing for
        # exact equality, which is what it already does.
        "arcturian_resolver": (
            arcturian_resolver if companion_mode == "arcturian" else None
        ),
        "resolver_followup": (
            _arcturian_resolver_followup_payload(arcturian_resolver)
            if companion_mode == "arcturian" else None
        ),
        # Turn-lifecycle correlation (#886). All three responses a turn
        # opens carry a server-set `kind` in response.metadata, which
        # Realtime echoes on response.created AND response.done — also
        # for cancelled ones. Bind on (metadata.kind + response_id), never
        # on arrival order. Add your own turn discriminator to metadata;
        # do not overwrite `kind`.
        "response_kind_field": (
            ARCTURIAN_RESPONSE_KIND_FIELD
            if companion_mode == "arcturian" else None
        ),
        # Reference template for the spoken turn (and greeting / resumed
        # narration, which differ only in the kind value). The adapter
        # builds these natively per wire v1 — this is shipped so it can
        # VALIDATE against the measured contract instead of re-deriving
        # it. The no-tool gate is not cosmetic: a response override
        # inherits the session tools, and omitting the fields produced a
        # voluntary resolver call in 2 of 2 measured runs.
        "primary_audio_response": (
            _arcturian_primary_audio_payload(request.read_tools, request.name_resolution)
            if companion_mode == "arcturian" else None
        ),
        # Fuer Alex' Muster #1035: Schlaegt ein Zug fehl, schickt der
        # Client DIESE Nutzlast und bekommt die eine Zeile, die er nicht
        # selbst bilden kann — die Absicht. Alles andere am Bericht baut
        # er aus eigenen Tatsachen, ohne Modell. Ist der Zug nicht
        # moeglich oder liefert er nichts, entsteht das Issue trotzdem:
        # die Absicht ergaenzt, sie bedingt nicht.
        "report_intent_response": (
            _arcturian_report_intent_payload()
            if companion_mode == "arcturian" else None
        ),
        # Schickt der Client diesen Zug, nachdem der Resolver eine Frage
        # nach einem Agenten erkannt hat, ist der Nachschlag erzwungen
        # statt erhofft. Nur mit `read_tools`, weil die Nutzlast sonst
        # ein Werkzeug forderte, das die Sitzung nicht fuehrt.
        "status_lookup_response": (
            _arcturian_status_lookup_payload()
            if companion_mode == "arcturian" and request.read_tools else None
        ),
        "response_kinds": (
            {
                "server_set": ARCTURIAN_RESPONSE_KINDS_SERVER,
                "adapter_set": ARCTURIAN_RESPONSE_KINDS_ADAPTER,
                "adapter_correlation_keys": ARCTURIAN_CORRELATION_KEYS,
            }
            if companion_mode == "arcturian" else None
        ),
        # The follow-up turn is part of the contract, not an optimisation:
        # letting the model volunteer the call reaches only ~12/18, and the
        # misses land on `concerned`. Served ready-made so no client has to
        # rediscover that the named-function tool_choice is silently
        # ignored by Realtime. Send this after the audio output has
        # drained, then project the returned affect. See issue #767.
        "affect_followup": _affect_followup_payload() if affect_projection else None,
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
async def realtime_cost_status(_: str = Depends(require_operator_key)):
    """Federation-shared Realtime cap state, same shape as the other
    /ai/{provider}/cost-status endpoints."""
    from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker
    return openai_realtime_cost_tracker.get_status()


@router.post("/realtime/cost-status/reset-hard-cap")
async def reset_realtime_hard_cap(_: str = Depends(require_operator_key)):
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
    x_usage_response: Optional[str] = Header(
        default=None,
        alias="X-Usage-Response",
        description=(
            "`minimal` liefert NUR accepted/deduped/cost_eur. Alles "
            "andere (auch fehlend) liefert die bisherige breite Form."
        ),
    ),
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
        cached_text_input_tokens=report.cached_text_input_tokens,
        cached_audio_input_tokens=report.cached_audio_input_tokens,
        duration_sec=report.duration_sec,
        voice_session_id=report.voice_session_id or report.session_id,
        usage_event_id=report.usage_event_id,
    )
    # Charge the per-profile budget guard with the same usage. Skip if
    # this report was a dedup-replay (already counted). The cost in EUR
    # is taken from what the cost tracker just computed for this row.
    zeilen_eur = None
    if result and result.get("accepted") and not result.get("deduped"):
        from ..services.openai_realtime_cost_tracker import openai_realtime_cost_tracker as _tracker
        # The tracker doesn't expose the per-row cost directly, but for
        # the budget guard we only need ``cost_eur`` of THIS turn. Re-
        # compute it locally — same pricing table, deterministic.
        # NB: _cost_for_session returns a (usd, eur) tuple, NOT a dict.
        try:
            _cost_usd, per_row_eur = _tracker._cost_for_session(
                model=report.model,
                audio_input_tokens=report.audio_input_tokens,
                audio_output_tokens=report.audio_output_tokens,
                text_input_tokens=report.text_input_tokens,
                text_output_tokens=report.text_output_tokens,
                cached_text_input_tokens=report.cached_text_input_tokens,
                cached_audio_input_tokens=report.cached_audio_input_tokens,
            )
            zeilen_eur = float(per_row_eur)
            realtime_budget_guard.confirm_usage_charge(
                profile_id=grant.profile_id,
                user_id=grant.sub,
                voice_session_id=report.voice_session_id or report.session_id or "",
                cost_eur=zeilen_eur,
            )
        except Exception as exc:
            # Charging the guard must NEVER tank the usage report —
            # the cost tracker is the source of truth, the guard is
            # the optimisation. Log and continue.
            logger.warning("budget_guard charge failed: %s", exc)

    status = openai_realtime_cost_tracker.get_status()
    status["deduped"] = bool(result and result.get("deduped"))
    status["accepted"] = bool(result and result.get("accepted"))
    # Kosten DIESER Meldung. Ohne sie kann der Aufrufer die Kosten
    # einer Sitzung nicht mitschreiben, ohne die Preistabelle zu
    # kopieren — und eine zweite Preistabelle driftet gegen meine.
    # `None`, wenn nichts verbucht wurde (Nullmeldung oder Dublette):
    # eine 0.0 hier waere von „hat nichts gekostet" nicht zu
    # unterscheiden.
    kosten = round(float(zeilen_eur), 6) if zeilen_eur is not None else None
    status["cost_eur"] = kosten

    # Schmale Form auf Wunsch: NUR was der Aufrufer fuer seine eigene
    # Buchhaltung braucht.
    #
    # Die breite Form traegt Gesamtausgaben, Monatsbudget und
    # Tokenzahlen ueber ALLE Profile — mehr, als ein einzelner
    # Aufrufer wissen muss. Gefaehrlich ist das hinter dem
    # `usage`-Scope nicht, aber es ist dieselbe Form, die im August
    # zugeschlagen hat, als `/ai/*/cost-status` Ausgaben und Budget
    # des Eigentuemers offen ausgeliefert hat (AppDevV2, 2026-08-07).
    #
    # Opt-in und nicht Vorgabe, weil ein Beschneiden der Antwort jeden
    # bestehenden Aufrufer braeche — und ich nicht sicher weiss, wer
    # sie heute alles liest. Wenn niemand mehr die breite Form
    # abruft, wird die schmale zur Vorgabe.
    if (x_usage_response or "").strip().lower() == "minimal":
        return {
            "accepted": status["accepted"],
            "deduped": status["deduped"],
            "cost_eur": kosten,
        }
    return status


class RealtimeSessionEndRequest(BaseModel):
    """Body for ``POST /ai/realtime/session/end`` — CloudV2 explicit
    end-of-session ping so the parallel-slot reservation frees up
    immediately on device-switch (Content-Post #1215, the 60-min
    orphan reaper would otherwise hold the slot).
    """

    voice_session_id: str = Field(
        ...,
        description=(
            "The same id used as session_id / companion_run_id when "
            "minting the token. Owner-scoped — only the user that "
            "originally reserved the slot can release it."
        ),
    )


@router.post("/realtime/session/heartbeat")
async def realtime_session_heartbeat(
    body: RealtimeSessionEndRequest,
    grant: VerifiedGrant = Depends(require_realtime_grant("mint")),
):
    """Owner-scoped heartbeat lease for an active reservation.

    Refreshes the session's last-activity timestamp so the orphan
    reaper does not prune the slot. The FE pings this on a ~30 s
    interval; once ``REALTIME_REAP_SECONDS=90`` is configured on the
    hosts, a crashed tab loses its slot in ~90 s instead of 60 min.

    Returns ``{alive: false}`` if the voice_session_id is no longer
    in the owner's active set — the FE should treat that as a hint
    to stop the local audio + drop the WebRTC PC.

    Scope: ``mint`` (same as reserve/release).
    """
    alive = realtime_budget_guard.refresh_lease(
        profile_id=grant.profile_id,
        user_id=grant.sub,
        voice_session_id=body.voice_session_id,
    )
    if not alive:
        # A heartbeat that finds nothing used to be a silent False. It
        # stayed silent for 18 of 18 beats across two live sessions
        # while the client, correctly, kept talking — the miss was only
        # visible in AppDevV2's device log, never here.
        #
        # A swallowed doubt is fine; an unobserved one is not. We name
        # the keys we DO hold for this owner so the mismatch is legible
        # at a glance instead of requiring someone to read the state
        # file, which is how this was eventually found.
        try:
            known = realtime_budget_guard.active_sessions_for(
                profile_id=grant.profile_id, user_id=grant.sub,
            )
        except Exception as exc:  # diagnostics must never break the path
            known = f"(unavailable: {type(exc).__name__})"
        logger.warning(
            "realtime_heartbeat MISS profile=%s user=%s asked=%s known=%s "
            "(client heartbeats an id we did not book under)",
            grant.profile_id, kurz_id(grant.sub), body.voice_session_id, known,
        )
    return {
        "alive": alive,
        "voice_session_id": body.voice_session_id,
    }


@router.post("/realtime/session/end")
async def realtime_session_end(
    body: RealtimeSessionEndRequest,
    grant: VerifiedGrant = Depends(require_realtime_grant("mint")),
):
    """Owner-scoped release of a pre-mint parallel-slot reservation.

    Idempotent: returns ``{released: false}`` if the session wasn't in
    the caller's active set (already released, never reserved, or
    belongs to another user). Returns ``{released: true}`` on the
    first successful release.

    Scope: ``mint`` — the same scope that booked the slot can release
    it. No separate ``release`` scope (Codex' v1 closed enum is
    mint/usage/devlog only).
    """
    released = realtime_budget_guard.release_by_voice_session(
        profile_id=grant.profile_id,
        user_id=grant.sub,
        voice_session_id=body.voice_session_id,
    )
    return {
        "released": released,
        "voice_session_id": body.voice_session_id,
    }


# ── Dev Narration-Log (CloudV2, Post #1215) ───────────────────────────


DEVLOG_ROOT = "/var/lib/api-ai/devlogs"
DEVLOG_SECRET_ENV = "DEVLOG_DEV_SECRET"
DEVLOG_CONSENT_ENV = "REALTIME_DEVLOG_REQUIRE_CONSENT"
DEVLOG_RETENTION_ENV = "REALTIME_DEVLOG_RETENTION_DAYS"
# Wie oft ein Arbeitsprozess hoechstens durchkehrt. Der Verfall
# muss nicht auf die Minute genau greifen; er muss ueberhaupt
# greifen, ohne bei jedem Schreibvorgang die Ablage abzugehen.
_DEVLOG_SWEEP_INTERVAL_SEC = 3600.0
_devlog_letzter_kehrgang = 0.0


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


def _devlog_retention_days() -> float:
    """Aufbewahrungsfrist in Tagen. 0 oder nicht gesetzt = kein Verfall.

    Ohne Zahl passiert nichts — genau wie heute. Der Wert ist Alex'
    Entscheidung (Issue #1267); hier steht nur die Mechanik, damit die
    Entscheidung eine Zeile ist und kein Umbau.
    """
    roh = os.environ.get(DEVLOG_RETENTION_ENV, "").strip()
    if not roh:
        return 0.0
    try:
        tage = float(roh)
    except ValueError:
        logger.warning(
            "%s=%r ist keine Zahl — Verfall bleibt AUS", DEVLOG_RETENTION_ENV, roh,
        )
        return 0.0
    return tage if tage > 0 else 0.0


def _devlog_kehre_aus(max_alter_tage: float) -> int:
    """Loescht abgelaufene Mitschnitte. Gibt die Anzahl zurueck.

    Das Alter kommt aus ``received_at`` IM Datensatz, nicht aus der
    mtime der Datei: Ein Umzug, ein `cp -r` oder eine Sicherung setzt
    Dateizeiten neu und wuerde die Frist stillschweigend verlaengern.
    Faellt das Feld aus, zieht die mtime als Notbehelf nach.

    Faellt hier etwas um, darf es den Schreibweg nicht mitreissen —
    ein nicht gekehrter Mitschnitt ist ein Aufraeumproblem, ein
    abgestuerzter Schreibvorgang ein Datenverlust.
    """
    if max_alter_tage <= 0 or not os.path.isdir(DEVLOG_ROOT):
        return 0
    grenze = time.time() - max_alter_tage * 86400.0
    geloescht = 0
    for eimer in os.listdir(DEVLOG_ROOT):
        eimer_pfad = os.path.join(DEVLOG_ROOT, eimer)
        if not os.path.isdir(eimer_pfad):
            continue
        for name in os.listdir(eimer_pfad):
            if not name.endswith(".json"):
                continue
            pfad = os.path.join(eimer_pfad, name)
            try:
                with open(pfad, "r", encoding="utf-8") as f:
                    alter = json.load(f).get("received_at")
                if not isinstance(alter, (int, float)):
                    alter = os.path.getmtime(pfad)
                if alter < grenze:
                    os.remove(pfad)
                    geloescht += 1
            except Exception as exc:
                logger.warning("Devlog-Kehrgang: %s uebersprungen (%s)", pfad, exc)
    if geloescht:
        logger.info(
            "Devlog-Kehrgang: %d Mitschnitte aelter als %.1f Tage geloescht",
            geloescht, max_alter_tage,
        )
    return geloescht


def _devlog_kehre_gedrosselt() -> None:
    """Hoechstens einmal je Intervall und Arbeitsprozess.

    Bei zwei uvicorn-Arbeitern kehrt jeder fuer sich — das Loeschen ist
    idempotent, doppelte Laeufe schaden nicht.
    """
    global _devlog_letzter_kehrgang
    tage = _devlog_retention_days()
    if tage <= 0:
        return
    jetzt = time.time()
    if jetzt - _devlog_letzter_kehrgang < _DEVLOG_SWEEP_INTERVAL_SEC:
        return
    _devlog_letzter_kehrgang = jetzt
    try:
        _devlog_kehre_aus(tage)
    except Exception as exc:
        logger.warning("Devlog-Kehrgang fehlgeschlagen: %s", exc)


def _devlog_consent_required() -> bool:
    """Ist der Einwilligungsriegel scharf?

    Default AUS. Das ist bewusst und nicht Bequemlichkeit: Der einzige
    heutige Anrufer (CloudV2) sendet das Feld noch nicht, und ein
    Riegel, der beim Ausrollen sofort greift, nimmt einem Gegenüber die
    Funktion weg, statt sie ihm zu überlassen. Wer die Fähigkeit nutzt,
    legt den Schalter um — nicht wer sie ausliefert.
    """
    return os.environ.get(DEVLOG_CONSENT_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _pruefe_devlog_einwilligung(consent: Optional[bool]) -> None:
    """Fail-closed, wenn der Riegel scharf ist.

    Alles ausser ``True`` ist eine Ablehnung — auch ``None``. Ein
    fehlendes Feld darf nicht als Zustimmung durchgehen, sonst ist der
    Riegel genau für den Client offen, der von ihm nichts weiss.
    """
    if not _devlog_consent_required():
        return
    if consent is True:
        return
    raise HTTPException(
        status_code=403,
        detail={
            "error": "devlog_retention_consent_required",
            "hint": (
                "Dauerhafte Aufbewahrung gesprochener Inhalte ist auf "
                "dieser Instanz zustimmungspflichtig. Sende "
                "retention_consent=true, wenn der SPRECHER zugestimmt "
                "hat. Ohne Zustimmung nichts senden — ein Mitschnitt "
                "ohne Einwilligung ist kein halber Mitschnitt."
            ),
            "field": "retention_consent",
        },
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
    # Vor jedem Schreiben, nicht danach: Der Fehler, den ich hier
    # gerade repariert habe, lag NACH `os.replace` — geschrieben und
    # trotzdem 500 gemeldet. Ein Riegel an derselben Stelle wäre ein
    # Riegel, der die Tür erst hinter dem Gast zuzieht.
    _pruefe_devlog_einwilligung(body.retention_consent)
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
        "retention_consent": body.retention_consent,
        "consent_enforced": _devlog_consent_required(),
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
    # Nach dem Schreiben, nicht davor: Der eigene Mitschnitt ist gerade
    # frisch, kann also nie das Opfer sein — und ein Kehrgang, der
    # klemmt, darf den Schreibvorgang nicht verzoegern.
    _devlog_kehre_gedrosselt()
    logger.info(
        "Devlog upsert: voice_session_id=%s owner=%s lines=%d",
        body.voice_session_id, owner_key, len(body.lines),
    )
    return {
        "accepted": True,
        "voice_session_id": body.voice_session_id,
        "owner": owner_key,
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
    profile_ids = sorted(host_profile_ids())
    key_ok = service_key_configured()
    return {
        "profile_id": profile,
        "profile_ids": profile_ids,
        "key_configured": key_ok,
        "grant_verifier_ready": bool(profile and key_ok),
        "cost_tracker_namespace": profile or "",
        "secret_version": os.environ.get(
            "REALTIME_GRANT_SECRET_VERSION", ""
        ),
        # Befund der Altwerkzeug-Messung. Steht hier und nicht nur im
        # journalctl, weil journalctl ohne sudo lautlos nichts liefert
        # — und eine Messung, die man nur mit dem richtigen Recht
        # sieht, wird als „keine Aufrufe" fehlgelesen.
        "legacy_tool_auth": legacy_auth_befund(),
    }


# ── Tool-routing proxy ────────────────────────────────────────────────


# Allowed Read tools — the only ones the browser is supposed to route
# through AiApi. Display-hint tools must NEVER reach this endpoint
# (the browser shorts them locally); we reject them so a bug there
# surfaces fast instead of silently going to OpenAI's expensive
# fail-mode of "model thinks it called the tool, never got an answer".
# Die zwei Produktwerkzeuge — eigene Menge, weil sie strenger
# behandelt werden als die uebrigen Lesewerkzeuge.
# ── Sprachen ─────────────────────────────────────────────────────────
#
# EINE Tabelle. Es gab bisher vier Kopien im selben Modul, und keine
# einzige kannte Spanisch — waehrend der Produktfinder `es` anbietet.
# Ein spanischer Kunde waehlte also seine Sprache und bekam Deutsch:
# kein Fehler, keine Meldung, nur die falsche Sprache.
#
# Die Kopien in den aelteren Pfaden ruehre ich hier NICHT an — das
# waere eine Aenderung an fremden, laufenden Personas ohne Anlass.
# Sie stehen als Befund im Bericht.
SPRACHNAMEN = {
    "de": "German",
    "en": "English",
    "sl": "Slovenian",
    "it": "Italian",
    "es": "Spanish",
}


def _sprachname(language: Optional[str]) -> str:
    """Sprachname fuer die Anweisung. Unbekanntes faellt auf Deutsch.

    Der Rueckfall ist eine Entscheidung, kein Zufall: Der Aussendienst
    dieses Kunden spricht Deutsch, und eine Sitzung ohne verstaendliche
    Sprache ist schlimmer als eine in der falschen.
    """
    schluessel = (language or "").strip().lower()[:2]
    return SPRACHNAMEN.get(schluessel, "German")


# ── Profilgebundener OpenAI-Schluessel ───────────────────────────────
#
# Der Produktfinder laeuft in der O'Neal-Demo auf dem Konto des Kunden
# (agentos1), nicht auf Alex'. Bis heute las der Mint prozessweit genau
# `OPENAI_API_KEY` — ein Konto fuer alle Profile.
#
# Muster wie beim Sitzungsdeckel: `OPENAI_API_KEY__<PROFIL>` gewinnt,
# sonst der globale Schluessel. Kein Profil ohne eigenen Eintrag
# aendert sein Verhalten.
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def _profil_suffix(profile_id: Optional[str]) -> str:
    """Profilkennung als Variablen-Suffix.

    Bindestriche und Punkte werden zu Unterstrichen, alles gross.
    Umgebungsvariablen tragen keine Bindestriche — ohne diese
    Uebersetzung waere der Eintrag fuer `product-finder` unsetzbar,
    und zwar lautlos: Er fiele auf den globalen Wert zurueck.
    Dieselbe Regel wie in `realtime_budget_guard._session_budget_eur`;
    ein Test haelt beide gegeneinander, damit sie nicht driften.
    """
    return (profile_id or "").strip().replace("-", "_").replace(".", "_").upper()


def _openai_key_fuer(profile_id: Optional[str]) -> tuple:
    """(Schluessel, Herkunft) fuer dieses Profil.

    Herkunft ist `profile` oder `global` — sie geht in die Protokoll-
    zeile des Mints. Ohne sie liesse sich hinterher nicht feststellen,
    WELCHES Konto eine Sitzung bezahlt hat, und genau das ist die
    Frage, um die es hier geht.

    Ein leer gesetzter Profilschluessel gilt als NICHT gesetzt: Eine
    leere Zeichenkette in der `.env` ist ein Tippfehler, kein Wunsch
    nach einem Mint ohne Schluessel.
    """
    suffix = _profil_suffix(profile_id)
    if suffix:
        eigener = (os.environ.get(f"{OPENAI_KEY_ENV}__{suffix}") or "").strip()
        if eigener:
            return eigener, "profile"
    return (os.environ.get(OPENAI_KEY_ENV) or "").strip(), "global"


PRODUCT_TOOL_NAMES = {"find_products", "refine_search", "product_details"}


# ── Kategorien: echte Slugs statt freier Woerter ──────────────────────
#
# Anlass (Alex' erster echter Kundendialog, 00:02): Das Modell fuellte
# `category` mit dem gehoerten Wort — „Bekleidung". oneal verlangt
# kanonische Slugs (`^[a-z0-9][a-z0-9-]*$`), antwortete 422, mein
# Fehlerzweig machte daraus `unavailable`, und der Sprecher sagte
# „Katalog nicht erreichbar". Ein deutsches Substantiv legte den
# Katalog lahm.
#
# Meine Haertung von vorhin filtert unbekannte FELDER. Das hier ist
# dieselbe Luecke eine Ebene tiefer: unbekannte WERTE. Ein Feld ohne
# Enum ist eine Einladung an das Modell, etwas zu erfinden — und im
# Sprachbetrieb erfindet es immer das Wort des Kunden.
#
# Die Liste kommt LIVE von oneal, nicht aus einer Abschrift: Eine
# handgepflegte Kopie waere am Tag der naechsten Katalogaenderung
# still falsch, und zwar wieder als „nicht erreichbar".

ONEAL_KATEGORIEN_TTL_SEC = 3600.0
_kategorien_cache: dict = {"werte": None, "geholt": 0.0}

# Slugs, die es im Katalog gibt, aber nicht im Kundengespraech:
# Ersatzteil-Sammelposten, Display-Material, Altmarken, ein
# Buchhaltungsposten. `z-spare-parts-helmets` ist mit 1357 Eintraegen
# sogar die GROESSTE Kategorie — gaebe ich sie dem Modell, schlaege
# eine Frage nach Helmen zuerst Ersatzvisiere vor.
_KATEGORIE_AUSSCHLUSS = {
    "displays", "merchandise-displays", "old-brands",
    "revenue-without-material-usage",
}


def _ist_kundenkategorie(slug: str) -> bool:
    return bool(slug) and not slug.startswith("z-") and slug not in _KATEGORIE_AUSSCHLUSS


# Rueckfallliste — Stand 2026-08-27, von `GET /v1/categories` geholt.
# Sie greift NUR, wenn oneal beim Mint nicht erreichbar ist. Ohne sie
# haette ein kurzer Ausfall dort ein Werkzeug ohne Kategorien zur
# Folge, und der Kunde bekaeme wieder „Katalog nicht erreichbar" —
# fuer einen Fehler, der laengst vorbei ist.
ONEAL_KATEGORIEN_RUECKFALL = (
    "adv-pants", "bags---backpacks", "boots-adventure", "boots-mx",
    "casual-wear", "face-masks", "gloves", "goggles", "grips",
    "handlebars", "helme-mtb-open-face", "helmets-mtb-full-face",
    "helmets-mx", "helmets-street", "jackets", "jerseys-mtb",
    "jerseys-offroad", "kids-wear-protection", "leather-suits-road",
    "leisure-accessories", "pants-mx", "pants--shorts-mtb",
    "protection-mtb", "protection-mx", "rain-wear", "shoes",
    "sunglasses", "transportation",
)


def _oneal_kategorien() -> List[str]:
    """Kundentaugliche Kategorie-Slugs, gecacht.

    Faellt NIE aus: Bei jedem Fehler kommt die Rueckfallliste. Ein
    Mint darf nicht daran scheitern, dass ein Katalogdienst gerade
    langsam ist — das Werkzeug bliebe sonst ohne Kategorien und der
    Kunde hoerte denselben Satz wie beim eigentlichen Fehler.
    """
    jetzt = time.time()
    if (_kategorien_cache["werte"]
            and jetzt - _kategorien_cache["geholt"] < ONEAL_KATEGORIEN_TTL_SEC):
        return _kategorien_cache["werte"]

    basis = (os.environ.get(ONEAL_SELECTION_BASE_ENV) or "").strip().rstrip("/")
    schluessel = os.environ.get(ONEAL_API_KEY_ENV) or ""
    werte = None
    if basis and schluessel:
        try:
            r = httpx.get(f"{basis}/v1/categories",
                          headers={"X-API-Key": schluessel}, timeout=4.0)
            if r.status_code == 200:
                roh = r.json()
                # oneal antwortet mit {"data": [...]}. Die anderen
                # Namen stehen als Toleranz daneben — aber `data` ist
                # der Fall, der WIRKLICH vorkommt. Ohne ihn lieferte
                # der Abruf HTTP 200 und null Eintraege, und mein
                # Leer-Schutz machte daraus lautlos die Rueckfallliste:
                # gruen an der Naht, tot am Ziel.
                liste = roh if isinstance(roh, list) else (
                    roh.get("data") or roh.get("items")
                    or roh.get("categories") or [])
                slugs = [
                    x if isinstance(x, str) else (x.get("slug") or "")
                    for x in liste
                ]
                gefiltert = sorted({s for s in slugs if _ist_kundenkategorie(s)})
                # Eine leere Antwort ist KEINE gueltige Kategorienliste.
                # Sie zu uebernehmen hiesse, ein leeres Regal fuer einen
                # geraeumten Laden zu halten.
                if gefiltert:
                    werte = gefiltert
            else:
                logger.warning("Kategorien HTTP %s — Rueckfallliste",
                               r.status_code)
        except Exception as exc:
            logger.warning("Kategorien nicht abrufbar (%s) — Rueckfallliste",
                           type(exc).__name__)

    if werte is None:
        werte = list(ONEAL_KATEGORIEN_RUECKFALL)
    _kategorien_cache["werte"] = werte
    _kategorien_cache["geholt"] = jetzt
    return werte

# Felder, die der SERVER setzt — nie das Modell, nie der Browser.
PRODUCT_SERVER_CONTROLLED = frozenset(
    {"brand", "collection_year", "selection_token"}
)


def _product_finder_kriterien_felder() -> set:
    """Die Kriterienfelder AUS dem Werkzeugschema.

    Einzige Wahrheitsquelle: Wer dem Schema ein Feld hinzufuegt, muss
    hier nichts nachziehen — und kann es auch nicht vergessen. Eine
    zweite, handgepflegte Liste waere genau die Stelle, an der ein neues
    Feld lautlos verworfen wuerde.
    """
    schema = _product_finder_tools(None)[0]["parameters"]
    felder = set((schema.get("properties") or {}).keys())
    # Serverseitig gesetzte Felder gehoeren NIE zu den Kriterien, auch
    # wenn sie im Schema stehen: Bei ungebundener Marke fuehrt das
    # Schema `brand`, damit das Modell danach fragen KANN — den Wert
    # setzt trotzdem der Sitzungs-Scope. Liesse ich ihn durch, koennte
    # ein Gespraech die Marke wechseln, an der Bindung vorbei.
    return felder - PRODUCT_SERVER_CONTROLLED


# ── Altwerkzeuge: erst messen, dann durchsetzen ───────────────────────
#
# Die uebrigen Lesewerkzeuge verlangen bis heute keinen Grant. Das ist
# kein Versehen, sondern die Zwischenstufe: Der Wanderlaut-Browser ruft
# sie seit Monaten ohne, eine harte Pflicht braeche ihn im selben
# Moment, in dem sie ausgeliefert wird.
#
# Was hier NICHT passiert: raten. Bevor ich eine Pflicht einschalte,
# muss ich WISSEN, ob der bestehende Aufrufer ueberhaupt etwas mitsendet
# — sonst ist die Umstellung ein Wurf mit verbundenen Augen. Deshalb
# zwei Stufen an EINEM Schalter:
#
#   REALTIME_LEGACY_TOOL_AUTH=off      (Vorgabe) — nur zaehlen
#   REALTIME_LEGACY_TOOL_AUTH=enforce            — Grant verlangen
#
# Die Messung kostet bewusst KEINEN Netzaufruf. Die Frage der Stufe 1
# lautet nicht „ist der Grant gueltig", sondern „kommt ueberhaupt
# einer" — und die beantwortet der Kopf allein. Ein Austausch gegen
# auth-api auf dem heissen Pfad kostete einen Roundtrip im 250-ms-
# Budget, fuer eine Antwort, die ich nicht brauche.
LEGACY_TOOL_AUTH_ENV = "REALTIME_LEGACY_TOOL_AUTH"

# Zaehler je (Werkzeug, mit/ohne Kopf). Prozesslokal und absichtlich
# schlicht: Er beantwortet eine einzige Frage und wird danach entfernt.
_legacy_auth_zaehler: dict = {}
_LEGACY_LOG_INTERVALL = 50


def _legacy_auth_modus() -> str:
    """`off` (Vorgabe) oder `enforce`. Unbekanntes faellt auf `off`.

    Fail-open ist hier richtig, obwohl es sonst falsch waere: Ein
    Tippfehler in der Variable darf nicht den laufenden Wanderlaut-
    Browser abschalten. Die Stelle, an der fail-closed zaehlt — die
    Produktwerkzeuge — haengt nicht an diesem Schalter.
    """
    roh = (os.environ.get(LEGACY_TOOL_AUTH_ENV) or "").strip().lower()
    if roh in {"off", "enforce"}:
        return roh
    if roh:
        logger.warning(
            "%s=%r unbekannt — Altwerkzeuge bleiben ungeschuetzt (off)",
            LEGACY_TOOL_AUTH_ENV, roh,
        )
    return "off"


def _hat_bearer(authorization: Optional[str]) -> bool:
    """Traegt der Aufruf ueberhaupt ein Bearer-Token?

    Prueft NICHT, ob es gueltig ist — das waere eine andere Frage und
    ein Netzaufruf. `True` heisst nur: hier ist etwas, das man
    einloesen koennte.
    """
    if not authorization:
        return False
    teile = authorization.split(None, 1)
    return len(teile) == 2 and teile[0].lower() == "bearer" and bool(teile[1].strip())


def _log_auth_presence(tool_name: str, authorization: Optional[str]) -> bool:
    """Zaehle, ob Altwerkzeug-Aufrufe einen Bearer mitbringen.

    Gibt zurueck, ob einer da war. Loggt gedrosselt, damit die Messung
    nicht selbst zum Problem wird: bei jedem Aufruf eine Zeile waere
    bei ~60 Werkzeugaufrufen je Sprachsitzung Rauschen, in dem der
    Befund untergeht.
    """
    da = _hat_bearer(authorization)
    schluessel = (tool_name, da)
    n = _legacy_auth_zaehler.get(schluessel, 0) + 1
    _legacy_auth_zaehler[schluessel] = n
    if n == 1 or n % _LEGACY_LOG_INTERVALL == 0:
        logger.info(
            "Altwerkzeug-Befund: %s %s Bearer — %sx seit Prozessstart "
            "(Modus=%s)",
            tool_name, "MIT" if da else "OHNE", n, _legacy_auth_modus(),
        )
    return da


def legacy_auth_befund() -> dict:
    """Der Befund als Datenstruktur, fuer den Statusendpunkt.

    Ohne diese Ausleitung stuende die Messung nur im journalctl — und
    journalctl liefert ohne sudo lautlos nichts zurueck. Eine Messung,
    die man nur mit dem richtigen Recht sieht, wird als „keine Aufrufe"
    fehlgelesen.
    """
    return {
        "mode": _legacy_auth_modus(),
        "counts": {
            f"{werkzeug}:{'mit' if da else 'ohne'}": n
            for (werkzeug, da), n in sorted(_legacy_auth_zaehler.items())
        },
    }

READ_TOOL_NAMES = {"knowledge_query", "pois_near", "narration_near", "osm_nearby",
                   "resolve_agent_name",
                   "agent_status",
                   # Produktfinder — beide lesend, beide ohne Produktdaten
                   # im Rueckgabewert (Bauplan #4831, Vertrag c887a89).
                   "find_products", "refine_search", "product_details"}


# Zahlwoerter aus #1037, deutsch und englisch. Die Faltung muss auf
# BEIDEN Seiten identisch sein, sonst findet sie nichts — deshalb steht
# die Tabelle hier vollstaendig statt „sinngemaess".
_ZAHLWOERTER = {
    "null": "0", "zero": "0", "eins": "1", "ein": "1", "one": "1",
    "zwei": "2", "two": "2", "drei": "3", "three": "3", "vier": "4",
    "four": "4", "fuenf": "5", "five": "5", "sechs": "6", "six": "6",
    "sieben": "7", "seven": "7", "acht": "8", "eight": "8",
    "neun": "9", "nine": "9", "zehn": "10", "ten": "10",
}


def _fold_agent_name(name: str) -> str:
    """Gesprochenen Namen auf die Vergleichsform falten (#1037).

    NFKD, Kleinschreibung, Umlaute ausgeschrieben, Zahlwoerter zu
    Ziffern, dann alles ausser [a-z0-9] entfernen. „drei D API" wird
    `3dapi`, „Cloud V zwei" wird `cloudv2`, „S W F M E" wird `swfme`.

    Bewusst NUR Faltung, keine Aehnlichkeit: Jaro-Winkler und Phonetik
    gehoeren nach cloud-api, wo Bestand, Zustand und Eigentuemer aller
    ~200 Agenten liegen. Zwei Implementierungen derselben Heuristik
    driften garantiert auseinander — die exakte Faltung dagegen ist
    reproduzierbar und laesst sich gegen dieselbe Tabelle pruefen.
    """
    import unicodedata
    # Umlaute ZUERST, dann NFKD. Die Reihenfolge in #1037 ist andersherum
    # notiert und funktioniert so nicht: NFKD zerlegt „ö" in „o" + ein
    # kombinierendes Zeichen, und danach findet `.replace("ö","oe")`
    # nichts mehr. Gemessen 2026-08-12: „Förderungen" faltete auf
    # `forderungen`, waehrend die Tabelle `foerderungen` verlangt — ein
    # Unterschied, der genau den Agenten unauffindbar macht, der in der
    # Tabelle als Beispiel steht.
    t = (name or "").lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    woerter = [_ZAHLWOERTER.get(w, w) for w in t.split()]
    return "".join(woerter)


async def _resolve_agent_by_fold(
    gehoert: str, base: str, hdrs: dict
) -> tuple[Optional[str], List[str]]:
    """Gefalteten Namen gegen die Sitzungsliste aufloesen.

    Liefert (eindeutiger Name, Kandidaten). Bei mehreren Treffern ist
    der erste Wert None und die Kandidaten sind gefuellt — dann wird
    NICHT geraten, sondern zurueckgefragt. `3DApi` und `3dApi`
    existieren beide und falten auf `3dapi`; genau dafuer ist der Fall
    da (#1037 nennt `Kitt` / `K.I.T.T.` als dasselbe Muster).
    """
    ziel = _fold_agent_name(gehoert)
    if not ziel:
        return None, []
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(f"{base}/api/sessions", headers=hdrs)
        namen = [
            (x.get("name") or x.get("session_name") or "")
            for x in ((r.json() or {}).get("sessions") or [])
        ] if r.status_code == 200 else []
    except Exception:
        return None, []
    treffer = [n for n in namen if n and _fold_agent_name(n) == ziel]
    if len(treffer) == 1:
        return treffer[0], treffer
    return None, treffer


async def _tool_agent_status(args: dict, authorization: Optional[str]) -> Any:
    """Agentenzustände lesen — UNTER DER KENNUNG DES AUFRUFERS.

    Der Bearer des Operators wird durchgereicht, nicht ersetzt. Zwei
    Gründe, beide von CloudV2 benannt: Er soll sehen, was ER sehen darf;
    und sobald Clouds Absicherung von `/api/agents/status` ausgeliefert
    ist, filtert der Endpunkt automatisch richtig, statt plötzlich leer
    zu antworten oder zu viel zu zeigen.

    Ohne Bearer wird NICHT anonym gelesen — heute antwortet der Endpunkt
    zwar noch ungefiltert, und genau das wäre der Moment, in dem ein
    Sprachagent Zustände fremder Nutzer vorliest.
    """
    if not authorization:
        return {"ok": False, "error": "no_caller_identity",
                "hint": "Der Client muss den Bearer des Operators durchreichen."}
    agent = (args.get("agent") or "").strip()
    base = os.getenv("CLOUD_API_URL", "https://cloud-api.arkserver.arkturian.com")
    hdrs = {"Authorization": authorization}

    if not agent:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{base}/api/agents/status", headers=hdrs)
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "error": r.text[:200]}
        # `data` bleibt `data` — der Web-Client kompaktiert genau ueber
        # `result.data.agents`. Beim Umbau auf `board` (2026-08-10) hatte
        # ich diesen Zweig mitbenannt; CloudV2-Codex hat es gefunden,
        # bevor es lief. Ohne den Schluessel waere die volle Agentenliste
        # unkompaktiert in den Modellkontext gelaufen.
        # `scope` ist das maschinenlesbare Merkmal, an dem der Client die
        # Uebersicht erkennt. `agent: "(alle)"` bleibt daneben stehen,
        # ist aber ANZEIGETEXT: deutsch, uebersetzbar, jederzeit
        # aenderbar. CloudV2-Codex hatte seine fail-closed-Pruefung an
        # genau dieses Literal gehaengt — dann bricht die Uebersicht in
        # dem Moment, in dem jemand das Wort anfasst.
        return {"ok": True, "scope": "all", "agent": "(alle)", "data": r.json()}

    async def _get(path: str, timeout: float) -> Optional[Any]:
        """Eine Quelle holen; Ausfall ist erlaubt und wird zu None.

        Kein `raise`: Fehlt eine der drei Quellen, soll die Antwort
        aermer werden, nicht ausbleiben. Ein Sprachagent, der auf eine
        langsame Verlaufsdatei wartet, haengt hoerbar.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}{path}", headers=hdrs)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    board, live, hist = await asyncio.gather(
        _get(f"/api/agents/{agent}/status", 5.0),
        _get(f"/api/sessions/{agent}/agent-state", 5.0),
        # Fuenf statt einem Zug: Ein einzelner letzter Zug ist zu wenig
        # fuer „was hat er gemacht" — und wenn dieser Zug zufaellig eine
        # Systemzustellung war, ist er schlechter als nichts. Alexander
        # am 2026-08-12 zu CHAP: Werkzeug lief, Antwort blieb aus.
        _get(f"/api/sessions/{agent}/history?limit=5", 8.0),
    )

    last_reply: Optional[str] = None
    last_reply_at: Optional[str] = None
    recent: List[dict] = []
    turns = (hist or {}).get("turns") or []
    for turn in reversed(turns):  # neueste zuerst
        texts = [
            (sec.get("content") or "").strip()
            for sec in (turn.get("sections") or [])
            if sec.get("kind") == "text"
        ]
        texts = [t for t in texts if t]
        if not texts:
            continue
        wann = turn.get("ended_at") or turn.get("timestamp")
        recent.append({"at": wann, "said": _condense_for_speech(texts[-1])})
        if last_reply is None:
            last_reply, last_reply_at = recent[0]["said"], wann

    # Das ALTER ist der Teil, der die Ehrlichkeit ueberhaupt erst
    # moeglich macht: Ohne es kann er „das Letzte ist fuenf Tage alt"
    # nicht sagen und schweigt stattdessen.
    age_days = _age_in_days(last_reply_at)

    state = (live or {}).get("state")

    # Nichts gefunden? Dann lag es vielleicht am gehoerten Namen. Alex,
    # 2026-08-12: „Teilweise durch die Spracherkennung stimmt der Name
    # nicht ganz und dann checkt er nicht." Er hatte recht — bis hierher
    # ging der Name unveraendert in drei Abfragen, und „drei D API"
    # trifft keinen Agenten.
    #
    # Nur EIN Nachversuch, nur bei exakter Faltung, nie geraten.
    # Ein UNBEKANNTER Agent liefert `state: "dead"`, nicht etwa nichts.
    # Die erste Fassung dieser Bedingung fragte `if not any([board,
    # state, last_reply])` — und `"dead"` ist wahr, also lief die
    # Aufloesung nie an. Gemessen am 2026-08-12 mit einem echten Aufruf
    # (`agent="drei D API"` -> `resolved_from: None`); alle Unit-Tests
    # waren gruen, weil die Doubles dort 500 liefern statt „dead".
    #
    # Es zaehlt also nicht „irgendetwas kam zurueck", sondern „etwas
    # BRAUCHBARES kam zurueck".
    _brauchbar = bool(board) or bool(last_reply) or (
        state not in (None, "", "dead", "unknown")
    )
    resolved_from: Optional[str] = None
    if not _brauchbar:
        # Clouds Endpunkt ZUERST — er kennt Bestand, Zustand und
        # Eigentuemer aller Agenten und entscheidet Gleichstaende
        # danach. Verifiziert 2026-08-12 gegen die Tabelle aus #1037,
        # 11 von 11: „drei D API" -> 3dApi, „Foerderungen" korrekt
        # gefaltet, „K I T T" ehrlich `ambiguous`. Er trennt sogar
        # `3dApi` von `3DApi`, was meine reine Faltung NICHT kann.
        #
        # Meine lokale Faltung bleibt nur als Rueckfall, wenn er nicht
        # antwortet — zwei Implementierungen derselben Heuristik driften
        # sonst auseinander (#1037).
        kanonisch, kandidaten = None, []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                rr_ = await client.post(
                    f"{base}/api/agents/resolve-name", headers=hdrs,
                    json={"spoken": agent, "limit": 5, "authorized_only": False},
                )
            if rr_.status_code == 200:
                d = rr_.json() or {}
                if d.get("decision") == "unique":
                    kanonisch = ((d.get("match") or {}).get("name")) or None
                elif d.get("decision") == "ambiguous":
                    kandidaten = [c.get("name") for c in (d.get("candidates") or [])
                                  if c.get("name")]
        except Exception:
            pass
        if not kanonisch and not kandidaten:
            kanonisch, kandidaten = await _resolve_agent_by_fold(agent, base, hdrs)
        if len(kandidaten) > 1:
            # `3DApi` und `3dApi` falten beide auf `3dapi`. Hier wird
            # zurueckgefragt, nicht gewaehlt — ein stiller Fehlgriff
            # liest den falschen Agenten vor.
            return {"ok": False, "agent": agent,
                    "error": "ambiguous_agent_name",
                    "candidates": kandidaten,
                    "hint": ("Mehrere Agenten passen auf diesen Namen. FRAG "
                             "ZURUECK und nenne die Kandidaten, waehle NICHT "
                             "selbst.")}
        if kanonisch and kanonisch != agent:
            resolved_from, agent = agent, kanonisch
            board, live, hist = await asyncio.gather(
                _get(f"/api/agents/{agent}/status", 5.0),
                _get(f"/api/sessions/{agent}/agent-state", 5.0),
                _get(f"/api/sessions/{agent}/history?limit=5", 8.0),
            )
            state = (live or {}).get("state")
            recent = []
            last_reply = last_reply_at = None
            for turn in reversed((hist or {}).get("turns") or []):
                texts = [(sec.get("content") or "").strip()
                         for sec in (turn.get("sections") or [])
                         if sec.get("kind") == "text"]
                texts = [t for t in texts if t]
                if not texts:
                    continue
                wann = turn.get("ended_at") or turn.get("timestamp")
                recent.append({"at": wann, "said": _condense_for_speech(texts[-1])})
                if last_reply is None:
                    last_reply, last_reply_at = recent[0]["said"], wann
            age_days = _age_in_days(last_reply_at)

    if not (bool(board) or bool(last_reply)
            or state not in (None, "", "dead", "unknown")):
        return {"ok": False, "agent": agent, "error": "nothing_readable",
                "hint": "Kein Zustand und keine Antwort lesbar — sag, dass "
                        "du zu diesem Agenten gerade nichts weisst, und "
                        "erfinde nichts."}

    return {
        "ok": True,
        "scope": "one",
        "agent": agent,
        "state": state,
        "board": board,
        "last_reply": last_reply,
        "last_reply_at": last_reply_at,
        "last_reply_age_days": age_days,
        # Gesetzt, wenn der gehoerte Name nicht traf und ueber die
        # Faltung aufgeloest wurde. Der Agent SOLL das sagen — „ich habe
        # das als 3dApi verstanden" ist eine ehrliche Auskunft, ein
        # stilles Umdeuten waere es nicht.
        "resolved_from": resolved_from,
        # Bis zu fuenf Zuege, neueste zuerst. `last_reply` bleibt als
        # Bequemlichkeit erhalten, damit CloudV2s Client nichts umbauen
        # muss — es ist immer `recent[0]["said"]`.
        "recent": recent,
    }


def _age_in_days(stamp: Optional[str]) -> Optional[int]:
    """Alter eines ISO-Zeitstempels in ganzen Tagen, oder None.

    Bewusst grob: Der Agent soll „vor fuenf Tagen" sagen koennen, nicht
    „vor 4 Tagen, 18 Stunden und 12 Minuten". Eine Genauigkeit, die
    niemand hoert, ist nur eine weitere Stelle, an der etwas falsch
    sein kann.
    """
    if not stamp:
        return None
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - t).days)
    except Exception:
        return None


_SPEECH_MAX_CHARS = 700


def _condense_for_speech(text: str) -> str:
    """Fliesstext einer Agentenantwort auf etwas Sprechbares kuerzen.

    Agentenantworten sind fuer Augen geschrieben: Markdown-Zeichen,
    Codebloecke, Tabellen, Werkzeug-JSON. Vorgelesen wird daraus
    Zeichensalat. Wir nehmen die Prosa und kappen an einer Satzgrenze,
    damit der Agent zitieren kann, ohne mitten im Wort abzubrechen.
    """
    body = re.sub(r"```.*?```", " ", text, flags=re.S)
    body = re.sub(r"^\s*[|>#-]+\s*", "", body, flags=re.M)
    body = re.sub(r"[*_`]+", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) <= _SPEECH_MAX_CHARS:
        return body
    cut = body[:_SPEECH_MAX_CHARS]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: stop + 1] if stop > 200 else cut).strip() + " …"


async def _tool_resolve_agent_name(args: dict, authorization: Optional[str]) -> Any:
    """Gehoerten Namen aufloesen — deterministisch, kein Modell (#1037).

    Alex' Vorgabe woertlich: "aber nicht ueber eine KI, sondern ueber
    eine semantische Aehnlichkeitssuche vom Namen her." Der Grund ist
    Reproduzierbarkeit: Eine Namensaufloesung, die bei gleicher Eingabe
    zweimal Verschiedenes liefert, ist schlimmer als eine, die zugibt,
    dass sie es nicht weiss.

    Serverseitig, weil nur cloud-api den vollstaendigen Bestand, den
    Zustand und den Eigentuemer kennt: `Kitt` und `K.I.T.T.` falten
    beide auf `kitt`, und nur der Zustand trennt sie.

    Wie ueberall hier: unter der Kennung des Aufrufers, nie mit einer
    erweiterten Identitaet.
    """
    if not authorization:
        return {"ok": False, "error": "no_caller_identity",
                "hint": "Der Client muss den Bearer des Operators durchreichen."}
    spoken = (args.get("spoken") or "").strip()
    if not spoken:
        return {"ok": False, "error": "empty_spoken_name"}
    base = os.getenv("CLOUD_API_URL", "https://cloud-api.arkserver.arkturian.com")
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(
                f"{base}/api/agents/resolve-name",
                headers={"Authorization": authorization},
                json={"spoken": spoken, "limit": 5, "authorized_only": False},
            )
    except Exception as e:
        return {"ok": False, "error": "resolver_unreachable", "detail": str(e)[:120]}
    if r.status_code == 404:
        # Der Endpunkt existiert noch nicht auf jedem Knoten. Ein
        # ehrliches "kann ich nicht" ist hier richtig — nicht raten.
        return {"ok": False, "error": "name_resolution_unavailable"}
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": r.text[:200]}
    return {"ok": True, **r.json()}


# ── Produktfinder-Werkzeuge (Bauplan #4831, Vertrag c887a89/bad029d) ──

ONEAL_SELECTION_BASE_ENV = "ONEAL_SELECTION_BASE_URL"
ONEAL_INTERNAL_KEY_ENV = "ONEAL_REALTIME_INTERNAL_KEY"
ONEAL_API_KEY_ENV = "ONEAL_API_KEY"

# Drei Zustaende, und die Trennung ist der Punkt: `matches` und `empty`
# kommen von der Gegenseite — sie weiss, ob nichts da ist. `unavailable`
# ist AUSSCHLIESSLICH unsere Uebersetzung eines technischen Fehlers — wir
# wissen nur, ob wir sie erreicht haben. Wer die beiden verwechselt, sagt
# einem Kunden "das fuehren wir nicht", waehrend der Katalog bloss
# klemmt: eine falsche Auskunft ueber das Sortiment.
_NICHT_ERREICHBAR = {"status": "unavailable"}


async def _tool_product_search(
    args: dict,
    session_id: Optional[str],
    verfeinern: bool,
    diagnose: Optional[dict] = None,
) -> dict:
    """`find_products` und `refine_search` — eine Route, ein Unterschied.

    Der Unterschied ist `base_selection_token`: ohne ihn eine neue
    Suche, mit ihm ein Kriterien-Patch, den die Gegenseite serverseitig
    mit der gespeicherten Auswahl zusammenfuehrt. Damit halten wir
    weder Filterzustand noch SQL-Semantik doppelt.

    **Das Ergebnis wird UNVERAENDERT durchgereicht**, `__app_command__`
    eingeschlossen. Der Browser-Core loest ihn heraus, fuehrt ihn lokal
    aus und schickt nur das bereinigte Resultat ans Modell. Entfernten
    wir ihn hier, oeffnete die Trefferflaeche nie — obwohl die Suche
    erfolgreich war (geklaert im Plan, belegt durch
    `ProductFinderRealtimeAdapter.test.ts`).
    """
    basis = (os.environ.get(ONEAL_SELECTION_BASE_ENV) or "").strip().rstrip("/")
    internal = os.environ.get(ONEAL_INTERNAL_KEY_ENV) or ""
    if not basis or not internal:
        logger.warning(
            "Produktsuche nicht konfiguriert (%s/%s fehlen) — als "
            "'unavailable' gemeldet", ONEAL_SELECTION_BASE_ENV,
            ONEAL_INTERNAL_KEY_ENV,
        )
        return dict(_NICHT_ERREICHBAR)

    # Marke und Jahr kommen aus dem SERVERSEITIGEN Sitzungs-Scope, nie
    # aus den Argumenten des Modells und nie vom Browser.
    scope = realtime_session_scope.lesen(session_id)
    if scope is None:
        # Unbekannter Scope ist NICHT dasselbe wie markenoffen. Fuer den
        # Sprecher klingt es gleich ("ich kann das nicht nachsehen"),
        # im Protokoll muss es unterscheidbar bleiben.
        logger.warning(
            "Produktsuche ohne Sitzungs-Scope (session_id=%r) — "
            "als 'unavailable' gemeldet", session_id,
        )
        return dict(_NICHT_ERREICHBAR)

    # Nur Felder weiterreichen, die im Werkzeugschema stehen.
    #
    # Das Schema traegt `additionalProperties: false`, aber ohne
    # `strict: true` ist das fuer OpenAI ein HINWEIS, keine Schranke —
    # das Modell darf ein Feld erfinden. Reichte ich es durch, antwortet
    # oneal mit 422, mein Fehlerzweig macht daraus `unavailable`, und
    # der Kunde hoert „Katalog gerade nicht erreichbar". Ein erfundenes
    # Wort des Modells sperrte damit den ganzen Katalog.
    #
    # Die Liste kommt aus dem Schema selbst, nicht aus einer zweiten
    # Aufzaehlung — zwei Listen fuer dieselbe Aussage driften.
    erlaubt = _product_finder_kriterien_felder()
    kriterien = {k: v for k, v in (args or {}).items() if k in erlaubt}
    verworfen = sorted(set(args or {}) - erlaubt
                       - {"selection_token", "brand", "collection_year"})
    if verworfen:
        logger.info(
            "Produktsuche: unbekannte Kriterien verworfen %s "
            "(Modell hat sie erfunden, Schema kennt sie nicht)", verworfen,
        )
        # Auch in den Umschlag, nicht nur ins Log: journalctl liefert
        # ohne sudo lautlos nichts, und im Sitzungsprotokoll stuende
        # sonst ein Aufruf OHNE dieses Feld — niemand koennte
        # rekonstruieren, dass das Modell es gesagt hat.
        if diagnose is not None:
            diagnose["dropped_criteria"] = verworfen
    nutzlast = {
        "session_id": session_id,
        "brand": scope.get("brand"),
        "collection_year": scope.get("collection_year"),
        "criteria": kriterien,
        # Das Token kommt aus dem SERVERSEITIGEN Scope, nicht vom
        # Modell: oneal entfernt es aus dem Werkzeugergebnis (0e3ea84),
        # der Browser ergaenzt es nicht (am Client-Code geprueft). Ein
        # vom Modell mitgeschicktes Token waere entweder erfunden oder
        # aus einer fremden Sitzung — beides nehme ich nicht.
        "base_selection_token": (
            realtime_session_scope.token_lesen(session_id)
            if verfeinern else None
        ),
    }
    kopf = {
        "X-Realtime-Internal-Key": internal,
        "Content-Type": "application/json",
    }
    if os.environ.get(ONEAL_API_KEY_ENV):
        kopf["X-API-Key"] = os.environ[ONEAL_API_KEY_ENV]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{basis}/v1/realtime/selections/search",
                json=nutzlast, headers=kopf,
            )
    except Exception as exc:
        logger.warning("Produktsuche nicht erreichbar: %s: %s",
                       type(exc).__name__, exc)
        return dict(_NICHT_ERREICHBAR)

    if r.status_code != 200:
        # Auch 4xx sind hier technische Fehler: Ein abgelaufenes Token
        # oder ein Schemafehler heisst nicht "es gibt nichts".
        logger.warning("Produktsuche HTTP %s: %s", r.status_code, r.text[:200])
        return dict(_NICHT_ERREICHBAR)
    try:
        ergebnis = r.json()
    except Exception as exc:
        logger.warning("Produktsuche: Antwort nicht lesbar (%s)", exc)
        return dict(_NICHT_ERREICHBAR)

    # Das ausgegebene Auswahl-Token an der Sitzung halten, damit eine
    # spaetere Verfeinerung darauf aufsetzen kann, OHNE dass das Modell
    # es je zu sehen bekommt. Fehlschlaege sind hier kein Grund, das
    # Ergebnis zu verwerfen: Der Kunde hat seine Treffer, nur die
    # naechste Verfeinerung waere dann eine neue Suche.
    # Was oneal von meinen Kriterien verworfen hat, gehoert in MEIN
    # Protokoll — sonst steht der Befund nur auf einer Seite. Ein
    # verworfenes Kriterium ist kein Fehler (der Kunde bekommt
    # Treffer), aber es ist der Fruehwarnwert dafuer, dass mein
    # Werkzeugschema und ihr Vokabular auseinanderlaufen.
    try:
        verworfen_dort = (
            ((ergebnis or {}).get("hints") or {}).get("ignored_criteria") or []
        )
        if verworfen_dort:
            logger.info(
                "Produktsuche: oneal hat Kriterien verworfen %s "
                "(Schema und Vokabular driften)", verworfen_dort,
            )
    except Exception:
        pass

    try:
        befehl = (ergebnis or {}).get("__app_command__") or {}
        token = ((befehl.get("args") or {}).get("selection_token")
                 if isinstance(befehl, dict) else None)
        if token:
            realtime_session_scope.token_merken(session_id, token)
    except Exception as exc:
        logger.warning("Auswahl-Token nicht gemerkt (%s)", exc)

    # Unveraendert zurueck — einschliesslich __app_command__. Was der
    # Browser bekommt, entscheidet nicht dieser Handler.
    return ergebnis


async def _tool_product_details(
    args: dict,
    session_id: Optional[str],
    diagnose: Optional[dict] = None,
) -> dict:
    """`product_details` — der Agent spricht ueber das offene Produkt.

    Der Unterschied zu den beiden Suchwerkzeugen: Hier kommt Produkt-
    TEXT in den Modellkontext, und das ist der Zweck. Die Auswahl
    trifft trotzdem der Server — das Modell nennt hoechstens eine
    Ordnungszahl, welches Produkt gemeint ist, entscheidet oneal aus
    der sitzungsgebundenen Auswahl.

    **Ergebnis unveraendert durchgereicht**, `status` eingeschlossen.
    `no_focus` und `no_such_position` sind ERFOLGE (HTTP 200), keine
    Fehler: „nichts geoeffnet" und „diese Nummer gibt es nicht" sind
    Auskuenfte, die der Sprecher geben soll. Nur ein echter Ausfall
    wird zu `unavailable` — der Unterschied entscheidet, ob der Kunde
    auf seinen Bildschirm schaut oder die Technik verdaechtigt.
    """
    basis = (os.environ.get(ONEAL_SELECTION_BASE_ENV) or "").strip().rstrip("/")
    internal = os.environ.get(ONEAL_INTERNAL_KEY_ENV) or ""
    if not basis or not internal:
        logger.warning(
            "Produktdetails nicht konfiguriert (%s/%s fehlen) — als "
            "'unavailable' gemeldet", ONEAL_SELECTION_BASE_ENV,
            ONEAL_INTERNAL_KEY_ENV,
        )
        return dict(_NICHT_ERREICHBAR)

    if not session_id:
        # Ohne Sitzung kann oneal weder Fokus noch Auswahl aufloesen.
        # Das ist kein Ausfall, aber auch kein Fachzustand, den ich
        # erfinden darf — also der ehrliche technische Zustand.
        logger.warning("Produktdetails ohne Sitzungskennung")
        return dict(_NICHT_ERREICHBAR)

    nutzlast: dict = {"session_id": session_id}

    # Ordnungszahlen: strikt 1-basiert (Vertrag mit OnealServ-Codex).
    # oneal weist 0 und negative Werte mit 422 ab — voellig richtig,
    # aber ein 422 landet bei mir im Ausfall-Zweig und der Kunde hoerte
    # „Katalog nicht erreichbar", weil das Modell „das nullte" gesagt
    # hat. Also verwerfe ich solche Werte HIER und frage nach dem
    # fokussierten Produkt: „Das nullte" hat keine Bedeutung, das
    # offene Produkt ist die beste verfuegbare.
    roh = (args or {}).get("position")
    if roh is not None:
        try:
            pos = int(roh)
        except (TypeError, ValueError):
            pos = 0
        if pos >= 1:
            nutzlast["position"] = pos
        else:
            logger.info(
                "Produktdetails: unbrauchbare Position %r verworfen — "
                "frage stattdessen das fokussierte Produkt", roh,
            )
            if diagnose is not None:
                diagnose["dropped_position"] = roh

    kopf = {
        "X-Realtime-Internal-Key": internal,
        "Content-Type": "application/json",
    }
    if os.environ.get(ONEAL_API_KEY_ENV):
        kopf["X-API-Key"] = os.environ[ONEAL_API_KEY_ENV]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{basis}/v1/realtime/selections/details",
                json=nutzlast, headers=kopf,
            )
    except Exception as exc:
        logger.warning("Produktdetails nicht erreichbar (%s)",
                       type(exc).__name__)
        return dict(_NICHT_ERREICHBAR)

    if r.status_code != 200:
        # Auch 4xx sind hier technische Fehler. Die Fachzustaende
        # kommen laut Vertrag mit 200 und einem `status`-Feld.
        logger.warning("Produktdetails HTTP %s: %s",
                       r.status_code, r.text[:200])
        return dict(_NICHT_ERREICHBAR)

    try:
        return r.json()
    except Exception as exc:
        logger.warning("Produktdetails: Antwort nicht lesbar (%s)", exc)
        return dict(_NICHT_ERREICHBAR)


@router.post("/realtime/tool/{tool_name}")
async def realtime_tool_call(
    tool_name: str = Path(..., description="Function name from the model"),
    body: RealtimeToolCall = ...,
    x_session_id: Optional[str] = Header(
        default=None,
        alias="X-Session-ID",
        description="Guide-api session id, stamped by the browser.",
    ),
    authorization: Optional[str] = Header(None),
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
    # Die beiden Produktwerkzeuge verlangen einen gueltigen Grant. Die
    # uebrigen Lesewerkzeuge NICHT — nicht weil das richtig waere,
    # sondern weil der Wanderlaut-Browser sie heute ohne ruft und eine
    # harte Pflicht ihn sofort braeche. Neue Faehigkeit von Tag eins
    # geschuetzt, alte Aufrufer stufenweise; der Rest steht als Befund
    # im Protokoll (siehe _log_auth_presence).
    if tool_name in PRODUCT_TOOL_NAMES:
        try:
            await exchange_and_verify(authorization, "mint")
        except Exception as exc:
            logger.warning(
                "Produktwerkzeug %s ohne gueltigen Grant abgewiesen: %s",
                tool_name, type(exc).__name__,
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "realtime_grant_required",
                    "tool": tool_name,
                    "hint": (
                        "Die Produktwerkzeuge laufen ueber einen "
                        "serverseitigen Pfad, der ein Bearer-Grant "
                        "haelt. Der Browser ruft sie nicht direkt."
                    ),
                },
            )

    # Altwerkzeuge: immer messen, nur bei geschaltetem Modus abweisen.
    # Die Messung steht VOR der Namenspruefung nicht — ein unbekanntes
    # Werkzeug ist kein Aufrufer, den ich zaehlen will.
    if tool_name in READ_TOOL_NAMES and tool_name not in PRODUCT_TOOL_NAMES:
        hat_grant = _log_auth_presence(tool_name, authorization)
        if _legacy_auth_modus() == "enforce":
            fehler = None
            if not hat_grant:
                fehler = "kein Bearer"
            else:
                try:
                    await exchange_and_verify(authorization, "mint")
                except Exception as exc:
                    fehler = type(exc).__name__
            if fehler:
                logger.warning(
                    "Altwerkzeug %s abgewiesen (%s) — Modus=enforce",
                    tool_name, fehler,
                )
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "realtime_grant_required",
                        "tool": tool_name,
                        "hint": (
                            "Dieser Host verlangt seit der Umstellung "
                            "auch fuer die Lesewerkzeuge ein Bearer-"
                            "Grant. Der Aufrufer muss denselben Grant "
                            "mitsenden, mit dem er die Sitzung geoeffnet "
                            "hat."
                        ),
                    },
                )

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
    # Auditdaten fahren im TRANSPORT-Umschlag mit, nicht im Domain-
    # Ergebnis (Tschepp-Codex2s Korrektur an meinem ersten Vorschlag):
    # Im Ergebnis wuerde der Details-Validator sie als
    # `invalid_tool_response` abweisen, waehrend die Suchvalidatoren sie
    # durchliessen — und dann saehe und BEZAHLTE das Modell meine
    # Diagnose. Der BFF schaelt den Umschlag ohnehin ab; er protokolliert
    # das Feld und gibt dem Browser nur `result`.
    diagnose: dict = {}
    result: Any
    try:
        if tool_name == "knowledge_query":
            result = await _tool_knowledge_query(args)
        elif tool_name == "pois_near":
            result = await _tool_pois_near(args)
        elif tool_name == "narration_near":
            result = await _tool_narration_near(args)
        elif tool_name == "osm_nearby":
            result = await _tool_osm_nearby(args)
        elif tool_name == "agent_status":
            result = await _tool_agent_status(args, authorization)
        elif tool_name == "find_products":
            result = await _tool_product_search(args, x_session_id, False, diagnose)
        elif tool_name == "refine_search":
            result = await _tool_product_search(args, x_session_id, True, diagnose)
        elif tool_name == "product_details":
            result = await _tool_product_details(args, x_session_id, diagnose)
        elif tool_name == "resolve_agent_name":
            result = await _tool_resolve_agent_name(args, authorization)
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
    umschlag = {
        "call_id": body.call_id,
        "tool": tool_name,
        "ok": True,
        "result": result,
        "elapsed_ms": elapsed_ms,
    }
    # Nur wenn tatsaechlich etwas verworfen wurde. Ein immer
    # vorhandenes leeres Feld erzeugt Rauschen im Protokoll und
    # verleitet dazu, es zu ignorieren.
    if diagnose:
        umschlag["diagnostics"] = diagnose
    return umschlag


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
        "X-API-KEY": os.getenv("GUIDE_API_KEY") or storage_api_key(),
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


async def _tool_osm_nearby(args: dict) -> dict:
    """Wraps ArTrack-API's OSM compact-nearby lookup.

    Unlike ``pois_near`` (track-bound, returns {count: 0} off-track),
    OSM nearby works EVERYWHERE on Earth — Overpass-style lookups
    against the global OSM corpus. Right tool when the Realtime guide
    needs to talk about the user's actual surroundings instead of
    track-curated POIs (GuideDevBot2 use case 2026-06-27).

    We hit the ``/osm/nearby/compact`` variant — pre-formatted
    ``Name (category, Xm) | …`` text, low token count, voice-friendly.
    The model can paraphrase it directly without parsing a JSON list.

    ArTrack-API uses ``lng`` (not ``lon``) for longitude — same
    translation as _tool_pois_near.
    """
    lat = float(args["lat"])
    lng = float(args.get("lng") or args.get("lon"))
    # 500m is the typical talking-distance radius (user walks past,
    # mentions what's around). The compact endpoint caps at 20 hits
    # internally so a slightly larger radius doesn't blow up tokens.
    radius_m = float(args.get("radius_m", 500))
    params = {"lat": lat, "lng": lng, "radius_m": radius_m}
    url = f"{_artrack_api_base()}/osm/nearby/compact"
    # Timeout posture: 5.0s (PR #90, 2026-06-28). Cold Overpass queries
    # after the 1h ArTrack cache TTL can briefly exceed 2.5s
    # (GuideDevBot2 observed 2530ms at 46.62/14.31). Tightening to 2.5s
    # caused the route to raise -> ok:false -> the model silently fell
    # back to LLM general knowledge — sounds smooth but the
    # 'real-surroundings' value is lost. Five seconds occasionally
    # produces a hearable hang for the user, but cached queries return
    # in ~17ms so the worst-case is rare per user-area. Honest hang
    # with real data beats smooth fallback to hallucinated knowledge.
    t_start = time.monotonic()
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    osm_elapsed_ms = int((time.monotonic() - t_start) * 1000)
    if osm_elapsed_ms > 2500:
        logger.warning(
            "osm_nearby slow cold-Overpass: %dms at lat=%.5f lng=%.5f "
            "radius_m=%.0f cached=%s — consider ArTrack cache-prewarm "
            "or longer TTL for popular grids",
            osm_elapsed_ms, lat, lng, radius_m, data.get("cached"),
        )
    # Compact endpoint returns {text, count, cached}. Pass it through
    # mostly verbatim — the text is the voice-payload, the rest is
    # diagnostic.
    return {
        "text": data.get("text") or "",
        "count": int(data.get("count") or 0),
        "cached": bool(data.get("cached") or False),
    }


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
