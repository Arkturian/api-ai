"""
Realtime Grant Verifier
=======================

Server-to-server grant exchange + local JWKS verify for CloudV2's
Voice-Companion mint path (Content-Post #1215, frozen v1 Auth-Contract).

Flow per protected ``/ai/realtime*`` request:

  1. Browser sends the normal ``User-JWT`` in ``Authorization: Bearer``.
  2. api-ai forwards that header to auth-api's grant endpoint with
     a service-shared ``X-API-KEY`` header. The user JWT NEVER lands
     in the JSON body and is NEVER logged.
  3. auth-api returns a capability-JWT (RS256, kid-pinned, 90s exp)
     describing the user's permitted ``profile_id``, ``scopes`` and
     ``limits``.
  4. api-ai verifies the grant locally against the pinned JWKS,
     enforces:
       * iss exact = ``auth-api.arkturian.com`` (SCHEMELESS)
       * alg = RS256 (allowlist; rejects ``none``/HS/EdDSA/ES256)
       * kid is present and in the cached JWKS
       * aud exact = ``ai-realtime:<host_profile_id>``
       * grant.profile_id == host REALTIME_PROFILE_ID
       * required ``scope`` ⊆ grant.scopes
       * iat/nbf/exp standard window
  5. On any failure: HTTP 403 with a generic error code; details land
     only in the redacted audit logger.

Design rules (Codex final, Post #1215):

  * NO positive cache. Local JWKS verify is sub-millisecond, and a
    cache would push admin-revoke latency to 60s. Every protected
    request makes one server-to-server exchange.
  * NO negative cache.
  * NO raw User-JWT in JSON bodies or logs.
  * JWKS in-memory cache only; re-fetch on unknown kid (rotation
    overlap-friendly).
  * fail-closed on any error. auth-api unreachable -> 403
    ``auth_down`` (locally generated, not an auth-api response).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────


AUTH_ISSUER = "auth-api.arkturian.com"  # schemeless — AuthAPI's frozen iss
JWKS_URL = "https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json"
GRANT_URL = "https://auth-api.arkturian.com/api/v1/auth/realtime-grant"
ALG_ALLOWLIST = ("RS256",)
AUD_PREFIX = "ai-realtime:"
GRANT_TIMEOUT_SEC = 5.0

REALTIME_PROFILE_ID_ENV = "REALTIME_PROFILE_ID"
SERVICE_KEY_ENV = "REALTIME_GRANT_SERVICE_KEY"
SERVICE_KEY_HEADER = "X-API-KEY"


# ── Exceptions ────────────────────────────────────────────────────────


class GrantError(Exception):
    """Base for all grant-flow failures.

    ``error_code`` becomes the public ``{error: <code>}`` response;
    ``audit_detail`` is the redacted line for journalctl. Never put
    PII or the raw User-JWT in either.
    """

    def __init__(self, error_code: str, audit_detail: str = "", status_code: int = 403):
        super().__init__(error_code)
        self.error_code = error_code
        self.audit_detail = audit_detail
        self.status_code = status_code


class GrantDenied(GrantError):
    """auth-api returned approved=false. Map the reason_code to public."""


class GrantUnverifiable(GrantError):
    """Grant JWT failed local cryptographic verify."""


class GrantWrongProfile(GrantError):
    """Grant's profile_id != this host's REALTIME_PROFILE_ID."""

    def __init__(self):
        super().__init__("realtime_wrong_profile", "host/grant profile mismatch")


class GrantScopeMissing(GrantError):
    """Endpoint required a scope not in the grant's scopes list."""


class AuthDown(GrantError):
    """auth-api unreachable / 5xx. AiApi local fail-closed."""

    def __init__(self, hint: str = ""):
        super().__init__("auth_down", hint, status_code=503)


class ServiceKeyMissing(GrantError):
    """REALTIME_GRANT_SERVICE_KEY env unset on this host."""

    def __init__(self):
        super().__init__(
            "realtime_grant_service_unconfigured",
            "REALTIME_GRANT_SERVICE_KEY env not set",
            status_code=503,
        )


class ProfileMisconfigured(GrantError):
    """REALTIME_PROFILE_ID env unset on this host."""

    def __init__(self):
        super().__init__(
            "realtime_profile_misconfigured",
            f"{REALTIME_PROFILE_ID_ENV} env not set",
            status_code=503,
        )


# ── Verified grant payload ────────────────────────────────────────────


@dataclass(frozen=True)
class VerifiedGrant:
    """Per-request grant view used by the FastAPI dependency."""

    sub: str
    tenant_id: str
    profile_id: str
    scopes: tuple
    jti: str
    grant_id: str
    iat: int
    exp: int
    max_parallel_sessions: int
    daily_budget_eur: float

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


# ── JWKS client (in-memory cache, kid-pinned, re-fetch on miss) ───────


_jwks_client: Optional[PyJWKClient] = None


def _jwks() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        # PyJWKClient handles in-memory caching + re-fetch on unknown kid
        # internally (cache_keys=True). Overlap-rotation friendly.
        _jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, lifespan=300)
    return _jwks_client


# ── Server-to-server grant exchange ───────────────────────────────────


def _unsafe_peek_jwt_claims(authorization_header: str) -> str:
    """Decode-without-verify of the forwarded Bearer to extract diagnostic
    claims (sub/kid/exp/iss) for redacted logging on deny paths.

    NEVER trust these values for authorization. The whole purpose of the
    server-to-server grant exchange is that auth-api is the only thing
    that *verifies* the JWT. This helper exists only so a 401 from
    auth-api carries enough fingerprint for cross-correlating with
    auth-api's own audit log.

    Returns a short, log-safe string like
    ``sub=74e8e363 kid=auth-primary exp_in=84s iss=auth-api.arkturian.com``.
    Returns an empty string on any decode failure — never raises.
    """
    try:
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            return ""
        token = authorization_header.split(None, 1)[1].strip()
        header = jwt.get_unverified_header(token)
        claims = jwt.decode(token, options={"verify_signature": False})
        sub = str(claims.get("sub") or "")
        iss = str(claims.get("iss") or "")
        kid = str(header.get("kid") or "")
        exp = claims.get("exp")
        now = int(time.time())
        if isinstance(exp, (int, float)):
            exp_in = int(exp) - now
            exp_str = f"exp_in={exp_in}s"
        else:
            exp_str = "exp=?"
        return (
            f"sub={sub[:8]} kid={kid} {exp_str} iss={iss}"
        ).strip()
    except Exception:
        return ""


async def _exchange_user_jwt_for_grant(authorization_header: str) -> dict:
    """POST the user JWT (still as Bearer, never as body) to auth-api
    plus our shared service key. Return the JSON envelope verbatim."""
    service_key = os.environ.get(SERVICE_KEY_ENV)
    if not service_key:
        raise ServiceKeyMissing()
    headers = {
        "Authorization": authorization_header,
        SERVICE_KEY_HEADER: service_key,
    }
    try:
        async with httpx.AsyncClient(timeout=GRANT_TIMEOUT_SEC) as client:
            resp = await client.post(GRANT_URL, headers=headers)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
        raise AuthDown(f"grant exchange transport: {type(exc).__name__}") from exc

    if resp.status_code == 401:
        # auth-api rejected service or user auth. Bubble out as deny —
        # but log enough forensic detail (claim fingerprint + body) to
        # cross-correlate with auth-api's own audit. Body is bounded to
        # 200 chars to keep journalctl rotation cheap; the JWT itself
        # is NEVER logged, only sub/kid/iss/exp from the (unverified)
        # header — same level of detail Auth gets from their own decode.
        peek = _unsafe_peek_jwt_claims(authorization_header)
        try:
            body_preview = (resp.text or "")[:200].replace("\n", " ")
        except Exception:
            body_preview = ""
        key_suffix = service_key[-6:] if service_key else ""
        raise GrantDenied(
            "realtime_grant_unauthorized",
            f"auth-api 401 body={body_preview!r} jwt[{peek}] key_suffix=...{key_suffix}",
            status_code=401,
        )
    if resp.status_code >= 500:
        raise AuthDown(f"auth-api {resp.status_code}")
    if resp.status_code != 200:
        raise AuthDown(f"auth-api unexpected {resp.status_code}")

    try:
        envelope = resp.json()
    except Exception as exc:
        raise AuthDown(f"auth-api body parse: {type(exc).__name__}") from exc
    return envelope


# ── Local JWKS verify of the capability JWT ───────────────────────────


def _verify_grant_jwt(grant_token: str, host_profile_id: str) -> VerifiedGrant:
    """Verify signature/claims of the capability JWT against the JWKS.

    Enforces: alg=RS256, iss exact, aud exact, kid present, exp/nbf/iat
    standard window, and matches the host's REALTIME_PROFILE_ID.
    """
    try:
        signing_key = _jwks().get_signing_key_from_jwt(grant_token)
    except Exception as exc:
        raise GrantUnverifiable(
            "realtime_grant_jwks_lookup_failed",
            f"{type(exc).__name__}",
        ) from exc

    expected_aud = f"{AUD_PREFIX}{host_profile_id}"
    try:
        claims = jwt.decode(
            grant_token,
            signing_key.key,
            algorithms=list(ALG_ALLOWLIST),
            audience=expected_aud,
            issuer=AUTH_ISSUER,
            options={"require": ["iss", "aud", "sub", "exp", "iat"]},
        )
    except jwt.InvalidAudienceError as exc:
        # aud mismatch is structurally the same as wrong-profile here.
        raise GrantWrongProfile() from exc
    except jwt.ExpiredSignatureError as exc:
        raise GrantUnverifiable("realtime_grant_expired", "exp passed") from exc
    except jwt.InvalidTokenError as exc:
        raise GrantUnverifiable(
            "realtime_grant_invalid",
            f"{type(exc).__name__}",
        ) from exc

    # Belt-and-suspenders: aud might pass via list, but we expect exact.
    aud_claim = claims.get("aud")
    if aud_claim != expected_aud and aud_claim != [expected_aud]:
        raise GrantWrongProfile()

    grant_profile = claims.get("profile_id")
    if grant_profile != host_profile_id:
        raise GrantWrongProfile()

    scopes = tuple(claims.get("scopes") or [])
    limits = claims.get("limits") or {}
    try:
        max_parallel = int(limits.get("max_parallel_sessions"))
        daily_budget = float(limits.get("daily_budget_eur"))
    except (TypeError, ValueError) as exc:
        raise GrantUnverifiable(
            "realtime_grant_limits_invalid",
            "limits claim missing/malformed",
        ) from exc

    return VerifiedGrant(
        sub=str(claims["sub"]),
        tenant_id=str(claims.get("tenant_id") or ""),
        profile_id=grant_profile,
        scopes=scopes,
        jti=str(claims.get("jti") or ""),
        grant_id=str(claims.get("grant_id") or ""),
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
        max_parallel_sessions=max_parallel,
        daily_budget_eur=daily_budget,
    )


# ── Public entry point used by the FastAPI dependency ────────────────


async def exchange_and_verify(
    authorization_header: str,
    required_scope: str,
) -> VerifiedGrant:
    """Full per-request flow: exchange the user JWT for a grant, verify
    it locally, enforce host profile + scope. Raises GrantError subclass
    on any failure; the route caller maps that to HTTPException.

    The audit logger gets a single redacted line per call so a flapping
    auth-api or a bad client is visible without leaking PII.
    """
    host_profile = os.environ.get(REALTIME_PROFILE_ID_ENV)
    if not host_profile:
        raise ProfileMisconfigured()

    envelope = await _exchange_user_jwt_for_grant(authorization_header)
    if not envelope.get("approved"):
        reason = envelope.get("reason_code") or "denied"
        # Closed enum from AuthAPI: not_enabled | limits_not_configured |
        # no_profile_key | profile_not_resolved | wrong_profile
        raise GrantDenied(f"realtime_{reason}", f"reason={reason}")

    token = envelope.get("grant_token")
    if not token:
        raise AuthDown("approved envelope without grant_token")

    grant = _verify_grant_jwt(token, host_profile)

    if not grant.has_scope(required_scope):
        raise GrantScopeMissing(
            "realtime_scope_missing",
            f"required={required_scope}",
        )

    # Standard redacted audit line — no PII, no raw token, no key bytes.
    logger.info(
        "realtime_grant ok scope=%s profile=%s tenant=%s jti=%s "
        "exp_in=%ds limits=%d/%.2f",
        required_scope,
        host_profile,
        grant.tenant_id,
        grant.jti[:8],
        max(0, grant.exp - int(time.time())),
        grant.max_parallel_sessions,
        grant.daily_budget_eur,
    )
    return grant


def host_profile_id() -> Optional[str]:
    """Return the host's pinned REALTIME_PROFILE_ID or None."""
    return os.environ.get(REALTIME_PROFILE_ID_ENV)


def service_key_configured() -> bool:
    return bool(os.environ.get(SERVICE_KEY_ENV))
