"""agentos_jwt — shared JWT verification for AgentOS services (#2987).

One file, no state outside `state_dir`. Can be installed as a package or
vendored as a single file (`agentos_jwt.py`); the header below carries the
version so a vendored copy stays identifiable.

    from agentos_jwt import Verifier, TokenRejected, RevocationUnknown

    verifier = Verifier(
        issuer="auth-api.arkturian.com",
        accept={"agent": None, "service": None, "user": None},
        state_dir=Path("/var/lib/<dienst>/agentos_jwt"),
    )
    claims = verifier.verify(token)          # sync
    claims = await verifier.averify(token)   # async

Rules (fail-closed):
  * RS256 only; the header `alg` is ignored. Required: exp, iat, sub, iss.
  * Keys by `kid` from the JWKS (TTL 300 s). An unknown kid triggers one
    reload, at most every 10 s. A token without kid is rejected. When the
    JWKS cannot be loaded, the last keys stay usable for 1 h, then 503.
  * `type` must be in `accept`; `accept[type] is None` means the token must
    not carry `aud`, otherwise `aud` must hit the given set.
  * For `agent` and `service` tokens: the revocation state from
    `/.well-known/revoked-agent-jtis` (jti list, `min_agent_iat`,
    `owner_min_iat`) is polled every 60 s and kept on disk. A failed poll
    never clears it. With no state ever loaded -> RevocationUnknown (503).
"""
from __future__ import annotations

__version__ = "1.0.1"

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional

import jwt
from jwt.algorithms import RSAAlgorithm

__all__ = [
    "Verifier",
    "TokenRejected",
    "RevocationUnknown",
    "KeysUnavailable",
    "REASONS",
    "TOKEN_TYPES",
    "__version__",
]

logger = logging.getLogger("agentos_jwt")

#: Token classes auth-api issues (#2975, #3004).
TOKEN_TYPES = frozenset({
    "user", "oauth_access", "delegated", "realtime_grant",
    "ephemeral_capability", "agent", "service",
})

#: Every value `TokenRejected.reason` can take.
REASONS = frozenset({
    "malformed", "bad_signature", "unknown_kid", "expired", "not_yet_valid",
    "missing_claim", "bad_issuer", "type_not_accepted", "aud_mismatch",
    "jti_revoked", "iat_before_cutoff", "owner_revoked",
})

_REVOCABLE = frozenset({"agent", "service"})
_SUPERADMIN_TYPES = frozenset({"user", "delegated"})
# Longest agent-token lifetime auth-api has ever issued is 365 d; a jti
# revoked longer ago than that cannot belong to a live token.
_JTI_KEEP_SECONDS = 366 * 86400

Fetch = Callable[[str, Optional[dict]], dict]
AFetch = Callable[[str, Optional[dict]], Awaitable[dict]]


class TokenRejected(Exception):
    """The token is not acceptable. Map to HTTP 401."""

    def __init__(self, reason: str, detail: str = ""):
        assert reason in REASONS, reason
        self.reason = reason
        super().__init__(f"{reason}{': ' + detail if detail else ''}")


class RevocationUnknown(Exception):
    """No revocation state was ever loaded. Map to HTTP 503."""


class KeysUnavailable(Exception):
    """No usable signing keys (never loaded, or older than the grace). 503."""


def _iso_to_epoch(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    if not isinstance(value, str):
        raise TypeError("timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class Verifier:
    """Verifies auth-api tokens. Thread-safe for `verify`, task-safe for
    `averify`. Create one per process and reuse it."""

    def __init__(
        self,
        *,
        issuer: str,
        accept: Mapping[str, Optional[Iterable[str]]],
        state_dir: Optional[os.PathLike] = None,
        ephemeral_state: bool = False,
        base_url: Optional[str] = None,
        jwks_url: Optional[str] = None,
        revocation_url: Optional[str] = None,
        untyped_as: Optional[str] = None,
        leeway: int = 30,
        jwks_ttl: int = 300,
        jwks_min_refresh: int = 10,
        jwks_grace: int = 3600,
        revocation_interval: int = 60,
        http_timeout: float = 5.0,
        fetch: Optional[Fetch] = None,
        afetch: Optional[AFetch] = None,
        clock: Callable[[], float] = time.time,
    ):
        if not issuer:
            raise ValueError("issuer is required")
        unknown = set(accept) - TOKEN_TYPES
        if unknown:
            raise ValueError(f"unknown token types in accept: {sorted(unknown)}")
        if not accept:
            raise ValueError("accept must name at least one token type")
        if untyped_as is not None and untyped_as not in accept:
            raise ValueError("untyped_as must be one of the accepted types")
        if jwks_ttl > 300:
            raise ValueError("jwks_ttl must be at most 300 s (#2987)")

        self.issuer = issuer
        self.accept: dict[str, Optional[frozenset]] = {
            t: (None if a is None else frozenset([a] if isinstance(a, str) else a))
            for t, a in accept.items()
        }
        base = (base_url or f"https://{issuer}").rstrip("/")
        self.jwks_url = jwks_url or f"{base}/api/v1/auth/.well-known/jwks.json"
        self.revocation_url = revocation_url or f"{base}/api/v1/auth/.well-known/revoked-agent-jtis"
        self.untyped_as = untyped_as
        self.leeway = leeway
        self.jwks_ttl = jwks_ttl
        self.jwks_min_refresh = jwks_min_refresh
        self.jwks_grace = jwks_grace
        self.revocation_interval = revocation_interval
        self.http_timeout = http_timeout
        self._fetch = fetch
        self._afetch = afetch
        self._clock = clock

        self._keys: dict[str, Any] = {}
        self._keys_loaded_at: Optional[float] = None
        self._keys_attempt_at: Optional[float] = None

        self._needs_revocation = bool(_REVOCABLE & set(self.accept))
        if self._needs_revocation and state_dir is None and not ephemeral_state:
            # Without a state file a restart during an auth-api outage would
            # answer 503 for every agent call; in production that must be a
            # deliberate choice, not a forgotten parameter.
            raise ValueError("state_dir is required when agent/service tokens are accepted "
                             "(ephemeral_state=True only for tests)")
        self._rev_path = Path(state_dir) / "revocation.json" if state_dir else None
        self._rev_loaded = False
        self._rev_attempt_at: Optional[float] = None
        self._since: Optional[str] = None
        self._jtis: dict[str, int] = {}
        self._min_iat: Optional[int] = None
        self._owner_min_iat: dict[str, int] = {}
        self._load_state()

        self._tlock = threading.Lock()
        self._alock: Optional[asyncio.Lock] = None

    # ── public API ──────────────────────────────────────────────────────

    def verify(self, token: str) -> dict:
        header = self._header(token)
        with self._tlock:
            self._sync_keys(header.get("kid"))
            if self._wants_revocation(token):
                self._sync_revocation()
        return self._check(token, header)

    async def averify(self, token: str) -> dict:
        header = self._header(token)
        if self._alock is None:
            self._alock = asyncio.Lock()
        async with self._alock:
            await self._async_keys(header.get("kid"))
            if self._wants_revocation(token):
                await self._async_revocation()
        return self._check(token, header)

    def _wants_revocation(self, token: str) -> bool:
        # Only agent/service tokens depend on the revocation state, so a
        # user request never waits for (or fails on) the poll. The unverified
        # type is just a hint; `_check` decides on the verified claims and
        # refuses a revocable type when no state is loaded.
        if not self._needs_revocation:
            return False
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError:
            return False
        if not isinstance(payload, dict):
            return False
        typ = payload.get("type")
        if typ is None:
            typ = self.untyped_as
        return isinstance(typ, str) and typ in _REVOCABLE

    def fastapi_dependency(self):
        """FastAPI dependency: returns the claims of a valid Bearer token.

        TokenRejected -> 401 {"error": "invalid_token", "reason": ...},
        RevocationUnknown / KeysUnavailable -> 503.
        """
        from fastapi import HTTPException, Request

        async def dependency(request: Request) -> dict:
            auth = request.headers.get("authorization", "")
            scheme, _, token = auth.partition(" ")
            if scheme.lower() != "bearer" or not token.strip():
                raise HTTPException(
                    status_code=401,
                    detail={"error": "invalid_token", "reason": "missing_token"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            try:
                return await self.averify(token.strip())
            except TokenRejected as exc:
                raise HTTPException(
                    status_code=401,
                    detail={"error": "invalid_token", "reason": exc.reason},
                    headers={"WWW-Authenticate": f'Bearer error="invalid_token"'},
                ) from None
            except (RevocationUnknown, KeysUnavailable) as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "auth_unavailable", "reason": type(exc).__name__},
                ) from None

        # Module uses postponed annotations; FastAPI needs the real class.
        dependency.__annotations__ = {"request": Request, "return": dict}
        return dependency

    # ── checks ─────────────────────────────────────────────────────────

    @staticmethod
    def _header(token: str) -> dict:
        try:
            return jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise TokenRejected("malformed") from None

    def _check(self, token: str, header: dict) -> dict:
        kid = header.get("kid")
        if not kid or kid not in self._keys:
            raise TokenRejected("unknown_kid")
        try:
            claims = jwt.decode(
                token,
                self._keys[kid],
                algorithms=["RS256"],
                issuer=self.issuer,
                leeway=self.leeway,
                options={"require": ["exp", "iat", "sub", "iss"], "verify_aud": False},
            )
        except jwt.ExpiredSignatureError:
            raise TokenRejected("expired") from None
        except jwt.ImmatureSignatureError:
            raise TokenRejected("not_yet_valid") from None
        except jwt.MissingRequiredClaimError as exc:
            raise TokenRejected("missing_claim", exc.claim) from None
        except jwt.InvalidIssuerError:
            raise TokenRejected("bad_issuer") from None
        except jwt.InvalidSignatureError:
            raise TokenRejected("bad_signature") from None
        except jwt.InvalidAlgorithmError:
            raise TokenRejected("bad_signature", "alg") from None
        except jwt.PyJWTError:
            raise TokenRejected("malformed") from None

        # Shape guards: a signed token from auth-api always has these types,
        # anything else is malformed (never a TypeError -> 500).
        typ = claims.get("type")
        if typ is None:
            typ = self.untyped_as
        if typ is not None and not isinstance(typ, str):
            self._reject(claims, "malformed")
        iat = claims.get("iat")
        if isinstance(iat, bool) or not isinstance(iat, (int, float)):
            self._reject(claims, "malformed")
        aud = claims.get("aud")
        if aud is None:
            auds = []
        elif isinstance(aud, str):
            auds = [aud]
        elif isinstance(aud, list) and all(isinstance(a, str) for a in aud):
            auds = aud
        else:
            self._reject(claims, "malformed")

        if typ not in self.accept:
            self._reject(claims, "type_not_accepted")
        allowed = self.accept[typ]
        if allowed is None:
            if auds:
                self._reject(claims, "aud_mismatch")
        elif not (set(auds) & allowed):
            self._reject(claims, "aud_mismatch")

        if typ in _REVOCABLE:
            if not self._rev_loaded:
                raise RevocationUnknown("no revocation state yet")
            jti = claims.get("jti")
            if not isinstance(jti, str) or not jti:
                self._reject(claims, "missing_claim")
            owner = claims.get("owner_id")
            if owner is not None and not isinstance(owner, str):
                self._reject(claims, "malformed")
            if jti in self._jtis:
                self._reject(claims, "jti_revoked")
            if self._min_iat is not None and iat < self._min_iat:
                self._reject(claims, "iat_before_cutoff")
            if owner and owner in self._owner_min_iat and iat < self._owner_min_iat[owner]:
                self._reject(claims, "owner_revoked")

        out = dict(claims)
        out["type"] = typ
        # Superadmin power only ever rides on a first-party login token or a
        # token exchanged from one; an OAuth client token (possibly from a
        # dynamically registered client) never carries it, whatever it says.
        out["is_superadmin"] = typ in _SUPERADMIN_TYPES and claims.get("is_superadmin") is True
        return out

    @staticmethod
    def _reject(claims: dict, reason: str):
        # Never the token itself; only identifiers of a signature-checked token.
        logger.info("token rejected reason=%s sub=%s jti=%s", reason, claims.get("sub"), claims.get("jti"))
        raise TokenRejected(reason)

    # ── JWKS ───────────────────────────────────────────────────────────

    def _keys_plan(self, kid: Optional[str]) -> bool:
        """True when a JWKS fetch should happen now."""
        now = self._clock()
        if self._keys_loaded_at is None:
            return self._keys_attempt_at is None or now - self._keys_attempt_at >= self.jwks_min_refresh
        stale = now - self._keys_loaded_at >= self.jwks_ttl
        missing = bool(kid) and kid not in self._keys
        recent_try = self._keys_attempt_at is not None and now - self._keys_attempt_at < self.jwks_min_refresh
        if missing and not recent_try:
            return True
        return stale and not recent_try

    def _keys_apply(self, doc: Optional[dict]) -> None:
        now = self._clock()
        self._keys_attempt_at = now
        if doc is not None:
            keys = {}
            for jwk in doc.get("keys", []):
                if jwk.get("kty") != "RSA" or not jwk.get("kid"):
                    continue
                if jwk.get("alg") not in (None, "RS256"):
                    continue
                if jwk.get("use") not in (None, "sig"):
                    continue
                try:
                    keys[jwk["kid"]] = RSAAlgorithm.from_jwk(json.dumps(jwk))
                except Exception:  # noqa: BLE001 — a broken key is skipped
                    logger.warning("jwks: unreadable key kid=%s skipped", jwk.get("kid"))
            if keys:
                self._keys = keys
                self._keys_loaded_at = now
                return
            logger.warning("jwks: no usable RS256 key in response")
        if self._keys_loaded_at is None:
            raise KeysUnavailable("JWKS never loaded")
        if now - self._keys_loaded_at > self.jwks_grace:
            raise KeysUnavailable("JWKS older than grace")

    def _sync_keys(self, kid):
        if not self._keys_plan(kid):
            return self._keys_guard()
        try:
            doc = (self._fetch or self._default_fetch)(self.jwks_url, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("jwks fetch failed: %s", exc)
            doc = None
        self._keys_apply(doc)

    async def _async_keys(self, kid):
        if not self._keys_plan(kid):
            return self._keys_guard()
        try:
            if self._afetch is not None:
                doc = await self._afetch(self.jwks_url, None)
            elif self._fetch is not None:
                doc = self._fetch(self.jwks_url, None)
            else:
                doc = await self._default_afetch(self.jwks_url, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("jwks fetch failed: %s", exc)
            doc = None
        self._keys_apply(doc)

    def _keys_guard(self):
        now = self._clock()
        if self._keys_loaded_at is None:
            raise KeysUnavailable("JWKS never loaded")
        if now - self._keys_loaded_at > self.jwks_grace:
            raise KeysUnavailable("JWKS older than grace")

    # ── revocation ─────────────────────────────────────────────────────

    def _rev_due(self) -> bool:
        return self._rev_attempt_at is None or self._clock() - self._rev_attempt_at >= self.revocation_interval

    def _rev_apply(self, doc: Optional[dict]) -> None:
        self._rev_attempt_at = self._clock()
        if doc is not None:
            try:
                # Build the complete new state first; swap it in only when
                # the whole response was readable.
                jtis = dict(self._jtis)
                for row in doc["revoked_jtis"]:
                    jti = row["jti"]
                    if not isinstance(jti, str):
                        raise TypeError("jti")
                    jtis[jti] = _iso_to_epoch(row.get("revoked_at")) or int(self._clock())
                min_iat = _iso_to_epoch(doc.get("min_agent_iat"))
                owners = dict(self._owner_min_iat)
                for k, v in (doc.get("owner_min_iat") or {}).items():
                    v = _iso_to_epoch(v)
                    if v is not None and v > owners.get(str(k), 0):
                        owners[str(k)] = v
                as_of = doc["as_of"]
                if not isinstance(as_of, str):
                    raise TypeError("as_of")
                horizon = int(self._clock()) - _JTI_KEEP_SECONDS
                jtis = {j: t for j, t in jtis.items() if t >= horizon}
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                logger.warning("revocation: unreadable response (%s), keeping last state", exc)
            else:
                self._jtis = jtis
                self._min_iat = min_iat  # current value, lowering included
                self._owner_min_iat = owners
                self._since = as_of
                self._rev_loaded = True
                self._save_state()
        if not self._rev_loaded:
            raise RevocationUnknown("no revocation state yet")

    def _sync_revocation(self):
        if not self._rev_due():
            return self._rev_guard()
        params = {"since": self._since} if self._since else None
        try:
            doc = (self._fetch or self._default_fetch)(self.revocation_url, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("revocation fetch failed: %s", exc)
            doc = None
        self._rev_apply(doc)

    async def _async_revocation(self):
        if not self._rev_due():
            return self._rev_guard()
        params = {"since": self._since} if self._since else None
        try:
            if self._afetch is not None:
                doc = await self._afetch(self.revocation_url, params)
            elif self._fetch is not None:
                doc = self._fetch(self.revocation_url, params)
            else:
                doc = await self._default_afetch(self.revocation_url, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("revocation fetch failed: %s", exc)
            doc = None
        self._rev_apply(doc)

    def _rev_guard(self):
        if not self._rev_loaded:
            raise RevocationUnknown("no revocation state yet")

    # ── persistence ────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not self._rev_path or not self._rev_path.exists():
            return
        try:
            fd = os.open(self._rev_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd) as fh:
                data = json.load(fh)
            jtis = {str(k): int(v) for k, v in data["jtis"].items()}
            min_iat = data.get("min_iat")
            if min_iat is not None:
                min_iat = int(min_iat)
            owners = {str(k): int(v) for k, v in data.get("owner_min_iat", {}).items()}
            since = data.get("since")
            if since is not None and not isinstance(since, str):
                raise TypeError("since")
            self._jtis, self._min_iat, self._owner_min_iat, self._since = jtis, min_iat, owners, since
            self._rev_loaded = True
        except Exception as exc:  # noqa: BLE001 — unreadable file = no state
            logger.warning("revocation state %s unreadable: %s", self._rev_path, exc)

    def _save_state(self) -> None:
        if not self._rev_path:
            return
        tmp = None
        try:
            self._rev_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Unique temp file per write (O_EXCL, 0600): several workers of
            # one service can save concurrently without clobbering each
            # other's half-written file; os.replace makes the swap atomic.
            fd, tmp = tempfile.mkstemp(dir=self._rev_path.parent, prefix=".revocation.", suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump({
                    "version": __version__,
                    "since": self._since,
                    "jtis": self._jtis,
                    "min_iat": self._min_iat,
                    "owner_min_iat": self._owner_min_iat,
                }, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._rev_path)
            tmp = None
        except OSError as exc:
            logger.warning("revocation state not saved: %s", exc)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ── default HTTP ───────────────────────────────────────────────────

    def _default_fetch(self, url: str, params: Optional[dict]) -> dict:
        import httpx

        r = httpx.get(url, params=params, timeout=self.http_timeout)
        r.raise_for_status()
        return r.json()

    async def _default_afetch(self, url: str, params: Optional[dict]) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
