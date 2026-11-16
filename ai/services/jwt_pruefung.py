"""JWT-Pruefung fuer api-ai ueber agentos_jwt v1.0.1 (#3155, #2972/#2970).

Vorher: eigene Pruefung mit PyJWKClient, ohne Widerruf und ohne Typ-
Allowlist. Jedes von auth-api signierte Token galt als „angemeldet“, auch
ein Realtime-Grant oder ein OAuth-Token fremder Clients.

Zwei Pruefer, beide einmal je Prozess:

- `allgemein()`: fuer alles ausser der Stimme. Nimmt `agent`, `service`,
  `user` (Portal) und `delegated` mit aud `ai-api` (Token-Tausch des
  MCP-Gateways, #3004) an. Widerruf ueber state_dir.
- `grant(profile_ids)`: nur fuer die Stimme. Nimmt ausschliesslich
  `realtime_grant` mit aud `ai-realtime:<profil>` an.

Aussteller und Adressen folgen denselben Umgebungsvariablen wie bisher
beim Realtime-Grant (REALTIME_AUTH_ISSUER, REALTIME_AUTH_JWKS_URL), damit
eine eigene auth-api-Instanz alles zusammen umstellt. Die Widerrufsliste
liegt neben der JWKS (oder AGENTOS_JWT_REVOCATION_URL).

Protokolliert wird nur reason, sub und jti, nie das Token.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException

from ai.agentos_jwt import KeysUnavailable, RevocationUnknown, TokenRejected, Verifier

logger = logging.getLogger(__name__)

AUD_DELEGATED = "ai-api"
AUD_PREFIX_GRANT = "ai-realtime:"

_lock = threading.Lock()
_allgemein: Optional[Verifier] = None
_grants: dict = {}


def _issuer() -> str:
    return os.getenv("REALTIME_AUTH_ISSUER", "auth-api.arkturian.com")


def _jwks_url() -> str:
    return os.getenv("REALTIME_AUTH_JWKS_URL",
                     "https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json")


def _revocation_url() -> str:
    eigen = os.getenv("AGENTOS_JWT_REVOCATION_URL")
    if eigen:
        return eigen
    return _jwks_url().rsplit("/", 1)[0] + "/revoked-agent-jtis"


def _state_dir() -> Path:
    pfad = Path(os.getenv("AGENTOS_JWT_STATE_DIR") or "/var/lib/api-ai/agentos_jwt")
    pfad.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(pfad, 0o700)
    except OSError:
        pass
    return pfad


def allgemein() -> Verifier:
    global _allgemein
    with _lock:
        if _allgemein is None:
            _allgemein = Verifier(
                issuer=_issuer(),
                accept={"agent": None, "service": None, "user": None,
                        "delegated": {AUD_DELEGATED}},
                state_dir=_state_dir(),
                jwks_url=_jwks_url(),
                revocation_url=_revocation_url(),
            )
        return _allgemein


def grant(profile_ids: Iterable[str]) -> Verifier:
    auds = frozenset(f"{AUD_PREFIX_GRANT}{p}" for p in profile_ids if p)
    with _lock:
        v = _grants.get(auds)
        if v is None:
            # realtime_grant ist nicht widerrufbar (nur agent/service), darum
            # ohne state_dir. Ohne Profil nimmt der Pruefer nichts an.
            v = Verifier(
                issuer=_issuer(),
                accept={"realtime_grant": auds or {AUD_PREFIX_GRANT + "-"}},
                jwks_url=_jwks_url(),
                revocation_url=_revocation_url(),
            )
            _grants[auds] = v
        return v


def _zuruecksetzen() -> None:
    """Nur fuer Tests."""
    global _allgemein
    with _lock:
        _allgemein = None
        _grants.clear()


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return token or None


def pruefen(authorization: Optional[str]) -> Optional[dict]:
    """Claims eines mitgeschickten Bearer-Tokens. Kein Bearer: None.
    Abgelehnt: 401. Pruefung gerade unmoeglich (keine Schluessel, keine
    Widerrufsliste): 503, damit ein Ausfall von auth-api nicht wie ein
    falsches Token aussieht."""
    token = _bearer(authorization)
    if token is None:
        return None
    try:
        pruefer = allgemein()
    except (OSError, ValueError) as e:
        # z. B. state_dir nicht anlegbar: lieber 503 als ein ungeprueftes Ja
        logger.error("jwt-Pruefer nicht baubar: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail={"error": "auth_unavailable",
                                                     "reason": type(e).__name__})
    try:
        return pruefer.verify(token)
    except TokenRejected as e:
        logger.info("jwt abgelehnt: reason=%s", e.reason)
        raise HTTPException(status_code=401, detail={"error": "invalid_token", "reason": e.reason})
    except (RevocationUnknown, KeysUnavailable) as e:
        logger.warning("jwt nicht pruefbar: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail={"error": "auth_unavailable",
                                                     "reason": type(e).__name__})


def claims_oder_none(authorization: Optional[str]) -> Optional[dict]:
    """Fuer Etiketten (wer ruft?): geprueft oder None, nie eine Ausnahme."""
    try:
        return pruefen(authorization)
    except HTTPException:
        return None
