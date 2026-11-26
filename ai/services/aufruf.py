"""Kontext des laufenden Aufrufs: Pfad, Aufrufer, geprufter Mandant.

Gesetzt von main._aufrufer_protokoll fuer jeden POST /ai/*, vor dem Aufruf.
Gelesen vom Kostenbuch (#4127) und bei Storage-Uploads (#4132).

- `aufrufer`/`quelle`: Etikett wie im Aufruferprotokoll (X-Agent-Name vor JWT-sub).
- `tenant`: NUR aus einem gepruften JWT (Signatur, Aussteller, Widerruf), nie aus
  einem Kopf. Ohne geprueftes JWT None.
"""
from __future__ import annotations

import contextvars
import re
from typing import Dict, Optional

KONTEXT: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("aufruf_kontext", default=None)

_MANDANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def mandant() -> Optional[str]:
    """Mandant aus dem geprueften JWT des laufenden Aufrufs, sonst None."""
    t = (KONTEXT.get() or {}).get("tenant")
    return t if isinstance(t, str) and _MANDANT_RE.fullmatch(t) else None


def herkunft_kopf(tenant: Optional[str] = None) -> Dict[str, str]:
    """#4132: Kopf fuer Storage-Uploads im Auftrag eines Agenten oder Menschen.

    storage-api setzt daraus `origin_tenant`, aber nur fuer Schluessel mit
    `may_assert_origin` und nur bei Uploads. Ohne geprueften Mandanten: kein Kopf
    (lieber keine Herkunft als eine falsche)."""
    t = tenant if isinstance(tenant, str) and _MANDANT_RE.fullmatch(tenant or "") else mandant()
    return {"X-Caller-Tenant": t} if t else {}
