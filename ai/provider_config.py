"""Provider-Konfiguration je Instanz — eine Wahrheit fuer zwei Fragen.

1. Welche Provider hat DIESE Instanz konfiguriert? (`/health` -> `providers`,
   read-only, ohne Geheimnisse, ohne Erzeugung — fuer Kundeninstanzen ohne
   unsere Toepfe.)
2. Was passiert, wenn ein Endpunkt einen Provider braucht, dessen Schluessel
   fehlt? Bisher an zwoelf Stellen ein 500 mit je eigenem Text — fuer den
   Aufrufer ununterscheidbar von einem Absturz (Steward, Davids Instanz,
   Issue #36). Jetzt ueberall 403 `provider_not_configured`: die Anfrage war
   in Ordnung, die Instanz darf/kann diesen Provider nicht — dieselbe Klasse
   wie das Billing-Tor.
"""

import os
from typing import Dict, List

from fastapi import HTTPException

# provider -> Env-Variablen, die ALLE gesetzt sein muessen
PROVIDER_ENV: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "minimax": ["MINIMAX_MULTIMODAL_API_KEY"],
    "elevenlabs": ["ELEVENLABS_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "kling": ["KLING_API_KEY"],
    "higgsfield": ["HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET"],
    "tencent_3d": ["TENCENT_SECRET_ID", "TENCENT_SECRET_KEY"],
    "aimlapi": ["AIMLAPI_KEY"],
}


def provider_configured(provider: str) -> bool:
    envs = PROVIDER_ENV.get(provider)
    if not envs:
        return False
    return all(bool((os.getenv(e) or "").strip()) for e in envs)


def providers_status() -> Dict[str, bool]:
    """Nur Wahrheitswerte — nie Schluessel, nie Teile davon."""
    return {p: provider_configured(p) for p in PROVIDER_ENV}


def provider_missing(provider: str) -> HTTPException:
    """Die eine Ausnahme fuer 'Schluessel fehlt' — 403, nicht 500."""
    envs = PROVIDER_ENV.get(provider, [provider.upper() + "_API_KEY"])
    return HTTPException(
        status_code=403,
        detail={
            "error": "provider_not_configured",
            "provider": provider,
            "env": envs,
            "hint": (f"{provider} ist auf dieser Instanz nicht konfiguriert "
                     f"({', '.join(envs)} fehlt in der Dienst-.env). "
                     "Welche Provider verfuegbar sind, zeigt GET /health -> providers."),
        },
    )
