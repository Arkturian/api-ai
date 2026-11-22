"""#3656: /ai/gemini auf einer Instanz ohne eigenes agy ueber den Kern.

Alex, 21./22.11.: P3 verhaelt sich wie der Kern und nutzt Gemini Hydrogen und
Carbon, ohne eigene Google-Anmeldung. Ist `GEMINI_KERN_URL` gesetzt, reicht
/ai/gemini reine Textaufrufe an `<GEMINI_KERN_URL>/ai/extern/hydrogen` weiter,
mit dem Schluessel aus `GEMINI_KERN_SCHLUESSEL_DATEI`.

Bewusst eng: Nur Text. Bilder, Verlauf, Persona, Werkzeuge und unbekannte
Modellnamen ergeben eine klare Ablehnung statt eines stillen Ausweichens.
Ohne `GEMINI_KERN_URL` aendert sich nichts.
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger("api-ai.gemini_kern")

URL_ENV = "GEMINI_KERN_URL"
# Freigabe je Aufrufer (P3-Cloud, 22.11.): nur diese Namen gehen zum Kern, zuerst
# nur "Cloud" (Sprechstile). Weitere erst nach Jaschas Ja, als Eintrag hier.
AUFRUFER_ENV = "GEMINI_KERN_AUFRUFER"
AUFRUFER_KOPF = "X-Extern-Aufrufer"
DATEI_ENV = "GEMINI_KERN_SCHLUESSEL_DATEI"
DATEI_VORGABE = "/etc/agentos/kern-hydrogen.key"
MODELL_VORGABE = "gemini-4-hydrogen-low"
# Muss zu extern_routes._MODELL passen; der Kern prueft selbst noch einmal.
MODELLE = re.compile(r"gemini-4-(hydrogen(-(low|medium|high|xhigh))?|carbon(-(low|medium|high))?)")


# Gesetzt waehrend /ai/extern/hydrogen: Dort laeuft agy lokal, nie wieder weiter
# zum Kern, auch wenn GEMINI_KERN_URL versehentlich auch hier gesetzt ist.
IM_KERN: "contextvars.ContextVar[bool]" = contextvars.ContextVar("gemini_im_kern", default=False)


def ziel() -> Optional[str]:
    if IM_KERN.get():
        return None
    url = (os.getenv(URL_ENV) or "").strip().rstrip("/")
    return url or None


def _schluessel() -> str:
    pfad = os.getenv(DATEI_ENV) or DATEI_VORGABE
    try:
        with open(pfad, encoding="ascii") as f:
            wert = f.read().strip()
    except (OSError, UnicodeDecodeError) as e:
        logger.error("gemini_kern: Schluesseldatei nicht lesbar (%s)", type(e).__name__)
        wert = ""
    if not wert:
        raise HTTPException(status_code=503, detail={
            "error": "cli_fehlt", "weg": "agy", "resets_at": None,
            "grund": "kern_schluessel_fehlt",
            "hint": "Gemini laeuft auf dieser Instanz ueber den Kern; die Schluesseldatei fehlt."})
    return wert


def _ablehnen(grund: str, text: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"error": "gemini_kern_nicht_moeglich", "grund": grund,
                                                  "hint": text, "routed_to": "kern"})


def rumpf(prompt, model: Optional[str]) -> dict:
    """Prueft und baut den Rumpf fuer /ai/extern/hydrogen. Wirft 422 statt still zu kuerzen."""
    text = prompt.prompt if isinstance(prompt.prompt, str) else None
    bilder = list(prompt.image_paths or [])
    if text is None:
        bilder += list(prompt.prompt.image_paths or []) + list(prompt.prompt.images or [])
        text = prompt.prompt.text
    if bilder:
        raise _ablehnen("bilder", "Ueber den Kern geht nur Text, keine Bilder.")
    if prompt.conversation_history:
        raise _ablehnen("verlauf", "Ueber den Kern geht kein conversation_history; den Verlauf in prompt schreiben.")
    if prompt.persona_variant:
        raise _ablehnen("persona", "persona_variant gibt es ueber den Kern nicht.")
    if prompt.tools:
        raise HTTPException(status_code=403, detail={"error": "tools_disabled"})
    modell = model or prompt.model or MODELL_VORGABE
    if not MODELLE.fullmatch(modell):
        raise _ablehnen("modell", f"Ueber den Kern gehen nur gemini-4-hydrogen und gemini-4-carbon "
                                  f"(mit Stufe), nicht {modell!r}.")
    out = {"prompt": text or "", "model": modell}
    if prompt.system:
        out["system"] = prompt.system
    return out


def freigegeben() -> set:
    return {n.strip() for n in (os.getenv(AUFRUFER_ENV) or "").split(",") if n.strip()}


def aufrufer_pruefen(request) -> str:
    """Name des Aufrufers (geprueftes JWT vor X-Agent-Name). Nicht freigegeben: 403,
    bevor irgendetwas die Instanz verlaesst."""
    from ai.services.bild_verbrauch import aufrufer
    kopf = request.headers if request is not None else {}
    name, _quelle = aufrufer(kopf.get("authorization"), kopf.get("x-agent-name"))
    if name not in freigegeben():
        logger.warning("gemini_kern: Aufrufer %r nicht freigegeben", name[:64])
        raise HTTPException(status_code=403, detail={
            "error": "gemini_kern_aufrufer_gesperrt", "aufrufer": name[:64], "routed_to": "kern",
            "hint": f"Gemini ueber den Kern ist je Aufrufer freigeschaltet ({AUFRUFER_ENV})."})
    return name


async def aufrufen(prompt, model: Optional[str], request=None) -> dict:
    """Ruft den Kern. Gibt die Antwort als dict zurueck oder wirft die Antwort des
    Kerns unveraendert als HTTPException weiter (feste Codes bleiben erhalten)."""
    import httpx
    url = ziel()
    wer = aufrufer_pruefen(request)
    nutzlast = rumpf(prompt, model)
    kopf = {"X-API-KEY": _schluessel(), AUFRUFER_KOPF: re.sub(r"[^A-Za-z0-9._-]", "_", wer)[:64]}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(330, connect=10)) as c:
            r = await c.post(f"{url}/ai/extern/hydrogen", json=nutzlast, headers=kopf)
    except Exception as e:                                    # noqa: BLE001
        logger.warning("gemini_kern: Kern nicht erreichbar (%s)", type(e).__name__)
        raise HTTPException(status_code=503, detail={
            "error": "gemini_kern_nicht_erreichbar", "weg": "agy", "resets_at": None,
            "hint": "Der Kern antwortet nicht; Gemini ist auf dieser Instanz gerade nicht verfuegbar."})
    try:
        inhalt = r.json()
    except ValueError:
        inhalt = None
    if r.status_code != 200 or not isinstance(inhalt, dict):
        detail = inhalt.get("detail") if isinstance(inhalt, dict) else None
        if not isinstance(detail, dict):
            detail = {"error": "gemini_kern_fehler", "status": r.status_code}
        raise HTTPException(status_code=r.status_code if r.status_code >= 400 else 502,
                            detail={**detail, "routed_to": "kern"}, headers=(
                                {"Retry-After": r.headers["Retry-After"]} if "Retry-After" in r.headers else None))
    inhalt["routed_to"] = "kern"
    return inhalt
