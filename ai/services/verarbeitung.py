"""#4095: Rolle "zustandslose Verarbeitung" — der Anbieter kommt aus der Instanz.

Alex, 26.11.2026: Abhaengige KI-Dienste (Intake der issue-api u. a.) nennen kein
Modell mehr. Jede Instanz legt in `KI_VERARBEITUNG` fest, wer die Arbeit macht;
der Rollout schreibt das Feld aus `tenants.json` (`ki_verarbeitung`).

Form (JSON oder nur der Anbietername):

    KI_VERARBEITUNG={"anbieter": "hydrogen", "modell": "gemini-4-hydrogen-low"}
    KI_VERARBEITUNG=codex

Anbieter:

- `hydrogen`: nur agy (Google-Abo), Vorgabe gemini-4-hydrogen-low. Betreiber.
- `codex`:    nur codex (ChatGPT-Abo).
- `kette`:    die Abo-Kette der Instanz in der Reihenfolge aus KI_KETTE.
- `openai`:   OpenAI-Platform-Schluessel der Instanz (/ai/openai, #3640) mit
              dessen Tagesgrenze OPENAI_TEXT_DAILY_MAX_USD. Ohne sie 503.

Fail-closed: Fehlt die Konfiguration oder ist sie ungueltig, kommt 503 mit klarer
Meldung. Es gibt keinen Rueckfall auf einen anderen Anbieter, schon gar nicht auf
DeepSeek: Ein bezahlter Weg laeuft nur, wenn die Instanz ihn ausdruecklich nennt.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

ENV = "KI_VERARBEITUNG"

#: Anbieter -> (Weg der Kette oder None, Vorgabemodell oder None)
ANBIETER = {
    "hydrogen": ("agy", "gemini-4-hydrogen-low"),
    "codex": ("codex", None),
    "kette": (None, None),
    "openai": (None, None),
}

_MODELL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


@dataclass(frozen=True)
class Konfig:
    anbieter: str
    modell: Optional[str]


def _ungueltig(grund: str) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "error": "verarbeitung_konfig_ungueltig", "grund": grund,
        "hint": f"{ENV} der Instanz pruefen (anbieter: {', '.join(ANBIETER)})."})


def konfig() -> Konfig:
    """Liest KI_VERARBEITUNG. Wirft 503, wenn sie fehlt oder ungueltig ist."""
    roh = (os.getenv(ENV) or "").strip()
    if not roh:
        raise HTTPException(status_code=503, detail={
            "error": "verarbeitung_nicht_konfiguriert",
            "hint": f"Diese Instanz hat keine {ENV}. Es laeuft kein Anbieter, auch kein Ersatz."})
    if roh.startswith("{"):
        try:
            d = json.loads(roh)
        except ValueError:
            raise _ungueltig("kein_json") from None
        if not isinstance(d, dict):
            raise _ungueltig("kein_objekt")
        anbieter = str(d.get("anbieter") or "").strip().lower()
        modell = d.get("modell")
    else:
        anbieter, modell = roh.lower(), None
    if anbieter not in ANBIETER:
        raise _ungueltig("anbieter_unbekannt")
    if modell is not None:
        if not isinstance(modell, str) or not _MODELL.match(modell.strip()):
            raise _ungueltig("modell_ungueltig")
        modell = modell.strip()
    return Konfig(anbieter=anbieter, modell=modell)


def schema_anweisung(schema: dict) -> str:
    return ("Antworte ausschliesslich mit genau einem JSON-Objekt, ohne Erklaertext und ohne "
            "Codeblock, nach diesem JSON-Schema:\n" + json.dumps(schema, ensure_ascii=False))


def json_lesen(text: str, schema: dict) -> dict:
    """Holt das JSON-Objekt aus der Antwort und prueft Typ und Pflichtfelder.

    Bewusst schlicht (kein jsonschema im Dienst): Objekt, `required` vorhanden.
    Wirft ValueError mit einem kurzen Grund."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    if not s.startswith("{"):
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("kein_json")
        s = s[a:b + 1]
    try:
        d = json.loads(s)
    except ValueError:
        raise ValueError("kein_json") from None
    if not isinstance(d, dict):
        raise ValueError("kein_objekt")
    fehlend = [k for k in (schema.get("required") or []) if isinstance(k, str) and k not in d]
    if fehlend:
        raise ValueError("pflichtfelder_fehlen:" + ",".join(fehlend[:10]))
    return d


def stand() -> dict:
    """Was wirkt am laufenden Prozess? Ohne Geheimnisse, ohne Modellaufruf."""
    from ai.services import openai_text as ot
    roh = (os.getenv(ENV) or "").strip()
    try:
        kf = konfig()
    except HTTPException as e:
        return {"konfiguriert": bool(roh), "gueltig": False,
                "fehler": e.detail.get("error"), "grund": e.detail.get("grund")}
    weg, vorgabe = ANBIETER[kf.anbieter]
    out = {"konfiguriert": True, "gueltig": True, "anbieter": kf.anbieter,
           "modell": kf.modell or vorgabe, "modell_aus_konfig": kf.modell is not None}
    if kf.anbieter == "openai":
        out["openai"] = {"schluessel": ot.schluessel() is not None, "tagesgrenze_usd": ot.grenze_usd()}
    elif kf.anbieter == "kette":
        from ai.services import kette as k
        out["reihe"] = k.kette()
    return out
