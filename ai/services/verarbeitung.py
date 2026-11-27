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
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger("api-ai.verarbeitung")

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
                "fehler": e.detail.get("error"), "grund": e.detail.get("grund"), "betreiber": betreiber_stand()}
    weg, vorgabe = ANBIETER[kf.anbieter]
    out = {"konfiguriert": True, "gueltig": True, "anbieter": kf.anbieter,
           "modell": kf.modell or vorgabe, "modell_aus_konfig": kf.modell is not None,
           "betreiber": betreiber_stand()}
    if kf.anbieter == "openai":
        out["openai"] = {"schluessel": ot.schluessel() is not None, "tagesgrenze_usd": ot.grenze_usd()}
    elif kf.anbieter == "kette":
        from ai.services import kette as k
        out["reihe"] = k.kette()
    return out


# --- #4135 (Plan 6567 A1): Zuweisung je Mandant ------------------------------
#
# Auf dem Kern gilt KI_VERARBEITUNG nur fuer den Betreiber. Ein anderer Mandant
# (aus dem GEPRUEFTEN JWT) bekommt ausschliesslich seine eigene Zuweisung:
# einen Schluesselweg (OpenAI mit Schluessel aus dem Tresor der auth-api) oder
# "keine". Abo-Wege (agy, codex, Claude) gehoeren dem Betreiber und laufen fuer
# Kunden nur auf deren eigenen Rechnern. Es gibt keinen Rueckfall.

BETREIBER_ENV = "BETREIBER_MANDANT"
MANDANTEN_DIR_ENV = "KI_MANDANTEN_DIR"
JWT_PFLICHT_ENV = "VERARBEITUNG_JWT_PFLICHT"
#: Anbieter, die ein Kunden-Mandant auf dem Kern haben darf.
KUNDEN_ANBIETER = ("openai",)
_VERWEIS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def betreiber() -> str:
    """Mandant, dem die Abos und Schluessel DIESER api-ai gehoeren. Auf dem Kern arkturian
    (Vorgabe), auf einer Kundeninstanz ihr eigener Mandant: BETREIBER_MANDANT=<TENANT_NAME>
    setzt der Rollout (P3 27.11.: ohne den Wert lief jedes Urteil der Instanz in 503)."""
    return (os.getenv(BETREIBER_ENV) or "arkturian").strip() or "arkturian"


def betreiber_stand() -> dict:
    """Fuer die Wirkungspruefung am Ziel: welcher Mandant gilt hier als eigener, und woher."""
    gesetzt = bool((os.getenv(BETREIBER_ENV) or "").strip())
    return {"mandant": betreiber(), "quelle": BETREIBER_ENV if gesetzt else "vorgabe"}


def jwt_pflicht() -> str:
    """beobachten (Vorgabe): ohne JWT weiter mit der Kern-Einstellung, aber im
    Journal; an: ohne gepruftes JWT 401. Scharf erst, wenn alle Aufrufer ein
    Dienst-JWT schicken (erst messen, dann verlangen)."""
    return "an" if (os.getenv(JWT_PFLICHT_ENV) or "").strip().lower() == "an" else "beobachten"


def _mandanten_dir():
    from pathlib import Path
    return Path(os.getenv(MANDANTEN_DIR_ENV) or "/var/lib/api-ai/mandanten")


@dataclass(frozen=True)
class MandantKonfig:
    mandant: str
    anbieter: str
    modell: Optional[str]
    verweis: Optional[str]
    tag_usd: float = 0.0
    monat_usd: float = 0.0


def _mandant_fehler(status: int, fehler: str, mandant: str, **extra) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": fehler, "mandant": mandant, **extra})


def mandant_konfig(mandant: str) -> MandantKonfig:
    """Zuweisung eines Kunden-Mandanten. Wirft 503 ohne oder mit ungueltiger
    Zuweisung und bei `keine`. Nie ein Rueckgriff auf KI_VERARBEITUNG."""
    p = _mandanten_dir() / f"{mandant}.json"
    try:
        roh = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _mandant_fehler(503, "verarbeitung_nicht_zugewiesen", mandant,
                              hint="Fuer diesen Mandanten ist kein KI-Konto zugewiesen. Es laeuft kein "
                                   "Anbieter, auch nicht der des Betreibers.") from None
    except OSError:
        raise _mandant_fehler(503, "verarbeitung_zuweisung_unlesbar", mandant) from None
    try:
        d = json.loads(roh)
    except ValueError:
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="kein_json") from None
    v = d.get("verarbeitung") if isinstance(d, dict) else None
    if v == "keine":
        raise _mandant_fehler(503, "verarbeitung_keine_ki", mandant,
                              hint="Der Mandant hat 'keine KI' gewaehlt.")
    if not isinstance(v, dict):
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="verarbeitung_fehlt")
    anbieter = str(v.get("anbieter") or "").strip().lower()
    if anbieter not in KUNDEN_ANBIETER:
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="anbieter_nicht_erlaubt",
                              erlaubt=list(KUNDEN_ANBIETER))
    modell = v.get("modell")
    if modell is not None and (not isinstance(modell, str) or not _MODELL.match(modell.strip())):
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="modell_ungueltig")
    verweis = v.get("verweis")
    if not isinstance(verweis, str) or not _VERWEIS.match(verweis):
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="verweis_fehlt")
    dk = d.get("deckel") if isinstance(d.get("deckel"), dict) else {}
    try:
        tag, monat = float(dk.get("tag_usd")), float(dk.get("monat_usd"))
    except (TypeError, ValueError):
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="deckel_fehlt") from None
    if not (0 < tag <= 1000 and 0 < monat <= 10000):
        raise _mandant_fehler(503, "verarbeitung_zuweisung_ungueltig", mandant, grund="deckel_fehlt")
    return MandantKonfig(mandant=mandant, anbieter=anbieter, modell=(modell or "").strip() or None,
                         verweis=verweis, tag_usd=tag, monat_usd=monat)


# --- 6586/I5: Weg je Aufrufer aus dem Zugangsregister -------------------------
#
# Den Weg (Konto, Modell, Tresor-Verweis) eines Kunden-Mandanten bestimmt der
# Dienstweg der cloud-api (`conversation` bzw. `prompt_agent`, erst der Mensch,
# sonst die Vorgabe des Mandanten). Bei uns bleibt nur der Deckel des Mandanten
# (Datei aus PUT /ai/mandanten/{t}/ki). Die alte `verarbeitung` dort zaehlt nicht mehr.

#: Dienste, die /ai/verarbeitung bedient (Kopf X-Dienst); ohne Kopf prompt_agent (issue-api).
VERARBEITUNG_DIENSTE = ("prompt_agent", "conversation")


def _zuweisung_datei(mandant: str) -> Optional[dict]:
    p = _mandanten_dir() / f"{mandant}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise _mandant_fehler(503, "deckel_fehlt", mandant, grund="zuweisung_unlesbar") from None
    return d if isinstance(d, dict) else None


def deckel(mandant: str, d: Optional[dict] = None) -> tuple:
    """(tag_usd, monat_usd) des Mandanten. Ohne gueltigen Deckel laeuft kein Schluessel: 503."""
    if d is None:
        d = _zuweisung_datei(mandant)
    dk = d.get("deckel") if isinstance(d, dict) and isinstance(d.get("deckel"), dict) else None
    try:
        tag, monat = float(dk["tag_usd"]), float(dk["monat_usd"])
    except (TypeError, KeyError, ValueError):
        raise _mandant_fehler(503, "deckel_fehlt", mandant,
                              hint="Ohne Tages- und Monatsdeckel des Mandanten laeuft kein Schluessel.") from None
    if not (0 < tag <= 1000 and 0 < monat <= 10000):
        raise _mandant_fehler(503, "deckel_fehlt", mandant, grund="deckel_ungueltig")
    return tag, monat


async def mandant_weg(mandant: str, dienst: str, mensch: Optional[str]) -> Optional[MandantKonfig]:
    """Schluesselweg eines Kunden-Mandanten aus dem Dienstweg. Auf dem Kern nur `weg: api`
    mit einem Anbieter aus KUNDEN_ANBIETER und Tresor-Verweis; None bei `weg: pool` (dann
    gilt KI_VERARBEITUNG); alles andere 503 mit Grund, nie ein Weg des Betreibers."""
    from ai.services import dienstweg
    datei = _zuweisung_datei(mandant)
    if isinstance(datei, dict) and datei.get("verarbeitung") == "keine":
        # Die ausdrueckliche Wahl des Mandanten "keine KI" gilt weiter, vor jedem Dienstweg.
        raise _mandant_fehler(503, "verarbeitung_keine_ki", mandant, hint="Der Mandant hat 'keine KI' gewaehlt.")
    w = await dienstweg.aufloesen(mandant, dienst, mensch)     # 403/503 mit Grund
    ort = {"weg": w.weg, "anbieter": w.anbieter, "quelle": w.quelle, "dienst": dienst}
    if w.weg == "pool":
        # Mandant im Pool des Kontoregisters (edera, p3; Cloud 27.11.): es gilt die
        # Einstellung der Instanz (KI_VERARBEITUNG), also die Abos des Betreibers.
        return None
    if w.weg != "api":
        raise _mandant_fehler(503, "weg_nicht_auf_dem_kern", mandant, **ort,
                              hint="Abo- und lokale Wege laufen auf den Rechnern des Mandanten "
                                   "(Knotenweg folgt). Es laeuft kein Weg des Betreibers.")
    if w.anbieter not in KUNDEN_ANBIETER:
        raise _mandant_fehler(503, "anbieter_nicht_unterstuetzt", mandant, **ort, erlaubt=list(KUNDEN_ANBIETER))
    if not (w.tresor_verweis and _VERWEIS.match(w.tresor_verweis)):
        raise _mandant_fehler(503, "konto_nicht_verfuegbar", mandant, grund="verweis_fehlt", **ort)
    modell = (w.modell or "").strip() or None
    if modell is not None and not _MODELL.match(modell):
        raise _mandant_fehler(503, "konto_nicht_verfuegbar", mandant, grund="modell_ungueltig", **ort)
    tag, monat = deckel(mandant, datei)
    return MandantKonfig(mandant=mandant, anbieter=w.anbieter, modell=modell, verweis=w.tresor_verweis,
                         tag_usd=tag, monat_usd=monat)


def urteil_sperre(mandant: Optional[str]) -> Optional[HTTPException]:
    """#4135 (A1.4, erster Teil): Urteile (/ai/jev, /ai/decisions) eines Kunden-Mandanten
    laufen nie ueber die Schluessel des Betreibers (TypeSafe, OpenAI). None = erlaubt
    (Betreiber bzw. eigener Mandant der Instanz, oder kein geprueftes JWT). Sonst 503:
    ohne Zuweisung, mit `urteil: keine`, und `urteil: openai` (Schluessel des Kunden folgt)."""
    if mandant is None or mandant == betreiber():
        return None
    p = _mandanten_dir() / f"{mandant}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _mandant_fehler(503, "urteil_nicht_zugewiesen", mandant, betreiber=betreiber(),
                               hint="Fuer diesen Mandanten ist kein Urteils-Konto zugewiesen. Ist das der "
                                    "eigene Mandant dieser Instanz, fehlt BETREIBER_MANDANT in der api-ai.")
    except (OSError, ValueError):
        return _mandant_fehler(503, "urteil_zuweisung_unlesbar", mandant)
    if not isinstance(d, dict) or d.get("urteil") != "openai":
        return _mandant_fehler(503, "urteil_keine", mandant, hint="Der Mandant hat fuer Urteile 'keine' gewaehlt.")
    return _mandant_fehler(503, "urteil_konto_folgt", mandant,
                           hint="Urteile mit dem Schluessel des Mandanten kommen mit #4135 A1.4. "
                                "Es laeuft kein Schluessel des Betreibers.")


def urteil_pruefen() -> None:
    """Wirft die Sperre aus urteil_sperre() fuer den Mandanten des laufenden Aufrufs."""
    from ai.services import aufruf
    e = urteil_sperre(aufruf.mandant())
    if e is not None:
        logger.warning("urteil gesperrt: mandant=%s %s", e.detail.get("mandant"), e.detail.get("error"))
        raise e
