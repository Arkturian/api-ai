"""Eigene codex-Konten fuer api-ai (#3165, Freigabe Alex 20.11.2026).

Bis #3165 lief codex als alex mit dessen Anmeldung (~/.codex/auth.json, ein
Link, den cloud-api beim Kontowechsel umhaengt). Jetzt laeuft codex als
Dienstbenutzer `apiai-codex`, je ChatGPT-Konto mit einer EIGENEN
Geraetecode-Anmeldung unter <wurzel>/<konto> (eigene Token-Kette, nie eine
Kopie aus /home/alex). Damit waehlt api-ai das Konto selbst.

Regel wie cloud-api `codex_konto_wechsel` (Bedingung Cloud 20.11.):
- gewechselt wird nur, wenn das aktive Konto fast voll ist (erreicht oder
  >= SCHWELLE_PCT),
- Ziel: Anmeldung ok, selbst nicht fast voll, mindestens VORSPRUNG_PCT Punkte
  besser; bei mehreren das mit dem geringsten Stand,
- hoechstens ein Wechsel je MIN_ABSTAND_S.
`account/rateLimits/read` misst das KONTO, nicht die Anmeldung: api-ai und die
Codex-Agenten sehen dieselben Zahlen und waehlen nach derselben Regel.

Gemessen wird ueber den Wrapper (`<konto> limits`), alle MESS_ABSTAND_S und
sofort, wenn ein Aufruf an einem Kontingent scheitert. Ereignisse (Wechsel,
kein Ziel, Anmeldung abgelehnt) stehen in GET /ai/codex/konten und gehen an
Login (ereignis_melden), sobald api-ai dafuer einen Ausweis hat.

Ohne API_AI_CODEX_KONTEN ist nichts davon aktiv.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

SCHWELLE_PCT = float(os.getenv("CODEX_WECHSEL_SCHWELLE_PCT", "95"))
VORSPRUNG_PCT = float(os.getenv("CODEX_WECHSEL_VORSPRUNG_PCT", "20"))
MIN_ABSTAND_S = float(os.getenv("CODEX_WECHSEL_MIN_ABSTAND_S", "3600"))
MESS_ABSTAND_S = float(os.getenv("CODEX_KONTEN_MESS_ABSTAND_S", "900"))
SOFORT_SPERRE_S = 120.0     # Messung nach Kontingentfehler hoechstens alle 2 min

_NAME = re.compile(r"^[a-z][a-z0-9]{1,31}$")
_UNAUTH_RE = re.compile(r"\b401\b|unauthori[sz]ed|revoked|invalid_grant|sign(ing)? in again", re.I)
_LIMIT_RE = re.compile(r"usage.?limit|rate.?limit.?reached|limit reached|hit your (weekly|daily|session|usage)? ?limit", re.I)

# api-ai laeuft mit mehreren Prozessen (uvicorn --workers 2): Zustand und
# Messungen liegen deshalb nur in der Datei, eine Runde laeuft unter flock,
# und jeder Prozess liest das aktive Konto bei jedem Aufruf neu.
_LOCK = threading.RLock()
_SOFORT_ZULETZT = 0.0


# --- Einstellungen -----------------------------------------------------------

def konten() -> List[str]:
    """Konten in Vorzugsreihenfolge aus API_AI_CODEX_KONTEN ("apopovic,tsaier")."""
    roh = (os.getenv("API_AI_CODEX_KONTEN") or "").split(",")
    return [k.strip() for k in roh if _NAME.fullmatch(k.strip())]


def aktiv_modus() -> bool:
    return bool(konten())


def wurzel() -> str:
    return (os.getenv("API_AI_CODEX_KONTEN_WURZEL") or "/var/lib/api-ai-codex/konten").rstrip("/")


def home(konto: str) -> str:
    if not _NAME.fullmatch(konto or ""):
        raise ValueError("Kontoname")
    return f"{wurzel()}/{konto}"


def konto_aus_home(pfad: Optional[str]) -> Optional[str]:
    """Kontoname, wenn `pfad` genau ein Konto-Home ist, sonst None."""
    p = (pfad or "").rstrip("/")
    w = wurzel() + "/"
    if not p.startswith(w):
        return None
    k = p[len(w):]
    return k if (_NAME.fullmatch(k) and k in konten()) else None


def _zustand_datei() -> str:
    return os.getenv("API_AI_CODEX_KONTEN_ZUSTAND") or "/var/lib/api-ai/codex_konten.json"


# --- Zustand -----------------------------------------------------------------

def _leer() -> dict:
    ks = konten()
    return {"aktiv": ks[0] if ks else None, "letzter_wechsel": 0.0, "stand": {},
            "wechsel": [], "ereignisse": [], "gemeldet": "", "abgelehnt": {}}


def _laden() -> dict:
    z = _leer()
    try:
        with open(_zustand_datei(), "r", encoding="utf-8") as fh:
            gelesen = json.load(fh)
        if isinstance(gelesen, dict):
            z.update({k: gelesen[k] for k in z if k in gelesen})
    except FileNotFoundError:
        pass
    except Exception as e:                                    # noqa: BLE001
        logger.warning("codex-konten: Zustand unlesbar (%s), beginne neu", type(e).__name__)
    if z.get("aktiv") not in konten():
        z["aktiv"] = konten()[0] if konten() else None
    return z


def _speichern(z: dict) -> None:
    pfad = _zustand_datei()
    tmp = pfad + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(z, fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, pfad)
    except OSError as e:
        logger.warning("codex-konten: Zustand nicht gespeichert: %s", type(e).__name__)


def aktiv() -> Optional[str]:
    with _LOCK:
        return _laden().get("aktiv")


def aktiv_home() -> Optional[str]:
    k = aktiv()
    return home(k) if k else None


def _zuruecksetzen() -> None:            # Tests
    global _SOFORT_ZULETZT
    _SOFORT_ZULETZT = 0.0


class _runden_sperre:
    """flock auf <zustand>.lock, nicht blockierend: haelt schon ein anderer
    Prozess die Runde, faellt diese aus (ok=False)."""

    def __init__(self):
        self.fh = None
        self.ok = False

    def __enter__(self):
        import fcntl
        try:
            self.fh = open(_zustand_datei() + ".lock", "a")
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.ok = True
        except OSError:
            self.ok = False
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()


# --- Messen ------------------------------------------------------------------

def limits_aus(antwort: dict) -> Optional[dict]:
    """Wie cloud-api codex_konto_wechsel.limits_aus: hoechste Auslastung der
    Fenster, fruehester Reset, erreicht ja/nein."""
    rl = ((antwort or {}).get("rateLimits") or {})
    fenster = [f for f in (rl.get("primary"), rl.get("secondary")) if isinstance(f, dict)]
    werte = [float(f["usedPercent"]) for f in fenster if f.get("usedPercent") is not None]
    if not werte and not rl.get("rateLimitReachedType"):
        return None
    resets = [int(f["resetsAt"]) for f in fenster if f.get("resetsAt")]
    return {"used_pct": max(werte) if werte else 100.0,
            "resets_at": min(resets) if resets else None,
            "erreicht": bool(rl.get("rateLimitReachedType"))}


def auswerten(zeilen: str) -> dict:
    """Ausgabe von `<wrapper> <konto> limits` (JSON-Zeilen der Antworten
    id 2 und 3) -> {state, detail, email, limits}. Wie cloud-api classify."""
    antworten: dict = {}
    for z in (zeilen or "").splitlines():
        try:
            o = json.loads(z)
        except ValueError:
            continue
        if isinstance(o, dict) and o.get("id") in (2, 3):
            antworten[o["id"]] = o
    acc, rl = antworten.get(2) or {}, antworten.get(3) or {}
    fehler = [str((o.get("error") or {}).get("message") or "") for o in (acc, rl) if o.get("error")]
    konto = ((acc.get("result") or {}).get("account") or {})
    erg = {"email": konto.get("email"), "limits": limits_aus(rl.get("result") or {})}
    for f in fehler:
        if _UNAUTH_RE.search(f):
            return {**erg, "state": "unauthorized", "detail": f[:300]}
    if not acc and not rl:
        return {**erg, "state": "probe_error", "detail": "keine Antwort vom app-server"}
    if acc.get("result") is not None and not konto:
        return {**erg, "state": "logged_out", "detail": "account/read ohne Konto"}
    if fehler:
        return {**erg, "state": "probe_error", "detail": fehler[0][:300]}
    if acc.get("result") is None or rl.get("result") is None:
        return {**erg, "state": "probe_error", "detail": "unvollstaendige Antwort"}
    return {**erg, "state": "ok", "detail": ""}


def _messbefehl(konto: str) -> List[str]:
    from ai.services import cli_dienst
    user = (os.getenv("CODEX_RUN_AS_USER") or "").strip()
    return ["sudo", "-n", "-u", user, os.path.join(cli_dienst.wrapper_dir() or "", cli_dienst.codex_wrapper()),
            konto, "limits"]


def messen(konto: str, lauf: Callable = subprocess.run) -> dict:
    try:
        r = lauf(_messbefehl(konto), capture_output=True, text=True, timeout=30)
        m = auswerten(r.stdout or "")
        if m["state"] == "probe_error" and r.returncode not in (0, None):
            m["detail"] = f"rc={r.returncode} {(r.stderr or '')[-200:]}".strip()
    except Exception as e:                                    # noqa: BLE001
        m = {"state": "probe_error", "detail": type(e).__name__, "email": None, "limits": None}
    m["gemessen_um"] = time.time()
    return m


# --- Entscheiden -------------------------------------------------------------

def _voll(l: Optional[dict]) -> bool:
    return bool(l) and (l["erreicht"] or l["used_pct"] >= SCHWELLE_PCT)


def _erreicht(l: Optional[dict]) -> bool:
    return bool(l) and (l["erreicht"] or l["used_pct"] >= 100)


def entscheide(aktiv_name: str, stand: dict) -> Optional[str]:
    """Zielkonto oder None. stand: konto -> {state, limits}."""
    a = stand.get(aktiv_name) or {}
    if not _voll(a.get("limits")):
        return None
    jetzt = a["limits"]["used_pct"]
    # Cloud 20.11.: Ist das Kontingent ERREICHT, entfaellt der Vorsprung --
    # sonst bliebe api-ai bei apopovic 100 % / tsaier 82 % tot stehen. Bei
    # „fast voll“ (95-99 %) bleiben die 20 Punkte.
    vorsprung = 0.0 if _erreicht(a["limits"]) else VORSPRUNG_PCT
    gut = [k for k, s in stand.items()
           if k != aktiv_name and s.get("state") == "ok" and s.get("limits")
           and not _voll(s["limits"]) and jetzt - s["limits"]["used_pct"] >= vorsprung]
    if not gut:
        return None
    return min(gut, key=lambda k: (stand[k]["limits"]["used_pct"], stand[k]["limits"].get("resets_at") or 0))


def _pct(stand: dict, k: str) -> Optional[float]:
    l = (stand.get(k) or {}).get("limits")
    return None if not l else round(float(l["used_pct"]), 1)


def kein_ziel_text(aktiv_name: str, stand: dict) -> str:
    """Wortlaut wie cloud-api `_kein_ziel_text` (R83): nennt den echten Grund."""
    l = (stand.get(aktiv_name) or {}).get("limits") or {}
    jetzt = l.get("used_pct", 100.0)
    wie = "erreicht" if l.get("erreicht") or jetzt >= 100 else "fast voll"
    kopf = f"api-ai codex: Konto {aktiv_name} ist {wie} ({jetzt:.0f} % verbraucht). "
    andere = [k for k, s in stand.items() if k != aktiv_name and s.get("limits") and s.get("state") == "ok"]
    if not andere:
        return kopf + "Es gibt kein anderes messbares Konto. api-ai wartet bis zum Reset."
    best = min(andere, key=lambda k: stand[k]["limits"]["used_pct"])
    b = stand[best]["limits"]["used_pct"]
    if _voll(stand[best]["limits"]):
        return kopf + f"Auch {best} ist voll ({b:.0f} %). api-ai wartet bis zum Reset."
    return kopf + (f"Kein Wechsel: {best} steht bei {b:.0f} %, das sind nur {jetzt - b:.0f} Punkte "
                   f"mehr Luft. Bei fast vollem Konto wird erst ab {VORSPRUNG_PCT:.0f} Punkten Abstand "
                   f"gewechselt, bei erreichtem Kontingent sofort.")


def _ereignis(z: dict, art: str, text: str, **daten) -> dict:
    e = {"zeit": time.time(), "art": art, "text": text, **daten}
    z["ereignisse"] = (z.get("ereignisse") or [])[-9:] + [e]
    logger.warning("codex-konten: %s", text)
    return e


def runde(melden: Optional[Callable[[dict], None]] = None, lauf: Callable = subprocess.run) -> dict:
    """Alle Konten messen, Anmeldungen pruefen, ggf. wechseln. Rueckgabe:
    {aktion: bleibt|gewechselt|kein_ziel|abstand, ...}."""
    if not aktiv_modus():
        return {"aktion": "aus"}
    with _runden_sperre() as sp:
        if not sp.ok:
            return {"aktion": "laeuft_schon"}
        return _runde(melden, lauf)


def _runde(melden, lauf) -> dict:
    gemessen = {k: messen(k, lauf=lauf) for k in konten()}
    neu: list = []
    with _LOCK:
        z = _laden()
        z["stand"] = gemessen
        stand = {k: dict(v) for k, v in gemessen.items()}
        # Anmeldung abgelehnt: einmal je Konto melden, bis sie wieder ok ist.
        for k, s in stand.items():
            war = bool((z.get("abgelehnt") or {}).get(k))
            if s.get("state") == "unauthorized" and not war:
                z.setdefault("abgelehnt", {})[k] = True
                neu.append(_ereignis(z, "anmeldung_abgelehnt",
                                     f"api-ai codex: Anmeldung von Konto {k} abgelehnt "
                                     f"({s.get('detail') or 'unauthorized'}). Erneuern kann nur der "
                                     f"Kontoinhaber per Geraetecode.", konto=k))
            elif s.get("state") == "ok" and war:
                z["abgelehnt"][k] = False
        a = z["aktiv"]
        ziel = entscheide(a, stand)
        if ziel is None:
            if _voll((stand.get(a) or {}).get("limits")):
                schl = f"kein_ziel:{a}"
                if z.get("gemeldet") != schl:
                    z["gemeldet"] = schl
                    neu.append(_ereignis(z, "kein_ziel", kein_ziel_text(a, stand), von=a,
                                         stand={k: _pct(stand, k) for k in stand}))
                erg = {"aktion": "kein_ziel", "von": a}
            else:
                z["gemeldet"] = ""
                erg = {"aktion": "bleibt", "aktiv": a}
        elif time.time() - float(z.get("letzter_wechsel") or 0) < MIN_ABSTAND_S:
            erg = {"aktion": "abstand", "von": a, "ziel": ziel}
        else:
            von_pct, nach_pct = _pct(stand, a), _pct(stand, ziel)
            w = {"zeit": time.time(), "von": a, "nach": ziel, "von_pct": von_pct, "nach_pct": nach_pct,
                 "grund": "erreicht" if (stand[a]["limits"] or {}).get("erreicht") else "fast_voll"}
            z["aktiv"], z["letzter_wechsel"], z["gemeldet"] = ziel, w["zeit"], ""
            z["wechsel"] = (z.get("wechsel") or [])[-9:] + [w]
            neu.append(_ereignis(z, "wechsel", f"api-ai codex: Wechsel {a} → {ziel} "
                                 f"({a} {von_pct:.0f} %, {ziel} {nach_pct:.0f} %)", von=a, nach=ziel))
            erg = {"aktion": "gewechselt", "von": a, "nach": ziel}
        _speichern(z)
    for e in neu:
        if melden:
            try:
                melden(e)
            except Exception as ex:                           # noqa: BLE001
                logger.warning("codex-konten: Meldung an Login gescheitert: %s", type(ex).__name__)
    return erg


def ist_kontingentfehler(text: str) -> bool:
    return bool(_LIMIT_RE.search(text or ""))


def nach_kontingentfehler(melden: Optional[Callable[[dict], None]] = None) -> Optional[dict]:
    """Ein Aufruf scheiterte am Kontingent: sofort messen statt bis zur
    naechsten Runde warten (hoechstens alle SOFORT_SPERRE_S)."""
    global _SOFORT_ZULETZT
    if not aktiv_modus():
        return None
    with _LOCK:
        if time.time() - _SOFORT_ZULETZT < SOFORT_SPERRE_S:
            return None
        _SOFORT_ZULETZT = time.time()
    return runde(melden=melden)


# --- Lesen -------------------------------------------------------------------

def uebersicht() -> dict:
    """Fuer GET /ai/codex/konten (Login, #3165). Keine Pfade, keine Token."""
    with _LOCK:
        z = _laden()
        ks = []
        for k in konten():
            s = (z.get("stand") or {}).get(k) or {}
            l = s.get("limits") or {}
            ks.append({"name": k, "email": s.get("email"),
                       "used_pct": None if l.get("used_pct") is None else round(float(l["used_pct"]), 1),
                       "resets_at": l.get("resets_at"), "state": s.get("state") or "unbekannt",
                       "gemessen_um": s.get("gemessen_um")})
        return {"aktiv": z.get("aktiv"), "konten": ks,
                "wechsel": list(z.get("wechsel") or [])[-10:],
                "ereignisse": list(z.get("ereignisse") or [])[-10:],
                "regel": {"schwelle_pct": SCHWELLE_PCT, "vorsprung_pct": VORSPRUNG_PCT,
                          "min_abstand_s": MIN_ABSTAND_S, "mess_abstand_s": MESS_ABSTAND_S}}
