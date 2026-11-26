"""#4127: Kostenbuch je Aufruf und Bericht fuer GET /ai/kosten.

Alex, 26.11.2026: „Ich brauche volle Kostentransparenz“ — TTS, STT, zustandslos,
Prompt Agent, Conversation-Tracking, getrennt nach Geld und Abo, auf Abruf.

Bis hierher gab es nur Monatszaehler je Anbieter (ohne Aufrufer) und das
Aufruferprotokoll (ohne Kosten). Das Kostenbuch verbindet beides: An jeder
Stelle, an der ein Zaehler bucht, kommt eine Zeile dazu, mit Pfad und Aufrufer
des laufenden Aufrufs (Kontext aus der Middleware `_aufrufer_protokoll`).

Datei: KOSTENBUCH_DIR (Vorgabe /var/lib/api-ai/kostenbuch)/<JJJJ-MM-TT>.jsonl,
Tag in Wiener Zeit. Eine Zeile:

    {"ts", "pfad", "aufrufer", "quelle", "anbieter", "art": "geld|abo|guthaben",
     "modell", "anzahl", "tokens_ein", "tokens_aus", "zeichen", "minuten",
     "credits", "usd", "gegenwert_usd", "genauigkeit"}

`usd` ist echtes Geld. Bei Abo-Wegen steht der API-Gegenwert, den die Zaehler
schon immer rechnen, nur in `gegenwert_usd`: er wird nie bezahlt.

Ein Fehler beim Buchen darf nie den Aufruf treffen (die Leistung ist erbracht).
"""
from __future__ import annotations

import contextvars
import fcntl
import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("api-ai.kostenbuch")

WIEN = ZoneInfo("Europe/Vienna")
EUR_USD = 1.05          # wie die Zaehler (EUR_USD_RATE)
ARTEN = ("geld", "abo", "guthaben")
GENAUIGKEITEN = ("gezaehlt", "geschaetzt")
GRUPPEN = ("weg", "aufrufer", "anbieter", "tag")
MAX_TAGE = 400

# Pfad und Aufrufer des laufenden Aufrufs; gesetzt von main._aufrufer_protokoll.
KONTEXT: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("kostenbuch_kontext", default=None)


def verzeichnis() -> Path:
    return Path(os.getenv("KOSTENBUCH_DIR") or "/var/lib/api-ai/kostenbuch")


def aufruferverzeichnis() -> Path:
    return Path(os.getenv("AUFRUFER_LOG_DIR") or "/var/lib/api-ai/aufrufer")


def _zahl(x, ganz: bool = False):
    try:
        v = float(x or 0)
    except (TypeError, ValueError):
        return 0
    if v != v or v < 0:
        return 0
    return int(v) if ganz else round(v, 8)


def buchen(*, anbieter: str, art: str, modell: Optional[str] = None, anzahl: int = 1,
           tokens_ein: int = 0, tokens_aus: int = 0, zeichen: int = 0, minuten: float = 0.0,
           credits: float = 0.0, usd: float = 0.0, gegenwert_usd: float = 0.0,
           genauigkeit: str = "gezaehlt", aufrufer: Optional[str] = None,
           pfad: Optional[str] = None, jetzt: Optional[datetime] = None) -> None:
    """Eine Zeile ins Kostenbuch. Aufrufer und Pfad kommen aus dem Kontext des
    laufenden Aufrufs, sofern nicht ausdruecklich genannt."""
    try:
        k = KONTEXT.get() or {}
        t = (jetzt or datetime.now(WIEN)).astimezone(WIEN)
        zeile = {
            "ts": t.isoformat(timespec="seconds"),
            "pfad": pfad or k.get("pfad"),
            "aufrufer": normalisieren(aufrufer or k.get("aufrufer")),
            "quelle": k.get("quelle") if not aufrufer else (k.get("quelle") or "angabe"),
            "anbieter": str(anbieter)[:40],
            "art": art if art in ARTEN else "geld",
            "modell": (str(modell)[:80] if modell else None),
            "anzahl": _zahl(anzahl, ganz=True),
            "tokens_ein": _zahl(tokens_ein, ganz=True),
            "tokens_aus": _zahl(tokens_aus, ganz=True),
            "zeichen": _zahl(zeichen, ganz=True),
            "minuten": _zahl(minuten),
            "credits": _zahl(credits),
            "usd": _zahl(usd),
            "gegenwert_usd": _zahl(gegenwert_usd),
            "genauigkeit": genauigkeit if genauigkeit in GENAUIGKEITEN else "gezaehlt",
        }
        d = verzeichnis()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{t:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:                                    # noqa: BLE001
        logger.warning("Kostenbuch: nicht gebucht (%s)", type(e).__name__)


# --- Einordnung -------------------------------------------------------------

def normalisieren(aufrufer: Optional[str]) -> str:
    a = str(aufrufer or "").strip()[:64]
    if a.startswith("agent:"):
        a = a[len("agent:"):]
    return a or "unbekannt"


#: Aufrufer, deren Arbeit zum Kleinhirn/Mini-Hirn der cloud-api gehoert.
_KLEINHIRN = {"cloud-api-jev", "cloud-api-kleinhirn", "Cloud", "Cloud-time", "CloudV2-time"}

#: Feste Schluessel der Wege (Vertrag mit CloudV2, #4127); die Anzeigenamen setzt das Portal.
#: `postfach_auswertung` ist reserviert (heute ordnet kein Dienst Mails per KI ein, Comm 26.11.).
WEGE = ("tts", "stt", "verarbeitung", "prompt_agent", "conversation", "postfach_auswertung",
        "bild_video", "kleinhirn", "jev_decisions", "realtime", "text_direkt", "rest")


def weg(pfad: Optional[str], aufrufer: Optional[str]) -> str:
    """Weg eines Aufrufs aus Pfad und Aufrufer. Aufrufer vor Pfad, weil Alex nach
    Diensten fragt (Prompt Agent, Conversation), die mehrere Pfade nutzen."""
    p = str(pfad or "")
    a = normalisieren(aufrufer)
    if a == "service:issue-api" and p.startswith(("/ai/verarbeitung", "/ai/deepseek", "/ai/claude")):
        return "prompt_agent"
    if a == "service:conversation-api":
        return "conversation"
    if a in _KLEINHIRN:
        return "kleinhirn"
    if p.startswith("/ai/transcribe"):
        return "stt"
    if p.startswith(("/ai/tts", "/ai/generate_speech", "/ai/narrate", "/ai/genmusic", "/ai/music",
                     "/ai/gensfx", "/ai/sfx", "/ai/audio")):
        return "tts"
    if p.startswith(("/ai/kette", "/ai/verarbeitung")):
        return "verarbeitung"
    if p.startswith(("/ai/genimage", "/ai/genvideo", "/ai/gen3d", "/ai/image", "/ai/video")):
        return "bild_video"
    if p.startswith(("/ai/jev", "/ai/decisions")):
        return "jev_decisions"
    if p.startswith("/ai/realtime"):
        return "realtime"
    if p.startswith(("/ai/chatgpt", "/ai/gemini", "/ai/claude", "/ai/deepseek", "/ai/m3", "/ai/grok",
                     "/ai/openai", "/ai/agy", "/ai/extern")):
        return "text_direkt"
    return "rest"


#: Keine Last, kein Geld: Meldungen von Google und Statusaufrufe.
_OHNE_LAST = ("/ai/gemini/gcp-budget-webhook", "/ai/gemini/cost-status", "/ai/realtime/session/heartbeat",
              "/ai/status/")

#: Alle Zaehler rechnen in USD; EUR nur mit Kurs als Feld (das Portal rechnet nicht um).
WAEHRUNG = "USD"


# --- Lesen ------------------------------------------------------------------

def _tage(von: date, bis: date) -> List[date]:
    return [von + timedelta(days=i) for i in range((bis - von).days + 1)]


def _jsonl(pfad: Path) -> Iterable[dict]:
    try:
        with open(pfad, encoding="utf-8") as f:
            for zeile in f:
                try:
                    d = json.loads(zeile)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    yield d
    except FileNotFoundError:
        return


def _aufrufer_aus_protokoll(d: dict) -> str:
    """IP-Herkunft ist 'ANONYM', nie verteilt."""
    if d.get("quelle") == "ip":
        return "ANONYM"
    return normalisieren(d.get("caller"))


def _erster_tag(verz: Path) -> Optional[date]:
    try:
        namen = sorted(p.name[:10] for p in verz.glob("*.jsonl"))
        return date.fromisoformat(namen[0]) if namen else None
    except (OSError, ValueError):
        return None


_MENGEN = ("buchungen", "stueck", "tokens_ein", "tokens_aus", "zeichen", "minuten", "credits",
           "betrag", "gegenwert_usd")


def _null() -> dict:
    return {k: 0 for k in _MENGEN}


def _runden(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        out[k] = round(v, 6) if isinstance(v, float) else v
    return out


def _env_zahl(name: str, vorgabe: Optional[float] = None) -> Optional[float]:
    roh = (os.getenv(name) or "").strip()
    if not roh:
        return vorgabe
    try:
        return float(roh)
    except ValueError:
        return vorgabe


def deckel(anbieter: str) -> Optional[dict]:
    """Bekannte Grenzen je Topf, aus derselben Umgebung wie die Zaehler."""
    tabelle = {
        "typesafe_jev": ("JEV_DAILY_MAX_USD", 2.0, "tag", "USD"),
        "openai_decisions": ("DECISIONS_DAILY_MAX_USD", 2.0, "tag", "USD"),
        "openai_text": ("OPENAI_TEXT_DAILY_MAX_USD", None, "tag", "USD"),
        "deepseek": ("DEEPSEEK_MONTHLY_BUDGET_EUR", 25.0, "monat", "EUR"),
        "minimax": ("MINIMAX_MONTHLY_BUDGET_EUR", 25.0, "monat", "EUR"),
        "openai_bild": ("OPENAI_MONTHLY_BUDGET_EUR", 50.0, "monat", "EUR"),
        "openai_realtime": ("OPENAI_REALTIME_MONTHLY_BUDGET_EUR", None, "monat", "EUR"),
        "elevenlabs": ("ELEVENLABS_MONTHLY_CHAR_CAP", 10000.0, "monat", "zeichen"),
        "agy": ("KETTE_AGY_TAG_MAX", 500.0, "tag", "aufrufe"),
    }
    if anbieter not in tabelle:
        return None
    env, vorgabe, zeitraum, einheit = tabelle[anbieter]
    wert = _env_zahl(env, vorgabe)
    if wert is None:
        return None
    return {"wert": wert, "einheit": einheit, "zeitraum": zeitraum}


def bericht(von: date, bis: date, gruppe: str = "weg", takt: Optional[str] = None) -> dict:
    """Bericht ueber [von, bis] (Wiener Kalendertage, beide einschliesslich).

    `zeilen`: je Gruppe (weg | aufrufer | anbieter | tag), Anbieter und Art eine
    Zeile mit Mengen aus dem Kostenbuch. Geld steht nur in Zeilen mit art=geld
    (`betrag` + `waehrung`); Abo-Zeilen tragen nur Mengen, nie eine Mischsumme.
    `aufrufe`: je Gruppe die Aufrufe aus dem Aufruferprotokoll (ok und alle),
    auch die ohne Buchung (Fehler, Abo-Aufrufe ohne Tokenangabe).
    `takt=tag` haengt an jede Zeile die Tagesreihe."""
    if gruppe not in GRUPPEN:
        raise ValueError(f"gruppe muss eine von {', '.join(GRUPPEN)} sein")
    if takt not in (None, "", "tag"):
        raise ValueError("takt nur 'tag'")
    if bis < von:
        raise ValueError("bis liegt vor von")
    if (bis - von).days + 1 > MAX_TAGE:
        raise ValueError(f"hoechstens {MAX_TAGE} Tage")
    mit_reihe = takt == "tag"

    def schluessel(tag: str, pfad, aufrufer, anbieter) -> str:
        if gruppe == "tag":
            return tag
        if gruppe == "aufrufer":
            return aufrufer
        if gruppe == "anbieter":
            return anbieter or "(ohne Buchung)"
        return weg(pfad, aufrufer)

    zeilen: Dict[tuple, dict] = {}
    aufrufe: Dict[str, dict] = {}
    toepfe: Dict[str, dict] = {}
    erfasst_bis: Optional[str] = None

    for t in _tage(von, bis):
        tag = t.isoformat()
        for d in _jsonl(aufruferverzeichnis() / f"{tag}.jsonl"):
            pfad = str(d.get("pfad") or "")
            if pfad.startswith(_OHNE_LAST) or gruppe == "anbieter":
                continue
            k = schluessel(tag, pfad, _aufrufer_aus_protokoll(d), None)
            z = aufrufe.setdefault(k, {"schluessel": k, "aufrufe": 0, "aufrufe_ok": 0, "_reihe": {}})
            ok = 0
            try:
                ok = 1 if 200 <= int(d.get("status") or 0) < 300 else 0
            except (TypeError, ValueError):
                pass
            z["aufrufe"] += 1
            z["aufrufe_ok"] += ok
            r = z["_reihe"].setdefault(tag, [0, 0])
            r[0] += 1
            r[1] += ok
        for d in _jsonl(verzeichnis() / f"{tag}.jsonl"):
            a = "ANONYM" if d.get("quelle") == "ip" else normalisieren(d.get("aufrufer"))
            art = d.get("art") if d.get("art") in ARTEN else "geld"
            anb = str(d.get("anbieter") or "?")
            k = schluessel(tag, d.get("pfad"), a, anb)
            z = zeilen.setdefault((k, anb, art), {**_null(), "_geschaetzt": False, "_reihe": {}})
            m = {"buchungen": 1, "stueck": _zahl(d.get("anzahl"), ganz=True) or 1,
                 "tokens_ein": _zahl(d.get("tokens_ein"), ganz=True),
                 "tokens_aus": _zahl(d.get("tokens_aus"), ganz=True),
                 "zeichen": _zahl(d.get("zeichen"), ganz=True), "minuten": _zahl(d.get("minuten")),
                 "credits": _zahl(d.get("credits")),
                 "betrag": _zahl(d.get("usd")) if art == "geld" else 0.0,
                 "gegenwert_usd": _zahl(d.get("gegenwert_usd"))}
            for f, v in m.items():
                z[f] += v
            if mit_reihe:
                r = z["_reihe"].setdefault(tag, _null())
                for f, v in m.items():
                    r[f] += v
            if d.get("genauigkeit") == "geschaetzt":
                z["_geschaetzt"] = True
            if art == "geld":
                tp = toepfe.setdefault(anb, {"anbieter": anb, "betrag": 0.0, "waehrung": WAEHRUNG})
                tp["betrag"] += m["betrag"]
            ts = d.get("ts")
            if isinstance(ts, str) and (erfasst_bis is None or ts > erfasst_bis):
                erfasst_bis = ts

    feld = {"weg": "weg", "aufrufer": "aufrufer", "anbieter": "anbieter_gruppe", "tag": "tag"}[gruppe]
    aus = []
    for (k, anb, art), z in sorted(zeilen.items(), key=lambda kv: (-kv[1]["betrag"], kv[0])):
        zeile = {feld: k, "anbieter": anb, "art": art,
                 **_runden({f: z[f] for f in _MENGEN if f not in ("betrag", "gegenwert_usd")}),
                 "genauigkeit": "geschaetzt" if z["_geschaetzt"] else "gezaehlt"}
        if feld == "anbieter_gruppe":
            zeile.pop("anbieter_gruppe")
        if art == "geld":
            zeile.update(betrag=round(z["betrag"], 6), waehrung=WAEHRUNG)
        else:
            # API-Gegenwert eines Abo-Wegs: nie bezahlt, nie mit Geld summieren.
            zeile["gegenwert_usd"] = round(z["gegenwert_usd"], 4)
        if mit_reihe:
            zeile["reihe"] = [{"tag": tg, **_runden(v if art == "geld" else {f: x for f, x in v.items() if f != "betrag"})}
                              for tg, v in sorted(z["_reihe"].items())]
        aus.append(zeile)

    aufruf_aus = []
    for k, z in sorted(aufrufe.items(), key=lambda kv: -kv[1]["aufrufe"]):
        e = {feld: k, "aufrufe": z["aufrufe"], "aufrufe_ok": z["aufrufe_ok"]}
        if mit_reihe:
            e["reihe"] = [{"tag": tg, "aufrufe": v[0], "aufrufe_ok": v[1]} for tg, v in sorted(z["_reihe"].items())]
        aufruf_aus.append(e)

    topf_aus = []
    for anb, tp in sorted(toepfe.items(), key=lambda kv: -kv[1]["betrag"]):
        e = {**tp, "betrag": round(tp["betrag"], 6)}
        dk = deckel(anb)
        if dk:
            e["deckel"] = dk
        topf_aus.append(e)
    summe = round(sum(tp["betrag"] for tp in toepfe.values()), 6)

    gesamt_aufrufe = sum(z["aufrufe"] for z in aufrufe.values()) or 0
    anonym = (aufrufe.get("ANONYM") or {}).get("aufrufe", 0) if gruppe == "aufrufer" else None
    luecken = []
    start = _erster_tag(verzeichnis())
    if start is None or start > von:
        luecken.append({"art": "vor_kostenbuch",
                        "bis": (start - timedelta(days=1)).isoformat() if start else bis.isoformat(),
                        "text": "Vor dem Start des Kostenbuchs (#4127) gibt es nur Aufrufe, keine Mengen und "
                                "kein Geld je Aufruf. Siehe monatszaehler und den Bericht an #4127."})
    a_start = _erster_tag(aufruferverzeichnis())
    if a_start is None or a_start > von:
        luecken.append({"art": "vor_aufruferprotokoll",
                        "bis": (a_start - timedelta(days=1)).isoformat() if a_start else bis.isoformat(),
                        "text": "Vor dem 30.10.2026 gibt es kein Aufruferprotokoll."})
    if anonym:
        luecken.append({"art": "ohne_kennung", "text": "Aufrufe ohne Kennung (ANONYM), nicht verteilt.",
                        "anteil_aufrufe": round(anonym / gesamt_aufrufe, 4) if gesamt_aufrufe else None})
    luecken.append({"art": "an_aiapi_vorbei", "anteil": None,
                    "text": "Dienste mit eigenem Schluessel und die Abos der Agenten (Claude, Codex) stehen hier "
                            "nicht. Kling und die gesperrte Google-API werden nicht gebucht."})
    luecken.append({"art": "geschaetzt", "anteil": None,
                    "text": "CapSolver je Flow-Erzeugung (0,3 Cent) ist geschaetzt; die Zeilen tragen "
                            "genauigkeit=geschaetzt."})
    return {
        "von": von.isoformat(), "bis": bis.isoformat(), "gruppe": gruppe, "takt": "tag" if mit_reihe else None,
        "stand": datetime.now(WIEN).isoformat(timespec="seconds"),
        "erfasst_bis": erfasst_bis,
        "kostenbuch_seit": start.isoformat() if start else None,
        "zeilen": aus,
        "aufrufe": aufruf_aus,
        "toepfe": topf_aus,
        "summe": {"betrag": summe, "waehrung": WAEHRUNG, "eur": round(summe / EUR_USD, 6),
                  "kurs_usd_je_eur": EUR_USD},
        "monatszaehler": monatszaehler(von, bis),
        "luecken": luecken,
    }


#: Monatszaehler je Anbieter: Datei-Praefix -> (anbieter, art)
_MONATSZAEHLER = {
    "openai_usage": ("openai_bild", "geld"),
    "openai_realtime_usage": ("openai_realtime", "geld"),
    "deepseek_usage": ("deepseek", "geld"),
    "minimax_usage": ("minimax", "geld"),
    "elevenlabs_usage": ("elevenlabs", "abo"),
    "codex_usage": ("codex", "abo"),
    "gemini_cli_usage": ("agy", "abo"),
    "claude_usage": ("claude_cli", "abo"),
}


def _daten_dir() -> Path:
    return Path(os.getenv("VERBRAUCH_BASIS_DIR") or "/var/lib/api-ai")


def monatszaehler(von: date, bis: date) -> List[dict]:
    """Die Monatssummen der bestehenden Zaehler fuer die Monate, die der Zeitraum
    beruehrt. Ganze Monate, nicht auf den Zeitraum gekuerzt (`monat` sagt es)."""
    monate = sorted({f"{t:%Y-%m}" for t in _tage(von, bis)})
    out = []
    for m in monate:
        for praefix, (anbieter, art) in _MONATSZAEHLER.items():
            p = _daten_dir() / f"{praefix}_{m}.json"
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            usd = _zahl(d.get("total_cost_usd"))
            out.append({
                "monat": m, "anbieter": anbieter, "art": art,
                "anfragen": d.get("request_count") or d.get("tts_calls"),
                "tokens_ein": d.get("total_input_tokens"), "tokens_aus": d.get("total_output_tokens"),
                "zeichen": d.get("chars_total"),
                **({"betrag": usd, "waehrung": WAEHRUNG} if art == "geld" else {"gegenwert_usd": usd}),
            })
    return out
