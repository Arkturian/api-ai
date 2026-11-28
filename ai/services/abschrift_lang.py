"""Abschrift langer Sendungen mit Wortankern und Wellenform.

Projekt 32 „KI-Schnitt für Radiosendungen", Paket A2. Vertrag
`radio-schnitt-v2` (Abschrift unverändert seit v1) (Content-Beitrag 6630, bestätigt von Story-Codex am
28.11.2026, Stand v35). Was hier steht, folgt dem Vertrag wörtlich:

- Wörter und Zeiten aus `whisper-1` (verbose_json, word), Sprecher aus
  `gpt-4o-transcribe-diarize`, Zuordnung je Wort nach größter Überdeckung.
- Lange Aufnahmen in Stücke mit Überlappung; an der Mitte der Überlappung
  wird geschnitten, so gibt es an den Nähten weder Lücke noch Doppel.
- Zeiten in ganzen Millisekunden, streng monoton.
- `abschrift_fp` = sha256 über `[[i, wort, start_ms, ende_ms], …]`
  (NFC, ohne Leerzeichen); Testvektor im Vertrag und im Test.
- Peaks: je Fenster `[k·sr/50, (k+1)·sr/50)` ein Paar int8 min/max,
  `werte = ceil(samples · 50 / sr)`, letztes Teilfenster mit.

Die reinen Funktionen hier rufen keinen Anbieter auf; der Ablauf mit
Anbieter und Storage steht in `ai/routes/abschrift_routes.py`.
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import unicodedata
from typing import Iterable, Optional

VERTRAG = "radio-schnitt-v2"
PEAKS_JE_SEKUNDE = 50

# Sprechertrennung schafft höchstens 1400 s je Aufruf; Abstand zur Grenze.
STUECK_MS = int(os.getenv("ABSCHRIFT_STUECK_S", "1320")) * 1000
UEBERLAPP_MS = int(os.getenv("ABSCHRIFT_UEBERLAPP_S", "30")) * 1000

# Ein Vorgabetext mit Fülllauten hält whisper-1 eher dazu an, „äh" mitzuschreiben
# (A1 misst, wie viel das bringt). Kein Inhalt, nur Stil.
FUELL_VORGABE = "Ähm, also, äh, ich meine, hm, ja. Äh, das ist, ähm, so."


# ── Fingerprint ──────────────────────────────────────────────────────────────

def fingerprint(woerter: Iterable[dict]) -> str:
    folge = [[int(w["i"]), unicodedata.normalize("NFC", str(w["wort"])), int(w["start_ms"]), int(w["ende_ms"])]
             for w in woerter]
    roh = json.dumps(folge, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(roh.encode("utf-8")).hexdigest()


# ── Stücke ───────────────────────────────────────────────────────────────────

def stuecke(dauer_ms: int, stueck_ms: int = None, ueberlapp_ms: int = None) -> list[tuple[int, int]]:
    """[(von_ms, bis_ms)] über die ganze Aufnahme, benachbarte Stücke überlappen."""
    stueck_ms = stueck_ms or STUECK_MS
    ueberlapp_ms = UEBERLAPP_MS if ueberlapp_ms is None else ueberlapp_ms
    if dauer_ms <= 0:
        return []
    if dauer_ms <= stueck_ms:
        return [(0, dauer_ms)]
    schritt = stueck_ms - ueberlapp_ms
    out, von = [], 0
    while True:
        bis = min(von + stueck_ms, dauer_ms)
        out.append((von, bis))
        if bis >= dauer_ms:
            return out
        von += schritt


def _ms(sekunden) -> int:
    return int(round(float(sekunden) * 1000))


def woerter_zusammenfuehren(teile: list[dict]) -> list[dict]:
    """teile: [{"von_ms", "bis_ms", "words": [{"word", "start", "end"} …]}] mit
    Zeiten relativ zum Stück (Sekunden, wie whisper sie liefert).

    Jedes Stück liefert die Wörter, deren Mitte vor der Naht liegt; die Naht
    ist die Mitte der Überlappung mit dem nächsten Stück. Danach fortlaufend
    nummeriert und streng monoton gemacht. Ein Wort, dessen Zeit dabei
    gestaucht werden musste, trägt `sicher: False`."""
    roh: list[dict] = []
    for k, t in enumerate(teile):
        links = 0 if k == 0 else (teile[k - 1]["bis_ms"] + t["von_ms"]) // 2
        rechts = None if k == len(teile) - 1 else (t["bis_ms"] + teile[k + 1]["von_ms"]) // 2
        for w in t.get("words") or []:
            text = unicodedata.normalize("NFC", str(w.get("word") or "").strip())
            if not text:
                continue
            s = t["von_ms"] + _ms(w["start"])
            e = t["von_ms"] + _ms(w["end"])
            mitte = (s + e) // 2
            if mitte < links or (rechts is not None and mitte >= rechts):
                continue
            roh.append({"wort": text, "start_ms": s, "ende_ms": e})
    roh.sort(key=lambda w: (w["start_ms"], w["ende_ms"]))
    out, letztes_ende = [], 0
    for w in roh:
        sicher = True
        s, e = w["start_ms"], w["ende_ms"]
        if s < letztes_ende:
            s, sicher = letztes_ende, False
        if e < s:
            e, sicher = s, False
        if e == s:
            sicher = False
        out.append({"i": len(out), "wort": w["wort"], "start_ms": s, "ende_ms": e, "sicher": sicher})
        letztes_ende = e
    return out


# ── Sprecher ─────────────────────────────────────────────────────────────────

def segmente_aus_diarize(teile: list[dict]) -> list[dict]:
    """teile: [{"von_ms", "bis_ms", "segments": [{"speaker", "start", "end"} …]}]
    → absolute Segmente [{"sprecher", "von_ms", "bis_ms"}], an den Nähten wie
    bei den Wörtern begrenzt."""
    # Etiketten gelten nur innerhalb eines Stücks: dasselbe „A" kann im
    # nächsten Stück ein anderer Mensch sein, und ein Sprecher bekommt dort
    # oft ein neues Etikett (Langtest 28.11.: ein Erzähler, drei Sprecher).
    # Deshalb je Stück eigene Etiketten und Abgleich über die Überlappung.
    roh: list[list[dict]] = []
    for k, t in enumerate(teile):
        roh.append([{"sprecher": f"{k}:{s.get('speaker') or '?'}",
                     "von_ms": t["von_ms"] + _ms(s["start"]), "bis_ms": t["von_ms"] + _ms(s["end"])}
                    for s in t.get("segments") or []])
    zuordnung: dict[str, str] = {}
    for k in range(len(teile)):
        if k == 0:
            for s in roh[0]:
                zuordnung.setdefault(s["sprecher"], s["sprecher"])
            continue
        lo, hi = teile[k]["von_ms"], teile[k - 1]["bis_ms"]
        deckung: dict[tuple, int] = {}
        for neu in roh[k]:
            for alt in roh[k - 1]:
                von, bis = max(lo, neu["von_ms"], alt["von_ms"]), min(hi, neu["bis_ms"], alt["bis_ms"])
                if bis > von:
                    schl = (neu["sprecher"], zuordnung.get(alt["sprecher"], alt["sprecher"]))
                    deckung[schl] = deckung.get(schl, 0) + (bis - von)
        vergeben: set = set()
        for (neu, alt), _d in sorted(deckung.items(), key=lambda x: -x[1]):
            if neu not in zuordnung and alt not in vergeben:
                zuordnung[neu] = alt
                vergeben.add(alt)
        for s in roh[k]:
            zuordnung.setdefault(s["sprecher"], s["sprecher"])
    out = []
    for k, t in enumerate(teile):
        links = 0 if k == 0 else (teile[k - 1]["bis_ms"] + t["von_ms"]) // 2
        rechts = None if k == len(teile) - 1 else (t["bis_ms"] + teile[k + 1]["von_ms"]) // 2
        for s in roh[k]:
            a, b = max(s["von_ms"], links), s["bis_ms"]
            if rechts is not None:
                b = min(b, rechts)
            if b > a:
                out.append({"sprecher": zuordnung[s["sprecher"]], "von_ms": a, "bis_ms": b})
    out.sort(key=lambda s: (s["von_ms"], s["bis_ms"]))
    return out


def _ueberlappungen(segmente: list[dict]) -> list[dict]:
    """Bereiche, in denen Segmente verschiedener Sprecher gleichzeitig laufen."""
    bereiche = []
    for i, a in enumerate(segmente):
        for b in segmente[i + 1:]:
            if b["von_ms"] >= a["bis_ms"]:
                break
            if b["sprecher"] != a["sprecher"]:
                von, bis = max(a["von_ms"], b["von_ms"]), min(a["bis_ms"], b["bis_ms"])
                if bis > von:
                    bereiche.append([von, bis])
    bereiche.sort()
    zus: list[list[int]] = []
    for von, bis in bereiche:
        if zus and von <= zus[-1][1]:
            zus[-1][1] = max(zus[-1][1], bis)
        else:
            zus.append([von, bis])
    return [{"von_ms": a, "bis_ms": b, "grund": "ueberlappung"} for a, b in zus]


def sprecher_zuordnen(woerter: list[dict], segmente: list[dict],
                      namen: Optional[dict] = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Gibt (woerter mit `sprecher`, sprecher-Liste, unsicher-Liste) zurück.

    Sprecher-IDs werden in der Reihenfolge des ersten Auftretens S1, S2 …
    `namen` bildet Diarize-Etiketten auf Namen ab (bekannte Sprecher)."""
    namen = namen or {}
    ids: dict[str, str] = {}
    for s in segmente:
        if s["sprecher"] not in ids:
            ids[s["sprecher"]] = f"S{len(ids) + 1}"
    unsicher = _ueberlappungen(segmente)
    j0 = 0
    for w in woerter:
        while j0 < len(segmente) and segmente[j0]["bis_ms"] <= w["start_ms"]:
            j0 += 1
        bester, beste = None, 0
        j = j0
        while j < len(segmente) and segmente[j]["von_ms"] < max(w["ende_ms"], w["start_ms"] + 1):
            s = segmente[j]
            d = min(s["bis_ms"], max(w["ende_ms"], w["start_ms"] + 1)) - max(s["von_ms"], w["start_ms"])
            if d > beste:
                bester, beste = s["sprecher"], d
            j += 1
        w["sprecher"] = ids.get(bester) if bester is not None else None
        if w["sprecher"] is None:
            w["sicher"] = False
        for u in unsicher:
            if u["von_ms"] < w["ende_ms"] and w["start_ms"] < u["bis_ms"]:
                w["sicher"] = False
                break
    # Wörter mit gestauchter oder fehlender Zeit als eigene Bereiche
    for w in woerter:
        if w["ende_ms"] == w["start_ms"]:
            unsicher.append({"von_ms": w["start_ms"], "bis_ms": w["ende_ms"], "grund": "zeit_unsicher"})
    unsicher.sort(key=lambda u: (u["von_ms"], u["bis_ms"]))
    sprecher = [{"id": sid, "name": namen.get(etikett.split(":", 1)[-1])} for etikett, sid in ids.items()]
    return woerter, sprecher, unsicher


# ── Wellenform ───────────────────────────────────────────────────────────────

def peaks_anzahl(samples: int, sample_rate: int) -> int:
    return -(-samples * PEAKS_JE_SEKUNDE // sample_rate)


def peaks(pcm_s16le_mono: bytes, sample_rate: int) -> tuple[bytes, int]:
    """Rohbytes int8 min,max je Fenster und die Zahl der Paare."""
    a = array.array("h")
    a.frombytes(pcm_s16le_mono[: len(pcm_s16le_mono) // 2 * 2])
    if array.array("h", [1]).tobytes() != b"\x01\x00":   # Grosses Endian: auf little drehen
        a.byteswap()
    n = len(a)
    werte = peaks_anzahl(n, sample_rate)
    out = bytearray(2 * werte)
    for k in range(werte):
        s = k * sample_rate // PEAKS_JE_SEKUNDE
        e = min(n, (k + 1) * sample_rate // PEAKS_JE_SEKUNDE)
        if e <= s:
            lo = hi = 0
        else:
            fenster = a[s:e]
            lo, hi = min(fenster), max(fenster)
        out[2 * k] = _int8(lo) & 0xFF
        out[2 * k + 1] = _int8(hi) & 0xFF
    return bytes(out), werte


def _int8(v: int) -> int:
    """s16 → int8 mit Abrundung zur Null hin; −32768 → −128, 32767 → 127."""
    return max(-128, min(127, int(v / 256)))


# ── Anfrage-Identität ────────────────────────────────────────────────────────

def anfrage_hash(rumpf: dict) -> str:
    """Fachlicher Rumpf: ohne `request_id` und ohne die signierte Adresse.
    Die Quelle gehört über `quelle_storage_id` dazu; den sha256 der Bytes
    vergleicht der Lauf selbst (409 quelle_geaendert)."""
    kern = {k: v for k, v in rumpf.items() if k not in ("request_id", "quelle_url", "confirm_api_billing")}
    roh = json.dumps(kern, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()


def abschrift_bauen(quelle: dict, woerter: list[dict], sprecher: list[dict], unsicher: list[dict],
                    peaks_info: dict, herkunft: dict) -> dict:
    return {
        "vertrag": VERTRAG,
        "quelle": quelle,
        "sprecher": sprecher,
        "woerter": [{"i": w["i"], "wort": w["wort"], "start_ms": w["start_ms"], "ende_ms": w["ende_ms"],
                     "sprecher": w.get("sprecher"), "sicher": bool(w.get("sicher", True))} for w in woerter],
        "unsicher": unsicher,
        "abschrift_fp": fingerprint(woerter),
        "peaks": peaks_info,
        "herkunft": herkunft,
    }
