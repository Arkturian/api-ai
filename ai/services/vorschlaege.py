"""Schnittvorschläge aus einer Abschrift (Projekt 32, Paket A3: deterministische Filter).

Vertrag `radio-schnitt-v2`, Content-Beitrag 6630, Abschnitt 2 und „v2“.
Hier stehen nur die Filter ohne KI:

- `wiederholung`: dasselbe Wort direkt hintereinander („wir wir“) und ein
  abgebrochener Wortanfang mit Bindestrich („Wi- Wir“). Entfernt wird die
  erste Fassung. Grammatisch mögliche Doppelungen (die die, das das …) lässt
  A3 aus, sie gehen mit Sicherheit an A4.
- `pause`: Stille zwischen zwei Wörtern, gemessen in den Peaks (20 ms), ab
  `ab_ms` als `kuerzen` auf `ziel_ms`.

Regeln aus dem Vertrag: Wortanker statt Sekunden, keine Grenze in `unsicher`
und an Wörtern mit `sicher: false`, nie zwei Stufe-1-Einträge an derselben
Lücke (sonst zusammengelegt mit `teile`), stabile `id`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Optional

VERTRAG = "radio-schnitt-v2"

# Doppelungen, die im Deutschen auch richtig sein können („Leute, die die Wahrheit …“).
GRAMMATISCH_MOEGLICH = frozenset({"der", "die", "das", "den", "dem", "des", "sie", "es", "er", "wir", "ihr"}) - {"wir"}

_SATZZEICHEN = re.compile(r"[\s.,;:!?…\"'„“”‚‘’()\[\]–—]+")


def _norm(wort: str) -> str:
    return _SATZZEICHEN.sub("", unicodedata.normalize("NFC", wort)).lower()


def eintrag_id(quelle_sha256: str, abschrift_fp: str, e: dict) -> str:
    kern = {k: e.get(k) for k in ("filter", "art", "von", "bis", "anker", "ziel_ms", "verweis")}
    kern["quelle_sha256"], kern["abschrift_fp"] = quelle_sha256, abschrift_fp
    roh = json.dumps(kern, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "e-" + hashlib.sha256(unicodedata.normalize("NFC", roh).encode("utf-8")).hexdigest()[:16]


def _in_unsicher(a_ms: int, b_ms: int, unsicher: list) -> bool:
    return any(u["von_ms"] <= b_ms and a_ms <= u["bis_ms"] for u in unsicher)


def _wortanker(w: dict) -> dict:
    return {"word_index": w["i"], "expected_word": w["wort"]}


def wiederholungen(woerter: list, unsicher: list) -> list[dict]:
    out = []
    for a, b in zip(woerter, woerter[1:]):
        if not (a.get("sicher", True) and b.get("sicher", True)):
            continue
        if _in_unsicher(a["start_ms"], b["ende_ms"], unsicher):
            continue
        na, nb = _norm(a["wort"]), _norm(b["wort"])
        if not na:
            continue
        stamm = na.rstrip("-")
        abgebrochen = a["wort"].rstrip().endswith(("-", "–")) and len(stamm) >= 1 and nb.startswith(stamm) and nb != stamm
        gleich = na == nb and na not in GRAMMATISCH_MOEGLICH
        if not (gleich or abgebrochen):
            continue
        out.append({
            "art": "raus", "filter": "wiederholung",
            "von": _wortanker(a), "bis": _wortanker(a),
            "begruendung": (f"„{a['wort']} {b['wort']}“, die erste Fassung ist abgebrochen" if abgebrochen
                            else f"„{a['wort']} {b['wort']}“, Wort doppelt gesprochen"),
            "sicherheit": None, "von_s": round(a["start_ms"] / 1000, 3), "bis_s": round(a["ende_ms"] / 1000, 3),
        })
    return out


def _stille_folgen(peaks: bytes, von_ms: int, bis_ms: int, schwelle: int) -> int:
    """Längste zusammenhängende Stille in [von_ms, bis_ms) in ms (Fenster je 20 ms)."""
    k0, k1 = von_ms // 20, min(len(peaks) // 2, -(-bis_ms // 20))
    best = lauf = 0
    for k in range(k0, k1):
        lo = peaks[2 * k] - 256 if peaks[2 * k] > 127 else peaks[2 * k]
        hi = peaks[2 * k + 1] - 256 if peaks[2 * k + 1] > 127 else peaks[2 * k + 1]
        if max(abs(lo), abs(hi)) <= schwelle:
            lauf += 1
            best = max(best, lauf)
        else:
            lauf = 0
    return best * 20


def schwelle_int8(db: float) -> int:
    """dBFS → Schwelle im int8-Raster der Peaks (Vollaussteuerung 127)."""
    return max(0, int(math.floor(127 * 10 ** (db / 20))))


def pausen(woerter: list, unsicher: list, peaks: bytes, ab_ms: int = 800, ziel_ms: int = 400,
           schwelle_db: float = -40.0) -> list[dict]:
    schwelle = schwelle_int8(schwelle_db)
    out = []
    for a, b in zip(woerter, woerter[1:]):
        if not (a.get("sicher", True) and b.get("sicher", True)):
            continue
        if b["start_ms"] - a["ende_ms"] < ab_ms or _in_unsicher(a["ende_ms"], b["start_ms"], unsicher):
            continue
        still = _stille_folgen(peaks, a["ende_ms"], b["start_ms"], schwelle)
        if still < ab_ms:
            continue
        out.append({
            "art": "kuerzen", "filter": "pause",
            "anker": {"nach_word_index": a["i"], "vor_word_index": b["i"],
                      "expected_nach": a["wort"], "expected_vor": b["wort"]},
            "luecke_ms": still, "ziel_ms": ziel_ms,
            "begruendung": f"Denkpause {still / 1000:.1f} s, gekürzt auf {ziel_ms / 1000:.1f} s".replace(".", ","),
            "sicherheit": None,
        })
    return out


def _luecken(e: dict) -> set:
    """Welche Lücken (als (nach, vor)) berührt ein Eintrag?"""
    if e["art"] == "kuerzen":
        a = e["anker"]
        return {(a["nach_word_index"], a["vor_word_index"])}
    i0, i1 = e["von"]["word_index"], e["bis"]["word_index"]
    return {(i0 - 1, i0), (i1, i1 + 1)}


def zusammenlegen(eintraege: list) -> list[dict]:
    """Vertrag v2 Punkt 3: nie zwei Stufe-1-Einträge an derselben Lücke. Ein
    Wort-`raus` übernimmt die Pause daneben als Teil (`teile`, `anker`, `ziel_ms`)."""
    raus = [e for e in eintraege if e["art"] == "raus"]
    kurz = [e for e in eintraege if e["art"] == "kuerzen"]
    belegt: dict = {}
    for e in raus:
        for l in _luecken(e):
            belegt.setdefault(l, e)
    uebrig = []
    for p in kurz:
        l = next(iter(_luecken(p)))
        haupt = belegt.get(l)
        if haupt is None:
            uebrig.append(p)
            continue
        haupt.setdefault("teile", [{"filter": haupt["filter"], "sicherheit": haupt.get("sicherheit")}])
        haupt["teile"].append({"filter": "pause", "sicherheit": p.get("sicherheit")})
        haupt["anker"], haupt["ziel_ms"], haupt["luecke_ms"] = p["anker"], p["ziel_ms"], p["luecke_ms"]
        # Ersatzschnitt (q-c2fefb7aeac7): gilt nicht jeder Teil automatisch, kürzt
        # story-api nur die Stille; das Wort bleibt hörbar.
        haupt["ohne_enthaltene"] = {"art": "kuerzen", "filter": "pause", "anker": p["anker"],
                                    "luecke_ms": p["luecke_ms"], "ziel_ms": p["ziel_ms"]}
        haupt["begruendung"] += f"; dazu {p['begruendung'][0].lower()}{p['begruendung'][1:]}"
    # zwei Wort-`raus` direkt hintereinander (dreifaches Wort): zum Bereich zusammenziehen
    raus.sort(key=lambda e: e["von"]["word_index"])
    zus: list = []
    for e in raus:
        if zus and e["von"]["word_index"] == zus[-1]["bis"]["word_index"] + 1 and "anker" not in zus[-1]:
            zus[-1]["bis"] = e["bis"]
            zus[-1]["bis_s"] = e["bis_s"]
            zus[-1]["begruendung"] = zus[-1]["begruendung"].split(",")[0] + ", mehrfach gesprochen"
            continue
        zus.append(e)
    return zus + uebrig


def erzeugen(abschrift: dict, peaks: bytes, abschrift_storage_id: Optional[int], filter_: dict) -> dict:
    woerter, unsicher = abschrift["woerter"], abschrift.get("unsicher") or []
    eintraege: list = []
    if filter_.get("wiederholung", True):
        eintraege += wiederholungen(woerter, unsicher)
    p = filter_.get("pause", {"ab_ms": 800, "ziel_ms": 400})
    if p:
        p = p if isinstance(p, dict) else {}
        eintraege += pausen(woerter, unsicher, peaks, ab_ms=int(p.get("ab_ms", 800)),
                            ziel_ms=int(p.get("ziel_ms", 400)), schwelle_db=float(p.get("schwelle_db", -40.0)))
    eintraege = zusammenlegen(eintraege)
    q, fp = abschrift["quelle"]["sha256"], abschrift["abschrift_fp"]
    for e in eintraege:
        e["id"] = eintrag_id(q, fp, e)
    eintraege.sort(key=lambda e: (e.get("von") or {}).get("word_index", (e.get("anker") or {}).get("nach_word_index", 0)))
    return {
        "vertrag": VERTRAG, "abschrift_storage_id": abschrift_storage_id, "abschrift_fp": fp, "quelle_sha256": q,
        "automatik_empfehlung": {"aeh": None, "fuellwort": None},
        "eintraege": eintraege,
    }


def datei_bytes(liste: dict) -> tuple[bytes, str]:
    """Die gespeicherten Bytes und ihr Fingerprint (Vertrag: SHA-256 der Dateibytes)."""
    roh = json.dumps(liste, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return roh, "sha256:" + hashlib.sha256(roh).hexdigest()
