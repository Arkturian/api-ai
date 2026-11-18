"""Pruefstand: Sichtbarkeit beim Hochladen automatisch vorschlagen (Cloud/Alex, 18.11.2026).

Die KI schlaegt nur zwischen `privat` und `intern` vor. `oeffentlich` vergibt
sie nie (Vorgabe Alex): das kommt nur aus einer ausdruecklichen Angabe oder
einer festen Regel. Entscheiden tut die Regel hier:

    intern  nur, wenn die KI `intern` waehlt UND P(intern) >= SCHWELLE,
    sonst   privat (Zweifel, Fehler, Zeitueberschreitung -> privat).

Sollwerte in tools/sichtbarkeit_faelle.json (erfundene Metadaten):
    privat  Kunden-, Vertrags-, Finanz-, Gesundheits- oder Personendaten,
            Anhaenge aus Mail/Messenger, Aufnahmen und Fotos von Menschen,
            Ausweise, Zugangsdaten, Logs, Backups, Exporte, unveroeffentlichte
            Kundenentwuerfe, Screenshots aus Browsern von Menschen, und
            alles ohne erkennbaren Zweck (generischer Name, kein Kontext).
    intern  Arbeitsmaterial der Agenten ohne Personen- oder Kundendaten:
            KI-erzeugte Bilder/Videos/Audio/3D, Diagramme, Doku, Testdateien,
            Vorschaubilder, eigene Designs.

Eingabe je Fall nur Metadaten (Dateiname, MIME, Groesse, Kontext, Sammlung,
hochladender Agent/Mensch). Keine Dateiinhalte.

Aufruf (auf arkserver, JWT aus dem Bot-Home, nie ausgegeben):
    python3 tools/sichtbarkeit_bench.py --engines jev,openai [--gemini 20] [--schwelle 0.8]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BASIS = "http://127.0.0.1:8000"
HELFER = "/home/alex/.cloud-api-bots/AiApi/mcp-auth-helper.sh"
FAELLE = Path(__file__).with_name("sichtbarkeit_faelle.json")

STUFEN = {
    "privat": ("Nur der hochladende Agent bzw. Mensch und ausdruecklich Freigegebene. Fuer alles, was "
               "Kunden-, Vertrags-, Finanz-, Gesundheits- oder Personendaten enthalten kann, Anhaenge aus "
               "Mail oder Messenger, Aufnahmen oder Fotos von Menschen, Ausweise, Zugangsdaten, Logs, "
               "Backups, Exporte, unveroeffentlichte Kundenentwuerfe, Screenshots aus dem Browser eines "
               "Menschen, und alles, dessen Zweck aus den Angaben nicht erkennbar ist."),
    "intern": ("Alle angemeldeten Agenten und Menschen des Mandanten. Nur fuer Arbeitsmaterial ohne "
               "Personen- oder Kundendaten: KI-erzeugte Bilder, Videos, Audio, Musik, 3D-Modelle, "
               "Diagramme, Dokumentation, Testdateien, Vorschaubilder, eigene Designs."),
}
FRAGE_NEUTRAL = ("Welche Sichtbarkeit passt fuer diese hochgeladene Datei? Beurteile nur die Angaben "
                 "(Dateiname, Typ, Groesse, Kontext, Sammlung, Hochlader); der Inhalt ist nicht bekannt.")
FRAGE = FRAGE_NEUTRAL + " Im Zweifel privat."


def token() -> str:
    out = json.loads(subprocess.check_output([HELFER]).decode())
    return out.get("Authorization") or out["authorization"]


def zustand(f: dict) -> str:
    return json.dumps({k: f.get(k) for k in ("datei", "mime", "bytes", "kontext", "sammlung", "von")},
                      ensure_ascii=False)


def post(pfad: str, body: dict, auth: str, timeout: float = 120) -> tuple:
    req = urllib.request.Request(BASIS + pfad, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": auth})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), (time.monotonic() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, (time.monotonic() - t0) * 1000, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:  # Zeitueberschreitung u. a.
        return None, (time.monotonic() - t0) * 1000, type(e).__name__


def frage_jev(f: dict, engine: str, auth: str, frage: str = FRAGE) -> dict:
    body = {"state": zustand(f),
            "questions": {"stufe": {"type": "choice", "instructions": frage, "criteria": STUFEN}}}
    d, ms, fehler = post(f"/ai/jev?engine={engine}", body, auth)
    if fehler or not d:
        return {"vorschlag": None, "p_intern": None, "ms": ms, "kosten": 0.0, "fehler": fehler}
    a = (d.get("answers") or d.get("result") or d).get("stufe") or {}
    p = a.get("probabilities") or {}
    return {"vorschlag": a.get("choice"), "p_intern": p.get("intern"), "ms": d.get("latency_ms") or ms,
            "kosten": float(d.get("cost_usd") or 0.0), "fehler": None}


def frage_gemini(f: dict, auth: str) -> dict:
    prompt = (FRAGE + "\n\nStufen:\n" + "\n".join(f"- {k}: {v}" for k, v in STUFEN.items()) +
              "\n\nAngaben: " + zustand(f) +
              '\n\nAntworte nur mit JSON: {"stufe": "privat"|"intern", "p_intern": Zahl 0..1}')
    d, ms, fehler = post("/ai/gemini", {"prompt": prompt, "max_tokens": 200, "temperature": 0}, auth, timeout=180)
    if fehler or not d:
        return {"vorschlag": None, "p_intern": None, "ms": ms, "kosten": 0.0, "fehler": fehler}
    text = str(d.get("response") or d.get("content") or d.get("text") or "")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except ValueError:
        j = {}
    return {"vorschlag": j.get("stufe"), "p_intern": j.get("p_intern"), "ms": ms, "kosten": 0.0,
            "fehler": None if j else "keine_json_antwort"}


def regel(ergebnis: dict, schwelle: float) -> str:
    p = ergebnis.get("p_intern")
    if ergebnis.get("vorschlag") == "intern" and isinstance(p, (int, float)) and p >= schwelle:
        return "intern"
    return "privat"


def auswerten(name: str, faelle: list, ergebnisse: list, schwelle: float) -> dict:
    roh = sum(1 for f, e in zip(faelle, ergebnisse) if e["vorschlag"] == f["soll"])
    entschieden = [regel(e, schwelle) for e in ergebnisse]
    treffer = sum(1 for f, s in zip(faelle, entschieden) if s == f["soll"])
    zu_offen = [f["id"] for f, s in zip(faelle, entschieden) if f["soll"] == "privat" and s == "intern"]
    zu_eng = [f["id"] for f, s in zip(faelle, entschieden) if f["soll"] == "intern" and s == "privat"]
    ms = [e["ms"] for e in ergebnisse if not e["fehler"]]
    return {"variante": name, "faelle": len(faelle), "roh_treffer": roh, "treffer_mit_regel": treffer,
            "zu_offen_ids": zu_offen, "zu_eng_ids": zu_eng,
            "fehler": sum(1 for e in ergebnisse if e["fehler"]),
            "ms_median": round(statistics.median(ms)) if ms else None,
            "ms_p95": round(sorted(ms)[max(0, int(len(ms) * 0.95) - 1)]) if ms else None,
            "kosten_je_upload_usd": round(sum(e["kosten"] for e in ergebnisse) / max(1, len(ergebnisse)), 8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="jev,openai")
    ap.add_argument("--gemini", type=int, default=0, help="Anzahl Faelle fuer agy/Gemini (Abo), 0 = aus")
    ap.add_argument("--schwelle", type=float, default=0.8)
    ap.add_argument("--roh", default="", help="Datei fuer die Rohergebnisse (JSON)")
    ap.add_argument("--neutral", action="store_true", help="Frage ohne 'Im Zweifel privat' (den Zweifel faengt die Regel ab)")
    a = ap.parse_args()
    faelle = json.loads(FAELLE.read_text())["faelle"]
    auth = token()
    berichte, roh = [], {}
    for eng in [e for e in a.engines.split(",") if e]:
        erg = [frage_jev(f, eng, auth, FRAGE_NEUTRAL if a.neutral else FRAGE) for f in faelle]
        name = eng + ("-neutral" if a.neutral else "")
        roh[name] = erg
        berichte.append(auswerten(name, faelle, erg, a.schwelle))
    if a.gemini:
        # ausgewogen: gleich viele privat und intern, feste Auswahl
        pr = [f for f in faelle if f["soll"] == "privat"][: a.gemini // 2]
        it = [f for f in faelle if f["soll"] == "intern"][: a.gemini - len(pr)]
        teil = pr + it
        erg = [frage_gemini(f, auth) for f in teil]
        roh["gemini"] = erg
        berichte.append(auswerten("gemini-agy", teil, erg, a.schwelle))
    if a.roh:
        Path(a.roh).write_text(json.dumps(roh, ensure_ascii=False, indent=1))
    print(json.dumps(berichte, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
