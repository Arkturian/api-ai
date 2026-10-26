#!/usr/bin/env python3
"""Vergleich Jev (TypeSafe System One) gegen die OpenAI Decisions API.

Auftrag Alex 26.10.2026: Was laeuft effizienter, performanter, intelligenter?

Misst gegen die ANBIETER direkt (nicht ueber api-ai), damit der Vergleich
die Motoren misst und nicht unseren Proxy. Gleiche Faelle, gleiche Fragen,
abwechselnd gesendet (kein Tageszeit-Vorteil fuer einen Motor), jeder Fall
mehrfach (Streuung).

Datensatz: SYNTHETISCH, keine echten Personen- oder Kundendaten (Regel aus
dem Jev-Fragment). Vier Familien aus unserem Alltag: Agenten-Zustand,
Issue-Schwere/Routing, Anfrage-Routing, sprachliche Fallen (Verneinung,
Ironie, Hoeflichkeitsfloskel).

Schluessel: TYPESAFE_API_KEY bzw. /home/alex/.secrets/typesafe-api.key,
OPENAI_API_KEY bzw. --openai-key-file. Werden nie ausgegeben.

Aufruf:
    python3 scripts/bench_decisions_vs_jev.py --runs 3 --out ergebnis.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEC_URL = "https://api.openai.com/v1/decisions"
PREIS_JEV = 0.042   # USD je Mio Eingabetoken (docs.typesafe.ai)
PREIS_DEC = 0.10    # USD je Mio ungecacht (OpenAI, GA 22.10.2026)
PREIS_DEC_CACHE = 0.01

# ── Fragen ──────────────────────────────────────────────────────────────
# Jede Frage: (typ, anweisung, kriterien). choice: dict wert->beschreibung,
# score: geordnete Liste. Label je Fall: predicate bool, choice wert,
# score Index.

ZUSTAND = {
    "blockiert": ("noul", "Ist der Agent blockiert und kommt ohne Eingriff nicht weiter?", None),
    "zustand": ("choice", "In welchem Zustand ist der Agent laut diesem Bildschirmausschnitt?", {
        "arbeitet": "fuehrt gerade aktiv Befehle oder Werkzeuge aus",
        "wartet_auf_mensch": "stellt eine Frage oder wartet auf eine Freigabe eines Menschen",
        "fehler": "ein Fehler oder eine Ablehnung hat die Arbeit gestoppt",
        "fertig": "hat die Aufgabe abgeschlossen und meldet das Ergebnis",
        "leerlauf": "nichts passiert, leere Eingabezeile ohne Kontext",
    }),
}
ISSUE = {
    "schwere": ("score", "Wie schwer ist dieses Problem fuer den Betrieb?",
                ["cosmetic", "minor", "major", "critical"]),
    "komponente": ("choice", "Welcher Dienst ist zustaendig?", {
        "storage": "Dateiablage, Uploads, Medienauslieferung, Bilder-Links",
        "content": "Beitraege, Collab-Dokumente, Abschnitte, Editor",
        "comm": "E-Mail, Telegram, WhatsApp, Discord, Benachrichtigungen an Menschen",
        "ai": "KI-Endpunkte, Modelle, Transkription, Bild- und Sprachgenerierung",
        "business": "Rechnungen, Honorarnoten, Kunden, Zahlungen",
        "other": "keiner der genannten",
    }),
}
ROUTING = {
    "ziel": ("choice", "An welchen Bereich gehoert diese Anfrage?", {
        "rechnung": "Rechnung oder Honorarnote erstellen, Zahlung pruefen",
        "bild": "ein Bild, Logo oder eine Grafik erzeugen",
        "nachricht": "einem Menschen eine Nachricht schicken",
        "wissen": "eine Wissens- oder Sachfrage beantworten",
        "code": "Programmcode aendern, deployen, Fehler im Code beheben",
        "other": "nichts davon",
    }),
    "dringend": ("noul", "Verlangt die Anfrage ausdruecklich eine Erledigung noch heute oder sofort?", None),
}
SPRACHE = {
    "problem": ("noul", "Meldet der Absender ein tatsaechlich bestehendes Problem?", None),
    "antwort_noetig": ("noul", "Erwartet der Absender eine inhaltliche Antwort?", None),
}

FAELLE = [
    # Agenten-Zustand
    ("zustand", ZUSTAND, "Failed to run tool: approval policy is never. The command was rejected.\n› Ask Codex to do anything",
     {"blockiert": True, "zustand": "fehler"}),
    ("zustand", ZUSTAND, "● Running pytest -q tests/ … 412 passed so far\n  ⎿ collecting …",
     {"blockiert": False, "zustand": "arbeitet"}),
    ("zustand", ZUSTAND, "Soll ich den Release jetzt auf main schieben oder erst noch den Hotfix abwarten?\n> ",
     {"blockiert": True, "zustand": "wartet_auf_mensch"}),
    ("zustand", ZUSTAND, "Fertig: 3 Dateien geaendert, Tests gruen, auf dev gepusht (a1b2c3d). Issue #77 geschlossen.\n> ",
     {"blockiert": False, "zustand": "fertig"}),
    ("zustand", ZUSTAND, "> ", {"blockiert": False, "zustand": "leerlauf"}),
    ("zustand", ZUSTAND, "Do you want to allow Bash(rm -rf build/)? 1. Yes 2. No, and tell Claude what to do differently",
     {"blockiert": True, "zustand": "wartet_auf_mensch"}),
    ("zustand", ZUSTAND, "Error: 401 Unauthorized — Invalid authentication credentials. Please run /login.\n> ",
     {"blockiert": True, "zustand": "fehler"}),
    ("zustand", ZUSTAND, "✻ Thinking… (esc to interrupt · 41s · ↓ 2.1k tokens)",
     {"blockiert": False, "zustand": "arbeitet"}),
    # Issue-Schwere und Zustaendigkeit
    ("issue", ISSUE, "Alle Uploads an den Storage scheitern seit 20 Minuten mit 500, Portal zeigt keine Bilder mehr.",
     {"schwere": 3, "komponente": "storage"}),
    ("issue", ISSUE, "Im Collab-Editor ist der Abstand unter Ueberschriften zwei Pixel zu gross.",
     {"schwere": 0, "komponente": "content"}),
    ("issue", ISSUE, "Die Transkription schneidet Audios ueber zehn Minuten still ab; das Ergebnis wirkt vollstaendig.",
     {"schwere": 2, "komponente": "ai"}),
    ("issue", ISSUE, "Telegram-Benachrichtigungen kommen gelegentlich doppelt an.",
     {"schwere": 1, "komponente": "comm"}),
    ("issue", ISSUE, "Honorarnoten-PDF zeigt die falsche Umsatzsteuer; drei Rechnungen sind schon an Kunden raus.",
     {"schwere": 2, "komponente": "business"}),
    ("issue", ISSUE, "Der Drucker im Buero hat kein Papier mehr.",
     {"schwere": 0, "komponente": "other"}),
    ("issue", ISSUE, "Die Datenbank des Content-Dienstes ist voll, keine Beitraege lassen sich mehr speichern, alle Agenten betroffen.",
     {"schwere": 3, "komponente": "content"}),
    ("issue", ISSUE, "Die Fehlermeldung bei falschem Bildformat hat einen Tippfehler: 'Formt' statt 'Format'.",
     {"schwere": 0, "komponente": "storage"}),
    # Anfrage-Routing
    ("routing", ROUTING, "Kannst du fuer den Kunden Huber eine Honorarnote ueber 12 Stunden erstellen?",
     {"ziel": "rechnung", "dringend": False}),
    ("routing", ROUTING, "Mach mir bitte ein Logo fuer das Wanderprojekt, gruen, minimalistisch.",
     {"ziel": "bild", "dringend": False}),
    ("routing", ROUTING, "Sag Peter sofort Bescheid, dass der Termin um 14 Uhr ausfaellt!",
     {"ziel": "nachricht", "dringend": True}),
    ("routing", ROUTING, "Wie hoch ist der Grossglockner?", {"ziel": "wissen", "dringend": False}),
    ("routing", ROUTING, "Der Deploy ist rot, bitte den NameError in main.py heute noch fixen.",
     {"ziel": "code", "dringend": True}),
    ("routing", ROUTING, "Ich geh jetzt Mittagessen.", {"ziel": "other", "dringend": False}),
    ("routing", ROUTING, "Wenn du mal Zeit hast, koenntest du die Rechnung vom Maerz nochmal pruefen? Eilt nicht.",
     {"ziel": "rechnung", "dringend": False}),
    ("routing", ROUTING, "Bitte bis heute Abend ein Titelbild fuer den Newsletter erzeugen.",
     {"ziel": "bild", "dringend": True}),
    # Sprachliche Fallen
    ("sprache", SPRACHE, "Das Deployment ist nicht kaputt, es dauert nur laenger als sonst. Alles gut.",
     {"problem": False, "antwort_noetig": False}),
    ("sprache", SPRACHE, "Na super, schon wieder ist der Server weg. Toll gemacht.",
     {"problem": True, "antwort_noetig": False}),
    ("sprache", SPRACHE, "Danke dir, passt so!", {"problem": False, "antwort_noetig": False}),
    ("sprache", SPRACHE, "Ich wollte nur sagen, dass es keinen Fehler mehr gibt — oder siehst du noch einen?",
     {"problem": False, "antwort_noetig": True}),
    ("sprache", SPRACHE, "Kein Stress, aber der Login geht seit gestern bei keinem von uns.",
     {"problem": True, "antwort_noetig": True}),
    ("sprache", SPRACHE, "The backup job didn't fail last night; I double-checked the logs.",
     {"problem": False, "antwort_noetig": False}),
    ("sprache", SPRACHE, "Weisst du, warum die Rechnungen doppelt verbucht sind?",
     {"problem": True, "antwort_noetig": True}),
    ("sprache", SPRACHE, "Unglaublich, wie schnell das jetzt laeuft. Respekt!",
     {"problem": False, "antwort_noetig": False}),
]


# ── Aufrufe ─────────────────────────────────────────────────────────────

def _post(url: str, key: str, body: dict, timeout: float = 30.0) -> tuple:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            code = r.status
    except urllib.error.HTTPError as e:
        code, data = e.code, {"raw": e.read()[:300].decode(errors="replace")}
    except Exception as e:  # Netz, Timeout
        code, data = 0, {"exc": type(e).__name__}
    return code, data, (time.perf_counter() - t0) * 1000


def jev_body(state, fragen):
    q = {}
    for n, (t, ins, krit) in fragen.items():
        q[n] = {"type": t, "instructions": ins}
        if krit is not None:
            q[n]["criteria"] = krit
    return {"model": "jev-latest", "state": state, "questions": q}


def dec_body(state, fragen):
    q = []
    for n, (t, ins, krit) in fragen.items():
        e = {"name": n, "instructions": ins}
        if t == "noul":
            e["type"] = "predicate"
        elif t == "choice":
            e["type"] = "choice"
            e["choices"] = [{"value": k, "description": v} for k, v in krit.items()]
        else:
            e["type"] = "score"
            e["levels"] = [{"label": s, "description": s} for s in krit]
        q.append(e)
    return {"model": "gpt-6-luna", "input": state, "questions": q}


def normalisieren(motor, data):
    """-> {name: {"p": float|None, "choice": str|None, "score": float|None,
    "probs": dict, "confidence": float|None}}"""
    out = {}
    if motor == "jev":
        for n, a in (data.get("answers") or {}).items():
            out[n] = {"p": a.get("noul"), "choice": a.get("choice"), "score": a.get("score"),
                      "probs": a.get("probabilities") or {}, "confidence": a.get("confidence")}
    else:
        for a in data.get("answers") or []:
            probs = {str(x.get("value")): x.get("probability") for x in a.get("probabilities") or []}
            out[a.get("name")] = {"p": a.get("probability"), "choice": a.get("choice"),
                                  "score": a.get("score"), "probs": probs,
                                  "confidence": a.get("confidence"), "type": a.get("type")}
    return out


# ── Auswertung ──────────────────────────────────────────────────────────

def bewerten(fragen, label, antwort):
    """-> Liste (art, richtig, brier|None, logloss|None, confidence|None)."""
    zeilen = []
    for n, soll in label.items():
        t = fragen[n][0]
        a = antwort.get(n)
        if not a:
            zeilen.append((t, False, None, None, None))
            continue
        if t == "noul":
            p = a["p"]
            if p is None:
                zeilen.append((t, False, None, None, None))
                continue
            y = 1.0 if soll else 0.0
            pk = min(max(p, 1e-4), 1 - 1e-4)
            zeilen.append((t, (p >= 0.5) == soll, (p - y) ** 2,
                           -(y * math.log(pk) + (1 - y) * math.log(1 - pk)), None))
        elif t == "choice":
            ps = a["probs"].get(soll)
            richtig = a["choice"] == soll
            ll = -math.log(min(max(ps or 0.0, 1e-4), 1.0)) if ps is not None else None
            brier = None
            if a["probs"]:
                brier = sum(((1.0 if k == soll else 0.0) - (v or 0.0)) ** 2 for k, v in a["probs"].items())
            zeilen.append((t, richtig, brier, ll, a.get("confidence")))
        else:
            sc = a["score"]
            richtig = sc is not None and round(sc) == soll
            zeilen.append((t, richtig, (abs(sc - soll) if sc is not None else None), None,
                           a.get("confidence")))
    return zeilen


def perzentil(werte, q):
    w = sorted(werte)
    if not w:
        return None
    i = min(len(w) - 1, max(0, int(round(q * (len(w) - 1)))))
    return w[i]


def zusammenfassen(lauf):
    ber = {}
    for motor in ("jev", "openai"):
        z = lauf[motor]
        lat = [r["ms"] for r in z if r["ok"]]
        tok = [r["tokens"] for r in z if r["ok"]]
        zeilen = [x for r in z if r["ok"] for x in r["bewertung"]]
        def antw(art, idx):
            v = [x[idx] for x in zeilen if x[0] == art and x[idx] is not None]
            return v
        def quote(art):
            v = [x[1] for x in zeilen if x[0] == art]
            return round(sum(v) / len(v), 3) if v else None
        alle = [x[1] for x in zeilen]
        falsche_conf = [x[4] for x in zeilen if not x[1] and x[4] is not None]
        richtige_conf = [x[4] for x in zeilen if x[1] and x[4] is not None]
        ber[motor] = {
            "requests": len(z), "fehler": sum(1 for r in z if not r["ok"]),
            "latency_ms_p50": round(perzentil(lat, 0.5) or 0), "latency_ms_p95": round(perzentil(lat, 0.95) or 0),
            "latency_ms_mean": round(statistics.mean(lat)) if lat else None,
            "input_tokens_mean": round(statistics.mean(tok)) if tok else None,
            "usd_je_1000_anfragen": round((statistics.mean(tok) if tok else 0) *
                                          (PREIS_JEV if motor == "jev" else PREIS_DEC) / 1e6 * 1000, 5),
            "genauigkeit_gesamt": round(sum(alle) / len(alle), 3) if alle else None,
            "genauigkeit_predicate": quote("noul"), "genauigkeit_choice": quote("choice"),
            "genauigkeit_score": quote("score"),
            "brier_predicate": round(statistics.mean(antw("noul", 2)), 4) if antw("noul", 2) else None,
            "logloss_predicate": round(statistics.mean(antw("noul", 3)), 4) if antw("noul", 3) else None,
            "brier_choice": round(statistics.mean(antw("choice", 2)), 4) if antw("choice", 2) else None,
            "score_abstand_mittel": round(statistics.mean(antw("score", 2)), 3) if antw("score", 2) else None,
            "confidence_bei_richtig": round(statistics.mean(richtige_conf), 3) if richtige_conf else None,
            "confidence_bei_falsch": round(statistics.mean(falsche_conf), 3) if falsche_conf else None,
        }
    return ber


def streuung(lauf, motor):
    """Mittlere Spannweite der Wahrscheinlichkeit/des Scores je Frage ueber
    die Wiederholungen — 0 heisst deterministisch."""
    werte = {}
    for r in lauf[motor]:
        if not r["ok"]:
            continue
        for n, a in r["antwort"].items():
            v = a["p"] if a["p"] is not None else (a["score"] if a["score"] is not None
                                                   else (a["probs"].get(a["choice"]) if a["choice"] else None))
            if v is not None:
                werte.setdefault((r["fall"], n), []).append(v)
    sp = [max(v) - min(v) for v in werte.values() if len(v) > 1]
    return round(statistics.mean(sp), 4) if sp else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="bench_decisions_vs_jev.json")
    ap.add_argument("--openai-key-file")
    ap.add_argument("--typesafe-key-file", default="/home/alex/.secrets/typesafe-api.key")
    a = ap.parse_args()
    ok = os.getenv("OPENAI_API_KEY") or (Path(a.openai_key_file).read_text().strip() if a.openai_key_file else "")
    tk = os.getenv("TYPESAFE_API_KEY") or Path(a.typesafe_key_file).read_text().strip()
    lauf = {"jev": [], "openai": []}
    for runde in range(a.runs):
        for i, (fam, fragen, state, label) in enumerate(FAELLE):
            # abwechselnd, wer zuerst geht
            reihenfolge = ("jev", "openai") if (i + runde) % 2 == 0 else ("openai", "jev")
            for motor in reihenfolge:
                if motor == "jev":
                    code, data, ms = _post(JEV_URL, tk, jev_body(state, fragen))
                    tokens = ((data or {}).get("usage") or {}).get("input_tokens")
                else:
                    code, data, ms = _post(DEC_URL, ok, dec_body(state, fragen))
                    tokens = ((data or {}).get("usage") or {}).get("input_tokens")
                antwort = normalisieren(motor, data) if code == 200 else {}
                lauf[motor].append({"fall": i, "familie": fam, "runde": runde, "ok": code == 200,
                                    "code": code, "ms": ms, "tokens": tokens or 0, "antwort": antwort,
                                    "bewertung": bewerten(fragen, label, antwort) if code == 200 else [],
                                    "fehler": None if code == 200 else data})
        print(f"Runde {runde + 1}/{a.runs} fertig", flush=True)
    ber = zusammenfassen(lauf)
    ber["jev"]["streuung"] = streuung(lauf, "jev")
    ber["openai"]["streuung"] = streuung(lauf, "openai")
    # Fehler je Fall (aus Runde 0) fuer die qualitative Auswertung
    falsch = {m: [] for m in ("jev", "openai")}
    for m in falsch:
        for r in lauf[m]:
            if r["runde"] == 0 and r["ok"]:
                fam, fragen, state, label = FAELLE[r["fall"]]
                for (n, soll), z in zip(label.items(), r["bewertung"]):
                    if not z[1]:
                        falsch[m].append({"fall": r["fall"], "text": state[:80], "frage": n, "soll": soll,
                                          "ist": r["antwort"].get(n)})
    Path(a.out).write_text(json.dumps({"bericht": ber, "falsch_runde0": falsch, "roh": lauf},
                                      ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"bericht": ber, "falsch_runde0": falsch}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
