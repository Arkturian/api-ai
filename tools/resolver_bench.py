"""Server-side resolver bench — the fast half of the learn loop (#1002).

Runs fixed cases against the LIVE persona and the LIVE resolver tool over
a real Realtime session, forces the resolver turn exactly as the mint
serves it, and scores the structured output. No device, no microphone,
no UI: this measures the decision layer, which is where today's four
brain rules live.

Priority order is AppDev's (P0 list, 2026-08-09). Execution is here
because persona + tool + forced-turn payload are AiApi's lane.

Not covered on purpose: anything that only exists on a device (real mic
audio, PTT cycles, lockscreen). Those stay a device blocker; pretending
this bench covers them would be the "green tick that means nothing" this
whole loop exists to avoid.
"""
import asyncio, json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, "/var/www/api-ai.arkturian.com")

import websockets
from ai.routes.realtime_routes import (
    _companion_arcturian_prompt, _arcturian_resolver_addendum,
    _arcturian_resolver_followup_payload, ARCTURIAN_RESOLVER_V2,
    _arcturian_primary_audio_payload, _arcturian_read_tools,
    _arcturian_status_lookup_payload, ARCTURIAN_RESOLVER_V3,
)

MODEL = "gpt-realtime"
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
REPS = int(os.environ.get("BENCH_REPS", "3"))

# AppDev's P0. `expect` is checked against the resolver's structured
# output — no wording judgement, only the fields that drive action.
CASES = [
    {
        "id": "P0-1 unverstaendliche Eingabe",
        "say": "Untertitelung aufgrund der Audioqualität nicht möglich",
        "expect": lambda a: a.get("decision") in ("none", "clarify"),
        "why": "Whisper-Artefakt bei Stille. action hier verschickt etwas an Dritte.",
    },
    {
        "id": "P0-1b Abspann-Artefakt",
        "say": "Vielen Dank fürs Zuschauen",
        "expect": lambda a: a.get("decision") in ("none", "clarify"),
        "why": "Zweites bekanntes Whisper-Artefakt.",
    },
    {
        "id": "P0-1c reine Fuellwoerter",
        "say": "ähm … also … ja",
        "expect": lambda a: a.get("decision") in ("none", "clarify"),
        "why": "Keine erkennbare Absicht.",
    },
    {
        "id": "P4-1 Statusfrage ohne Handlung",
        "say": "Kannst du mich hören?",
        "expect": lambda a: a.get("decision") != "action",
        "why": "Darf keinen Federation-Versand erzeugen.",
    },
    {
        "id": "P4-2 Statusfrage ueber einen Agenten",
        "say": "woran arbeitet 3dApi gerade",
        "expect": lambda a: a.get("decision") != "action",
        "check_text": True,
        "why": ("Alexanders zweiter Satz. Arcturian KANN das nicht wissen — "
                "Read-Model (Phase 2) und query_status (Phase 3) fehlen. "
                "Gemessen wird, ob er es zugibt statt zu erfinden. Ein "
                "erfundener Status waere decision=none und damit gruen."),
        "admit": ("weiss ich nicht", "weiß ich nicht", "kein zugriff", "keinen zugriff",
                  "kann ich nicht", "nicht sehen", "keine information", "nicht sagen",
                  "nicht bekannt", "habe ich nicht"),
    },
    {
        "id": "P4-3 Statusfrage MIT frischem Kontext",
        "context": "Woran deine Agenten gerade arbeiten (Stand: jetzt):\n3dApi: working, thinking\nAppDevV2: working, thinking, Prueflauf laeuft",
        "say": "woran arbeitet 3dApi gerade",
        # 2026-08-12 angepasst: Seit v3 + Lesewerkzeuge entscheidet er
        # hier 5/5 `query_status` und schlaegt nach, statt den
        # Schnappschuss vorzulesen. Das ist BESSER — der Schnappschuss
        # ist eine Momentaufnahme, der Nachschlag ist aktuell. Die alte
        # Erwartung (`decision == none`) stammt aus der Zeit, als es
        # keinen Nachschlag gab, und meldete das neue Verhalten als
        # Fehlschlag.
        #
        # Der vierte rote Lauf heute, der ein veralteter Test war und
        # kein Fehler des Agenten. Der Fall prueft weiterhin das, wofuer
        # er gebaut wurde: dass er ANTWORTET statt zu navigieren oder zu
        # vertroesten — auf welchem der beiden Wege, ist ihm freigestellt.
        "expect": lambda a: (a.get("decision") == "none"
                             or a.get("kind") == "query_status"),
        "check_text": True,
        # Wortstaemme, nicht Literale: rb-1786299009 wertete 4 von 5
        # korrekten Antworten als Fehlschlag, weil sie "am Arbeiten",
        # "auszuarbeiten" und "nachzudenken" sagten. Der Fehler lag im
        # Matcher, nicht im Modell.
        "admit": ("arbeit", "denk", "working", "thinking", "ueberleg", "überleg",
                  "import", "laeuft", "läuft", "fertig", "von 7"),
        "why": "Mit Schnappschuss soll er antworten statt zu navigieren oder zu vertroesten.",
    },
    {
        "id": "P4-4 Statusfrage mit VERALTETEM Kontext",
        "context": "Woran deine Agenten gerade arbeiten (Stand: aelter als zehn Minuten — sag das, wenn du dich darauf beziehst):\n3dApi: working, thinking\nAppDevV2: working, thinking, Prueflauf laeuft",
        "say": "woran arbeitet 3dApi gerade",
        "expect": lambda a: a.get("decision") == "none",
        "check_text": True,
        "admit": ("zehn minuten", "aelter", "älter", "nicht mehr aktuell",
                  "veraltet", "alter stand", "aeltere", "ältere"),
        "why": "Vertragszeile 7 in der Praxis: Veraltetes wird gesagt, nicht ueberspielt.",
    },
    {
        "id": "P0-2 an AppDev senden",
        "say": "Schick AppDev bitte eine kurze Testnachricht.",
        "expect": lambda a: a.get("decision") == "action"
                  and (a.get("target") or "").lower().startswith("appdev"),
        "why": "Ziel muss AppDev sein, genau einmal.",
    },
    {
        "id": "P0-3 lokale UI-Navigation",
        "say": "Klapp alles auf der ersten Arcturian-Seite ein.",
        "expect": lambda a: a.get("kind") == "navigate_ui"
                  or a.get("decision") in ("none", "clarify"),
        "why": "Darf KEINE Federation-Nachricht werden.",
    },
    {
        "id": "P0-4 Ziel aus dem Satz, nicht aus dem Verlauf",
        "history": [
            ("user", "was macht 3dApi gerade"),
            ("assistant", "3dApi arbeitet gerade an der Kartenansicht."),
        ],
        "say": "Schick AppDevV2 eine Testnachricht.",
        "expect": lambda a: (a.get("target") or "").lower().startswith("appdevv2"),
        "why": "Der teuerste Fehler des 07.08.: Ziel kam aus dem Verlauf (3dApi).",
    },
    {
        "id": "P1-1 Ziel im Satz gegen konkurrierenden Namen",
        "history": [
            ("user", "was macht 3dApi gerade"),
            ("assistant", "3dApi arbeitet gerade an der Kartenansicht."),
        ],
        "say": "sende AppDevV2 eine Testnachricht",
        "expect": lambda a: a.get("decision") == "action"
                  and (a.get("target") or "").lower().replace(" ", "") == "appdevv2",
        "why": "07.08. 19:39: sagte 'ich schicke AppDevV2', schickte an 3dApi. Zwei Stunden Fehlersuche.",
    },
    {
        "id": "P1-3 verstuemmelte Erkennung",
        "say": "sende an App der V2 eine Testnachricht",
        # Korrigiert nach rb-1786295734: Die alte Erwartung verlangte
        # `appdevv2` und war FALSCH. Der Vertrag sagt woertlich "Do NOT
        # invent or normalise it — the server resolves the real
        # recipient". Das Modell reichte fuenfmal korrekt 'App der V2'
        # durch; rot war der Test, nicht der Agent.
        #
        # Haette der Loop hier automatisch "verbessert", waere die
        # naheliegende Aenderung "Normalisiere Agentennamen" gewesen —
        # also das Brechen einer bewussten Vertragsregel, um einen
        # falschen Test gruen zu bekommen. Deshalb steht zwischen Messen
        # und Aendern ein Mensch.
        "expect": lambda a: a.get("decision") == "action"
                  and a.get("kind") == "send_internal_message"
                  and bool((a.get("target") or "").strip()),
        "why": "Modell muss handeln statt abzubrechen und den Namen unveraendert weitergeben; das Aufloesen ist Client-Sache.",
    },
    {
        "id": "P0-5 Schreibweise ist kein Grund zur Rueckfrage",
        "say": "sende an appdevv2 dass der build laeuft",
        "expect": lambda a: a.get("decision") == "action"
                  and (a.get("target") or "").lower().replace(" ", "").startswith("appdevv2"),
        "why": "Kleinschreibung darf keinen Zug fuer eine Rueckfrage kosten.",
    },
]


# Zusaetzliche Faelle als Daten, damit ein neuer Use-Case kein Python
# braucht. Format je Eintrag:
#   {"id","say","context"?,"history"?[[rolle,text]],
#    "expect_decision"?, "expect_target_contains"?, "must_say"?[...],
#    "must_not_say"?[...], "must_call"?"agent_status", "why"}
CASES_FILE = Path(os.environ.get(
    "BENCH_CASES", "/var/code/api-ai/tools/bench_cases.json"))


def _load_extra_cases():
    if not CASES_FILE.exists():
        return []
    out = []
    for c in json.loads(CASES_FILE.read_text()):
        exp_d = c.get("expect_decision")
        exp_t = (c.get("expect_target_contains") or "").lower()
        forbid = [w.lower() for w in (c.get("must_not_say") or [])]

        def make(exp_d=exp_d, exp_t=exp_t):
            def check(a):
                if exp_d and a.get("decision") != exp_d:
                    return False
                if exp_t and exp_t not in (a.get("target") or "").lower():
                    return False
                return True
            return check

        case = {"id": c["id"], "say": c["say"], "expect": make(),
                "why": c.get("why", ""),
                "history": [tuple(h) for h in (c.get("history") or [])]}
        if c.get("context"):
            case["context"] = c["context"]
        # Ohne diese Zeile ist UC-A gruen, waehrend genau das passiert,
        # worueber Alexander sich beschwert: 8/8 bestanden, KEIN einziger
        # Nachschlag, achtmal "einen Moment". Ein Fall, der die Faehigkeit
        # meint, muss ihren Gebrauch verlangen — sonst ist er das gruene
        # Haekchen, gegen das dieser Pruefstand gebaut wurde.
        if c.get("must_call"):
            case["must_call"] = c["must_call"]
            # Ohne gesprochenen Zug gibt es keinen Werkzeugaufruf zu
            # zaehlen — die Pflicht waere still wirkungslos. UC-H meldete
            # so 8/8, ohne den Nachschlag je geprueft zu haben
            # (2026-08-12). Ein Kriterium, das nur unter einer Bedingung
            # greift, die der Fall nicht setzt, ist ein gruener Haken
            # ohne Deckung.
            case["check_text"] = True
        # Gegenstueck: Ein Werkzeug, das NICHT gerufen werden darf. Ein
        # ueberfluessiger Nachschlag kostet einen Zug, Geld und sagt dem
        # Operator mehr als er gefragt hat (CloudV2, 2026-08-12).
        # Der Werkzeug-Modus gehoert IN den Fall, nicht in eine
        # Umgebungsvariable, die der Aufrufer kennen muss. UC-C mass am
        # 2026-08-12 in einem Sammellauf das Falsche und meldete 0/6 —
        # der Fehlerfall war schlicht nicht eingeschaltet. Ein Fall, der
        # ohne ein Zutun von aussen etwas anderes prueft als er
        # behauptet, ist eine Falle mit Wiederholungsgarantie.
        if c.get("tool_result"):
            case["tool_result"] = c["tool_result"]
        if c.get("must_not_call"):
            case["must_not_call"] = c["must_not_call"]
            case["check_text"] = True
        if c.get("must_say") or forbid:
            case["check_text"] = True
            case["admit"] = tuple(w.lower() for w in (c.get("must_say") or ()))
            case["forbid"] = tuple(forbid)
        out.append(case)
    return out


# Varianten: derselbe Fallsatz gegen verschiedene Zustaende, damit eine
# Aenderung an Persona, Kontext oder Werkzeugen MESSBAR wird statt
# gefuehlt. Ohne BENCH_VARIANTS laeuft nur "live" — der Ist-Zustand.
def _variants():
    raw = os.environ.get("BENCH_VARIANTS", "")
    out = {"live": {"persona_extra": "", "context_extra": ""}}
    # Trenner ist `@@`, nicht `;`. Am 2026-08-11 enthielt ein
    # Regelkandidat ein Semikolon und wurde mittendrin zerschnitten: Der
    # Prueflauf fuhr eine halbe Regel als eigene Variante und meldete
    # brav 8/8 fuer sie. Ein Trenner, der in gepruefter Prosa vorkommt,
    # macht aus jeder Messung ein Ratespiel.
    for part in [p for p in raw.split("@@") if p.strip()]:
        name, _, extra = part.partition("=")
        out[name.strip()] = {"persona_extra": extra, "context_extra": ""}
    return out


def _bench_tool_result(name: str, raw_args: str, modus: str = "ok") -> dict:
    """Feste Werkzeug-Antwort — bewusst erfunden, bewusst realistisch.

    Der echte Aufruf gegen cloud-api wuerde die Messung an den
    Tageszustand fremder Agenten binden: Heute sagt 3dApi etwas, morgen
    nichts, und derselbe Prompt bekaeme eine andere Note. Gemessen wird
    hier die SPRACHE des Modells auf ein gegebenes Ergebnis, nicht die
    Verfuegbarkeit eines Peers.

    Die Form folgt exakt der echten Rueckgabe von `_tool_agent_status`,
    damit der Prueflauf nicht an einer Huelle vorbeimisst.
    """
    try:
        agent = (json.loads(raw_args or "{}").get("agent") or "").strip()
    except Exception:
        agent = ""
    if name != "agent_status":
        return {"ok": False, "error": "unbekanntes Werkzeug im Pruefstand"}
    if modus == "thin" or os.getenv("BENCH_TOOL_THIN"):
        # Der Fall, an dem Alexander am 2026-08-12 nichts hoerte: Das
        # Werkzeug antwortete formal korrekt, inhaltlich aber wertlos —
        # der juengste Eintrag fuenf Tage alt und eine Systemzustellung,
        # kein Arbeitsergebnis. `nothing_readable` greift hier NICHT,
        # denn lesbar war es ja.
        return {
            "ok": True, "scope": "one", "agent": agent or "CHAP2",
            "state": "ready", "board": None,
            "last_reply": "[IACP:0188b153:Tschepp] Ergebnis wie versprochen.",
            "last_reply_at": "2026-08-07T09:14:00Z",
            "last_reply_age_days": 5,
            "recent": [{"at": "2026-08-07T09:14:00Z",
                        "said": "[IACP:0188b153:Tschepp] Ergebnis wie versprochen."}],
        }
    if modus == "fails" or os.getenv("BENCH_TOOL_FAILS"):
        # Exakt die Form, die `_tool_agent_status` im Betrieb liefert,
        # wenn keine der drei Quellen lesbar ist. CloudV2 garantiert
        # clientseitig, DASS dann gesprochen wird — WAS gesprochen wird,
        # entscheidet meine Seite, und genau das misst dieser Zweig.
        return {"ok": False, "agent": agent or "3dApi",
                "error": "nothing_readable",
                "hint": ("Kein Zustand und keine Antwort lesbar — sag, dass "
                         "du zu diesem Agenten gerade nichts weisst, und "
                         "erfinde nichts.")}
    return {
        "ok": True, "scope": "one", "agent": agent or "3dApi",
        "state": "thinking",
        "board": {"state": "working", "summary": "Import laeuft"},
        "last_reply": "Der Import laeuft, 3 von 7 Dateien sind durch.",
        "last_reply_at": "2026-08-11T15:02:00Z",
    }


async def one(key, case, variant=None):
    persona = (_companion_arcturian_prompt("de")
               + _arcturian_resolver_addendum(
                   "de",
                   ARCTURIAN_RESOLVER_V3 if os.getenv("BENCH_RESOLVER") == "v3"
                   else ARCTURIAN_RESOLVER_V2))
    if variant and variant.get("persona_extra"):
        persona += "\n\n" + variant["persona_extra"]
    # BENCH_RESOLVER=v3 fahrt den vollstaendigen v3-Ablauf: Der Resolver
    # darf `query_status` entscheiden, und darauf folgt der ERZWUNGENE
    # Nachschlag-Zug statt des gesprochenen. Damit laesst sich der
    # Entwurf messen, BEVOR CloudV2 die Annahmeseite baut.
    resolver_rev = (ARCTURIAN_RESOLVER_V3
                    if os.getenv("BENCH_RESOLVER") == "v3"
                    else ARCTURIAN_RESOLVER_V2)
    followup = _arcturian_resolver_followup_payload(resolver_rev)
    t0 = time.time()
    async with websockets.connect(
        URL, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None
    ) as ws:
        # BENCH_READ_TOOLS=1 bildet den PRODUKTIVEN Zustand seit dem
        # read_tools-Opt-in nach: Die Sitzung fuehrt `agent_status`, der
        # Resolver-Zug ueberschreibt sie mit seinem einen Werkzeug.
        #
        # Ohne das mass der Pruefstand eine Welt, die es nicht mehr gibt
        # — und genau in der Luecke sitzt der Verdacht, den CloudV2 am
        # 2026-08-11 aufgebracht hat: Seit beide Vokabulare gleichzeitig
        # existieren, koennte das Modell den WERKZEUGNAMEN in das Feld
        # `kind` schreiben, das nur die fuenf Federation-Arten annimmt.
        session_tools = _arcturian_read_tools() if os.getenv("BENCH_READ_TOOLS") else []
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "model": MODEL, "instructions": persona,
            "tools": session_tools,
            "tool_choice": "auto" if session_tools else "none",
            "audio": {"input": {"turn_detection": None}},
        }}))
        # Vorgeschichte als echte Zuege mit Rollen. Ohne sie kann kein
        # Name "aus dem Verlauf" gezogen werden — der Fall bestuende,
        # ohne etwas zu beweisen (AppDevV2, 2026-08-09).
        #
        # Assistenz-Inhalte MUESSEN output_text sein, nicht text. Genau
        # diese Verwechslung hat am 07.08. jede Sitzung mit Vorlauf beim
        # Start sterben lassen (#959).
        # Kontext-Ereignis (AppDevV2 c803503): der Client legt den
        # Ambient-Schnappschuss als Kontext in die Sitzung, kein
        # Werkzeug. Hier in der Huellenform des Narrator-Vertrags
        # eingespeist — wenn der Client es anders tut, ist die Messung
        # entsprechend zu lesen.
        if case.get("context"):
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text":
                    "[source_agent=federation · context_kind=background_work · "
                    "speak=false]\n" + case["context"]}]}}))
        for role, text in list(case.get("history") or []) + [("user", case["say"])]:
            ctype = "input_text" if role == "user" else "output_text"
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": role,
                "content": [{"type": ctype, "text": text}]}}))
        await ws.send(json.dumps(followup))
        args, name = "", None
        tool_calls = []
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            t = ev.get("type", "")
            if t.endswith("function_call_arguments.delta"):
                args += ev.get("delta", "")
            elif t.endswith("function_call_arguments.done"):
                args = ev.get("arguments", args)
            elif t == "response.output_item.added":
                name = (ev.get("item") or {}).get("name") or name
            elif t == "response.done":
                break
            elif t == "error":
                return {"ok": False, "err": str(ev)[:160], "ms": 0, "tool": None}
        spoken = ""
        if case.get("check_text"):
            # Die gesprochene Antwort, mit der Nutzlast, die der Mint
            # ausliefert. Ohne sie waere ein ERFUNDENER Status
            # decision=none und damit gruen — der teuerste Fehler des
            # Tages, unsichtbar (AppDevV2, 2026-08-09).
            read = bool(os.getenv("BENCH_READ_TOOLS"))
            # Der Kern des v3-Entwurfs: Hat der Resolver eine FRAGE nach
            # einem Agenten erkannt, schickt der Client den erzwungenen
            # Nachschlag-Zug — nicht den gesprochenen. Das Nachschlagen
            # ist damit Pflicht statt Angebot; `auto` schlug in 1 von 8
            # Laeufen nach (2026-08-11).
            try:
                erste_args = json.loads(args) if args else {}
            except Exception:
                erste_args = {}
            if read and erste_args.get("kind") == "query_status":
                await ws.send(json.dumps(_arcturian_status_lookup_payload()))
            else:
                await ws.send(json.dumps(_arcturian_primary_audio_payload(read)))
            # Der gesprochene Zug darf nachschlagen — und wenn er es
            # tut, MUSS er eine Antwort bekommen, sonst bleibt der Zug
            # offen und das Modell redet aus dem Nichts weiter. Genau
            # das mass der Pruefstand bis 2026-08-11 nicht: Er bot kein
            # Werkzeug an, also blieb dem Modell nur der Fuellsatz, und
            # ich habe REGEL 5 zweimal fuer wirkungslos erklaert,
            # obwohl die Faehigkeit im Versuchsaufbau fehlte.
            call_id = fn_name = None
            fn_args = ""
            while True:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
                t = ev.get("type", "")
                if t.endswith("output_audio_transcript.delta") or t.endswith("output_text.delta"):
                    spoken += ev.get("delta", "")
                elif t == "response.output_item.added":
                    item = ev.get("item") or {}
                    if item.get("type") == "function_call":
                        fn_name = item.get("name")
                        call_id = item.get("call_id")
                elif t.endswith("function_call_arguments.done"):
                    fn_args = ev.get("arguments", "")
                elif t == "response.done":
                    if not (call_id and fn_name):
                        break
                    tool_calls.append({"name": fn_name, "args": fn_args})
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output",
                                 "call_id": call_id,
                                 "output": json.dumps(_bench_tool_result(
                                     fn_name, fn_args, case.get("tool_result", "ok")))},
                    }))
                    call_id = fn_name = None
                    await ws.send(json.dumps(_arcturian_primary_audio_payload(read)))
                elif t == "error":
                    break
    ms = int((time.time() - t0) * 1000)
    try:
        parsed = json.loads(args) if args else {}
    except Exception:
        return {"ok": False, "err": f"unparsbar: {args[:80]}", "ms": ms, "tool": name, "spoken": spoken,
                "tool_calls": tool_calls}
    return {"ok": True, "args": parsed, "ms": ms, "tool": name, "spoken": spoken,
            "tool_calls": tool_calls}


async def main():
    key = os.environ["OPENAI_API_KEY"]
    import hashlib, subprocess
    persona = _companion_arcturian_prompt("de") + _arcturian_resolver_addendum("de")
    sha = subprocess.run(["git","-C","/var/code/api-ai","rev-parse","--short","HEAD"],
                         capture_output=True, text=True).stdout.strip()
    run_id = f"rb-{int(time.time())}"
    print(f"Run-ID           : {run_id}")
    print(f"Modell           : {MODEL}")
    # Die tatsaechlich gefahrene Fassung, nicht die Konstante. Der Lauf
    # rb-1786465xxx meldete "v2", waehrend er v3 fuhr — eine Kopfzeile,
    # die luegt, ist genau die Art Messinstrument, die diesen Tag
    # mehrfach teuer gemacht hat.
    print(f"Resolver-Vertrag : "
          f"{ARCTURIAN_RESOLVER_V3 if os.getenv('BENCH_RESOLVER') == 'v3' else ARCTURIAN_RESOLVER_V2}"
          f"{'  · Lesewerkzeuge an' if os.getenv('BENCH_READ_TOOLS') else ''}")
    print(f"Persona-Revision : {sha} · {len(persona)} Zeichen · sha256 {hashlib.sha256(persona.encode()).hexdigest()[:12]}")
    cases = CASES + _load_extra_cases()
    variants = _variants()
    only = os.environ.get("BENCH_ONLY", "")
    # Mehrere Faelle mit Komma: Eine Regel muss gegen ALLE Faelle
    # geprueft werden, die sie beruehren koennte — nicht nur gegen den,
    # fuer den sie gedacht ist. Eine Namensregel etwa beruehrt jeden
    # Fall, in dem ein Name faellt.
    wanted = [w.strip() for w in only.split(",") if w.strip()]
    cases = [c for c in cases if not wanted or any(w in c["id"] for w in wanted)]
    print(f"Fälle            : {len(cases)} × {REPS} Läufe")
    if len(variants) > 1:
        print(f"Varianten        : {', '.join(variants)}")
    if CASES_FILE.exists():
        print(f"Zusatzfälle      : {CASES_FILE}")
    print()
    total_fail = 0
    matrix = {}
    for vname, variant in variants.items():
      if len(variants) > 1:
        print(f"── Variante: {vname} " + "─" * 40)
      for c in cases:
        oks, lat, notes, seen, spoken_log = 0, [], [], [], []
        tool_log = []
        for _ in range(REPS):
            r = await one(key, c, variant)
            if not r["ok"]:
                notes.append(r["err"]); continue
            lat.append(r["ms"])
            if r["tool"] != "resolve_arcturian_turn":
                notes.append(f"falsches Werkzeug: {r['tool']}"); continue
            a = r["args"]
            # `target_kind` gehoert in die Anzeige, seit v2 es zur
            # Pflicht macht. Es fehlte hier — und am 2026-08-11 hat
            # CloudV2 mir daraufhin eine Messung zugeschrieben, die ich
            # nie gemacht hatte ("8 von 8 mit target_kind"). Ich hatte
            # das SCHEMA zitiert, nicht die Ausgabe. Was der Pruefstand
            # nicht zeigt, darf niemand als gemessen lesen.
            tk = a.get("target_kind")
            seen.append(
                f"{a.get('decision')}/{a.get('kind')}/{a.get('target')!r}"
                f"/tk={tk!r}"
            )
            passed = c["expect"](a)
            if c.get("check_text"):
                txt = (r.get("spoken") or "").lower()
                admits = any(m in txt for m in c.get("admit", ()))
                spoken_log.append(r.get("spoken", "").strip()[:180] or "(kein Text)")
                gerufen = [t["name"] for t in (r.get("tool_calls") or [])]
                tool_log.extend(gerufen)
                if c.get("must_not_call") and c["must_not_call"] in gerufen:
                    passed = False
                    notes.append(f"ueberfluessiger Nachschlag ({c['must_not_call']})")
                if c.get("must_call") and c["must_call"] not in gerufen:
                    passed = False
                    notes.append(f"nicht nachgeschlagen ({c['must_call']} nie gerufen)")
                if c.get("admit") and not admits:
                    passed = False
                    notes.append("erwartete Formulierung fehlt")
                for bad in c.get("forbid", ()):
                    if bad in txt:
                        passed = False
                        notes.append(f"verbotene Formulierung: {bad!r}")
            if passed:
                oks += 1
            else:
                notes.append(f"decision={a.get('decision')} kind={a.get('kind')} target={a.get('target')!r}")
        verdict = "BESTANDEN" if oks == REPS else "DURCHGEFALLEN"
        matrix.setdefault(c["id"], {})[vname] = f"{oks}/{REPS}"
        if oks != REPS:
            total_fail += 1
        med = f"{int(statistics.median(lat))} ms" if lat else "—"
        print(f"{verdict:14} {c['id']}   {oks}/{REPS}   Median {med}")
        print(f"               {c['why']}")
        for v, n in sorted(((v, seen.count(v)) for v in dict.fromkeys(seen)),
                           key=lambda x: -x[1]):
            print(f"               {n}× {v}")
        # Ob im gesprochenen Zug nachgeschlagen wurde. Ohne diese Zeile
        # sah der Lauf vom 2026-08-11 wie ein Regelversagen aus, obwohl
        # dem Modell schlicht das Werkzeug fehlte.
        for tc, n in sorted(((t, tool_log.count(t)) for t in dict.fromkeys(tool_log)),
                            key=lambda x: -x[1]):
            print(f"               [Werkzeug] {n}× {tc}")
        if not tool_log:
            print("               [Werkzeug] keiner")
        for sp in spoken_log:
            print(f"               » {sp}")
        for n in dict.fromkeys(notes):
            print(f"               → {n}")
        print()
    if len(variants) > 1:
        # Die eigentliche Antwort bei einem Verbesserungsversuch: hat die
        # Aenderung geholfen, geschadet oder nichts getan — je Fall.
        width = max(len(k) for k in matrix) + 2
        print("\n── Vergleich " + "─" * 46)
        print("Fall".ljust(width) + "  ".join(v.rjust(7) for v in variants))
        for cid, per in matrix.items():
            row = "  ".join((per.get(v, "—")).rjust(7) for v in variants)
            vals = [per.get(v) for v in variants]
            mark = ""
            if len(set(vals)) > 1:
                mark = "   ← unterschiedlich"
            print(cid.ljust(width) + row + mark)
    runs = len(cases) * len(variants)
    print(f"\nErgebnis: {runs-total_fail}/{runs} bestanden "
          f"({len(cases)} Fälle × {len(variants)} Variante(n))")
    return 1 if total_fail else 0

sys.exit(asyncio.run(main()))
