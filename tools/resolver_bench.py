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
sys.path.insert(0, "/var/www/api-ai.arkturian.com")

import websockets
from ai.routes.realtime_routes import (
    _companion_arcturian_prompt, _arcturian_resolver_addendum,
    _arcturian_resolver_followup_payload, ARCTURIAN_RESOLVER_V2,
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
        "history": "Wir haben vorhin lange über 3dApi gesprochen.",
        "say": "Schick AppDevV2 eine Testnachricht.",
        "expect": lambda a: (a.get("target") or "").lower().startswith("appdevv2"),
        "why": "Der teuerste Fehler des 07.08.: Ziel kam aus dem Verlauf (3dApi).",
    },
    {
        "id": "P0-5 Schreibweise ist kein Grund zur Rueckfrage",
        "say": "sende an appdevv2 dass der build laeuft",
        "expect": lambda a: a.get("decision") == "action"
                  and (a.get("target") or "").lower().replace(" ", "").startswith("appdevv2"),
        "why": "Kleinschreibung darf keinen Zug fuer eine Rueckfrage kosten.",
    },
]


async def one(key, case):
    persona = _companion_arcturian_prompt("de") + _arcturian_resolver_addendum("de")
    followup = _arcturian_resolver_followup_payload(ARCTURIAN_RESOLVER_V2)
    t0 = time.time()
    async with websockets.connect(
        URL, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None
    ) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "model": MODEL, "instructions": persona,
            "tools": [], "tool_choice": "none",
            "audio": {"input": {"turn_detection": None}},
        }}))
        for text in ([case["history"]] if case.get("history") else []) + [case["say"]]:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": text}]}}))
        await ws.send(json.dumps(followup))
        args, name = "", None
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
    ms = int((time.time() - t0) * 1000)
    try:
        parsed = json.loads(args) if args else {}
    except Exception:
        return {"ok": False, "err": f"unparsbar: {args[:80]}", "ms": ms, "tool": name}
    return {"ok": True, "args": parsed, "ms": ms, "tool": name}


async def main():
    key = os.environ["OPENAI_API_KEY"]
    import hashlib, subprocess
    persona = _companion_arcturian_prompt("de") + _arcturian_resolver_addendum("de")
    sha = subprocess.run(["git","-C","/var/code/api-ai","rev-parse","--short","HEAD"],
                         capture_output=True, text=True).stdout.strip()
    run_id = f"rb-{int(time.time())}"
    print(f"Run-ID           : {run_id}")
    print(f"Modell           : {MODEL}")
    print(f"Resolver-Vertrag : {ARCTURIAN_RESOLVER_V2}")
    print(f"Persona-Revision : {sha} · {len(persona)} Zeichen · sha256 {hashlib.sha256(persona.encode()).hexdigest()[:12]}")
    print(f"Fälle            : {len(CASES)} × {REPS} Läufe\n")
    total_fail = 0
    for c in CASES:
        oks, lat, notes = 0, [], []
        for _ in range(REPS):
            r = await one(key, c)
            if not r["ok"]:
                notes.append(r["err"]); continue
            lat.append(r["ms"])
            if r["tool"] != "resolve_arcturian_turn":
                notes.append(f"falsches Werkzeug: {r['tool']}"); continue
            if c["expect"](r["args"]):
                oks += 1
            else:
                a = r["args"]
                notes.append(f"decision={a.get('decision')} kind={a.get('kind')} target={a.get('target')!r}")
        verdict = "BESTANDEN" if oks == REPS else "DURCHGEFALLEN"
        if oks != REPS:
            total_fail += 1
        med = f"{int(statistics.median(lat))} ms" if lat else "—"
        print(f"{verdict:14} {c['id']}   {oks}/{REPS}   Median {med}")
        print(f"               {c['why']}")
        for n in dict.fromkeys(notes):
            print(f"               → {n}")
        print()
    print(f"Ergebnis: {len(CASES)-total_fail}/{len(CASES)} Fälle bestanden")
    return 1 if total_fail else 0

sys.exit(asyncio.run(main()))
