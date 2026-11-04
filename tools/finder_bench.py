"""Produktfinder-Pruefstand: ruft der Berater das Werkzeug, statt zu raten?

Anlass (OnealServ, 04.11.2026): Bei offenem Produkt fragte der Kunde
„Siehst du das ausgewählte Produkt?" — das Modell rief kein Werkzeug und
sagte „Öffne am besten eins". Bei „mehr in Blau" fragte es nur zurück.

Misst die ENTSCHEIDUNG des ersten Antwortzugs (welches Werkzeug, oder
nur Text) gegen eine echte Realtime-Sitzung mit der Persona und den acht
Werkzeugen, so wie der Mint sie ausliefert. Textmodus, kein Audio: Die
Wortwahl wird nicht bewertet, nur ob nachgesehen bzw. gesucht wird.

Aufruf (Schluessel aus der Dienst-.env, nie auf der Konsole):
  sudo -E /var/www/api-ai.arkturian.com/venv/bin/python tools/finder_bench.py
Umgebung: BENCH_REPS (Vorgabe 6), BENCH_SRC (Codebaum, Vorgabe: dieser),
BENCH_LANG (Vorgabe de), BENCH_AUDIO=1 (Audio-Ausgabe wie im Finder),\nBENCH_ONLY=<Fall-Praefix>.
Schalter wie im Dienst: REALTIME_FINDER_PRESENTATION=on fuer die Faelle E/F.
Kosten: ~0,02 EUR je Lauf (rund 3,5k Text-Token Eingabe).
"""
import asyncio, json, os, sys
from pathlib import Path

SRC = os.environ.get("BENCH_SRC") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, SRC)
import logging
logging.disable(logging.CRITICAL)
import websockets
from ai.routes import realtime_routes as rr

MODEL = "gpt-realtime"
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
REPS = int(os.environ.get("BENCH_REPS", "6"))
LANG = os.environ.get("BENCH_LANG", "de")

SUCHE = ("find_products", "refine_search")
VORLAUF = [
    ("user", "Zeig mir MX-Jerseys."),
    ("call", "find_products", {"product_type": "Jersey", "sport": ["MX"]},
     {"shown": 12, "price_min": 39.9, "price_max": 89.9,
      "sizes": ["S", "M", "L", "XL", "XXL"], "applied_limit": 12}),
    ("assistant", "Ich zeig dir zwölf MX-Jerseys zwischen 40 und 90 Euro."),
]
GRUSS = [("assistant", "Hi! Ich helf dir, das Passende zu finden. Was suchst du?")]
FAELLE = [
    {"id": "A0 offenes Produkt direkt nach dem Gruss", "vorlauf": GRUSS,
     "say": "Siehst du das ausgewählte Produkt? Kannst du mir darüber etwas sagen?",
     "gut": ("product_details",)},
    {"id": "A offenes Produkt (Originalsatz)", "vorlauf": VORLAUF,
     "say": "Siehst du das ausgewählte Produkt? Kannst du mir darüber etwas sagen?",
     "gut": ("product_details",)},
    {"id": "A2 das hier", "vorlauf": VORLAUF,
     "say": "Was ist das hier für ein Material?",
     "gut": ("product_details",)},
    {"id": "B mehr Auswahl in Blau", "vorlauf": VORLAUF,
     "say": "Hast du noch mehr in Blau? Ich hätte gern mehr Auswahl.",
     "gut": SUCHE},
    {"id": "C Filter Preis", "vorlauf": VORLAUF,
     "say": "Nur die unter fünfzig Euro bitte.",
     "gut": SUCHE},
    {"id": "D Filter Farbe ohne Vorlauf", "vorlauf": [],
     "say": "Ich such einen roten Helm.",
     "gut": SUCHE},
    {"id": "E Outfit = Konfigurator", "vorlauf": GRUSS,
     "say": "Stell mir ein komplettes MX-Outfit in Blau zusammen.",
     "gut": SUCHE, "args": lambda a: a.get("presentation") == "outfit"
        and 3 <= (a.get("per_category_limit") or 0) <= 5},
    {"id": "F mehrere Kategorien = Auswahl", "vorlauf": GRUSS,
     "say": "Zeig mir alle blauen Helme, Jerseys und Hosen.",
     "gut": SUCHE, "args": lambda a: a.get("presentation") == "selection"
        and not a.get("per_category_limit")},
]


def _key() -> str:
    for z in Path("/var/www/api-ai.arkturian.com/.env").read_text().splitlines():
        if z.startswith("OPENAI_API_KEY="):
            return z.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENAI_API_KEY fehlt")


async def lauf(fall, key):
    persona = rr._product_finder_prompt(LANG, "O'Neal", ui_tools=True)
    tools = rr._product_finder_tools("O'Neal") + rr._product_finder_ui_tools()
    async with websockets.connect(
            URL, additional_headers={"Authorization": f"Bearer {key}"},
            max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "model": MODEL, "instructions": persona,
            "tools": tools, "tool_choice": "auto",
            "output_modalities": (["audio"] if os.environ.get("BENCH_AUDIO")
                                  else ["text"]),
            "audio": {"input": {"turn_detection": None}}}}))
        for i, e in enumerate(fall["vorlauf"]):
            if e[0] == "call":
                cid = f"c{i}"
                await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                    "type": "function_call", "call_id": cid, "name": e[1],
                    "arguments": json.dumps(e[2])}}))
                await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                    "type": "function_call_output", "call_id": cid,
                    "output": json.dumps(e[3])}}))
            else:
                ct = "input_text" if e[0] == "user" else "output_text"
                await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                    "type": "message", "role": e[0],
                    "content": [{"type": ct, "text": e[1]}]}}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": fall["say"]}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        werkzeug, text, args = None, "", ""
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            t = ev.get("type", "")
            if t == "response.output_item.added":
                it = ev.get("item") or {}
                if it.get("type") == "function_call" and not werkzeug:
                    werkzeug = it.get("name")
            elif t == "response.function_call_arguments.done" and not args:
                args = ev.get("arguments", "")
            elif t in ("response.output_text.delta",
                       "response.output_audio_transcript.delta"):
                text += ev.get("delta", "")
            elif t == "error":
                return {"err": str(ev)[:200]}
            elif t == "response.done":
                try:
                    a = json.loads(args) if args else {}
                except Exception:
                    a = {}
                return {"werkzeug": werkzeug, "args": a, "text": text.strip()[:120]}


async def main():
    key = _key()
    ergebnis = {}
    for fall in FAELLE:
        if os.environ.get("BENCH_ONLY") and not fall["id"].startswith(os.environ["BENCH_ONLY"]):
            continue
        treffer = 0
        for n in range(REPS):
            r = await lauf(fall, key)
            ok = (r.get("werkzeug") in fall["gut"]
                  and fall.get("args", lambda a: True)(r.get("args") or {}))
            treffer += ok
            print(f"  {fall['id']} #{n+1}: {'OK ' if ok else 'XX '}"
                  f"{r.get('werkzeug') or '-'} {json.dumps(r.get('args') or {}, ensure_ascii=False)[:110]} | "
                  f"{r.get('text') or r.get('err', '')}",
                  flush=True)
        ergebnis[fall["id"]] = f"{treffer}/{REPS}"
    print(json.dumps({"src": SRC, "lang": LANG, "ergebnis": ergebnis},
                     ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
