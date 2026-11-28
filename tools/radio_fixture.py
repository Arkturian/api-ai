#!/usr/bin/env python3
"""Prüf-Fixture für Projekt 32 (Vertrag radio-schnitt-v1, Beitrag 6630, Abschnitt „Gemeinsame Prüf-Fixture").

Erzeugt eine synthetische Aufnahme mit zwei Stimmen (OpenAI-TTS, wenige Cent),
Stereo 48 kHz, exakt 12,345 s:
  - „wir wir" (Wortwiederholung), ein „äh" in einer Lücke, zwei Laute in derselben Lücke,
  - eine Pause von 1,6 s, ein Bereich mit zwei Sprechern gleichzeitig,
  - eine inhaltliche Wiederholung ohne Neues.
Keine echten Personendaten.

Aufruf (braucht OPENAI_API_KEY): radio_fixture.py <ausgabe_dir>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ZEILEN = [
    # (Stimme, Text, Start in s)
    ("alloy", "Guten Morgen. Heute reden wir wir über die Kinderbetreuung.", 0.00),
    ("onyx", "Danke. Also, äh, ähm, Kinderbetreuung ist wichtig.", 3.60),
    ("alloy", "Und was heißt das konkret?", 7.90),          # danach 1,6 s Pause
    ("onyx", "Wie gesagt, Kinderbetreuung ist wichtig.", 9.20),
    ("alloy", "Genau.", 9.90),                              # spricht in Zeile 4 hinein
]
DAUER_S = 12.345
SR = 48000


def tts(stimme: str, text: str, ziel: Path) -> None:
    from openai import OpenAI
    r = OpenAI().audio.speech.create(model="gpt-4o-mini-tts", voice=stimme, input=text,
                                     response_format="wav")
    ziel.write_bytes(r.read())


def main() -> None:
    aus = Path(sys.argv[1])
    aus.mkdir(parents=True, exist_ok=True)
    teile = []
    for n, (stimme, text, start) in enumerate(ZEILEN):
        p = aus / f"zeile_{n}.wav"
        if not p.exists():
            tts(stimme, text, p)
        teile.append((p, start))
    eingaben, filter_, marken = [], [], []
    for n, (p, start) in enumerate(teile):
        eingaben += ["-i", str(p)]
        filter_.append(f"[{n}:a]aresample={SR},aformat=channel_layouts=stereo,adelay={int(start * 1000)}:all=1[a{n}]")
        marken.append(f"[a{n}]")
    filter_.append("".join(marken) + f"amix=inputs={len(teile)}:normalize=0,apad,atrim=end_sample={int(DAUER_S * SR)}[out]")
    ziel = aus / "fixture.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *eingaben, "-filter_complex", ";".join(filter_),
                    "-map", "[out]", "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", str(ziel)], check=True)
    print(ziel)


if __name__ == "__main__":
    main()
