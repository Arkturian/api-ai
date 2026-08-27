#!/usr/bin/env python3
"""Findet Kundennamen und Kundendaten im api-ai-Kern.

Warum es das gibt (Entscheidung Alex 2026-08-27): Beim Setzen des
Freigabe-Tags fiel auf, dass nicht nur Kundennamen im Kern stehen,
sondern Kundendaten — die Markenliste und 28 Katalogkategorien eines
Kunden als Konstanten. Bei jedem weiteren Kunden laege damit fremdes
Sortimentswissen im ausgelieferten Code.

**Meldend, nicht heilend.** Der Waechter faerbt nichts gruen, indem er
etwas entfernt; er zeigt, was da ist.

**Die Verschaerfung gegenueber dem Auftrag:** Er blockiert nicht erst,
wenn die Baseline 0 ist, sondern schon dann, wenn ein Treffer
DAZUKOMMT. Ein Waechter, der einen wachsenden Berg nur meldet,
gewoehnt seine Leser daran, dass er rot ist — und dann faellt der
naechste Eintrag nicht mehr auf. Der Berg darf schrumpfen, nie
wachsen.

Aufruf:
    python3 scripts/tenant_leak_check.py            # pruefen
    python3 scripts/tenant_leak_check.py --update   # Baseline neu schreiben
"""

from __future__ import annotations

import os
import re
import sys

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUELLE = os.path.join(WURZEL, "ai")
BASELINE = os.path.join(WURZEL, "tenant-leak-baseline.txt")

# Muster mit Begruendung. Der Text erscheint im Bericht — wer einen
# Treffer sieht, soll ohne Nachschlagen wissen, warum er stoert.
MUSTER = [
    (r"O'Neal|ONE Industries|Kini Red Bull", "Kundenname (Marke)"),
    (r"\b(helmets-mx|jerseys-offroad|jerseys-mtb|pants-mx|boots-mx|"
     r"protection-mx|protection-mtb|kids-wear-protection|goggles|"
     r"adv-pants|bags---backpacks|boots-adventure|casual-wear|"
     r"face-masks|gloves|grips|handlebars|helme-mtb-open-face|"
     r"helmets-mtb-full-face|helmets-street|jackets|leather-suits-road|"
     r"leisure-accessories|pants--shorts-mtb|rain-wear|shoes|"
     r"sunglasses|transportation)\b", "Katalog-Slug eines Kunden"),
    (r"[Ww]anderlaut", "Kundenname (Guide)"),
    (r"[Tt]scheppa", "Kundenname (Tscheppaschlucht)"),
    (r"def _companion_\w+_prompt|def _product_finder_prompt",
     "Persona als Code statt als Text je Profil"),
]

# Dateien, die per Definition kundenbezogen sein duerfen — heute keine.
AUSNAHMEN = ()


def _treffer() -> list:
    gefunden = []
    for wurzel, _, dateien in os.walk(QUELLE):
        if "__pycache__" in wurzel:
            continue
        for name in sorted(dateien):
            if not name.endswith(".py"):
                continue
            pfad = os.path.join(wurzel, name)
            rel = os.path.relpath(pfad, WURZEL)
            if rel in AUSNAHMEN:
                continue
            try:
                zeilen = open(pfad, encoding="utf-8").read().splitlines()
            except Exception:
                continue
            for nr, zeile in enumerate(zeilen, 1):
                for muster, grund in MUSTER:
                    if re.search(muster, zeile):
                        gefunden.append(f"{rel}:{nr}\t{grund}")
                        break
    return sorted(set(gefunden))


def _baseline() -> set:
    if not os.path.exists(BASELINE):
        return set()
    return {
        z.rstrip("\n") for z in open(BASELINE, encoding="utf-8")
        if z.strip() and not z.startswith("#")
    }


def main() -> int:
    jetzt = _treffer()
    if "--update" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8") as f:
            f.write("# Bekannte Kundenbezuege im Kern. Ziel: leer.\n")
            f.write("# Neue Eintraege lassen den Waechter fehlschlagen —\n")
            f.write("# der Berg darf schrumpfen, nie wachsen.\n")
            for z in jetzt:
                f.write(z + "\n")
        print(f"Baseline geschrieben: {len(jetzt)} Eintraege")
        return 0

    alt = _baseline()
    neu = [z for z in jetzt if z not in alt]
    weg = [z for z in alt if z not in jetzt]

    print(f"Kundenbezuege im Kern: {len(jetzt)}  (Baseline {len(alt)})")
    if weg:
        print(f"\n{len(weg)} beseitigt — Baseline mit --update nachziehen:")
        for z in weg[:10]:
            print("  - " + z)

    if neu:
        print(f"\nNEU HINZUGEKOMMEN ({len(neu)}) — das ist der Fehler:")
        for z in neu:
            print("  + " + z)
        return 1

    if not alt and jetzt:
        print("\nBaseline ist leer, es gibt aber Treffer — blockierend.")
        return 1

    if not jetzt:
        print("\nKern ist kundenfrei.")
    else:
        print("\nKeine neuen Bezuege. Der Berg schrumpft oder steht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
