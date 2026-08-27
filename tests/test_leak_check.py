"""Der Leak-Wächter muss selbst geprüft sein.

Ein Wächter, der nichts findet, sieht aus wie ein sauberer Kern. Die
Tests halten ihn deshalb gegen eingebaute Treffer — sonst wäre er die
Behauptung, kundenfrei zu sein, statt der Nachweis.
"""

import os
import subprocess
import sys

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKRIPT = os.path.join(WURZEL, "scripts", "tenant_leak_check.py")


def _lauf(*args):
    return subprocess.run([sys.executable, SKRIPT, *args],
                          capture_output=True, text=True, cwd=WURZEL)


def test_laeuft_und_meldet_den_stand():
    r = _lauf()
    assert "Kundenbezuege im Kern:" in r.stdout


def test_heute_gruen_weil_nichts_neu_ist():
    """Meldend, nicht blockierend — solange nichts dazukommt."""
    assert _lauf().returncode == 0


def test_er_findet_die_bekannten_stellen():
    """Fände er nichts, wäre der Kern scheinbar sauber."""
    basis = os.path.join(WURZEL, "tenant-leak-baseline.txt")
    inhalt = open(basis, encoding="utf-8").read()
    # Welche Gruende noch vorkommen, aendert sich mit jedem Schritt des
    # Umbaus — die Katalog-Slugs sind seit dem Datenschnitt weg. Der
    # Test prueft deshalb, dass der Waechter UEBERHAUPT etwas benennt
    # und die Baseline nicht leer behauptet, was sie nicht ist.
    eintraege = [z for z in inhalt.splitlines()
                 if z.strip() and not z.startswith("#")]
    r = _lauf()
    stand = int(r.stdout.split("Kern:")[1].split()[0])
    assert len(eintraege) == stand, "Baseline und Stand muessen uebereinstimmen"
    if stand:
        assert any("\t" in z for z in eintraege), "Grund fehlt am Eintrag"


def test_ein_neuer_bezug_laesst_ihn_fehlschlagen(tmp_path):
    """Die Verschärfung gegenüber dem Auftrag: Er blockiert nicht erst
    bei Baseline 0, sondern sobald etwas DAZUKOMMT. Ein Wächter, der
    einen wachsenden Berg nur meldet, gewöhnt seine Leser daran, dass
    er rot ist."""
    ziel = os.path.join(WURZEL, "ai", "_leak_probe_tmp.py")
    try:
        with open(ziel, "w", encoding="utf-8") as f:
            f.write("MARKE = \"O'Neal\"\n")
        r = _lauf()
        assert r.returncode == 1
        assert "NEU HINZUGEKOMMEN" in r.stdout
        assert "_leak_probe_tmp.py" in r.stdout
    finally:
        if os.path.exists(ziel):
            os.remove(ziel)
    assert _lauf().returncode == 0, "nach dem Aufräumen wieder grün"


def test_beseitigte_stellen_werden_gemeldet_nicht_verschwiegen():
    """Sonst merkt niemand, dass die Baseline nachzuziehen wäre — und
    sie bliebe als Ausrede für spätere Treffer stehen."""
    import inspect
    quelle = open(SKRIPT, encoding="utf-8").read()
    assert "beseitigt" in quelle
    assert "--update" in quelle
