"""Der Mint-Pfad darf keine Variable vor ihrer Zuweisung lesen (#1038).

Am 2026-08-11 um 19:42 warf JEDE Arcturian-Sprachsitzung einen 500:
`cannot access local variable 'arcturian_resolver'`. Der Zusatz war
versionsabhaengig geworden und las die Fassung — die Zuweisung stand
25 Zeilen weiter unten.

**Warum 160 gruene Tests das nicht gefunden haben:** Keiner von ihnen
ruft `mint_realtime_token` auf. Sie pruefen Bausteine (Werkzeuge,
Nutzlasten, Persona-Text) einzeln und in der richtigen Reihenfolge —
die Funktion, die sie zusammensetzt, war unbetreten. Ein Fehler, der
nur beim Zusammensetzen entsteht, ist fuer eine solche Suite unsichtbar.

Diese Datei prueft den Rumpf deshalb STATISCH: Jede lokale Variable
muss zugewiesen sein, bevor sie gelesen wird. Das faengt die ganze
Fehlerklasse, ohne eine Realtime-Sitzung zu brauchen.
"""

import ast
import inspect

from ai.routes import realtime_routes as rr


def _verstoesse(fname: str = "mint_realtime_token"):
    """Erster Lesezugriff vor erster Zuweisung — pro Name, nach Zeile.

    NICHT ueber `ast.walk` mit mitlaufender Menge: `walk` laeuft in
    Breite, nicht in Quelltextreihenfolge. Die erste Fassung dieses
    Waechters tat genau das und meldete die kaputte Datei als sauber —
    ein Test, der den Fehler nicht faengt, fuer den er geschrieben
    wurde. Gegengeprueft an `867ec87` (findet ihn) und am Fix (sauber).
    """
    baum = ast.parse(inspect.getsource(rr))
    knoten = next(
        k for k in ast.walk(baum)
        if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef)) and k.name == fname
    )
    von_aussen = {a.arg for a in knoten.args.args} | {a.arg for a in knoten.args.kwonlyargs}
    modul_namen = {n.name.split(".")[0] for n in ast.walk(baum) if isinstance(n, ast.alias)}
    modul_namen |= {k.name for k in baum.body
                    if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    modul_namen |= {t.id for k in baum.body if isinstance(k, ast.Assign)
                    for t in k.targets if isinstance(t, ast.Name)}

    erste_zuweisung, erster_lesezugriff = {}, {}
    for n in ast.walk(knoten):
        if not isinstance(n, ast.Name):
            continue
        ziel = erste_zuweisung if isinstance(n.ctx, ast.Store) else (
            erster_lesezugriff if isinstance(n.ctx, ast.Load) else None)
        if ziel is not None:
            ziel[n.id] = min(ziel.get(n.id, n.lineno), n.lineno)

    out = []
    for name, lese in erster_lesezugriff.items():
        if name in von_aussen or name in modul_namen or name in dir(__builtins__):
            continue
        zu = erste_zuweisung.get(name)
        if zu is not None and lese < zu:
            out.append((name, lese, zu))
    return sorted(out)


def test_mint_liest_keine_variable_vor_ihrer_zuweisung():
    v = _verstoesse()
    assert not v, (
        "Im Mint wird eine Variable gelesen, bevor sie zugewiesen ist — "
        f"jeder Aufruf dieses Pfads ist ein 500: {v}"
    )


def test_die_fassung_steht_vor_den_anweisungen():
    """Die konkrete Reihenfolge, die 19:42 gebrochen hat.

    Nicht redundant zum statischen Waechter: Diese Zusicherung benennt
    die Stelle, damit ein spaeterer Umbau nicht stillschweigend
    zurueckfaellt.
    """
    quelle = inspect.getsource(rr)
    zuweisung = quelle.index("arcturian_resolver = request.arcturian_resolver")
    verwendung = quelle.index("instructions += _arcturian_resolver_addendum")
    assert zuweisung < verwendung, (
        "Die Resolver-Fassung wird gelesen, bevor sie feststeht"
    )
