"""Jeder globale Namensgriff im Realtime-Modul muss auflösbar sein.

Ein Name, den Python als *global* übersetzt, den es aber nirgends
gibt: `owner` in `realtime_devlog_upsert`. Die Umstellung auf den
stabilen Eigner-Schlüssel (`owner_key`) hat zwei Vorkommen stehen
lassen — in der Protokollzeile und im Rückgabewert.

Der Verwandte von `arcturian_resolver` (500 auf jedem Arcturian-Mint,
~1 h), aber NICHT derselbe Fall und von diesem Test NICHT abgedeckt:
Dort war der Name lokal und nur zu früh benutzt (UnboundLocalError),
er steht also brav in `co_varnames`. Wer hier grün liest, hat über die
Reihenfolge von Zuweisungen nichts erfahren.

Was ihn so teuer macht: Er ist beim Import unsichtbar und wird erst
beim Aufruf zur Ausnahme. Im Devlog-Fall lag die Fundstelle NACH
`os.replace` — die Datei wurde geschrieben, danach flog der NameError,
und der Anrufer bekam 500 für einen Vorgang, der stattgefunden hat.

Der Test misst am echten Codeobjekt (`co_names` gegen `module.__dict__`),
nicht am Syntaxbaum: Ein AST-Lauf über Modul-Globals ist mir genau hier
schon einmal falsch geraten (Top-Level ist `ast.Assign`, nicht `ast.Name`),
und ein Werkzeug, das die Sache nicht anzeigt, hat sie nicht gemessen.
"""

import builtins
import types

from ai.routes import realtime_routes as rr


def _codeobjekte(obj):
    """Alle Codeobjekte, die WIRKLICH in diesem Modul stehen.

    Die Einschränkung auf `co_filename` ist nicht Kosmetik: `dir(rr)`
    liefert auch importierte Funktionen mit (`Depends`, `Field`,
    `exchange_and_verify`), deren Globals im FREMDEN Modul leben. Ohne
    den Filter meldet der Test 30 Fehlalarme und keinen echten Treffer —
    ein Wächter, der immer schreit, wird abgeschaltet.
    """
    gesehen = set()
    stapel = [getattr(obj, n, None) for n in dir(obj)]
    while stapel:
        wert = stapel.pop()
        code = getattr(wert, "__code__", None)
        while code is None and hasattr(wert, "__wrapped__"):
            wert = wert.__wrapped__
            code = getattr(wert, "__code__", None)
        if isinstance(wert, types.CodeType):
            code = wert
        if code is None or id(code) in gesehen:
            continue
        gesehen.add(id(code))
        if code.co_filename == obj.__file__:
            yield code
        stapel.extend(
            c for c in code.co_consts if isinstance(c, types.CodeType)
        )


def test_kein_globaler_name_ohne_ziel():
    vorhanden = set(vars(rr)) | set(dir(builtins))
    quelle = open(rr.__file__).read()
    fehlend = set()
    for code in _codeobjekte(rr):
        # Lokale Bindungen zählen mit: `import re` IM Rumpf landet in
        # co_names UND in co_varnames — das ist kein globaler Griff.
        lokal = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_freevars)
        for gesucht in code.co_names:
            if gesucht in vorhanden or gesucht in lokal:
                continue
            # co_names führt auch Attributnamen (`a.b`) — die sind hier
            # nicht unterscheidbar, also über die Quelle aussortieren.
            if f".{gesucht}" in quelle:
                continue
            fehlend.add((code.co_name, gesucht))
    assert not fehlend, (
        "Globale Namen ohne Ziel — beim AUFRUF ein NameError, nicht beim "
        f"Import: {sorted(fehlend)}"
    )
