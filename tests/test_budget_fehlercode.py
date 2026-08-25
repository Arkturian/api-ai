"""Der Fehlercode behauptet über das Fenster GAR NICHTS mehr (#1314).

Er hieß `daily_budget_exceeded`, unabhängig davon, worauf
`REALTIME_BUDGET_WINDOW` stand. Beide Clients übersetzten ihn wörtlich —
Alexanders Stimme sagte fünf Tage lang „morgen wieder verfügbar",
während das Fenster erst am Monatsersten aufging. Im August kostete
derselbe Fall ihn nochmals fünf Tage.

**Warum `budget_exceeded` und nicht `monthly_budget_exceeded`:** Alex'
Auftrag lautete wörtlich „mach den Fehlercode fenster-richtig", AppDev
empfahl im Review den fenster-LOSEN Namen. AppDevs Fassung ist die
bessere, und zwar aus demselben Grund, der den Fehler erzeugt hat: Ein
Fenster im Namen ist eine zweite Wahrheitsquelle neben dem Feld
`window`. Genau so ist der alte Name entstanden — er hat eine
Umstellung nicht mitgemacht und alterte lautlos. Ein Name, der nichts
behauptet, kann auch nicht falsch werden.
"""

import pytest

from ai.services import realtime_budget_guard as g


def test_code_nennt_kein_fenster():
    e = g.BudgetExceeded("alex-default", 32.21, 30.0)
    assert e.error_code == "budget_exceeded"
    assert "daily" not in e.error_code
    assert "monthly" not in e.error_code


@pytest.mark.parametrize("fenster", ["daily", "monthly"])
def test_code_bleibt_gleich_egal_welches_fenster(monkeypatch, fenster):
    """Der Kern der Umstellung: Der Name haengt nicht mehr am Schalter.

    Wer den Code auf Gleichheit prueft, darf durch ein Umkonfigurieren
    des Fensters nicht kaputtgehen — und wer das Fenster wissen will,
    muss ins Feld schauen.
    """
    monkeypatch.setattr(g, "BUDGET_WINDOW", fenster)
    e = g.BudgetExceeded("p", 5.0, 1.0)
    assert e.error_code == "budget_exceeded"
    assert e.public_fields["window"] == fenster


def test_das_fenster_steht_im_feld_und_nur_dort(monkeypatch):
    monkeypatch.setattr(g, "BUDGET_WINDOW", "monthly")
    e = g.BudgetExceeded("p", 32.21, 30.0)
    assert e.public_fields["window"] == "monthly"
    assert e.public_fields["resets_at"].split("T")[0].endswith("-01")


def test_alter_klassenname_faengt_weiterhin():
    """`except DailyBudgetExceeded` muss weiter fangen, was es meint.

    Ein umbenanntes Symbol, das still nicht mehr fängt, wäre ein
    Ausfall an genau der Stelle, die Geld begrenzt — der Aufruf liefe
    dann durch statt gesperrt zu werden.
    """
    assert g.DailyBudgetExceeded is g.BudgetExceeded
    try:
        raise g.BudgetExceeded("p", 5.0, 1.0)
    except g.DailyBudgetExceeded as e:
        assert e.error_code == "budget_exceeded"


def test_reserve_wirft_den_neuen_typ():
    """Der Pfad, der wirklich sperrt — nicht nur die Klasse daneben."""
    import inspect
    quelle = inspect.getsource(g.reserve_mint)
    assert "raise BudgetExceeded(" in quelle
    assert "raise DailyBudgetExceeded(" not in quelle


def test_kein_alter_code_mehr_auf_der_leitung():
    """Die Zeichenkette darf im Modul nur noch in Prosa vorkommen.

    Gemessen wird der QUELLTEXT ohne Kommentare und Docstrings: Die
    Vorgeschichte gehört dokumentiert, aber nichts darf den alten Code
    mehr AUSGEBEN.
    """
    import ast
    baum = ast.parse(open(g.__file__).read())
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
            # Docstrings sind Expr-Statements; die zaehlen nicht.
            pass
    ausgaben = [
        n.value for n in ast.walk(baum)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and n.value == "daily_budget_exceeded"
    ]
    assert not ausgaben, "der alte Code steht noch als Wert im Modul"
