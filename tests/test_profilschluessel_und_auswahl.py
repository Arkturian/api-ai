"""Vertikalschnitt 3: profilgebundener OpenAI-Key, Sortierung, Fokus-Zwang.

Drei Befunde aus echten Sitzungen (Owner-Test 2026-08-27):

* „Zeig mir die fünf besten Helme" → 782 Treffer und Rückfragen.
* „Welche Größe habe ich gewählt?" → „Nein."
* „Erzähl mir was über den gewählten Helm" → „Bitte öffnen Sie einen",
  obwohl `focused:true` im Protokoll stand. Das Modell hat GERATEN,
  statt nachzusehen.

Alle drei sind aus dem Gesprächsprotokoll gefunden worden, das es
gestern noch nicht gab.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")
from ai.routes import realtime_routes as rr  # noqa: E402


# ───────────────────────── profilgebundener Schlüssel

@pytest.fixture(autouse=True)
def keys_weg(monkeypatch):
    for n in list(os.environ):
        if n.startswith("OPENAI_API_KEY__"):
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "global")


def test_ohne_profileintrag_bleibt_alles_wie_bisher():
    """Kein Profil ohne eigenen Eintrag ändert sein Verhalten."""
    assert rr._openai_key_fuer("alex-default") == ("global", "global")
    assert rr._openai_key_fuer(None) == ("global", "global")


def test_profileintrag_gewinnt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY__PRODUCT_FINDER", "kunde")
    assert rr._openai_key_fuer("product-finder") == ("kunde", "profile")


@pytest.mark.parametrize("schreibweise", [
    "product-finder", "product_finder", "PRODUCT-FINDER", " product-finder ",
])
def test_beide_schreibweisen_treffen_denselben_eintrag(monkeypatch, schreibweise):
    """oneal schreibt `product-finder`, Cloud schreibt `product_finder`.
    Ohne Normalisierung wäre die Hälfte der Aufrufe still auf dem
    falschen Konto gelandet — ohne Fehler, ohne Meldung."""
    monkeypatch.setenv("OPENAI_API_KEY__PRODUCT_FINDER", "kunde")
    assert rr._openai_key_fuer(schreibweise)[0] == "kunde"


def test_leerer_profileintrag_gilt_als_nicht_gesetzt(monkeypatch):
    """Eine leere Zeile in der .env ist ein Tippfehler, kein Wunsch
    nach einem Mint ohne Schlüssel."""
    monkeypatch.setenv("OPENAI_API_KEY__PRODUCT_FINDER", "   ")
    assert rr._openai_key_fuer("product-finder") == ("global", "global")


def test_die_herkunft_wird_protokolliert():
    """Ohne sie ließe sich hinterher nicht feststellen, WELCHES Konto
    eine Sitzung bezahlt hat — genau die Frage, um die es geht."""
    import inspect
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert "_openai_key_fuer(grant.profile_id)" in quelle
    assert "OpenAI-Konto=%s" in quelle


def test_normalisierung_stimmt_mit_dem_budgetwaechter_ueberein():
    """Zwei Regeln für dieselbe Aussage driften. Hier wird es gemerkt."""
    from ai.services import realtime_budget_guard as g
    import inspect
    quelle = inspect.getsource(g._session_budget_eur)
    assert '.replace("-", "_").replace(".", "_").upper()' in quelle
    assert rr._profil_suffix("a-b.c") == "A_B_C"


# ───────────────────────── Sortierung und Menge im Schema

@pytest.mark.parametrize("werkzeug", ["find_products", "refine_search"])
def test_beide_suchwerkzeuge_kennen_limit_und_sort(werkzeug):
    t = [x for x in rr._product_finder_tools("O'Neal") if x["name"] == werkzeug][0]
    p = t["parameters"]["properties"]
    assert p["limit"] == {"type": "integer", "minimum": 1, "maximum": 50}
    assert p["sort"]["enum"] == ["newest", "price_desc", "price_asc"]


def test_details_bekommt_keine_sortierung():
    """Es wählt nicht aus, es fragt nach dem bereits Gewählten."""
    t = [x for x in rr._product_finder_tools(None)
         if x["name"] == "product_details"][0]
    assert "limit" not in t["parameters"]["properties"]
    assert "sort" not in t["parameters"]["properties"]


def test_limit_und_sort_gehen_als_kriterien_mit():
    """Vertrag mit OnealServ-Codex: beide liegen INNERHALB `criteria`,
    damit `refine_search` sie vom Basis-Token erbt. Mein vorhandener
    Kriterienpfad reicht sie deshalb durch — kein Sonderweg."""
    felder = rr._product_finder_kriterien_felder()
    assert "limit" in felder and "sort" in felder


# ───────────────────────── Persona

def test_der_sprecher_nennt_nicht_das_sortiment():
    """`count` ist die GEZEIGTE Menge, nicht die gefundene. „Wir haben
    fünf" wäre eine falsche Auskunft über den Sortimentsumfang."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "WIE VIELE DU ZEIGST" in t
    assert "niemals 'wir " in t


def test_der_sprecher_sagt_nicht_die_besten():
    """Der Katalog kennt keine Bewertung. Neu und teuer ist keine
    Qualität — und der Außendienstler gibt die Behauptung weiter."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "sag NICHT 'die besten'" in t
    assert "fuenf neuesten" in t


def test_der_sprecher_liest_nur_applied_werte():
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "applied_sort" in t and "applied_limit" in t
    assert "nie aus dem, was du" in t


def test_keine_groessenrueckfrage_bei_wenigen_treffern():
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "frag " in t and "NICHT nach der Groesse" in t


def test_der_sprecher_muss_erst_nachsehen_bevor_er_ablehnt():
    """Der Befund aus dem Log: `focused:true`, aber das Modell
    antwortete „bitte öffnen Sie einen" — ohne das Werkzeug zu rufen.
    Es hat geraten, was auf dem Bildschirm steht."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "ERST NACHSEHEN" in t
    assert "Behaupte NIE ohne Aufruf" in t


def test_der_sprecher_nutzt_die_gewaehlte_variante():
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "gewaehlte Groesse oder Farbe" in t


# ───────────── Einstiegskategorie ist kein Zaun (Log 15:28)

def test_der_einstieg_begrenzt_die_suche_nicht():
    """Befund aus `7e5d7d86`: Einstieg über MX-Helme, Frage nach einem
    Oberteil → 0 Treffer, weil die Suche im Einstiegs-Scope lief.
    Ohne Scope liefert dieselbe Suche fünf."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "KEIN ZAUN" in t


def test_vorbelegung_wird_ausgesprochen():
    """Sucht der Sprecher still im Zaun und meldet „nichts gefunden",
    ist das die dritte Verwechslung derselben Achse: „führen wir
    nicht" gegen „ich habe nur hier gesucht"."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "defaults_applied" in t
    assert "dort nicht" in t


def test_die_drei_verwechslungen_stehen_alle_in_der_persona():
    """Sortiment / Störung / Vorbelegung — drei verschiedene Gründe
    für „nichts", die im Gespräch gleich klingen."""
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "fuehren wir davon nichts" in t      # Sortiment
    assert "Katalog antwortet gerade" in t       # Stoerung
    assert "dort nicht" in t                     # Vorbelegung


def test_die_gewaehlte_groesse_gilt_nur_fuer_dieses_produkt():
    """Folgefehler meiner eigenen Regel, gefunden an OnealServ-Codex'
    Befund `ee0142b`: Dort wurde die Helmgröße `S/M` serverseitig in
    eine Jersey-Suche übernommen. Der Server räumt das jetzt aus —
    meine Persona hätte es wieder hineingeredet, weil „frag nicht nach
    der Größe" ohne Bindung an das Produkt dastand.
    """
    t = rr._product_finder_prompt("de", "O'Neal")
    assert "NUR fuer dieses Produkt" in t
    assert "Wechselt er die " in t
    assert "darfst und sollst du neu fragen" in t
