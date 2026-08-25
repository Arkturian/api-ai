"""Betriebsart `product-finder` — Phase 1d des freigegebenen Bauplans.

Quelle: Content-Post #4831, Product-Section `p-dabc0367a5d6`, abgezeichnet
von OnealServ-Codex und Tschepp-Codex2 am 2026-08-25.

Die Regeln hier sind keine Stilfragen. Jede einzelne stammt aus einem
Fehler, den jemand schon bezahlt hat.
"""

import pytest

from ai.routes import realtime_routes as rr


# ─────────────────────────────────────────────────── Betriebsart

def test_modus_ist_angemeldet():
    assert "product-finder" in rr.SUPPORTED_COMPANION_MODES


def test_mint_verdrahtet_die_betriebsart():
    """Ein Modus in der Liste ohne Zweig im Mint waere eine Attrappe."""
    import inspect
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert 'companion_mode == "product-finder"' in quelle
    assert "_product_finder_prompt(" in quelle
    assert "_product_finder_tools(" in quelle


# ─────────────────────────────────────────────────── Werkzeuge

def test_genau_zwei_werkzeuge():
    namen = [t["name"] for t in rr._product_finder_tools()]
    assert namen == ["find_products", "refine_search"]


def test_kein_anzeigewerkzeug_in_der_hand_des_modells():
    """Tschepp-Codex2s Korrektur am ersten Entwurf.

    Ich hatte `show_products(ids)` als Modellwerkzeug vorgeschlagen.
    Dann kann das Modell IDs *erfinden* — bei einem Aussichtspunkt ist
    das peinlich, bei einem Produkt oeffnet es einen Artikel, den es
    nicht gibt, oder den falschen Preis. Der Anzeigebefehl reist
    stattdessen im geprueften Serverresultat.
    """
    namen = [t["name"] for t in rr._product_finder_tools()]
    assert not [n for n in namen if n.startswith("show_")]
    assert not [n for n in namen if "display" in n or "render" in n]


def test_werkzeugbeschreibung_verlangt_die_kompakte_form():
    """Der Vertrag steht in der Beschreibung, die das Modell liest —
    nicht nur in einem Dokument, das niemand zur Laufzeit aufschlaegt."""
    for t in rr._product_finder_tools():
        d = t["description"]
        assert "keine Produkt-IDs" in d
        assert "keine Namen" in d
        assert "Einzelpreise" in d


def test_kein_kriterium_heisst_wie_ein_rueckgabefeld():
    """Kein `ids`, `name`, `description` als EINGABE — sonst laedt die
    Werkzeugform selbst dazu ein, Produktdaten hin- und herzureichen."""
    for t in rr._product_finder_tools():
        felder = set(t["parameters"]["properties"])
        assert not (felder & {"ids", "product_ids", "name", "description"})


def test_verfeinern_braucht_das_token_der_laufenden_suche():
    """Ohne Token waere `refine_search` eine zweite Volltextsuche —
    und die Trefferfolge nicht mehr dieselbe."""
    t = [x for x in rr._product_finder_tools() if x["name"] == "refine_search"][0]
    assert t["parameters"]["required"] == ["selection_token"]


@pytest.mark.parametrize("werkzeug", ["find_products", "refine_search"])
def test_zusatzfelder_sind_verboten(werkzeug):
    t = [x for x in rr._product_finder_tools() if x["name"] == werkzeug][0]
    assert t["parameters"]["additionalProperties"] is False


# ─────────────────────────────────────────────────── Persona

def test_persona_nennt_kein_einziges_werkzeug():
    """Die teuerste Zeile des Arcturian-Tages, hier als Test.

    Gemessen: 8/8 sauber ohne Erwaehnung, 1/8 mit Aufforderung, 0/8 mit
    ausdruecklichem Verbot. Ein Verbot ist eine Erwaehnung.
    """
    text = rr._product_finder_prompt("de")
    for name in ("find_products", "refine_search", "show_product_results",
                 "prepare_cart", "submit_order", "compare_selected"):
        assert name not in text, f"{name} steht in der Persona"


def test_persona_traegt_die_marken_aber_keine_farb_oder_groessenlisten():
    """Die Grenze verlaeuft bei der ANZAHL der Werte, nicht bei der
    Wichtigkeit: 3 Marken kosten fast nichts, 50 Farben und 239
    Groessen kosten sie in JEDER Antwort erneut."""
    text = rr._product_finder_prompt("de")
    for marke in rr.PRODUCT_FINDER_BRANDS:
        assert marke in text
    # Ein paar echte Groessen- und Farbwerte duerfen NICHT als Liste
    # dastehen. `L` als Buchstabe kommt in Prosa vor — deshalb auf
    # Aufzaehlungen pruefen, nicht auf Einzelvorkommen.
    for liste in ("XS, S, M", "S, M, L", "schwarz, weiss", "rot, blau"):
        assert liste not in text


def test_persona_trennt_leer_von_gestoert():
    """Tschepps Fehlklasse: „Ausfall ist kein leeres Regal."

    Im Produktkontext ist die Verwechslung schlimmer als peinlich —
    „das fuehren wir nicht" bei einer Stoerung ist eine falsche
    Auskunft ueber das Sortiment, gegenueber einem Kunden, der danach
    woanders kauft.
    """
    text = rr._product_finder_prompt("de")
    assert "fuehren wir davon nichts" in text
    assert "Katalog antwortet gerade" in text
    assert "FALSCHE" in text and "AUSKUNFT" in text


def test_persona_sagt_zur_passform_ehrlich_nein():
    """0 von 35.796 Varianten tragen Messwerte (OnealServ-Codex, live
    geprueft). Der Satz faellt in jedem zweiten Gespraech — er braucht
    eine wahre Antwort, keine Ausrede."""
    text = rr._product_finder_prompt("de")
    assert "Koerpermassen" in text
    assert "kannst du NICHT" in text or "kann ich" in text
    assert "Ich zeig Ihnen, was der Hersteller angibt" in text


def test_persona_verbietet_erfundene_produktnamen():
    text = rr._product_finder_prompt("de")
    assert "Erfinde niemals einen Produktnamen" in text
