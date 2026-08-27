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

def test_genau_vier_werkzeuge():
    """Seit #1404 drei. Die Zahl steht hier fest, damit ein viertes
    Werkzeug nicht unbemerkt in die Hand des Modells wandert — die
    Werkzeugliste IST die Angriffsflaeche."""
    namen = [t["name"] for t in rr._product_finder_tools()]
    assert namen == ["find_products", "refine_search",
                     "cart_details", "product_details"]


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
    nicht nur in einem Dokument, das niemand zur Laufzeit aufschlaegt.

    Gilt fuer die SUCHWERKZEUGE. `product_details` ist die bewusste
    Ausnahme (#1404): Es liefert Name, Material und Ausstattung, weil
    der Agent genau darueber sprechen soll. Die Grenze verschiebt sich
    dort vom „was darf das Modell sehen" zum „wer waehlt aus" — und
    auswaehlen darf weiterhin nur der Server.
    """
    kompakt = [t for t in rr._product_finder_tools()
               if t["name"] in ("find_products", "refine_search")]
    assert len(kompakt) == 2
    for t in kompakt:
        d = t["description"]
        assert "keine Produkt-IDs" in d
        assert "keine Namen" in d
        assert "Einzelpreise" in d


def test_details_ist_die_einzige_ausnahme_von_der_kompakten_form():
    """Die Ausnahme muss EINE bleiben. Ruecken weitere Werkzeuge nach,
    die Produkttext liefern, soll dieser Test rot werden und nicht
    stillschweigend mitwachsen."""
    mit_text = sorted(t["name"] for t in rr._product_finder_tools()
                      if "keine Namen" not in t["description"])
    # ZWEI benannte Ausnahmen, bewusst erweitert (Warenkorb, #1426):
    # `product_details` liefert Produkttext, weil der Agent darueber
    # sprechen soll; `cart_details` liefert die Artikel des Kunden,
    # weil es SEINE sind. Ein dritter Kandidat faerbt diesen Test
    # wieder rot — das ist der Zweck, nicht das Hindernis.
    assert mit_text == ["cart_details", "product_details"]


def test_kein_kriterium_heisst_wie_ein_rueckgabefeld():
    """Kein `ids`, `name`, `description` als EINGABE — sonst laedt die
    Werkzeugform selbst dazu ein, Produktdaten hin- und herzureichen."""
    for t in rr._product_finder_tools():
        felder = set(t["parameters"]["properties"])
        assert not (felder & {"ids", "product_ids", "name", "description"})


def test_verfeinern_verlangt_kein_token_mehr_vom_modell():
    """Umgekehrt seit #1398.

    Der Gedanke stimmt weiter — ohne Token waere `refine_search` eine
    zweite Volltextsuche. Falsch war nur, WER das Token liefert. Es
    darf den Modellkontext nie erreichen (oneal `0e3ea84`), also kann
    das Modell es nicht mitschicken; ein Pflichtfeld dafuer erzeugt nur
    erfundene Werte. Der Server haelt es an der Sitzung.
    """
    t = [x for x in rr._product_finder_tools() if x["name"] == "refine_search"][0]
    assert not t["parameters"].get("required")
    assert "selection_token" not in t["parameters"]["properties"]


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


# ────────────────────────────── Sitzungs-Scope (OnealServ-Codex, #4831)

def test_markennamen_sind_die_echten_facet_werte():
    """Der Fehler, den ich hier selbst gemacht habe.

    Ich hatte `("O'Neal", "ONE", "KINI")` geschrieben — abgeleitet aus
    Kurzlabels in einer Zusammenfassung, nicht aus dem Katalog. Der
    Enum haette jeden echten Wert abgewiesen, und zwar erst beim ersten
    Kundengespraech. Wer Bezeichner aus Prosa ableitet, rechnet mit
    einer Schreibweise, die nie jemand zugesagt hat.
    """
    assert rr.PRODUCT_FINDER_BRANDS == ("O'Neal", "ONE Industries", "Kini Red Bull")


def test_markenoffen_ist_ein_zustand_kein_versehen():
    """`brand=None` heisst markenoffener Katalog — die Flows `open` und
    `direct` ueberspringen die Marke bewusst. Dann MUSS das Kriterium
    waehlbar bleiben, sonst kann der Agent dort gar nicht filtern."""
    offen = {t["name"]: t for t in rr._product_finder_tools(None)}
    assert "brand" in offen["find_products"]["parameters"]["properties"]
    gebunden = {t["name"]: t for t in rr._product_finder_tools("O'Neal")}
    assert "brand" not in gebunden["find_products"]["parameters"]["properties"]


def test_persona_nennt_bei_bindung_nur_die_eine_marke():
    text = rr._product_finder_prompt("de", "Kini Red Bull")
    assert "Kini Red Bull" in text
    assert "ONE Industries" not in text


@pytest.mark.parametrize("werkzeug", ["find_products", "refine_search"])
def test_jahr_ist_kein_modell_kriterium(werkzeug):
    """Das Jahr ist Sitzungs-Scope und wird serverseitig injiziert.

    Ohne Angabe setzt der Katalog sein eigenes Jahr. Fuehrt der
    Sprachpfad es nicht mit, trifft seine Auswahl womoeglich eine
    andere Menge als die sichtbare Oberflaeche — **und nichts schlaegt
    dabei fehl**. Genau diese Sorte lautloser Abweichung.
    """
    for gebunden in (None, "O'Neal"):
        t = {x["name"]: x for x in rr._product_finder_tools(gebunden)}[werkzeug]
        assert "collection_year" not in t["parameters"]["properties"]


def test_mint_kennt_die_beiden_scope_felder():
    felder = rr.RealtimeTokenRequest.model_fields
    for name in ("brand", "collection_year", "entry_selection"):
        assert name in felder, f"{name} fehlt an der Anfrage"
        assert felder[name].default is None


def test_unbekannte_marke_wird_abgewiesen_statt_fallengelassen():
    """Fail-closed, und zwar AUFGERUFEN statt im Quelltext gesucht.

    Die erste Fassung dieses Tests suchte nur die Zeichenkette
    `"error": "unknown_brand"` im Quelltext. Sie blieb gruen, als ich
    die Bedingung zur Gegenprobe auf `if False:` setzte — die
    Fehlermeldung stand ja weiter da, nur unerreichbar. Deshalb sitzt
    die Pruefung jetzt in einer eigenen Funktion, die man rufen kann.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        rr._product_finder_brand("Fox Racing")
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "unknown_brand"
    assert exc.value.detail["brand"] == "Fox Racing"
    assert sorted(rr.PRODUCT_FINDER_BRANDS) == exc.value.detail["known"]


@pytest.mark.parametrize("roh,erwartet", [
    (None, None), ("", None), ("   ", None),
    ("O'Neal", "O'Neal"), ("  Kini Red Bull  ", "Kini Red Bull"),
])
def test_markenpruefung_normalisiert_und_laesst_markenoffen_zu(roh, erwartet):
    assert rr._product_finder_brand(roh) == erwartet


def test_mint_benutzt_die_pruefung():
    import inspect
    assert "_product_finder_brand(request.brand)" in inspect.getsource(
        rr.mint_realtime_token)


def test_jahr_wird_im_mint_gesetzt_auch_ohne_angabe():
    import inspect
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert "PRODUCT_FINDER_DEFAULT_YEAR" in quelle


# ───────────────────── Form der Kriterien (committeter Vertrag c887a89)

def test_sport_ist_eine_liste_mit_MX_und_MTB():
    """Zweite geratene Schreibweise an derselben Stelle.

    Ich hatte `{"type": "string", "description": "moto oder mtb"}`
    geschrieben. Der committete Vertrag sagt: Liste, Werte `MX`/`MTB`.
    Beide Male haette das Modell etwas geliefert, das die Route mit
    `422` abweist — sichtbar erst im Gespraech.
    """
    t = {x["name"]: x for x in rr._product_finder_tools()}["find_products"]
    sport = t["parameters"]["properties"]["sport"]
    assert sport["type"] == "array"
    assert sport["items"]["enum"] == ["MX", "MTB"]


def test_kategorie_ist_eine_liste_von_slugs():
    t = {x["name"]: x for x in rr._product_finder_tools()}["find_products"]
    kat = t["parameters"]["properties"]["category"]
    assert kat["type"] == "array"
    assert kat["items"]["type"] == "string"
