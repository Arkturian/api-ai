"""Preise und zwischengespeicherte Eingabe im Realtime-Zähler (#1283).

Vorgeschichte: Am 2026-08-22 meldete der Realtime-Agent „Monthly Cap
Reached". CloudV2 maß nach und fand, dass 64 % des Monatstopfs auf
Text-EINGABE entfielen — 4.736.883 Tokens bei 963 Antworten, also 4.919
je Antwort, gegenüber 598 Sekunden Sprechzeit im ganzen Monat.

Der Befund stimmte. Die Rechnung daneben nicht, gleich zweifach:

1. `gpt-realtime` stand mit 5,00 USD je 1M Text-Eingabe in der Tabelle,
   die Preisliste sagt 4,00 — und mit 20,00 statt 16,00 für Ausgabe.
2. Zwischengespeicherte Eingabe wurde voll berechnet. Gemessen sind
   3.608 der 4.919 Tokens je Antwort der stabile Vorbau (Persona 2.378,
   Resolver-Zusatz 555, Werkzeuge 675) — also 73 %. Genau dieser Teil
   wird von OpenAI nach der ersten Antwort zwischengespeichert und
   kostet dann ein Zehntel.

Die Lehre, die den Test rechtfertigt: Der Zähler war in sich stimmig —
die vier Posten trafen den gemeldeten Betrag auf den Cent. Das ist kein
Beleg für Richtigkeit, sondern nur für Konsistenz. Ein Deckel, der auf
einer zu hohen Zahl zumacht, sperrt echte Arbeit aus.
"""

import pytest

from ai.services.openai_realtime_cost_tracker import (
    EUR_USD_RATE,
    OPENAI_REALTIME_PRICING,
    OpenAIRealtimeCostTracker,
)

# Abgerufen 2026-08-22, developers.openai.com/api/docs/pricing.
PREISLISTE_GPT_REALTIME = {
    "text_input_per_1m": 4.0,
    "text_input_cached_per_1m": 0.40,
    "audio_input_per_1m": 32.0,
    "audio_input_cached_per_1m": 0.40,
    "text_output_per_1m": 16.0,
    "audio_output_per_1m": 64.0,
}

# Der reale Augustverbrauch, aus dem Monatszähler auf arkserver.
AUGUST = dict(
    audio_input_tokens=146_876,
    audio_output_tokens=116_272,
    text_input_tokens=4_736_883,
    text_output_tokens=52_315,
)


@pytest.fixture
def zaehler():
    return OpenAIRealtimeCostTracker()


# ──────────────────────────────────────────────────────── Preistabelle

@pytest.mark.parametrize("posten,preis", sorted(PREISLISTE_GPT_REALTIME.items()))
def test_preise_entsprechen_der_liste(posten, preis):
    """Jeder Posten einzeln, damit die Meldung den Fehler benennt.

    Eine Sammelprüfung auf das ganze Dict sagt nur „ungleich" — bei
    sechs Zahlen ist das die Hälfte der Arbeit, die ein Test abnehmen
    soll.
    """
    assert OPENAI_REALTIME_PRICING["gpt-realtime"][posten] == preis


def test_rueckfall_zaehlt_zwischenspeicher_voll():
    """Unbekanntes Modell darf sich nicht billig rechnen.

    Sonst liesse sich der Deckel durch einen Tippfehler im Modellnamen
    umgehen — und zwar am stärksten genau über den Posten, der hier neu
    dazukam.
    """
    d = OPENAI_REALTIME_PRICING["default"]
    assert d["text_input_cached_per_1m"] == d["text_input_per_1m"]
    assert d["audio_input_cached_per_1m"] == d["audio_input_per_1m"]


# ─────────────────────────────────────────────── Zwischenspeicher-Logik

def test_zwischenspeicher_ist_teilmenge_nicht_zusatz(zaehler):
    """Der Fehler, der am nächsten liegt: die Zahlen addieren.

    OpenAI meldet die zwischengespeicherten Tokens INNERHALB von
    `input_tokens`. Wer sie aufaddiert, verdoppelt still den grössten
    Posten — der Zähler wäre dann wieder in sich stimmig und wieder
    falsch.
    """
    voll, _ = zaehler._cost_for_session("gpt-realtime", 0, 0, 1_000_000, 0)
    ganz, _ = zaehler._cost_for_session(
        "gpt-realtime", 0, 0, 1_000_000, 0, cached_text_input_tokens=1_000_000
    )
    assert voll == pytest.approx(4.0)
    assert ganz == pytest.approx(0.40)   # nicht 4,40 und nicht 4,00


def test_haelfte_zwischengespeichert_liegt_dazwischen(zaehler):
    kosten, _ = zaehler._cost_for_session(
        "gpt-realtime", 0, 0, 1_000_000, 0, cached_text_input_tokens=500_000
    )
    assert kosten == pytest.approx(0.5 * 4.0 + 0.5 * 0.40)


@pytest.mark.parametrize("gemeldet", [2_000_000, 10 ** 9])
def test_zu_hohe_meldung_erzeugt_keinen_abzug(zaehler, gemeldet):
    """Ein kaputter Client darf den Monatszähler nicht nach unten treiben.

    Ohne Kappung ergibt `cached > total` einen negativen Summanden — und
    damit einen Weg, unter dem Deckel zu bleiben, indem man Unsinn
    meldet.
    """
    kosten, _ = zaehler._cost_for_session(
        "gpt-realtime", 0, 0, 1_000_000, 0, cached_text_input_tokens=gemeldet
    )
    assert kosten == pytest.approx(0.40)
    assert kosten > 0


def test_negative_meldung_wird_ignoriert(zaehler):
    kosten, _ = zaehler._cost_for_session(
        "gpt-realtime", 0, 0, 1_000_000, 0, cached_text_input_tokens=-5
    )
    assert kosten == pytest.approx(4.0)


def test_audio_hat_denselben_schutz(zaehler):
    ganz, _ = zaehler._cost_for_session(
        "gpt-realtime", 1_000_000, 0, 0, 0, cached_audio_input_tokens=1_000_000
    )
    assert ganz == pytest.approx(0.40)


# ───────────────────────────────────────────── der reale Augustverbrauch

def test_august_ohne_zwischenspeicher_kostet_weniger_als_gezaehlt(zaehler):
    """Die Preiskorrektur allein, ohne jede Annahme über den Cache.

    Gezählt wurden 36,87 USD. Das ist der Anteil, der schon feststeht,
    bevor irgendjemand den Zwischenspeicher misst.
    """
    usd, eur = zaehler._cost_for_session("gpt-realtime", **AUGUST)
    assert usd == pytest.approx(31.93, abs=0.01)
    assert eur == pytest.approx(usd / EUR_USD_RATE)
    assert usd < 36.87


def test_august_mit_zwischenspeicher_bleibt_unter_dem_deckel(zaehler):
    """Bei 73 % Vorbau — dem GEMESSENEN Anteil, nicht einem geschätzten.

    Der Deckel stand auf 35,00 EUR und schlug zu. Diese Prüfung sagt
    nicht, wie hoch die Trefferquote des Zwischenspeichers wirklich war
    — die weiss niemand, weil die Zahl nie gemeldet wurde. Sie sagt nur:
    Wäre der Vorbau als das verbucht worden, was er ist, hätte der
    Deckel nicht zugemacht.
    """
    anteil = int(AUGUST["text_input_tokens"] * 3608 / 4919)
    _usd, eur = zaehler._cost_for_session(
        "gpt-realtime", cached_text_input_tokens=anteil, **AUGUST
    )
    assert eur < 35.0


# ───────────────────────────────────────────────── Weitergabe im Verbund

def test_satellit_reicht_den_zwischenspeicher_an_den_master_weiter():
    """Sonst verliert genau der Host den Rabatt, der ihn meldet.

    arkturian bedient `alex-default` und ist NICHT der Master. Fehlen
    die Felder in der Weiterleitung, rechnet der Master wieder voll —
    der Fehler wäre behoben und trotzdem wirkungslos.
    """
    import inspect

    from ai.services import openai_realtime_cost_tracker as m

    quelle = inspect.getsource(m.OpenAIRealtimeCostTracker._post_to_master)
    assert '"cached_text_input_tokens": cached_text_input_tokens,' in quelle
    assert '"cached_audio_input_tokens": cached_audio_input_tokens,' in quelle


def test_browser_vertrag_kennt_die_felder():
    from ai.routes.realtime_routes import RealtimeUsageReport

    felder = RealtimeUsageReport.model_fields
    for name in ("cached_text_input_tokens", "cached_audio_input_tokens"):
        assert name in felder
        # Vorgabe 0: Ein Client, der das Feld nicht kennt, zählt wie
        # bisher — zu VIEL, nie zu wenig. Ein Default > 0 wäre eine
        # geschenkte Ermässigung an jeden, der schweigt.
        assert felder[name].default == 0
