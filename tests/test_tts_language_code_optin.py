"""Sprach-Zwang bei ElevenLabs — Opt-in, nie automatisch.

Anlass (Knowledge, 2026-08-16): Alexanders slowenische Aufnahmen
klingen kroatisch. `context.language='sl'` erreicht `/ai/tts/narrate`
korrekt — und wurde dort fallen gelassen: An ElevenLabs gingen nur
`text`, `model_id` und `voice_settings`. `eleven_multilingual_v2` kennt
ohnehin keinen `language_code` und raet die Sprache aus dem Text; bei
slowenischem Text liegt Kroatisch nahe.

**Warum Opt-in und nicht aus `context.language` abgeleitet:** Der Zwang
wirkt nur mit den neueren Modellen, und ein stiller Modellwechsel
aendert den KLANG. Alex' rund 70 slowenische Toene sind mit dem
heutigen Klang aufgenommen — wer sie neu erzeugt, soll das entscheiden,
nicht erleiden.

Ob der Zwang das Ergebnis hoerbar verbessert, ist NICHT bewiesen. Nur
dass er nie benutzt wurde. Das entscheidet ein Ohr an zwei Clips, nicht
dieser Test.
"""

import inspect

from ai.services import narration_service as ns
from ai.services import tts_service as ts


def test_beide_konfigurationen_kennen_das_feld_und_lassen_es_leer():
    """Standard None = bitweise dasselbe Verhalten wie vorher."""
    assert ns.NarrationConfig().language_code is None
    assert ts.ElevenLabsTTSConfig(voice_id="x").language_code is None


def test_narrate_reicht_nur_durch_wenn_gesetzt():
    """Ohne Wert darf der Schluessel gar nicht erst im Aufruf stehen.

    Ein `language_code: None` an ElevenLabs waere nicht dasselbe wie
    sein Fehlen — und bei `multilingual_v2` ein unbekannter Parameter.
    """
    quelle = inspect.getsource(ns)
    assert 'if request.config.language_code:' in quelle
    assert '_tts_args["language_code"] = request.config.language_code' in quelle


def test_zweiter_pfad_hat_dieselbe_luecke_geschlossen():
    """`/ai/tts/speak` teilt den Defekt — halb reparieren erzeugt genau
    die „geht bei narrate, geht nicht bei speak"-Raetsel."""
    quelle = inspect.getsource(ts)
    assert 'if config.language_code:' in quelle          # SDK-Pfad
    assert '"language_code": config.language_code,' in quelle  # REST-Pfad
    # Der REST-Pfad filtert None heraus, statt ihn zu senden.
    assert 'if v is not None}' in quelle


def test_kein_automatisches_ableiten_aus_der_sprache():
    """Die Ableitung waere ein stiller Klangwechsel im Bestand."""
    quelle = inspect.getsource(ns)
    for verboten in ("language_code = request.context.language",
                     "language_code=request.context.language"):
        assert verboten not in quelle, (
            "language_code wird automatisch aus context.language "
            "abgeleitet — das aendert den Klang bestehender Bestaende "
            "ohne Zutun des Aufrufers"
        )
