"""Analyze-Schritt des Dialog-Builders: Cue-Typen und Fehlerkanal.

Gegen den echten Fehler gehalten (Alex' admin.arkturian.com/dialog.php,
gemeldet 2026-09-12): Das Planungsmodell lieferte einen sauber parsenden
Plan, in dem jede Sprechzeile `"type": "speech"` trug. Der Verbraucher
vergleicht strikt auf `"dialog"`, der Riegel zaehlte null Dialogeintraege
und warf 502 `no_dialog_cues` — und der Endpunkt goss diese 502 in eine
500 mit dem Dict als Text, sodass der Aufrufer weder Status noch
Fehlercode auswerten konnte.

Die drei Faelle: Ziel (Synonym wird Dialog), Gegenfall (echter Leerplan
bleibt leer, der Riegel muss weiter greifen), Nachbarfall (bereits
korrekte Typen werden nicht angefasst).
"""

import pytest
from fastapi import HTTPException

from ai.services.audio_drama_service import normalize_production_cues


def test_synonyme_werden_zu_dialog():
    plan = {"production_cues": [
        {"type": "speech", "speaker": "Anna", "text": "Hallo."},
        {"type": "Line", "speaker": "Bob", "text": "Servus."},
        {"type": "dialogue", "speaker": "Anna", "text": "Na endlich."},
    ]}
    geaendert = normalize_production_cues(plan)
    assert geaendert == 3
    assert {c["type"] for c in plan["production_cues"]} == {"dialog"}


def test_ohne_typ_entscheiden_die_felder():
    plan = {"production_cues": [
        {"speaker": "Anna", "text": "Ich habe keinen Typ."},
        {"description": "Door slams shut"},
    ]}
    normalize_production_cues(plan)
    assert plan["production_cues"][0]["type"] == "dialog"
    assert plan["production_cues"][1]["type"] == "sfx"


def test_korrekte_typen_bleiben_unberuehrt():
    plan = {"production_cues": [
        {"type": "dialog", "speaker": "Anna", "text": "Passt."},
        {"type": "sfx", "description": "Rain"},
        {"type": "silence", "duration_ms": 400},
    ]}
    assert normalize_production_cues(plan) == 0
    assert [c["type"] for c in plan["production_cues"]] == ["dialog", "sfx", "silence"]


def test_leerplan_bleibt_leer():
    """Der Gegenfall: ohne Sprechzeilen darf nichts erfunden werden —
    sonst liefe die Produktion in den Mixer und stuerbe dort."""
    plan = {"production_cues": [{"type": "sfx", "description": "Wind"}]}
    assert normalize_production_cues(plan) == 0
    assert not [c for c in plan["production_cues"] if c["type"] == "dialog"]


def test_kein_plan_kein_absturz():
    assert normalize_production_cues({}) == 0
    assert normalize_production_cues({"production_cues": None}) == 0
    assert normalize_production_cues({"production_cues": ["kaputt", 3]}) == 0


@pytest.mark.asyncio
async def test_endpunkt_reicht_502_unveraendert_durch(monkeypatch):
    """Der eigentliche Melde-Fehler: die 502 kam als 500-Text an."""
    from ai.routes import audio_ai_routes as a
    from ai.services.tts_models import SpeechRequest

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self):
            raise HTTPException(
                status_code=502,
                detail={"error": "no_dialog_cues", "hint": "Retry the request."},
            )

    monkeypatch.setattr(a, "AudioDramaGenerator", _Generator)

    req = SpeechRequest(
        id="t-1",
        timestamp="2026-09-12T00:00:00Z",
        content={"text": "Anna: Hallo."},
        config={"dialog_mode": True, "analyze_only": True},
    )

    with pytest.raises(HTTPException) as exc:
        await a.generate_speech_endpoint(req, api_key="placeholder")

    assert exc.value.status_code == 502
    assert exc.value.detail["error"] == "no_dialog_cues"


@pytest.mark.asyncio
async def test_analyse_verdrahtet_die_normalisierung(monkeypatch):
    """Haelt die Verdrahtung, nicht nur die Funktion: ein Plan, der nur
    `"type": "speech"` kennt, darf den no_dialog_cues-Riegel NICHT mehr
    ausloesen. Ohne den Aufruf im Analysepfad faellt dieser Test."""
    import json
    import httpx

    from ai.services import audio_drama_service as ads
    from ai.services.tts_models import SpeechRequest

    plan = {"production_cues": [
        {"type": "speech", "speaker": "Anna", "gender": "female", "text": "Hallo."},
    ], "music": []}

    class _Antwort:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": json.dumps(plan)}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Antwort()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    req = SpeechRequest(
        id="t-2",
        timestamp="2026-09-12T00:00:00Z",
        content={"text": "Anna: Hallo."},
        config={"dialog_mode": True, "analyze_only": True},
    )
    gen = ads.AudioDramaGenerator(req, "placeholder", image_gen_func=None)

    ergebnis = await gen._analyze_script()
    assert [c["type"] for c in ergebnis["production_cues"]] == ["dialog"]
