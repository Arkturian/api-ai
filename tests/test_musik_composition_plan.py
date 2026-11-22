"""/ai/genmusic_eleven mit composition_plan (DaVinci, 22.11.): feste Abschnitte,
vor dem bezahlten Aufruf geprueft, Plan statt Prompt an ElevenLabs."""
import asyncio

import pytest
from fastapi import HTTPException

from ai.routes import music_generation as mg

PLAN = {"positive_global_styles": ["cinematic", "96 bpm", "d minor"],
        "negative_global_styles": ["vocals", "singing"],
        "sections": [{"section_name": "Pre-dawn", "positive_local_styles": ["felt piano"],
                      "negative_local_styles": [], "duration_ms": 10000, "lines": []},
                     {"section_name": "Drop", "positive_local_styles": ["epic drums"],
                      "negative_local_styles": [], "duration_ms": 52500, "lines": []}]}


@pytest.mark.parametrize("plan", [
    {"sections": []},
    {"sections": [{"duration_ms": 2000}]},
    {"sections": [{"duration_ms": 130000}]},
    {"sections": [{"duration_ms": 100000}] * 4},
    {"sections": [{"duration_ms": 5000, "lines": "la la"}]},
    {"sections": [{"duration_ms": 5000, "unbekannt": 1}]},
    {"sections": [{"duration_ms": 5000}], "extra": 1},
    {"sections": [{"duration_ms": True}]},
    "kein dict",
])
def test_ungueltiger_plan_vor_dem_aufruf(plan):
    with pytest.raises(HTTPException) as e:
        mg.plan_pruefen(plan)
    assert e.value.status_code == 422


def test_plan_geht_statt_prompt_an_den_anbieter(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    gesehen = {}

    class R:
        status_code = 200
        headers = {"content-type": "audio/mpeg"}
        content = b"ID3" + b"\0" * 100

    class C:
        def __init__(self, *a, timeout=None, **k):
            gesehen["timeout"] = timeout
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            gesehen["json"] = json
            return R()

    class Obj:
        id = 1
        file_url = "https://example/1"

    async def speichern(**k):
        return Obj()
    from ai.services.elevenlabs_cost_tracker import elevenlabs_cost_tracker as t
    gebucht = {}
    monkeypatch.setattr(t, "pre_check", lambda *a, **k: None)
    monkeypatch.setattr(t, "track_music", lambda **k: gebucht.update(k))
    monkeypatch.setattr(mg.httpx, "AsyncClient", C)
    monkeypatch.setattr(mg, "save_file_and_record", speichern)
    plan = mg.plan_pruefen(PLAN)
    asyncio.run(mg.generate_music_elevenlabs("Der Shot", 30000, composition_plan=plan))
    assert gesehen["json"] == {"composition_plan": plan, "model_id": "music_v1"}
    assert "prompt" not in gesehen["json"] and gesehen["timeout"].read == 180.0
    assert gebucht["seconds_requested"] == pytest.approx(62.5)
