"""Devlog-Zeitpunkte im OpenAPI ohne anyOf (C#-SDK, 11.11.2026).

Der echte Fehler: Optional[Any] erschien als anyOf [{}, {type: null}], der
C#-Generator erzeugte daraus `Option<AnyOf>`, ein Typ, den es nicht gibt.
Die Neuerzeugung des SDK (PR #3) brach mit 9x CS0246 ab.
"""
from ai.routes.realtime_routes import DevlogLine, DevlogUpsertRequest


def _props(modell):
    return modell.model_json_schema()["properties"]


def test_zeitpunkte_ohne_anyof_im_schema():
    for modell, felder in ((DevlogLine, ["ts"]), (DevlogUpsertRequest, ["started_at", "ended_at"])):
        for f in felder:
            assert "anyOf" not in _props(modell)[f], (modell.__name__, f)


def test_pruefung_bleibt_offen():
    assert DevlogLine(ts="2026-11-11T14:00:00Z").ts == "2026-11-11T14:00:00Z"
    assert DevlogLine(ts=1794405188000).ts == 1794405188000
    assert DevlogLine().ts is None
    r = DevlogUpsertRequest(voice_session_id="v", started_at=1, ended_at=None)
    assert r.started_at == 1 and r.ended_at is None
