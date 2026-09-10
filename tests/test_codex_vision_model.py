"""Bild-Aufrufe ueber /ai/chatgpt (Storage-Safety-Gate, 10.09., 26 Objekte):
gpt-5.4-mini ist im Abo nicht mehr auswaehlbar, die Vision-Vorgabe lief NACH
dem Alias-Mapping und machte aus 'codex-default' wieder ein --model. Gegen
den echten Fehler gehalten: kein Weg von aussen fuehrte zu einem Bild-Aufruf
ohne --model."""

from ai.routes import text_ai_routes as t


def test_pseudo_aliasse_heissen_kein_model():
    for m in ("codex-default", "default", "auto", "codex", " Codex-Default "):
        assert t._codex_model_or_none(m) is None
    assert t._codex_model_or_none("gpt-5.6-luna") == "gpt-5.6-luna"
    assert t._codex_model_or_none(None) is None


def test_vision_vorgabe_ist_luna_und_respektiert_codex_default(monkeypatch):
    monkeypatch.delenv("CODEX_VISION_MODEL", raising=False)
    assert t._codex_vision_model() == "gpt-5.6-luna"
    monkeypatch.setenv("CODEX_VISION_MODEL", "codex-default")
    assert t._codex_vision_model() is None                 # Env-Alias -> kein --model
    monkeypatch.setenv("CODEX_VISION_MODEL", "gpt-5.6-sol")
    assert t._codex_vision_model() == "gpt-5.6-sol"


def test_modellablehnung_wird_erkannt_erfolg_nicht():
    assert t._codex_model_rejected('{"type":"error","message":"Model metadata for `gpt-5.4-mini` not found. Defaulting"}\n{"type":"turn.failed"}')
    assert t._codex_model_rejected('{"type":"error","message":"gpt-5.4-mini is not supported when using Codex with a ChatGPT account"}')
    assert not t._codex_model_rejected('{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}')
    assert not t._codex_model_rejected("")
