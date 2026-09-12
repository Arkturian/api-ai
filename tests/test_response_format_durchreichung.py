"""`response_format` fuer 3DPresenter (#1696).

Durchgereicht wird es nur auf den beiden API-direkten, abgerechneten
Pfaden /ai/deepseek und /ai/m3 — dort kennt der Anbieter den Schalter.
Die CLI-Pfade koennen ihn nicht erzwingen; dort wird er mit 422
abgewiesen statt still verschluckt, sonst haelt der Aufrufer Prosa
fuer JSON.

Gegen den echten Fehler gehalten: der Gegenfall prueft, dass ohne Feld
kein `response_format` im Anbieteraufruf landet (sonst zwaenge die
Aenderung jeden Altaufrufer in JSON), und der Nachbarfall, dass eine
unbekannte Form 422 ergibt statt eines bezahlten Aufrufs.
"""

import types

import pytest
from fastapi import HTTPException

from ai.routes import text_ai_routes as t


class _Antwort:
    def __init__(self):
        nachricht = types.SimpleNamespace(content='{"ok": true}')
        self.choices = [types.SimpleNamespace(message=nachricht, finish_reason="stop")]
        self.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)


def _client(gesehen):
    class _Completions:
        async def create(self, **kw):
            gesehen.update(kw)
            return _Antwort()

    class _C:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(completions=_Completions())

    return _C


@pytest.fixture(autouse=True)
def _umgebung(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("MINIMAX_MULTIMODAL_API_KEY", "sk-test")
    import openai
    self_gesehen = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", _client(self_gesehen))
    yield self_gesehen


def _prompt(**kw):
    return t.Prompt(prompt="Gib mir JSON.", confirm_api_billing=True, **kw)


@pytest.mark.asyncio
async def test_deepseek_reicht_json_object_durch(_umgebung):
    await t.deepseek_endpoint(
        _prompt(response_format={"type": "json_object"}),
        model=None,
        api_key="placeholder",
    )
    assert _umgebung["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_m3_reicht_json_object_durch(_umgebung):
    await t.m3_endpoint(
        _prompt(response_format={"type": "json_object"}),
        model=None,
        api_key="placeholder",
    )
    assert _umgebung["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_ohne_feld_bleibt_der_aufruf_unveraendert(_umgebung):
    await t.deepseek_endpoint(_prompt(), model=None, api_key="placeholder")
    assert "response_format" not in _umgebung


@pytest.mark.asyncio
async def test_unbekannte_form_wird_abgewiesen(_umgebung):
    with pytest.raises(HTTPException) as exc:
        await t.deepseek_endpoint(
            _prompt(response_format={"type": "yaml"}),
            model=None,
            api_key="placeholder",
        )
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "unsupported_response_format"
    assert "response_format" not in _umgebung


@pytest.mark.asyncio
async def test_form_ohne_typ_wird_abgewiesen(_umgebung):
    with pytest.raises(HTTPException) as exc:
        await t.deepseek_endpoint(
            _prompt(response_format={"schema": {}}),
            model=None,
            api_key="placeholder",
        )
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "invalid_response_format"


@pytest.mark.parametrize("endpunkt", ["claude", "chatgpt", "grok", "gemini"])
def test_cli_pfade_weisen_ab(endpunkt):
    with pytest.raises(HTTPException) as exc:
        t._weise_response_format_ab(
            _prompt(response_format={"type": "json_object"}), endpoint=endpunkt
        )
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "response_format_unsupported_on_endpoint"
    assert exc.value.detail["structured_endpoints"] == ["/ai/deepseek", "/ai/m3"]


def test_cli_pfade_ohne_feld_lassen_durch():
    assert t._weise_response_format_ab(_prompt(), endpoint="claude") is None


@pytest.mark.parametrize(
    "funktion,endpunkt",
    [
        (t.claude_endpoint, "claude"),
        (t.chatgpt_endpoint, "chatgpt"),
        (t.grok_endpoint, "grok"),
        (t.gemini_endpoint, "gemini"),
    ],
)
def test_cli_endpunkte_rufen_die_abweisung_auf(funktion, endpunkt):
    """Haelt die Verdrahtung: ein CLI-Endpunkt ohne diesen Aufruf wuerde
    das Feld still verschlucken. Der Aufruf steht absichtlich VOR dem
    umschliessenden try, sonst faengt dessen except die 422 wieder ein."""
    import inspect

    quelle = inspect.getsource(funktion)
    assert f'_weise_response_format_ab(prompt, endpoint="{endpunkt}")' in quelle
    vor_try = quelle.split("\n    try:")[0]
    assert "_weise_response_format_ab" in vor_try
