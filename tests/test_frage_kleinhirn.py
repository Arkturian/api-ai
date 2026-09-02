"""`frage_kleinhirn` — den Wahrnehmungsstand eines Agenten holen (#4907 B5).

Owner-Entscheid (Alex, 02.09. 08:22, #4909 q-143ca1a5523d): drei Sichten
auf EINEN Agenten, keine Praesenz-Instanz. Das Werkzeug holt NUR den
Kontextblock (`GET /api/sessions/<agent>/presence-context`); das
Sprachmodell antwortet selbst. Die echten Fehler, gegen die hier gehalten
wird: (1) nie eine Session ansprechen (kein POST, kein /input), (2) der
Umschlag (ref, source_refs) geht in `diagnostics`, nie ins Ergebnis,
(3) der Bearer des MENSCHEN wird durchgereicht, nie ersetzt.
"""

import asyncio
import pytest

from ai.routes import realtime_routes as rr


class _Resp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class _FakeClient:
    """Spielt cloud-api: nur presence-context."""
    context = None            # (status_code, data) oder Exception
    gets = []
    posts = []
    params = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        type(self).gets.append((url, headers))
        type(self).params.append(params or {})
        c = type(self).context
        if isinstance(c, Exception):
            raise c
        if url.endswith("/presence-context") and c is not None:
            return _Resp(*c)
        return _Resp(404, {})

    async def post(self, url, headers=None, json=None):
        type(self).posts.append((url, json, headers))
        return _Resp(200, {"ok": True})


@pytest.fixture(autouse=True)
def _schnell(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _FakeClient)
    _FakeClient.context = None
    _FakeClient.gets = []
    _FakeClient.posts = []
    _FakeClient.params = []
    yield


def _run(args, auth="Bearer x", diagnose=None):
    return asyncio.run(rr._tool_frage_kleinhirn(args, auth, diagnose))


KONTEXT = {
    "ref": "abc123", "presence_of": "CloudV2", "mode": "text",
    "stand": "2026-09-02T06:20:14.515Z", "derivative_count": 8,
    "source_refs": [{}, {}, {}],
    "block": "[PRESENCE-CONTEXT ref=abc123 of=CloudV2 stand=02.09._06:20]\n"
             "▸ 02.09. 06:01 CloudV2 hat die Linse ausgeliefert.",
}


def test_ohne_bearer_keine_anonyme_frage():
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"}, auth=None)
    assert out["ok"] is False and out["status"] == "no_caller_identity"
    assert _FakeClient.gets == []


def test_happy_path_kontext_und_stand_umschlag_nur_in_diagnose():
    _FakeClient.context = (200, KONTEXT)
    diag = {}
    out = _run({"agent": "CloudV2", "frage": "Was macht er gerade?"}, diagnose=diag)
    assert out["ok"] is True and out["status"] == "ok" and out["agent"] == "CloudV2"
    assert out["kontext"].startswith("[PRESENCE-CONTEXT")
    assert out["stand"] == "2026-09-02T06:20:14.515Z" and out["derivative_count"] == 8
    # Umschlag NICHT im Ergebnis, sondern daneben
    assert "ref" not in out and "source_refs" not in out and "antwort" not in out
    assert diag["kleinhirn"]["ref"] == "abc123" and diag["kleinhirn"]["source_refs"] == 3
    url, hdrs = _FakeClient.gets[0]
    assert url.endswith("/api/sessions/CloudV2/presence-context")
    assert hdrs["Authorization"] == "Bearer x"
    # Frage geht als Audit-Metadatum mit (generated_for_question), sonst nichts
    assert _FakeClient.params[0] == {"question": "Was macht er gerade?"}


def test_keine_session_wird_angesprochen():
    """Owner-Entscheid 02.09.: keine Praesenz-Instanz. Das Werkzeug darf
    nie etwas in eine Session schreiben — kein /input, kein POST."""
    _FakeClient.context = (200, KONTEXT)
    _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert _FakeClient.posts == []
    assert all("/input" not in u and "/sessions/all" not in u for u, _ in _FakeClient.gets)


def test_404_ist_no_context_nicht_erfunden():
    _FakeClient.context = (404, {})
    out = _run({"agent": "Niemand", "frage": "Was machst du?"})
    assert out["ok"] is False and out["status"] == "no_context"
    assert "kontext" not in out


def test_leerer_block_ist_no_context():
    _FakeClient.context = (200, {**KONTEXT, "block": "   "})
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert out["status"] == "no_context"


def test_fehler_wird_unavailable_nicht_erfunden():
    _FakeClient.context = RuntimeError("timeout")
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert out["ok"] is False and out["status"] == "unavailable"
    assert "kontext" not in out
    _FakeClient.context = (502, {})
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert out["status"] == "unavailable" and out["http"] == 502


def test_werkzeug_ist_in_allowlist_und_schema():
    assert "frage_kleinhirn" in rr.READ_TOOL_NAMES
    schema = next(t for t in rr._arcturian_read_tools() if t["name"] == "frage_kleinhirn")
    assert schema["parameters"]["required"] == ["agent", "frage"]
    assert schema["parameters"]["additionalProperties"] is False
    # Das Sprachmodell antwortet SELBST — die Anweisung steht im Schema.
    assert "SELBST" in schema["description"] and "nie als sein Ich" in schema["description"]


def test_frage_kleinhirn_nicht_im_erzwungenen_status_lookup_zug():
    """Im `status_lookup`-Zug ist `tool_choice: required` — das Modell kann
    keinen Satz vorausschicken; die Liste bleibt einelementig (#4531 Z. 8)."""
    payload = rr._arcturian_status_lookup_payload()
    namen = [t["name"] for t in payload["response"]["tools"]]
    assert "agent_status" in namen
    assert "frage_kleinhirn" not in namen
    assert payload["response"]["tool_choice"] == "required"


def test_bearer_des_menschen_wird_durchgereicht_nie_ersetzt(monkeypatch):
    """Die Datengrenze (#4531 Z. 116) haengt daran, dass cloud-api den Bearer
    des MENSCHEN sieht — ein Agenten-/Dienst-Token saehe die Foederation."""
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst-token-der-NICHT-benutzt-werden-darf")
    _FakeClient.context = (200, KONTEXT)
    _run({"agent": "CloudV2", "frage": "Was machst du?"}, auth="Bearer mensch-123")
    url, hdrs = _FakeClient.gets[0]
    assert hdrs == {"Authorization": "Bearer mensch-123"}
    assert "dienst-token" not in str(hdrs)
