"""`frage_kleinhirn` — die Praesenz-Session eines Agenten befragen (#4907 B5).

Der echte Fehler, gegen den diese Tests gehalten sind (Cloud, 01.09.):
Praesenz NIE ueber den Namen aufloesen, nur ueber `runtime.presence_of`.
Eine Session, die zufaellig `<Agent>-Presence` heisst, aber keine Praesenz
ist, darf nicht angesprochen werden. Und: der Umschlag (ref, source_refs)
geht in `diagnostics`, nie ins Ergebnis — das Modell soll Audit-Daten
weder sehen noch bezahlen.
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
    """Spielt cloud-api: Sessionliste, Eingabe, History-Sequenz."""
    sessions = []
    history_seq = []          # Liste von Turn-Dicts, je Abruf einer (letzter bleibt)
    posts = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        if url.endswith("/api/sessions/all"):
            return _Resp(200, {"sessions": type(self).sessions})
        if "/history?limit=1" in url:
            seq = type(self).history_seq
            turn = seq.pop(0) if len(seq) > 1 else (seq[0] if seq else None)
            return _Resp(200, {"turns": [turn] if turn else []})
        return _Resp(404, {})

    async def post(self, url, headers=None, json=None):
        type(self).posts.append((url, json, headers))
        return _Resp(200, {"ok": True})


@pytest.fixture(autouse=True)
def _schnell(monkeypatch):
    monkeypatch.setattr(rr.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(rr, "_KLEINHIRN_POLL_S", 0.0)
    monkeypatch.setattr(rr, "_KLEINHIRN_TIMEOUT_S", 0.5)
    _FakeClient.sessions = []
    _FakeClient.history_seq = []
    _FakeClient.posts = []
    yield


def _run(args, auth="Bearer x", diagnose=None):
    return asyncio.run(rr._tool_frage_kleinhirn(args, auth, diagnose))


PRAESENZ = {"name": "CloudV2-Presence",
            "runtime": {"kind": "presence", "presence_of": "CloudV2"}}
ALT = {"uuid": "alt", "ended_at": "t0", "sections": [{"kind": "text", "content": "alt"}]}
NEU = {"uuid": "neu", "ended_at": "t1",
       "sections": [{"kind": "text", "content": "Ich arbeite gerade an der **Linse**."}],
       "presence": {"ref": "abc123", "stand": "2026-09-01T20:13:00Z", "mode": "voice",
                    "derivative_count": 8, "source_refs": [{}, {}, {}]}}


def test_ohne_bearer_keine_anonyme_frage():
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"}, auth=None)
    assert out["ok"] is False and out["status"] == "no_caller_identity"


def test_praesenz_nur_ueber_runtime_nicht_ueber_namen():
    # Eine Session HEISST wie eine Praesenz, ist aber keine.
    _FakeClient.sessions = [{"name": "CloudV2-Presence",
                             "runtime": {"kind": "core", "presence_of": ""}}]
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert out["status"] == "no_presence"
    assert _FakeClient.posts == []      # nichts angesprochen


def test_happy_path_antwort_ohne_umschlag_umschlag_in_diagnose():
    _FakeClient.sessions = [PRAESENZ]
    _FakeClient.history_seq = [ALT, ALT, NEU]
    diag = {}
    out = _run({"agent": "CloudV2", "frage": "Was machst du gerade?"}, diagnose=diag)
    assert out["ok"] is True and out["status"] == "ok"
    assert out["antwort"].startswith("Ich arbeite gerade an der Linse")
    assert "**" not in out["antwort"]
    assert out["stand"] == "2026-09-01T20:13:00Z" and out["mode"] == "voice"
    assert out["derivative_count"] == 8
    # Umschlag NICHT im Ergebnis, sondern daneben
    assert "ref" not in out and "source_refs" not in out
    assert diag["kleinhirn"]["ref"] == "abc123" and diag["kleinhirn"]["source_refs"] == 3
    # Eingabe: Menschen-Pfad mit [SPRACHE], ohne _origin, mit client_message_id
    url, body, hdrs = _FakeClient.posts[0]
    assert url.endswith("/api/sessions/CloudV2-Presence/input")
    assert body["data"].startswith("[SPRACHE] ") and "_origin" not in body
    assert body["client_message_id"]
    assert hdrs["Authorization"] == "Bearer x"


def test_timeout_wird_unavailable_nicht_erfunden():
    _FakeClient.sessions = [PRAESENZ]
    _FakeClient.history_seq = [ALT]      # nie ein neuer fertiger Turn
    out = _run({"agent": "CloudV2", "frage": "Was machst du?"})
    assert out["ok"] is False and out["status"] == "unavailable"
    assert "antwort" not in out


def test_werkzeug_ist_in_allowlist_und_schema():
    assert "frage_kleinhirn" in rr.READ_TOOL_NAMES
    namen = [t["name"] for t in rr._arcturian_read_tools()]
    assert "frage_kleinhirn" in namen
    schema = next(t for t in rr._arcturian_read_tools() if t["name"] == "frage_kleinhirn")
    assert schema["parameters"]["required"] == ["agent", "frage"]
    assert schema["parameters"]["additionalProperties"] is False
