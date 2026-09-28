"""Jev (TypeSafe System One) ueber api-ai, Anforderung Content #5083
(Jev/Alex, 28.09.2026). Durchreichen statt nachbauen, grob validieren,
Kosten + Latenz ergaenzen, Tagesbudget gegen Endlosschleifen, 429/529 mit
Backoff, Schluessel nie in Antwort oder Log."""
import asyncio

import pytest
from fastapi import HTTPException

from ai.routes import jev_routes as j
from ai.services import jev_service as s

ANTWORT = {"model": "jev-1.13.0",
           "answers": {"echt": {"type": "noul", "noul": 0.86}},
           "usage": {"input_tokens": 386, "output_tokens": 63}}


class _R:
    def __init__(self, code, body=None, headers=None):
        self.status_code = code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = str(self._body)
    def json(self):
        return self._body


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-geheim-nie-ausgeben")
    monkeypatch.setenv("JEV_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("JEV_DAILY_MAX_REQUESTS", "3")
    monkeypatch.setenv("JEV_DAILY_MAX_USD", "2")
    calls = []
    antworten = []

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None, **k):
            calls.append({"url": url, "json": json, "headers": headers})
            return antworten.pop(0) if antworten else _R(200, ANTWORT)

    monkeypatch.setattr(s.httpx, "AsyncClient", _Client)
    schlaf = []
    async def fake_sleep(t): schlaf.append(t)
    monkeypatch.setattr(s.asyncio, "sleep", fake_sleep)
    return calls, antworten, schlaf


def _req(**kw):
    b = dict(state="Hallo, Angebot bitte", questions={"echt": {"type": "noul", "instructions": "Echte Anfrage?"}})
    b.update(kw)
    return j.JevRequest(**b)


def test_erfolg_reicht_durch_und_ergaenzt(umgebung):
    calls, _, _ = umgebung
    r = asyncio.run(j.jev_endpoint(_req(), x_agent_name="Jev"))
    assert r["answers"] == ANTWORT["answers"] and r["model"] == "jev-1.13.0"
    assert r["cost_usd"] == pytest.approx(386 * 0.042 / 1e6)
    assert isinstance(r["latency_ms"], int)
    c = calls[0]
    assert c["url"] == "https://api.typesafe.ai/v1/systemone"
    assert c["headers"]["Authorization"] == "Bearer ts-geheim-nie-ausgeben"
    assert c["json"]["model"] == "jev-latest" and c["json"]["state"] == "Hallo, Angebot bitte"
    assert "ts-geheim" not in str(r)


def test_feste_version_wird_durchgereicht(umgebung):
    calls, _, _ = umgebung
    asyncio.run(j.jev_endpoint(_req(model="jev-1.13.0"), x_agent_name=None))
    assert calls[0]["json"]["model"] == "jev-1.13.0"


@pytest.mark.parametrize("fragen,fehler", [
    ({}, "questions_required"),
    ({"x": {"type": "frei", "instructions": "?"}}, "invalid_question_type"),
    ({"x": {"type": "noul"}}, "instructions_required"),
    ({"x": {"type": "choice", "instructions": "?", "criteria": {str(i): "o" for i in range(256)}}}, "too_many_choice_options"),
    ({"x": {"type": "choice", "instructions": "?"}}, "criteria_required"),
    ({"x": {"type": "score", "instructions": "?", "criteria": ["nur eine"]}}, "score_levels_2_to_10"),
])
def test_grobe_validierung_vor_dem_aufruf(umgebung, fragen, fehler):
    calls, _, _ = umgebung
    with pytest.raises(HTTPException) as e:
        asyncio.run(j.jev_endpoint(_req(questions=fragen), x_agent_name=None))
    assert e.value.status_code == 422 and e.value.detail["error"] == fehler
    assert calls == []


def test_zu_gross_ist_413_ohne_aufruf(umgebung):
    calls, _, _ = umgebung
    with pytest.raises(HTTPException) as e:
        asyncio.run(j.jev_endpoint(_req(state="x" * 400_001), x_agent_name=None))
    assert e.value.status_code == 413 and calls == []


def test_upstream_4xx_wird_durchgereicht(umgebung):
    calls, antworten, _ = umgebung
    antworten.append(_R(422, {"detail": "questions.x.criteria missing"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(j.jev_endpoint(_req(), x_agent_name=None))
    assert e.value.status_code == 422
    assert e.value.detail["upstream_status"] == 422 and "criteria" in str(e.value.detail["upstream_body"])


def test_429_wiederholt_mit_retry_after(umgebung):
    calls, antworten, schlaf = umgebung
    antworten += [_R(429, {"detail": "rate"}, {"retry-after": "2"}), _R(529, {"detail": "overloaded"})]
    r = asyncio.run(j.jev_endpoint(_req(), x_agent_name=None))
    assert r["model"] == "jev-1.13.0" and len(calls) == 3
    assert schlaf[0] == 2.0 and schlaf[1] > 0


def test_tagesbudget_anfragen(umgebung):
    calls, _, _ = umgebung
    for _ in range(3):
        asyncio.run(j.jev_endpoint(_req(), x_agent_name="Schleife"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(j.jev_endpoint(_req(), x_agent_name="Schleife"))
    assert e.value.status_code == 429 and e.value.detail["error"] == "jev_daily_budget_exceeded"
    assert len(calls) == 3


def test_nutzung_je_aufrufer(umgebung):
    asyncio.run(j.jev_endpoint(_req(), x_agent_name="Jev"))
    asyncio.run(j.jev_endpoint(_req(), x_agent_name=None))
    st = s.status()
    assert st["by_caller"]["Jev"]["requests"] == 1 and st["by_caller"]["(unbekannt)"]["requests"] == 1
    assert st["requests"] == 2 and st["input_tokens"] == 772
    assert st["models_seen"] == {"jev-1.13.0": 2}


def test_ohne_schluessel_503(umgebung, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setattr(s, "_SCHLUESSEL_DATEI", "/gibt/es/nicht")
    with pytest.raises(HTTPException) as e:
        asyncio.run(j.jev_endpoint(_req(), x_agent_name=None))
    assert e.value.status_code == 503 and e.value.detail["error"] == "typesafe_key_missing"


def test_ist_bezahlpfad():
    import main
    assert main._ist_bezahlpfad("/ai/jev")


# --- Etikett aus geprueftem JWT (Jev-Befund 1, 28.09.) --------------------

def test_etikett_aus_geprueftem_jwt(umgebung, monkeypatch):
    monkeypatch.setattr(j, "_jwt_sub_geprueft", lambda auth: "agent:Jev" if auth == "Bearer gut" else None)
    asyncio.run(j.jev_endpoint(_req(), x_agent_name=None, authorization="Bearer gut"))
    asyncio.run(j.jev_endpoint(_req(), x_agent_name=None, authorization="Bearer gefaelscht"))
    asyncio.run(j.jev_endpoint(_req(), x_agent_name="Gateway-Agent", authorization="Bearer gut"))
    by = s.status()["by_caller"]
    assert by["Jev"]["requests"] == 1 and "agent:Jev" not in by
    assert by["(jwt-ungueltig)"]["requests"] == 1
    assert by["Gateway-Agent"]["requests"] == 1          # Gateway-Kopf hat Vorrang


def test_jwt_pruefung_lehnt_ungeprueftes_ab():
    assert j._jwt_sub_geprueft("Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiJhZ2VudDpGYWxzY2gifQ.") is None
    assert j._jwt_sub_geprueft(None) is None and j._jwt_sub_geprueft("Basic x") is None


def test_gateway_und_direkt_eine_zeile(umgebung, monkeypatch):
    monkeypatch.setattr(j, "_jwt_sub_geprueft", lambda auth: "agent:Jev")
    asyncio.run(j.jev_endpoint(_req(), x_agent_name=None, authorization="Bearer x"))
    asyncio.run(j.jev_endpoint(_req(), x_agent_name="agent:Jev", authorization=None))
    asyncio.run(j.jev_endpoint(_req(), x_agent_name="Jev", authorization=None))
    assert s.status()["by_caller"]["Jev"]["requests"] == 3
