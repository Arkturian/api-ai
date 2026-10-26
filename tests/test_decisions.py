"""OpenAI Decisions API ueber api-ai (Auftrag Alex 26.10.2026): native Route
/ai/decisions, A/B-Schalter /ai/jev?engine=openai, gleiches JWT-Tor und
Tagesbudget wie Jev, Kosten mit Cache-Anteil. Antwortform am 26.10. am
echten Endpunkt gemessen (siehe ECHT unten)."""
import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import jev_routes as j
from ai.services import decisions_service as ds

# Gemessene Antwort von POST /v1/decisions (26.10.2026), unveraendert.
ECHT = {"model": "gpt-6-luna", "answers": [
    {"type": "predicate", "name": "blockiert", "probability": 1.0},
    {"type": "choice", "name": "art", "choice": "fehler", "probabilities": [
        {"value": "fehler", "probability": 0.8}, {"value": "erfolg", "probability": 0.06},
        {"value": "other", "probability": 0.14}], "confidence": 0.7},
    {"type": "score", "name": "dringlichkeit", "score": 1.12, "probabilities": [
        {"value": 0, "label": "gar nicht", "probability": 0.32},
        {"value": 1, "label": "bald", "probability": 0.24},
        {"value": 2, "label": "sofort", "probability": 0.44}], "confidence": 0.0}],
    "usage": {"input_tokens": 438, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
              "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 438}}

JEV_BODY = {"state": "Failed to run tool: approval policy is never.",
            "questions": {
                "blockiert": {"type": "noul", "instructions": "Blockiert?"},
                "art": {"type": "choice", "instructions": "Art?",
                        "criteria": {"fehler": "Fehler", "erfolg": "Erfolg", "other": "anderes"}},
                "dringlichkeit": {"type": "score", "instructions": "Wie dringend?",
                                  "criteria": ["gar nicht", "bald", "sofort"]}}}


# ── Uebersetzer ──────────────────────────────────────────────────────────

def test_jev_zu_decisions():
    b = ds.jev_zu_decisions(JEV_BODY["state"], JEV_BODY["questions"])
    assert b["model"] == "gpt-6-luna" and b["input"] == JEV_BODY["state"]
    q = {x["name"]: x for x in b["questions"]}
    assert q["blockiert"]["type"] == "predicate"
    assert q["art"]["choices"][0] == {"value": "fehler", "description": "Fehler"}
    assert [l["label"] for l in q["dringlichkeit"]["levels"]] == ["gar nicht", "bald", "sofort"]


def test_strukturierter_state_geht_als_input_json():
    b = ds.jev_zu_decisions({"severity": "major"}, {"a": {"type": "noul", "instructions": "?"}})
    teil = b["input"][0]["content"][0]
    assert b["input"][0]["role"] == "user" and teil == {"type": "input_json", "data": {"severity": "major"}}


def test_decisions_zu_jev_aus_echter_antwort():
    a = ds.decisions_zu_jev(ECHT)
    assert a["answers"]["blockiert"] == {"type": "noul", "noul": 1.0}
    assert a["answers"]["art"]["choice"] == "fehler" and a["answers"]["art"]["probabilities"]["other"] == 0.14
    sc = a["answers"]["dringlichkeit"]
    assert sc["score"] == 1.12 and sc["legend"]["2"] == "sofort" and sc["probabilities"]["0"] == 0.32
    assert a["usage"]["input_tokens"] == 438


def test_refusal_wird_nicht_verworfen():
    a = ds.decisions_zu_jev({"answers": [{"type": "refusal", "name": "x", "refusal": "nein"}]})
    assert a["answers"]["x"] == {"type": "refusal", "refusal": "nein"}


def test_kosten_mit_cacheanteil():
    u = {"input_tokens": 2_000_000, "input_tokens_details": {"cached_tokens": 1_000_000}}
    assert ds.kosten(u) == pytest.approx(0.10 + 0.01)
    assert ds.kosten(ECHT["usage"]) == pytest.approx(438 * 0.10 / 1e6)


# ── Route und Tor ────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DECISIONS_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("DECISIONS_DAILY_MAX_REQUESTS", "3")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-geheim-nie-ausgeben")
    monkeypatch.setattr(j, "_jwt_sub_geprueft", lambda a: "agent:Test" if a == "Bearer gut" else None)
    aufrufe = []

    async def fake(body, **k):
        aufrufe.append(body)
        return 200, ECHT
    monkeypatch.setattr(ds, "aufrufen", fake)
    c = TestClient(main.app)
    c.aufrufe = aufrufe
    return c


GUT = {"Authorization": "Bearer gut"}
NATIV = {"input": "x", "questions": [{"name": "a", "type": "predicate", "instructions": "?"}]}


@pytest.mark.parametrize("pfad", ["/ai/decisions", "/ai/decisions/cost-status"])
def test_ohne_jwt_401(client, pfad):
    r = client.post(pfad, json=NATIV) if pfad == "/ai/decisions" else client.get(pfad)
    assert r.status_code == 401 and client.aufrufe == []


def test_nativ_mit_kosten_und_latenz(client):
    r = client.post("/ai/decisions", json=NATIV, headers=GUT)
    assert r.status_code == 200
    d = r.json()
    assert d["answers"][0]["name"] == "blockiert" and "latency_ms" in d and d["cost_usd"] > 0
    assert "sk-geheim" not in r.text
    st = client.get("/ai/decisions/cost-status", headers=GUT).json()
    assert st["requests"] == 1 and st["by_caller"]["Test"]["questions"] == 1


@pytest.mark.parametrize("fragen,fehler", [
    ([], "questions_required"),
    ([{"type": "predicate", "instructions": "?"}], "question_name_required"),
    ([{"name": "a", "type": "noul", "instructions": "?"}], "invalid_question_type"),
    ([{"name": "a", "type": "predicate", "instructions": "?"}] * 2, "duplicate_question_name"),
    ([{"name": "a", "type": "choice", "instructions": "?"}], "choices_required"),
    ([{"name": "a", "type": "score", "instructions": "?", "levels": [{"label": "x"}]}], "score_levels_2_to_10"),
])
def test_validierung_vor_anbieteraufruf(client, fragen, fehler):
    r = client.post("/ai/decisions", json={"input": "x", "questions": fragen}, headers=GUT)
    assert r.status_code == 422 and r.json()["detail"]["error"] == fehler and client.aufrufe == []


def test_budget_greift(client):
    for _ in range(3):
        assert client.post("/ai/decisions", json=NATIV, headers=GUT).status_code == 200
    r = client.post("/ai/decisions", json=NATIV, headers=GUT)
    assert r.status_code == 429 and r.json()["detail"]["error"] == "decisions_daily_budget_exceeded"


def test_jev_route_mit_engine_openai_antwortet_in_jev_form(client, monkeypatch):
    from ai.services import jev_service as js
    async def nie(*a, **k):
        raise AssertionError("TypeSafe darf bei engine=openai nicht gerufen werden")
    monkeypatch.setattr(js, "aufrufen", nie)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    r = client.post("/ai/jev?engine=openai", json=JEV_BODY, headers=GUT)
    assert r.status_code == 200
    d = r.json()
    assert d["engine"] == "openai" and d["answers"]["blockiert"]["noul"] == 1.0
    assert client.aufrufe[0]["questions"][0]["type"] == "predicate"


def test_jev_route_ohne_engine_bleibt_typesafe(client, monkeypatch, tmp_path):
    from ai.services import jev_service as js
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("JEV_USAGE_DIR", str(tmp_path / "jev"))
    async def jev(body, **k):
        return 200, {"model": "jev-1.14.0", "answers": {}, "usage": {"input_tokens": 1}}, {}
    monkeypatch.setattr(js, "aufrufen", jev)
    r = client.post("/ai/jev", json=JEV_BODY, headers=GUT)
    assert r.status_code == 200 and r.json()["model"] == "jev-1.14.0" and client.aufrufe == []


def test_betreiber_schaltet_motor_per_env(client, monkeypatch):
    """Umschalten fuer alle Aufrufer ohne Client-Aenderung — und zurueck."""
    from ai.services import jev_service as js
    async def nie(*a, **k):
        raise AssertionError("TypeSafe darf nicht gerufen werden")
    monkeypatch.setattr(js, "aufrufen", nie)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("JEV_DEFAULT_ENGINE", "openai")
    r = client.post("/ai/jev", json=JEV_BODY, headers=GUT)
    assert r.status_code == 200 and r.json()["engine"] == "openai"
