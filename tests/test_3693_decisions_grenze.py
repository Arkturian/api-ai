"""#3693: Die Tagesgrenze von Decisions haelt hart, auch bei gleichzeitigen
Aufrufen, langen Ruempfen und Zeitueberschreitungen. Nur synthetische Inhalte."""
import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

from ai.routes import decisions_routes as d
from ai.services import decisions_service as s

GRENZE = 0.50


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    monkeypatch.setenv("DECISIONS_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-nur-fuer-tests")
    monkeypatch.setenv("DECISIONS_DAILY_MAX_USD", str(GRENZE))
    monkeypatch.setenv("DECISIONS_DAILY_MAX_REQUESTS", "100000")
    for n in ("DECISIONS_MAX_ZEICHEN", "DECISIONS_CALLER_LIMITS", "DECISIONS_PRICE_PER_1M_INPUT_USD"):
        monkeypatch.delenv(n, raising=False)


def rumpf(zeichen: int) -> dict:
    return {"model": "gpt-6-luna", "input": "a" * zeichen,
            "questions": [{"name": "q", "type": "predicate", "instructions": "x"}]}


def stand() -> dict:
    return s.status()


def antwort_fuer(body, faktor=1.0):
    groesse = len(json.dumps(body, ensure_ascii=False))
    return 200, {"usage": {"input_tokens": int(groesse * faktor)}, "results": []}


def test_parallele_lange_aufrufe_bleiben_unter_der_grenze(monkeypatch):
    async def langsam(body, timeout_s=15.0, versuche=3):
        await asyncio.sleep(0.05)
        return antwort_fuer(body)
    monkeypatch.setattr(s, "aufrufen", langsam)

    async def alle():
        return await asyncio.gather(*[d.ausfuehren(rumpf(390_000), "test", versuche=3) for _ in range(20)],
                                    return_exceptions=True)
    erg = asyncio.run(alle())
    ok = [e for e in erg if isinstance(e, dict)]
    abgewiesen = [e for e in erg if isinstance(e, HTTPException) and e.status_code == 429]
    st = stand()
    assert ok and abgewiesen and len(ok) + len(abgewiesen) == 20
    assert st["cost_usd"] <= GRENZE and st["reserved_usd"] == 0 and st["reserved_requests"] == 0


def test_knapp_unter_der_grenze_geht_kein_grosser_aufruf_mehr(monkeypatch):
    monkeypatch.setattr(s, "aufrufen", lambda *a, **k: asyncio.sleep(0, antwort_fuer(a[0])))
    s.buchen("test", 1, {"input_tokens": 0}, GRENZE - 0.01)
    with pytest.raises(HTTPException) as e:
        asyncio.run(d.ausfuehren(rumpf(200_000), "test", versuche=1))
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) > 0
    assert stand()["cost_usd"] <= GRENZE


def test_zeitueberschreitung_wird_mit_der_schaetzung_gebucht(monkeypatch):
    async def haengt(body, timeout_s=15.0, versuche=3):
        raise httpx.ReadTimeout("zu langsam")
    monkeypatch.setattr(s, "aufrufen", haengt)
    with pytest.raises(HTTPException) as e:
        asyncio.run(d.ausfuehren(rumpf(100_000), "test", versuche=1))
    st = stand()
    erwartet = s.schaetzung(len(json.dumps(rumpf(100_000))), 1)
    assert e.value.status_code == 502 and st["cost_usd"] == pytest.approx(erwartet, rel=1e-6)
    assert st["errors"] == 1 and st["reserved_usd"] == 0


def test_zeitueberschreitungen_halten_die_grenze(monkeypatch):
    async def haengt(body, timeout_s=15.0, versuche=3):
        raise httpx.ReadTimeout("zu langsam")
    monkeypatch.setattr(s, "aufrufen", haengt)
    codes = []
    for _ in range(30):
        try:
            asyncio.run(d.ausfuehren(rumpf(390_000), "test", versuche=3))
        except HTTPException as e:
            codes.append(e.status_code)
    assert 429 in codes and stand()["cost_usd"] <= GRENZE


def test_nicht_gesendet_kostet_nichts(monkeypatch):
    async def weg(body, timeout_s=15.0, versuche=3):
        raise httpx.ConnectError("keine Verbindung")
    monkeypatch.setattr(s, "aufrufen", weg)
    with pytest.raises(HTTPException):
        asyncio.run(d.ausfuehren(rumpf(1000), "test", versuche=1))
    st = stand()
    assert st["cost_usd"] == 0 and st["errors"] == 1 and st["reserved_usd"] == 0


@pytest.mark.parametrize("grenze,zeichen,code", [(None, 400_100, 413), (None, 399_000, 200),
                                                 ("32000", 32_500, 413), ("32000", 31_000, 200)])
def test_groesse(monkeypatch, grenze, zeichen, code):
    if grenze:
        monkeypatch.setenv("DECISIONS_MAX_ZEICHEN", grenze)
    monkeypatch.setenv("DECISIONS_DAILY_MAX_USD", "100")
    monkeypatch.setattr(s, "aufrufen", lambda *a, **k: asyncio.sleep(0, antwort_fuer(a[0], 0.25)))
    try:
        asyncio.run(d.ausfuehren(rumpf(zeichen), "test", versuche=1))
        ist = 200
    except HTTPException as e:
        ist = e.status_code
    assert ist == code


def test_jev_openai_weg_hat_dieselbe_groessengrenze(monkeypatch):
    """/ai/jev?engine=openai laeuft ueber d.ausfuehren; die Grenze von Jev zweigte vorher ab."""
    from ai.routes import jev_routes as j
    from ai.services import decisions_service as ds
    monkeypatch.setenv("DECISIONS_MAX_ZEICHEN", "32000")
    gerufen = []
    monkeypatch.setattr(s, "aufrufen", lambda *a, **k: gerufen.append(1) or asyncio.sleep(0, antwort_fuer(a[0])))
    req = j.JevRequest(state="a" * 40_000, questions={"q": {"type": "noul", "instructions": "x"}})
    with pytest.raises(HTTPException) as e:
        asyncio.run(j._ueber_decisions(req, None, "test", 1))
    assert e.value.status_code == 413 and gerufen == []


def test_grenze_je_aufrufer_mit_reservierung(monkeypatch):
    monkeypatch.setenv("DECISIONS_DAILY_MAX_USD", "100")
    monkeypatch.setenv("DECISIONS_CALLER_LIMITS", json.dumps({"kunde": {"requests": 0, "usd": 0.05}}))

    async def langsam(body, timeout_s=15.0, versuche=3):
        await asyncio.sleep(0.05)
        return antwort_fuer(body)
    monkeypatch.setattr(s, "aufrufen", langsam)

    async def alle():
        return await asyncio.gather(*[d.ausfuehren(rumpf(100_000), "kunde", versuche=1) for _ in range(10)],
                                    return_exceptions=True)
    asyncio.run(alle())
    kunde = stand()["by_caller"]["kunde"]
    assert kunde["cost_usd"] <= 0.05 and kunde["reserved_usd"] == 0


def test_status_zeigt_reservierungen(monkeypatch):
    s.reservieren("test", 0.1)
    st = stand()
    assert st["reserved_usd"] == pytest.approx(0.1) and st["reserved_requests"] == 1
    assert st["limits"]["max_chars_per_request"] == 400_000
    s.abrechnen("test", 0.1, 1, {"input_tokens": 10}, 0.000001)
    assert stand()["reserved_usd"] == 0
