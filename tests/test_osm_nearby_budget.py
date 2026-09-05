"""osm_nearby: Zeitbudget der Quelle durchreichen, Default 12 s, Proxy-Timeout
ueber dem Budget (GuideDevBot2, Post 4913 T17: 9/9 ok:false nach 5,06 s,
weil der Proxy bei 5 s kappte, waehrend ArTrack 5–6 s auf drei Spiegel
verteilte)."""

import asyncio
import types

from ai.routes import realtime_routes as rr


class _Resp:
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d


def _fake_client(store):
    class _C:
        def __init__(self, *a, timeout=None, **k): store["timeout"] = timeout
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            store["url"] = url; store["params"] = params
            return _Resp({"text": "Kirche (place_of_worship, 80m)", "count": 1, "cached": False})
    return _C


def test_default_12_und_timeout_ueber_budget(monkeypatch):
    store = {}
    monkeypatch.setattr(rr.httpx, "AsyncClient", _fake_client(store))
    out = asyncio.run(rr._tool_osm_nearby({"lat": 47.27, "lng": 11.39}))
    assert store["params"]["budget_s"] == 12.0
    assert store["timeout"] > 12.0
    assert out["count"] == 1 and out["budget_s"] == 12.0


def test_budget_aus_argumenten_wird_durchgereicht_und_geklemmt(monkeypatch):
    store = {}
    monkeypatch.setattr(rr.httpx, "AsyncClient", _fake_client(store))
    asyncio.run(rr._tool_osm_nearby({"lat": 47.27, "lng": 11.39, "budget_s": 8}))
    assert store["params"]["budget_s"] == 8.0 and store["timeout"] == 11.0
    asyncio.run(rr._tool_osm_nearby({"lat": 47.27, "lng": 11.39, "budget_s": 99}))
    assert store["params"]["budget_s"] == 30.0                      # ArTrack-Obergrenze


def test_unbrauchbares_budget_faellt_auf_default():
    assert rr._osm_budget_s("zwoelf") == 12.0
    assert rr._osm_budget_s(None) == 12.0
    assert rr._osm_budget_s(float("nan")) == 12.0
    assert rr._osm_budget_s(0.1) == 0.5


def test_degradierte_quelle_wird_durchgereicht_nicht_als_leer_verkauft(monkeypatch):
    """ArTrack 200 + count 0 + degraded (alle Spiegel gescheitert) ist KEIN
    'nichts in der Naehe' — gemessen am Goldenen Dachl 05.09. 14:58."""
    store = {}

    class _C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            return _Resp({"text": "", "count": 0, "cached": False, "degraded": True,
                          "degraded_reason": "all Overpass mirrors failed: https://maps.mail.ru/…"})
    monkeypatch.setattr(rr.httpx, "AsyncClient", _C)
    out = asyncio.run(rr._tool_osm_nearby({"lat": 47.2685, "lng": 11.3933}))
    assert out["count"] == 0 and out["degraded"] is True
    assert "mirrors failed" in out["degraded_reason"] and "NICHT" in out["hint"]
