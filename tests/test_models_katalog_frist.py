"""Katalog-Frist (Automation, 23.09.2026): ein verpasster 03:00-Lauf darf
nicht sofort den statischen Fallback (6 Modelle) ausloesen. Bis 49 h gilt
der Katalog, ab 25 h als `stale: true` markiert; danach Fallback."""
import asyncio, json, os, time

import pytest

from ai.routes import text_ai_routes as t


@pytest.fixture
def katalog(tmp_path, monkeypatch):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"providers": {"codex": {"available": ["gpt-6-sol", "gpt-5.6-luna"], "default": "gpt-6-sol"}}}))
    monkeypatch.setattr(t, "_MODELS_STATE_PATH", p)
    def alter(h):
        ts = time.time() - h * 3600
        os.utime(p, (ts, ts))
    return alter


def _ids(r):
    return [m["id"] for m in r["models"]]


def test_frisch_nicht_stale(katalog):
    katalog(2)
    r = asyncio.run(t.list_text_models())
    assert "gpt-6-sol" in _ids(r) and r["stale"] is False


def test_verpasster_lauf_30h_liefert_katalog_markiert(katalog):
    katalog(30)
    r = asyncio.run(t.list_text_models())
    assert "gpt-6-sol" in _ids(r)
    assert r["stale"] is True and r["max_age_seconds"] == 49 * 3600


def test_ueber_49h_fallback(katalog):
    katalog(50)
    r = asyncio.run(t.list_text_models())
    assert "gpt-6-sol" not in _ids(r)
