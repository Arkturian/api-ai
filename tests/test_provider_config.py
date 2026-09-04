"""Fehlender Provider-Schluessel -> 403 provider_not_configured statt 500;
/health -> providers zeigt nur Wahrheitswerte (Steward/Davids Instanz,
Issue #36, via Cloud 05.09.)."""

import pytest
from fastapi import HTTPException

import main
from ai import provider_config as pc
from ai.clients import minimax_client


def test_providers_status_nur_bool_und_alle_envs_noetig(monkeypatch):
    for e in ("HIGGSFIELD_API_KEY", "HIGGSFIELD_API_SECRET", "OPENAI_API_KEY"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("HIGGSFIELD_API_KEY", "nur-der-key")
    st = pc.providers_status()
    assert st["openai"] is True
    assert st["higgsfield"] is False              # Secret fehlt
    assert set(map(type, st.values())) == {bool}
    assert "sk-x" not in str(st)


def test_provider_missing_ist_403_mit_klarer_meldung():
    exc = pc.provider_missing("minimax")
    assert isinstance(exc, HTTPException) and exc.status_code == 403
    assert exc.detail["error"] == "provider_not_configured"
    assert exc.detail["provider"] == "minimax"
    assert "MINIMAX_MULTIMODAL_API_KEY" in exc.detail["env"]
    assert "/health" in exc.detail["hint"]


def test_minimax_client_ohne_schluessel_wirft_403(monkeypatch):
    monkeypatch.delenv("MINIMAX_MULTIMODAL_API_KEY", raising=False)
    with pytest.raises(HTTPException) as ei:
        minimax_client._api_key()
    assert ei.value.status_code == 403 and ei.value.detail["error"] == "provider_not_configured"


def test_health_traegt_providers(monkeypatch):
    monkeypatch.setattr(main, "_cli_status", lambda name="codex": {"ok": True})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    h = main.health_check()
    assert h["providers"]["openai"] is True and "minimax" in h["providers"]
