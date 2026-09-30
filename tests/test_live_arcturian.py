"""Arcturian ueber GPT-Live (Alex 30.09.2026): companion_mode=arcturian am
Live-Endpunkt. Stimme = kurze Arcturian-Sprechpersona; Backend (Responses-
Delegation) = der freigegebene Arcturian-Prompt + Resolver-Zusatz, Werkzeug
resolve_arcturian_turn mit tool_choice required. Ausfuehrung des Resolvers
bleibt beim Portal (wie heute)."""
import pytest
from fastapi import HTTPException

from ai.routes import realtime_routes as rr


def _req(**kw):
    b = dict(sdp="v=0", companion_mode="arcturian", language="de", confirm_api_billing=True)
    b.update(kw)
    return rr.RealtimeLiveSdpRequest(**b)


def test_arcturian_live_konfiguration():
    s = rr._live_session_config(_req())
    assert s["model"] == "gpt-live-1"
    assert "Arcturian" in s["instructions"] and len(s["instructions"]) < 3000
    d = s["delegation"]["responses"]
    # "required" bricht in GPT-Live die Fortsetzung (gemessen 30.09.)
    assert d["tool_choice"] == "auto"
    assert [t["name"] for t in d["tools"]] == ["resolve_arcturian_turn"]
    assert d["instructions"].startswith(rr._companion_arcturian_prompt("de")[:200])
    assert rr._arcturian_resolver_addendum("de", rr.DEFAULT_ARCTURIAN_RESOLVER) in d["instructions"]


def test_resolver_version_und_lesewerkzeuge():
    v3 = sorted(rr.SUPPORTED_ARCTURIAN_RESOLVERS)[-1]
    s = rr._live_session_config(_req(arcturian_resolver=v3, read_tools=True))
    namen = [t["name"] for t in s["delegation"]["responses"]["tools"]]
    assert namen[0] == "resolve_arcturian_turn" and "agent_status" in namen
    assert rr._arcturian_resolver_addendum("de", v3) in s["delegation"]["responses"]["instructions"]


def test_unbekannter_resolver_422():
    with pytest.raises(HTTPException) as e:
        rr._live_session_config(_req(arcturian_resolver="agentos.arcturian-action.v9"))
    assert e.value.status_code == 422


def test_unbekannter_modus_422():
    with pytest.raises(HTTPException) as e:
        rr._live_session_config(_req(companion_mode="product-finder"))
    assert e.value.status_code == 422 and e.value.detail["error"] == "unsupported_live_companion_mode"


def test_bildschirm_opt_in_im_backend():
    s = rr._live_session_config(_req(screen_tools=True))
    namen = [t["name"] for t in s["delegation"]["responses"]["tools"]]
    assert "screen_capture" in namen and "look_at_screen" in namen


def test_ohne_modus_bleibt_prototyp():
    s = rr._live_session_config(rr.RealtimeLiveSdpRequest(sdp="v=0", confirm_api_billing=True))
    assert s["delegation"]["responses"]["tool_choice"] == "auto"
    assert "resolve_arcturian_turn" not in [t["name"] for t in s["delegation"]["responses"]["tools"]]
