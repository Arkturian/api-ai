"""#4135 (Plan 6567 A1.2): /ai/verarbeitung je Mandant aus dem geprueften JWT.

Ein Kunden-Mandant (z. B. lesser) bekommt auf dem Kern nur seine eigene
Zuweisung (Schluesselweg) oder "keine KI" -- nie die Abos des Betreibers
(KI_VERARBEITUNG), nie dessen OpenAI-Schluessel. Ohne JWT: beobachten bzw. 401.
"""
import asyncio
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import jwt_pruefung
from ai.services import verarbeitung as v


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    def verboten(*a, **kw):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **kw):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("API_ACCESS_KEY", v.JWT_PFLICHT_ENV, v.BETREIBER_ENV, "OPENAI_TEXT_DAILY_MAX_USD",
              "API_AI_DIENST_CLIENT_ID", "API_AI_DIENST_SECRET", "API_AI_DIENST_TOKEN_URL", "TRESOR_URL"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv(v.ENV, "hydrogen")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-kern-schluessel")
    monkeypatch.setenv(v.MANDANTEN_DIR_ENV, str(tmp_path / "mandanten"))
    monkeypatch.setenv("KOSTENBUCH_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path / "auf"))
    monkeypatch.setenv("VERBRAUCH_DIR", str(tmp_path / "verbrauch"))
    monkeypatch.setenv("KI_KETTE", "agy,codex,claude")

    tokens = {"Bearer lesser": {"sub": "service:conversation-api", "tenant": "lesser"},
              "Bearer ark": {"sub": "service:conversation-api", "tenant": "arkturian"}}
    monkeypatch.setattr(jwt_pruefung, "claims_oder_none", lambda a: tokens.get(a))
    return tmp_path


@pytest.fixture
def wege(monkeypatch):
    gerufen = []

    async def agy(prompt, model=None, api_key=None, request=None):
        gerufen.append("agy")
        return t.AIResponse(response="antwort", model=model)

    async def nie(*a, **kw):
        raise AssertionError("anderer Weg gerufen")

    async def kern_openai(*a, **kw):
        raise AssertionError("OpenAI-Schluessel des Kerns gerufen")
    monkeypatch.setattr(t, "gemini_endpoint", agy)
    monkeypatch.setattr(t, "chatgpt_endpoint", nie)
    monkeypatch.setattr(t, "_claude_mit_rechten", nie)
    monkeypatch.setattr(t, "deepseek_endpoint", nie)
    monkeypatch.setattr(t, "_openai_einmal", kern_openai)
    return gerufen


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def _post(client, token=None, **kopf):
    h = dict(kopf)
    if token:
        h["Authorization"] = token
    return client.post("/ai/verarbeitung", json={"prompt": "Ordne ein"}, headers=h)


def _zuweisen(tmp_path, mandant, inhalt):
    d = tmp_path / "mandanten"
    d.mkdir(exist_ok=True)
    (d / f"{mandant}.json").write_text(json.dumps(inhalt) if not isinstance(inhalt, str) else inhalt)


def test_kunde_ohne_zuweisung_bekommt_nie_die_abos_des_betreibers(client, wege):
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "verarbeitung_nicht_zugewiesen"
    assert wege == []


def test_kunde_mit_keine_ki(client, wege, umgebung):
    _zuweisen(umgebung, "lesser", {"verarbeitung": "keine", "urteil": "keine"})
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "verarbeitung_keine_ki" and wege == []


@pytest.mark.parametrize("anbieter", ["hydrogen", "codex", "kette", "claude"])
def test_kunde_darf_keinen_abo_weg_zuweisen(client, wege, umgebung, anbieter):
    _zuweisen(umgebung, "lesser", {"verarbeitung": {"anbieter": anbieter, "verweis": "lesser-openai"}})
    r = _post(client, "Bearer lesser")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["grund"] == "anbieter_nicht_erlaubt" and wege == []


def test_kunde_ohne_dienstausweis_kein_konto(client, wege, umgebung):
    _zuweisen(umgebung, "lesser", {"verarbeitung": {"anbieter": "openai", "verweis": "lesser-openai",
                                                     "modell": "gpt-5.4-mini"},
                                   "deckel": {"tag_usd": 1, "monat_usd": 5}})
    r = _post(client, "Bearer lesser")
    d = r.json()["detail"]
    assert r.status_code == 503 and d["error"] == "konto_nicht_verfuegbar" and d["grund"] == "dienstausweis_fehlt"
    assert wege == []


@pytest.mark.parametrize("inhalt,grund", [
    ("{kaputt", "kein_json"),
    ({"verarbeitung": {"anbieter": "openai"}}, "verweis_fehlt"),
    ({"verarbeitung": {"anbieter": "openai", "verweis": "x", "modell": "a b"}}, "modell_ungueltig"),
    ({"urteil": "keine"}, "verarbeitung_fehlt"),
    ({"verarbeitung": {"anbieter": "openai", "verweis": "x"}}, "deckel_fehlt"),
])
def test_ungueltige_zuweisung(client, wege, umgebung, inhalt, grund):
    _zuweisen(umgebung, "lesser", inhalt)
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["grund"] == grund and wege == []


def test_betreiber_mit_jwt_nimmt_die_kern_einstellung(client, wege):
    r = _post(client, "Bearer ark")
    assert r.status_code == 200, r.text
    assert r.json()["anbieter"] == "hydrogen" and wege == ["agy"]


def test_ohne_jwt_beobachten_laeuft_mit_kern_einstellung(client, wege, caplog):
    with caplog.at_level("WARNING"):
        r = _post(client, **{"X-Agent-Name": "service:issue-api"})
    assert r.status_code == 200 and wege == ["agy"]
    assert any("verarbeitung OHNE JWT" in m for m in caplog.messages)


def test_ohne_jwt_mit_pflicht_401(client, wege, monkeypatch):
    monkeypatch.setenv(v.JWT_PFLICHT_ENV, "an")
    r = _post(client, **{"X-Agent-Name": "service:issue-api"})
    assert r.status_code == 401 and r.json()["detail"]["error"] == "jwt_required" and wege == []


def test_gefaelschter_mandant_im_kopf_zaehlt_nicht(client, wege, monkeypatch):
    """Ungepruftes JWT oder ein Kopf machen niemanden zum Kunden oder Betreiber."""
    monkeypatch.setenv(v.JWT_PFLICHT_ENV, "an")
    r = _post(client, "Bearer gefaelscht", **{"X-Caller-Tenant": "arkturian"})
    assert r.status_code == 401 and wege == []


def test_betreiber_ist_einstellbar(client, wege, monkeypatch):
    monkeypatch.setenv(v.BETREIBER_ENV, "lesser")
    monkeypatch.setenv("KI_VERARBEITUNG", "hydrogen")
    # Auf einer Kundeninstanz ist der eigene Mandant der Betreiber; arkturian waere dort fremd.
    assert _post(client, "Bearer lesser").status_code == 200
    assert _post(client, "Bearer ark").json()["detail"]["error"] == "verarbeitung_nicht_zugewiesen"


def test_kostenbuch_traegt_den_mandanten(client, wege, umgebung):
    _post(client, "Bearer ark")
    from ai.services import kostenbuch as kb
    kb.KONTEXT.set({"pfad": "/ai/x", "aufrufer": "a", "quelle": "jwt", "tenant": "lesser"})
    kb.buchen(anbieter="x", art="geld", usd=0.1)
    zeilen = [json.loads(z) for p in (umgebung / "kb").glob("*.jsonl") for z in p.read_text().splitlines()]
    assert zeilen[-1]["tenant"] == "lesser"


# --- A1.1/A1.5: Zuweisung setzen und lesen ------------------------------------

ADMIN_LESSER = {"sub": "17", "type": "user", "mandant_rechte": {"lesser": ["ai:zuweisung:verwalten"]}}
TOKENS = {
    "Bearer admin-lesser": ADMIN_LESSER,
    "Bearer admin-ele": {"sub": "18", "type": "user", "mandant_rechte": {"ele": ["ai:zuweisung:verwalten"]}},
    "Bearer nur-tenant": {"sub": "19", "type": "user", "tenant": "lesser"},
    "Bearer dienst-mit-map": {"sub": "service:x", "type": "service",
                              "mandant_rechte": {"lesser": ["ai:zuweisung:verwalten"]}},
    "Bearer kosten": {"sub": "agent:AiApi", "type": "agent", "permissions": ["ai:kosten:lesen"]},
}
ZUWEISUNG = {"verarbeitung": {"anbieter": "openai", "verweis": "lesser-openai-2026", "modell": "gpt-5.4-mini"},
             "urteil": "openai", "deckel": {"tag_usd": 1.0, "monat_usd": 20.0}}


@pytest.fixture
def rechte(monkeypatch):
    from ai.routes import verbrauch_routes as vr
    monkeypatch.setattr(jwt_pruefung, "pruefen", lambda a: TOKENS.get(a))

    async def kein_profil(a):
        return None
    monkeypatch.setattr(vr, "_profil", kein_profil)


def _put(client, token, tenant="lesser", body=None):
    return client.put(f"/ai/mandanten/{tenant}/ki", json=body if body is not None else ZUWEISUNG,
                      headers={"Authorization": token} if token else {})


def test_admin_des_mandanten_setzt_und_verarbeitung_liest_es(client, rechte, wege, umgebung):
    r = _put(client, "Bearer admin-lesser")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["zugewiesen"] and d["verarbeitung"]["anbieter"] == "openai" and d["deckel"]["tag_usd"] == 1.0
    assert d["verarbeitung"]["verweis"] == "lesser-opena…"          # nur gekuerzt
    # Die Zuweisung wirkt: Conversation von lesser bekommt den Schluesselweg, nicht die Kern-Abos
    v2 = _post(client, "Bearer lesser")
    assert v2.status_code == 503 and v2.json()["detail"]["error"] == "konto_nicht_verfuegbar" and wege == []
    assert (umgebung / "mandanten" / "protokoll.jsonl").read_text().count('"mandant": "lesser"') == 1


@pytest.mark.parametrize("token,status", [
    (None, 401),
    ("Bearer admin-ele", 403),            # Admin eines anderen Mandanten
    ("Bearer nur-tenant", 403),           # tenant-Claim ist kein Recht (Schluessel der Map zaehlt)
    ("Bearer dienst-mit-map", 403),       # nur Menschen-Tokens
    ("Bearer kosten", 403),               # Lesen ja, Setzen nein
])
def test_setzen_nur_mit_recht_fuer_genau_diesen_mandanten(client, rechte, umgebung, token, status):
    assert _put(client, token).status_code == status
    assert not (umgebung / "mandanten" / "lesser.json").exists()


def test_lesen_mit_kostenrecht_und_fremder_admin_nicht(client, rechte):
    assert client.get("/ai/mandanten/lesser/ki", headers={"Authorization": "Bearer kosten"}).status_code == 200
    assert client.get("/ai/mandanten/lesser/ki", headers={"Authorization": "Bearer admin-ele"}).status_code == 403


def test_betreiber_ist_nicht_zuweisbar(client, rechte, monkeypatch):
    TOKENS["Bearer admin-ark"] = {"sub": "1", "type": "user", "mandant_rechte": {"arkturian": ["ai:zuweisung:verwalten"]}}
    r = _put(client, "Bearer admin-ark", tenant="arkturian")
    assert r.status_code == 422 and r.json()["detail"]["error"] == "betreiber_nicht_zuweisbar"


@pytest.mark.parametrize("body,fehler", [
    ({"verarbeitung": {"anbieter": "openai", "verweis": "x"}}, "deckel_pflicht"),
    ({"verarbeitung": "keine", "urteil": "openai"}, "urteil_ohne_konto"),
])
def test_regeln_der_zuweisung(client, rechte, body, fehler):
    r = _put(client, "Bearer admin-lesser", body=body)
    assert r.status_code == 422 and r.json()["detail"]["error"] == fehler


@pytest.mark.parametrize("body", [
    {"verarbeitung": {"anbieter": "hydrogen", "verweis": "x"}, "deckel": {"tag_usd": 1, "monat_usd": 2}},
    {"verarbeitung": {"anbieter": "openai", "verweis": "x", "schluessel": "sk-…"},
     "deckel": {"tag_usd": 1, "monat_usd": 2}},
    {"verarbeitung": "keine", "api_key": "sk-…"},
])
def test_kein_abo_weg_und_kein_geheimnis_im_rumpf(client, rechte, umgebung, body):
    assert _put(client, "Bearer admin-lesser", body=body).status_code == 422
    assert not (umgebung / "mandanten" / "lesser.json").exists()


def test_keine_ki_setzen(client, rechte, wege):
    assert _put(client, "Bearer admin-lesser", body={"verarbeitung": "keine"}).status_code == 200
    assert _post(client, "Bearer lesser").json()["detail"]["error"] == "verarbeitung_keine_ki" and wege == []


def test_verbrauch_nur_des_eigenen_mandanten(client, rechte, umgebung):
    from ai.services import kostenbuch as kb
    for tenant, usd in (("lesser", 0.25), ("arkturian", 9.0), ("lesser", 0.5)):
        m = kb.KONTEXT.set({"pfad": "/ai/verarbeitung", "aufrufer": "x", "quelle": "jwt", "tenant": tenant})
        kb.buchen(anbieter="openai_text", art="geld", usd=usd)
        kb.KONTEXT.reset(m)
    d = client.get("/ai/mandanten/lesser/ki", headers={"Authorization": "Bearer kosten"}).json()
    assert d["verbrauch"]["heute"] == {"usd": 0.75, "buchungen": 2} and d["verbrauch"]["waehrung"] == "USD"



# --- A1.3: Schluesselweg mit dem Schluessel des Kunden aus dem Tresor --------------

@pytest.fixture
def kunde(monkeypatch, umgebung, wege):
    """Tresor und OpenAI nachgestellt; der echte _openai_einmal laeuft."""
    import httpx
    from ai.routes import text_ai_routes as tr
    from ai.services import openai_text as ot
    from ai.services import tresor
    monkeypatch.setattr(tr, "_openai_einmal", ECHT_OPENAI_EINMAL)
    gesehen = {"tresor": [], "openai_keys": []}

    async def tresor_lesen(verweis, mandant, zweck, aufrufer, anbieter):
        gesehen["tresor"].append((verweis, mandant, zweck, anbieter))
        return "sk-kunde-lesser"
    monkeypatch.setattr(tresor, "lesen", tresor_lesen)

    class _A:
        status_code = 200

        def json(self):
            return {"model": "gpt-6-luna", "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}

    async def post(self, url, json=None, headers=None, **kw):
        gesehen["openai_keys"].append((headers or {}).get("Authorization"))
        return _A()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    _zuweisen(umgebung, "lesser", {"verarbeitung": {"anbieter": "openai", "verweis": "lesser-openai",
                                                     "modell": "gpt-6-luna"},
                                   "deckel": {"tag_usd": 0.5, "monat_usd": 5.0}})
    return gesehen


from ai.routes import text_ai_routes as _tr_modul  # noqa: E402
ECHT_OPENAI_EINMAL = _tr_modul._openai_einmal


def test_kunde_rechnet_mit_seinem_schluessel_und_seinen_zaehlern(client, kunde, wege, umgebung):
    r = _post(client, "Bearer lesser")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["text"] == "ok" and d["konto"] == {"mandant": "lesser", "art": "schluessel_des_mandanten"}
    assert kunde["tresor"] == [("lesser-openai", "lesser", "verarbeitung", "openai")]
    assert kunde["openai_keys"] == ["Bearer sk-kunde-lesser"]          # nie sk-kern-schluessel
    assert wege == []
    verbrauch = umgebung / "verbrauch"
    assert list((verbrauch / "mandanten" / "lesser").glob("openai_text_*.json"))   # Zaehler des Mandanten
    assert not list(verbrauch.glob("openai_text_2*.json"))                      # Kern-Zaehler unberuehrt
    zeilen = [json.loads(z) for p in (umgebung / "kb").glob("*.jsonl") for z in p.read_text().splitlines()]
    k = [z for z in zeilen if z["anbieter"] == "openai_kunde"]
    assert len(k) == 1 and k[0]["art"] == "kunde_geld" and k[0]["tenant"] == "lesser" and k[0]["usd"] > 0
    assert "sk-kunde" not in json.dumps(zeilen)


def test_kunden_geld_nie_in_der_summe_des_betreibers(client, kunde, umgebung):
    _post(client, "Bearer lesser")
    from datetime import datetime
    from ai.services import kostenbuch as kb
    heute = datetime.now(kb.WIEN).date()
    b = kb.bericht(heute, heute, "weg")
    z = [x for x in b["zeilen"] if x["anbieter"] == "openai_kunde"]
    assert z and z[0]["zahlt"] == "kunde" and z[0]["betrag"] > 0
    assert b["summe"]["betrag"] == 0 and b["toepfe"] == []


def test_tagesdeckel_des_kunden_429_ohne_wechsel(client, kunde, wege, umgebung):
    d = umgebung / "verbrauch" / "mandanten" / "lesser"
    d.mkdir(parents=True)
    from datetime import datetime
    from ai.services import openai_text as ot
    tag = datetime.now(ot.WIEN).date().isoformat()
    (d / f"openai_text_{tag}.json").write_text(json.dumps({"usd": 0.5, "anfragen": 3}))
    r = _post(client, "Bearer lesser")
    assert r.status_code == 429 and r.json()["detail"]["grund"] == "tagesgrenze"
    assert kunde["openai_keys"] == [] and wege == []


def test_monatsdeckel_des_kunden(client, kunde, umgebung):
    d = umgebung / "verbrauch" / "mandanten" / "lesser"
    d.mkdir(parents=True)
    from datetime import datetime
    from ai.services import openai_text as ot
    monat = datetime.now(ot.WIEN).strftime("%Y-%m")
    (d / f"openai_text_{monat}-01.json").write_text(json.dumps({"usd": 5.0}))
    r = _post(client, "Bearer lesser")
    assert r.status_code == 429 and kunde["openai_keys"] == []


@pytest.mark.parametrize("grund", ["tresor_eintrag_unbekannt", "tresor_lesen_abgelehnt", "tresor_anbieter_passt_nicht"])
def test_tresor_fehler_ohne_ausweichen(client, kunde, wege, monkeypatch, grund):
    from ai.services import tresor

    async def kaputt(*a, **kw):
        raise tresor._nicht_verfuegbar(grund, "lesser")
    monkeypatch.setattr(tresor, "lesen", kaputt)
    r = _post(client, "Bearer lesser")
    assert r.status_code == 503 and r.json()["detail"]["grund"] == grund
    assert kunde["openai_keys"] == [] and wege == []


# --- Tresor-Client gegen den Vertrag der auth-api ------------------------------------

def test_tresor_client_vertrag(monkeypatch):
    import httpx
    from ai.services import tresor
    monkeypatch.setenv("API_AI_DIENST_CLIENT_ID", "cid")
    monkeypatch.setenv("API_AI_DIENST_SECRET", "geheim")
    monkeypatch.setenv("API_AI_DIENST_TOKEN_URL", "https://auth.example/api/v1/auth/oauth/token")
    monkeypatch.setattr(tresor, "_TOKEN", {"wert": None, "bis": 0.0})
    aufrufe = []

    class _R:
        def __init__(self, code, d):
            self.status_code, self._d = code, d

        def json(self):
            return self._d

    async def post(self, url, data=None, **kw):
        aufrufe.append(("post", url, data["grant_type"]))
        return _R(200, {"access_token": "dienst-jwt", "expires_in": 900})

    async def get(self, url, params=None, headers=None, **kw):
        aufrufe.append(("get", url, params, headers["Authorization"]))
        return _R(200, {"wert": "sk-x", "anbieter": "openai"})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    wert = asyncio.run(tresor.lesen("lesser/openai 1", "lesser", "verarbeitung", "service:conversation-api-lesser", "openai"))
    assert wert == "sk-x"
    assert aufrufe[0] == ("post", "https://auth.example/api/v1/auth/oauth/token", "client_credentials")
    assert aufrufe[1] == ("get", "https://auth.example/api/v1/tresor/lesser%2Fopenai%201",
                          {"mandant": "lesser", "zweck": "verarbeitung", "aufrufer": "service:conversation-api-lesser"},
                          "Bearer dienst-jwt")
