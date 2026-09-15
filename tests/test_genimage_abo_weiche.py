"""Kleine Aufloesungen laufen ueber das ChatGPT-Abo, nicht ueber den
bezahlten Pfad (Alex, 15.09.2026). Die Weiche ist ehrlich: exakte Groesse
durch Einpassen, Rueckfall benannt, nie still ein anderes Bild."""
import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from ai.routes import image_ai_routes as g
from ai.services import codex_imagegen as c


def _png(w, h, mode="RGB"):
    im = Image.new(mode, (w, h), (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30))
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def _masse(data):
    return Image.open(io.BytesIO(data)).size


# --- Einpassen -----------------------------------------------------------

def test_einpassen_verkleinert_und_schneidet_mittig():
    data, info = c.passe_ein(_png(1254, 1254), 1024, 1024)
    assert _masse(data) == (1024, 1024)
    assert info["source"] == [1254, 1254] and info["upscaled"] is False


def test_einpassen_bewahrt_alpha():
    data, _ = c.passe_ein(_png(1254, 1254, "RGBA"), 1024, 1024)
    assert Image.open(io.BytesIO(data)).mode == "RGBA"


def test_einpassen_anderes_seitenverhaeltnis_deckt_und_schneidet():
    data, info = c.passe_ein(_png(1672, 941), 1536, 1024)     # gemessene Abo-Groesse fuer 16:9
    assert _masse(data) == (1536, 1024)
    assert info["scale"] == pytest.approx(1024 / 941, abs=1e-3) and info["upscaled"] is True


def test_einpassen_kleine_hochskalierung_erlaubt_und_benannt():
    data, info = c.passe_ein(_png(940, 1672), 1024, 1536)
    assert _masse(data) == (1024, 1536) and info["upscaled"] is True and info["scale"] < 1.1


def test_einpassen_verweigert_zu_kleine_quelle():
    with pytest.raises(c.QuelleZuKlein) as e:
        c.passe_ein(_png(800, 800), 1024, 1024)
    assert e.value.scale == pytest.approx(1.28)


def test_seitenverhaeltnis_aus_masse():
    assert c.seitenverhaeltnis_aus(1024, 1024, None) == "1:1"
    assert c.seitenverhaeltnis_aus(1536, 1024, "1:1") in ("3:2", "16:9", "4:3")   # naechstes erlaubtes, quer
    assert c.seitenverhaeltnis_aus(1024, 1536, "1:1") in ("2:3", "9:16", "3:4")   # hoch
    assert c.seitenverhaeltnis_aus(None, None, "16:9") == "16:9"


# --- Weiche --------------------------------------------------------------

def _req(**kw):
    basis = dict(prompt="ein Tor", model="gpt-image-2")
    basis.update(kw)
    return g.ImageGenRequest(**basis)


def test_abo_eignung():
    assert g._abo_geeignet(_req())[0] is True                                  # 1024x1024 Vorgabe
    assert g._abo_geeignet(_req(width=1024, height=1536))[0] is True
    assert g._abo_geeignet(_req(width=1536, height=1024))[0] is True
    assert g._abo_geeignet(_req(width=2048, height=1152))[0] is False
    assert g._abo_geeignet(_req(width=3840, height=2160))[0] is False
    assert g._abo_geeignet(_req(image_size="4K"))[0] is False
    assert g._abo_geeignet(_req(reference_image_urls=["https://x/1.png"]))[0] is False
    assert g._abo_geeignet(_req(route="api"))[0] is False
    assert g._abo_geeignet(_req(model="minimax-image-01"))[0] is False


def test_unbekannte_route_ist_422():
    with pytest.raises(HTTPException) as e:
        asyncio.run(g._generate_image_einmal(_req(route="egal"), "x"))
    assert e.value.status_code == 422 and e.value.detail["error"] == "unknown_route"


def _verdrahten(monkeypatch, abo_fehler=None):
    aufrufe = {"abo": [], "api": []}

    async def fake_abo(**kw):
        aufrufe["abo"].append(kw)
        if abo_fehler:
            raise abo_fehler
        return {"id": 1, "image_url": "u", "file_url": "u", "storage_object_id": 1, "request_id": "codex_x",
                "width": kw.get("width"), "height": kw.get("height"), "provider": "codex-imagegen",
                "billing": "subscription", "size_guaranteed": True}

    async def fake_api(**kw):
        aufrufe["api"].append(kw)
        return {"id": 2, "image_url": "p", "file_url": "p", "storage_object_id": 2,
                "width": kw.get("width"), "height": kw.get("height"), "provider": "openai"}

    monkeypatch.setattr(c, "generate_with_codex_imagegen", fake_abo)
    monkeypatch.setattr(g, "generate_with_openai_image", fake_api)
    monkeypatch.setattr(g, "_check_openai_billing_gate", lambda *a, **k: None)
    return aufrufe


def test_kleine_groesse_laeuft_ueber_das_abo_und_sagt_es(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    r = asyncio.run(g._generate_image_einmal(_req(width=1024, height=1536), "x"))
    assert aufrufe["api"] == [] and len(aufrufe["abo"]) == 1
    assert aufrufe["abo"][0]["width"] == 1024 and aufrufe["abo"][0]["height"] == 1536
    assert aufrufe["abo"][0]["aspect_ratio"] == c.seitenverhaeltnis_aus(1024, 1536, "1:1")
    assert r["model"] == "codex-imagegen" and r["model_requested"] == "gpt-image-2"
    assert r["routed_via"] == "subscription" and r["billing"] == "subscription"


def test_grosse_groesse_bleibt_bezahlt(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    r = asyncio.run(g._generate_image_einmal(_req(width=2048, height=1152, confirm_api_billing=True), "x"))
    assert aufrufe["abo"] == [] and len(aufrufe["api"]) == 1
    assert r["model"] == "gpt-image-2" and r["routed_via"] == "api" and r.get("model_requested") is None


def test_route_api_erzwingt_bezahlt(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    r = asyncio.run(g._generate_image_einmal(_req(route="api", confirm_api_billing=True), "x"))
    assert aufrufe["abo"] == [] and len(aufrufe["api"]) == 1 and r["routed_via"] == "api"


def test_route_subscription_erzwingt_abo_auch_fuer_gpt_image_2(monkeypatch):
    aufrufe = _verdrahten(monkeypatch)
    r = asyncio.run(g._generate_image_einmal(_req(route="subscription"), "x"))
    assert len(aufrufe["abo"]) == 1 and aufrufe["api"] == []
    assert r["model"] == "codex-imagegen" and r["routed_via"] == "subscription"


def test_abo_scheitert_mit_kostenzustimmung_faellt_benannt_auf_bezahlt(monkeypatch):
    aufrufe = _verdrahten(monkeypatch, abo_fehler=HTTPException(status_code=502, detail={"error": "codex_imagegen_failed"}))
    r = asyncio.run(g._generate_image_einmal(_req(confirm_api_billing=True), "x"))
    assert len(aufrufe["abo"]) == 1 and len(aufrufe["api"]) == 1
    assert r["routed_via"] == "api" and r["subscription_attempt"]["error"] == "codex_imagegen_failed"
    assert r["model"] == "gpt-image-2"


def test_abo_scheitert_ohne_kostenzustimmung_ist_fehler_nicht_rechnung(monkeypatch):
    aufrufe = _verdrahten(monkeypatch, abo_fehler=HTTPException(status_code=502, detail={"error": "codex_imagegen_failed"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(g._generate_image_einmal(_req(), "x"))
    assert e.value.status_code == 502 and aufrufe["api"] == []
    assert "confirm_api_billing" in str(e.value.detail)


def test_verdrahtung_sitzt_im_rumpf():
    import inspect
    q = inspect.getsource(g._generate_image_einmal)
    assert "_abo_geeignet(" in q and "routed_via" in q
