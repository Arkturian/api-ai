"""Referenzbilder auf dem Abo-Weg (codex image_gen), 3DNerd/Alex 11.10.

Gegen den echten Fehler gehalten: bis dahin wies api-ai Referenzen auf dem
Abo-Weg ab, obwohl das CLI sie annimmt (gemessen: Merkmale, die nur im
Bild standen, kamen im Ergebnis an). Die Tests halten fest, dass die
Bilder wirklich am codex-Aufruf haengen und nichts still verloren geht.
"""
import asyncio
import io
import json
import subprocess
import types

import pytest
from fastapi import HTTPException
from PIL import Image

from ai.clients import storage_client as sc
from ai.routes import image_ai_routes as g
from ai.routes import text_ai_routes as t
from ai.services import codex_imagegen as c


def _png(w=1536, h=1024):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def aufbau(monkeypatch, tmp_path):
    """codex, Download und Storage als Attrappen; alles dazwischen echt."""
    monkeypatch.delenv("CODEX_RUN_AS_USER", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    ziel = tmp_path / "generated_images" / "sitzung" / "bild.png"
    ziel.parent.mkdir(parents=True)
    spur = {"cmd": None, "stdin": None, "geladen": [], "gespeichert": [], "ref_dateien": []}

    def fake_cli(cmd, env, timeout=300, cwd="/", input=None):
        spur["cmd"], spur["stdin"] = list(cmd), input
        # Die angehaengten Dateien muessen beim Start existieren und lesbar sein.
        for i, teil in enumerate(cmd):
            if teil == "-i":
                spur["ref_dateien"].append(open(cmd[i + 1], "rb").read())
        ziel.write_bytes(_png())
        zeile = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps({"path": str(ziel), "width": 1536, "height": 1024})}})
        return subprocess.CompletedProcess(cmd, 0, stdout=zeile + "\n", stderr="")

    async def fake_download(urls):
        spur["geladen"].append(list(urls))
        return [(f"ref_{i}.png", f"BILD{i}".encode(), "image/png") for i, _ in enumerate(urls)]

    async def fake_save(**kw):
        spur["gespeichert"].append(kw)
        return types.SimpleNamespace(id=42, file_url="https://storage/42")

    monkeypatch.setattr(t, "_run_cli_with_pgid", fake_cli)
    monkeypatch.setattr(g, "_download_reference_images", fake_download)
    monkeypatch.setattr(sc, "save_file_and_record", fake_save)
    return spur


def _lauf(**kw):
    return asyncio.run(c.generate_with_codex_imagegen(
        prompt="Produktfoto", collection_id="ai-generated-images", link_id=None, **kw))


def test_referenzen_haengen_am_codex_aufruf(aufbau):
    urls = ["https://api-storage.arkturian.com/storage/media/1",
            "https://api-storage.arkturian.com/storage/media/2"]
    out = _lauf(reference_image_urls=urls)
    cmd = aufbau["cmd"]
    assert cmd.count("-i") == 2
    # `-i` ist variadisch: ohne `--` verschluckt es den Prompt-Platzhalter.
    assert cmd[-2:] == ["--", "-"]
    assert aufbau["geladen"] == [urls]
    assert aufbau["ref_dateien"] == [b"BILD0", b"BILD1"]      # die echten Bytes, in Reihenfolge
    assert "2 Referenzbilder" in aufbau["stdin"] and "Vorlage an image_gen" in aufbau["stdin"]
    assert out["reference_images_used"] == 2 and out["billing"] == "subscription"


def test_ohne_referenz_bleibt_der_aufruf_wie_bisher(aufbau):
    out = _lauf()
    assert "-i" not in aufbau["cmd"] and "--" not in aufbau["cmd"] and aufbau["cmd"][-1] == "-"
    assert aufbau["geladen"] == [] and "Referenzbild" not in aufbau["stdin"]
    assert out["reference_images_used"] == 0


def test_mehr_als_vier_referenzen_422_vor_jedem_lauf(aufbau):
    with pytest.raises(HTTPException) as e:
        _lauf(reference_image_urls=[f"https://api-storage.arkturian.com/storage/media/{i}" for i in range(5)])
    assert e.value.status_code == 422 and e.value.detail["max"] == c.MAX_REFERENZEN
    assert aufbau["cmd"] is None and aufbau["geladen"] == []


def test_unlesbare_referenz_stoppt_vor_codex(aufbau, monkeypatch):
    async def kaputt(urls):
        raise HTTPException(status_code=422, detail={"error": "reference_image_fetch_failed"})
    monkeypatch.setattr(g, "_download_reference_images", kaputt)
    with pytest.raises(HTTPException) as e:
        _lauf(reference_image_urls=["https://api-storage.arkturian.com/storage/media/130161"])
    assert e.value.detail["error"] == "reference_image_fetch_failed"
    assert aufbau["cmd"] is None and aufbau["gespeichert"] == []


def test_private_gilt_auch_mit_referenz(aufbau):
    _lauf(reference_image_urls=["https://api-storage.arkturian.com/storage/media/1"], private=True)
    gespeichert = aufbau["gespeichert"][0]
    assert gespeichert["private"] is True and gespeichert["is_public"] is False


def test_kopien_fuer_den_codex_benutzer_werden_aufgeraeumt(aufbau, monkeypatch, tmp_path):
    kopie = tmp_path / "kopie.png"

    def fake_kopieren(pfade):
        kopie.write_bytes(b"BILD0")
        return [str(kopie)], [str(kopie)]
    monkeypatch.setattr(t, "bilder_fuer_codex_benutzer", fake_kopieren)
    _lauf(reference_image_urls=["https://api-storage.arkturian.com/storage/media/1"])
    assert aufbau["cmd"][aufbau["cmd"].index("-i") + 1] == str(kopie)   # codex bekommt die lesbare Kopie
    assert not kopie.exists()                                           # und sie ist danach weg
