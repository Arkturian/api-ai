"""#2937 Folgefehler (3DTischler, 16.11. 12:21 UTC): Abo-Bild mit Referenz -> 422.

Als apiai legt codex-imagegen seinen Arbeitsordner unter arbeit/ an statt
unter /tmp. Die Referenzen landen dort, und bilder_fuer_codex_benutzer
pruefte sie gegen VISION_LOKALE_WURZELN (/mnt/backup-disk, /tmp, /var/tmp):
local_path_rejected path_outside_allowed_roots. Die bisherigen Referenz-Tests
ersetzten bilder_fuer_codex_benutzer durch eine Attrappe und liefen ohne
Dienstmodus, darum blieb es gruen.
"""
import asyncio
import io
import json
import os
import pwd
import subprocess
import types

import pytest
from fastapi import HTTPException
from PIL import Image

from ai.clients import storage_client as sc
from ai.routes import image_ai_routes as g
from ai.routes import text_ai_routes as t
from ai.services import cli_dienst, cli_katalog
from ai.services import codex_imagegen as c


def _png(w=1536, h=1024):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _fremder_benutzer():
    for name in ("alex", "nobody", "root"):
        try:
            pw = pwd.getpwnam(name)
        except KeyError:
            continue
        if pw.pw_uid != os.geteuid():
            return name
    pytest.skip("kein zweiter Benutzer")


@pytest.fixture
def dienst(monkeypatch, tmp_path):
    ein, arb, text, disk = (tmp_path / n for n in ("eingang", "arbeit", "codex-text", "disk"))
    for d in (ein, arb, text, disk):
        d.mkdir()
    monkeypatch.setenv("API_AI_CLI_WRAPPER_DIR", "/usr/local/libexec/api-ai")
    monkeypatch.setenv("API_AI_CLI_EINGANG", str(ein))
    monkeypatch.setenv("API_AI_CLI_ARBEIT", str(arb))
    monkeypatch.setenv("CODEX_TEXT_HOME", str(text))
    monkeypatch.setenv("CODEX_RUN_AS_USER", _fremder_benutzer())
    # Wie am Ziel: arbeit/ liegt NICHT unter den Vision-Wurzeln. (tmp_path
    # liegt unter /tmp und waere sonst ueber die Vorgabe erlaubt.)
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(disk))
    monkeypatch.setattr(os, "chown", lambda *a, **k: (_ for _ in ()).throw(AssertionError("chown")))
    return types.SimpleNamespace(eingang=ein, arbeit=arb, text=text, disk=disk)


@pytest.fixture
def lauf(dienst, monkeypatch):
    spur = {"refs": [], "cmd": None}
    ziel = dienst.text / "generated_images" / "s" / "bild.png"

    def fake_cli(cmd, env, timeout=300, cwd="/", input=None, phasen=None):
        spur["cmd"] = list(cmd)
        for i, teil in enumerate(cmd):
            if teil == "-i":
                spur["refs"].append((cmd[i + 1], open(cmd[i + 1], "rb").read()))
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_bytes(_png())
        zeile = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps({"path": str(ziel), "width": 1536, "height": 1024})}})
        return subprocess.CompletedProcess(cmd, 0, stdout=zeile + "\n", stderr="")

    async def fake_download(urls):
        return [(f"ref_{i}_ab12cd.png", f"BILD{i}".encode(), "image/png") for i, _ in enumerate(urls)]

    async def fake_save(**kw):
        return types.SimpleNamespace(id=42, file_url="https://storage/42")

    monkeypatch.setattr(t, "_run_cli_with_pgid", fake_cli)
    monkeypatch.setattr(g, "_download_reference_images", fake_download)
    monkeypatch.setattr(sc, "save_file_and_record", fake_save)
    monkeypatch.setattr(cli_katalog, "codex_config", lambda haupt: ("gpt-6.1-terra", "low"))
    monkeypatch.setattr(c, "codex_home", lambda: dienst.text)

    def starten():
        return asyncio.run(c.generate_with_codex_imagegen(
            prompt="Tisch", collection_id="ai-generated-images", link_id=None,
            width=1536, height=1024, reference_image_urls=["https://storage/signiert/1"]))
    return types.SimpleNamespace(spur=spur, starten=starten)


def test_abo_bild_mit_referenz_im_dienstmodus(lauf, dienst):
    lauf.starten()
    [(pfad, daten)] = lauf.spur["refs"]
    assert daten == b"BILD0"
    # codex bekommt die Kopie im Eingang (der Wrapper laesst -i nur dort zu).
    assert pfad.startswith(str(dienst.eingang) + os.sep)


def test_fremder_pfad_unter_arbeit_bleibt_gesperrt(dienst):
    """Nur der eigene Arbeitsordner gilt, nicht ganz arbeit/ (fremde Laeufe)."""
    eigener = cli_dienst.arbeitsordner("codex-imagegen-")
    fremder = cli_dienst.arbeitsordner("codex-imagegen-")
    bild = os.path.join(fremder, "ref_0.png")
    open(bild, "wb").write(b"PNG")
    with pytest.raises(HTTPException) as e:
        t.bilder_fuer_codex_benutzer([bild], eigene_ordner=(eigener,))
    assert e.value.status_code == 422
    assert "path_outside_allowed_roots" in e.value.detail["reason"]


def test_ohne_eigene_ordner_bleibt_die_alte_pruefung(dienst):
    d = cli_dienst.arbeitsordner("codex-imagegen-")
    bild = os.path.join(d, "ref_0.png")
    open(bild, "wb").write(b"PNG")
    with pytest.raises(HTTPException):
        t.bilder_fuer_codex_benutzer([bild])
