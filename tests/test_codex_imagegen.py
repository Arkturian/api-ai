"""Bilderzeugung ueber das ChatGPT-Abo (codex image_gen) und #1797.

Gemessen am 12.09. headless als Dienst-User: das Werkzeug liefert Bilder
ohne OPENAI_API_KEY, waehlt die Groesse selbst (1,3-2,4 MP), kann echte
Transparenz, und legt die Datei unter $CODEX_HOME/generated_images ab.

Was dieser Test festhaelt:
- Ziel: Pfad aus der Antwort, Masse aus dem PNG, nicht aus der Behauptung.
- Gegenfall: ein Pfad ausserhalb generated_images wird NICHT gelesen —
  das Modell nennt den Pfad, ein beliebiger Pfad waere ein Dateiabfluss.
- Nachbarfall: 4K oder Kanten > 2048 werden abgewiesen statt still
  unterboten; das ist die eine Zusage, die dieser Pfad nicht halten kann.
- #1797: negative_prompt wird im OpenAI-Pfad in den Prompt gefaltet und
  im Abo-Pfad in die Anweisung; ohne Negativ bleibt der Prompt unberuehrt.
"""

import inspect
import struct
import zlib

import pytest
from fastapi import HTTPException

from ai.services import codex_imagegen as ci
from ai.routes import image_ai_routes as g


def _png(w, h, rgba=False):
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if rgba else 2, 0, 0, 0)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


# ---------------------------------------------------- Masse aus der Datei

def test_png_masse_aus_ihdr():
    assert ci.png_masse(_png(2048, 1152)) == (2048, 1152)
    assert ci.png_masse(_png(940, 1672, rgba=True)) == (940, 1672)


def test_kein_png_wird_abgewiesen():
    with pytest.raises(ValueError):
        ci.png_masse(b"GIF89a....")


# ---------------------------------------------------- Antwort lesen

def test_pfad_aus_der_json_zeile():
    text = 'Bild erstellt.\n{"path": "/home/x/.codex/generated_images/s/a.png", "width": 1, "height": 2}\n'
    assert ci.pfad_aus_antwort(text) == "/home/x/.codex/generated_images/s/a.png"


def test_ohne_pfadzeile_kein_raten():
    assert ci.pfad_aus_antwort("Ich habe ein Bild gemacht, es liegt irgendwo.") is None
    assert ci.pfad_aus_antwort('{"width": 1}') is None


def test_jsonl_agent_message_und_fehler():
    raw = "\n".join([
        '{"type":"item.completed","item":{"type":"reasoning","text":"..."}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"path\\": \\"/p.png\\"}"}}',
        '{"type":"turn.completed","usage":{"input_tokens":1}}',
    ])
    text, fehler = ci.agentenantwort_aus_jsonl(raw)
    assert '"path"' in text and fehler is None
    text2, fehler2 = ci.agentenantwort_aus_jsonl('{"type":"turn.failed","error":{"message":"quota"}}')
    assert text2 == "" and fehler2 == "quota"


# ---------------------------------------------------- Pfad-Waechter

def test_nur_generated_images_wird_gelesen(tmp_path):
    home = tmp_path / ".codex"
    gut = home / "generated_images" / "sitzung" / "bild.png"
    gut.parent.mkdir(parents=True)
    gut.write_bytes(_png(1, 1))
    assert ci.pfad_ist_erlaubt(str(gut), home)
    # Gegenfaelle: ausserhalb, Traversal, falsche Endung
    (home / "auth.json").write_text("{}")
    assert not ci.pfad_ist_erlaubt(str(home / "auth.json"), home)
    assert not ci.pfad_ist_erlaubt(str(home / "generated_images" / ".." / "auth.json"), home)
    assert not ci.pfad_ist_erlaubt("/etc/passwd", home)
    boese = home / "generated_images" / "s" / "x.py"
    boese.parent.mkdir(parents=True, exist_ok=True)
    boese.write_text("")
    assert not ci.pfad_ist_erlaubt(str(boese), home)


# ---------------------------------------------------- Groessenzusage

def test_4k_wird_abgewiesen():
    with pytest.raises(HTTPException) as exc:
        ci.pruefe_groessenwunsch("4K", 1024, 1024)
    assert exc.value.status_code == 422
    assert exc.value.detail["paid_alternative"] == "gpt-image-2"


def test_kante_ueber_2048_wird_abgewiesen():
    with pytest.raises(HTTPException):
        ci.pruefe_groessenwunsch(None, 3840, 2160)


def test_vorgabewerte_gehen_durch():
    ci.pruefe_groessenwunsch(None, 1024, 1024)
    ci.pruefe_groessenwunsch("2K", 2048, 1152)


# ---------------------------------------------------- Anweisung

def test_anweisung_traegt_negativ_transparenz_und_verbote():
    a = ci.baue_anweisung("ein Apfel", "Text, Wasserzeichen", "16:9", "transparent")
    assert "16:9" in a
    assert "transparent" in a
    assert "NICHT im Bild enthalten: Text, Wasserzeichen" in a
    assert "KEINE Datei" in a and "KEINE Shell" in a
    assert '"path"' in a


def test_unbekanntes_seitenverhaeltnis_faellt_auf_quadrat():
    assert "Seitenverhaeltnis: 1:1." in ci.baue_anweisung("x", None, "7:5", None)


# ---------------------------------------------------- #1797 im OpenAI-Pfad

def test_negativ_wird_in_den_prompt_gefaltet():
    assert g.prompt_mit_negativ("ein Tor", "text labels, watermarks") == \
        "ein Tor\n\nDo not include: text labels, watermarks."
    assert g.prompt_mit_negativ("ein Tor", None) == "ein Tor"
    assert g.prompt_mit_negativ("ein Tor", "   ") == "ein Tor"


def test_openai_zweig_ruft_die_faltung():
    """Haelt die Verdrahtung: ohne diesen Aufruf steht negative_prompt
    wieder nur im Schema."""
    quelle = inspect.getsource(g.generate_image_endpoint)
    assert "prompt=prompt_mit_negativ(request.prompt, request.negative_prompt)" in quelle
    assert "generate_with_codex_imagegen(" in quelle


def test_modellnamen_des_abo_pfads():
    for name in ("codex-imagegen", "gpt-image-2-abo", "imagegen"):
        assert g.MODEL_MAPPING[name] == "codex-imagegen"
    assert g.is_codex_image_model("codex-imagegen")
    assert not g.is_openai_image_model("codex-imagegen")
