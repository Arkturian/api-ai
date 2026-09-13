"""POST /ai/audio/mix — deterministisch, ohne Anbieteraufruf.

Vertrag Content #4976, p-b12b0d4d6062. Ziel: zwei Quellen an zwei
Startzeiten ergeben eine Datei mit der geplanten Laenge, und zweimal
dieselbe Eingabe ergibt dieselben Bytes. Gegenfall: Grenzen (Spurzahl,
Laenge, Format) -> 422 vor jedem Download. Nachbarfall: die Reihenfolge
der Spuren aendert den Hash nicht; kein Zaehler, keine Sperre wird
beruehrt — hier wird nichts erzeugt.
"""

import subprocess
import struct
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException

from ai.routes import audio_mix_routes as m


def _stille(pfad: Path, sekunden: float, ton_hz: int = 0, rate: int = 44100):
    n = int(sekunden * rate)
    import math
    frames = bytearray()
    for i in range(n):
        v = int(12000 * math.sin(2 * math.pi * ton_hz * i / rate)) if ton_hz else 0
        frames += struct.pack("<h", v)
    with wave.open(str(pfad), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(bytes(frames))


def _req(**kw):
    basis = dict(tracks=[m.MixTrack(audio_id=1, start_s=0.0), m.MixTrack(audio_id=2, start_s=1.5)])
    basis.update(kw)
    return m.MixRequest(**basis)


# ---------------------------------------------------- Grenzen vor dem Download

def test_grenzen():
    with pytest.raises(HTTPException) as e:
        m.pruefe_grenzen(m.MixRequest(tracks=[]))
    assert e.value.detail["error"] == "no_tracks"
    with pytest.raises(HTTPException) as e:
        m.pruefe_grenzen(m.MixRequest(tracks=[m.MixTrack(audio_id=i) for i in range(33)]))
    assert e.value.detail["error"] == "too_many_tracks"
    with pytest.raises(HTTPException) as e:
        m.pruefe_grenzen(_req(output_format="ogg"))
    assert e.value.detail["error"] == "unsupported_output_format"
    with pytest.raises(HTTPException) as e:
        m.pruefe_grenzen(_req(duration_s=1801))
    assert e.value.detail["error"] == "duration_too_long"
    m.pruefe_grenzen(_req())


# ---------------------------------------------------- Hash: Reihenfolge egal

def test_hash_unabhaengig_von_spurreihenfolge():
    a = _req()
    b = m.MixRequest(tracks=list(reversed(a.tracks)))
    assert m.mix_hash(a) == m.mix_hash(b)
    c = _req(); c.tracks[1].start_s = 2.0
    assert m.mix_hash(a) != m.mix_hash(c)
    d = _req(save_options={"is_public": False}, request_id="abc-12345")
    assert m.mix_hash(a) != m.mix_hash(d)           # Sichtbarkeit ist Teil des Ergebnisses
    e = _req(request_id="abc-12345")
    assert m.mix_hash(a) == m.mix_hash(e)           # die Kennung selbst nicht


# ---------------------------------------------------- Filtergraph

def test_filtergraph_platziert_und_schneidet():
    tracks = [m.MixTrack(audio_id=1, start_s=0.0), m.MixTrack(audio_id=2, start_s=1.5, source_offset_s=0.5, duration_s=1.0, gain_db=-6)]
    graph, gesamt, laengen = m.filtergraph(tracks, [2.0, 3.0], 44100, None, False)
    assert laengen[1] == (1.5, 1.0)
    assert "adelay=1500|1500" in graph
    assert "atrim=start=0.500:end=1.500" in graph
    assert "volume=-6.00dB" in graph
    assert "amix=inputs=2:dropout_transition=0,volume=2" in graph   # kein normalize= (ffmpeg < 4.4 auf arkturian)
    assert "normalize=" not in graph.split("amix")[1].split(",")[0]
    assert graph.count("apad=whole_dur=2.500") == 2
    assert gesamt == 2.5                       # Spur 2: 1.5 + 1.0
    graph2, gesamt2, _ = m.filtergraph(tracks, [2.0, 3.0], 44100, 10.0, False)
    assert gesamt2 == 10.0 and "atrim=end=10.000[out]" in graph2


# ---------------------------------------------------- echte Rechnung mit ffmpeg

def test_mischen_liefert_geplante_laenge_und_gleiche_bytes(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _stille(a, 1.0, ton_hz=440); _stille(b, 1.0, ton_hz=880)
    tracks = [m.MixTrack(audio_id=1, start_s=0.0), m.MixTrack(audio_id=2, start_s=1.5)]
    z1, z2 = tmp_path / "m1.wav", tmp_path / "m2.wav"
    g1, gesamt, _ = m.mischen([a, b], tracks, 44100, "wav", None, False, z1)
    m.mischen([a, b], tracks, 44100, "wav", None, False, z2)
    assert gesamt == 2.5
    assert abs(m._dauer(z1) - 2.5) < 0.05
    assert z1.read_bytes() == z2.read_bytes()   # deterministisch
    # zweite Spur beginnt wirklich bei 1,5 s: davor (1,0-1,5 s) ist Stille
    # volumedetect meldet auf Stufe info; mit -v error bleibt stderr leer.
    def pegel(ab, dauer):
        # -ss/-t VOR -i: ausgangsseitiges Suchen greift erst nach dem
        # Filter, volumedetect saehe dann die ganze Datei (gemessen 13.09.).
        r = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(ab), "-t", str(dauer), "-i", str(z1),
                            "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
        z = [l for l in r.stderr.splitlines() if "mean_volume" in l]
        assert z, r.stderr[-300:]
        return float(z[-1].split("mean_volume:")[1].split("dB")[0])
    assert pegel(1.05, 0.4) < -60          # Luecke zwischen den Spuren: Stille
    assert pegel(0.2, 0.4) > -30           # erste Spur: Ton
    assert pegel(1.8, 0.4) > -30           # zweite Spur ab 1,5 s: Ton


def test_kein_anbieter_kein_zaehler():
    import inspect
    q = inspect.getsource(m)
    assert "elevenlabs" not in q.lower().replace("nicht neu gebaut", "")  # keine Sperre, kein Zaehler
    assert "openai" not in q.lower()
    from main import _ist_bezahlpfad
    assert not _ist_bezahlpfad("/ai/audio/mix")


def test_route_registriert():
    from main import app
    assert any(getattr(r, "path", "") == "/ai/audio/mix" for r in app.routes)


# ---------------------------------------------------- Review q-d98ba93140f0

def test_quelleueberlauf_wird_genannt_nicht_gekappt():
    tr = [m.MixTrack(audio_id=7, start_s=0.0, source_offset_s=0.5, duration_s=2.0)]
    with pytest.raises(HTTPException) as e:
        m.filtergraph(tr, [2.0], 44100, None, False)        # 0.5 + 2.0 > 2.0
    assert e.value.detail["error"] == "source_overrun" and e.value.detail["audio_id"] == 7
    with pytest.raises(HTTPException) as e2:
        m.filtergraph([m.MixTrack(audio_id=7, source_offset_s=3.0)], [2.0], 44100, None, False)
    assert e2.value.detail["field"] == "source_offset_s"
    m.filtergraph([m.MixTrack(audio_id=7, source_offset_s=0.5, duration_s=1.5)], [2.0], 44100, None, False)  # exakt bis Ende: ok


def test_feste_gesamtdauer_wird_respektiert():
    tr = [m.MixTrack(audio_id=1, start_s=1.0)]
    with pytest.raises(HTTPException) as e:
        m.filtergraph(tr, [2.0], 44100, 2.5, False)         # endet bei 3.0 > 2.5
    assert e.value.detail["error"] == "track_exceeds_duration"
    _, gesamt, _ = m.filtergraph(tr, [2.0], 44100, 5.0, False)
    assert gesamt == 5.0


def test_nicht_endliche_zahlen_422():
    with pytest.raises(HTTPException) as e:
        m.pruefe_grenzen(m.MixRequest(tracks=[m.MixTrack(audio_id=1, start_s=float("inf"))]))
    assert e.value.detail["error"] == "non_finite_value"


def test_mix_status_route_und_trennung(monkeypatch, tmp_path):
    from ai.services import narrate_jobs as nj
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path))
    import asyncio
    nj.reservieren("mix-probe-0001", "h", kind="mix")
    d = asyncio.run(m.audio_mix_status("mix-probe-0001"))
    assert d["kind"] == "mix" and d["state"] == "running"
    nj.reservieren("sprech-probe-01", "h", kind="narrate")
    with pytest.raises(HTTPException) as e:
        asyncio.run(m.audio_mix_status("sprech-probe-01"))
    assert e.value.status_code == 404
    from main import app
    assert any(getattr(r, "path", "") == "/ai/audio/mix/{request_id}" for r in app.routes)


def test_frames_werden_dekodiert_gezaehlt(tmp_path):
    a = tmp_path / "a.wav"; _stille(a, 1.0, ton_hz=440)
    z = tmp_path / "z.wav"
    m.mischen([a], [m.MixTrack(audio_id=1, start_s=0.25)], 44100, "wav", None, False, z)
    frames, sr, ch = m._frames(z)
    assert sr == 44100 and ch == 2
    assert abs(frames - round(1.25 * 44100)) <= 1          # Ein-Sample-Regel
