"""Projekt 32, Paket A2: Abschrift langer Sendungen (Vertrag radio-schnitt-v1, Beitrag 6630).

Ziel: Wörter aus Stücken ohne Lücke und ohne Doppel an den Nähten,
Fingerprint nach Testvektor, Peaks-Anzahl nach Vertrag, request_id-Regeln.
Gegenfälle: ohne confirm_api_billing 403; gleiche request_id mit anderem
Rumpf 409; erneuerte quelle_url startet keinen zweiten Lauf.
"""
import asyncio
import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.services import abschrift_lang as al
from ai.routes import abschrift_routes as ar


# ── Fingerprint ──────────────────────────────────────────────────────────────

def test_fingerprint_testvektor():
    w = [{"i": 0, "wort": "Guten", "start_ms": 420, "ende_ms": 710},
         {"i": 1, "wort": "Morgen", "start_ms": 730, "ende_ms": 1180}]
    assert al.fingerprint(w) == "sha256:4e812defa248a72b18f00643f23082127e595d1e28bf4e73add6d6fc5369136a"


def test_fingerprint_nfc():
    zerlegt = "Mädchen"       # a + kombinierendes Trema
    fertig = "Mädchen"
    a = al.fingerprint([{"i": 0, "wort": zerlegt, "start_ms": 0, "ende_ms": 10}])
    b = al.fingerprint([{"i": 0, "wort": fertig, "start_ms": 0, "ende_ms": 10}])
    assert a == b


# ── Peaks ────────────────────────────────────────────────────────────────────

def test_peaks_anzahl_vertragsbeispiel():
    assert al.peaks_anzahl(172871040, 48000) == 180074


def test_peaks_letztes_teilfenster_und_werte():
    sr = 48000
    n = int(12.345 * sr)                     # letztes Fenster unvollständig
    pcm = bytearray()
    for i in range(n):
        pcm += struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / sr)))
    daten, werte = al.peaks(bytes(pcm), sr)
    assert werte == math.ceil(n * 50 / sr) == 618
    assert len(daten) == 2 * werte
    lo, hi = struct.unpack("bb", daten[0:2])
    assert -63 <= lo <= -60 and 60 <= hi <= 63           # 16000/256 = 62,5
    lo, hi = struct.unpack("bb", daten[-2:])                # Teilfenster mit Inhalt
    assert lo < 0 < hi


def test_int8_grenzen():
    assert al._int8(-32768) == -128 and al._int8(32767) == 127 and al._int8(-255) == 0


# ── Stücke und Nähte ─────────────────────────────────────────────────────────

def test_stuecke_decken_ab_und_ueberlappen():
    st = al.stuecke(3_600_000, 1_320_000, 30_000)
    assert st[0][0] == 0 and st[-1][1] == 3_600_000
    for (a, b), (c, d) in zip(st, st[1:]):
        assert c == b - 30_000 and b - a <= 1_320_000
    assert al.stuecke(600_000, 1_320_000, 30_000) == [(0, 600_000)]


def _w(text, s, e):
    return {"word": text, "start": s, "end": e}


def test_naht_ohne_doppel_und_ohne_luecke():
    # Stück 0: 0–100 s, Stück 1: 70–170 s, Überlappung 70–100, Naht bei 85 s.
    t0 = {"von_ms": 0, "bis_ms": 100_000,
          "words": [_w("eins", 80.0, 80.5), _w("zwei", 84.0, 84.6), _w("drei", 86.0, 86.4), _w("vier", 99.0, 99.8)]}
    t1 = {"von_ms": 70_000, "bis_ms": 170_000,
          "words": [_w("eins", 10.0, 10.5), _w("zwei", 14.0, 14.6), _w("drei", 16.0, 16.4),
                    _w("vier", 29.0, 29.8), _w("fünf", 40.0, 40.5)]}
    w = al.woerter_zusammenfuehren([t0, t1])
    assert [x["wort"] for x in w] == ["eins", "zwei", "drei", "vier", "fünf"]
    assert [x["i"] for x in w] == list(range(5))
    assert all(a["ende_ms"] <= b["start_ms"] for a, b in zip(w, w[1:]))
    assert w[2]["start_ms"] == 86_000          # „drei" kommt aus Stück 1 (nach der Naht)


def test_monoton_und_unsicher_markiert():
    t = {"von_ms": 0, "bis_ms": 10_000,
         "words": [_w("a", 1.0, 1.5), _w("b", 1.4, 1.9), _w("c", 2.0, 2.0)]}
    w = al.woerter_zusammenfuehren([t])
    assert w[1]["start_ms"] == 1500 and w[1]["sicher"] is False
    assert w[2]["sicher"] is False


# ── Sprecher ─────────────────────────────────────────────────────────────────

def test_sprecher_und_ueberlappung():
    w = al.woerter_zusammenfuehren([{"von_ms": 0, "bis_ms": 20_000, "words": [
        _w("Hallo", 0.5, 1.0), _w("Gast", 5.0, 5.5), _w("zugleich", 9.2, 9.8)]}])
    seg = al.segmente_aus_diarize([{"von_ms": 0, "bis_ms": 20_000, "segments": [
        {"speaker": "B", "start": 0.0, "end": 4.0},
        {"speaker": "A", "start": 4.5, "end": 10.0},
        {"speaker": "B", "start": 9.0, "end": 12.0}]}])
    w, sprecher, unsicher = al.sprecher_zuordnen(w, seg)
    assert [s["id"] for s in sprecher] == ["S1", "S2"]        # Reihenfolge des Auftretens
    assert w[0]["sprecher"] == "S1" and w[1]["sprecher"] == "S2"
    assert {"von_ms": 9000, "bis_ms": 10000, "grund": "ueberlappung"} in unsicher
    assert w[2]["sicher"] is False and w[0]["sicher"] is True


# ── Anfrage-Identität ────────────────────────────────────────────────────────

def test_anfrage_hash_ohne_signierte_adresse():
    a = {"request_id": "rs-0000001", "quelle_storage_id": 5, "quelle_url": "https://x/1?sig=a", "sprache": "de"}
    b = dict(a, quelle_url="https://x/1?sig=b")
    c = dict(a, quelle_storage_id=6)
    assert al.anfrage_hash(a) == al.anfrage_hash(b)
    assert al.anfrage_hash(a) != al.anfrage_hash(c)


# ── Route ────────────────────────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(ar, "_aufrufer", lambda *a, **k: "test")
    gestartet = []

    async def lauf_ersatz(rid, req, url):
        gestartet.append((rid, url))
    monkeypatch.setattr(ar, "_lauf", lauf_ersatz)
    a = FastAPI()
    a.include_router(ar.router, prefix="/ai")
    a.state.gestartet = gestartet
    return a


def _rumpf(**kw):
    d = {"request_id": "rs-test-0001", "quelle_storage_id": 42, "quelle_url": "https://s/42?sig=1",
         "confirm_api_billing": True}
    d.update(kw)
    return d


def test_ohne_billing_403(app):
    r = TestClient(app).post("/ai/audio/abschrift", json=_rumpf(confirm_api_billing=False))
    assert r.status_code == 403 and r.json()["detail"]["error"] == "billing_noetig"
    assert app.state.gestartet == []


def test_erneuerte_adresse_derselbe_job_anderer_rumpf_409(app):
    c = TestClient(app)
    r1 = c.post("/ai/audio/abschrift", json=_rumpf())
    assert r1.status_code == 202 and r1.json()["job_id"] == "rs-test-0001"
    r2 = c.post("/ai/audio/abschrift", json=_rumpf(quelle_url="https://s/42?sig=2"))
    assert r2.status_code == 200 and r2.json()["wiederholt"] is True
    assert len(app.state.gestartet) == 1                     # keine zweite Buchung
    r3 = c.post("/ai/audio/abschrift", json=_rumpf(quelle_storage_id=43))
    assert r3.status_code == 409 and r3.json()["detail"]["error"] == "request_id_konflikt"


def test_neuanlauf_nur_vor_dem_anbieter(app):
    from ai.services import narrate_jobs as nj
    c = TestClient(app)
    c.post("/ai/audio/abschrift", json=_rumpf())
    nj.stufe("rs-test-0001", "laden")
    nj.scheitern("rs-test-0001", 502, {"error": "quelle_nicht_lesbar"})
    r = c.post("/ai/audio/abschrift", json=_rumpf(quelle_url="https://s/42?sig=3"))
    assert r.status_code == 202 and len(app.state.gestartet) == 2
    nj.stufe("rs-test-0001", "anbieter")
    nj.scheitern("rs-test-0001", 500, {"error": "x"})
    r = c.post("/ai/audio/abschrift", json=_rumpf())
    assert r.status_code == 200 and r.json()["status"] == "fehler"
    assert r.json()["neuanlauf_moeglich"] is False and len(app.state.gestartet) == 2


# ── Lauf mit Ersatz-Anbieter ─────────────────────────────────────────────────

def _ton_wav(pfad: Path, sekunden=3.0, sr=48000):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={sekunden}:r={sr}",
                    "-ac", "2", str(pfad)], check=True)


def test_lauf_ende_zu_ende(tmp_path, monkeypatch):
    from ai.services import narrate_jobs as nj
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    quelle = tmp_path / "q.wav"
    _ton_wav(quelle)

    async def holen(url, ziel):
        ziel.write_bytes(quelle.read_bytes())
        return quelle.stat().st_size, "abc"

    async def woerter(mp3, sprache, vorgabe, minuten):
        assert vorgabe == al.FUELL_VORGABE
        return [_w("Guten", 0.42, 0.71), _w("Morgen", 0.73, 1.18)]

    async def diarize(mp3, sprache, namen, refs, minuten):
        return [{"speaker": "A", "start": 0.0, "end": 2.0}]

    async def speichern(rid, erg, peaks):
        erg["abschrift_storage_id"] = 7
    monkeypatch.setattr(ar, "_quelle_holen", holen)
    monkeypatch.setattr(ar, "_whisper_woerter", woerter)
    monkeypatch.setattr(ar, "_diarize", diarize)
    monkeypatch.setattr(ar, "_referenzen_aus", lambda mp3, segs: ([], []))
    monkeypatch.setattr(ar, "_speichern", speichern)
    rid = "rs-lauf-0001"
    nj.reservieren(rid, "h", kind=ar.KIND)
    req = ar.AbschriftAnfrage(request_id=rid, quelle_storage_id=1, confirm_api_billing=True)
    asyncio.run(ar._lauf(rid, req, "https://s/1"))
    d = nj.lesen(rid)
    assert d["state"] == "done", d.get("error")
    e = d["result"]
    assert e["vertrag"] == "radio-schnitt-v2"
    assert e["quelle"]["samples"] == 144000 and e["quelle"]["sample_rate"] == 48000 and e["quelle"]["kanaele"] == 2
    assert e["peaks"]["werte"] == 150
    assert e["abschrift_fp"] == "sha256:4e812defa248a72b18f00643f23082127e595d1e28bf4e73add6d6fc5369136a"
    assert [w["sprecher"] for w in e["woerter"]] == ["S1", "S1"]
    assert ar._peaks_pfad(rid).stat().st_size == 300


def test_sprecher_ueber_stuecke_abgeglichen():
    # Langtest 28.11.: ein Erzähler bekam pro Stück ein neues Etikett (S1, dann S3).
    # Stück 0: A erzählt bis 80 s, B (Gast) bis 100 s. Stück 1 (ab 70 s) nennt
    # dieselben Menschen „X" (Erzähler) und „Y" (Gast); beide kommen in der Überlappung vor.
    teile = [
        {"von_ms": 0, "bis_ms": 100_000, "segments": [
            {"speaker": "A", "start": 0.0, "end": 80.0},
            {"speaker": "B", "start": 80.0, "end": 100.0}]},
        {"von_ms": 70_000, "bis_ms": 170_000, "segments": [
            {"speaker": "X", "start": 0.0, "end": 10.0},
            {"speaker": "Y", "start": 10.0, "end": 30.0},
            {"speaker": "X", "start": 30.0, "end": 100.0}]},
    ]
    seg = al.segmente_aus_diarize(teile)
    w = al.woerter_zusammenfuehren([{"von_ms": 0, "bis_ms": 170_000, "words": [
        _w("eins", 10.0, 10.5), _w("gast", 90.0, 90.5), _w("weiter", 150.0, 150.5)]}])
    w, sprecher, _ = al.sprecher_zuordnen(w, seg)
    assert [s["id"] for s in sprecher] == ["S1", "S2"]          # nur zwei Menschen
    assert [x["sprecher"] for x in w] == ["S1", "S2", "S1"]


def test_sprechername_ueber_praefix():
    seg = al.segmente_aus_diarize([{"von_ms": 0, "bis_ms": 10_000, "segments": [
        {"speaker": "Sabrina", "start": 0.0, "end": 5.0}]}])
    w = al.woerter_zusammenfuehren([{"von_ms": 0, "bis_ms": 10_000, "words": [_w("Hallo", 1.0, 1.5)]}])
    _, sprecher, _ = al.sprecher_zuordnen(w, seg, {"Sabrina": "Sabrina"})
    assert sprecher == [{"id": "S1", "name": "Sabrina"}]
