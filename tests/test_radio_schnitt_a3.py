"""Projekt 32, Paket A3: deterministische Filter (Vertrag radio-schnitt-v2, Beitrag 6630).

Ziel: Stotterwort und abgebrochener Wortanfang als `raus`, Denkpause in Stille als
`kuerzen`, nie zwei Einträge an derselben Lücke, stabile ids, Datei-Fingerprint =
SHA-256 der Bytes. Gegenfälle: grammatisch mögliche Doppelung, unsichere Wörter,
Lücke mit Ton, falscher abschrift_fp.
"""
import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.services import vorschlaege as vs
from ai.routes import abschrift_routes as ar


def _w(i, wort, s, e, sicher=True):
    return {"i": i, "wort": wort, "start_ms": s, "ende_ms": e, "sprecher": "S1", "sicher": sicher}


def _peaks(dauer_ms, laut=((0, 10**9),), pegel=60):
    """Peaks mit Ton in den Bereichen `laut` (ms), sonst Stille."""
    n = dauer_ms // 20
    out = bytearray(2 * n)
    for k in range(n):
        t = k * 20
        if any(a <= t < b for a, b in laut):
            out[2 * k], out[2 * k + 1] = (-pegel) & 0xFF, pegel
    return bytes(out)


# ── Wiederholung ─────────────────────────────────────────────────────────────

def test_stotterwort_und_abgebrochener_anfang():
    w = [_w(0, "Heute", 0, 300), _w(1, "wir", 400, 600), _w(2, "wir", 650, 850),
         _w(3, "Wi-", 900, 1000), _w(4, "Wirtschaft.", 1050, 1600)]
    e = vs.wiederholungen(w, [])
    assert [(x["von"]["word_index"], x["von"]["expected_word"]) for x in e] == [(1, "wir"), (3, "Wi-")]
    assert all(x["art"] == "raus" and x["filter"] == "wiederholung" and x["von"] == x["bis"] for x in e)


def test_grammatisch_moegliche_doppelung_und_unsicher_bleiben_draussen():
    w = [_w(0, "Leute,", 0, 300), _w(1, "die", 350, 450), _w(2, "die", 500, 600),
         _w(3, "ja", 700, 800), _w(4, "ja", 850, 950, sicher=False),
         _w(5, "so", 1000, 1100), _w(6, "so", 1150, 1250)]
    assert vs.wiederholungen(w, [{"von_ms": 990, "bis_ms": 1300, "grund": "ueberlappung"}]) == []


# ── Pause ────────────────────────────────────────────────────────────────────

def test_pause_in_stille_und_luecke_mit_ton():
    w = [_w(0, "Frage", 0, 1000), _w(1, "Also", 2600, 3000), _w(2, "gut", 4200, 4600)]
    # Ton bis 1000, Stille 1000–2600 (1,6 s), Ton ab 2600 bis 4600 durchgehend (Lücke 3000–4200 hat Ton)
    peaks = _peaks(5000, laut=((0, 1000), (2600, 4600)))
    p = vs.pausen(w, [], peaks, ab_ms=800, ziel_ms=400)
    assert len(p) == 1
    assert p[0]["anker"] == {"nach_word_index": 0, "vor_word_index": 1, "expected_nach": "Frage", "expected_vor": "Also"}
    assert p[0]["luecke_ms"] == 1600 and p[0]["ziel_ms"] == 400 and p[0]["art"] == "kuerzen"


def test_schwelle_int8():
    assert vs.schwelle_int8(-40) == 1 and vs.schwelle_int8(-20) == 12 and vs.schwelle_int8(0) == 127


# ── Zusammenlegen, ids ───────────────────────────────────────────────────────

def _abschrift(woerter):
    return {"quelle": {"sha256": "q" * 64}, "abschrift_fp": "sha256:" + "a" * 64, "woerter": woerter, "unsicher": []}


def test_stotterwort_neben_pause_wird_ein_eintrag_mit_ersatzschnitt():
    w = [_w(0, "dass", 0, 300), _w(1, "wir", 350, 550), _w(2, "wir", 2000, 2200), _w(3, "gehen", 2250, 2600)]
    peaks = _peaks(3000, laut=((0, 550), (2000, 2600)))           # Stille 550–2000
    liste = vs.erzeugen(_abschrift(w), peaks, 7, {"wiederholung": True, "pause": {"ab_ms": 800, "ziel_ms": 400}})
    e = liste["eintraege"]
    assert len(e) == 1                                              # nie zwei Einträge an derselben Lücke
    assert e[0]["art"] == "raus" and [t["filter"] for t in e[0]["teile"]] == ["wiederholung", "pause"]
    assert e[0]["ohne_enthaltene"]["art"] == "kuerzen" and e[0]["ohne_enthaltene"]["luecke_ms"] == 1440
    assert liste["automatik_empfehlung"] == {"aeh": None, "fuellwort": None}


def test_id_stabil_und_fachlich_gebunden():
    q, fp = "q" * 64, "sha256:" + "a" * 64
    basis = {"art": "kuerzen", "filter": "pause", "anker": {"nach_word_index": 0}, "ziel_ms": 400,
             "begruendung": "Denkpause 1,6 s"}
    a = vs.eintrag_id(q, fp, basis)
    assert vs.eintrag_id(q, fp, dict(basis, begruendung="anders formuliert")) == a
    assert vs.eintrag_id(q, fp, dict(basis, ziel_ms=300)) != a
    assert vs.eintrag_id(q, "sha256:" + "b" * 64, basis) != a


def test_andere_pausenschwelle_laesst_stotter_id_gleich():
    w = [_w(0, "wir", 0, 200), _w(1, "wir", 250, 450), _w(2, "gehen", 500, 800)]
    peaks = _peaks(1000)
    a = vs.erzeugen(_abschrift(w), peaks, 7, {"wiederholung": True, "pause": {"ab_ms": 800}})
    b = vs.erzeugen(_abschrift(w), peaks, 7, {"wiederholung": True, "pause": {"ab_ms": 500}})
    assert [x["id"] for x in a["eintraege"]] == [x["id"] for x in b["eintraege"]]


def test_datei_fingerprint_ist_sha_der_bytes():
    daten, fp = vs.datei_bytes({"vertrag": "radio-schnitt-v2", "eintraege": []})
    assert fp == "sha256:" + hashlib.sha256(daten).hexdigest()


# ── Route ────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from ai.services import narrate_jobs as nj
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(ar, "_aufrufer", lambda *a, **k: "test")

    async def kein_storage(**kw):
        raise RuntimeError("kein storage im test")
    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "save_file_and_record", kein_storage)
    w = [_w(0, "wir", 0, 200), _w(1, "wir", 250, 450), _w(2, "gehen", 500, 800)]
    abschrift = dict(_abschrift(w), abschrift_storage_id=99)
    nj.reservieren("rs-abschrift-01", "h", kind=ar.KIND)
    nj.abschliessen("rs-abschrift-01", abschrift)
    ar._peaks_pfad("rs-abschrift-01").write_bytes(_peaks(1000))
    a = FastAPI()
    a.include_router(ar.router, prefix="/ai")
    return TestClient(a)


def _rumpf(**kw):
    d = {"request_id": "rs-vorschlag-01", "abschrift_storage_id": 99, "abschrift_fp": "sha256:" + "a" * 64}
    d.update(kw)
    return d


def test_route_liefert_datei_mit_passendem_fingerprint(client):
    r = client.post("/ai/audio/vorschlaege", json=_rumpf())
    assert r.status_code == 202, r.text
    e = r.json()["ergebnis"]
    assert e["anzahl"] == 1 and e["vertrag"] == "radio-schnitt-v2" and e["storage_fehler"] == "RuntimeError"
    daten = client.get("/ai/audio/vorschlaege/rs-vorschlag-01/datei").content
    assert e["vorschlaege_fp"] == "sha256:" + hashlib.sha256(daten).hexdigest()
    assert json.loads(daten)["eintraege"][0]["von"]["expected_word"] == "wir"
    # gleiche request_id, gleicher Rumpf: derselbe Job
    assert client.post("/ai/audio/vorschlaege", json=_rumpf()).json()["wiederholt"] is True


def test_route_falscher_fingerprint_und_unbekannte_abschrift(client):
    r = client.post("/ai/audio/vorschlaege", json=_rumpf(request_id="rs-vorschlag-02", abschrift_fp="sha256:" + "b" * 64))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "abschrift_veraltet"
    r = client.post("/ai/audio/vorschlaege", json=_rumpf(request_id="rs-vorschlag-03", abschrift_storage_id=5))
    assert r.status_code == 422 and r.json()["detail"]["error"] == "abschrift_unbekannt"
