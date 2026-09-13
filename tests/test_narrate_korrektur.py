"""Korrektur ohne Neusprechen (Aufnahme 125030, Reel #1818, 13.09.2026):
Dauer ungerundet, Enden geklemmt und benannt, Woerter aus dem
gespeicherten Zeichen-Alignment neu gruppiert, Herkunft am Ergebnis,
Replay liefert die Korrektur."""
import asyncio

import pytest
from fastapi import HTTPException

from ai.services import narrate_jobs as nj
from ai.services import narration_service as n
from ai.routes import narration_routes as r


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path))
    yield


RID = "story-korrektur-test-0001"


def _fertig(result, alignment=None):
    nj.anlegen(RID, "h", "narrate")
    nj.abschliessen(RID, result)
    if alignment is not None:
        nj.nebendatei_schreiben(RID, "alignment", alignment)


# --- Klemme -------------------------------------------------------------

def test_klemme_setzt_ende_auf_dauer_und_benennt_es():
    w = [{"word": "eine", "start": 135.559, "end": 135.71},
         {"word": "andere.", "start": 135.768, "end": 136.534}]
    raus, geklemmt, ms, weg = n.klemme_wortzeiten(w, 6021120 / 44100)
    assert geklemmt == 1 and weg == 0
    assert raus[-1]["end"] == 6021120 / 44100 and raus[-1]["end"] < 136.534
    assert 0.6 < ms < 0.8            # 136.534 - 136.53333 = 0.67 ms
    assert raus[0] == w[0]           # unberuehrt


def test_klemme_verwirft_statt_zu_erfinden():
    w = [{"word": "x", "start": 10.0, "end": 10.4}]
    raus, geklemmt, ms, weg = n.klemme_wortzeiten(w, 9.5)
    assert raus == [] and geklemmt == 1 and weg == 1


def test_klemme_ohne_ueberhang_aendert_nichts():
    w = [{"word": "x", "start": 0.0, "end": 0.4}]
    assert n.klemme_wortzeiten(w, 1.0) == (w, 0, 0.0, 0)
    assert n.klemme_wortzeiten(w, None)[0] == w


# --- korrigieren --------------------------------------------------------

def test_korrektur_traegt_herkunft_und_bewahrt_original():
    _fertig({"audio_id": 1, "word_timestamps": [{"word": "a\n\nb", "start": 0.0, "end": 1.0}]})
    d = nj.korrigieren(RID, {"audio_id": 1, "word_timestamps": []}, [{"what": "test"}], "test")
    h = d["result"]["corrected_from"]
    assert h["correction_no"] == 1 and h["changes"] == [{"what": "test"}] and h["source"] == "test"
    assert h["previous_result_sha256"] == nj.dict_hash(d["result_original"], ohne=())
    assert d["result_original"]["word_timestamps"][0]["word"] == "a\n\nb"
    d2 = nj.korrigieren(RID, {"audio_id": 1}, [], "test")
    assert d2["result"]["corrected_from"]["correction_no"] == 2
    assert d2["result_original"]["word_timestamps"][0]["word"] == "a\n\nb"   # das erste bleibt


def test_korrektur_nur_auf_fertigem_auftrag():
    nj.anlegen(RID, "h", "narrate")
    with pytest.raises(nj.KorrekturAbgelehnt) as e:
        nj.korrigieren(RID, {}, [], "test")
    assert e.value.status_code == 409
    with pytest.raises(nj.KorrekturAbgelehnt) as e:
        nj.korrigieren("gibt-es-nicht-1", {}, [], "test")
    assert e.value.status_code == 404


def test_grabstein_nimmt_nebendatei_mit(monkeypatch):
    import os, time
    _fertig({"audio_id": 1}, {"chunks": []})
    p = nj._pfad(RID)
    alt = time.time() - (nj.TTL_DAYS + 1) * 86400
    os.utime(p, (alt, alt))
    assert nj.aufraeumen() == 1
    assert nj.nebendatei_lesen(RID, "alignment") is None
    assert "result_original" not in nj.lesen(RID)


# --- Route /correct -----------------------------------------------------

def _align(text, dt=0.5, offset=0.0):
    chars = list(text)
    return {"chunk": 0, "time_offset": offset, "characters": chars,
            "character_start_times_seconds": [i * dt for i in range(len(chars))],
            "character_end_times_seconds": [(i + 1) * dt for i in range(len(chars))]}


def test_correct_gruppiert_neu_und_klemmt():
    # Alt: ein verschmolzenes Token und ein Ende hinter der Datei.
    text = "gehen?\n\nUnd"
    dauer_alt = 5.4
    _fertig({"audio_id": 7, "duration_seconds": dauer_alt, "frame_count": 238140, "sample_rate": 44100,
             "word_timestamps": [{"word": text, "start": 0.0, "end": 5.5}],
             "dramatic_script": text, "original_text": text},
            {"chunks": [_align(text)]})
    d = asyncio.run(r.narrate_correct(RID))
    w = d["result"]["word_timestamps"]
    assert [x["word"] for x in w] == ["gehen?", "Und"]
    assert w[0] == {"word": "gehen?", "start": 0.0, "end": 3.0}
    assert w[1]["start"] == 4.0 and w[1]["end"] == 238140 / 44100      # 5.5 -> Dauer
    assert d["result"]["word_timestamps_clamped"] == 1
    assert d["result"]["alignment_clamped_ms"] == pytest.approx(100.0, abs=0.01)
    assert d["result"]["duration_seconds"] == 238140 / 44100
    assert d["result"]["corrected_from"]["correction_no"] == 1
    assert d["result"]["corrected_from"]["changes"][0]["words_before"] == 1
    assert d["result"]["corrected_from"]["changes"][0]["words_after"] == 2
    assert d["result_original"]["word_timestamps"][0]["word"] == text
    # Replay liefert die Korrektur.
    assert nj.lesen(RID)["result"]["word_timestamps"][1]["word"] == "Und"


def test_correct_ohne_alignment_ist_409_und_spricht_nicht():
    _fertig({"audio_id": 7, "duration_seconds": 1.0, "word_timestamps": []})
    with pytest.raises(HTTPException) as e:
        asyncio.run(r.narrate_correct(RID))
    assert e.value.status_code == 409 and e.value.detail["error"] == "alignment_not_stored"
    assert nj.lesen(RID)["result"].get("corrected_from") is None


def test_alignment_route():
    _fertig({"audio_id": 7}, {"chunks": [_align("ab")]})
    assert asyncio.run(r.narrate_alignment(RID))["chunks"][0]["characters"] == ["a", "b"]
    nj.anlegen("ohne-alignment-0001", "h", "narrate"); nj.abschliessen("ohne-alignment-0001", {})
    with pytest.raises(HTTPException) as e:
        asyncio.run(r.narrate_alignment("ohne-alignment-0001"))
    assert e.value.status_code == 404 and e.value.detail["error"] == "alignment_not_stored"
    with pytest.raises(HTTPException) as e:
        asyncio.run(r.narrate_alignment("gibt-es-nicht-1"))
    assert e.value.status_code == 404


def test_statusrouten_verdecken_nichts():
    from main import app
    pfade = [getattr(x, "path", "") for x in app.routes]
    assert "/ai/tts/narrate/{request_id}/alignment" in pfade
    assert "/ai/tts/narrate/{request_id}/correct" in pfade
    # preview ist POST, die Statusroute GET — kein Schatten; ein GET auf
    # "preview" faengt die Statusroute selbst als 404 ab.
    with pytest.raises(HTTPException) as e:
        asyncio.run(r.narrate_status("preview"))
    assert e.value.status_code == 404
