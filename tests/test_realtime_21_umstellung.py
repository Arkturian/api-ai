"""#1881: gpt-realtime-2.1 als Vorgabe, gpt-realtime abgekuendigt (Abschaltung
20.01.2027). Gemessen 15.09.2026: 2.1 meldet in response.done.usage bei
Antworten mit reiner Textausgabe KEINE Eingabetoken (input_tokens 0, bei
Audioausgabe 857), obwohl der Kontext verbraucht wird (Rate-Limit-Rest
sank um ~1 800 bei gemeldeten 64). Der Zaehler schaetzt diese Eingabe aus
der letzten gemeldeten Antwort derselben Sitzung und benennt es."""
import asyncio

import pytest

from ai.routes import realtime_routes as r
from ai.services import openai_realtime_cost_tracker as k


def test_vorgabe_und_unterstuetzte_modelle():
    assert r.DEFAULT_REALTIME_MODEL == "gpt-realtime-2.1"
    assert {"gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime"} <= r.SUPPORTED_REALTIME_MODELS


@pytest.mark.parametrize("modell,posten,preis", [
    ("gpt-realtime-2.1", "text_input_per_1m", 4.0),
    ("gpt-realtime-2.1", "text_input_cached_per_1m", 0.40),
    ("gpt-realtime-2.1", "audio_input_per_1m", 32.0),
    ("gpt-realtime-2.1", "audio_input_cached_per_1m", 0.40),
    ("gpt-realtime-2.1", "text_output_per_1m", 24.0),      # 16 bei gpt-realtime
    ("gpt-realtime-2.1", "audio_output_per_1m", 64.0),
    ("gpt-realtime-2.1-mini", "text_input_per_1m", 0.60),
    ("gpt-realtime-2.1-mini", "text_input_cached_per_1m", 0.06),
    ("gpt-realtime-2.1-mini", "audio_input_per_1m", 10.0),
    ("gpt-realtime-2.1-mini", "audio_input_cached_per_1m", 0.30),
    ("gpt-realtime-2.1-mini", "text_output_per_1m", 2.40),
    ("gpt-realtime-2.1-mini", "audio_output_per_1m", 20.0),
])
def test_preise_2_1_entsprechen_der_liste_vom_15_09(modell, posten, preis):
    assert k.OPENAI_REALTIME_PRICING[modell][posten] == preis


def test_modelliste_nennt_abkuendigung_und_vorgabe():
    d = asyncio.run(r.list_realtime_models())
    ids = {m["id"]: m for m in d["models"]}
    assert ids["gpt-realtime-2.1"]["default"] is True
    assert ids["gpt-realtime"]["default"] is False
    assert ids["gpt-realtime"]["shutdown"] == "2027-01-20"
    assert ids["gpt-realtime"]["replacement"] == "gpt-realtime-2.1"
    assert "gpt-realtime-2.1-mini" in ids


# --- Eingabeschaetzung fuer 2.1 ------------------------------------------

@pytest.fixture
def zaehler(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_REALTIME_COST_TRACKER_DATA_DIR", str(tmp_path))
    z = object.__new__(k.OpenAIRealtimeCostTracker)
    z._initialized = False
    z.__init__()
    z.master_url = None
    z._reset_monthly_data()
    z._save_data()
    return z


def _melden(z, model, sid, eid, **kw):
    basis = dict(audio_input_tokens=0, audio_output_tokens=0, text_input_tokens=0, text_output_tokens=0,
                 cached_text_input_tokens=0, cached_audio_input_tokens=0)
    basis.update(kw)
    return z.track_session(model=model, voice_session_id=sid, usage_event_id=eid, **basis)


def test_textantwort_ohne_eingabe_wird_aus_der_vorigen_antwort_geschaetzt(zaehler):
    z = zaehler
    # Audio-Antwort meldet Eingabe: 15 000 Text (14 000 zwischengespeichert), 300 Audio.
    a = _melden(z, "gpt-realtime-2.1", "s1", "resp_1", text_input_tokens=15000, cached_text_input_tokens=14000,
                audio_input_tokens=300, audio_output_tokens=200, text_output_tokens=40)
    assert a["accepted"] and not a.get("input_estimated")
    vorher = z._usage_data["total_cost_usd"]
    # Affekt-Nachzug: reine Textausgabe, Eingabe 0 gemeldet.
    b = _melden(z, "gpt-realtime-2.1", "s1", "resp_2", text_output_tokens=30)
    assert b["accepted"] and b["input_estimated"] is True
    st = z._usage_data["by_model"]["gpt-realtime-2.1"]
    assert st["estimated_input_responses"] == 1
    assert st["estimated_input_tokens"] == 15300
    # Kosten: 1 000 Text voll (4.0) + 14 000 zwischengespeichert (0.40) + 300 Audio (32.0) + 30 Textausgabe (24.0)
    erwartet = (1000 * 4.0 + 14000 * 0.40 + 300 * 32.0 + 30 * 24.0) / 1e6
    assert z._usage_data["total_cost_usd"] - vorher == pytest.approx(erwartet, rel=1e-6)


def test_ohne_vorige_antwort_keine_schaetzung_aber_gezaehlt(zaehler):
    z = zaehler
    b = _melden(z, "gpt-realtime-2.1", "s2", "resp_1", text_output_tokens=30)
    assert b["accepted"] and b.get("input_estimated") is False
    assert z._usage_data["by_model"]["gpt-realtime-2.1"]["input_unknown_responses"] == 1


def test_altes_modell_wird_nicht_geschaetzt(zaehler):
    """gpt-realtime meldet die Eingabe auch bei Textausgabe (gemessen 742);
    eine Null dort ist eine Null."""
    z = zaehler
    _melden(z, "gpt-realtime", "s3", "resp_1", text_input_tokens=700, text_output_tokens=10)
    b = _melden(z, "gpt-realtime", "s3", "resp_2", text_output_tokens=10)
    assert b.get("input_estimated") is False
    assert "estimated_input_responses" not in z._usage_data["by_model"]["gpt-realtime"] or \
        z._usage_data["by_model"]["gpt-realtime"]["estimated_input_responses"] == 0


def test_schaetzung_ueberlebt_den_prozess(zaehler, tmp_path, monkeypatch):
    """Zwei Worker je Host: die letzte Eingabe einer Sitzung liegt in der
    Datei, nicht nur im Speicher."""
    z = zaehler
    _melden(z, "gpt-realtime-2.1", "s4", "resp_1", text_input_tokens=5000, text_output_tokens=10)
    z2 = object.__new__(k.OpenAIRealtimeCostTracker)
    z2._initialized = False
    z2.__init__()
    z2.master_url = None
    b = _melden(z2, "gpt-realtime-2.1", "s4", "resp_2", text_output_tokens=10)
    assert b["input_estimated"] is True
