"""Der Einwilligungsriegel vor der dauerhaften Sprachablage (#1267).

CloudV2 fand im nie zusammengeführten Zweig `arcturian-continuity-unmerged`
einen ausdrücklichen Browser-Riegel mit der Begründung: die Verfügbarkeit
eines Endpunkts darf dauerhaften Sprachtext nicht einschalten. Der Riegel
kam nie auf `main` — und genau das trat ein. Auf arkturian lagen am
2026-08-22 74 Sitzungen mit 1166 Zeilen wörtlicher Rede, ohne Frist, ohne
Aufräum-Job, ohne Zustimmung.

Die Lehre daraus steckt in der Verteilung der Zuständigkeit: Ein Riegel im
Browser bewacht nur den Client, der ihn liest. Er gehört hierher.

Der Schalter steht auf AUS. Das ist der Unterschied zwischen einer
Fähigkeit und einem Übergriff: Wer die Ablage benutzt, entscheidet über
den Riegel — nicht wer ihn ausliefert.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ai.routes import realtime_routes as rr


def _grant():
    return SimpleNamespace(
        sub="11111111-2222-3333-4444-555555555555",
        tenant_id="arkturian",
        profile_id="alex-default",
    )


def _body(consent=None):
    return rr.DevlogUpsertRequest(
        voice_session_id="vs_test_1267",
        agent="CloudV2",
        retention_consent=consent,
        lines=[rr.DevlogLine(role="you", text="gesprochener Satz")],
    )


def _ruf(body):
    return asyncio.run(rr.realtime_devlog_upsert(body=body, grant=_grant()))


@pytest.fixture
def ablage(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "DEVLOG_ROOT", str(tmp_path))
    return tmp_path


def _dateien(p):
    return list(p.glob("*/*.json"))


# ─────────────────────────────────────────── Schalter AUS (heutiger Stand)

def test_ohne_schalter_bleibt_alles_wie_es_war(ablage, monkeypatch):
    """Ausrollen darf dem einzigen Anrufer nichts wegnehmen.

    CloudV2 sendet das Feld heute nicht. Käme der Riegel scharf heraus,
    wäre die Mitschnitt-Funktion beim Deploy still kaputt — dieselbe
    Sorte Schaden, nur mit umgekehrtem Vorzeichen.
    """
    monkeypatch.delenv(rr.DEVLOG_CONSENT_ENV, raising=False)
    antwort = _ruf(_body(consent=None))
    assert antwort["accepted"] is True
    assert len(_dateien(ablage)) == 1


def test_zustimmung_wird_mitgeschrieben_auch_ohne_schalter(ablage, monkeypatch):
    """Sonst wäre nach dem Umlegen niemand nachweislich einverstanden."""
    monkeypatch.delenv(rr.DEVLOG_CONSENT_ENV, raising=False)
    _ruf(_body(consent=True))
    rec = json.loads(_dateien(ablage)[0].read_text())
    assert rec["retention_consent"] is True
    assert rec["consent_enforced"] is False


# ────────────────────────────────────────────────────── Schalter AN

@pytest.mark.parametrize("wert", ["1", "true", "yes", "on", "TRUE"])
def test_schalter_versteht_die_ueblichen_schreibweisen(monkeypatch, wert):
    monkeypatch.setenv(rr.DEVLOG_CONSENT_ENV, wert)
    assert rr._devlog_consent_required() is True


@pytest.mark.parametrize("consent", [None, False])
def test_scharf_ohne_zustimmung_schreibt_NICHTS(ablage, monkeypatch, consent):
    """Der eigentliche Satz dieses Tests: nichts auf der Platte.

    Ein 403, der trotzdem schreibt, wäre kein Riegel — und genau diese
    Reihenfolge ging hier schon einmal schief: Der NameError im selben
    Endpunkt lag hinter `os.replace`, die Datei war da und der Anrufer
    bekam einen Fehler. Deshalb wird hier das Verzeichnis geprüft und
    nicht der Statuscode allein.

    `None` gehört ausdrücklich dazu: Ein Client, der von dem Feld nichts
    weiss, darf nicht der sein, für den der Riegel offen steht.
    """
    monkeypatch.setenv(rr.DEVLOG_CONSENT_ENV, "true")
    with pytest.raises(HTTPException) as exc:
        _ruf(_body(consent=consent))
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "devlog_retention_consent_required"
    # Der Client muss erfahren, WAS er senden soll, nicht nur dass er darf.
    assert exc.value.detail["field"] == "retention_consent"
    assert _dateien(ablage) == []


def test_scharf_mit_zustimmung_geht_durch(ablage, monkeypatch):
    monkeypatch.setenv(rr.DEVLOG_CONSENT_ENV, "true")
    antwort = _ruf(_body(consent=True))
    assert antwort["accepted"] is True
    rec = json.loads(_dateien(ablage)[0].read_text())
    assert rec["retention_consent"] is True
    assert rec["consent_enforced"] is True
