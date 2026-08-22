"""Aufbewahrungsfrist für Sprachmitschnitte (#1267, zweite Hälfte).

Der Einwilligungsriegel regelt, ob geschrieben werden darf. Diese Hälfte
regelt, wie lange das Geschriebene liegen bleibt — die Frage, die
CloudV2 wörtlich gestellt hat („falls dauerhaft: wie lange, und wo?")
und auf die die Antwort bis heute „unbegrenzt" lautete: 74 Sitzungen
vom 24. Juni bis 17. August, kein Aufräum-Job.

Auch hier ohne Zahl kein Verhalten. Die Frist ist Alex' Entscheidung;
hier steht nur die Mechanik.
"""

import json
import os
import time

import pytest

from ai.routes import realtime_routes as rr


@pytest.fixture
def ablage(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "DEVLOG_ROOT", str(tmp_path))
    monkeypatch.setattr(rr, "_devlog_letzter_kehrgang", 0.0)
    return tmp_path


def _lege_ab(wurzel, name, alter_tage, mit_feld=True):
    eimer = wurzel / "eimer"
    eimer.mkdir(exist_ok=True)
    p = eimer / f"{name}.json"
    empfangen = time.time() - alter_tage * 86400
    rec = {"voice_session_id": name, "lines": [{"text": "gesprochen"}]}
    if mit_feld:
        rec["received_at"] = empfangen
    p.write_text(json.dumps(rec))
    os.utime(p, (empfangen, empfangen))
    return p


# ───────────────────────────────────────────────── ohne Zahl kein Verfall

@pytest.mark.parametrize("wert", [None, "", "0", "-3", "keine Zahl"])
def test_ohne_brauchbare_zahl_wird_nichts_geloescht(ablage, monkeypatch, wert):
    """Auch Unsinn im Schalter darf nicht löschen.

    Ein `REALTIME_DEVLOG_RETENTION_DAYS=dreissig` das als 0 gelesen wird
    ist harmlos; eines, das als „alles weg" gelesen würde, wäre ein
    Datenverlust durch Tippfehler.
    """
    if wert is None:
        monkeypatch.delenv(rr.DEVLOG_RETENTION_ENV, raising=False)
    else:
        monkeypatch.setenv(rr.DEVLOG_RETENTION_ENV, wert)
    alt = _lege_ab(ablage, "uralt", 400)
    rr._devlog_kehre_gedrosselt()
    assert alt.exists()


@pytest.mark.parametrize("tage", [0, 0.0, -1, -365])
def test_kehrgang_selbst_loescht_ohne_frist_nichts(ablage, tage):
    """Die zweite Verteidigungslinie, direkt geprüft.

    Die Tests darüber gehen alle durch `_devlog_kehre_gedrosselt`, das
    schon bei `tage <= 0` aussteigt — die gleichlautende Prüfung IM
    Kehrgang war damit ungetestet. Ich habe sie zur Gegenprobe entfernt
    und die Suite blieb grün: ein Kriterium, das an eine Bedingung
    gekoppelt ist, die der Fall gar nicht erreicht.

    Was dabei auf dem Spiel steht: `_devlog_kehre_aus(0)` setzt die
    Grenze auf JETZT — jede Datei ist älter, also fällt der ganze
    Bestand. Ein Aufruf mit 0 muss deshalb nichts tun und nicht alles.
    """
    alt = _lege_ab(ablage, "bestand", 400)
    assert rr._devlog_kehre_aus(tage) == 0
    assert alt.exists()


# ──────────────────────────────────────────────────────── mit Zahl

def test_aelteres_faellt_neueres_bleibt(ablage, monkeypatch):
    monkeypatch.setenv(rr.DEVLOG_RETENTION_ENV, "30")
    alt = _lege_ab(ablage, "alt", 45)
    frisch = _lege_ab(ablage, "frisch", 5)
    assert rr._devlog_kehre_aus(30) == 1
    assert not alt.exists()
    assert frisch.exists()


def test_grenzfall_knapp_darunter_bleibt(ablage, monkeypatch):
    monkeypatch.setenv(rr.DEVLOG_RETENTION_ENV, "30")
    knapp = _lege_ab(ablage, "knapp", 29.9)
    assert rr._devlog_kehre_aus(30) == 0
    assert knapp.exists()


def test_alter_kommt_aus_dem_datensatz_nicht_aus_der_dateizeit(ablage):
    """Der Grund, warum `received_at` gewinnt.

    Eine Sicherung, ein `cp -r` oder ein Umzug setzt die Dateizeit auf
    jetzt. Wer danach die mtime befragt, verlängert die Frist
    stillschweigend — die Datei ist scheinbar frisch und bleibt ewig
    liegen. Hier ist die Datei per mtime taufrisch und laut Datensatz
    ein Jahr alt.
    """
    eimer = ablage / "eimer"
    eimer.mkdir()
    p = eimer / "verschoben.json"
    p.write_text(json.dumps({"received_at": time.time() - 365 * 86400}))
    os.utime(p, None)  # mtime = jetzt, wie nach einem Kopiervorgang
    assert rr._devlog_kehre_aus(30) == 1
    assert not p.exists()


def test_ohne_received_at_zieht_die_dateizeit_nach(ablage):
    """Notbehelf für Altbestand — die 74 vorhandenen tragen das Feld,
    aber ein handgeschriebener Datensatz muss trotzdem verfallen."""
    alt = _lege_ab(ablage, "ohnefeld", 100, mit_feld=False)
    assert rr._devlog_kehre_aus(30) == 1
    assert not alt.exists()


def test_kaputte_datei_stoppt_den_kehrgang_nicht(ablage):
    """Sonst rettet eine einzige unlesbare Datei den ganzen Rest vor
    dem Verfall — und niemand merkt es."""
    eimer = ablage / "eimer"
    eimer.mkdir()
    (eimer / "kaputt.json").write_text("{kein json")
    alt = _lege_ab(ablage, "alt", 90)
    assert rr._devlog_kehre_aus(30) == 1
    assert not alt.exists()
    assert (eimer / "kaputt.json").exists()   # bleibt liegen, sichtbar


# ────────────────────────────────────────────────────────── Drosselung

def test_zweiter_aufruf_kehrt_nicht_sofort_erneut(ablage, monkeypatch):
    monkeypatch.setenv(rr.DEVLOG_RETENTION_ENV, "30")
    rr._devlog_kehre_gedrosselt()
    _lege_ab(ablage, "danach", 90)
    rr._devlog_kehre_gedrosselt()          # zu früh — soll aussetzen
    assert (ablage / "eimer" / "danach.json").exists()


def test_nach_dem_intervall_kehrt_er_wieder(ablage, monkeypatch):
    monkeypatch.setenv(rr.DEVLOG_RETENTION_ENV, "30")
    rr._devlog_kehre_gedrosselt()
    alt = _lege_ab(ablage, "danach", 90)
    monkeypatch.setattr(
        rr, "_devlog_letzter_kehrgang",
        time.time() - rr._DEVLOG_SWEEP_INTERVAL_SEC - 1,
    )
    rr._devlog_kehre_gedrosselt()
    assert not alt.exists()
