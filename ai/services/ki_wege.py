"""Welche Modellwege hat diese Instanz? (#3639, Alex 21.11.)

Die Pipeline misst damit vor und nach jedem Lauf, ob jeder genannte Weg da ist.
Ohne Modellaufruf. Nicht im oeffentlichen /health, denn das wuerde zeigen,
welche Abos ein Kunde hat.

`zustand`: fehlt | nicht_angemeldet | angemeldet (vorhanden, unbewiesen) |
ok (letzter echter Aufruf gelang) | kontingent (zeitweise, `bis` falls bekannt)
| fehler.

agy und codex melden `bis` und `letzter_erfolg` noch nicht: Dafuer muessen ihre
Routen Ausgaenge buchen, das kommt mit den festen Fehlercodes `cli_fehlt` und
`kontingent` (#3639, Kommentar 11202).
"""
from __future__ import annotations

import os
import pwd
import shutil
from typing import Callable

from ai.services import cli_dienst, openai_text

_AGY_TOKEN = (".gemini", "antigravity-cli", "antigravity-oauth-token")


def _weg(zustand: str, **extra) -> dict:
    return {"zustand": zustand, "bis": None, "letzter_erfolg": None, **extra}


def _agy_home() -> str:
    user = (os.getenv("AGY_RUN_AS_USER") or "").strip()
    if user:
        return pwd.getpwnam(user).pw_dir
    return os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir


def agy() -> dict:
    if not shutil.which("agy") and not cli_dienst.aktiv():
        return _weg("fehlt")
    try:
        os.stat(os.path.join(_agy_home(), *_AGY_TOKEN))
    except FileNotFoundError:
        return _weg("nicht_angemeldet")
    except KeyError:
        return _weg("fehler", grund="agy_benutzer_fehlt")
    except PermissionError:
        # Fehlendes Leserecht sagt nichts ueber die Anmeldung.
        return _weg("fehler", grund="anmeldung_nicht_lesbar")
    except OSError:
        return _weg("fehler", grund="anmeldung_nicht_pruefbar")
    return _weg("angemeldet")


def codex(cli_status: Callable[[str], dict]) -> dict:
    s = cli_status("codex") or {}
    if not s.get("ok"):
        fehler = str(s.get("error") or "")
        if "not found" in fehler:
            return _weg("fehlt")
        return _weg("fehler", grund="cli_nicht_startbar")
    auth = s.get("auth") or {}
    if auth.get("ok") is True:
        return _weg("angemeldet")
    grund = auth.get("reason")
    if grund == "not_readable_by_service" or auth.get("ok") is None:
        return _weg("fehler", grund="anmeldung_nicht_lesbar")
    if grund in ("absent", "expired", "unreadable"):
        return _weg("nicht_angemeldet", grund=grund)
    return _weg("fehler", grund=str(grund or "unbekannt")[:40])


def wege(cli_status: Callable[[str], dict]) -> dict:
    return {"wege": {"agy": agy(), "codex": codex(cli_status),
                     "openai_schluessel": openai_text.weg_zustand()}}
