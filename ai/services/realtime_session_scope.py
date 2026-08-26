"""Sitzungs-Scope fuer den Produktfinder — Marke, Jahr, Einstieg.

**Warum es diesen Speicher gibt.** OnealServs Suchroute erwartet
`brand` und `collection_year` von uns. Beide werden beim Mint an die
Sitzung gebunden — aber der Werkzeug-Dispatch kennt diesen Zustand
nicht: Er bekommt `X-Session-ID` und die Argumente des Modells, sonst
nichts. Es gibt genau drei Stellen, an denen der Scope herkommen
koennte, und zwei davon sind ausgeschlossen:

* **Der Client schickt ihn mit** — dann ist er browserseitig behauptet.
  Genau das schliesst die Regel aus, dass der oeffentliche Browser-Key
  keine Sitzungsidentitaet ist.
* **Das Modell setzt ihn** — ausgeschlossen: `brand` und
  `collection_year` wurden bewusst aus den Werkzeugkriterien entfernt,
  damit es sie nicht setzen KANN.
* **Der Server haelt ihn.** Bleibt uebrig, und ist richtig.

Bauform bewusst wie der Devlog-Speicher: eine JSON-Datei mit
`flock`, atomar ersetzt, mit Verfallszeit. Kein Redis — es gibt hier
keinen verantwortlichen Dienst mit Ausfallvertrag, und eine
Abhaengigkeit ohne Eigentuemer gehoert niemandem (Argument von
OnealServ-Codex, hier uebernommen).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCOPE_PATH = "/var/lib/api-ai/realtime_session_scope.json"

# Eine Sprachsitzung dauert Minuten, nicht Stunden. Laeuft der Scope
# ab, waehrend sie noch laeuft, faellt der naechste Werkzeugaufruf
# fail-closed aus — das ist unangenehm, aber ehrlich. Ein Scope ohne
# Verfall waere dagegen ein Speicher, der nur waechst.
SCOPE_TTL_SEC = 3600.0


def _laden() -> dict:
    try:
        with open(SCOPE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Sitzungs-Scope nicht lesbar (%s) — leer behandelt", exc)
        return {}


def _schreiben(daten: dict) -> None:
    os.makedirs(os.path.dirname(SCOPE_PATH), exist_ok=True)
    tmp = SCOPE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(daten, f, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp, SCOPE_PATH)


def merken(
    session_id: str,
    brand: Optional[str],
    collection_year: int,
    entry_selection: Optional[dict] = None,
) -> None:
    """Scope beim Mint ablegen. Ohne `session_id` passiert nichts.

    Abgelaufene Eintraege werden beim Schreiben mit ausgekehrt — der
    Speicher raeumt sich an der Stelle auf, an der er ohnehin
    angefasst wird, statt einen eigenen Aufraeum-Job zu brauchen.
    """
    if not session_id:
        logger.info("Sitzungs-Scope nicht abgelegt: keine session_id")
        return
    jetzt = time.time()
    daten = {
        k: v for k, v in _laden().items()
        if isinstance(v, dict) and (v.get("expires_at") or 0) > jetzt
    }
    daten[session_id] = {
        "brand": brand,
        "collection_year": collection_year,
        "entry_selection": entry_selection,
        "expires_at": jetzt + SCOPE_TTL_SEC,
    }
    _schreiben(daten)


def lesen(session_id: Optional[str]) -> Optional[dict]:
    """Scope einer Sitzung. `None` heisst: nicht bekannt oder abgelaufen.

    Der Aufrufer darf `None` NICHT als "markenoffen" lesen. Markenoffen
    ist ein Scope mit `brand=None`; unbekannt ist gar kein Scope. Die
    beiden zu verwechseln hiesse, eine gebundene Sitzung stillschweigend
    ueber den ganzen Katalog laufen zu lassen — dieselbe Klasse wie
    "Ausfall ist kein leeres Regal", nur eine Ebene hoeher.
    """
    if not session_id:
        return None
    eintrag = _laden().get(session_id)
    if not isinstance(eintrag, dict):
        return None
    if (eintrag.get("expires_at") or 0) <= time.time():
        return None
    return eintrag


def token_merken(session_id: Optional[str], token: Optional[str]) -> bool:
    """Das zuletzt ausgegebene Auswahl-Token an der Sitzung halten.

    Warum serverseitig: Das Token darf NICHT in den Modellkontext —
    oneal hat es deshalb aus dem Werkzeugergebnis entfernt (`0e3ea84`).
    Damit kann das Modell es bei einer Verfeinerung auch nicht
    mitschicken, und der Browser ergaenzt es nicht (am Client-Code
    geprueft). Also haelt es der Server, genau wie Marke und Jahr.

    Dieselbe Doktrin, ein Satz: **Was das Modell nicht wissen soll,
    haelt der Server.**

    Gibt zurueck, ob abgelegt wurde. `False` heisst: keine Sitzung
    bekannt oder kein Token dabei — beides normal (eine Suche ohne
    Treffer liefert keins) und deshalb kein Fehler.
    """
    if not session_id or not token:
        return False
    daten = _laden()
    eintrag = daten.get(session_id)
    if not isinstance(eintrag, dict):
        # Kein Scope, kein Token. Ein Token ohne Sitzung waere heimatlos
        # und wuerde beim naechsten Lesen niemandem gehoeren.
        return False
    if (eintrag.get("expires_at") or 0) <= time.time():
        return False
    eintrag["last_selection_token"] = token
    daten[session_id] = eintrag
    _schreiben(daten)
    return True


def token_lesen(session_id: Optional[str]) -> Optional[str]:
    """Das zuletzt gemerkte Auswahl-Token oder `None`.

    `None` heisst „es gab noch keine Suche in dieser Sitzung" — der
    Aufrufer muss daraus eine NEUE Suche machen, nicht eine
    Verfeinerung von nichts.
    """
    eintrag = lesen(session_id)
    if not eintrag:
        return None
    token = eintrag.get("last_selection_token")
    return token if isinstance(token, str) and token else None


def vergessen(session_id: str) -> bool:
    """Beim Sitzungsende aufraeumen."""
    daten = _laden()
    if session_id in daten:
        del daten[session_id]
        _schreiben(daten)
        return True
    return False
