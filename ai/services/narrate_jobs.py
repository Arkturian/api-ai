"""Auftragsbuch fuer /ai/tts/narrate: request_id, dauerhafter Status,
payloadgebundene Idempotenz.

Story-Codex (13.09., unter Alex' Audio-Auftrag): ein Aufrufer, der in
seinen Timeout laeuft, hatte bisher keinen Anker auf das Ergebnis — der
Server sprach zu Ende und speicherte, der Client wusste es nicht und
sprach neu: doppelt bezahlt. Vertrag in Content #4976, p-b12b0d4d6062.

Regeln, die hier gelten:
- Eine Kennung ist an ihren Inhalt gebunden (sha256 der kanonischen
  Nutzlast). Dieselbe Kennung mit anderem Inhalt ist ein Fehler, keine
  Ueberschreibung.
- `done` wird ohne Anbieteraufruf wiederholt (`replayed`).
- `running` wird nie ein zweites Mal gesprochen (409) — ausser der
  Eintrag ist verwaist (`stale`, > 10 min ohne Abschluss), dann meldet
  der Status das ausdruecklich, und ein neuer POST darf sprechen.
- `failed` vor dem Sprechen (`pre_tts`) darf wiederholt werden; `failed`
  nach dem Sprechen (`tts`/`save`) nicht automatisch — Zeichen koennen
  verbraucht sein, das entscheidet ein Mensch (409).
- Je Host eigener Bestand (Datei), nicht foederationsweit.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Kein fuehrender Punkt: die Kennung ist ein Dateiname in jobs_dir, und
# ".." oder ".versteckt" haben dort nichts verloren. Kein Trennzeichen
# ist erlaubt, also keine Pfadtraversierung — der Name-Vergleich unten
# haelt das zusaetzlich fest.
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_:-][A-Za-z0-9_.:-]{7,127}$")
STALE_AFTER_S = 600
TTL_DAYS = 30
_PRE_TTS_STATUS = {402, 422, 429}


def jobs_dir() -> Path:
    return Path(os.getenv("NARRATE_JOBS_DIR", "/var/lib/api-ai/narrate_jobs"))


def request_id_ok(request_id: str) -> bool:
    if not REQUEST_ID_RE.match(request_id or ""):
        return False
    return Path(request_id).name == request_id and ".." not in request_id


def payload_hash(req: Any) -> str:
    """Kanonische Form dessen, was das ERGEBNIS bestimmt — einschliesslich
    `save_options` (Speicherung, Sichtbarkeit): dieselbe Kennung mit
    `is_public: false` bekaeme sonst beim Replay die oeffentliche Datei
    des ersten Aufrufs, ein falsches Ergebnis, das richtig aussieht
    (Story-Codex/Story, Review q-d98ba93140f0). Nur `request_id` und
    `respeak_stale` bleiben draussen: sie sind Steuerung, nicht Inhalt."""
    d = req.model_dump() if hasattr(req, "model_dump") else dict(req)
    kern = {k: d.get(k) for k in ("text", "character", "context", "config", "collection_id", "link_id", "save_options")}
    roh = json.dumps(kern, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()


def dict_hash(d: dict, ohne=("request_id",)) -> str:
    """Hash einer beliebigen Nutzlast (gensfx, genmusic_eleven) — alles
    zaehlt, was das Ergebnis bestimmt; nur die Steuerfelder bleiben draussen."""
    k = {a: b for a, b in dict(d).items() if a not in set(ohne)}
    roh = json.dumps(k, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()


def _pfad(request_id: str) -> Path:
    return jobs_dir() / f"{request_id}.json"


class ClaimConflict(Exception):
    """Ein anderer Prozess haelt gerade die Sperre auf diese Kennung."""


@contextlib.contextmanager
def sperre(request_id: str, wartezeit_s: float = 2.0):
    """Kurze, prozessuebergreifende Sperre je Kennung (Datei per os.link —
    atomar auf POSIX, auch ueber Prozesse). Nur fuer Pruefung + Uebergang
    gehalten, Millisekunden. Wer sie nicht bekommt, bekommt ClaimConflict
    -> 409, nie einen zweiten Anbieteraufruf (Story-Codex, Restpunkt 1)."""
    jobs_dir().mkdir(parents=True, exist_ok=True)
    lock = jobs_dir() / f"{request_id}.lock"
    tmp = jobs_dir() / f".{request_id}.{os.getpid()}.{time.time_ns()}.locktmp"
    tmp.write_text(str(os.getpid()))
    frist = time.time() + wartezeit_s
    try:
        while True:
            try:
                os.link(tmp, lock)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 30:
                        lock.unlink(missing_ok=True)   # verwaiste Sperre (Prozess starb)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() > frist:
                    raise ClaimConflict(request_id)
                time.sleep(0.02)
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)


def _atomar_neu(request_id: str, d: dict) -> bool:
    """Datei mit INHALT atomar anlegen: tmp schreiben, dann os.link auf den
    Zielnamen (scheitert mit EEXIST, wenn es ihn gibt). Ein Leser sieht
    nie eine leere Datei — das war die Luecke von O_EXCL + spaeterem
    write (Story-Codex, Restpunkt 1)."""
    tmp = jobs_dir() / f".{request_id}.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    try:
        os.link(tmp, _pfad(request_id))
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def lesen(request_id: str) -> Optional[dict]:
    p = _pfad(request_id)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("state") == "running":
        try:
            alter = time.time() - datetime.fromisoformat(d["updated_at"]).timestamp()
        except Exception:
            alter = STALE_AFTER_S + 1
        d["stale"] = alter > STALE_AFTER_S
    return d


def schreiben(d: dict) -> None:
    jobs_dir().mkdir(parents=True, exist_ok=True)
    d["updated_at"] = datetime.now().isoformat()
    tmp = _pfad(d["request_id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_pfad(d["request_id"]))


def reservieren(request_id: str, hash_: str, kind: str = "narrate", extra: Optional[dict] = None) -> Optional[dict]:
    """Atomar: O_EXCL. Zwei gleichzeitige Aufrufe mit derselben Kennung —
    genau einer bekommt die Reservierung, der andere None und liest den
    Eintrag des Gewinners. `exists()+write()` liesse beide sprechen."""
    jobs_dir().mkdir(parents=True, exist_ok=True)
    d = {"request_id": request_id, "kind": kind, "state": "running", "payload_hash": hash_,
         "stage": "pre_tts", "created_at": datetime.now().isoformat(),
         "updated_at": datetime.now().isoformat()}
    if extra:
        d.update(extra)
    with sperre(request_id):
        return d if _atomar_neu(request_id, d) else None


def uebernehmen(request_id: str, hash_: str, kind: str, erlaubt) -> Optional[dict]:
    """Wiederanspruch (nach failed/pre_tts, failed/prepare oder ausdruecklich
    nach stale) — atomar unter der Sperre: nur wenn der Eintrag noch im
    erwarteten Zustand ist, wird er auf running gesetzt. Zwei gleichzeitige
    Wiederansprueche: genau einer gewinnt, der andere bekommt None."""
    with sperre(request_id):
        alt = lesen(request_id) or {}
        if not erlaubt(alt):
            return None
        d = {"request_id": request_id, "kind": kind, "state": "running", "payload_hash": hash_,
             "stage": "pre_tts", "created_at": alt.get("created_at") or datetime.now().isoformat(),
             "attempts": int(alt.get("attempts", 0)) + 1}
        schreiben(d)
        return d


def anlegen(request_id: str, hash_: str, kind: str = "narrate") -> dict:
    """Nur fuer Tests und Werkzeuge: ueberschreibt ohne Zustandspruefung.
    Im Antwortpfad `reservieren` (erster Anspruch) oder `uebernehmen`
    (Wiederanspruch) nehmen — beide sind atomar."""
    alt = lesen(request_id) or {}
    d = {"request_id": request_id, "kind": kind, "state": "running", "payload_hash": hash_,
         "stage": "pre_tts", "created_at": alt.get("created_at") or datetime.now().isoformat(),
         "attempts": int(alt.get("attempts", 0)) + 1}
    schreiben(d)
    return d


# Was die Stufen kosten koennen — damit "vor dem Sprechen" nicht als
# "kostenlos" gelesen wird (Story-Codex, Restpunkt 2).
STAGE_SEMANTICS = {
    "pre_tts": "vor jedem Modell- und Anbieteraufruf; Neuanlauf kostet nichts",
    "prepare": "dramaturgische Aufbereitung ueber /ai/chatgpt (codex-CLI, ChatGPT-Abo: keine gezaehlten Einheiten, Kontingent des Abos); Neuanlauf wiederholt diesen Aufruf, verbraucht keine ElevenLabs-Zeichen",
    "tts": "Anbieteraufruf laeuft; Zeichen koennen verbraucht sein -> kein automatischer Neuanlauf",
    "save": "Anbieteraufruf abgeschlossen, Speicherung lief; Zeichen verbraucht -> kein automatischer Neuanlauf",
    "done": "Ergebnis gespeichert",
}


def stufe(request_id: str, stage: str) -> None:
    d = lesen(request_id)
    if d:
        d["stage"] = stage
        schreiben(d)


def abschliessen(request_id: str, result: dict) -> None:
    d = lesen(request_id) or {"request_id": request_id}
    d.update({"state": "done", "result": result, "stage": "done"})
    d.pop("stale", None)
    schreiben(d)


def scheitern(request_id: str, status_code: Optional[int], detail: Any) -> str:
    """Ordnet den Fehler einer Stufe zu. Vor dem Sprechen (Sperren, 422)
    ist ein erneuter Versuch harmlos; danach koennen Zeichen weg sein."""
    d = lesen(request_id) or {"request_id": request_id}
    stage = d.get("stage", "pre_tts")
    if stage == "tts" and status_code in _PRE_TTS_STATUS:
        # Sperren (402/429) und 422 kommen VOR dem Anbieteraufruf
        failed_stage = "pre_tts"
    else:
        failed_stage = stage if stage != "done" else "save"
    d.update({"state": "failed", "failed_stage": failed_stage,
              "error": {"status_code": status_code, "detail": detail if isinstance(detail, (dict, str)) else str(detail)}})
    d.pop("stale", None)
    schreiben(d)
    return failed_stage


def aufraeumen() -> int:
    """Nach TTL_DAYS wird das ERGEBNIS entfernt, die Kennung bleibt als
    Grabstein (`tombstone: true`). Eine freigegebene Kennung koennte eine
    alte Wiederholung neu bezahlen lassen (Review q-d98ba93140f0, Punkt 1)."""
    grenze = time.time() - TTL_DAYS * 86400
    n = 0
    try:
        for p in jobs_dir().glob("*.json"):
            if p.stat().st_mtime < grenze:
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if d.get("tombstone"):
                    continue
                d.pop("result", None)
                d["tombstone"] = True
                d["tombstoned_at"] = datetime.now().isoformat()
                p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                n += 1
    except OSError:
        pass
    return n
