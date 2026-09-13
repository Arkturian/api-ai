"""
ElevenLabs Cost Tracker
=======================

Zaehlt, was ElevenLabs abrechnet, und sperrt VOR dem Sprechen.

Anlass (Story, 13.09.2026): Die Story-Engine soll Ton ableiten —
Erzaehler, Figurenstimmen, Atmosphaere, Geraeusche, Musik. Alle vier
laufen bei uns ueber ElevenLabs (TTS in tts_service/narration_service,
Geraeusche in gensfx + AudioDramaGenerator, Musik in
generate_music_elevenlabs), und keiner davon stand in einem Zaehler.
10 000 Zeichen inklusive, danach ``payg`` — und bei uns bremste nichts.
Das ist die Konstellation vom Mai (209,56 EUR, anderer Anbieter).

Was hier bewusst anders ist als bei den anderen Trackern:

* **Einheit ist das Zeichen, nicht der Euro.** ElevenLabs rechnet TTS in
  Zeichen ab; die eigene Abrechnung (``/v1/user/subscription``) fuehrt
  ``character_count``/``character_limit``. Einen Euro-Preis je Zeichen
  erfinde ich nicht — er wird nur gerechnet, wenn
  ``ELEVENLABS_PRICE_PER_1K_CHARS_USD`` gesetzt ist.
* **Zwei Sperren, zwei Codes.** 429 ``monthly_api_cap_reached``: unser
  Monatsdeckel in Zeichen (``ELEVENLABS_MONTHLY_CHAR_CAP``, Vorgabe
  10 000 = das Inklusivkontingent). 402 ``elevenlabs_quota_exhausted``:
  ElevenLabs' eigenes Kontingent ist laut Abrechnung aufgebraucht —
  auf ``payg`` wuerde der naechste Aufruf trotzdem sprechen und
  abrechnen; genau das wird hier gestoppt, solange
  ``ELEVENLABS_BLOCK_BEYOND_INCLUDED`` (Vorgabe true) steht.
* **Geraeusche und Musik zaehlen sichtbar, aber nicht im Zeichendeckel**,
  weil ich ihre Abrechnungseinheit nicht belegen kann. Sie stehen als
  eigene Zaehler (Aufrufe, angeforderte Sekunden) im Status; der
  Hard-Cap und die Kontingentpruefung greifen auch fuer sie.

Master/Client wie die Geschwister: arkserver ist Master, arkturian
meldet ueber ``/internal/elevenlabs-cost-shared-state`` mit demselben
``COST_TRACKER_SHARED_SECRET``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

EUR_USD_RATE = 1.05
_SUBSCRIPTION_CACHE_S = 60.0


class ElevenLabsCostTracker:
    _instance: Optional["ElevenLabsCostTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ElevenLabsCostTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.monthly_char_cap = int(os.getenv("ELEVENLABS_MONTHLY_CHAR_CAP", "10000"))
        self.price_per_1k_chars_usd = float(os.getenv("ELEVENLABS_PRICE_PER_1K_CHARS_USD", "0") or 0)
        self.block_beyond_included = (
            os.getenv("ELEVENLABS_BLOCK_BEYOND_INCLUDED", "true").lower() == "true"
        )
        self.block_on_cap = os.getenv("ELEVENLABS_BLOCK_ON_CAP", "true").lower() == "true"
        self.data_dir = Path(os.getenv("ELEVENLABS_COST_TRACKER_DATA_DIR", "/var/lib/api-ai"))
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.alert_thresholds = [80, 95, 100]
        self.master_url = os.getenv("ELEVENLABS_COST_TRACKER_MASTER_URL", "").rstrip("/")
        self.shared_secret = os.getenv("COST_TRACKER_SHARED_SECRET", "")

        self._usage_data: dict = {}
        self._alerts_sent: set = set()
        self._data_lock = threading.Lock()
        self._master_status_cache: Optional[dict] = None
        self._master_status_cache_ts: float = 0.0
        self._subscription_cache: Optional[dict] = None
        self._subscription_cache_ts: float = 0.0
        self._load_data()
        logger.info(
            "ElevenLabsCostTracker: cap=%d Zeichen/Monat, Preis je 1k=%s, mode=%s",
            self.monthly_char_cap,
            self.price_per_1k_chars_usd or "nicht gesetzt",
            "client" if self.master_url else "master",
        )

    # ── Ablage ───────────────────────────────────────────────────────

    @property
    def data_file(self) -> Path:
        return self.data_dir / f"elevenlabs_usage_{datetime.now().strftime('%Y-%m')}.json"

    def _reset_monthly_data(self) -> None:
        self._usage_data = {
            "month": datetime.now().strftime("%Y-%m"),
            "chars_total": 0,
            "tts_calls": 0,
            "audio_seconds_total": 0.0,
            "sfx_calls": 0,
            "sfx_seconds_requested": 0.0,
            "music_calls": 0,
            "music_seconds_requested": 0.0,
            "total_cost_usd": 0.0,
            "total_cost_eur": 0.0,
            "by_caller": {},
            "alerts_sent": [],
            "created_at": datetime.now().isoformat(),
        }
        self._alerts_sent = set()

    def _load_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    self._usage_data = json.load(f)
                self._alerts_sent = set(self._usage_data.get("alerts_sent", []))
            else:
                self._reset_monthly_data()
        except Exception as e:
            logger.error("elevenlabs_cost_tracker: Laden fehlgeschlagen: %s", e)
            self._reset_monthly_data()

    def _save_data(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._usage_data["alerts_sent"] = list(self._alerts_sent)
            self._usage_data["last_updated"] = datetime.now().isoformat()
            with open(self.data_file, "w") as f:
                json.dump(self._usage_data, f, indent=2)
            self._usage_data["_file_mtime"] = self.data_file.stat().st_mtime
        except Exception as e:
            logger.error("elevenlabs_cost_tracker: Speichern fehlgeschlagen: %s", e)

    def _maybe_reload_from_file(self) -> None:
        try:
            f = self.data_file
            if not f.exists():
                return
            mtime = f.stat().st_mtime
            if mtime > float(self._usage_data.get("_file_mtime", 0)) + 0.001:
                with open(f, "r") as fh:
                    fresh = json.load(fh)
                fresh["_file_mtime"] = mtime
                self._usage_data = fresh
                self._alerts_sent = set(fresh.get("alerts_sent", []))
        except Exception as e:
            logger.debug("elevenlabs_cost_tracker: reload uebersprungen (%s)", e)

    def _ensure_month(self) -> None:
        if self._usage_data.get("month") != datetime.now().strftime("%Y-%m"):
            self._reset_monthly_data()

    # ── Zaehlen ──────────────────────────────────────────────────────

    def track_tts(self, chars: int, caller: str = "tts", audio_seconds: Optional[float] = None) -> None:
        self._track("tts", caller, chars=int(chars), audio_seconds=float(audio_seconds or 0))

    def track_audio_seconds(self, seconds: float, caller: str = "narrate") -> None:
        """Nachbuchung der GEMESSENEN Dauer: die Zeichen werden in tts_service
        beim Empfang der Bytes gebucht, die Dauer kennt erst narrate nach der
        Messung. Story (13.09.): eine Null nach 18,5 s echter Sprache ist eine
        Null, die nicht stimmt. Zaehlt keinen Aufruf, nur Sekunden."""
        if not seconds or seconds <= 0:
            return
        self._track("seconds", caller, audio_seconds=float(seconds))

    def track_sfx(self, caller: str = "gensfx", seconds_requested: Optional[float] = None) -> None:
        self._track("sfx", caller, seconds_requested=float(seconds_requested or 0))

    def track_music(self, caller: str = "genmusic_eleven", seconds_requested: Optional[float] = None) -> None:
        self._track("music", caller, seconds_requested=float(seconds_requested or 0))

    def _track(self, modality: str, caller: str, **units) -> None:
        if self.master_url and self.shared_secret:
            try:
                self._post_to_master(modality, caller, units)
                return
            except Exception as e:
                logger.error("elevenlabs_cost_tracker: Master nicht erreichbar (%s); zaehle lokal", e)
        self._track_local(modality, caller, **units)

    def _track_local(self, modality: str, caller: str, **units) -> None:
        with self._data_lock:
            self._maybe_reload_from_file()
            self._ensure_month()
            d = self._usage_data
            if modality == "tts":
                chars = int(units.get("chars", 0))
                d["chars_total"] = int(d.get("chars_total", 0)) + chars
                d["tts_calls"] = int(d.get("tts_calls", 0)) + 1
                d["audio_seconds_total"] = float(d.get("audio_seconds_total", 0)) + float(units.get("audio_seconds", 0))
                if self.price_per_1k_chars_usd:
                    usd = self.price_per_1k_chars_usd * chars / 1000.0
                    d["total_cost_usd"] = float(d.get("total_cost_usd", 0)) + usd
                    d["total_cost_eur"] = float(d.get("total_cost_eur", 0)) + usd / EUR_USD_RATE
            elif modality == "seconds":
                d["audio_seconds_total"] = float(d.get("audio_seconds_total", 0)) + float(units.get("audio_seconds", 0))
            elif modality == "sfx":
                d["sfx_calls"] = int(d.get("sfx_calls", 0)) + 1
                d["sfx_seconds_requested"] = float(d.get("sfx_seconds_requested", 0)) + float(units.get("seconds_requested", 0))
            elif modality == "music":
                d["music_calls"] = int(d.get("music_calls", 0)) + 1
                d["music_seconds_requested"] = float(d.get("music_seconds_requested", 0)) + float(units.get("seconds_requested", 0))
            bc = d.setdefault("by_caller", {}).setdefault(caller, {"calls": 0, "chars": 0})
            if modality != "seconds":
                bc["calls"] += 1
                bc["chars"] += int(units.get("chars", 0))
            bc["audio_seconds"] = round(float(bc.get("audio_seconds", 0)) + float(units.get("audio_seconds", 0)), 3)
            self._save_data()
            self._check_thresholds()
        logger.info("ElevenLabs gezaehlt: %s/%s %s (Monat: %d Zeichen)", modality, caller, units, self._usage_data.get("chars_total", 0))

    # ── Sperren ──────────────────────────────────────────────────────

    def chars_used(self) -> int:
        self._maybe_reload_from_file()
        return int(self._usage_data.get("chars_total", 0))

    def is_cap_exceeded(self, chars_planned: int = 0) -> bool:
        return self.monthly_char_cap > 0 and (self.chars_used() + max(0, chars_planned)) > self.monthly_char_cap

    def subscription_snapshot(self, force: bool = False) -> Optional[dict]:
        """ElevenLabs' eigene Abrechnung, 60 s zwischengespeichert. None,
        wenn nicht erreichbar — dann entscheidet nur der lokale Deckel."""
        now = time.time()
        if not force and self._subscription_cache is not None and now - self._subscription_cache_ts < _SUBSCRIPTION_CACHE_S:
            return self._subscription_cache
        key = (os.getenv("ELEVENLABS_API_KEY") or "").strip('"').strip("'")
        if not key:
            return None
        try:
            with httpx.Client(timeout=6.0) as client:
                r = client.get("https://api.elevenlabs.io/v1/user/subscription", headers={"xi-api-key": key})
            if r.status_code != 200:
                return None
            data = r.json()
            snap = {
                "tier": data.get("tier"),
                "character_count": int(data.get("character_count") or 0),
                "character_limit": int(data.get("character_limit") or 0),
                "next_reset_unix": data.get("next_character_count_reset_unix"),
                "fetched_at": datetime.now().isoformat(),
            }
            snap["remaining"] = max(0, snap["character_limit"] - snap["character_count"])
            self._subscription_cache, self._subscription_cache_ts = snap, now
            return snap
        except Exception as e:
            logger.warning("elevenlabs_cost_tracker: Abrechnung nicht lesbar (%s)", e)
            return None

    def pre_check(self, chars_planned: int = 0, endpoint: str = "elevenlabs") -> None:
        """VOR dem Aufruf: wirft 429 (unser Deckel) oder 402 (Kontingent
        beim Anbieter aufgebraucht). Wirft nichts, wenn beides frei ist."""
        self._maybe_reload_from_file()
        if self._usage_data.get("hard_cap_active"):
            raise HTTPException(status_code=429, detail={
                "error": "monthly_api_cap_reached", "provider": "elevenlabs", "endpoint": endpoint,
                "reason": self._usage_data.get("hard_cap_reason") or "hard cap",
            })
        if self.master_url and self.shared_secret:
            try:
                st = self._fetch_master_status()
                used = int(st.get("chars_used", 0))
                cap = int(st.get("monthly_char_cap", self.monthly_char_cap))
                if self.block_on_cap and cap > 0 and used + chars_planned > cap:
                    raise HTTPException(status_code=429, detail={
                        "error": "monthly_api_cap_reached", "provider": "elevenlabs", "endpoint": endpoint,
                        "chars_used": used, "chars_planned": chars_planned, "monthly_char_cap": cap,
                    })
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("elevenlabs_cost_tracker: Master nicht erreichbar (%s); lokale Sicht", e)
                self._pre_check_local(chars_planned, endpoint)
        else:
            self._pre_check_local(chars_planned, endpoint)
        if self.block_beyond_included:
            snap = self.subscription_snapshot()
            if snap and snap["character_limit"] > 0 and snap["remaining"] < max(1, chars_planned):
                raise HTTPException(status_code=402, detail={
                    "error": "elevenlabs_quota_exhausted", "endpoint": endpoint,
                    "hint": ("ElevenLabs meldet das Inklusivkontingent als aufgebraucht; auf "
                             "payg wuerde der Aufruf trotzdem sprechen und abrechnen. "
                             "ELEVENLABS_BLOCK_BEYOND_INCLUDED=false laesst das bewusst zu."),
                    "character_count": snap["character_count"], "character_limit": snap["character_limit"],
                    "chars_planned": chars_planned, "tier": snap["tier"],
                })

    def _pre_check_local(self, chars_planned: int, endpoint: str) -> None:
        if self.block_on_cap and self.is_cap_exceeded(chars_planned):
            raise HTTPException(status_code=429, detail={
                "error": "monthly_api_cap_reached", "provider": "elevenlabs", "endpoint": endpoint,
                "chars_used": self.chars_used(), "chars_planned": chars_planned,
                "monthly_char_cap": self.monthly_char_cap,
            })

    # ── Status / Kill-Switch / Alarme ────────────────────────────────

    def get_status(self) -> dict:
        """Im Client-Modus (arkturian) die Sicht des Masters — die lokale
        Datei bleibt dort bei null, weil jeder Aufruf an den Master
        gemeldet wird. Story-Codex las am 13.09. am oeffentlichen Ziel
        chars_used=0 nach vier echten Aufrufen; der Master hatte 276.
        Ein Status, der den falschen Zaehler zeigt, ist schlimmer als keiner."""
        if self.master_url and self.shared_secret:
            try:
                st = dict(self._fetch_master_status())
                st["view"] = "master"
                st["master_url"] = self.master_url
                return st
            except Exception as e:
                logger.warning("elevenlabs_cost_tracker: Master-Status nicht lesbar (%s); lokale Sicht", e)
        st = self._local_status()
        st["view"] = "local"
        return st

    def _local_status(self) -> dict:
        self._maybe_reload_from_file()
        with self._data_lock:
            d = self._usage_data
            used = int(d.get("chars_total", 0))
            cap = self.monthly_char_cap
            return {
                "provider": "elevenlabs",
                "unit": "characters",
                # Gemessen 12./13.09.: unsere Zeichen (Laenge des gesendeten
                # Texts) und ElevenLabs' `character_count` laufen nicht 1:1 —
                # 177 -> +48, 276 -> +76, beide Male ~0,27. Was der Anbieter
                # zaehlt, ist nicht belegt; unser Deckel ist damit konservativ.
                # Nur Messwerte, keine Verallgemeinerung (Story-Codex, 13.09.):
                # aus zwei Serien folgt keine Garantie, dass der interne
                # Deckel stets frueher sperrt.
                "note": ("chars_used = intern gezaehlte Textzeichen (Laenge des gesendeten Texts). "
                         "ElevenLabs' subscription.character_count ist eine andere Zaehlweise: gemessen "
                         "12./13.09. 177 intern -> +48 beim Anbieter, 276 intern -> +76. Beide Zahlen "
                         "getrennt lesen; die Anbieterzaehlweise ist nicht geklaert."),
                "audio_seconds_note": ("Summe der in narrate GEMESSENEN Dauern (Nachbuchung nach der Messung); "
                                       "Pfade ohne Messung (Hoerspiel, gensfx) tragen hier nichts bei."),
                "month": d.get("month"),
                "chars_used": used,
                "monthly_char_cap": cap,
                "usage_percentage": round(used / cap * 100, 2) if cap > 0 else 0.0,
                "chars_remaining": max(0, cap - used) if cap > 0 else None,
                "tts_calls": d.get("tts_calls", 0),
                "audio_seconds_total": round(float(d.get("audio_seconds_total", 0)), 3),
                "sfx_calls": d.get("sfx_calls", 0),
                "sfx_seconds_requested": round(float(d.get("sfx_seconds_requested", 0)), 1),
                "music_calls": d.get("music_calls", 0),
                "music_seconds_requested": round(float(d.get("music_seconds_requested", 0)), 1),
                "sfx_music_in_char_cap": False,
                "price_per_1k_chars_usd": self.price_per_1k_chars_usd or None,
                "total_cost_eur": round(float(d.get("total_cost_eur", 0)), 4) if self.price_per_1k_chars_usd else None,
                "by_caller": d.get("by_caller", {}),
                "cap_exceeded": cap > 0 and used >= cap,
                "hard_cap_active": bool(d.get("hard_cap_active")),
                "hard_cap_reason": d.get("hard_cap_reason", ""),
                "block_beyond_included": self.block_beyond_included,
                "alerts_sent": list(self._alerts_sent),
                "last_updated": d.get("last_updated"),
            }

    def trip_hard_cap(self, reason: str = "") -> None:
        with self._data_lock:
            self._usage_data["hard_cap_active"] = True
            self._usage_data["hard_cap_reason"] = reason
            self._save_data()
        logger.warning("elevenlabs_cost_tracker: hard cap AKTIV (%s)", reason or "-")

    def clear_hard_cap(self) -> dict:
        with self._data_lock:
            was = bool(self._usage_data.get("hard_cap_active"))
            self._usage_data["hard_cap_active"] = False
            self._usage_data["hard_cap_reason"] = ""
            self._save_data()
        return {"was_active": was, "cleared": True}

    def _check_thresholds(self) -> None:
        if self.monthly_char_cap <= 0:
            return
        pct = int(self._usage_data.get("chars_total", 0)) / self.monthly_char_cap * 100
        for th in self.alert_thresholds:
            if pct >= th and th not in self._alerts_sent:
                self._send_telegram_alert(th)
                self._alerts_sent.add(th)
                self._usage_data["alerts_sent"] = list(self._alerts_sent)

    def _send_telegram_alert(self, threshold: int) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        d = self._usage_data
        emoji = "🚨" if threshold >= 100 else ("⚠️" if threshold >= 95 else "📊")
        msg = (
            f"{emoji} <b>ElevenLabs {threshold}% des Zeichen-Deckels</b>\n\n"
            f"<b>Zeichen:</b> {int(d.get('chars_total', 0)):,} / {self.monthly_char_cap:,}\n"
            f"<b>TTS-Aufrufe:</b> {d.get('tts_calls', 0)} · <b>SFX:</b> {d.get('sfx_calls', 0)} · <b>Musik:</b> {d.get('music_calls', 0)}\n"
            f"<b>Status:</b> {'⛔ GESPERRT' if threshold >= 100 else '✅ aktiv'}"
        )
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage",
                    json={"chat_id": self.telegram_chat_id, "text": msg, "parse_mode": "HTML"},
                )
        except Exception as e:
            logger.error("Telegram-Alarm fehlgeschlagen: %s", e)

    # ── Master/Client ────────────────────────────────────────────────

    def _post_to_master(self, modality: str, caller: str, units: dict) -> None:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                f"{self.master_url}/internal/elevenlabs-cost-shared-state",
                json={"modality": modality, "caller": caller, "units": units,
                      "source_host": os.environ.get("API_AI_HOST_KEY") or os.uname().nodename.split(".")[0]},
                headers={"X-Internal-Auth": self.shared_secret},
            )
            r.raise_for_status()
            try:
                self._master_status_cache, self._master_status_cache_ts = r.json(), time.time()
            except Exception:
                pass

    def _fetch_master_status(self) -> dict:
        now = time.time()
        if self._master_status_cache is not None and now - self._master_status_cache_ts < 10.0:
            return self._master_status_cache
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{self.master_url}/internal/elevenlabs-cost-shared-state",
                           headers={"X-Internal-Auth": self.shared_secret})
            r.raise_for_status()
            data = r.json()
        self._master_status_cache, self._master_status_cache_ts = data, now
        return data


elevenlabs_cost_tracker = ElevenLabsCostTracker()
