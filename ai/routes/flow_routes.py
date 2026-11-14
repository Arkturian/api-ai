"""
Google-Flow-Routen — Veo/Omni-Videos ueber useapi.net (Alex, 14.11.2026).

POST /ai/genvideo/flow               — Auftrag abgeben (Text oder Startbild)
GET  /ai/genvideo/flow/status/{id}   — Stand abfragen; fertige Clips landen
                                       einmal im Storage und werden einmal gebucht
GET  /ai/genvideo/flow/account       — verknuepfte Konten, Guthaben, Captchas

Hintergrund: Alex hat ein Google-Ultra-Abo. Flow hat keine offizielle API;
useapi.net haelt eine Anmeldung in seinem Google-Konto und reicht Auftraege
an Flow weiter. Gerechnet wird in Flow-Credits des Ultra-Kontos, dazu je
Erzeugung eine Captcha-Loesung (300 frei, danach ein bezahlter Anbieter).
Deshalb dieselbe Bremse wie bei allen Medienpfaden: `confirm_api_billing`.
Den Euro-Deckel der API-Pfade beruehrt das nicht, es ist ein eigener Topf.

Upstream: https://api.useapi.net/v1/google-flow (Bearer USEAPI_TOKEN).
Die Ergebnis-Links von Google verfallen nach ~6 h, darum der Storage.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

FLOW_BASE = os.getenv("USEAPI_FLOW_BASE", "https://api.useapi.net/v1/google-flow")

# Modelle und erlaubte Dauern laut useapi-Doku (14.11.2026). Flow nimmt
# falsche Werte mit 400 zurueck, aber erst nach der Captcha-Loesung — also
# hier pruefen, bevor etwas verbraucht wird.
VEO_DAUERN = (4, 6, 8)
MODELLE: Dict[str, tuple] = {
    "veo-3.1-fast": VEO_DAUERN,
    "veo-3.1-quality": (8,),
    "veo-3.1-lite": VEO_DAUERN,
    "veo-3.1-lite-low-priority": VEO_DAUERN,
    "omni-flash": (4, 6, 8, 10),
    "omni-flash-1.2": tuple(range(3, 16)),
}
VORGABE_MODELL = os.getenv("FLOW_DEFAULT_MODEL", "veo-3.1-fast")

# Credits je Clip mit Ultra (Doku 14.11.2026). Nur Schaetzung fuer
# max_credits und die Antwort; die Wahrheit ist `remainingCredits`.
_CREDITS_FEST = {"veo-3.1-fast": 10, "veo-3.1-quality": 100, "veo-3.1-lite": 5,
                 "veo-3.1-lite-low-priority": 0}
_OMNI_FLASH_720 = {4: 7, 6: 10, 8: 12, 10: 15}
_OMNI_FLASH_360 = {4: 4, 6: 5, 8: 6, 10: 7}


def credits_schaetzung(modell: str, dauer: int, aufloesung: str, anzahl: int) -> Optional[int]:
    if modell in _CREDITS_FEST:
        je = _CREDITS_FEST[modell]
    elif modell == "omni-flash":
        je = (_OMNI_FLASH_360 if aufloesung == "360p" else _OMNI_FLASH_720).get(dauer)
    elif modell == "omni-flash-1.2":
        basis = 1 if aufloesung == "360p" else 3
        je = basis * min(dauer, 10) + 2 * basis * max(0, dauer - 10)
    else:
        je = None
    return None if je is None else je * anzahl


_SEITEN = {"landscape": "landscape", "16:9": "landscape", "quer": "landscape",
           "portrait": "portrait", "9:16": "portrait", "hoch": "portrait"}


def _token() -> str:
    t = os.getenv("USEAPI_TOKEN", "")
    if not t:
        raise HTTPException(status_code=503, detail={
            "error": "flow_not_configured",
            "hint": "USEAPI_TOKEN ist auf diesem Host nicht gesetzt (/etc/api-ai/useapi.env).",
        })
    return t


def _client(timeout: float = 60.0) -> httpx.AsyncClient:
    """Eigene Fabrik, damit Tests den Transport tauschen koennen."""
    return httpx.AsyncClient(timeout=timeout)


def _fehler_text(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text[:300]


async def _flow(method: str, pfad: str, *, json_body: Optional[dict] = None,
                inhalt: Optional[bytes] = None, content_type: Optional[str] = None,
                timeout: float = 60.0) -> Dict[str, Any]:
    kopf = {"Authorization": f"Bearer {_token()}"}
    if content_type:
        kopf["Content-Type"] = content_type
    async with _client(timeout=timeout) as c:
        r = await c.request(method, f"{FLOW_BASE}{pfad}", json=json_body, content=inhalt, headers=kopf)
    if r.status_code >= 400:
        detail = {"error": "flow_error", "flow_status": r.status_code, "flow": _fehler_text(r)}
        if r.status_code == 429 and r.headers.get("retry-after"):
            detail["retry_after_s"] = r.headers.get("retry-after")
        if r.status_code == 596:
            detail["hint"] = ("Die Google-Anmeldung bei useapi ist abgelaufen. Alex muss das Konto "
                              "einmal neu verknuepfen: https://useapi.net/assets/setup-browser/google-flow.html")
        # 4xx von Flow reichen wir als solche durch (der Aufrufer kann sie
        # beheben), alles andere ist ein Upstream-Fehler.
        status = r.status_code if r.status_code in (400, 402, 403, 404, 408, 429) else 502
        raise HTTPException(status_code=status, detail=detail)
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail={"error": "flow_non_json", "text": r.text[:200]})


# --- Auftragsablage ---------------------------------------------------------

def _ablage() -> Path:
    return Path(os.getenv("FLOW_JOBS_DIR") or "/var/lib/api-ai/flow_jobs")


_JOB_ID = re.compile(r"^[A-Za-z0-9:._@%-]{1,200}$")


def _job_pfad(job_id: str) -> Path:
    if not _JOB_ID.match(job_id or ""):
        raise HTTPException(status_code=422, detail={"error": "invalid_job_id"})
    sicher = job_id.replace("/", "_").replace(":", "_")
    return _ablage() / f"{sicher}.json"


@asynccontextmanager
async def _gesperrt(job_id: str, frist_s: float = 300.0):
    """Sperre je Auftrag ueber Worker hinweg. Nicht blockierend versucht:
    ein blockierendes flock im selben Prozess hielte die Ereignisschleife an,
    waehrend der Halter gerade hochlaedt — Verklemmung."""
    d = _ablage()
    d.mkdir(parents=True, exist_ok=True)
    ende = time.monotonic() + frist_s
    with open(_job_pfad(job_id).with_suffix(".lock"), "a") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > ende:
                    raise HTTPException(status_code=409, detail={"error": "flow_job_busy",
                                                                 "hint": "Wird gerade gesichert; gleich erneut abfragen."})
                await asyncio.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _lesen(job_id: str) -> Dict[str, Any]:
    p = _job_pfad(job_id)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _schreiben(job_id: str, daten: Dict[str, Any]) -> None:
    p = _job_pfad(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


# --- Startbilder -------------------------------------------------------------

def _bild_adresse(wert: str) -> str:
    """Storage-ID oder Adresse -> Adresse."""
    w = str(wert).strip()
    if w.isdigit():
        basis = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com").rstrip("/")
        return f"{basis}/storage/media/{int(w)}?variant=full"
    return w


async def _bild_hochladen(wert: str) -> str:
    """Bild holen (SSRF-geprueft, Storage-Schluessel nur fuer eigene Links) und
    bei useapi als Asset ablegen. Liefert die mediaGenerationId."""
    from ai.clients import storage_abruf as sa
    url = _bild_adresse(wert)
    grenze = 20 * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as c:
            async with sa.oeffne(c, url, mit_schluessel=True) as r:
                typ = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                buf = bytearray()
                async for teil in r.aiter_bytes(65536):
                    buf.extend(teil)
                    if len(buf) > grenze:
                        raise HTTPException(status_code=422, detail={"error": "image_too_large",
                                                                     "max_mb": 20})
    except HTTPException:
        raise
    except sa.PrivateDatei:
        raise HTTPException(status_code=403, detail=sa.private_media_detail(url))
    except sa.GesperrteDatei:
        raise HTTPException(status_code=451, detail=sa.gesperrt_detail(url))
    except sa.ZielGesperrt as e:
        raise HTTPException(status_code=422, detail=sa.ziel_gesperrt_detail(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": "image_fetch_failed",
                                                     "url": sa.ohne_anfrage(url), "reason": type(e).__name__})
    if typ not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(status_code=422, detail={"error": "image_type_not_supported", "got": typ,
                                                     "allowed": ["image/png", "image/jpeg", "image/webp"]})
    antwort = await _flow("POST", "/assets", inhalt=bytes(buf), content_type=typ)
    mid = (antwort.get("mediaGenerationId") or {}).get("mediaGenerationId")
    if not mid:
        raise HTTPException(status_code=502, detail={"error": "flow_asset_without_id"})
    return mid


# --- Ergebnis sichern --------------------------------------------------------

async def _video_sichern(url: str, collection_id: Optional[str], link_id: Optional[str],
                         name: str) -> Dict[str, Any]:
    return await _datei_sichern(url, None, "video/mp4", collection_id, link_id, name)


async def _datei_sichern(url: Optional[str], inhalt: Optional[bytes], mime: str,
                         collection_id: Optional[str], link_id: Optional[str], name: str,
                         privat: bool = False) -> Dict[str, Any]:
    """Ergebnis von Google (Link oder Bytes) in den Storage legen."""
    from ai.clients.storage_client import storage_api_key
    storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    grenze = int(os.getenv("FLOW_MAX_VIDEO_MB", "500")) * 1024 * 1024
    try:
        async with _client(timeout=300.0) as c:
            if inhalt is not None:
                buf = bytearray(inhalt)
            else:
                async with c.stream("GET", url, follow_redirects=True) as r:
                    r.raise_for_status()
                    buf = bytearray()
                    async for teil in r.aiter_bytes(65536):
                        buf.extend(teil)
                        if len(buf) > grenze:
                            raise ValueError("file exceeds size cap")
                    mime = (r.headers.get("content-type") or mime).split(";")[0].strip() or mime
            daten = {"is_public": "false" if privat else "true", "analyze": "false", "ai_mode": "none",
                     "reuse_existing": "false"}
            if privat:
                daten["private"] = "true"
            if collection_id:
                daten["collection_id"] = collection_id
            if link_id:
                daten["link_id"] = link_id
            up = await c.post(f"{storage_url}/storage/upload",
                              files={"file": (name, bytes(buf), mime)},
                              data=daten, headers={"X-API-KEY": storage_api_key()})
            up.raise_for_status()
            sid = up.json().get("id")
            return {"storage_object_id": sid,
                    "storage_url": f"https://api-storage.arkturian.com/storage/media/{sid}" if sid else None}
    except Exception as e:
        logger.warning("flow: Sichern fehlgeschlagen: %s", type(e).__name__)
        return {"storage_object_id": None, "save_error": type(e).__name__}


def _clips(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(((job.get("response") or {}).get("media")) or [])


async def _abschliessen(job_id: str, job: Dict[str, Any], *, sichern: bool,
                        collection_id: Optional[str], link_id: Optional[str]) -> Dict[str, Any]:
    """Fertigen Auftrag einmal sichern und einmal buchen. Mehrfaches Abfragen
    (MCP-Agenten pollen) darf weder doppelt hochladen noch doppelt zaehlen."""
    async with _gesperrt(job_id):
        akte = _lesen(job_id)
        gesichert: Dict[str, Any] = dict(akte.get("gesichert") or {})
        offen = False
        for i, m in enumerate(_clips(job)):
            mid = m.get("mediaGenerationId") or f"clip{i}"
            if mid in gesichert and gesichert[mid].get("storage_object_id"):
                continue
            if not sichern:
                continue
            if not m.get("videoUrl"):
                offen = True  # Google liefert den Link manchmal spaeter nach
                continue
            gesichert[mid] = await _video_sichern(m["videoUrl"], akte.get("collection_id", collection_id),
                                                  akte.get("link_id", link_id), f"flow_{i}_{int(time.time())}.mp4")
        akte["gesichert"] = gesichert
        if not akte.get("gebucht") and _clips(job):
            from ai.services import bild_verbrauch as bv
            ids = [g.get("storage_object_id") for g in gesichert.values() if g.get("storage_object_id")]
            bv.buchen(agent=akte.get("agent") or bv.UNBEKANNT, quelle=akte.get("quelle") or "keine",
                      art="video", anzahl=len(_clips(job)),
                      model=((job.get("request") or {}).get("model")) or akte.get("model"),
                      routed_via="google-flow", kosten_usd=None,
                      storage_id=ids[0] if len(ids) == 1 else (ids or None), request_id=job_id)
            akte["gebucht"] = True
        _schreiben(job_id, akte)
    return {"gesichert": gesichert, "links_ausstehend": offen}


def _antwort(job_id: str, job: Dict[str, Any], extra: Optional[dict] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"job_id": job_id, "status": job.get("status") or "unknown"}
    clips = []
    gesichert = (extra or {}).get("gesichert") or {}
    for i, m in enumerate(_clips(job)):
        mid = m.get("mediaGenerationId") or f"clip{i}"
        g = gesichert.get(mid) or {}
        clips.append({"media_generation_id": m.get("mediaGenerationId"),
                      "storage_object_id": g.get("storage_object_id"),
                      "storage_url": g.get("storage_url"),
                      "flow_url": m.get("videoUrl"),
                      "save_error": g.get("save_error")})
    if clips:
        out["clips"] = clips
    resp = job.get("response") or {}
    if resp.get("remainingCredits") is not None:
        out["remaining_credits"] = resp.get("remainingCredits")
    if job.get("status") == "failed":
        out["error"] = job.get("error")
        out["error_code"] = job.get("code")
        if resp.get("failureReasons"):
            out["failure_reasons"] = resp.get("failureReasons")
    if (extra or {}).get("links_ausstehend"):
        out["hint"] = "Google hat noch nicht alle Video-Links geliefert; spaeter erneut abfragen."
    return out


# --- Endpunkte ---------------------------------------------------------------

class FlowVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Szene/Bewegung")
    model: Optional[str] = Field(default=None, description=f"{', '.join(MODELLE)}; Vorgabe {VORGABE_MODELL}")
    aspect_ratio: str = Field(default="landscape", description="landscape (16:9) | portrait (9:16)")
    duration: int = Field(default=8, description="Sekunden; Veo 4/6/8, quality nur 8, omni-flash 4-10, omni-flash-1.2 3-15")
    resolution: str = Field(default="720p", description="720p | 360p (360p nur Omni)")
    count: int = Field(default=1, ge=1, le=4, description="Clips je Auftrag (1-4), jeder kostet")
    start_image: Optional[str] = Field(default=None, description="Startbild: Storage-ID oder https-Adresse")
    end_image: Optional[str] = Field(default=None, description="Endbild (braucht start_image)")
    wait_for_result: bool = Field(default=False, description="true = bis zum Ende warten (60-180 s)")
    timeout_s: int = Field(default=300, ge=30, le=600)
    save_to_storage: bool = Field(default=True, description="Google-Links verfallen nach ~6 h")
    collection_id: Optional[str] = Field(default="ai-generated-video")
    link_id: Optional[str] = None
    max_credits: Optional[int] = Field(default=None, description="Abbruch vor dem Auftrag, wenn die Schaetzung hoeher liegt")
    confirm_api_billing: Optional[bool] = Field(
        default=False, description="Pflicht true: verbraucht Flow-Credits des Ultra-Kontos und Captcha-Loesungen")


@router.post("/genvideo/flow")
async def flow_video(request: FlowVideoRequest,
                     authorization: Optional[str] = Header(default=None),
                     x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name")):
    """Video ueber Google Flow (Veo 3.1 / Omni) mit Alex' Ultra-Konto."""
    if not request.confirm_api_billing:
        raise HTTPException(status_code=403, detail={
            "error": "api_billing_confirmation_required", "endpoint": "flow-video",
            "hint": ("Verbraucht Flow-Credits des Google-Ultra-Kontos und je Clip eine Captcha-Loesung. "
                     "`confirm_api_billing: true` mitschicken."),
        })
    modell = (request.model or VORGABE_MODELL).strip()
    if modell not in MODELLE:
        raise HTTPException(status_code=422, detail={"error": "unknown_model", "got": modell,
                                                     "available": list(MODELLE)})
    if request.duration not in MODELLE[modell]:
        raise HTTPException(status_code=422, detail={"error": "duration_not_supported", "model": modell,
                                                     "got": request.duration, "allowed": list(MODELLE[modell])})
    seite = _SEITEN.get(request.aspect_ratio.strip().lower())
    if not seite:
        raise HTTPException(status_code=422, detail={"error": "invalid_aspect_ratio",
                                                     "allowed": ["landscape", "portrait"]})
    aufl = request.resolution.strip().lower()
    if aufl not in ("720p", "360p") or (aufl == "360p" and modell.startswith("veo")):
        raise HTTPException(status_code=422, detail={"error": "invalid_resolution", "got": request.resolution,
                                                     "hint": "720p; 360p nur bei omni-flash*"})
    if request.end_image and not request.start_image:
        raise HTTPException(status_code=422, detail={"error": "end_image_requires_start_image"})
    schaetzung = credits_schaetzung(modell, request.duration, aufl, request.count)
    if request.max_credits is not None and schaetzung is not None and schaetzung > request.max_credits:
        raise HTTPException(status_code=422, detail={"error": "over_max_credits",
                                                     "estimated_credits": schaetzung,
                                                     "max_credits": request.max_credits})
    _token()  # 503 vor jedem Bild-Upload

    body: Dict[str, Any] = {"prompt": request.prompt, "model": modell, "aspectRatio": seite,
                            "duration": request.duration, "resolution": aufl,
                            "count": request.count, "async": True}
    if request.start_image:
        body["startImage"] = await _bild_hochladen(request.start_image)
    if request.end_image:
        body["endImage"] = await _bild_hochladen(request.end_image)

    from ai.services import bild_verbrauch as bv
    agent, quelle = bv.aufrufer(authorization, x_agent_name)
    antwort = await _flow("POST", "/videos", json_body=body)
    job_id = antwort.get("jobid") or antwort.get("jobId")
    if not job_id:
        raise HTTPException(status_code=502, detail={"error": "flow_without_job_id"})
    async with _gesperrt(job_id):
        _schreiben(job_id, {"agent": agent, "quelle": quelle, "model": modell,
                            "collection_id": request.collection_id, "link_id": request.link_id,
                            "angelegt": time.time(), "estimated_credits": schaetzung})
    logger.info("flow: Auftrag %s (%s, %ss, x%d) von %s", job_id, modell, request.duration,
                request.count, agent)

    if not request.wait_for_result:
        return {"job_id": job_id, "status": antwort.get("status") or "created", "model": modell,
                "estimated_credits": schaetzung,
                "hint": f"Stand: GET /ai/genvideo/flow/status/{job_id} (meist 60-180 s)"}

    frist = time.monotonic() + request.timeout_s
    job: Dict[str, Any] = antwort
    while time.monotonic() < frist:
        await asyncio.sleep(10)
        job = await _flow("GET", f"/jobs/{job_id}")
        if job.get("status") in ("completed", "failed"):
            break
    return await _stand_ausgeben(job_id, job, request.save_to_storage, request.collection_id,
                                 request.link_id, schaetzung)


async def _stand_ausgeben(job_id, job, sichern, collection_id, link_id, schaetzung=None):
    extra = None
    if job.get("status") == "completed":
        extra = await _abschliessen(job_id, job, sichern=sichern,
                                    collection_id=collection_id, link_id=link_id)
    out = _antwort(job_id, job, extra)
    if schaetzung is not None:
        out["estimated_credits"] = schaetzung
    return out


@router.get("/genvideo/flow/account")
async def flow_account():
    """Verknuepfte Google-Konten (Adresse gekuerzt), Gesundheit, Captcha-Guthaben."""
    konten = await _flow("GET", "/accounts")
    try:
        captcha = await _flow("GET", "/accounts/captcha-providers")
    except HTTPException as e:
        captcha = {"error": e.detail}

    def kurz(mail: str) -> str:
        name, _, dom = str(mail).partition("@")
        return f"{name[:1]}***@{dom}" if dom else "***"

    liste = []
    for mail, info in (konten or {}).items():
        eintrag = {"account": kurz(mail)}
        if isinstance(info, dict):
            for k in ("health", "credits", "remainingCredits", "tier", "error", "quarantine"):
                if k in info:
                    eintrag[k] = info[k]
        liste.append(eintrag)
    return {"accounts": liste, "linked": len(liste), "captcha": captcha,
            "setup": "https://useapi.net/assets/setup-browser/google-flow.html" if not liste else None}


@router.get("/genvideo/flow/status/{job_id:path}")
async def flow_status(job_id: str, save_to_storage: bool = True):
    """Stand eines Auftrags. Fertige Clips werden einmal gesichert und gebucht."""
    _job_pfad(job_id)  # Form pruefen
    job = await _flow("GET", f"/jobs/{job_id}")
    return await _stand_ausgeben(job_id, job, save_to_storage, None, None)


# --- Bilder (Nano Banana in Flow, Alex 14.11.2026) ---------------------------

BILD_MODELLE = ("nano-banana-2-lite", "nano-banana-2", "nano-banana-pro")
BILD_VORGABE = os.getenv("FLOW_IMAGE_DEFAULT_MODEL", "nano-banana-2")
BILD_SEITEN = ("16:9", "4:3", "1:1", "3:4", "9:16", "auto")


class FlowImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = Field(default=None, description=f"{', '.join(BILD_MODELLE)}; Vorgabe {BILD_VORGABE}")
    aspect_ratio: str = Field(default="1:1", description="16:9 | 4:3 | 1:1 | 3:4 | 9:16 | auto (nur mit Referenzen)")
    count: int = Field(default=1, ge=1, le=4, description="Bilder je Auftrag; useapi nimmt sonst 4")
    seed: Optional[int] = Field(default=None, ge=0)
    reference_images: Optional[List[str]] = Field(default=None, description="bis 10, Storage-ID oder https-Adresse")
    collection_id: Optional[str] = Field(default="ai-generated-image")
    link_id: Optional[str] = None
    confirm_api_billing: Optional[bool] = Field(
        default=False, description="Pflicht true: je Auftrag eine Captcha-Loesung (300 frei), ggf. Flow-Credits")


def _bilder(antwort: Dict[str, Any]) -> List[Dict[str, Any]]:
    medien = antwort.get("media") or (antwort.get("response") or {}).get("media") or []
    out = []
    for m in medien:
        g = ((m or {}).get("image") or {}).get("generatedImage") or {}
        if g:
            out.append(g)
    return out


@router.post("/genimage/flow")
async def flow_image(request: FlowImageRequest,
                     authorization: Optional[str] = Header(default=None),
                     x_agent_name: Optional[str] = Header(default=None, alias="X-Agent-Name")):
    """Bild ueber Google Flow (Nano Banana) mit Alex' Ultra-Konto; nur Abo, kein Bezahl-Rueckfall."""
    if not request.confirm_api_billing:
        raise HTTPException(status_code=403, detail={
            "error": "api_billing_confirmation_required", "endpoint": "flow-image",
            "hint": ("Laeuft ueber das Google-Ultra-Abo (Flow) und verbraucht je Auftrag eine "
                     "Captcha-Loesung. `confirm_api_billing: true` mitschicken."),
        })
    modell = (request.model or BILD_VORGABE).strip()
    if modell not in BILD_MODELLE:
        raise HTTPException(status_code=422, detail={"error": "unknown_model", "got": modell,
                                                     "available": list(BILD_MODELLE)})
    seite = request.aspect_ratio.strip()
    if seite not in BILD_SEITEN:
        raise HTTPException(status_code=422, detail={"error": "invalid_aspect_ratio", "allowed": list(BILD_SEITEN)})
    refs = list(request.reference_images or [])
    if seite == "auto" and not refs:
        raise HTTPException(status_code=422, detail={"error": "auto_requires_reference"})
    if len(refs) > 10:
        raise HTTPException(status_code=422, detail={"error": "too_many_references", "max": 10})
    _token()

    body: Dict[str, Any] = {"prompt": request.prompt, "model": modell, "aspectRatio": seite,
                            "count": request.count}
    if request.seed is not None:
        body["seed"] = request.seed
    for i, ref in enumerate(refs, start=1):
        body[f"reference_{i}"] = await _bild_hochladen(str(ref))

    from ai.services import bild_verbrauch as bv
    agent, quelle = bv.aufrufer(authorization, x_agent_name)
    antwort = await _flow("POST", "/images", json_body=body, timeout=240.0)  # sync, meist 10-20 s
    job_id = antwort.get("jobId") or antwort.get("jobid")
    bilder = []
    for i, g in enumerate(_bilder(antwort)):
        inhalt = None
        if not g.get("fifeUrl") and g.get("encodedImage"):
            import base64
            inhalt = base64.b64decode(g["encodedImage"])
        gesichert = await _datei_sichern(g.get("fifeUrl"), inhalt, "image/png", request.collection_id,
                                         request.link_id, f"flow_{modell}_{i}_{int(time.time())}.png")
        bilder.append({"storage_object_id": gesichert.get("storage_object_id"),
                       "image_url": gesichert.get("storage_url"),
                       "media_generation_id": g.get("mediaGenerationId"),
                       "seed": g.get("seed"), "save_error": gesichert.get("save_error")})
        bv.buchen(agent=agent, quelle=quelle, art="bild", anzahl=1, model=modell,
                  routed_via="google-flow", kosten_usd=None,
                  storage_id=gesichert.get("storage_object_id"), request_id=job_id)
    if not bilder:
        raise HTTPException(status_code=502, detail={"error": "flow_no_images", "job_id": job_id})
    out: Dict[str, Any] = {"job_id": job_id, "model": modell, "routed_via": "google-flow", "images": bilder,
                           "image_url": bilder[0]["image_url"], "storage_object_id": bilder[0]["storage_object_id"]}
    if antwort.get("remainingCredits") is not None:
        out["remaining_credits"] = antwort.get("remainingCredits")
    logger.info("flow: Bild %s (%s, x%d) von %s", job_id, modell, len(bilder), agent)
    return out
