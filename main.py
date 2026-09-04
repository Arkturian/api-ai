#!/usr/bin/env python3
"""
Arkturian AI API
================

Production-ready AI/ML API for the Arkturian platform.

Features:
- Text AI: Claude, ChatGPT, Gemini
- Image AI: Generation, Upscaling, Depth Maps
- Audio AI: TTS, SFX, Music Generation
- Dialog System: Multi-character conversations with TTS

Author: Arkturian Team
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import hmac
import logging
import os
from ai.provider_config import providers_status as _providers_status

# Import routes
from ai.routes import text_ai_routes, image_ai_routes, audio_ai_routes, dialog_routes, video_ai_routes, narration_routes, image_generation_routes, translate_routes, internal_routes, dashboard_routes, music_ai_routes, realtime_routes, hunyuan3d_routes, kling_routes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Arkturian AI API",
    version="1.0.0",
    description="AI/ML services for the Arkturian platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- #1184
# NUR MESSEN, NICHTS ABWEISEN.
#
# Befund vom 2026-08-17: Text- und Medien-Endpunkte sind ohne jede
# Anmeldung aus dem Internet erreichbar — `get_api_key()` ist ein
# Platzhalter, und der nginx davor prueft nichts.
#
# Einen Schluesselzwang einzuschalten braeche jeden bestehenden
# Aufrufer auf einen Schlag (guide-api, automation-api, knowledge-api,
# swfme, die MCP-Werkzeuge). Deshalb ZUERST die Frage beantworten, die
# die Entscheidung ueberhaupt erst treffbar macht: **Wer ruft heute
# ohne Schluessel?**
#
# Diese Middleware weist NICHTS ab und aendert KEINE Antwort. Sie
# schreibt eine Zeile, damit die Migration auf Zahlen steht statt auf
# Vermutungen. Der Zwang kommt spaeter und ist Alexanders Entscheidung.
# ---------------------------------------------------------------- #1184
# ZUGANGSSPERRE — Alexanders Entscheidung vom 2026-08-18.
#
# Statischer `X-API-KEY`. Wer ihn nicht traegt, kommt nicht durch.
# Legitime Traeger: die MCP-Server, KIs die Alexander den Schluessel
# direkt gibt, und das admin-Dashboard.
#
# ZWEI Dinge, die hier bewusst so sind:
#
# 1. **Ohne gesetzten Schluessel wird NICHT gesperrt.** Sonst waere der
#    Dienst in der Sekunde des Ausrollens tot — bevor irgendein
#    Aufrufer den Schluessel hat. Der Start warnt dann laut. Scharf
#    wird es, sobald `API_ACCESS_KEY` je Host gesetzt ist; das ist ein
#    Konfigurationsschritt, kein Deploy.
#
# 2. **Der GCP-Budget-Webhook ist ausgenommen.** Google kann keinen
#    Schluessel von uns tragen. Gemessen am 2026-08-18: 30 Aufrufe in
#    sieben Tagen, zuletzt 08:58 — er feuert, die Ausnahme ist NICHT
#    gegenstandslos. Wuerde er gesperrt, verstummten die
#    Kostenwarnungen, die es nur gibt, weil im Mai 209,56 EUR
#    unbemerkt abflossen.
_API_KEY_ENV = "API_ACCESS_KEY"
_KEY_HEADER = "x-api-key"
# Pfade, die per Bauart keinen Schluessel tragen koennen.
# Pfade ohne X-API-KEY-Zwang. Zwei sehr verschiedene Gruende:
#
# * Der GCP-Webhook KANN keinen Schluessel tragen (fremder Absender).
# * `/ai/realtime/token` hat eine STAERKERE eigene Pruefung — Nutzer-JWT
#   gegen den Aussteller plus Vollmacht. Ein zusaetzlicher statischer
#   Schluessel wuerde dort nichts sichern, sondern nur den Browser
#   aussperren: Der schickt zurecht keinen API-Schluessel, denn ein
#   Schluessel im Browser ist oeffentlich.
#
#   Gemessen auf Davids Kundeninstanz am 2026-08-19: Der Browser bekam
#   `401 api_key_required`, bevor mein JWT-Pfad ueberhaupt lief. Meine
#   Sperre von gestern haette den Realtime-Zugang jeder Instanz
#   erschlagen, sobald jemand `API_ACCESS_KEY` setzt — auch unseren,
#   beim Scharfschalten.
_OFFENE_PFADE = (
    "/ai/gemini/gcp-budget-webhook",
    "/ai/realtime/token",
)

if not os.getenv(_API_KEY_ENV):
    # Laut, nicht still: Eine Sperre, die mangels Konfiguration nicht
    # greift, ist genau die Art Luecke, die man fuer geschlossen haelt.
    logging.getLogger("api-ai.authwatch").warning(
        "%s ist NICHT gesetzt — die Zugangssperre (#1184) ist AUS, "
        "alle /ai/*-Endpunkte sind offen erreichbar. Schluessel setzen, "
        "um sie scharf zu schalten.", _API_KEY_ENV,
    )


@app.middleware("http")
async def _require_api_key(request, call_next):
    erwartet = os.getenv(_API_KEY_ENV)
    if not erwartet:
        return await call_next(request)
    pfad = request.url.path
    if not pfad.startswith("/ai/") or pfad in _OFFENE_PFADE:
        return await call_next(request)
    geliefert = request.headers.get(_KEY_HEADER) or ""
    # `compare_digest` statt `==`: gleiche Laufzeit unabhaengig davon,
    # an welcher Stelle zwei Schluessel sich unterscheiden.
    if not (geliefert and hmac.compare_digest(geliefert, erwartet)):
        logging.getLogger("api-ai.authwatch").warning(
            "GESPERRT path=%s client=%s ua=%s (#1184 — %s)",
            pfad,
            (request.client.host if request.client else "?"),
            (request.headers.get("user-agent") or "?")[:60],
            "kein Schluessel" if not geliefert else "falscher Schluessel",
        )
        return JSONResponse(
            status_code=401,
            content={"detail": {
                "error": "api_key_required",
                "hint": ("Dieser Dienst verlangt den Kopf `X-API-KEY`. "
                         "Den Schluessel vergibt Alexander."),
            }},
        )
    return await call_next(request)


@app.middleware("http")
async def _log_auth_presence(request, call_next):
    try:
        pfad = request.url.path
        if pfad.startswith("/ai/") and request.method == "POST":
            hat_bearer = bool(request.headers.get("authorization"))
            hat_key = bool(request.headers.get("x-api-key"))
            if not (hat_bearer or hat_key):
                # Nur den anonymen Fall protokollieren — der belegte ist
                # der Normalfall und wuerde das Journal fluten.
                ua = (request.headers.get("user-agent") or "?")[:80]
                # Zusatzkopfzeilen NUR fuer den einen unidentifizierten
                # Aufrufer (#1184): `axios/1.13.5` ruft /ai/claude, und
                # weder sein Prozess noch sein Quelltext liess sich
                # zuordnen — die Kandidaten laufen als root, Cloud hat
                # meine Vermutung widerlegt. Vielleicht verraet ihn eine
                # eigene Kopfzeile.
                #
                # Maskiert wird nach dem Muster aus der Betriebsregel:
                # zwischen ':' und '@' (URL-Form) UND klassische
                # Zuweisungen. Diese Aufrufe sind per Definition
                # anmeldungsfrei, aber „anonym" heisst nicht „harmlos" —
                # ein Sitzungskennzeichen in einer Kopfzeile gehoert
                # nicht ungefiltert ins Journal.
                extra = ""
                if "axios" in ua:
                    import re as _re
                    teile = []
                    for k, v in request.headers.items():
                        if k.lower() in ("user-agent", "accept", "connection",
                                         "host", "content-length"):
                            continue
                        v = _re.sub(r":[^:@]*@", ":***@", str(v))
                        if _re.search(r"(key|token|secret|auth|cookie)", k, _re.I):
                            v = "***"
                        teile.append(f"{k}={v[:60]}")
                    extra = " kopfzeilen[" + " ".join(teile[:8]) + "]"
                logging.getLogger("api-ai.authwatch").warning(
                    "ANONYM path=%s client=%s ua=%s%s (#1184 — nicht abgewiesen)",
                    pfad,
                    (request.client.host if request.client else "?"),
                    ua, extra,
                )
    except Exception:
        # Eine Messung darf den Dienst nie stoeren.
        pass
    return await call_next(request)


# Include routers
app.include_router(text_ai_routes.router, prefix="/ai", tags=["Text AI"])
app.include_router(image_ai_routes.router, prefix="/ai", tags=["Image AI"])
app.include_router(video_ai_routes.router, prefix="/ai", tags=["Video AI"])
app.include_router(audio_ai_routes.router, prefix="/ai", tags=["Audio AI"])
app.include_router(dialog_routes.router, prefix="/ai/dialog", tags=["Dialog System"])
app.include_router(image_generation_routes.router, prefix="/ai/scene", tags=["Scene Images"])
app.include_router(narration_routes.router, prefix="/ai", tags=["Narration TTS"])
app.include_router(music_ai_routes.router, prefix="/ai", tags=["Music AI"])
app.include_router(hunyuan3d_routes.router, prefix="/ai", tags=["Hunyuan 3D"])
app.include_router(kling_routes.router, prefix="/ai", tags=["Kling Video"])
app.include_router(translate_routes.router, prefix="/ai", tags=["Translation"])
app.include_router(realtime_routes.router, prefix="/ai", tags=["Realtime AI"])
app.include_router(internal_routes.router, prefix="/internal", tags=["Internal"])
app.include_router(dashboard_routes.router, prefix="/ai", tags=["Status Dashboard"])

# Static Realtime Test-HP — talk-to-the-model demo for OpenAI gpt-realtime
# and ElevenLabs Conv. AI. Served at /ai/realtime/test/.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(os.path.join(_STATIC_DIR, "realtime-test")):
    app.mount(
        "/ai/realtime/test",
        StaticFiles(directory=os.path.join(_STATIC_DIR, "realtime-test"), html=True),
        name="realtime-test",
    )

# Health check
_CLI_STATUS_TTL_S = 300.0
_cli_status_cache: dict = {}


def _cli_status(name: str = "codex") -> dict:
    """Ist das CLI, an dem ein ganzer Pfad haengt, aufrufbar — mit dem
    Environment DIESES Prozesses (PATH der Unit, CODEX_HOME)?

    Anlass 2026-09-04 (Automation/Guardian): auf arkturian zog der
    naechtliche Updater codex 0.153.2 ohne Plattformpaket, `codex
    --version` warf, jeder /ai/chatgpt-Aufruf endete mit 500 — und /health
    sagte weiter „healthy“, der Katalog „einsatzbereit“. Ein Health-Check,
    der das CLI nicht anfasst, prueft nur den Python-Prozess. 5 min
    Cache, damit Monitore den Dienst nicht mit npm-Starts fluten."""
    import subprocess
    import time as _t
    now = _t.monotonic()
    c = _cli_status_cache.get(name)
    if c and now - c["at"] < _CLI_STATUS_TTL_S:
        return c["status"]
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    try:
        r = subprocess.run([name, "--version"], capture_output=True, text=True,
                           timeout=10, env=env)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        ok = r.returncode == 0 and bool(out)
        status = {"ok": ok, "version": out.split()[-1] if ok else None,
                  "error": None if ok else (err or out or f"exit {r.returncode}")[-300:]}
    except FileNotFoundError:
        status = {"ok": False, "version": None, "error": f"{name}: not found on PATH"}
    except Exception as exc:
        status = {"ok": False, "version": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    status["checked_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S%z")
    _cli_status_cache[name] = {"at": now, "status": status}
    return status


def _realtime_faehigkeiten() -> list:
    """Welche Realtime-Werkzeuge diese Instanz kennt.

    Warum das hier steht (OnealServ-Codex, 2026-08-27): `/health`
    meldete keinen nachpruefbaren Stand, und die Frage „laeuft dort
    schon der neue Code" liess sich von aussen nicht beantworten.

    Ein Commit-Hash waere die naheliegende Antwort und die schlechtere:
    Der Deploy packt ein tar aus, das mitgelieferte `.git` bleibt auf
    einem alten Stand stehen — genau daran habe ich am 2026-08-26 eine
    Stunde verloren. Was man wirklich wissen will, ist nicht „welcher
    Commit", sondern „kann diese Instanz X". Also antworte ich darauf.

    Faellt der Import aus, bleibt die Liste leer statt den Health-Check
    mitzureissen: Ein Gesundheitsbericht, der selbst krank werden
    kann, ist keiner.
    """
    try:
        from ai.routes.realtime_routes import READ_TOOL_NAMES
        return sorted(READ_TOOL_NAMES)
    except Exception:
        return []


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arkturian-ai-api",
        "version": "1.0.0",
        "realtime_tools": _realtime_faehigkeiten(),
        # Der codex-Pfad (Kleinhirn, Prosa, Sol/Luna) haengt an einem
        # Node-CLI; ein gebrochenes CLI ist ein gebrochener Dienst.
        "clis": {"codex": _cli_status("codex")},
        # Welche Provider DIESE Instanz konfiguriert hat — read-only, ohne
        # Geheimnisse, ohne Erzeugung (Kundeninstanzen ohne unsere Toepfe).
        "providers": _providers_status(),
    }

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "service": "Arkturian AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "text_ai": [
                "POST /ai/claude",
                "POST /ai/chatgpt",
                "POST /ai/gemini",
                "POST /ai/gemini/vision"
            ],
            "image_ai": [
                "POST /ai/genimage (Higgsfield default)",
                "GET /ai/genimage/models",
                "POST /ai/upscale",
                "POST /ai/gendepth"
            ],
            "video_ai": [
                "POST /ai/genvideo (Image-to-Video)",
                "GET /ai/genvideo/status/{request_id}",
                "POST /ai/genvideo/cancel/{request_id}",
                "GET /ai/genvideo/models"
            ],
            "translation": [
                "POST /ai/translate",
                "POST /ai/translate/batch",
                "GET /ai/translate/languages"
            ],
            "audio_ai": [
                "POST /ai/generate_speech",
                "POST /ai/gensfx",
                "POST /ai/genmusic",
                "POST /ai/genmusic_eleven",
                "POST /ai/transcribe"
            ],
            "dialog": [
                "POST /ai/dialog/start",
                "GET /ai/dialog/status",
                "POST /ai/dialog/cancel"
            ]
        }
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )

# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Arkturian AI API starting up...")
    logger.info("✅ All AI services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("👋 Arkturian AI API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,  # Different port from artrack(8001) and storage(8002)
        reload=True,
        log_level="info"
    )
