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
import logging
import os

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
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arkturian-ai-api",
        "version": "1.0.0"
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
