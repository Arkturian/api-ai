"""SSRF-Sperre fuer Adressen vom Aufrufer (#2744, gemeldet von P3s Cloud
ueber Cloud am 12.11.2026).

Vorher holte api-ai jede Adresse und folgte Umleitungen ohne Schema- oder
Zielpruefung: `reference_image_urls` (genimage, audio/mix), `file_url`
(transcribe) und Vision-Adressen mit `/storage/media/` auf beliebigem Host.
So liessen sich interne, unauthentifizierte Endpunkte abrufen
(127.0.0.1:<port>, 10.x, 169.254.169.254).

Kein Netz und kein echtes DNS: `_aufloesen` und MockTransport stellt der
Test selbst.
"""
import httpx
import pytest
from fastapi import HTTPException

from ai.clients import storage_abruf as s
from ai.routes import audio_ai_routes as a
from ai.routes import image_ai_routes as g
from ai.routes import text_ai_routes as t

pytestmark = pytest.mark.ziel_echt

OEFFENTLICH = "93.184.216.34"
DNS = {
    "bilder.example": [OEFFENTLICH],
    "intern.example": ["10.0.0.5"],
    "gemischt.example": [OEFFENTLICH, "10.0.0.5"],
    "api-storage.arkturian.com": ["127.0.0.1"],   # wie /etc/hosts auf arkserver
}


@pytest.fixture(autouse=True)
def _umgebung(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)
    monkeypatch.delenv("AI_REFERENZ_FREIGABE", raising=False)

    def aufloesen(host, port):
        if host in DNS:
            return DNS[host]
        # IP-Literale loest getaddrinfo zu sich selbst auf
        import ipaddress
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            raise s.socket.gaierror("unbekannt")

    monkeypatch.setattr(s, "_aufloesen", aufloesen)


def _gesperrt(url):
    with pytest.raises(s.ZielGesperrt) as e:
        s.ziel_pruefen(url)
    return e.value.grund


# --- Pruefung ------------------------------------------------------------

def test_oeffentliches_https_wird_mit_der_geprueften_ip_verbunden():
    verbinden, kopf, ext = s.ziel_pruefen("https://bilder.example/a.png?x=1")
    assert verbinden == f"https://{OEFFENTLICH}/a.png?x=1"
    assert kopf == {"Host": "bilder.example"}
    assert ext == {"sni_hostname": "bilder.example"}       # Zertifikat gilt fuer den Namen


@pytest.mark.parametrize("url", [
    "https://127.0.0.1/x", "https://127.0.0.1:8477/ki-vorlage/a.png", "https://10.1.2.3/x",
    "https://192.168.1.1/x", "https://172.16.0.1/x", "https://169.254.169.254/latest/meta-data/",
    "https://0.0.0.0/x", "https://[::1]/x", "https://[fc00::1]/x", "https://[fe80::1]/x",
    "https://224.0.0.1/x", "https://[::ffff:127.0.0.1]/x", "https://100.64.0.1/x",
    "https://intern.example/x",
])
def test_interne_ziele_sind_gesperrt(url):
    assert _gesperrt(url) == "internal_address"


def test_gemischte_dns_antwort_ist_gesperrt():
    """Eine oeffentliche und eine interne Adresse: Rebinding-Versuch."""
    assert _gesperrt("https://gemischt.example/x") == "internal_address"


def test_http_nach_draussen_ist_gesperrt():
    assert _gesperrt("http://bilder.example/a.png") == "https_required"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://bilder.example/", "ftp://bilder.example/x"])
def test_fremde_schemata_sind_gesperrt(url):
    assert _gesperrt(url) == "scheme_not_allowed"


def test_unaufloesbar_ist_gesperrt():
    assert _gesperrt("https://gibtsnicht.example/x") == "unresolvable"


# --- Freigabeliste -------------------------------------------------------

def test_freigabe_greift_genau(monkeypatch):
    """P3: Webdeveloper liefert Einmal-Adressen unter genau diesem Praefix."""
    monkeypatch.setenv("AI_REFERENZ_FREIGABE", "http://127.0.0.1:8477/ki-vorlage/")
    verbinden, kopf, ext = s.ziel_pruefen("http://127.0.0.1:8477/ki-vorlage/abc.png?t=1")
    assert verbinden == "http://127.0.0.1:8477/ki-vorlage/abc.png?t=1" and ext == {}
    for daneben in [
        "http://127.0.0.1:8478/ki-vorlage/abc.png",      # anderer Port
        "http://127.0.0.1:8477/ki-vorlagex/abc.png",     # Praefix nur angeschnitten
        "http://127.0.0.1:8477/admin",                   # anderer Pfad
        "http://localhost:8477/ki-vorlage/abc.png",      # anderer Host
        "https://127.0.0.1:8477/ki-vorlage/abc.png",     # anderes Schema
        "http://127.0.0.1:8477/ki-vorlage/../admin",     # Punkt-Segment
        "http://127.0.0.1:8477/ki-vorlage/%2e%2e/admin", # kodiert
    ]:
        with pytest.raises(s.ZielGesperrt):
            s.ziel_pruefen(daneben)


def test_eigener_storage_auch_auf_der_schleife(monkeypatch):
    """Auf arkserver zeigt api-storage.arkturian.com per /etc/hosts auf
    127.0.0.1. Die eigene Storage-Adresse muss weiter gehen — nur /storage/."""
    verbinden, kopf, _ = s.ziel_pruefen("https://api-storage.arkturian.com/storage/media/1")
    assert verbinden == "https://127.0.0.1/storage/media/1" and kopf["Host"] == "api-storage.arkturian.com"
    assert _gesperrt("https://api-storage.arkturian.com/admin") == "internal_address"


# --- Abrufwege -----------------------------------------------------------

def _mock(antwort):
    gesehen = []

    def handler(request):
        gesehen.append((str(request.url), request.headers.get("host")))
        return antwort(request)

    return httpx.MockTransport(handler), gesehen


@pytest.mark.asyncio
async def test_referenz_umleitung_auf_die_schleife_wird_nicht_verfolgt():
    def antwort(request):
        if request.headers.get("host") == "bilder.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:9000/geheim"})
        return httpx.Response(200, content=b"GEHEIM")

    transport, gesehen = _mock(antwort)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(HTTPException) as e:
            await g._hole_referenz(client, "https://bilder.example/a.png", "")
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "reference_image_url_blocked"
    assert e.value.detail["url"] == "https://bilder.example/a.png"     # Aufrufer-Adresse, kein internes Ziel
    assert "127.0.0.1" not in str(e.value.detail)
    assert [h for _, h in gesehen] == ["bilder.example"]               # zweiter Sprung nie abgeschickt


@pytest.mark.asyncio
async def test_referenz_umleitung_auf_internen_namen_wird_nicht_verfolgt():
    def antwort(request):
        return httpx.Response(302, headers={"location": "https://intern.example/x"})

    transport, gesehen = _mock(antwort)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(HTTPException) as e:
            await g._hole_referenz(client, "https://bilder.example/a.png", "")
    assert e.value.detail["code"] == "internal_address" and len(gesehen) == 1


@pytest.mark.asyncio
async def test_referenz_direkt_auf_metadaten_geht_nie_raus(monkeypatch):
    transport, gesehen = _mock(lambda r: httpx.Response(200, content=b"x"))
    echt = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: echt(transport=transport, **k))
    with pytest.raises(HTTPException) as e:
        await g._download_reference_images(["https://169.254.169.254/latest/meta-data/"])
    assert e.value.status_code == 422 and gesehen == []


@pytest.mark.asyncio
async def test_fehlertext_traegt_keine_antwortinhalte(monkeypatch):
    transport, _ = _mock(lambda r: httpx.Response(500, content=b"interner Stacktrace mit Passwort"))
    echt = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: echt(transport=transport, **k))
    with pytest.raises(HTTPException) as e:
        await g._download_reference_images(["https://bilder.example/a.png?sig=1&exp=2"])
    assert e.value.detail == {"error": "reference_image_fetch_failed",
                              "url": "https://bilder.example/a.png", "status": 500}


@pytest.mark.asyncio
async def test_transcribe_auf_internes_ziel_ist_422(monkeypatch):
    transport, gesehen = _mock(lambda r: httpx.Response(200, content=b"x"))
    echt = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: echt(transport=transport, **k))
    with pytest.raises(HTTPException) as e:
        await a._fetch_audio_from_url("http://10.0.0.7:8080/intern.mp3")
    assert e.value.status_code == 422 and e.value.detail["error"] == "file_url_blocked"
    assert gesehen == []


def test_vision_holt_keine_storage_pfade_auf_fremden_internen_hosts(monkeypatch):
    """medien_adressen nimmt jede Adresse mit /storage/media/ — auch
    http://127.0.0.1:<port>/storage/media/1. Die Sperre haelt sie auf."""
    transport, gesehen = _mock(lambda r: httpx.Response(200, content=b"\x89PNG"))
    echt = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **k: echt(transport=transport, **k))
    out = t._download_storage_images("schau http://127.0.0.1:9000/storage/media/1 an")
    assert out == [] and gesehen == []


def test_storage_ids_werden_zu_eigenen_adressen():
    r = g.ImageGenRequest(prompt="x", reference_storage_ids=[12, 13])
    assert r.reference_image_urls == [
        "https://api-storage.arkturian.com/storage/media/12?variant=full",
        "https://api-storage.arkturian.com/storage/media/13?variant=full",
    ]
    assert r.reference_storage_ids is None
