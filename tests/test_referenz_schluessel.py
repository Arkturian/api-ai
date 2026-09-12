"""Der Storage-Schluessel verlaesst die eigenen Hosts nicht (#1794).

Gefunden beim Story-Abnahme-Review, isoliert nachgestellt. Zwei Wege
nach draussen, beide bestaetigt:

1. Die Entscheidung fiel per Teilstueck des PFADES
   (``"/storage/media/" in url``). Jede fremde Adresse, die diesen Pfad
   nachbaut, bekam den Schluessel.
2. ``follow_redirects=True``: httpx entfernt beim Hostwechsel nur
   ``Authorization``, eigene Kopfzeilen wie ``X-API-KEY`` nimmt es mit.
   Eine echte Storage-Adresse, die auf einen fremden Host umleitet, trug
   ihn also dorthin.

Kein echter Schluessel, kein Netz: MockTransport.
"""

import httpx
import pytest

from ai.routes import image_ai_routes as g

SCHLUESSEL = "test-schluessel-nicht-echt"


@pytest.fixture(autouse=True)
def _eigener_storage(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    monkeypatch.delenv("STORAGE_EXTRA_HOSTS", raising=False)


# ---------------------------------------------------- Hostpruefung

def test_eigener_storage_bekommt_den_schluessel():
    assert g._darf_schluessel_sehen("https://api-storage.arkturian.com/storage/media/17")


def test_fremder_host_mit_demselben_pfad_bekommt_ihn_nicht():
    """Der gemeldete Fall. Frueher: Treffer, weil der Pfad passte."""
    assert not g._darf_schluessel_sehen("https://fremder-host.example/storage/media/17")


def test_eigener_name_als_teilstueck_fremder_hosts_zaehlt_nicht():
    """Nachbarfall: Hostvergleich, nicht Teilstueck."""
    for url in (
        "https://api-storage.arkturian.com.boese.example/storage/media/1",
        "https://boese.example/?x=api-storage.arkturian.com/storage/media/1",
    ):
        assert not g._darf_schluessel_sehen(url), url


def test_ohne_https_kein_schluessel():
    assert not g._darf_schluessel_sehen("http://api-storage.arkturian.com/storage/media/1")


def test_zusatzhost_nur_ausdruecklich(monkeypatch):
    url = "https://api-storage.oneal.eu/storage/media/1"
    assert not g._darf_schluessel_sehen(url)
    monkeypatch.setenv("STORAGE_EXTRA_HOSTS", "api-storage.oneal.eu")
    assert g._darf_schluessel_sehen(url)


# ---------------------------------------------------- Umleitungen

def _client(gesehen):
    def handler(request: httpx.Request) -> httpx.Response:
        gesehen.append((str(request.url), request.headers.get("x-api-key")))
        if request.url.host == "api-storage.arkturian.com" and request.url.path.startswith("/storage/media/"):
            return httpx.Response(302, headers={"location": "https://abfluss.example/bild.png"})
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


@pytest.mark.asyncio
async def test_umleitung_auf_fremden_host_traegt_den_schluessel_nicht_mit():
    gesehen = []
    async with _client(gesehen) as client:
        r = await g._hole_referenz(client, "https://api-storage.arkturian.com/storage/media/9", SCHLUESSEL)
    assert r.status_code == 200
    assert len(gesehen) == 2
    erste_url, erster_kopf = gesehen[0]
    zweite_url, zweiter_kopf = gesehen[1]
    assert erster_kopf == SCHLUESSEL          # eigener Host: ja
    assert "abfluss.example" in zweite_url
    assert zweiter_kopf is None               # fremder Host: nein


@pytest.mark.asyncio
async def test_endlose_umleitung_wird_abgebrochen():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://kreis.example/weiter"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await g._hole_referenz(client, "https://kreis.example/start", SCHLUESSEL)
        assert exc.value.detail["error"] == "reference_image_too_many_redirects"


@pytest.mark.asyncio
async def test_fremde_adresse_wird_anonym_geholt():
    gesehen = []
    async with _client(gesehen) as client:
        await g._hole_referenz(client, "https://fremder-host.example/storage/media/1", SCHLUESSEL)
    assert gesehen[0][1] is None


# ---------------------------------------------------- Ursprung, nicht Hostname
# Restfall von Story-Codex (#1794, 12.09. nach dem ersten Fix): die erste
# Fassung verglich den Hostnamen und warf den Port weg. Ein fremder
# Dienst auf demselben Host, anderer Port, bekam den Schluessel.


def test_anderer_port_am_erlaubten_host_bekommt_ihn_nicht():
    assert g._darf_schluessel_sehen("https://api-storage.arkturian.com/storage/media/1")
    assert not g._darf_schluessel_sehen("https://api-storage.arkturian.com:8443/storage/media/1")
    assert not g._darf_schluessel_sehen("https://api-storage.arkturian.com:444/storage/media/1")


def test_ausdruecklicher_port_443_ist_derselbe_ursprung():
    """Nachbarfall: 443 ausgeschrieben ist kein anderer Ursprung."""
    assert g._darf_schluessel_sehen("https://api-storage.arkturian.com:443/storage/media/1")


def test_konfigurierter_port_zaehlt(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "https://api-storage.oneal.eu:8443")
    assert g._darf_schluessel_sehen("https://api-storage.oneal.eu:8443/storage/media/1")
    assert not g._darf_schluessel_sehen("https://api-storage.oneal.eu/storage/media/1")


def test_unsinniger_port_wird_abgewiesen():
    assert not g._darf_schluessel_sehen("https://api-storage.arkturian.com:99999/storage/media/1")


@pytest.mark.asyncio
async def test_umleitung_auf_anderen_port_traegt_den_schluessel_nicht_mit():
    gesehen = []

    def handler(request: httpx.Request) -> httpx.Response:
        gesehen.append((str(request.url), request.headers.get("x-api-key")))
        if request.url.port is None:
            return httpx.Response(302, headers={"location": "https://api-storage.arkturian.com:8443/storage/media/1"})
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        await g._hole_referenz(client, "https://api-storage.arkturian.com/storage/media/1", SCHLUESSEL)
    assert gesehen[0][1] == SCHLUESSEL
    assert gesehen[1][1] is None


# ---------------------------------------------------- Schleife: http erlaubt
# oneal spricht seinen Storage als http://127.0.0.1:8001 an (12.09.
# gelesen). Reine https-Pflicht haette die Instanz ihre eigenen privaten
# Referenzen anonym holen lassen. Auf der Schleife geht nichts auf die
# Leitung — dort ist http kein Leck.


def test_schleife_http_mit_konfiguriertem_port(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "http://127.0.0.1:8001")
    assert g._darf_schluessel_sehen("http://127.0.0.1:8001/storage/media/1")
    assert not g._darf_schluessel_sehen("http://127.0.0.1:8002/storage/media/1")
    assert not g._darf_schluessel_sehen("http://127.0.0.1/storage/media/1")


def test_http_nach_draussen_bleibt_verboten(monkeypatch):
    """Der Gegenfall: die Ausnahme gilt nur der Schleife, nie einem Namen."""
    monkeypatch.setenv("STORAGE_API_URL", "http://api-storage.arkturian.com")
    assert not g._darf_schluessel_sehen("http://api-storage.arkturian.com/storage/media/1")


def test_umleitung_von_schleife_nach_draussen_traegt_nichts_mit(monkeypatch):
    monkeypatch.setenv("STORAGE_API_URL", "http://127.0.0.1:8001")
    gesehen = []

    def handler(request):
        gesehen.append((str(request.url), request.headers.get("x-api-key")))
        if request.url.host == "127.0.0.1":
            return httpx.Response(302, headers={"location": "http://abfluss.example/x.png"})
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    import asyncio

    async def lauf():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            await g._hole_referenz(client, "http://127.0.0.1:8001/storage/media/1", SCHLUESSEL)

    asyncio.run(lauf())
    assert gesehen[0][1] == SCHLUESSEL
    assert gesehen[1][1] is None
