"""Adressen holen, ohne den Storage-Schluessel herzugeben.

Einziger Ort, der entscheidet, ob eine Adresse den Dienstschluessel sieht.
Adressen und IDs vom Aufrufer holt AiApi ohne Schluessel (#2659): Diese Routen
pruefen keinen Aufrufer, mit Schluessel koennte jeder nicht oeffentliche Dateien
lesen lassen. Es gehen oeffentliche Dateien und signierte Adressen. Wo der
Schluessel doch einmal gilt (``mit_schluessel=True``), je Sprung neu geprueft:

- Nur der eigene Ursprung bekommt ihn (Issue #1794, #2654). Ein Teilstueck
  des Pfades reicht nicht, Umleitungen tragen ihn nicht mit.
- Signierte Adressen bekommen ihn nie: Die Signatur ist die Berechtigung.
- Private Dateien liest auch der Schluessel nicht. Das ist gewollt. Ein Dienst
  liest sie nur ueber eine signierte Adresse im Auftrag eines gebundenen
  Aufrufers (Post 5782). Wer scheitert, bekommt ``private_media`` statt eines
  stillen oder irrefuehrenden Fehlers.
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager, contextmanager
from typing import List, Optional
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit

import httpx

from ai.clients.storage_client import storage_api_key

UMLEITUNGEN = (301, 302, 303, 307, 308)
MAX_SPRUENGE = 3
PRIVATE_MEDIA_HINWEIS = (
    "Private Datei: Der Dienstschluessel liest sie nicht. Als Aufrufer mit "
    "Bindung eine signierte Adresse holen (assets_download) und diese "
    "uebergeben. Ohne Bindung muss der Besitzer sie teilen (assets_share)."
)


class ZuVieleUmleitungen(Exception):
    """Mehr Spruenge als MAX_SPRUENGE."""


class PrivateDatei(Exception):
    """Storage verweigert die Datei mit ``private_media``."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"private_media: {ohne_anfrage(url)}")


class GesperrteDatei(Exception):
    """Storage verweigert die Datei wegen des Safety-Urteils (451).

    Die Signatur hebt nur die Zugriffspruefung auf, nicht das Urteil."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"quarantined: {ohne_anfrage(url)}")


def _ursprung(url: str):
    """(Host, effektiver Port) einer https-Adresse — oder None.

    Der Port gehoert zum Ursprung. `api-storage.arkturian.com:8443` ist
    ein anderer Dienst als `api-storage.arkturian.com`, auch wenn der
    Hostname gleich lautet (Story-Codex, #1794 Restfall: die erste
    Fassung verglich nur den Hostnamen und gab dem fremden Port den
    Schluessel).
    """
    p = urlsplit(url or "")
    host = (p.hostname or "").lower()
    if not host:
        return None
    # http nur auf der Schleife: dort verlaesst der Schluessel den Rechner
    # nicht, das Leitungs-Argument greift nicht. oneal spricht seinen
    # eigenen Storage als http://127.0.0.1:8001 an — mit reiner
    # https-Pflicht haette die Instanz ihre eigenen privaten Referenzen
    # anonym geholt und waere daran gescheitert (gelesen am 12.09.).
    schleife = host in ("127.0.0.1", "localhost", "::1")
    if p.scheme == "https":
        vorgabe = 443
    elif p.scheme == "http" and schleife:
        vorgabe = 80
    else:
        return None
    try:
        port = p.port or vorgabe
    except ValueError:
        return None
    return (host, port)


def _storage_urspruenge() -> set:
    """Urspruenge (Host, Port), die den Storage-Schluessel sehen duerfen.

    Quelle ist die konfigurierte Storage-Adresse; weitere nur
    ausdruecklich ueber STORAGE_EXTRA_HOSTS (Komma-Liste, mit oder ohne
    Schema/Port). Eine Instanz mit eigenem Storage (oneal) traegt ihren
    dort ein.
    """
    roh = [os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")]
    roh += [t for t in (os.getenv("STORAGE_EXTRA_HOSTS", "") or "").split(",") if t.strip()]
    urspruenge = set()
    for eintrag in roh:
        eintrag = eintrag.strip()
        if not eintrag:
            continue
        if "//" not in eintrag:
            eintrag = "https://" + eintrag
        u = _ursprung(eintrag)
        if u:
            urspruenge.add(u)
    return urspruenge


def ist_signiert(url: str) -> bool:
    """Traegt die Adresse eine Storage-Signatur?"""
    q = parse_qs(urlsplit(url or "").query)
    return bool(q.get("sig") and q.get("exp"))


def darf_schluessel_sehen(url: str) -> bool:
    """Darf DIESE Adresse den Storage-Schluessel bekommen?

    Geprueft wird der URSPRUNG (https, Host, effektiver Port), nicht ein
    Teilstueck des Pfades. Der fruehere Test ``"/storage/media/" in url``
    traf jede fremde Adresse, die diesen Pfad nur nachbaut —
    `https://fremder-host/storage/media/1` bekam den Schluessel (Issue
    #1794, beim Story-Abnahme-Review isoliert nachgestellt).

    https ist Pflicht — ausser auf der Schleife (127.0.0.1, localhost,
    ::1), wo nichts auf die Leitung geht. Ueber http nach draussen waere
    der Schluessel lesbar, und ein erzwungener Rueckfall auf http ist der
    billigste Weg, ihn abzugreifen.
    """
    if ist_signiert(url):
        return False
    u = _ursprung(url)
    if u is None or u not in _storage_urspruenge():
        return False
    return "/storage/" in (urlsplit(url).path or "")


def ohne_anfrage(url: str) -> str:
    """Adresse ohne Query und Fragment, fuer Protokolle und Fehlertexte."""
    p = urlsplit(str(url or ""))
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def _ist_private_media(status: int, inhalt: bytes) -> bool:
    if status != 403:
        return False
    try:
        detail = json.loads(inhalt or b"{}").get("detail")
    except (ValueError, AttributeError):
        return False
    return isinstance(detail, dict) and detail.get("code") == "private_media"


def private_media_detail(url: str) -> dict:
    """Antwortkoerper fuer den 403 an den Aufrufer."""
    return {"error": "private_media", "url": ohne_anfrage(url), "hint": PRIVATE_MEDIA_HINWEIS}


def gesperrt_detail(url: str) -> dict:
    """Antwortkoerper fuer den 451 an den Aufrufer."""
    return {"error": "quarantined", "url": ohne_anfrage(url),
            "hint": "Storage hat die Datei wegen des Safety-Urteils gesperrt."}


def medien_adressen(text: str) -> List[str]:
    """Storage-Adressen aus freiem Text, je Objekt die brauchbarste.

    Der Text ist oft eine eingefuegte Werkzeug-Antwort (JSON). Anfuehrungszeichen
    und Backslashes beenden deshalb die Adresse: Vorher lief der Treffer bis zum
    naechsten Leerzeichen weiter und zerstoerte die Signatur (Post 5782, 132824
    und 133117). Steht zu einem Objekt eine signierte Adresse da, fallen die
    unsignierten weg: Sie koennen bei privaten Dateien nur scheitern.
    """
    treffer = [u.rstrip(".,;:)]}>") for u in re.findall(r"""https?://[^\s"'`<>\\]+""", text or "")
               if "/storage/media/" in u]
    treffer = list(dict.fromkeys(treffer))
    signierte = {urlsplit(u).path for u in treffer if ist_signiert(u)}
    return [u for u in treffer if ist_signiert(u) or urlsplit(u).path not in signierte]


def _kopf(ziel: str, kette_eigen: bool) -> dict:
    return {"X-API-KEY": storage_api_key()} if kette_eigen and darf_schluessel_sehen(ziel) else {}


@asynccontextmanager
async def oeffne(client: httpx.AsyncClient, url: str, mit_schluessel: bool = False):
    """Streamt ``url`` und entscheidet je Sprung neu ueber den Schluessel.

    Nach einem Sprung ueber einen fremden Host gibt es keinen Schluessel mehr:
    Eine fremde Adresse darf nicht per Umleitung auf den eigenen Storage zeigen
    und so nicht oeffentliche Dateien lesen (Post 5782).

    Der Client muss ``follow_redirects=False`` haben. Raises PrivateDatei,
    GesperrteDatei, ZuVieleUmleitungen oder httpx.HTTPStatusError.
    """
    ziel = url
    eigen = mit_schluessel
    for _ in range(MAX_SPRUENGE + 1):
        eigen = eigen and darf_schluessel_sehen(ziel)
        async with client.stream("GET", ziel, headers=_kopf(ziel, eigen)) as r:
            ort = r.headers.get("location") if r.status_code in UMLEITUNGEN else None
            if ort:
                ziel = urljoin(ziel, ort)
                continue
            if r.status_code == 403 and _ist_private_media(403, await r.aread()):
                raise PrivateDatei(url)
            if r.status_code == 451:
                raise GesperrteDatei(url)
            r.raise_for_status()
            yield r
            return
    raise ZuVieleUmleitungen(ohne_anfrage(url))


@contextmanager
def oeffne_sync(client: httpx.Client, url: str, mit_schluessel: bool = False):
    """Wie ``oeffne``, fuer synchrone Aufrufer."""
    ziel = url
    eigen = mit_schluessel
    for _ in range(MAX_SPRUENGE + 1):
        eigen = eigen and darf_schluessel_sehen(ziel)
        with client.stream("GET", ziel, headers=_kopf(ziel, eigen)) as r:
            ort = r.headers.get("location") if r.status_code in UMLEITUNGEN else None
            if ort:
                ziel = urljoin(ziel, ort)
                continue
            if r.status_code == 403 and _ist_private_media(403, r.read()):
                raise PrivateDatei(url)
            if r.status_code == 451:
                raise GesperrteDatei(url)
            r.raise_for_status()
            yield r
            return
    raise ZuVieleUmleitungen(ohne_anfrage(url))


def ist_private_antwort(r: httpx.Response) -> bool:
    """Fuer Aufrufer, die die ganze Antwort schon gelesen haben."""
    return _ist_private_media(r.status_code, r.content)


class AnfrageAusProtokoll(logging.Filter):
    """Entfernt die Query aus den Zeilen des httpx-Protokolls.

    httpx schreibt je Anfrage die volle Adresse ins Journal, also auch die
    Signatur privater Dateien und Schluessel in Anbieter-Adressen (#2654).
    Die Zeile selbst bleibt, sie wird fuer die Fehlersuche gebraucht.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args: Optional[tuple] = record.args if isinstance(record.args, tuple) else None
        if args and len(args) >= 2:
            record.args = (args[0], ohne_anfrage(args[1])) + args[2:]
        return True
