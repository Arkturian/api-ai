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

import ipaddress
import json
import logging
import os
import re
import socket
from urllib.parse import unquote
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


# ── Ziel-Sperre gegen SSRF (#2744) ───────────────────────────────────────
#
# Adressen vom Aufrufer duerfen nur nach draussen zeigen. Ohne diese Sperre
# konnte jeder Aufrufer api-ai interne, unauthentifizierte Endpunkte holen
# lassen (127.0.0.1:<port>, private Netze, 169.254.169.254), auch ueber eine
# Umleitung. Geprueft wird JE SPRUNG die aufgeloeste IP, und verbunden wird
# mit genau dieser IP (gegen DNS-Rebinding zwischen Pruefung und Verbindung).

class ZielGesperrt(Exception):
    """Adresse zeigt auf ein nicht erlaubtes Ziel (Schema, internes Netz)."""

    def __init__(self, url: str, grund: str):
        self.url = url
        self.grund = grund
        super().__init__(f"{grund}: {ohne_anfrage(url)}")


ZIEL_GESPERRT_HINWEIS = (
    "Nur https-Adressen auf oeffentliche Ziele. Interne Quellen gibt die "
    "Instanz ueber AI_REFERENZ_FREIGABE frei; Referenzbilder gehen auch als "
    "Storage-ID."
)


def ziel_gesperrt_detail(e: "ZielGesperrt", error: str = "url_blocked") -> dict:
    """Fehlertext nach aussen: Grund und Adresse ohne Query, keine IP und
    keine Antwortinhalte des Ziels."""
    return {"error": error, "code": e.grund, "url": ohne_anfrage(e.url), "hint": ZIEL_GESPERRT_HINWEIS}


def _vorgabe_port(scheme: str) -> Optional[int]:
    return {"https": 443, "http": 80}.get(scheme)


def _pfad_sauber(pfad: str) -> bool:
    """Keine Punkt-Segmente (auch kodiert): `/ki-vorlage/../admin` darf das
    Praefix nicht erfuellen."""
    teile = unquote(unquote(pfad or "/")).replace("\\", "/").split("/")
    return not any(t in (".", "..") for t in teile)


def _freigaben() -> list:
    """AI_REFERENZ_FREIGABE: Komma-Liste `schema://host:port/pfadpraefix`.
    Ohne Port gilt der Standardport des Schemas, ohne Pfad `/`."""
    out = []
    for roh in (os.getenv("AI_REFERENZ_FREIGABE", "") or "").split(","):
        roh = roh.strip()
        if not roh:
            continue
        p = urlsplit(roh)
        scheme = (p.scheme or "").lower()
        host = (p.hostname or "").lower()
        if scheme not in ("http", "https") or not host:
            continue
        try:
            port = p.port or _vorgabe_port(scheme)
        except ValueError:
            continue
        out.append((scheme, host, port, p.path or "/"))
    return out


def ist_freigegeben(url: str) -> bool:
    """Bewusst interne Quelle? Eigene Storage-Urspruenge (nur `/storage/`)
    oder ein Eintrag aus AI_REFERENZ_FREIGABE — Schema, Host und Port exakt,
    Pfad mit Praefix, ohne Punkt-Segmente."""
    p = urlsplit(url or "")
    pfad = p.path or "/"
    if not _pfad_sauber(pfad):
        return False
    u = _ursprung(url)
    if u is not None and u in _storage_urspruenge() and pfad.startswith("/storage/"):
        return True
    scheme = (p.scheme or "").lower()
    host = (p.hostname or "").lower()
    try:
        port = p.port or _vorgabe_port(scheme)
    except ValueError:
        return False
    for f_scheme, f_host, f_port, f_pfad in _freigaben():
        if (scheme, host, port) == (f_scheme, f_host, f_port) and pfad.startswith(f_pfad):
            return True
    return False


def _ip_oeffentlich(ip) -> bool:
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved and not ip.is_unspecified


def _aufloesen(host: str, port: int) -> list:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


def ziel_pruefen(url: str) -> tuple:
    """-> (verbinden_mit_url, kopfzeilen, extensions). Raises ZielGesperrt.

    Freigegeben: beliebiges Ziel, aber mit der aufgeloesten IP verbunden.
    Sonst: nur https, und JEDE aufgeloeste Adresse muss oeffentlich sein
    (eine gemischte Antwort ist ein Rebinding-Versuch)."""
    p = urlsplit(url or "")
    scheme = (p.scheme or "").lower()
    host = p.hostname
    if scheme not in ("http", "https") or not host:
        raise ZielGesperrt(url, "scheme_not_allowed")
    freigegeben = ist_freigegeben(url)
    if scheme != "https" and not freigegeben:
        raise ZielGesperrt(url, "https_required")
    try:
        port = p.port or _vorgabe_port(scheme)
    except ValueError:
        raise ZielGesperrt(url, "invalid_port")
    try:
        ips = _aufloesen(host, port)
    except (socket.gaierror, UnicodeError, OSError):
        raise ZielGesperrt(url, "unresolvable")
    if not ips:
        raise ZielGesperrt(url, "unresolvable")
    if not freigegeben:
        for roh in ips:
            try:
                ip = ipaddress.ip_address(roh.split("%")[0])
            except ValueError:
                raise ZielGesperrt(url, "internal_address")
            if not _ip_oeffentlich(ip):
                raise ZielGesperrt(url, "internal_address")
    ip = ips[0]
    netloc_ip = f"[{ip}]" if ":" in ip else ip
    if p.port:
        netloc_ip = f"{netloc_ip}:{p.port}"
    verbinden = urlunsplit((scheme, netloc_ip, p.path, p.query, ""))
    kopf = {"Host": p.netloc.rsplit("@", 1)[-1]}
    ext = {"sni_hostname": host} if scheme == "https" else {}
    return verbinden, kopf, ext


async def ziel_pruefen_async(url: str) -> tuple:
    """Wie ``ziel_pruefen``; die DNS-Aufloesung laeuft im Thread, nicht in
    der Ereignisschleife."""
    import asyncio
    return await asyncio.to_thread(ziel_pruefen, url)


def _kopf(ziel: str, kette_eigen: bool) -> dict:
    return {"X-API-KEY": storage_api_key()} if kette_eigen and darf_schluessel_sehen(ziel) else {}


@asynccontextmanager
async def oeffne(client: httpx.AsyncClient, url: str, mit_schluessel: bool = False):
    """Streamt ``url`` und entscheidet je Sprung neu ueber den Schluessel.

    Nach einem Sprung ueber einen fremden Host gibt es keinen Schluessel mehr:
    Eine fremde Adresse darf nicht per Umleitung auf den eigenen Storage zeigen
    und so nicht oeffentliche Dateien lesen (Post 5782).

    Der Client muss ``follow_redirects=False`` haben. Raises PrivateDatei,
    GesperrteDatei, ZielGesperrt, ZuVieleUmleitungen oder httpx.HTTPStatusError.
    """
    ziel = url
    eigen = mit_schluessel
    for _ in range(MAX_SPRUENGE + 1):
        eigen = eigen and darf_schluessel_sehen(ziel)
        verbinden, kopf, ext = await ziel_pruefen_async(ziel)
        async with client.stream("GET", verbinden, headers={**kopf, **_kopf(ziel, eigen)},
                                 extensions=ext) as r:
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
        verbinden, kopf, ext = ziel_pruefen(ziel)
        with client.stream("GET", verbinden, headers={**kopf, **_kopf(ziel, eigen)},
                           extensions=ext) as r:
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
