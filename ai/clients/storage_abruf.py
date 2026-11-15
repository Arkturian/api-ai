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
logger = logging.getLogger(__name__)
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


def ist_safety_pending(status: int, inhalt: bytes) -> bool:
    """451 mit code safety_pending: Storage prueft die Datei noch (ca. 1 min
    nach dem Hochladen). Vorlaeufig, kein Urteil — warten hilft."""
    if status != 451:
        return False
    try:
        import json as _json
        d = _json.loads(inhalt or b"{}")
        d = d.get("detail", d) if isinstance(d, dict) else {}
        return isinstance(d, dict) and d.get("code") == "safety_pending"
    except Exception:
        return False


class GesperrteDatei(Exception):
    """Storage verweigert die Datei wegen des Safety-Urteils (451).

    Die Signatur hebt nur die Zugriffspruefung auf, nicht das Urteil."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"quarantined: {ohne_anfrage(url)}")


class SicherheitAusstehend(GesperrteDatei):
    """451 safety_pending: noch keine Freigabe, kein Urteil. Wer kann, wartet."""


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


def signatur_urspruenge() -> set:
    """Urspruenge, deren Links eine Storage-Signatur (`exp`/`kid`/`sig`)
    tragen koennen. NUR fuer die Idempotenz-Normalisierung, nie fuer den
    Schluessel: wer hier steht, bekommt dadurch keinen Schluessel.

    Die eigenen Storage-Urspruenge plus die Namen, unter denen Storage
    signierte Links ausgibt (STORAGE_SIGNATUR_HOSTS, Vorgabe der
    kanonische `api-storage.arkturian.com`). Am Ziel gefunden 14.11.:
    arkserver hat STORAGE_API_URL=api-storage.arkserver…, die signierten
    Links lauten auf den kanonischen Namen. Ohne diese Liste blieb die
    Signatur im Hash, und der Neuanlauf von arkturian-daily-2026-11-14-bild
    scheiterte an request_id_payload_mismatch (Post 5771)."""
    urspruenge = set(_storage_urspruenge())
    roh = os.getenv("STORAGE_SIGNATUR_HOSTS", "api-storage.arkturian.com") or ""
    for eintrag in roh.split(","):
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


SAFETY_WARTEN_S = float(os.getenv("REFERENZ_SAFETY_WARTEN_S", "90"))


def ausstehend_detail(url: str) -> dict:
    """Antwortkoerper, wenn die Pruefung auch nach dem Warten nicht fertig ist."""
    return {"error": "reference_safety_pending", "code": "safety_pending", "url": ohne_anfrage(url),
            "retry_after_s": 30,
            "hint": "Storage prueft die frisch hochgeladene Datei noch (meist ~1 min). Gleich erneut senden."}


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
    "Nur https-Adressen auf Port 443 und oeffentliche Ziele, nicht auf die "
    "eigenen Knoten der Foederation. Interne Quellen gibt die "
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


# Eigene oeffentliche Knoten-Adressen (#3033, Befund 108): nach Bereich
# "global", aber eine Anfrage dorthin landet auf unseren eigenen Hosts. Dort
# erreicht sie Dienste, die ohne nginx auf 0.0.0.0 lauschen, oder nginx-
# Freigaben, die diesen Adressen trauen. Gleiche Liste wie
# share-proxy/safe_fetch.php, dazu tommy und arkturians IPv6 (Stand Cloud,
# 15.11.). pdrei ist abgemeldet, seine alte Adresse 172.104.235.73 gehoert
# inzwischen jemand anderem (#2595) und steht deshalb NICHT hier.
# Ergaenzen ohne Release: AI_EIGENE_IPS (Komma-Liste, IPs oder Netze).
_EIGENE_IPS = (
    "192.46.232.130", "2a01:7e01::f03c:94ff:feb3:d807",   # arkturian (Hub)
    "188.65.73.164",                                        # arkserver, mac-NAT
    "62.225.187.68", "62.225.187.66",                       # oneal
    "217.149.167.157",                                      # tommy (NAT)
)
# IPv6-Praefixe, die eine beliebige IPv4-Adresse einbetten: NAT64 (RFC 6052,
# RFC 8215). Python zaehlt 64:ff9b::/96 als global.
_V4_EINBETTEND = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"),
                  ipaddress.ip_network("2002::/16"))
_lokale_cache: Optional[list] = None


def _lokale_adressen() -> list:
    """Adressen der eigenen Schnittstellen (IPv4 per ioctl, IPv6 aus /proc)."""
    global _lokale_cache
    if _lokale_cache is not None:
        return _lokale_cache
    out = []
    try:
        import fcntl
        import struct
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            for _i, name in socket.if_nameindex():
                try:
                    roh = fcntl.ioctl(sk.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))
                    out.append(socket.inet_ntoa(roh[20:24]))
                except OSError:
                    pass
    except Exception:
        pass
    try:
        with open("/proc/net/if_inet6") as fh:
            for zeile in fh:
                h = zeile.split()[0]
                out.append(str(ipaddress.IPv6Address(int(h, 16))))
    except Exception:
        pass
    _lokale_cache = out
    return out


def _eigene_netze() -> list:
    netze = []
    roh = list(_EIGENE_IPS) + [x.strip() for x in (os.getenv("AI_EIGENE_IPS", "") or "").split(",")]
    for eintrag in roh + _lokale_adressen():
        if not eintrag:
            continue
        try:
            netze.append(ipaddress.ip_network(eintrag, strict=False))
        except ValueError:
            logger.warning("AI_EIGENE_IPS: %r ist keine Adresse", eintrag)
    return netze


def _als_v4(ip):
    """::ffff:a.b.c.d und ::a.b.c.d (IPv4-kompatibel, veraltet) -> IPv4."""
    if ip.version == 6:
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        n = int(ip)
        if n >> 32 == 0 and n > 1:
            return ipaddress.IPv4Address(n & 0xFFFFFFFF)
    return ip


def _eingebettet(ip):
    """Wie _als_v4, dazu NAT64 (letzte 32 Bit) und 6to4 (Bits 16-48)."""
    ip = _als_v4(ip)
    if ip.version == 6:
        if ip in _V4_EINBETTEND[0]:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if ip.sixtofour:
            return ip.sixtofour
    return ip


def ist_eigene_adresse(ip) -> bool:
    """Eigene Knoten oder eigene Schnittstelle, auch eingebettet in IPv6.
    Nur oeffentliche Bereiche zaehlen hier; Schleife und LAN meldet die
    Bereichspruefung als internal_address."""
    ip = _eingebettet(ip)
    if not (ip.is_global and not ip.is_unspecified):
        return False
    return any(ip.version == n.version and ip in n for n in _eigene_netze())


def _bereich_oeffentlich(ip) -> bool:
    ip = _als_v4(ip)
    if ip.version == 6 and any(ip in n for n in _V4_EINBETTEND):
        return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved and not ip.is_unspecified


def _ip_oeffentlich(ip) -> bool:
    return not ist_eigene_adresse(ip) and _bereich_oeffentlich(ip)


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
            if ist_eigene_adresse(ip):
                raise ZielGesperrt(url, "own_node_address")
            if not _bereich_oeffentlich(ip):
                raise ZielGesperrt(url, "internal_address")
        if port != 443:
            # Wie storage-api safe_fetch (#3033): fremde Ziele nur auf 443.
            # Ein oeffentlicher Host auf 22/6379/5432 ist nie eine Bildquelle.
            raise ZielGesperrt(url, "port_not_allowed")
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
                if ist_safety_pending(451, await r.aread()):
                    raise SicherheitAusstehend(url)
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
                if ist_safety_pending(451, r.read()):
                    raise SicherheitAusstehend(url)
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
