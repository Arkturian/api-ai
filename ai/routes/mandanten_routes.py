"""#4135 (Plan 6567 A1.1/A1.5): KI-Zuweisung je Mandant lesen und setzen.

GET /ai/mandanten/{tenant}/ki  -- Zuweisung, Konto ohne Wert, Deckel, Verbrauch heute/Monat
PUT /ai/mandanten/{tenant}/ki  -- Zuweisung setzen (kein Geheimnis: der Schluessel liegt im
                                  Tresor der auth-api, hier steht nur sein Verweis)

Wer darf (R1, AuthAPI 26.11.):
- Menschen-JWT (`type == "user"`) mit `ai:zuweisung:verwalten` in
  `mandant_rechte[<tenant aus dem Pfad>]` -- Vergleich mit dem SCHLUESSEL der Map,
  nicht mit `tenant` (den tragen nur Dienst-Tokens);
- der Betreiber-Schluessel bzw. ein Superadmin (/me);
- Lesen zusaetzlich mit `ai:kosten:lesen`.
Der Betreiber selbst wird hier nicht zugewiesen: fuer ihn gilt KI_VERARBEITUNG.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Literal, Optional, Union

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ai.services import verarbeitung as v

logger = logging.getLogger("api-ai.mandanten")
router = APIRouter()

RECHT_ZUWEISUNG = "ai:zuweisung:verwalten"
_MANDANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class KontoWahl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anbieter: Literal["openai"]
    verweis: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    modell: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class Deckel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag_usd: float = Field(gt=0, le=1000)
    monat_usd: float = Field(gt=0, le=10000)


class Zuweisung(BaseModel):
    """Ohne Konto keine KI. `deckel` ist Pflicht, sobald ein Konto genannt ist."""
    model_config = ConfigDict(extra="forbid")
    verarbeitung: Union[Literal["keine"], KontoWahl]
    urteil: Literal["openai", "keine"] = "keine"
    deckel: Optional[Deckel] = None


def _mandant_pruefen(tenant: str) -> str:
    if not _MANDANT_RE.fullmatch(tenant or ""):
        raise HTTPException(status_code=422, detail={"error": "mandant_ungueltig"})
    return tenant


async def _darf(tenant: str, authorization: Optional[str], x_api_key: Optional[str], lesen: bool) -> str:
    import asyncio
    from ai import auth_guard as _ag
    if _ag.ist_betreiber(x_api_key):
        return "betreiber"
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail={"error": "authentication_required"})
    from ai.services import jwt_pruefung
    claims = await asyncio.to_thread(jwt_pruefung.pruefen, authorization) or {}   # 401/503
    rechte = claims.get("mandant_rechte")
    if claims.get("type") == "user" and isinstance(rechte, dict):
        liste = rechte.get(tenant)
        if isinstance(liste, list) and RECHT_ZUWEISUNG in liste:
            return str(claims.get("sub") or "admin-mandant")
    if lesen:
        perms = claims.get("permissions")
        if isinstance(perms, list) and "ai:kosten:lesen" in perms:
            return str(claims.get("sub") or "kosten")
    from ai.routes import verbrauch_routes as vr
    profil = await vr._profil(authorization)
    if profil is not None and profil.get("is_superadmin"):
        return "superadmin"
    raise HTTPException(status_code=403, detail={"error": "zuweisung_nicht_freigegeben",
                                                 "required": RECHT_ZUWEISUNG, "mandant": tenant})


def _datei(tenant: str):
    return v._mandanten_dir() / f"{tenant}.json"


def _lesen(tenant: str) -> Optional[dict]:
    try:
        d = json.loads(_datei(tenant).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail={"error": "verarbeitung_zuweisung_unlesbar", "mandant": tenant})


def _verbrauch(tenant: str) -> dict:
    """Geld und Mengen des Mandanten aus dem Kostenbuch: heute und laufender Monat."""
    from ai.services import kostenbuch as kb
    heute = datetime.now(kb.WIEN).date()
    out = {"heute": {"usd": 0.0, "buchungen": 0}, "monat": {"usd": 0.0, "buchungen": 0}}
    for tag in kb._tage(heute.replace(day=1), heute):
        for z in kb._jsonl(kb.verzeichnis() / f"{tag.isoformat()}.jsonl"):
            if z.get("tenant") != tenant:
                continue
            usd = kb._zahl(z.get("usd")) if z.get("art") == "geld" else 0.0
            for k in (["monat", "heute"] if tag == heute else ["monat"]):
                out[k]["usd"] += usd
                out[k]["buchungen"] += 1
    for k in out:
        out[k]["usd"] = round(out[k]["usd"], 6)
    return {**out, "waehrung": kb.WAEHRUNG}


def _ansicht(tenant: str, d: Optional[dict]) -> dict:
    if d is None:
        return {"mandant": tenant, "zugewiesen": False, "verarbeitung": None, "urteil": "keine",
                "deckel": None, "verbrauch": _verbrauch(tenant)}
    vb = d.get("verarbeitung")
    konto = None
    if isinstance(vb, dict):
        verweis = str(vb.get("verweis") or "")
        # Nur der Verweis (kein Geheimnis), gekuerzt; der Wert liegt im Tresor.
        konto = {"anbieter": vb.get("anbieter"), "modell": vb.get("modell"),
                 "verweis": verweis[:12] + ("…" if len(verweis) > 12 else "")}
    return {"mandant": tenant, "zugewiesen": True,
            "verarbeitung": "keine" if vb == "keine" else konto,
            "urteil": d.get("urteil") or "keine", "deckel": d.get("deckel"),
            "geaendert_am": d.get("geaendert_am"), "geaendert_von": d.get("geaendert_von"),
            "verbrauch": _verbrauch(tenant)}


@router.get("/mandanten/{tenant}/ki")
async def mandant_ki_lesen(tenant: str, authorization: Optional[str] = Header(default=None),
                           x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")):
    tenant = _mandant_pruefen(tenant)
    await _darf(tenant, authorization, x_api_key, lesen=True)
    import asyncio
    d = await asyncio.to_thread(_lesen, tenant)
    return JSONResponse(await asyncio.to_thread(_ansicht, tenant, d), headers={"Cache-Control": "no-store"})


@router.put("/mandanten/{tenant}/ki")
async def mandant_ki_setzen(tenant: str, body: Zuweisung, authorization: Optional[str] = Header(default=None),
                            x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY")):
    tenant = _mandant_pruefen(tenant)
    wer = await _darf(tenant, authorization, x_api_key, lesen=False)
    if tenant == v.betreiber():
        raise HTTPException(status_code=422, detail={
            "error": "betreiber_nicht_zuweisbar",
            "hint": "Fuer den Betreiber gilt KI_VERARBEITUNG der Instanz."})
    if isinstance(body.verarbeitung, KontoWahl) and body.deckel is None:
        raise HTTPException(status_code=422, detail={"error": "deckel_pflicht",
                                                     "hint": "Mit Konto sind tag_usd und monat_usd Pflicht."})
    if body.urteil == "openai" and not isinstance(body.verarbeitung, KontoWahl):
        raise HTTPException(status_code=422, detail={"error": "urteil_ohne_konto"})
    import asyncio
    daten = body.model_dump(mode="json")
    daten.update(geaendert_am=datetime.now().astimezone().isoformat(timespec="seconds"), geaendert_von=wer[:64])

    def schreiben():
        p = _datei(tenant)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
        with open(p.parent / "protokoll.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": daten["geaendert_am"], "mandant": tenant, "von": daten["geaendert_von"],
                                "verarbeitung": "keine" if body.verarbeitung == "keine" else "openai",
                                "urteil": body.urteil}, ensure_ascii=False) + "\n")
    await asyncio.to_thread(schreiben)
    logger.info("mandant %s: KI-Zuweisung gesetzt von %s (%s)", tenant, wer[:64],
                "keine" if body.verarbeitung == "keine" else "openai")
    return JSONResponse(await asyncio.to_thread(_ansicht, tenant, daten), headers={"Cache-Control": "no-store"})
