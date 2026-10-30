"""Modelllisten aus den installierten CLIs (Content #5740, 30.10.2026).

cloud-api erhebt je Knoten `codex debug models`, `agy models` und die
claude-Aliase ohne KI nach `cli-models.json` (cloud-api `cli_catalog.py`).
Diese Datei ist die Wahrheit ueber das, was die CLIs dieses Knotens
anbieten. api-ai liest sie und baut daraus die Provider-Eintraege fuer
claude, codex und gemini in derselben Form wie `models.json`, damit
`/ai/models` und die agy-Effort-Vorgabe nichts anderes lesen muessen.

Was die Datei nicht liefert, kommt weiter aus dem Basis-Eintrag
(`models.json`), solange es ihn gibt: `param_name`, `supported_via`,
`min_cli_version`, Notizen. grok steht nicht in der Datei und bleibt bei
`models.json`. Die API-Anbieter ohne CLI (minimax, deepseek, glm) liegen
versioniert in `ai/data/api_anbieter.json`.

Gibt es die Datei nicht (arkturian: keine cloud-api), ist sie unlesbar
oder aelter als die Frist, gilt alles wie vorher.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

STANDARD_PFAD = "/var/lib/cloud-api/cli-models.json"
API_ANBIETER_PFAD = Path(__file__).resolve().parents[1] / "data" / "api_anbieter.json"
_AGY_SLUG = re.compile(r"^(?P<familie>.+)-(?P<stufe>low|medium|high)$")
_STUFEN = ("low", "medium", "high", "xhigh", "max", "ultra")
_VERSION = re.compile(r"\d+(?:\.\d+)+")


def pfad() -> Path:
    return Path(os.getenv("CLI_MODELS_PATH") or STANDARD_PFAD)


def lesen(max_alter_s: int) -> Optional[dict]:
    """Datei lesen, wenn vorhanden, gueltig und juenger als die Frist."""
    p = pfad()
    try:
        if not p.exists():
            return None
        d = json.loads(p.read_text())
    except Exception as e:
        logger.warning("cli-models.json nicht lesbar (%s): %s", p, e)
        return None
    if not isinstance(d, dict) or not isinstance(d.get("clis"), dict):
        logger.warning("cli-models.json ohne 'clis' — unbekanntes Format, ignoriert")
        return None
    erhoben = d.get("erhoben_am")
    if not isinstance(erhoben, (int, float)) or time.time() - erhoben > max_alter_s:
        logger.warning("cli-models.json aelter als %d h — ignoriert", max_alter_s // 3600)
        return None
    return d


def _fassung(eintrag: dict) -> Optional[str]:
    m = _VERSION.search(str(eintrag.get("fassung") or ""))
    return m.group(0) if m else None


def codex_config(codex_home: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Was ein codex-Aufruf ohne Modell tatsaechlich bekommt: `model` und
    `model_reasoning_effort` aus der config.toml, mit der api-ai codex startet."""
    home = codex_home or os.getenv("CODEX_HOME") or "/home/alex/.codex"
    try:
        cfg = tomllib.loads((Path(home) / "config.toml").read_text())
    except Exception:
        return None, None
    m, e = cfg.get("model"), cfg.get("model_reasoning_effort")
    return (m if isinstance(m, str) and m else None), (e if isinstance(e, str) and e else None)


def _codex(eintrag: dict, basis: dict, config: Tuple[Optional[str], Optional[str]]) -> Optional[dict]:
    modelle = [m for m in eintrag.get("modelle") or [] if m.get("id") and m.get("sichtbar", True)]
    if not modelle:
        return None
    modelle.sort(key=lambda m: m.get("rang") if isinstance(m.get("rang"), int) else 10**6)
    ids = [m["id"] for m in modelle]
    cfg_modell, cfg_effort = config
    p = copy.deepcopy(basis)
    p["available"] = ids
    p["default"] = cfg_modell if cfg_modell in ids else ids[0]
    eff = p.setdefault("efforts", {})
    eff.setdefault("param_name", "effort")
    eff.setdefault("supported_via", "-c reasoning_effort= override")
    per_model = {m["id"]: list(m.get("aufwand") or []) for m in modelle}
    alle = {s for v in per_model.values() for s in v}
    eff["available"] = [s for s in _STUFEN if s in alle] + sorted(alle - set(_STUFEN))
    eff["per_model"] = per_model
    eff["per_model_default"] = {m["id"]: m.get("aufwand_vorgabe") for m in modelle if m.get("aufwand_vorgabe")}
    if cfg_effort:
        eff["default"] = cfg_effort
    return p


def _gemini(eintrag: dict, basis: dict) -> Optional[dict]:
    """agy fuehrt nur suffigierte Slugs; daraus Familien + Stufen. Nur
    gemini-*: /ai/gemini prueft nur die (Fremdmodelle in agy sind eine
    Frage des Regelwerks in cloud-api, nicht dieses Endpunkts)."""
    familien: Dict[str, list] = {}
    for m in eintrag.get("modelle") or []:
        t = _AGY_SLUG.match(str(m.get("id") or ""))
        if t and t["familie"].startswith("gemini-"):
            familien.setdefault(t["familie"], []).append(t["stufe"])
    if not familien:
        return None
    p = copy.deepcopy(basis)
    ids = list(familien)
    p["available"] = ids
    if p.get("default") not in ids:
        p["default"] = ids[0]
    p.setdefault("underlying_cli", "agy")
    eff = p.setdefault("efforts", {})
    eff.setdefault("param_name", "effort")
    eff.setdefault("supported_via", "slug_suffix_composition")
    eff["required"] = True
    per_model = {f: [s for s in _STUFEN if s in st] for f, st in familien.items()}
    eff["per_model"] = per_model
    eff["available"] = [s for s in _STUFEN if any(s in v for v in per_model.values())]
    # Die Vorgabe bleibt, was sie war (sonst medium). Passt sie einem
    # Modell nicht, entscheidet per_model in _agy_effort_vorgabe — nicht
    # ein stiller Wechsel der Vorgabe fuer alle.
    eff.setdefault("default", "medium")
    return p


def _claude(eintrag: dict, basis: dict) -> Optional[dict]:
    ids = [m["id"] for m in eintrag.get("modelle") or [] if m.get("id")]
    if not ids:
        return None
    p = copy.deepcopy(basis)
    p["available"] = ids
    if p.get("default") not in ids:
        p["default"] = "sonnet" if "sonnet" in ids else ids[0]
    return p


def api_anbieter() -> Dict[str, dict]:
    """minimax, deepseek, glm: keine CLI, Liste versioniert im Repo."""
    try:
        d = json.loads(API_ANBIETER_PFAD.read_text())
        return {k: v for k, v in (d.get("providers") or {}).items() if isinstance(v, dict)}
    except Exception as e:
        logger.warning("api_anbieter.json nicht lesbar: %s", e)
        return {}


def zusammenfuehren(basis: Optional[Dict[str, dict]], cli: Optional[dict],
                    config: Optional[Tuple[Optional[str], Optional[str]]] = None) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Provider-Eintraege + Quelle je Provider.

    basis: `providers` aus models.json (oder None). cli: gelesene Datei
    (oder None). Die CLI-Erhebung ersetzt claude/codex/gemini; API-Anbieter
    kommen aus dem Repo; der Rest (grok) bleibt bei der Basis."""
    basis = basis or {}
    providers: Dict[str, dict] = {k: copy.deepcopy(v) for k, v in basis.items()}
    quellen = {k: "automation-curated" for k in providers}
    for k, v in api_anbieter().items():
        p = copy.deepcopy(providers.get(k) or {})
        p.update(v)
        p["discovery_method"] = "api-ai"
        providers[k] = p
        quellen[k] = "api-ai"
    if not cli:
        return providers, quellen
    clis = cli.get("clis") or {}
    bauer = {
        "codex": ("codex", lambda e, b: _codex(e, b, config if config is not None else codex_config())),
        "gemini": ("agy", _gemini),
        "claude": ("claude", _claude),
    }
    for name, (cli_name, bau) in bauer.items():
        eintrag = clis.get(cli_name)
        if not isinstance(eintrag, dict):
            continue
        neu = bau(eintrag, basis.get(name) or {})
        if not neu:
            continue
        neu["cli_version"] = _fassung(eintrag) or neu.get("cli_version")
        neu["cli_path"] = eintrag.get("binaer")
        neu["discovery_method"] = "cli"
        if eintrag.get("fehler"):
            neu["cli_error"] = eintrag["fehler"]
            neu["cli_stand_von"] = eintrag.get("stand_von")
        providers[name] = neu
        quellen[name] = "cli"
    return providers, quellen
