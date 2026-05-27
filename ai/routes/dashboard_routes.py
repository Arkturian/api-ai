"""
Dashboard Routes
================

Operator status view — one quick glance to see which provider/CLI/model
is healthy across the federation.

  • GET /ai/status            → HTML page (browser-friendly)
  • GET /ai/status.json       → JSON snapshot (machines / monitoring)
  • POST /ai/status/smoketest → run live "hello" probes against each
                                provider and stream results
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_MODELS_JSON = Path("/var/lib/api-ai/models.json")
_LAST_RUN    = Path("/var/lib/api-ai/last_run")
_PROVIDERS   = ("claude", "codex", "gemini")
_ENDPOINTS   = {
    "claude": "http://127.0.0.1:8000/ai/claude",
    "codex":  "http://127.0.0.1:8000/ai/chatgpt",
    "gemini": "http://127.0.0.1:8000/ai/gemini",
}


def _read_models_json() -> dict[str, Any]:
    if not _MODELS_JSON.exists():
        return {"_missing": True}
    try:
        data = json.loads(_MODELS_JSON.read_text())
        data["_mtime"] = _MODELS_JSON.stat().st_mtime
        data["_age_seconds"] = int(time.time() - data["_mtime"])
        return data
    except Exception as e:
        return {"_error": str(e)}


def _read_last_run() -> str | None:
    if not _LAST_RUN.exists():
        return None
    try:
        return _LAST_RUN.read_text().strip()
    except Exception:
        return None


def _cli_version(cli: str) -> str:
    """Best-effort `<cli> --version`. Returns short string."""
    try:
        out = subprocess.run(
            [cli, "--version"],
            capture_output=True, text=True, timeout=6,
        )
        return (out.stdout or out.stderr).strip().split("\n")[0] or "(empty)"
    except FileNotFoundError:
        return "(not installed)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


async def _smoke_probe(provider: str, model: str | None) -> dict[str, Any]:
    """Send a tiny "hello" prompt to the provider's local endpoint."""
    url = _ENDPOINTS[provider]
    body = {"prompt": "Reply with one word: pong.", "model": model or ""}
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=20.0) as cx:
            r = await cx.post(url, json=body)
        latency_ms = int((time.time() - start) * 1000)
        ok = 200 <= r.status_code < 300
        body_preview = r.text[:160].replace("\n", " ")
        return {
            "provider": provider,
            "model": model,
            "ok": ok,
            "http": r.status_code,
            "latency_ms": latency_ms,
            "preview": body_preview,
        }
    except Exception as e:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "http": None,
            "latency_ms": int((time.time() - start) * 1000),
            "preview": f"(exception: {type(e).__name__}: {e})",
        }


def _build_snapshot(include_local_cli: bool = True) -> dict[str, Any]:
    models = _read_models_json()
    snap: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_json": models,
        "last_run_marker": _read_last_run(),
        "providers": {},
        "host_local": os.uname().nodename.split(".")[0].lower(),
    }
    providers_info = (models.get("providers") or {}) if not models.get("_missing") else {}
    for p in _PROVIDERS:
        info = providers_info.get(p, {}) or {}
        cli_ver_live = _cli_version(p) if include_local_cli else None
        snap["providers"][p] = {
            "default": info.get("default"),
            "available": info.get("available") or [],
            "cli_version_recorded": info.get("cli_version"),
            "cli_version_live": cli_ver_live,
            "subscription_locked": info.get("subscription_locked", False),
            "discovery_method": info.get("discovery_method"),
        }
    return snap


# ── JSON snapshot (no smoke tests — fast) ──────────────────────────────
@router.get("/status.json")
async def status_json() -> dict[str, Any]:
    return _build_snapshot(include_local_cli=True)


# ── Live smoke tests (slow — runs real API calls) ──────────────────────
@router.post("/status/smoketest")
async def status_smoketest() -> dict[str, Any]:
    snap = _build_snapshot(include_local_cli=False)
    tasks = []
    for p in _PROVIDERS:
        default = snap["providers"][p].get("default")
        tasks.append(_smoke_probe(p, default))
    results = await asyncio.gather(*tasks)
    snap["smoke"] = {r["provider"]: r for r in results}
    return snap


# ── HTML dashboard ─────────────────────────────────────────────────────
_HTML_TMPL = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>api-ai status — {host}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ box-sizing: border-box; }}
body {{ font: 14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#0e1118; color:#e6e9ef; margin:0; padding:24px; }}
h1 {{ font-size:20px; margin:0 0 4px 0; color:#fff; }}
h1 small {{ font-weight:normal; color:#7a8499; font-size:13px; }}
.meta {{ color:#7a8499; margin-bottom:24px; font-size:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }}
.card {{ background:#161a24; border:1px solid #232838; border-radius:8px; padding:16px; }}
.card h2 {{ margin:0 0 12px 0; font-size:16px; display:flex; align-items:center; gap:10px; }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.dot.g {{ background:#3ddc84; box-shadow:0 0 8px #3ddc84; }}
.dot.y {{ background:#ffcc4d; box-shadow:0 0 8px #ffcc4d; }}
.dot.r {{ background:#ff5e57; box-shadow:0 0 8px #ff5e57; }}
.dot.gray {{ background:#4a5168; }}
.row {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #232838; font-size:13px; }}
.row:last-child {{ border-bottom:none; }}
.row .k {{ color:#7a8499; }}
.row .v {{ color:#e6e9ef; font-family:ui-monospace,SF Mono,monospace; text-align:right; word-break:break-all; max-width:60%; }}
.row .v.warn {{ color:#ffcc4d; }}
.row .v.err {{ color:#ff5e57; }}
.models-list {{ margin-top:10px; padding:8px 12px; background:#0e1118; border-radius:6px; font-family:ui-monospace,SF Mono,monospace; font-size:11.5px; color:#9aa4b8; max-height:120px; overflow:auto; }}
.models-list .default {{ color:#3ddc84; font-weight:600; }}
.actions {{ margin-top:20px; }}
button {{ background:#1f2533; color:#e6e9ef; border:1px solid #2e3548; padding:8px 16px; border-radius:6px; cursor:pointer; font:inherit; }}
button:hover {{ background:#262d3f; }}
button:disabled {{ opacity:0.5; cursor:wait; }}
.smoke-result {{ margin-top:12px; padding:8px 12px; border-radius:6px; font-size:12px; font-family:ui-monospace,SF Mono,monospace; }}
.smoke-result.ok {{ background:#0f2e1c; color:#3ddc84; border:1px solid #1d4a31; }}
.smoke-result.err {{ background:#2e0f0f; color:#ff5e57; border:1px solid #4a1d1d; }}
.smoke-result.pending {{ background:#1f2533; color:#9aa4b8; }}
.banner {{ padding:10px 14px; border-radius:6px; margin-bottom:16px; font-size:13px; }}
.banner.warn {{ background:#2e2410; color:#ffcc4d; border:1px solid #5c4920; }}
.banner.err  {{ background:#2e0f0f; color:#ff5e57; border:1px solid #5c1d1d; }}
</style>
</head><body>
<h1>api-ai status <small>· host {host}</small></h1>
<div class="meta">snapshot {ts} · models.json {mjson_age} · last_run_marker: {last_run}</div>
{banner}
<div class="grid">{cards}</div>
<div class="actions">
  <button id="probe">▶  run live smoke-test</button>
  <span id="probestatus" style="margin-left:10px;color:#7a8499;font-size:12px;"></span>
</div>
<div id="smoke" class="grid" style="margin-top:16px;"></div>
<script>
document.getElementById('probe').addEventListener('click', async (ev) => {{
  const b = ev.target;
  const s = document.getElementById('probestatus');
  b.disabled = true; s.textContent = 'probing 3 providers…';
  try {{
    const r = await fetch('/ai/status/smoketest', {{method:'POST'}});
    const d = await r.json();
    const out = document.getElementById('smoke');
    out.innerHTML = '';
    for (const p of ['claude','codex','gemini']) {{
      const sm = d.smoke[p];
      const card = document.createElement('div'); card.className = 'card';
      card.innerHTML = `
        <h2><span class="dot ${{sm.ok?'g':'r'}}"></span>${{p}} smoke</h2>
        <div class="row"><span class="k">model</span><span class="v">${{sm.model||'(default)'}}</span></div>
        <div class="row"><span class="k">http</span><span class="v">${{sm.http??'-'}}</span></div>
        <div class="row"><span class="k">latency</span><span class="v">${{sm.latency_ms}} ms</span></div>
        <div class="smoke-result ${{sm.ok?'ok':'err'}}">${{(sm.preview||'').slice(0,160)}}</div>`;
      out.appendChild(card);
    }}
    s.textContent = 'done · ' + new Date().toLocaleTimeString();
  }} catch (e) {{
    s.textContent = 'failed: ' + e;
  }} finally {{
    b.disabled = false;
  }}
}});
</script>
</body></html>"""


def _provider_card(name: str, info: dict[str, Any]) -> str:
    available = info.get("available") or []
    default = info.get("default")
    cli_rec = info.get("cli_version_recorded") or "—"
    cli_live = info.get("cli_version_live") or "—"
    sub_locked = info.get("subscription_locked", False)
    method = info.get("discovery_method") or "—"

    # Status dot logic: green if has available models, yellow if locked
    # without models, red if no live CLI.
    if "(not installed)" in cli_live or "(error" in cli_live:
        dot = "r"
    elif not available:
        dot = "y"
    else:
        dot = "g"

    cli_diff = ""
    if cli_rec != "—" and cli_live not in ("—", "(not installed)", "(timeout)"):
        # Best-effort comparison — strip whitespace/etc and string-compare.
        if cli_rec.strip() != cli_live.strip():
            cli_diff = '<span class="v warn">drift</span>'

    models_html = ""
    if available:
        items = []
        for m in available:
            cls = "default" if m == default else ""
            items.append(f'<span class="{cls}">{m}</span>')
        models_html = '<div class="models-list">' + " · ".join(items) + "</div>"

    sub_lock_html = (
        '<div class="row"><span class="k">subscription_locked</span>'
        '<span class="v warn">true</span></div>'
        if sub_locked else ""
    )

    return f"""
<div class="card">
  <h2><span class="dot {dot}"></span>{name}</h2>
  <div class="row"><span class="k">default</span><span class="v">{default or "—"}</span></div>
  <div class="row"><span class="k">available</span><span class="v">{len(available)} models</span></div>
  <div class="row"><span class="k">cli (recorded)</span><span class="v">{cli_rec}</span></div>
  <div class="row"><span class="k">cli (live)</span><span class="v">{cli_live}</span>{cli_diff}</div>
  <div class="row"><span class="k">discovery</span><span class="v">{method}</span></div>
  {sub_lock_html}
  {models_html}
</div>
"""


@router.get("/status", response_class=HTMLResponse)
async def status_html() -> Response:
    snap = _build_snapshot(include_local_cli=True)
    models = snap["models_json"]

    banner = ""
    if models.get("_missing"):
        banner = '<div class="banner err">models.json is MISSING — /ai/models will serve fallback</div>'
    elif models.get("_error"):
        banner = f'<div class="banner err">models.json read error: {models["_error"]}</div>'
    elif models.get("_age_seconds", 0) > 25 * 3600:
        banner = f'<div class="banner warn">models.json is {models["_age_seconds"] // 3600}h old (>25h) — /ai/models will fall back</div>'

    cards = "".join(_provider_card(p, snap["providers"][p]) for p in _PROVIDERS)
    age = models.get("_age_seconds")
    mjson_age = f"{age//60}m ago" if age is not None else "missing"
    html = _HTML_TMPL.format(
        host=snap["host_local"],
        ts=snap["ts"],
        mjson_age=mjson_age,
        last_run=snap["last_run_marker"] or "—",
        banner=banner,
        cards=cards,
    )
    return HTMLResponse(content=html)
