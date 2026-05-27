#!/usr/bin/env python3
"""
api-ai-models-discovery.py
--------------------------
Discovers available AI models per provider and writes a machine-readable
cache the api-ai endpoints consume. Designed to be invoked daily by a
systemd timer (and on demand via POST /internal/notify-cli-update).

Per-provider strategy:

  Claude   — query Anthropic /v1/models (admin path needs ANTHROPIC_API_KEY,
             but we fall back to a stable hardcoded alias list since the
             CLI exposes only sonnet/opus/haiku regardless).

  Codex    — the OpenAI /v1/models endpoint LIES about which models a
             ChatGPT-subscription account can actually use. We do an
             active smoke-test loop: spawn `codex exec -c model="X"` for
             each candidate; a 400 'not supported when using Codex with
             a ChatGPT account' marks it rejected, a 200 marks it
             accepted. Subscription default (no --model) is always
             included if it works at all.

  Gemini   — client.models.list() filtered to those advertising
             `generateContent`. Optional smoke if we want stricter
             validation, but Google's list is reliable enough.

Output: /var/lib/api-ai/models.json (atomic via temp+rename).
Previous file is kept as models.prev.json for diffing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from socket import gethostname


def short_host() -> str:
    """Return the short federation host-key (e.g. ``arkserver`` instead of
    ``arkserver.arkturian.com``). Keeps our models.json `host` field aligned
    with Automation's orchestrator POST body, which uses the short form."""
    h = gethostname().lower().split(".")[0]
    # Future-proofing: if someone runs this on a host whose hostname doesn't
    # match the federation key (e.g. `aiserver.oneal.eu` → key=oneal),
    # allow an explicit override via env.
    return os.environ.get("API_AI_HOST_KEY") or h

OUT_DIR = Path(os.environ.get("API_AI_STATE_DIR", "/var/lib/api-ai"))
OUT_FILE = OUT_DIR / "models.json"
PREV_FILE = OUT_DIR / "models.prev.json"
LOG_FILE = Path(os.environ.get("API_AI_LOG", "/var/log/api-ai-maintenance.log"))


# ──────────────────────────────────────────────────────────── helpers ──

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{ts()} [discovery] {msg}\n"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(line)
    except Exception:
        pass
    sys.stderr.write(line)


def cli_version(binary: str) -> str | None:
    """Return the --version output of a CLI binary, or None if missing."""
    try:
        r = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        # Strip trailing newline, take first line
        return (r.stdout or r.stderr or "").strip().splitlines()[0] if (r.stdout or r.stderr) else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ─────────────────────────────────────────────────────────── claude ──

CLAUDE_KNOWN_ALIASES = ["sonnet", "opus", "haiku"]


def discover_claude() -> dict:
    """Claude CLI accepts model aliases (sonnet/opus/haiku). The full
    model-id list is hidden behind subscription tier — query when
    possible, otherwise return the aliases as ground-truth."""
    version = cli_version("claude")
    info = {
        "default": "sonnet",
        "available": CLAUDE_KNOWN_ALIASES,
        "cli_version": version,
        "discovery_method": "hardcoded-aliases",
    }
    # Optional: try /v1/models via SDK if ANTHROPIC_API_KEY is set
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=key)
            models = client.models.list(limit=50)
            ids = [m.id for m in models.data if hasattr(m, "id")]
            if ids:
                info["model_ids"] = sorted(ids)
                info["discovery_method"] = "anthropic-api"
        except Exception as e:
            log(f"claude: /v1/models lookup failed: {e}")
    return info


# ──────────────────────────────────────────────────────────── codex ──

CODEX_CANDIDATES = [
    # Subscription-allowed (verified working historically)
    None,                        # "no model flag" = subscription default
    # Names worth re-probing in case OpenAI opens subscription access
    "gpt-5", "gpt-5-codex", "gpt-5-chat", "gpt-5-chat-latest",
    "gpt-5-mini", "gpt-5-thinking",
    "o3", "o4-mini", "codex-mini", "codex-1",
    "gpt-4o", "gpt-4o-mini", "gpt-4.1",
]


def discover_codex() -> dict:
    """Smoke-test every candidate. ChatGPT-subscription Codex returns
    400 'not supported when using Codex with a ChatGPT account' for
    rejected models. Surviving names are recorded as available; the
    sentinel `None` (= no --model flag) is recorded as 'codex-default'."""
    version = cli_version("codex")
    available: list[str] = []
    rejected: list[str] = []
    info: dict = {
        "default": None,
        "available": available,
        "rejected": rejected,
        "cli_version": version,
        "discovery_method": "smoke-test",
        "subscription_locked": True,  # corrected to False below if any --model passes
    }
    if not version:
        log("codex: CLI not installed, skipping discovery")
        info["discovery_method"] = "skipped-no-cli"
        return info

    for model in CODEX_CANDIDATES:
        cmd = [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        label = model or "codex-default"
        if model:
            cmd.extend(["-c", f'model="{model}"'])
        cmd.append("pong")  # minimal prompt
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                env={**os.environ},
            )
            out = (r.stdout or "") + (r.stderr or "")
            if "is not supported when using Codex with a ChatGPT account" in out:
                rejected.append(label)
                continue
            # Any agent_message means it worked, even if the answer is short
            if '"type":"item.completed"' in out and "agent_message" in out:
                available.append(label)
                if model is not None:
                    info["subscription_locked"] = False
                if info["default"] is None and label == "codex-default":
                    info["default"] = "codex-default"
            else:
                # Inconclusive — record but not as available
                rejected.append(f"{label}?unclear")
        except subprocess.TimeoutExpired:
            rejected.append(f"{label}?timeout")
        except Exception as e:
            log(f"codex: probe {label} failed: {e}")
            rejected.append(f"{label}?error")

    # Ensure default field
    if "codex-default" in available and info["default"] is None:
        info["default"] = "codex-default"
    return info


# ─────────────────────────────────────────────────────────── gemini ──

def discover_gemini() -> dict:
    version = cli_version("gemini")
    info: dict = {
        "default": "gemini-2.5-flash",
        "available": [],
        "cli_version": version,
        "discovery_method": "google-genai-api",
    }
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        info["discovery_method"] = "no-api-key"
        return info
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        models = client.models.list()
        ids: list[str] = []
        for m in models:
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions and name:
                ids.append(name)
        info["available"] = sorted(set(ids))
        # If our default isn't in the list, fall back to the first
        # "gemini-*-flash" / -pro alias that IS in the list.
        if info["default"] not in info["available"]:
            for fallback in ["gemini-2.5-flash", "gemini-flash-latest",
                             "gemini-2.5-pro", "gemini-pro-latest"]:
                if fallback in info["available"]:
                    info["default"] = fallback
                    break
    except Exception as e:
        log(f"gemini: list failed: {e}")
        info["discovery_method"] = f"error: {type(e).__name__}"
    return info


# ─────────────────────────────────────────────────────── main / IO ──

def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print result to stdout instead of writing models.json")
    ap.add_argument("--skip-codex-smoke", action="store_true",
                    help="Skip the slow Codex smoke-test loop (~10×30s)")
    args = ap.parse_args()

    log("discovery start")

    # Move existing models.json → models.prev.json for diff alerting later
    if OUT_FILE.exists() and not args.dry_run:
        try:
            OUT_FILE.replace(PREV_FILE)
        except Exception as e:
            log(f"warn: could not rotate prev: {e}")

    result = {
        "updated_at": ts(),
        "host": short_host(),
        "providers": {
            "claude": discover_claude(),
            "gemini": discover_gemini(),
        },
    }
    if args.skip_codex_smoke:
        result["providers"]["codex"] = {
            "skipped": True,
            "reason": "--skip-codex-smoke flag",
            "cli_version": cli_version("codex"),
        }
    else:
        result["providers"]["codex"] = discover_codex()

    if args.dry_run:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    write_atomic(OUT_FILE, result)
    log(f"wrote {OUT_FILE} (codex={len(result['providers']['codex'].get('available', []))} "
        f"claude={len(result['providers']['claude'].get('available', []))} "
        f"gemini={len(result['providers']['gemini'].get('available', []))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
