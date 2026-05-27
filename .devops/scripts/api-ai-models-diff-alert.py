#!/usr/bin/env python3
"""
api-ai-models-diff-alert.py
---------------------------
Compares /var/lib/api-ai/models.json against /var/lib/api-ai/models.prev.json
and sends a Telegram alert on meaningful changes:

  • New provider became available / removed
  • Model added or removed from available list
  • Default model changed (defensive — usually means upstream
    deprecated the previous default)
  • CLI version bumped (informational)

No-op if either file is missing or the diff is empty. Designed to run
*after* api-ai-models-discovery.py as a systemd unit step.

Env vars needed:
  TELEGRAM_BOT_TOKEN       — Bot to send from
  TELEGRAM_ADMIN_CHAT_ID   — Chat to send to
Both are loaded by systemd via EnvironmentFile (api-ai's .env file).
Silent no-op if either is missing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed in the running interpreter", file=sys.stderr)
    sys.exit(0)

STATE = Path(os.environ.get("API_AI_STATE_DIR", "/var/lib/api-ai"))
LOG = Path(os.environ.get("API_AI_LOG", "/var/log/api-ai-maintenance.log"))
CUR = STATE / "models.json"
PREV = STATE / "models.prev.json"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{ts()} [diff-alert] {msg}\n"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(line)
    except Exception:
        pass
    sys.stderr.write(line)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        log(f"warn: could not parse {path}: {e}")
        return None


def diff_providers(prev: dict, cur: dict) -> list[str]:
    """Return a list of human-readable change lines (HTML-safe)."""
    lines: list[str] = []
    p_provs = (prev or {}).get("providers", {})
    c_provs = (cur or {}).get("providers", {})

    # Provider-level add/remove
    for added in sorted(set(c_provs) - set(p_provs)):
        lines.append(f"➕ provider <b>{added}</b> appeared")
    for removed in sorted(set(p_provs) - set(c_provs)):
        lines.append(f"➖ provider <b>{removed}</b> disappeared")

    # Per-provider field diffs
    for name in sorted(set(p_provs) & set(c_provs)):
        p = p_provs[name] or {}
        c = c_provs[name] or {}

        # CLI version
        pv, cv = p.get("cli_version"), c.get("cli_version")
        if pv != cv and (pv or cv):
            lines.append(f"🆙 <b>{name}</b> CLI: <code>{pv or '?'}</code> → <code>{cv or '?'}</code>")

        # Default model change
        pd, cd = p.get("default"), c.get("default")
        if pd != cd:
            lines.append(f"🎯 <b>{name}</b> default: <code>{pd}</code> → <code>{cd}</code>")

        # Subscription-lock state
        ps, cs = p.get("subscription_locked"), c.get("subscription_locked")
        if ps != cs and (ps is not None or cs is not None):
            lines.append(f"🔒 <b>{name}</b> subscription_locked: {ps} → {cs}")

        # Available-list deltas
        pa = set(p.get("available") or [])
        ca = set(c.get("available") or [])
        added = sorted(ca - pa)
        removed = sorted(pa - ca)
        if added:
            lines.append(f"➕ <b>{name}</b> new models: {', '.join(f'<code>{m}</code>' for m in added)}")
        if removed:
            lines.append(f"➖ <b>{name}</b> removed models: {', '.join(f'<code>{m}</code>' for m in removed)}")

    return lines


def main() -> int:
    cur = load(CUR)
    prev = load(PREV)
    if cur is None:
        log(f"current models.json missing at {CUR} — nothing to compare")
        return 0
    if prev is None:
        log("no prev — first run, skipping alert")
        return 0

    diffs = diff_providers(prev, cur)
    if not diffs:
        log("no meaningful changes")
        return 0

    bot = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
    if not bot or not chat:
        log(f"changes detected but no Telegram creds in env — would have sent:\n  " + "\n  ".join(diffs))
        return 0

    host = cur.get("host", "?")
    header = f"🛠 <b>api-ai models.json changed on {host}</b>\n<i>{cur.get('updated_at', ts())}</i>\n"
    body = "\n".join(diffs)
    full = header + "\n" + body
    if len(full) > 3500:  # Telegram cap ~4096; leave room for parse-mode HTML
        full = full[:3500] + "\n…(truncated)"

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                json={"chat_id": chat, "text": full, "parse_mode": "HTML"},
            )
        if r.status_code != 200:
            log(f"telegram non-200: {r.status_code} {r.text[:200]}")
            return 0  # don't fail the whole timer step on telegram problems
        log(f"alerted on {len(diffs)} change(s)")
    except Exception as e:
        log(f"telegram send failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
