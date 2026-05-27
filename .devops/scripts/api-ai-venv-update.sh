#!/usr/bin/env bash
# api-ai-venv-update.sh
# ---------------------
# Updates the Python SDKs api-ai depends on inside its venv. Designed
# to run as a systemd-timer step on any host that hosts api-ai.
#
# Handles silent SDK breaks like:
#   • google-generativeai → google-genai migration (2026-05-22)
#   • openai SDK adding new transcribe params
#   • anthropic SDK schema changes
#
# Idempotent: if the venv path doesn't exist on the host, exits 0 cleanly
# so the same shipping unit can run on hosts without api-ai installed.

set -uo pipefail

VENV="${API_AI_VENV:-/var/www/api-ai.arkturian.com/venv}"
LOG="${API_AI_LOG:-/var/log/api-ai-maintenance.log}"
PIP="$VENV/bin/pip"

# Packages we explicitly track. Pinned to the upstream names — not the
# distro PyPI mirrors. Order matters only in that google-genai must come
# after the legacy google-generativeai is replaced (we do not pin
# google-generativeai because we deliberately migrated off of it).
PACKAGES=(
  google-genai          # current Gemini SDK (replaces google-generativeai)
  google-generativeai   # legacy SDK still imported in some code paths; keep current
  openai                # OpenAI SDK (whisper / chatgpt / etc.)
  anthropic             # Anthropic SDK (used for /v1/models discovery)
  deep-translator       # Google-Translate wrapper used by /ai/translate
)

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf '%s [venv-update] %s\n' "$(ts)" "$*" >> "$LOG"; }

if [[ ! -x "$PIP" ]]; then
  # No api-ai on this host — skip silently. The unit can be installed
  # on every federation node and only the relevant ones do real work.
  log "skip: no venv at $VENV"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"

log "begin update in $VENV"

# Capture current versions
declare -A BEFORE
for pkg in "${PACKAGES[@]}"; do
  ver=$("$PIP" show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')
  BEFORE[$pkg]="${ver:-not-installed}"
done

# Upgrade pip itself (quiet — minor version bumps are noise)
"$PIP" install --quiet --upgrade pip 2>>"$LOG" || log "warn: pip self-upgrade failed (continuing)"

# Bulk upgrade. --upgrade-strategy eager would re-resolve transitive
# deps; we keep it default to avoid surprise breaks across packages.
"$PIP" install --quiet --upgrade "${PACKAGES[@]}" 2>>"$LOG"
rc=$?
if (( rc != 0 )); then
  log "error: pip install exit $rc"
  exit "$rc"
fi

# Diff
changed=0
for pkg in "${PACKAGES[@]}"; do
  after=$("$PIP" show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')
  before="${BEFORE[$pkg]}"
  if [[ "$before" != "$after" ]]; then
    log "  $pkg: $before → ${after:-removed}"
    ((changed++))
  fi
done

if (( changed == 0 )); then
  log "no version changes (all $((${#PACKAGES[@]})) packages already current)"
else
  log "$changed package(s) updated — consider restarting api-ai.service if a runtime-loaded module changed"
fi

log "end update (rc=0)"
exit 0
