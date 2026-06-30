"""
Text AI Routes
==============

Endpoints for text-based AI models:
- Claude (Anthropic)
- ChatGPT (OpenAI)
- Gemini (Google)
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import asyncio
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────────────────────────────
# Concurrency caps — keep the host from melting.
#
# Each CLI subprocess is a fresh node process at ~350 MB resident; on a
# 3.8 GB host shared with 30 other services we cannot let an unbounded
# fan-out of /ai/<provider> requests spin up unlimited subprocesses in
# parallel. Without these caps, 5 concurrent calls swap-thrash and
# load-avg climbs to >30 (verified incident 2026-05-30).
#
# Per-provider semaphores. Defaults tuned for a small box; override via
# env if running on something beefier. Worst-case parallel CLIs:
#   gemini 2 + claude 1 + codex 1 = 4 × ~350MB ≈ 1.4 GB resident
# Leaves ~2.4 GB for uvicorn workers + the rest of the box.
#
# When the semaphore is saturated, requests wait up to ACQUIRE_TIMEOUT
# seconds for a slot and then return 503 with a Retry-After so callers
# back off instead of piling up more pending requests.
_CLI_MAX = {
    "gemini": int(os.getenv("API_AI_CLI_MAX_GEMINI", "2")),
    "claude": int(os.getenv("API_AI_CLI_MAX_CLAUDE", "1")),
    "codex":  int(os.getenv("API_AI_CLI_MAX_CODEX",  "1")),
}
_CLI_SEM: dict[str, asyncio.Semaphore] = {}
_ACQUIRE_TIMEOUT = float(os.getenv("API_AI_CLI_ACQUIRE_TIMEOUT", "60"))


def _get_cli_semaphore(provider: str) -> asyncio.Semaphore:
    """Lazy-create per-provider semaphore (must be inside an event loop)."""
    sem = _CLI_SEM.get(provider)
    if sem is None:
        sem = asyncio.Semaphore(_CLI_MAX.get(provider, 1))
        _CLI_SEM[provider] = sem
    return sem


def _run_cli_with_pgid(cmd, env, timeout=300, cwd="/"):
    """Run a CLI subprocess as its own process-group leader.

    Two reasons we don't just use ``subprocess.run`` here:

    1. ``start_new_session=True`` makes the child its own pgid leader, so
       a timeout kill can take out the whole group via ``os.killpg`` —
       gemini-cli spawns helper procs (model-router, telemetry) that
       would otherwise survive a plain ``proc.kill()``.

    2. On TimeoutExpired we ``killpg(SIGKILL)`` the whole group, then
       reap, so orphan node procs cannot accumulate.

    KillMode=control-group on the systemd unit (default for Type=simple)
    handles the SERVICE-level restart cleanup separately; this helper
    covers WORKER-level deaths (uvicorn child crash, multiprocessing
    fork weirdness) where systemd doesn't notice.

    Returns a CompletedProcess with the same shape ``subprocess.run``
    would have. Raises ``subprocess.TimeoutExpired`` on timeout after
    cleanup, same as ``subprocess.run``.
    """
    import signal as _signal
    import subprocess
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        text=True,
        start_new_session=True,  # own pgid for clean group-kill
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Take out the whole process group, not just the direct child
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Reap so we don't leak zombies
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        raise
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _check_api_billing_gate(confirmed: Optional[bool], endpoint: str) -> None:
    """Reject the request unless:
      1. caller explicitly opted in to GCP-billed Gemini API via
         ``confirm_api_billing: true`` in the request body, AND
      2. the monthly cap has not been reached.

    Cap mechanics: see ``ai.services.cost_tracker``. Per-host budget is
    half of Alex' total monthly cap (15 EUR), since we run on two hosts
    and don't (yet) have a federation-coherent shared counter — the
    approximation reaches at-most 15 EUR total in the worst case where
    one host alone hits its cap.

    Raises:
      HTTPException(403) when the body flag is missing — protects against
        accidental API spending from a misconfigured caller.
      HTTPException(429) when the monthly cap is reached — hard cutoff for
        all callers including those who sent confirm_api_billing=true.
        See the cost-incident postmortem (Mai 2026, 209 EUR) for context.
    """
    from ..services.cost_tracker import cost_tracker

    if not confirmed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "api_billing_confirmation_required",
                "endpoint": endpoint,
                "hint": ("This path hits the Gemini API (GCP-billed, not "
                         "subscription). Send `confirm_api_billing: true` in "
                         "the request body to acknowledge billing exposure."),
            },
        )

    if cost_tracker.should_block_request():
        status = cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_api_cap_reached",
                "endpoint": endpoint,
                "spent_eur": round(status.get("total_cost_eur", 0.0), 2),
                "budget_eur": status.get("monthly_budget_eur"),
                "hint": ("Per-host monthly cap reached. Cap resets at the "
                         "start of the next calendar month. Subscription "
                         "endpoints (/ai/claude, /ai/chatgpt) are unaffected."),
            },
        )


def _download_storage_images(prompt_text):
    # Scan the prompt for storage media URLs, download each to a HOME-rooted
    # vision-tmp dir; return [(url, local_path)]. Lets the CLI read a LOCAL
    # file instead of WebFetching the URL -> deterministic for codex/gemini.
    # X-API-KEY bypasses the storage quarantine gate (re-scan of flagged imgs).
    # Caller MUST delete the local files after the CLI call.
    #
    # Path-under-cwd, NOT /tmp and NOT just under $HOME: gemini-cli v0.46+
    # enforces a workspace sandbox scoped to the **cwd** (not $HOME) even
    # with --no-sandbox --yolo --skip-trust. Anything outside the cwd
    # resolves "outside the allowed workspace", the internal read_file
    # tool-call silently fails, and the model answers from the text
    # prompt alone — confident hallucinated species. Verified 2026-06-15
    # via SWFME's Knowledge pipeline (bee image -> "Gentiana verna @1.0").
    # Service runs CLI with cwd=$HOME/.aiapi-neutral, so images must go
    # UNDER that directory (.aiapi-neutral/.vision/) so the workspace
    # check passes and the @path attaches the image as a multimodal part.
    import re, uuid, pwd
    try:
        import httpx
    except Exception:
        return []
    urls = [u.rstrip('.,;:)]}>') for u in re.findall(r'https?://\S+', prompt_text or '') if '/storage/media/' in u]
    out = []
    cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
    img_dir = os.path.join(cli_home, ".aiapi-neutral", ".vision")
    try:
        os.makedirs(img_dir, exist_ok=True)
    except OSError:
        img_dir = "/tmp"
    for url in dict.fromkeys(urls):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as c:
                r = c.get(url, headers={"X-API-KEY": os.getenv("STORAGE_API_KEY", "Inetpass1")})
                r.raise_for_status()
            ct = r.headers.get('content-type', '')
            ext = 'png' if 'png' in ct else ('webp' if 'webp' in ct else 'jpg')
            p = os.path.join(img_dir, 'aiimg_' + uuid.uuid4().hex[:10] + '.' + ext)
            with open(p, 'wb') as fh:
                fh.write(r.content)
            out.append((url, p))
            logger.info('localized storage image -> ' + p)
        except Exception as e:
            logger.warning('localize: download failed for ' + url + ': ' + str(e))
    return out


def _check_minimax_billing_gate(confirmed: Optional[bool], endpoint: str) -> None:
    """MiniMax-side twin of ``_check_api_billing_gate``.

    Same default-deny semantics, separate cost-tracker (25 EUR/month cap).
    Reject the request unless the caller explicitly opted in via
    ``confirm_api_billing: true`` AND the MiniMax monthly cap has not
    been reached.

    Raises:
      HTTPException(403) when the body flag is missing.
      HTTPException(429) when the cap is reached — hard cutoff for ALL
        callers including those who sent confirm_api_billing=true.
    """
    from ..services.minimax_cost_tracker import minimax_cost_tracker

    if not confirmed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "api_billing_confirmation_required",
                "endpoint": endpoint,
                "provider": "minimax",
                "hint": ("This path hits the MiniMax API (pay-as-you-go, "
                         "billed against MINIMAX_MULTIMODAL_API_KEY). Send "
                         "`confirm_api_billing: true` in the request body to "
                         "acknowledge billing exposure."),
            },
        )

    if minimax_cost_tracker.should_block_request():
        status = minimax_cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_api_cap_reached",
                "endpoint": endpoint,
                "provider": "minimax",
                "spent_eur": round(status.get("total_cost_eur", 0.0), 2),
                "budget_eur": status.get("monthly_budget_eur"),
                "hint": ("MiniMax monthly cap reached. Cap resets at the "
                         "start of the next calendar month. Subscription "
                         "endpoints (/ai/claude, /ai/chatgpt) are unaffected."),
            },
        )


def _check_deepseek_billing_gate(confirmed: Optional[bool], endpoint: str) -> None:
    """DeepSeek-side twin of ``_check_minimax_billing_gate``.

    Same default-deny + cap-check semantics. Cap is 25 EUR/month by
    default (env: ``DEEPSEEK_MONTHLY_BUDGET_EUR``). Separate envelope
    from Gemini (15), MiniMax (25), OpenAI (50) — DeepSeek burn doesn't
    block other providers and vice versa.

    Raises:
      HTTPException(403) when the body flag is missing.
      HTTPException(429) when the cap is reached.
    """
    from ..services.deepseek_cost_tracker import deepseek_cost_tracker

    if not confirmed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "api_billing_confirmation_required",
                "endpoint": endpoint,
                "provider": "deepseek",
                "hint": ("DeepSeek V4 is pay-as-you-go billed against "
                         "DEEPSEEK_API_KEY (separate from MiniMax + OpenAI "
                         "+ GCP caps). Approx: deepseek-v4-flash ≈ "
                         "$0.07/1M input + $0.27/1M output, deepseek-v4-pro "
                         "≈ $0.27/1M input + $1.10/1M output. Send "
                         "`confirm_api_billing: true` to acknowledge."),
            },
        )

    if deepseek_cost_tracker.should_block_request():
        status = deepseek_cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_api_cap_reached",
                "endpoint": endpoint,
                "provider": "deepseek",
                "spent_eur": round(status.get("total_cost_eur", 0.0), 2),
                "budget_eur": status.get("monthly_budget_eur"),
                "hint": ("DeepSeek monthly cap reached. Cap resets at the "
                         "start of the next calendar month."),
            },
        )


async def _acquire_cli_slot(provider: str):
    """Wait up to ACQUIRE_TIMEOUT for a CLI slot; raise 503 on overflow."""
    sem = _get_cli_semaphore(provider)
    started = time.monotonic()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=_ACQUIRE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(
            f"{provider} CLI semaphore saturated for {_ACQUIRE_TIMEOUT}s "
            f"(max={_CLI_MAX.get(provider)}); rejecting with 503"
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": (
                    f"{provider} CLI is saturated. Server is protecting the "
                    "host from swap-thrash. Retry with backoff."
                ),
                "code": "cli_capacity",
                "max_concurrent": _CLI_MAX.get(provider),
                "queued_wait_seconds": round(time.monotonic() - started, 1),
                "retry_after": 5,
            },
            headers={"Retry-After": "5"},
        )
    return sem


# ──────────────────────────────────────────────────────────────────────
# Federation persona injection
#
# Background: the /ai/<provider> endpoints shell a CLI subprocess. Those
# CLIs authenticate via OAuth credentials resolved from the credentialed
# bot-home that the systemd unit points at:
#   claude  -> $CLAUDE_CONFIG_DIR/.credentials.json
#   codex   -> $CODEX_HOME/auth.json
#   gemini  -> $HOME/.gemini/oauth_creds.json
# A refresh loop keeps those tokens fresh. We therefore must NOT repoint
# CLAUDE_CONFIG_DIR/CODEX_HOME/HOME at a fresh per-endpoint persona dir —
# that would strip the credentials and 401 every call (the cred-drift
# landmine). Instead we fetch the *rendered persona string* from cloud-api
# and inject it as a system prompt:
#   claude -> --append-system-prompt <persona>   (alongside any caller system)
#   codex  -> prepended to the prompt text        (no --system-prompt flag)
#   gemini -> prepended to the prompt text         (no --system-prompt flag)
# This is federation-fragment-aware (toggle per endpoint via the virtual
# bot's disabled_fragments on cloud-api) yet credential-safe.
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "https://cloud-api.arkserver.arkturian.com")
PERSONAS_DIR = Path(os.getenv("AI_API_PERSONAS_DIR", "/var/lib/api-ai/personas"))
PERSONA_TTL_S = int(os.getenv("AI_API_PERSONA_TTL", "60"))
# endpoint token -> instruction filename rendered by cloud-api per agent
_PERSONA_INSTR = {"claude": "CLAUDE.md", "chatgpt": "AGENTS.md", "gemini": "GEMINI.md"}


def persona_dir_for(endpoint: str, variant: str) -> Path:
    return PERSONAS_DIR / f"api-ai-{endpoint}-{variant}"


async def get_persona_bundle(endpoint: str, variant: Optional[str]) -> dict:
    """Return ``{"rendered": str|None, "allowed_tools": list[str]}`` for
    ``api-ai-<endpoint>-<variant>``.

    Fetches cloud-api's ``/api/sessions/<vbot>/effective-prompt`` (which since
    the MCP-toolset rollout also carries ``allowed_tools``) and caches the full
    response JSON on disk (TTL ``PERSONA_TTL_S``). Disk cache survives worker
    restarts and is shared across uvicorn workers. Best-effort: returns an empty
    bundle when ``variant`` is falsy or the fetch fails, so the endpoint always
    proceeds (bare) rather than erroring.
    """
    import json as _json
    empty = {"rendered": None, "allowed_tools": []}
    if not variant:
        return empty
    import httpx
    vbot = f"api-ai-{endpoint}-{variant}"
    cache_dir = PERSONAS_DIR / vbot
    cache = cache_dir / "effective.json"

    def _parse(path: Path):
        try:
            d = _json.loads(path.read_text(encoding="utf-8"))
            return {
                "rendered": ((d.get("rendered") or "").strip() or None),
                "allowed_tools": d.get("allowed_tools") or [],
            }
        except Exception:
            return None

    # Serve from cache while fresh
    try:
        if cache.exists() and (time.time() - cache.stat().st_mtime) <= PERSONA_TTL_S:
            cached = _parse(cache)
            if cached is not None:
                return cached
    except OSError:
        pass

    # (Re)fetch from cloud-api
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{CLOUD_API_URL}/api/sessions/{vbot}/effective-prompt")
            r.raise_for_status()
            payload = r.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(_json.dumps(payload), encoding="utf-8")
        return {
            "rendered": ((payload.get("rendered") or "").strip() or None),
            "allowed_tools": payload.get("allowed_tools") or [],
        }
    except Exception as e:
        logger.warning(f"persona fetch failed for {vbot}: {e}; falling back to stale/none")
        stale = _parse(cache) if cache.exists() else None
        return stale if stale is not None else empty


def _load_host_mcp_servers() -> dict:
    """Return the host bot-home ``.claude.json`` ``mcpServers`` map (name ->
    server config WITH its auth ``headersHelper``). Located via
    CLAUDE_CONFIG_DIR so it matches the dir the CLI authenticates from."""
    import json as _json
    cfg_dir = os.getenv("CLAUDE_CONFIG_DIR") or os.getenv("HOME") or ""
    path = Path(cfg_dir) / ".claude.json"
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"could not read MCP servers from {path}: {e}")
        return {}

    def _find(obj):
        if isinstance(obj, dict):
            srv = obj.get("mcpServers")
            if isinstance(srv, dict) and srv:
                return srv
            for v in obj.values():
                r = _find(v)
                if r:
                    return r
        return {}

    return _find(data)


def build_mcp_config(persona_dir: Path, allowed_tools: list) -> Optional[Path]:
    """Materialise a claude ``--mcp-config`` file holding exactly the MCP
    servers referenced by ``allowed_tools`` (server name = the ``mcp__<server>__``
    prefix), sourced WITH auth from the host ``.claude.json``. Returns the path,
    or ``None`` when no servers resolve.

    Why explicit ``--mcp-config`` instead of the CLAUDE_CONFIG_DIR servers: the
    persona path uses ``--setting-sources project`` which strips user-scoped MCP
    servers (verified — knowledge/artrack disappear). An explicit ``--mcp-config``
    + ``--strict-mcp-config`` is independent of setting-sources, so persona
    suppression and MCP tools coexist. Credentials still load from
    CLAUDE_CONFIG_DIR/.credentials.json regardless.
    """
    import json as _json
    wanted = set()
    for t in (allowed_tools or []):
        if t.startswith("mcp__"):
            parts = t.split("__")
            if len(parts) >= 2 and parts[1]:
                wanted.add(parts[1])
    if not wanted:
        return None
    host = _load_host_mcp_servers()
    servers = {n: host[n] for n in wanted if n in host}
    if not servers:
        logger.warning(
            f"allowed_tools reference servers {sorted(wanted)} but none found in host .claude.json"
        )
        return None
    try:
        persona_dir.mkdir(parents=True, exist_ok=True)
        path = persona_dir / ".mcp.json"
        path.write_text(_json.dumps({"mcpServers": servers}), encoding="utf-8")
        return path
    except OSError as e:
        logger.warning(f"could not write mcp-config: {e}")
        return None


# Request/Response Models
class PromptText(BaseModel):
    """Nested text/images structure for legacy compatibility"""
    text: str
    images: Optional[List[str]] = None  # Base64 encoded images (legacy, not supported by CLI)
    image_paths: Optional[List[str]] = None  # Local file paths - Claude CLI reads these directly

class Prompt(BaseModel):
    prompt: Union[str, PromptText]  # Support both string and nested object
    system: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    conversation_history: Optional[List[Dict[str, str]]] = None
    image_paths: Optional[List[str]] = None  # Top-level image paths for convenience
    # Reasoning depth knob. Maps to claude `--effort`, codex
    # `-c model_reasoning_effort=`, gemini = currently not supported by
    # CLI (logged + ignored). Frontend should source the allowed values
    # from /ai/models providers_meta.<provider>.efforts.available.
    effort: Optional[str] = None
    # GCP-cost-incident hardening (2026-05-30): explicit opt-in flag required
    # for any path that hits the Gemini *API* (vision/transcribe SDK calls).
    # The /ai/gemini text path is hard-503'd until OAuth subscription is set
    # up — for that path this flag has no effect. For vision + transcribe-gemini
    # the call is rejected with 403 unless `true`, and 429'd if the monthly
    # cap is exceeded regardless of confirmation.
    confirm_api_billing: Optional[bool] = False
    # Federation persona injection (opt-in). When set (e.g. "default" /
    # "test"), the rendered persona of the virtual bot
    # ``api-ai-<endpoint>-<variant>`` is fetched from cloud-api and injected
    # as a system prompt (see ``get_persona_prompt``). None = no injection,
    # i.e. the historical "bare CLI" behaviour — kept as the default so
    # existing callers (storage/review/knowledge) are unaffected.
    persona_variant: Optional[str] = None


class AIResponse(BaseModel):
    response: str
    message: Optional[str] = None  # Alias for 'response' for legacy compatibility
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-populate message field from response if not provided
        if not self.message and self.response:
            self.message = self.response


# Placeholder for API key validation
def get_api_key():
    # TODO: Implement API key validation
    return "placeholder"


@router.post("/claude", response_model=AIResponse)
async def claude_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Claude AI endpoint via Claude Code CLI (claude -p)

    Features:
    - Uses logged-in Claude account (no API costs on your bill!)
    - Claude Code in print mode (-p) for non-interactive use
    - JSON output for token tracking
    - Supports --model parameter (sonnet, opus, haiku)
    - System prompt support via --system-prompt
    - Cost tracking at /ai/claude/cost-status

    Note: Claude CLI does NOT support images. For vision tasks, use /ai/gemini/vision instead.
    """
    import subprocess
    import asyncio
    import json as json_module
    from ..services.claude_cost_tracker import claude_cost_tracker

    try:
        # Extract prompt text
        prompt_text = ""

        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
            image_paths = prompt.image_paths or []
        else:
            prompt_text = prompt.prompt.text
            # Collect image paths from nested or top-level
            image_paths = prompt.prompt.image_paths or prompt.image_paths or []
            if prompt.prompt.images and not image_paths:
                # Legacy base64 images - Claude CLI can't handle these
                logger.warning(
                    f"Claude CLI does not support base64 images - ignoring {len(prompt.prompt.images)} images. "
                    f"Use image_paths with local file paths instead."
                )

        # Append image paths to prompt - Claude CLI will read and analyze them
        if image_paths:
            # image_paths may be local file paths OR https URLs (storage media).
            # Claude CLI reads local files and WebFetches URLs from the prompt,
            # so fold ALL refs in (old os.path.exists filter dropped URLs).
            paths_text = "\n".join(image_paths)
            prompt_text = f"{prompt_text}\n\nAnalysiere folgende Bilder:\n{paths_text}"
            logger.info(f"Added {len(image_paths)} image ref(s) to prompt for Claude CLI")

        logger.info(f"Calling claude -p with prompt length: {len(prompt_text)} chars")

        # Build CLI command with default model sonnet (cost-effective)
        # NOTE: --dangerously-skip-permissions is NOT compatible with
        # root/sudo (the CLI refuses to start). uvicorn here runs as
        # root, so we leave the flag off — in -p (print) mode there are
        # no tool calls anyway, so no permission prompts to bypass.
        _imgs = _download_storage_images(prompt_text)
        for _u, _p in _imgs:
            prompt_text = prompt_text.replace(_u, _p)
        if _imgs:
            logger.info("claude: localized %d storage image(s) to /tmp" % len(_imgs))
        cmd = ["claude", "-p", prompt_text, "--output-format", "json"]

        # Add model - default to sonnet (günstig), user can override with opus/haiku
        selected_model = model or "sonnet"
        cmd.extend(["--model", selected_model])

        # System prompt + federation persona (opt-in).
        #
        # When a persona is active it becomes the AUTHORITATIVE base system
        # prompt via --system-prompt, and we pass --setting-sources project to
        # suppress loading the user-level CLAUDE.md from CLAUDE_CONFIG_DIR (the
        # ~43KB bot-home federation identity) which would otherwise be loaded as
        # memory and override the small persona — verified: with a plain
        # --append-system-prompt the bot-home identity wins and the model still
        # claims to be AiApi with IACP tools. Credentials live in
        # .credentials.json and load regardless of --setting-sources, so OAuth
        # is unaffected (verified is_error=false). A caller-supplied system
        # prompt then layers ON TOP of the persona base via append.
        bundle = await get_persona_bundle("claude", prompt.persona_variant)
        persona = bundle["rendered"]
        allowed_tools = bundle["allowed_tools"]
        # CLEAN DEFAULT: always suppress the user-level CLAUDE.md from
        # CLAUDE_CONFIG_DIR (the bot-home identity, e.g. AiApi's 43KB federation
        # persona) so a bare endpoint call starts with NO inherited Vorwissen.
        # An ephemeral one-shot call only carries what the caller defines
        # (persona + tools). Credentials in .credentials.json load regardless of
        # --setting-sources (verified is_error=false).
        cmd.extend(["--setting-sources", "project"])
        if persona:
            cmd.extend(["--system-prompt", persona])
            if prompt.system:
                cmd.extend(["--append-system-prompt", prompt.system])
            logger.info(
                f"Injected persona api-ai-claude-{prompt.persona_variant} ({len(persona)} chars)"
            )
        elif prompt.system:
            cmd.extend(["--system-prompt", prompt.system])

        # MCP tool-gating (opt-in, bot-bound). When the virtual bot declares
        # allowed_tools, the call becomes an AGENTIC multi-turn run: we re-add
        # exactly those servers via an explicit --mcp-config (independent of
        # --setting-sources, which would otherwise strip them) and whitelist the
        # tools so no interactive permission prompt is needed. Bumps the timeout
        # since tool round-trips take longer than a single-shot inference.
        claude_cli_timeout = 300
        if allowed_tools and prompt.persona_variant:
            mcp_path = build_mcp_config(
                persona_dir_for("claude", prompt.persona_variant), allowed_tools
            )
            if mcp_path:
                cmd.extend([
                    "--mcp-config", str(mcp_path), "--strict-mcp-config",
                    "--allowedTools", ",".join(allowed_tools),
                ])
                claude_cli_timeout = int(os.getenv("API_AI_CLI_AGENTIC_TIMEOUT", "600"))
                logger.info(
                    f"Tool-gated agentic call: {len(allowed_tools)} allowed_tools, "
                    f"servers from {mcp_path}, timeout={claude_cli_timeout}s"
                )

        # Reasoning depth — claude CLI accepts: low, medium, high, xhigh, max
        if prompt.effort:
            cmd.extend(["--effort", prompt.effort])

        # Call claude -p in subprocess
        def run_claude_cli():
            import os
            import pwd
            import shlex

            # Check if running as root - if so, run claude as a non-root user to enable permission bypass
            cli_user = os.getenv("CLI_USER", "alex")  # Default to alex
            running_as_root = os.getuid() == 0

            # Determine HOME directory
            if running_as_root:
                try:
                    cli_home = pwd.getpwnam(cli_user).pw_dir
                except KeyError:
                    cli_home = f"/home/{cli_user}"
            else:
                cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir

            # Build environment
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            # Honor HOME + CLAUDE_CONFIG_DIR from the systemd unit if set;
            # only fall back to cli_home when nothing was provided. The
            # canonical credential lookup path is $CLAUDE_CONFIG_DIR/.credentials.json
            # — on arkserver the drop-in points this at a cloud-api bot
            # home so the auto-login + refresh loop manages the OAuth token.
            env.setdefault("HOME", cli_home)

            # IMPORTANT: Remove ANTHROPIC_API_KEY so Claude CLI uses OAuth credentials
            env.pop("ANTHROPIC_API_KEY", None)

            # Run claude in its own process-group so timeout cleanup
            # can take out helper procs alongside the main CLI. See
            # _run_cli_with_pgid for the orphan-defense rationale.
            result = _run_cli_with_pgid(cmd, env=env, timeout=claude_cli_timeout, cwd="/")
            return result

        # Concurrency cap: bound the number of in-flight claude CLI
        # subprocesses so we don't swap-thrash the host (see module top).
        sem = await _acquire_cli_slot("claude")
        try:
            result = await asyncio.to_thread(run_claude_cli)
        finally:
            sem.release()
            for _u, _p in _imgs:
                try:
                    os.remove(_p)
                except Exception:
                    pass

        # Log the result for debugging
        logger.info(f"Claude CLI returncode: {result.returncode}")
        logger.info(f"Claude CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stdout:
            logger.info(f"Claude CLI stdout preview: {result.stdout[:500]}")
        if result.stderr:
            logger.info(f"Claude CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        # Claude CLI returns errors in JSON stdout with is_error: true
        # So we need to parse JSON FIRST before checking returncode
        if raw_output:
            try:
                cli_response = json_module.loads(raw_output)

                # Check for error in JSON response (Claude returns is_error: true)
                if cli_response.get("is_error"):
                    error_msg = cli_response.get("result", "Unknown error")
                    logger.error(f"Claude CLI error (from JSON): {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Claude error: {error_msg}")

            except json_module.JSONDecodeError as e:
                # Not valid JSON - check if it's a plain error
                if result.returncode != 0:
                    error_msg = result.stderr or raw_output or "Unknown error from claude CLI"
                    logger.error(f"Claude CLI error (non-JSON): {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")

                # Otherwise treat as plain text response (unusual but possible)
                logger.warning(f"Claude CLI returned non-JSON output: {raw_output[:200]}")
                return AIResponse(
                    response=raw_output,
                    model="claude-cli",
                    tokens_used=None,
                    finish_reason="stop"
                )
        else:
            # No stdout - check stderr for errors
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from claude CLI (empty response)"
                logger.error(f"Claude CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Claude CLI error: {error_msg}")

            raise HTTPException(status_code=500, detail="Claude CLI returned empty response")

        # Track usage
        claude_cost_tracker.track_usage(cli_response)

        # Extract response text
        response_text = cli_response.get("result", "")

        # Calculate total tokens
        usage = cli_response.get("usage", {})
        tokens_used = (
            usage.get("input_tokens", 0) +
            usage.get("output_tokens", 0) +
            usage.get("cache_read_input_tokens", 0)
        )

        # Determine model used. When system context is cached (e.g. CLAUDE.md)
        # multiple models can appear in modelUsage — haiku for the cache layer
        # plus the requested model for the actual inference. Pick the one that
        # actually generated output tokens so the response shows the user-facing
        # model, not the cache-tier helper.
        model_usage = cli_response.get("modelUsage", {})
        if model_usage:
            model_name = max(
                model_usage.items(),
                key=lambda kv: kv[1].get("outputTokens", 0)
                              if isinstance(kv[1], dict) else 0
            )[0]
        else:
            model_name = "claude-cli"

        logger.info(
            f"Claude CLI response: {len(response_text)} chars, "
            f"${cli_response.get('total_cost_usd', 0):.4f}, "
            f"{tokens_used} tokens"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Claude CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("Claude CLI not found - is 'claude' installed?")
        raise HTTPException(status_code=500, detail="Claude CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Claude error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/claude/cost-status")
async def claude_cost_status():
    """
    Get current Claude CLI cost tracking status.

    Returns:
    - Current month's usage and costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational - Claude CLI uses your subscription.
    """
    from ..services.claude_cost_tracker import claude_cost_tracker
    return claude_cost_tracker.get_status()


@router.post("/chatgpt", response_model=AIResponse)
async def chatgpt_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    ChatGPT endpoint via OpenAI Codex CLI

    Features:
    - Uses logged-in ChatGPT account (no API costs on your bill!)
    - Codex CLI in exec mode for non-interactive use
    - JSONL output for response extraction
    - Default model: o4-mini (fast and cost-effective)
    - System prompt support
    - Cost tracking at /ai/chatgpt/cost-status

    Note: Uses your ChatGPT Plus/Pro subscription - no API billing.
    """
    import subprocess
    import asyncio
    import json as json_module
    from ..services.codex_cost_tracker import codex_cost_tracker

    try:
        # Extract prompt text and image paths
        prompt_text = ""
        image_paths = []

        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
            image_paths = prompt.image_paths or []
        else:
            prompt_text = prompt.prompt.text
            # Collect image paths from nested or top-level
            image_paths = prompt.prompt.image_paths or prompt.image_paths or []
            if prompt.prompt.images and not image_paths:
                logger.warning("Codex CLI does not support base64 images - use image_paths instead")

        # Codex CLI has no --system-prompt, so prepend system prompt to user prompt
        if prompt.system:
            prompt_text = f"{prompt.system}\n\n{prompt_text}"
            logger.info(f"Prepended system prompt ({len(prompt.system)} chars) to user prompt")

        # Federation persona (opt-in). Codex has no system-prompt flag, so we
        # prepend it ahead of any caller system prompt -> persona, system, user.
        persona = (await get_persona_bundle("chatgpt", prompt.persona_variant))["rendered"]
        if persona:
            prompt_text = f"{persona}\n\n{prompt_text}"
            logger.info(f"Injected persona api-ai-chatgpt-{prompt.persona_variant} ({len(persona)} chars)")

        logger.info(f"Calling codex exec with prompt length: {len(prompt_text)} chars")

        # Build CLI command
        # --dangerously-bypass-approvals-and-sandbox is the Codex
        # equivalent of Gemini's --no-sandbox --yolo: full network +
        # filesystem access, no confirmation prompts. Needed for the
        # model to actually fetch URLs in the prompt instead of
        # filename-guessing.
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox"]

        # Only add model if explicitly specified by user
        # (ChatGPT subscription has limited model access, let Codex use its default)
        selected_model = model  # None if not specified
        if selected_model:
            cmd.extend(["--model", selected_model])

        # Reasoning depth via Codex config-override. Key confirmed by
        # Automation against OpenAI docs (May 2026). Supported levels for
        # GPT-5 family (our defaults gpt-5.5/5.4/5.3-codex): minimal, low,
        # medium, high, xhigh. `minimal` is GPT-5-only — older models will
        # reject and the upstream error surfaces verbatim via our 400/502
        # mapping.
        if prompt.effort:
            cmd.extend(["-c", f"model_reasoning_effort={prompt.effort}"])

        _imgs = _download_storage_images(prompt_text)
        if _imgs:
            for _u, _p in _imgs:
                prompt_text = prompt_text.replace(_u, "(siehe angehaengtes Bild)")
            image_paths = list(image_paths) + [_p for _u, _p in _imgs]
            logger.info("codex: localized %d storage image(s) to /tmp -> -i" % len(_imgs))

        # Image inputs: simple-append pattern (same as claude_endpoint:518-524).
        # The CLI reads paths/URLs out of the prompt text natively — no `-i`
        # flag dance needed. Local paths, https URLs, mixed list — all fine.
        # The earlier `-i` injection + later 400-with-hint defense were both
        # over-engineered; Alex confirmed (2026-06-13) the canonical
        # convention is "just tell the model where the image is in the
        # prompt and the CLI fetches it".
        if image_paths:
            paths_text = "\n".join(image_paths)
            prompt_text = (
                f"{prompt_text}\n\nBild-Pfade (lokal oder URL):\n{paths_text}"
            )
            logger.info(
                f"Added {len(image_paths)} image ref(s) to prompt for codex-CLI "
                f"(append-pattern, no -i flag)"
            )

        # Add prompt as argument
        cmd.append(prompt_text)

        # Call codex exec in subprocess
        def run_codex_cli():
            import os
            import pwd
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # Remove OPENAI_API_KEY so Codex CLI uses OAuth credentials
            env.pop("OPENAI_API_KEY", None)

            # Own-pgid + killpg-on-timeout — see _run_cli_with_pgid docstring
            result = _run_cli_with_pgid(cmd, env=env, timeout=300)
            return result

        sem = await _acquire_cli_slot("codex")
        try:
            result = await asyncio.to_thread(run_codex_cli)
        finally:
            sem.release()
            for _u, _p in _imgs:
                try:
                    os.remove(_p)
                except Exception:
                    pass

        # Log the result for debugging
        logger.info(f"Codex CLI returncode: {result.returncode}")
        logger.info(f"Codex CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stderr:
            logger.info(f"Codex CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        if not raw_output:
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from codex CLI (empty response)"
                logger.error(f"Codex CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Codex CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail="Codex CLI returned empty response")

        # Parse JSONL output - each line is a separate JSON object
        # We need to find the agent_message item and usage info
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        model_name = selected_model or "codex-default"  # Track as codex-default if no model specified
        upstream_error: Optional[str] = None  # captured if Codex emits a turn.failed

        for line in raw_output.split('\n'):
            if not line.strip():
                continue
            try:
                event = json_module.loads(line)

                # Extract agent message
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                # Extract usage info
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                # Capture upstream error so we can return a useful message
                # instead of an opaque 500. Codex emits either a top-level
                # {"type":"error",...} or a {"type":"turn.failed","error":{...}}.
                if event.get("type") in ("error", "turn.failed"):
                    err_payload = event.get("error") if event.get("type") == "turn.failed" else event
                    err_msg = err_payload.get("message") if isinstance(err_payload, dict) else None
                    if err_msg:
                        upstream_error = err_msg

            except json_module.JSONDecodeError:
                continue

        if not response_text:
            # Classify the upstream error so the caller gets an actionable 4xx
            # instead of an opaque 500. The "X model is not supported when using
            # Codex with a ChatGPT account" message is by far the most common —
            # the ChatGPT subscription path locks Codex to its built-in default
            # model and rejects any --model override. Tell the caller exactly
            # that so the frontend can stop sending the model param.
            if upstream_error and "not supported when using Codex with a ChatGPT account" in upstream_error:
                logger.warning(
                    f"Codex rejected model={model_name!r} (ChatGPT subscription "
                    f"does not allow model selection). Suggesting caller drop "
                    f"the model param."
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Codex CLI (ChatGPT subscription) does not support model "
                        f"selection. Requested model={model_name!r} was rejected. "
                        f"Omit the 'model' query parameter to use the subscription "
                        f"default (codex-default)."
                    ),
                )
            if upstream_error:
                logger.error(f"Codex upstream error: {upstream_error}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Codex CLI upstream error: {upstream_error}",
                )
            logger.error(f"No agent_message found in Codex output: {raw_output[:500]}")
            raise HTTPException(status_code=500, detail="No response from Codex CLI")

        # Track usage
        codex_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Codex CLI response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Codex CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Codex CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("Codex CLI not found - is 'codex' installed?")
        raise HTTPException(status_code=500, detail="Codex CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Codex error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/chatgpt/cost-status")
async def chatgpt_cost_status():
    """
    Get current Codex CLI cost tracking status.

    Returns:
    - Current month's usage and costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational - Codex CLI uses your subscription.
    """
    from ..services.codex_cost_tracker import codex_cost_tracker
    return codex_cost_tracker.get_status()


async def _gemini_vision_with_paths(
    prompt_text: str,
    image_paths: List[str],
    model: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> AIResponse:
    """
    Helper: Use Google Generative AI API for vision analysis with local files.

    Args:
        prompt_text: The text prompt
        image_paths: List of local file paths to images
        model: Optional model override (default: gemini-2.0-flash)
        system_prompt: Optional system prompt to prepend
    """
    import os
    import google.generativeai as genai

    # Configure API
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured for vision")

    genai.configure(api_key=google_api_key)

    # Prepend system prompt if provided
    if system_prompt:
        prompt_text = f"{system_prompt}\n\n{prompt_text}"

    # Build content parts
    content_parts = []

    # Add images from file paths
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                image_data = f.read()

            # Detect mime type from extension
            ext = os.path.splitext(img_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".heic": "image/heic"
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            content_parts.append({
                "mime_type": mime_type,
                "data": image_data
            })
            logger.info(f"Loaded image for Gemini Vision: {img_path} ({len(image_data)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")

    if not content_parts:
        raise HTTPException(status_code=400, detail="No valid images could be loaded")

    # Add text prompt
    content_parts.append(prompt_text)

    # Generate response
    model_name = model or "gemini-2.0-flash"
    gemini_model = genai.GenerativeModel(model_name)

    try:
        response = gemini_model.generate_content(content_parts)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Extract token count if available
        tokens_used = None
        if hasattr(response, 'usage_metadata'):
            tokens_used = getattr(response.usage_metadata, 'total_token_count', None)

        logger.info(f"Gemini Vision API response: {len(response_text)} chars")

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )
    except Exception as e:
        logger.error(f"Gemini Vision API error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini Vision error: {str(e)}")


@router.post("/gemini", response_model=AIResponse)
async def gemini_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """
    Gemini endpoint — currently DISABLED at the handler entry.

    Background (2026-05-30 cost incident, 209.56 EUR / Mai):
    The gemini-CLI was running in API-billing mode because GOOGLE_API_KEY
    was loaded via dotenv and forwarded to the subprocess as GEMINI_API_KEY.
    The "subscription" claim never held — no OAuth flow was ever set up for
    Alex' Google account on these hosts.

    Until a proper `gemini /authorize` flow is run per host (Login broker
    follow-up), the text path is intentionally 503. Image-bearing requests
    are routed to `_gemini_vision_with_paths`, which is API-billed and gated
    behind the `confirm_api_billing` body-flag + monthly cap.
    """
    import subprocess
    import asyncio
    import os
    import json as json_module
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    # Extract prompt text and image paths
    prompt_text = ""
    image_paths = []

    if isinstance(prompt.prompt, str):
        prompt_text = prompt.prompt
        image_paths = prompt.image_paths or []
    else:
        prompt_text = prompt.prompt.text
        image_paths = prompt.prompt.image_paths or prompt.image_paths or []

    # Images go through the gemini CLI over the OAuth subscription (FREE) — not
    # the paid Gemini Vision API. With --no-sandbox --yolo the CLI reads local
    # file paths AND WebFetches image URLs from the prompt. OAuth subscription
    # is set up on these hosts now, so the old hard-503 is gone.
    if image_paths:
        paths_text = "\n".join(image_paths)
        prompt_text = f"{prompt_text}\n\nAnalysiere folgende Bilder:\n{paths_text}"
        logger.info(f"Folding {len(image_paths)} image ref(s) into gemini CLI prompt (subscription path)")

    # --- dead code below (kept for the OAuth re-enable path) -------------------

    try:
        # Gemini CLI has no --system-prompt, so prepend system prompt to user prompt
        if prompt.system:
            prompt_text = f"{prompt.system}\n\n{prompt_text}"
            logger.info(f"Prepended system prompt ({len(prompt.system)} chars) to user prompt")

        # Federation persona (opt-in). No system-prompt flag -> prepend ahead
        # of any caller system prompt -> persona, system, user.
        persona = (await get_persona_bundle("gemini", prompt.persona_variant))["rendered"]
        if persona:
            prompt_text = f"{persona}\n\n{prompt_text}"
            logger.info(f"Injected persona api-ai-gemini-{prompt.persona_variant} ({len(persona)} chars)")

        logger.info(f"Calling gemini CLI with prompt length: {len(prompt_text)} chars")

        # Build CLI command — agy (Google Antigravity CLI) replaces the
        # deprecated `gemini` CLI as of 2026-06-30. agy's flag set is
        # smaller: a single --dangerously-skip-permissions auto-approves
        # all tool calls (replaces --no-sandbox + --yolo + --skip-trust
        # together). --output-format json is supported and returns a
        # cleaner flat schema than gemini (see usage parsing below).
        cmd = ["agy", "--output-format", "json",
               "--dangerously-skip-permissions"]
        if model:
            cmd.extend(["--model", model])
        # Gemini CLI does not expose `thinking_budget` (verified May 2026 —
        # only --model/--raw-output/--no-sandbox/--yolo/--skip-trust). The
        # API knob exists but only via direct SDK call. We log + ignore so
        # callers using a generic /ai/<provider>?effort= contract don't
        # break — the frontend should ground-truth supported_via=='none'
        # from /ai/models providers_meta.gemini.efforts and disable the
        # dropdown there.
        if prompt.effort:
            logger.info(
                f"effort={prompt.effort!r} requested but Gemini CLI does "
                f"not support thinking_budget; ignoring."
            )
        _imgs = _download_storage_images(prompt_text)
        for _u, _p in _imgs:
            prompt_text = prompt_text.replace(_u, "@" + _p)
        if _imgs:
            logger.info("agy: localized %d storage image(s) -> @path" % len(_imgs))
        # agy requires an explicit -p/--print flag for non-interactive mode;
        # without it the CLI tries to open a TTY (bubbletea) and aborts in
        # the service context with "could not open TTY: /dev/tty".
        cmd.extend(["-p", prompt_text])

        # Call gemini in subprocess
        def run_gemini_cli():
            import os
            import pwd
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # CLEAN DEFAULT (cwd isolation): gemini discovers GEMINI.md by
            # walking the cwd up the directory tree. From the service WorkingDir
            # (/var/www/...) that walk reaches `/` and picks up `/GEMINI.md`
            # (the host Dev-Engineer context) as Vorwissen — verified leak. The
            # walk STOPS at $HOME, so running from a scratch dir UNDER cli_home
            # (with no GEMINI.md) yields no inherited Vorwissen (verified: the
            # "are you the dev engineer" probe flips JA->NEIN). The persona, when
            # set, is prepended to the prompt as before and stays authoritative.
            neutral_cwd = os.path.join(cli_home, ".aiapi-neutral")
            try:
                os.makedirs(neutral_cwd, exist_ok=True)
            except OSError:
                neutral_cwd = "/"

            # Hardening 2026-05-30 (GCP cost incident, 209.56 EUR Mai):
            # The previous conditional `if google_key: env["GEMINI_API_KEY"] = google_key`
            # caused the CLI to fall back to Gemini-API-billing-mode instead of
            # OAuth subscription whenever GOOGLE_API_KEY was loaded via dotenv.
            # Removed entirely — gemini-CLI must authenticate via OAuth subscription
            # only. Until a proper /authorize flow is set up per host (Login broker
            # follow-up), the /ai/gemini endpoint returns 503 at the handler entry.

            # Own-pgid + killpg-on-timeout — critical for gemini-cli
            # specifically because it forks helper procs (model router,
            # telemetry) that won't die with a plain proc.kill()
            result = _run_cli_with_pgid(cmd, env=env, timeout=300, cwd=neutral_cwd)
            return result

        sem = await _acquire_cli_slot("gemini")
        try:
            result = await asyncio.to_thread(run_gemini_cli)
        finally:
            sem.release()
            for _u, _p in _imgs:
                try:
                    os.remove(_p)
                except Exception:
                    pass

        # Log the result for debugging
        logger.info(f"Gemini CLI returncode: {result.returncode}")
        logger.info(f"Gemini CLI stdout length: {len(result.stdout) if result.stdout else 0}")
        if result.stderr:
            logger.info(f"Gemini CLI stderr: {result.stderr[:200]}")

        raw_output = result.stdout.strip() if result.stdout else ""

        if not raw_output:
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error from gemini CLI (empty response)"
                logger.error(f"Gemini CLI error (empty stdout): {error_msg}")
                raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error_msg}")
            raise HTTPException(status_code=500, detail="Gemini CLI returned empty response")

        # Parse JSON output — agy schema (flat):
        # {"conversation_id": "...", "status": "SUCCESS",
        #  "response": "...", "duration_seconds": <float>,
        #  "num_turns": <int>,
        #  "usage": {"input_tokens": ..., "output_tokens": ...,
        #            "thinking_tokens": ..., "total_tokens": ...}}
        try:
            cli_response = json_module.loads(raw_output)
        except json_module.JSONDecodeError as e:
            logger.error(f"Failed to parse agy CLI JSON: {e}")
            logger.error(f"Raw output: {raw_output[:500]}")
            # Return raw output as response if JSON parsing fails
            return AIResponse(
                response=raw_output,
                model="agy-cli",
                tokens_used=None,
                finish_reason="stop"
            )

        # Extract response text + strip trailing newline agy appends
        response_text = (cli_response.get("response") or "").rstrip("\n")

        # Extract token info from flat usage block.
        # thinking_tokens are upfront compute (planning) — pack them into
        # input_tokens for the cost-tracker so they're billed at the same
        # rate as prompt tokens. The cost-tracker schema only has
        # input/output buckets, no thinking-specific entry.
        usage = cli_response.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        thinking_tokens = int(usage.get("thinking_tokens", 0))
        input_tokens += thinking_tokens
        # agy doesn't echo the model name in the JSON response; fall back
        # to the requested model or a generic identifier.
        model_name = (model or "agy-cli")

        # Track usage
        gemini_cli_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Gemini CLI response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except subprocess.TimeoutExpired:
        logger.error("Gemini CLI timeout after 300 seconds")
        raise HTTPException(status_code=504, detail="Gemini CLI timeout - prompt may be too long")
    except FileNotFoundError:
        logger.error("agy CLI not found - is 'agy' installed and on PATH?")
        raise HTTPException(status_code=500, detail="Gemini CLI not installed on server")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini CLI error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/gemini/cost-status")
async def gemini_cost_status():
    """
    Get current Gemini CLI usage tracking status.

    Returns:
    - Current month's usage and estimated costs
    - Token breakdown by model
    - Request statistics

    Note: Costs are informational only - Gemini CLI uses free tier (no actual billing).
    """
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker
    return gemini_cli_cost_tracker.get_status()


@router.post("/gemini/vision", response_model=AIResponse)
async def gemini_vision_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Gemini Vision endpoint via Google Generative AI API.

    Features:
    - Supports multimodal input (text + images)
    - Uses Gemini 2.0 Flash for vision tasks
    - Base64 encoded images in prompt.images array
    - Perfect for image analysis, safety checks, content moderation

    Note: Uses GOOGLE_API_KEY from environment.
    """
    import os
    import base64
    import uuid
    import httpx

    # Vision runs over the gemini CLI / OAuth subscription (FREE) now — NOT the
    # paid Gemini Vision API. base64 images can't be handed to the CLI directly
    # (it hallucinates on local files outside its workspace), so each image is
    # stashed in storage with a short TTL (auto-purged), and the resulting
    # public URL is passed to /ai/gemini, where the CLI WebFetches + analyses it.
    # Endpoint kept for compatibility (storage/review/knowledge call it w/ base64).
    try:
        prompt_text = ""
        images_b64 = []
        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                images_b64 = prompt.prompt.images

        storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
        storage_key = os.getenv("STORAGE_API_KEY", "Inetpass1")

        image_urls = []
        if images_b64:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for i, img_b64 in enumerate(images_b64):
                    if img_b64.startswith("data:"):
                        header, b64_data = img_b64.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                    else:
                        b64_data = img_b64
                        mime_type = "image/jpeg"
                    ext = (mime_type.split("/")[-1] or "jpg").replace("jpeg", "jpg")
                    img_bytes = base64.b64decode(b64_data)
                    resp = await client.post(
                        f"{storage_url}/storage/upload",
                        files={"file": (f"vision_{uuid.uuid4().hex[:8]}_{i}.{ext}", img_bytes, mime_type)},
                        data={"is_public": "true", "analyze": "false", "ai_mode": "none",
                              "ttl_hours": "1", "reuse_existing": "false"},
                        headers={"X-API-KEY": storage_key},
                    )
                    resp.raise_for_status()
                    sid = resp.json().get("id")
                    if not sid:
                        raise HTTPException(status_code=502, detail=f"vision upload bridge: no id ({resp.text[:160]})")
                    image_urls.append(f"{storage_url}/storage/media/{sid}")
                    logger.info(f"Vision bridge: image {i+1} -> media/{sid} (ttl 1h)")

        if model:
            logger.info(f"Vision: ignoring requested model={model!r}; gemini CLI uses its default")
        delegate = Prompt(prompt=prompt_text, system=prompt.system, image_paths=image_urls or None)
        return await gemini_endpoint(delegate, model=None, api_key=api_key)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini Vision (CLI bridge) error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/gemini/send-report")
async def gemini_send_report():
    """
    Manually trigger a usage report via Telegram.

    Note: Shows CLI usage statistics (no actual costs - free tier).
    """
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    # Get status and send via Telegram
    status = gemini_cli_cost_tracker.get_status()

    import httpx
    import os
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

    if telegram_token and telegram_chat:
        message = f"""
📊 <b>Gemini CLI Usage Report</b>

<b>Month:</b> {status.get('month', 'N/A')}
<b>Requests:</b> {status.get('request_count', 0):,}
<b>Input Tokens:</b> {status.get('total_input_tokens', 0):,}
<b>Output Tokens:</b> {status.get('total_output_tokens', 0):,}

<b>Estimated Value:</b> ${status.get('total_cost_usd', 0):.4f} (if API)

<i>Note: Gemini CLI uses free tier - no actual costs!</i>
"""
        try:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json={"chat_id": telegram_chat, "text": message.strip(), "parse_mode": "HTML"})
        except Exception:
            pass

    return {"status": "Report sent", "usage": status}


@router.post("/gemini/gcp-budget-webhook")
async def gcp_budget_webhook(request: Request):
    """
    Webhook endpoint for GCP Budget Pub/Sub push notifications.

    Configure in GCP:
    1. Create Push Subscription on gemini-budget-alerts Topic
    2. Endpoint URL: https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook

    This is a BACKUP alert system with ~24h delay.
    Primary alerting is done via real-time token tracking.

    NOTE: Uses NotificationManager to limit alerts to max 1 per day.
    """
    import base64
    import json
    from ..services.cost_tracker import cost_tracker
    from ..services.notification_manager import notification_manager

    try:
        body = await request.json()

        # Decode Pub/Sub message
        message_data = body.get("message", {}).get("data", "")
        if message_data:
            decoded = base64.b64decode(message_data).decode()
            budget_data = json.loads(decoded)

            budget_name = budget_data.get("budgetDisplayName", "Unknown")
            cost_amount = budget_data.get("costAmount", 0)
            budget_amount = budget_data.get("budgetAmount", 0)
            threshold = budget_data.get("alertThresholdExceeded", 0) * 100

            logger.info(f"GCP Budget webhook received: {threshold:.0f}% threshold for {budget_name}")

            # GCP-side hard-cap (cost-incident hardening 2026-05-30):
            # When Google itself reports 100% threshold reached, persistently
            # block all API-billed /ai/* calls regardless of the local
            # counter view. The flag survives process restarts (lives in
            # the monthly usage file) and is cleared only via the dedicated
            # POST /ai/gemini/cost-status/reset-hard-cap endpoint (or by
            # month rollover when a new usage file is rotated in).
            if threshold >= 100:
                cost_tracker.trip_gcp_hard_cap(
                    reason=(
                        f"GCP webhook reported {threshold:.0f}% of "
                        f"{budget_name}: ${cost_amount:.2f} / ${budget_amount:.2f}"
                    )
                )

            # Check if we can send notification (cooldown: 24h)
            if not notification_manager.can_send("gcp_budget_alert"):
                next_allowed = notification_manager.get_next_allowed("gcp_budget_alert")
                logger.info(
                    f"GCP Budget alert suppressed (cooldown). "
                    f"Next allowed: {next_allowed.isoformat() if next_allowed else 'now'}"
                )
                return {"status": "ok", "notification": "suppressed_cooldown"}

            # Determine emoji and status
            if threshold >= 100:
                emoji = "🚨"
                status = "GCP: BUDGET EXCEEDED"
            elif threshold >= 80:
                emoji = "⚠️"
                status = "GCP: WARNING"
            else:
                emoji = "📊"
                status = "GCP: INFO"

            message = f"""
{emoji} <b>{status}</b>

<b>Budget:</b> {budget_name}
<b>Threshold:</b> {threshold:.0f}%
<b>GCP Cost:</b> ${cost_amount:.2f} / ${budget_amount:.2f}

<i>Note: GCP data has ~24h delay</i>
<i>Real-time status: /ai/gemini/cost-status</i>
"""
            cost_tracker._send_telegram_message(message.strip())
            notification_manager.mark_sent("gcp_budget_alert")
            logger.info(f"GCP Budget webhook: sent alert for {threshold:.0f}% threshold")

        return {"status": "ok", "notification": "sent"}

    except Exception as e:
        logger.error(f"GCP Budget webhook error: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/gemini/cost-status/reset-hard-cap")
async def reset_gcp_hard_cap(api_key: str = Depends(get_api_key)):
    """Manually clear the persistent GCP hard-cap flag.

    Use only after confirming GCP-side budget has reset (month rollover)
    or after intentional acknowledgement that further spending is OK.
    The webhook will re-trip the flag automatically the next time GCP
    reports 100% threshold reached, so this is safe to call without
    risking silent re-opening.
    """
    from ..services.cost_tracker import cost_tracker
    result = cost_tracker.clear_gcp_hard_cap()
    logger.warning(
        f"GCP hard-cap manually reset (was_active={result['was_active']})"
    )
    return result


@router.get("/notifications/status")
async def get_notification_status():
    """
    Get notification manager status.

    Shows cooldown state for all notification types.
    """
    from ..services.notification_manager import notification_manager
    return notification_manager.get_status()


@router.post("/notifications/reset/{notification_type}")
async def reset_notification_cooldown(notification_type: str):
    """
    Reset cooldown for a specific notification type.

    Args:
        notification_type: Type to reset (e.g., "gcp_budget_alert")
    """
    from ..services.notification_manager import notification_manager
    notification_manager.reset(notification_type)
    return {"status": "reset", "notification_type": notification_type}


@router.get("/models")
async def list_text_models(
    provider: Optional[str] = None,
    group_by: Optional[str] = None,
):
    """List available text AI models.

    Live source: ``/var/lib/api-ai/models.json`` produced by the daily
    ``api-ai-maintenance.timer`` (CLI version + smoke-test driven). When
    that file is missing or stale we return a conservative hardcoded
    fallback so existing clients don't break.

    Query params:
        provider: filter results to a single provider (claude/codex/gemini).
            Case-insensitive. Models from other providers are dropped.
        group_by: when set to ``provider``, the response gains an extra
            ``by_provider`` field with the shape ``{provider: [model_ids]}``
            for clients that want grouped dropdowns without re-pivoting
            the flat ``models`` array.
    """
    import json as _json
    import time
    from pathlib import Path as _Path

    state_path = _Path("/var/lib/api-ai/models.json")
    if state_path.exists():
        try:
            age = time.time() - state_path.stat().st_mtime
            if age <= 25 * 3600:  # accept up to 25h (timer is daily + jitter)
                data = _json.loads(state_path.read_text())
                providers = data.get("providers", {})
                models = []
                for prov_name, prov_info in providers.items():
                    endpoint = {
                        "claude": "/ai/claude",
                        "codex":  "/ai/chatgpt",
                        "gemini": "/ai/gemini",
                    }.get(prov_name, f"/ai/{prov_name}")
                    for m in (prov_info.get("available") or []):
                        models.append({
                            "id": m,
                            "provider": prov_name,
                            "endpoint": endpoint,
                            "is_default": (m == prov_info.get("default")),
                            "cli_version": prov_info.get("cli_version"),
                        })
                # Apply ?provider= filter (case-insensitive)
                if provider:
                    p = provider.lower().strip()
                    models = [m for m in models if m["provider"] == p]
                    providers = {k: v for k, v in providers.items() if k == p}

                response = {
                    # _source is "automation-curated" when Automation's
                    # orchestrator pushed via POST /internal/notify-cli-update
                    # with a `models` payload (KI-validated). Otherwise
                    # "discovery" (local script). Surface that distinction
                    # so callers can tell which side wrote the data.
                    "source": data.get("_source", "discovery"),
                    "updated_at": data.get("updated_at"),
                    "host": data.get("host"),
                    "stale_age_seconds": int(age),
                    "models": models,
                    "providers_meta": {
                        name: {
                            "default": info.get("default"),
                            "subscription_locked": info.get("subscription_locked", False),
                            "cli_version": info.get("cli_version"),
                            "discovery_method": info.get("discovery_method"),
                            # Reasoning-depth spec — Automation's curated
                            # block, see /internal/notify-cli-update payload
                            # schema. Shape: {param_name, available[],
                            # default, supported_via, note?}. None when the
                            # current payload doesn't include it.
                            "efforts": info.get("efforts"),
                        }
                        for name, info in providers.items()
                    },
                }
                # Optional grouped view: {provider: [ids]} for clients that
                # want pre-pivoted dropdowns
                if group_by == "provider":
                    grouped: dict[str, list[str]] = {}
                    for m in models:
                        grouped.setdefault(m["provider"], []).append(m["id"])
                    response["by_provider"] = grouped
                return response
            logger.warning(f"models.json is {age/3600:.1f}h old (>25h), using fallback")
        except Exception as e:
            logger.warning(f"Failed to read models.json: {e} — using fallback")

    # Fallback: minimal hardcoded list that's known to work today
    fallback_models = [
        {"id": "sonnet", "provider": "claude", "endpoint": "/ai/claude", "is_default": True},
        {"id": "opus",   "provider": "claude", "endpoint": "/ai/claude"},
        {"id": "haiku",  "provider": "claude", "endpoint": "/ai/claude"},
        {"id": "codex-default", "provider": "codex",  "endpoint": "/ai/chatgpt", "is_default": True},
        {"id": "gemini-2.5-flash",      "provider": "gemini", "endpoint": "/ai/gemini", "is_default": True},
        {"id": "gemini-2.5-flash-lite", "provider": "gemini", "endpoint": "/ai/gemini"},
        {"id": "gemini-2.5-pro",        "provider": "gemini", "endpoint": "/ai/gemini"},
    ]
    if provider:
        p = provider.lower().strip()
        fallback_models = [m for m in fallback_models if m["provider"] == p]
    response = {"source": "fallback", "models": fallback_models}
    if group_by == "provider":
        grouped: dict[str, list[str]] = {}
        for m in fallback_models:
            grouped.setdefault(m["provider"], []).append(m["id"])
        response["by_provider"] = grouped
    return response



# ──────────────────────────────────────────────────────────────────────
# MiniMax M3 — API-direct text endpoint (PR-6)
#
# Architectural choice rationale (Minimax-bot IACP cee14357, 2026-06-10):
# M3 IS available via claude-CLI with
# ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic + override env vars.
# We deliberately do NOT use that path for /ai/m3 because:
#   1. claude-CLI injects its agent system-prompt + tool definitions
#      (~10-30k tokens) per call → real PAYG money on a text-completion
#      endpoint that shouldn't be agentic.
#   2. CLI cold-start adds seconds of latency per request.
#   3. Caller gets claude-code's agent behaviour, not the raw M3 model.
#
# So /ai/m3 goes direct OpenAI-compat against api.minimax.io/v1, billing
# via the multimodal pay-as-you-go key.

@router.post("/m3", response_model=AIResponse)
async def m3_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """MiniMax M3 text endpoint — API-direct via OpenAI /v1/chat/completions.

    Pay-as-you-go billed against the federation-shared 25 EUR/month cap.
    Subscription endpoints (/ai/claude, /ai/chatgpt) remain the right
    choice for cost-free flat-rate text; /ai/m3 is for callers who
    specifically want M3's behaviour and accept PAYG billing.
    """
    import os
    from ..services.minimax_cost_tracker import minimax_cost_tracker

    _check_minimax_billing_gate(
        prompt.confirm_api_billing, endpoint="m3"
    )

    api_key_val = os.getenv("MINIMAX_MULTIMODAL_API_KEY", "")
    if not api_key_val:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "minimax_api_key_missing",
                "hint": "MINIMAX_MULTIMODAL_API_KEY not configured in service env.",
            },
        )

    if isinstance(prompt.prompt, str):
        user_text = prompt.prompt
    else:
        user_text = prompt.prompt.text

    messages: list = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    if prompt.conversation_history:
        messages.extend(prompt.conversation_history)
    messages.append({"role": "user", "content": user_text})

    selected_model = model or "MiniMax-M3"
    logger.info(
        f"Calling MiniMax M3 API: model={selected_model}, "
        f"messages={len(messages)}, max_tokens={prompt.max_tokens}, "
        f"temp={prompt.temperature}"
    )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai SDK not installed in api-ai venv",
        )

    client = AsyncOpenAI(
        api_key=api_key_val,
        base_url=os.getenv("MINIMAX_TEXT_BASE_URL", "https://api.minimax.io/v1"),
    )

    try:
        resp = await client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=prompt.max_tokens or 1000,
            temperature=prompt.temperature if prompt.temperature is not None else 0.7,
        )
    except Exception as e:
        logger.error(f"MiniMax M3 upstream error: {e}")
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_m3_upstream_error", "exc": str(e)[:300]},
        )

    if not resp.choices:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_m3_no_choices"},
        )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

    if in_tok or out_tok:
        minimax_cost_tracker.track_text(
            model="minimax-m3",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    return AIResponse(
        response=text,
        message=text,
        model=selected_model,
        tokens_used=in_tok + out_tok if (in_tok or out_tok) else None,
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
    )


# ──────────────────────────────────────────────────────────────────────
# DeepSeek V4 — API-direct text endpoint (Minimax IACP 315228fe, 2026-06-14)
#
# DeepSeek's OpenAI-compatible endpoint at api.deepseek.com/v1 mirrors
# our /ai/m3 architecture exactly: OpenAI SDK with base_url override +
# bearer auth via DEEPSEEK_API_KEY + PAYG billing tracked through
# deepseek_cost_tracker (25 EUR/month default cap).
#
# Two models:
#   deepseek-v4-flash — cheaper, ~$0.07/M in + $0.27/M out
#   deepseek-v4-pro   — capable, ~$0.27/M in + $1.10/M out
#
# Both surface a Chain-of-Thought reasoning block in usage.reasoning_tokens
# similar to MiniMax M3's <think>...</think>. Callers wanting deterministic
# JSON output should account for the reasoning-token budget when setting
# max_tokens (raise to 1000+ for non-trivial structured outputs).

@router.post("/deepseek", response_model=AIResponse)
async def deepseek_endpoint(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """DeepSeek V4 text endpoint — API-direct via OpenAI /v1/chat/completions.

    Pay-as-you-go billed against the federation-shared 25 EUR/month cap.
    """
    import os
    from ..services.deepseek_cost_tracker import deepseek_cost_tracker

    _check_deepseek_billing_gate(
        prompt.confirm_api_billing, endpoint="deepseek"
    )

    api_key_val = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key_val:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "deepseek_api_key_missing",
                "hint": "DEEPSEEK_API_KEY not configured in service env.",
            },
        )

    if isinstance(prompt.prompt, str):
        user_text = prompt.prompt
    else:
        user_text = prompt.prompt.text

    messages: list = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    if prompt.conversation_history:
        messages.extend(prompt.conversation_history)
    messages.append({"role": "user", "content": user_text})

    selected_model = model or "deepseek-v4-flash"
    logger.info(
        f"Calling DeepSeek API: model={selected_model}, "
        f"messages={len(messages)}, max_tokens={prompt.max_tokens}, "
        f"temp={prompt.temperature}"
    )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai SDK not installed in api-ai venv",
        )

    client = AsyncOpenAI(
        api_key=api_key_val,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )

    try:
        resp = await client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=prompt.max_tokens or 1000,
            temperature=prompt.temperature if prompt.temperature is not None else 0.7,
        )
    except Exception as e:
        logger.error(f"DeepSeek upstream error: {e}")
        raise HTTPException(
            status_code=502,
            detail={"error": "deepseek_upstream_error", "exc": str(e)[:300]},
        )

    if not resp.choices:
        raise HTTPException(
            status_code=502,
            detail={"error": "deepseek_no_choices"},
        )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

    if in_tok or out_tok:
        deepseek_cost_tracker.track_text(
            model=selected_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    return AIResponse(
        response=text,
        message=text,
        model=selected_model,
        tokens_used=in_tok + out_tok if (in_tok or out_tok) else None,
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
    )
