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
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


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
            import os
            valid_paths = [p for p in image_paths if os.path.exists(p)]
            if valid_paths:
                paths_text = "\n".join(valid_paths)
                prompt_text = f"{prompt_text}\n\nAnalysiere folgende Bilder:\n{paths_text}"
                logger.info(f"Added {len(valid_paths)} image paths to prompt for Claude CLI")
            else:
                logger.warning(f"None of the {len(image_paths)} image paths exist: {image_paths}")

        logger.info(f"Calling claude -p with prompt length: {len(prompt_text)} chars")

        # Build CLI command with default model sonnet (cost-effective)
        # NOTE: --dangerously-skip-permissions is NOT compatible with
        # root/sudo (the CLI refuses to start). uvicorn here runs as
        # root, so we leave the flag off — in -p (print) mode there are
        # no tool calls anyway, so no permission prompts to bypass.
        cmd = ["claude", "-p", prompt_text, "--output-format", "json"]

        # Add model - default to sonnet (günstig), user can override with opus/haiku
        selected_model = model or "sonnet"
        cmd.extend(["--model", selected_model])

        # Add system prompt if provided
        if prompt.system:
            cmd.extend(["--system-prompt", prompt.system])

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

            # Run claude directly - root has access to all files, no permission bypass needed
            # The key is setting HOME to a non-root user's home so Claude credentials work
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
                cwd="/"
            )
            return result

        result = await asyncio.to_thread(run_claude_cli)

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

        # Add image paths via -i flag (Codex supports multiple -i flags)
        if image_paths:
            import os
            for img_path in image_paths:
                if os.path.exists(img_path):
                    cmd.extend(["-i", img_path])
                    logger.info(f"Added image to Codex command: {img_path}")
                else:
                    logger.warning(f"Image path does not exist: {img_path}")

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

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            return result

        result = await asyncio.to_thread(run_codex_cli)

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
    api_key: str = Depends(get_api_key)
):
    """
    Gemini endpoint - auto-routes to CLI or Vision API

    Features:
    - Text-only: Uses Gemini CLI (free tier, no API costs)
    - With images: Uses Google Generative AI API for vision analysis
    - Supports image_paths for local file analysis
    - Usage tracking at /ai/gemini/cost-status

    Note: CLI uses free tier. Vision API uses GOOGLE_API_KEY.
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

    # If we have image paths, use Google Vision API instead of CLI
    if image_paths:
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if valid_paths:
            logger.info(f"Routing to Gemini Vision API for {len(valid_paths)} images")
            return await _gemini_vision_with_paths(prompt_text, valid_paths, model, prompt.system)
        else:
            logger.warning(f"None of {len(image_paths)} image paths exist - falling back to CLI")

    try:
        # Gemini CLI has no --system-prompt, so prepend system prompt to user prompt
        if prompt.system:
            prompt_text = f"{prompt.system}\n\n{prompt_text}"
            logger.info(f"Prepended system prompt ({len(prompt.system)} chars) to user prompt")

        logger.info(f"Calling gemini CLI with prompt length: {len(prompt_text)} chars")

        # Build CLI command
        # --no-sandbox + --yolo give the CLI full network + filesystem
        # access without per-tool-call confirmations. Required so the
        # CLI can actually WebFetch URLs in the prompt (image judging,
        # documentation lookup, ...) — without these flags the model
        # falls back to filename-only heuristics.
        # --skip-trust bypasses the "trusted directory" prompt that
        # otherwise blocks the CLI in headless/service contexts.
        cmd = ["gemini", "--output-format", "json",
               "--no-sandbox", "--yolo", "--skip-trust"]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt_text)

        # Call gemini in subprocess
        def run_gemini_cli():
            import os
            import pwd
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            cli_home = os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir
            env["HOME"] = cli_home

            # Gemini CLI expects GEMINI_API_KEY, but we use GOOGLE_API_KEY
            google_key = os.getenv("GOOGLE_API_KEY", "")
            if google_key:
                env["GEMINI_API_KEY"] = google_key

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            return result

        result = await asyncio.to_thread(run_gemini_cli)

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

        # Parse JSON output
        # Format: {"response": "...", "stats": {"models": {"gemini-2.5-flash-lite": {"tokens": {...}}}}}
        try:
            cli_response = json_module.loads(raw_output)
        except json_module.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini CLI JSON: {e}")
            logger.error(f"Raw output: {raw_output[:500]}")
            # Return raw output as response if JSON parsing fails
            return AIResponse(
                response=raw_output,
                model="gemini-cli",
                tokens_used=None,
                finish_reason="stop"
            )

        # Extract response text
        response_text = cli_response.get("response", "")

        # Extract token info from stats
        input_tokens = 0
        output_tokens = 0
        model_name = "gemini-cli"

        stats = cli_response.get("stats", {})
        models = stats.get("models", {})

        # Get the first model from stats (usually gemini-2.5-flash-lite)
        for m_name, m_data in models.items():
            model_name = m_name
            tokens = m_data.get("tokens", {})
            input_tokens = tokens.get("prompt", 0)
            output_tokens = tokens.get("candidates", 0)
            break

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
        logger.error("Gemini CLI not found - is 'gemini' installed?")
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
    import google.generativeai as genai
    from ..services.gemini_cli_cost_tracker import gemini_cli_cost_tracker

    try:
        # Get API key from environment
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
        if not google_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        # Configure Gemini
        genai.configure(api_key=google_api_key)

        # Extract prompt text and images
        prompt_text = ""
        images_b64 = []

        if isinstance(prompt.prompt, str):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt.prompt.text
            if prompt.prompt.images:
                images_b64 = prompt.prompt.images

        logger.info(f"Calling Gemini Vision API with prompt length: {len(prompt_text)} chars, images: {len(images_b64)}")

        # Select model - use gemini-2.0-flash for vision (fast and capable)
        model_name = model or "gemini-2.0-flash"
        gemini_model = genai.GenerativeModel(model_name)

        # Build content parts
        content_parts = []

        # Add images first (better for vision tasks)
        for i, img_b64 in enumerate(images_b64):
            try:
                # Handle data URL format: "data:image/jpeg;base64,..."
                if img_b64.startswith("data:"):
                    # Extract mime type and base64 data
                    header, b64_data = img_b64.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                else:
                    # Assume raw base64 JPEG
                    b64_data = img_b64
                    mime_type = "image/jpeg"

                # Decode and create image part
                image_data = base64.b64decode(b64_data)
                content_parts.append({
                    "mime_type": mime_type,
                    "data": image_data
                })
                logger.info(f"Added image {i+1}: {mime_type}, {len(image_data)} bytes")
            except Exception as e:
                logger.warning(f"Failed to process image {i+1}: {e}")
                continue

        # Add text prompt
        content_parts.append(prompt_text)

        # Generate response
        response = gemini_model.generate_content(content_parts)

        # Extract response text
        response_text = response.text if response.text else ""

        # Extract token counts if available
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        # Track usage
        gemini_cli_cost_tracker.track_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_name
        })

        tokens_used = input_tokens + output_tokens

        logger.info(
            f"Gemini Vision response: {len(response_text)} chars, "
            f"{tokens_used} tokens ({input_tokens}in/{output_tokens}out)"
        )

        return AIResponse(
            response=response_text,
            model=model_name,
            tokens_used=tokens_used,
            finish_reason="stop"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Gemini Vision error: {error_msg}")
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

