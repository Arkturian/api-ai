#!/usr/bin/env python3
"""Benchmark all chat-class models across Claude/ChatGPT/Gemini endpoints.
Concurrent with a bounded pool so we don't slam the CLI subprocess layer."""
import asyncio, json, time, sys
import httpx

BASE = "https://api-ai.arkturian.com"
PROMPT = "Antworte mit EXAKT diesem Wort: PONG"
TIMEOUT = 90

# Gemini chat/general subset — skip tts/image/research/agent specializations
GEMINI = [
    "gemini-2.0-flash", "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
    "gemini-3-flash-preview", "gemini-3-pro-preview",
    "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest",
    "gemma-4-26b-a4b-it", "gemma-4-31b-it",
]
CLAUDE = ["sonnet", "opus", "haiku"]
CHATGPT = [None]  # subscription only allows default

CONCURRENCY = 4  # don't blast more than 4 subprocess CLI calls at once
sem = asyncio.Semaphore(CONCURRENCY)


async def call(client: httpx.AsyncClient, provider: str, model: str | None):
    url = f"{BASE}/ai/{provider}"
    if model:
        url += f"?model={model}"
    payload = {"prompt": PROMPT}
    started = time.monotonic()
    async with sem:
        try:
            r = await client.post(url, json=payload, timeout=TIMEOUT)
            elapsed = time.monotonic() - started
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:120]}
            return {
                "provider": provider,
                "model": model or "(default)",
                "http": r.status_code,
                "elapsed_s": round(elapsed, 2),
                "answer": (data.get("response") or "")[:50] if isinstance(data, dict) else "",
                "actual_model": data.get("model") if isinstance(data, dict) else None,
                "tokens": data.get("tokens_used") if isinstance(data, dict) else None,
                "error": data.get("detail") if isinstance(data, dict) and not r.is_success else None,
            }
        except Exception as e:
            elapsed = time.monotonic() - started
            return {
                "provider": provider, "model": model or "(default)",
                "http": "EXC", "elapsed_s": round(elapsed, 2),
                "answer": "", "error": str(e)[:120],
            }


async def main():
    tasks = []
    async with httpx.AsyncClient() as client:
        for m in GEMINI:  tasks.append(call(client, "gemini", m))
        for m in CLAUDE:  tasks.append(call(client, "claude", m))
        for m in CHATGPT: tasks.append(call(client, "chatgpt", m))
        results = await asyncio.gather(*tasks)

    # sort by provider then elapsed
    results.sort(key=lambda r: (r["provider"], r["elapsed_s"]))

    # markdown table
    print(f"{'PROVIDER':<8} {'MODEL':<40} {'HTTP':<5} {'TIME':<8} {'ANSWER':<25} TOKENS")
    print("-" * 110)
    for r in results:
        ans = (r["answer"] or "—")[:24]
        toks = str(r["tokens"]) if r["tokens"] is not None else "—"
        err = ""
        if r.get("error") and r["http"] != 200:
            err = f"  ⚠ {str(r['error'])[:60]}"
        print(f"{r['provider']:<8} {r['model']:<40} {str(r['http']):<5} {r['elapsed_s']:>5.2f}s  {ans:<25} {toks}{err}")

    # summary
    ok = [r for r in results if r["http"] == 200]
    fail = [r for r in results if r["http"] != 200]
    print()
    print(f"✓ {len(ok)} ok   ✗ {len(fail)} fail   total {len(results)} models")
    if ok:
        fastest = min(ok, key=lambda r: r["elapsed_s"])
        slowest = max(ok, key=lambda r: r["elapsed_s"])
        print(f"fastest: {fastest['provider']}/{fastest['model']} ({fastest['elapsed_s']}s)")
        print(f"slowest: {slowest['provider']}/{slowest['model']} ({slowest['elapsed_s']}s)")

asyncio.run(main())
