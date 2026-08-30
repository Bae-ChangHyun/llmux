"""OpenAI 호환 엔드포인트 공통 헬퍼 (vllm / llama-server 재사용)."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import urllib.request

log = logging.getLogger(__name__)

BENCH_PROMPT = "Explain the theory of relativity in about 150 words."
BENCH_MAX_TOKENS = 200
# Qwen-family chat templates gate reasoning on this; other templates ignore
# unknown kwargs, so it is safe to send unconditionally.
BENCH_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


async def chat_completion_bench(
    port: int | str,
    model: str,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    timeout: int = 600,
) -> dict:
    """단일 /v1/chat/completions 호출 → {elapsed, usage}."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": dict(BENCH_CHAT_TEMPLATE_KWARGS),
        }
    ).encode()

    loop = asyncio.get_running_loop()

    def _do() -> dict:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        elapsed = time.time() - t0
        d = json.loads(raw, strict=False)
        return {"elapsed": elapsed, "usage": d.get("usage", {})}

    return await loop.run_in_executor(None, _do)


async def run_bench(
    port: int | str,
    model: str,
    *,
    prompt: str = BENCH_PROMPT,
    max_tokens: int = BENCH_MAX_TOKENS,
    runs: int = 3,
    warmup: int = 1,
) -> dict:
    """Warm up, then measure `runs` completions and summarize with the median.

    A single cold call conflates model/CUDA-graph warmup with steady-state
    decode speed, so the first number a user sees is always pessimistic and
    not reproducible. Discard `warmup` calls, then report the median of `runs`
    (median, not mean — one scheduler hiccup shouldn't move the headline).
    """
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    if warmup < 0:
        raise ValueError(f"warmup must be at least 0, got {warmup}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    for _ in range(warmup):
        await chat_completion_bench(
            port, model, prompt=prompt, max_tokens=max_tokens
        )

    results: list[dict] = []
    for _ in range(runs):
        r = await chat_completion_bench(
            port, model, prompt=prompt, max_tokens=max_tokens
        )
        usage = r.get("usage")
        if not isinstance(usage, dict) or "completion_tokens" not in usage:
            raise RuntimeError(
                "benchmark response is missing usage.completion_tokens"
            )
        try:
            tokens = int(usage["completion_tokens"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid usage.completion_tokens: {usage['completion_tokens']!r}"
            ) from exc
        if tokens < 0:
            raise RuntimeError(f"invalid negative completion token count: {tokens}")
        elapsed = float(r.get("elapsed", 0.0) or 0.0)
        if elapsed <= 0:
            raise RuntimeError(f"invalid benchmark elapsed time: {elapsed}")
        tps = tokens / elapsed
        results.append({"tokens": tokens, "elapsed": elapsed, "tps": tps})

    all_tps = [r["tps"] for r in results]
    return {
        "model": model,
        "runs": results,
        "median_tps": statistics.median(all_tps),
        "min_tps": min(all_tps),
        "max_tps": max(all_tps),
    }


async def list_served_models(port: int | str, timeout: int = 5) -> list[str]:
    """GET /v1/models → id 리스트. 실패 시 []. 실패 원인은 DEBUG 로그로.

    Callers (benchmark, readiness) treat `[]` as 'no model served yet'
    regardless of cause; DEBUG logging lets `--log-level=DEBUG` surface the
    actual exception without breaking that contract.
    """
    loop = asyncio.get_running_loop()

    def _do() -> list[str]:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout
            ) as r:
                d = json.loads(r.read())
            return [m.get("id", "") for m in d.get("data", []) if m.get("id")]
        except Exception as exc:
            log.debug("list_served_models(%s) failed: %s: %s",
                      port, type(exc).__name__, exc)
            return []

    return await loop.run_in_executor(None, _do)
