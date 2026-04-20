"""OpenAI 호환 엔드포인트 공통 헬퍼 (vllm / llama-server 재사용)."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request


async def chat_completion_bench(
    port: int | str,
    model: str,
    prompt: str = "Explain the theory of relativity in about 150 words.",
    max_tokens: int = 200,
    timeout: int = 600,
) -> dict:
    """단일 /v1/chat/completions 호출 → {elapsed, usage}."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
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


async def list_served_models(port: int | str, timeout: int = 5) -> list[str]:
    """GET /v1/models → id 리스트. 실패 시 []."""
    loop = asyncio.get_running_loop()

    def _do() -> list[str]:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout
            ) as r:
                d = json.loads(r.read())
            return [m.get("id", "") for m in d.get("data", []) if m.get("id")]
        except Exception:
            return []

    return await loop.run_in_executor(None, _do)
