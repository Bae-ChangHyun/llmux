"""HF 모델 메모리 추정 (hf-mem) — backend agnostic."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        env[k.strip()] = v
    return env


async def estimate_model_memory(model_id: str, hf_token: str | None = None) -> str:
    """HF repo id → "~X.YGB" 문자열. 실패 시 사람이 읽을 수 있는 에러 문자열."""
    try:
        from hf_mem import arun  # type: ignore[import-not-found]

        if hf_token is None:
            project_root = Path(__file__).resolve().parents[2]
            common_env = _parse_env_file(project_root / ".env.common")
            hf_token = common_env.get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")

        kwargs: dict = {"model_id": model_id, "experimental": True}
        if hf_token and not hf_token.startswith("your_"):
            kwargs["hf_token"] = hf_token

        try:
            result = await arun(**kwargs)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "kv-cache-dtype" in msg or "kv_cache_dtype" in msg:
                kwargs["kv_cache_dtype"] = "fp8"
                result = await arun(**kwargs)
            else:
                raise

        mem_bytes = getattr(result, "memory", 0) or 0
        kv_bytes = getattr(result, "kv_cache", 0) or 0
        total_bytes = getattr(result, "total_memory", None) or (mem_bytes + kv_bytes)
        if not total_bytes:
            return "estimation failed (no data)"
        total_gb = total_bytes / (1024**3)
        mem_gb = mem_bytes / (1024**3)
        kv_gb = kv_bytes / (1024**3)
        if kv_gb > 0:
            return f"~{total_gb:.1f}GB (model: {mem_gb:.1f}GB + KV: {kv_gb:.1f}GB)"
        return f"~{total_gb:.1f}GB"
    except Exception as exc:
        err = str(exc)
        if "403" in err:
            return "gated model - HF_TOKEN required"
        if "404" in err or "not found" in err.lower():
            return "model not found on HuggingFace"
        return "estimation failed"
