from __future__ import annotations

import os

from tui.common.env import parse_env_file
from tui.common.ssl_ctx import redact_sensitive_text


async def estimate_model_memory(model_id: str, hf_token: str | None = None) -> str:
    try:
        from hf_mem import arun  # type: ignore[import-not-found]
    except ImportError as exc:
        return f"hf-mem import failed: {type(exc).__name__}: {exc}"

    try:
        if hf_token is None:
            from tui.common.profile_store import PROJECT_ROOT

            common_env = parse_env_file(PROJECT_ROOT / ".env.common")
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
        if isinstance(mem_bytes, dict):
            return (
                "hf-mem runtime failed: multiple GGUF files found; "
                "select a specific GGUF file"
            )
        kv_bytes = getattr(result, "kv_cache", 0) or 0
        total_bytes = getattr(result, "total_memory", None) or (mem_bytes + kv_bytes)
        if not total_bytes:
            return "hf-mem runtime failed: no memory data returned"
        total_gb = total_bytes / (1024**3)
        mem_gb = mem_bytes / (1024**3)
        kv_gb = kv_bytes / (1024**3)
        if kv_gb > 0:
            return f"~{total_gb:.1f}GB (model: {mem_gb:.1f}GB + KV: {kv_gb:.1f}GB)"
        return f"~{total_gb:.1f}GB"
    except Exception as exc:
        err = redact_sensitive_text(str(exc), (hf_token or "",))
        if "403" in err:
            return "gated model - HF_TOKEN required"
        if "404" in err or "not found" in err.lower():
            return "model not found on HuggingFace"
        exc_module = type(exc).__module__.split(".", 1)[0]
        exc_name = type(exc).__name__
        network_modules = {"aiohttp", "httpcore", "httpx", "requests", "urllib3"}
        network_markers = (
            "connection",
            "network",
            "offline",
            "timed out",
            "timeout",
            "dns",
            "ssl",
            "http status",
        )
        if isinstance(exc, (ConnectionError, TimeoutError)) or (
            exc_module in network_modules
            or any(marker in err.lower() for marker in network_markers)
        ):
            return f"hf-mem network failed: {exc_name}: {err}"
        return f"hf-mem runtime failed: {exc_name}: {err}"
