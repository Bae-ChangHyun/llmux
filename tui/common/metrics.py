"""Prometheus `/metrics` polling → live token throughput (tok/s).

Both servers expose cumulative token counters on `/metrics`:

* vLLM (on by default):   `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`
* llama-server (`--metrics`): `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`

Counter *names* have drifted across releases, so the parser matches both
families by prefix rather than pinning an exact metric id, and sums every
label-set line for a given counter (vLLM labels its counters by model name).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import urllib.request

log = logging.getLogger(__name__)

# Line-prefix → which counter it feeds. Matched in order; the first hit wins.
_PROMPT_PREFIXES = (
    "vllm:prompt_tokens_total",
    "llamacpp:prompt_tokens_total",
)
_GENERATION_PREFIXES = (
    "vllm:generation_tokens_total",
    "llamacpp:tokens_predicted_total",
)

# Extra live-monitor metrics. Counters/gauges: match the metric name exactly and
# sum every label-set line. Both engines name the same concept differently, so
# each field lists both. Absent families stay None (server without that metric).
_COUNTER_GAUGE_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": _PROMPT_PREFIXES,
    "generation_tokens": _GENERATION_PREFIXES,
    "requests_running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
    "requests_waiting": ("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
    "kv_cache_usage": ("vllm:gpu_cache_usage_perc", "llamacpp:kv_cache_usage_ratio"),
}
# Histogram bases (vLLM only). avg = _sum / _count. `_bucket` lines are ignored.
_HISTOGRAM_BASES: dict[str, str] = {
    "ttft": "vllm:time_to_first_token_seconds",
    "tpot": "vllm:time_per_output_token_seconds",
}


@dataclass
class ServerMetrics:
    """A single /metrics snapshot. Every field is None when the engine doesn't
    expose that family (e.g. llama.cpp has no TTFT/TPOT histograms)."""

    prompt_tokens: float | None = None       # cumulative counter
    generation_tokens: float | None = None   # cumulative counter
    requests_running: float | None = None    # gauge
    requests_waiting: float | None = None    # gauge
    kv_cache_usage: float | None = None       # gauge, 0..1
    ttft_sum: float | None = None            # histogram sum (seconds)
    ttft_count: float | None = None
    tpot_sum: float | None = None
    tpot_count: float | None = None

    def token_counters(self) -> tuple[float, float] | None:
        if self.prompt_tokens is None and self.generation_tokens is None:
            return None
        return (self.prompt_tokens or 0.0, self.generation_tokens or 0.0)


def _metric_name(line: str) -> str:
    return line.split("{", 1)[0].split(None, 1)[0]


def _metric_value(line: str) -> float | None:
    parts = line.rsplit(None, 1)
    if len(parts) != 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def parse_server_metrics(text: str) -> ServerMetrics:
    """Parse the live-monitor fields out of a Prometheus exposition body.

    Fields absent from the body stay None. Counter/gauge values are summed
    across label sets; histogram `_sum`/`_count` are summed separately so the
    caller can derive a per-window average from successive samples.
    """
    sums: dict[str, float] = {}
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = _metric_name(line)

        for field, prefixes in _COUNTER_GAUGE_FIELDS.items():
            if name in prefixes:
                value = _metric_value(line)
                if value is not None:
                    sums[field] = sums.get(field, 0.0) + value
                    seen.add(field)
                break
        else:
            for field, base in _HISTOGRAM_BASES.items():
                if name == f"{base}_sum" or name == f"{base}_count":
                    value = _metric_value(line)
                    if value is not None:
                        key = f"{field}_sum" if name.endswith("_sum") else f"{field}_count"
                        sums[key] = sums.get(key, 0.0) + value
                        seen.add(key)
                    break

    return ServerMetrics(**{k: sums[k] for k in seen})


def parse_token_counters(text: str) -> tuple[float, float] | None:
    """Sum prompt/generation token counters out of a Prometheus exposition body.

    Returns `(prompt_tokens, generation_tokens)`, or None when neither counter
    is present (server up but metrics disabled, or an unknown metric naming
    scheme — either way there is nothing to derive a rate from).
    """
    prompt = 0.0
    generation = 0.0
    found = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(_PROMPT_PREFIXES):
            bucket = "prompt"
        elif line.startswith(_GENERATION_PREFIXES):
            bucket = "generation"
        else:
            continue

        # `name{label="v"} 123.0` or `name 123.0` — the value is the last field.
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue

        if bucket == "prompt":
            prompt += value
        else:
            generation += value
        found = True

    return (prompt, generation) if found else None


async def fetch_token_counters(
    port: int | str, timeout: int = 2
) -> tuple[float, float] | None:
    """GET /metrics → (prompt_tokens, generation_tokens). None if unreachable.

    A stopped container, a server without `--metrics`, and a bad response all
    collapse to None on purpose — the caller renders "n/a" and keeps polling.
    """
    loop = asyncio.get_running_loop()

    def _do() -> tuple[float, float] | None:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=timeout
            ) as r:
                body = r.read().decode(errors="replace")
        except Exception as exc:
            log.debug(
                "fetch_token_counters(%s) failed: %s: %s",
                port,
                type(exc).__name__,
                exc,
            )
            return None
        return parse_token_counters(body)

    return await loop.run_in_executor(None, _do)


async def fetch_server_metrics(
    port: int | str, timeout: int = 2
) -> ServerMetrics | None:
    """GET /metrics → ServerMetrics, or None if the server is unreachable."""
    loop = asyncio.get_running_loop()

    def _do() -> ServerMetrics | None:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=timeout
            ) as r:
                body = r.read().decode(errors="replace")
        except Exception as exc:
            log.debug(
                "fetch_server_metrics(%s) failed: %s: %s",
                port, type(exc).__name__, exc,
            )
            return None
        return parse_server_metrics(body)

    return await loop.run_in_executor(None, _do)


class ThroughputTracker:
    """Turn cumulative counters into per-second rates via successive deltas."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[float, float, float]] = {}

    def update(
        self, key: str, counters: tuple[float, float], now: float
    ) -> tuple[float, float] | None:
        """Feed one sample; return `(prompt_tps, generation_tps)` or None.

        None on the first sample for a key (no baseline to diff against) and
        whenever a counter goes backwards — a server restart resets counters to
        zero, and diffing across that would emit a large negative rate. Reset
        the baseline instead and wait for the next tick.
        """
        prompt, generation = counters
        prev = self._last.get(key)
        self._last[key] = (prompt, generation, now)

        if prev is None:
            return None

        prev_prompt, prev_generation, prev_now = prev
        dt = now - prev_now
        if dt <= 0:
            return None
        if prompt < prev_prompt or generation < prev_generation:
            return None

        return (
            (prompt - prev_prompt) / dt,
            (generation - prev_generation) / dt,
        )

    def forget(self, key: str) -> None:
        self._last.pop(key, None)
