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
from dataclasses import dataclass, field
import logging
import math
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

# Live-monitor counter/gauge fields. Each maps to the metric-name families that
# express the same concept; llama.cpp and vLLM name them differently, and vLLM
# has renamed some across releases (kv_cache_usage_perc ← gpu_cache_usage_perc).
# The first family seen for a field wins, so a server exposing both the old and
# new name never double-counts; within one family, label sets are summed.
_COUNTER_GAUGE_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": _PROMPT_PREFIXES,
    "generation_tokens": _GENERATION_PREFIXES,
    "requests_running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
    "requests_waiting": ("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
    "kv_cache_usage": (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "llamacpp:kv_cache_usage_ratio",
    ),
    "prefix_hits": ("vllm:prefix_cache_hits_total",),
    "prefix_queries": ("vllm:prefix_cache_queries_total",),
    "ext_prefix_hits": ("vllm:external_prefix_cache_hits_total",),
    "ext_prefix_queries": ("vllm:external_prefix_cache_queries_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "requests_finished": ("vllm:request_success_total",),
    "prompt_tps_gauge": ("llamacpp:prompt_tokens_seconds",),
    "gen_tps_gauge": ("llamacpp:predicted_tokens_seconds",),
}

# Histogram fields → the metric bases that feed them. `inter_token_latency` is
# the current vLLM per-output-token histogram; `time_per_output_token` is its
# older name. vLLM only — llama.cpp exposes no latency histograms.
_HISTOGRAM_BASES: dict[str, tuple[str, ...]] = {
    "ttft": ("vllm:time_to_first_token_seconds",),
    "e2e": ("vllm:e2e_request_latency_seconds",),
    "tpot": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
    "queue": ("vllm:request_queue_time_seconds",),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
    "infer": ("vllm:request_inference_time_seconds",),
}


@dataclass
class Hist:
    """One Prometheus histogram, summed across label sets."""

    sum: float = 0.0
    count: float = 0.0
    buckets: dict[float, float] = field(default_factory=dict)  # le → cumulative

    def avg(self) -> float | None:
        return self.sum / self.count if self.count > 0 else None

    def quantile(self, q: float) -> float | None:
        """Approximate the q-quantile (0..1) from cumulative buckets, Prometheus
        `histogram_quantile` style with linear interpolation inside the bucket."""
        if self.count <= 0 or not self.buckets:
            return None
        ordered = sorted(self.buckets.items())  # by le; math.inf sorts last
        target = q * self.count
        prev_le, prev_c = 0.0, 0.0
        for le, cum in ordered:
            if cum >= target:
                if math.isinf(le):
                    return prev_le or None
                span = cum - prev_c
                if span <= 0:
                    return le
                return prev_le + (le - prev_le) * (target - prev_c) / span
            prev_le, prev_c = le, cum
        return prev_le or None


@dataclass
class MetricsSnapshot:
    """A single /metrics snapshot. Every field stays None when the engine does
    not expose that family (e.g. llama.cpp has no latency histograms, no prefix
    cache breakdown, no preemption counter)."""

    backend: str = "unknown"                 # "vllm" | "llamacpp" | "unknown"
    prompt_tokens: float | None = None       # cumulative counter
    generation_tokens: float | None = None   # cumulative counter
    requests_running: float | None = None    # gauge
    requests_waiting: float | None = None    # gauge
    kv_cache_usage: float | None = None      # gauge, 0..1
    prefix_hits: float | None = None         # counter (tokens)
    prefix_queries: float | None = None      # counter (tokens)
    ext_prefix_hits: float | None = None     # counter (tokens)
    ext_prefix_queries: float | None = None  # counter (tokens)
    preemptions: float | None = None         # counter
    requests_finished: float | None = None   # counter (request_success_total)
    prompt_tps_gauge: float | None = None    # llama.cpp reports tok/s directly
    gen_tps_gauge: float | None = None
    ttft: Hist | None = None
    e2e: Hist | None = None
    tpot: Hist | None = None
    queue: Hist | None = None
    prefill: Hist | None = None
    decode: Hist | None = None
    infer: Hist | None = None

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


def _bucket_le(line: str) -> float | None:
    """Extract the `le` label from a histogram `_bucket` line, or None."""
    lo = line.find('le="')
    if lo < 0:
        return None
    lo += 4
    hi = line.find('"', lo)
    if hi < 0:
        return None
    token = line[lo:hi]
    try:
        return math.inf if token in ("+Inf", "Inf") else float(token)
    except ValueError:
        return None


def parse_snapshot(text: str) -> MetricsSnapshot:
    """Parse a full monitor snapshot out of a Prometheus exposition body.

    Counter/gauge fields are summed across label sets (first-seen family wins).
    Histograms accumulate `_sum`, `_count`, and every `_bucket{le=…}` so the
    caller can derive both window averages and quantiles.
    """
    cg: dict[str, float] = {}
    chosen: dict[str, str] = {}   # field → the family name that first matched it
    hists: dict[str, Hist] = {}
    backend = "unknown"

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = _metric_name(line)
        if backend == "unknown":
            if name.startswith("vllm:"):
                backend = "vllm"
            elif name.startswith("llamacpp:"):
                backend = "llamacpp"

        for fld, families in _COUNTER_GAUGE_FIELDS.items():
            if name in families:
                if chosen.get(fld, name) != name:
                    break  # a different family already owns this field
                value = _metric_value(line)
                if value is not None:
                    chosen[fld] = name
                    cg[fld] = cg.get(fld, 0.0) + value
                break
        else:
            for fld, bases in _HISTOGRAM_BASES.items():
                matched = False
                for base in bases:
                    if name == f"{base}_sum":
                        value = _metric_value(line)
                        if value is not None:
                            hists.setdefault(fld, Hist()).sum += value
                        matched = True
                    elif name == f"{base}_count":
                        value = _metric_value(line)
                        if value is not None:
                            hists.setdefault(fld, Hist()).count += value
                        matched = True
                    elif name == f"{base}_bucket":
                        le = _bucket_le(line)
                        value = _metric_value(line)
                        if le is not None and value is not None:
                            b = hists.setdefault(fld, Hist()).buckets
                            b[le] = b.get(le, 0.0) + value
                        matched = True
                    if matched:
                        break
                if matched:
                    break

    return MetricsSnapshot(backend=backend, **cg, **hists)


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


async def fetch_snapshot(
    port: int | str, timeout: int = 2
) -> MetricsSnapshot | None:
    """GET /metrics → MetricsSnapshot, or None if the server is unreachable."""
    loop = asyncio.get_running_loop()

    def _do() -> MetricsSnapshot | None:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/metrics", timeout=timeout
            ) as r:
                body = r.read().decode(errors="replace")
        except Exception as exc:
            log.debug(
                "fetch_snapshot(%s) failed: %s: %s",
                port, type(exc).__name__, exc,
            )
            return None
        return parse_snapshot(body)

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
