"""Shared parsing + validation for .env-style files.

Single source of truth for what every backend treats as a well-formed
`.env.common`. Backends used to carry their own near-identical parsers
(`tui.backends.{vllm,llamacpp}.backend{_storage,}._parse_env_file`) and a
mismatched validator (vLLM checked HF_CACHE_PATH up front; llama.cpp let
docker discover the missing/relative path during compose interpolation).
The mismatched validator surfaced as a confusing late failure; this module
removes the asymmetry.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def parse_env_file(path: Path | str) -> dict[str, str]:
    """Parse a .env file into a dict.

    Skips blank lines and comments. Strips matched leading/trailing single or
    double quotes from values, and unwraps single-token shell quoting via
    shlex (so `KEY="foo bar"` and `KEY='foo bar'` both yield `foo bar`).
    Missing file → empty dict, never raises.
    """
    p = Path(path)
    data: dict[str, str] = {}
    if not p.exists():
        return data
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if " #" in value and not value.startswith(("'", '"')):
            value = value[: value.index(" #")].rstrip()
        try:
            parsed = shlex.split(value, comments=False, posix=True)
        except ValueError:
            parsed = []
        if len(parsed) == 1:
            value = parsed[0]
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        data[key.strip()] = value
    return data


def validate_common_env(
    common_env_path: Path, *, require_lora_base_path: bool = False
) -> tuple[bool, list[str]]:
    """Validate `.env.common` for shared cross-backend invariants.

    Returns `(ok, messages)`. Messages are user-facing lines (already prefixed
    with `Error:` or `Warning:` where appropriate) suitable for yielding to a
    log stream.

    Currently enforced:
      * `.env.common` exists.
      * `HF_CACHE_PATH` is set and absolute (compose mounts it directly into
        `/root/.cache/huggingface`, so a missing or relative value fails late
        with a cryptic docker error).
      * When `require_lora_base_path=True` (vLLM with `ENABLE_LORA=true`),
        `LORA_BASE_PATH` is set and absolute.
    """
    if not common_env_path.exists():
        return False, [
            "Error: .env.common not found.",
            "Create it from .env.common.example before starting containers.",
        ]
    common = parse_env_file(common_env_path)
    hf_cache_path = common.get("HF_CACHE_PATH", "")
    if not hf_cache_path:
        return False, ["Error: HF_CACHE_PATH is not set in .env.common"]
    if not os.path.isabs(hf_cache_path):
        return False, [
            f"Error: HF_CACHE_PATH must be an absolute path. Current value: {hf_cache_path}"
        ]
    if require_lora_base_path:
        lora_base_path = common.get("LORA_BASE_PATH", "")
        if not lora_base_path:
            return False, [
                "Error: ENABLE_LORA=true but LORA_BASE_PATH is not set in .env.common"
            ]
        if not os.path.isabs(lora_base_path):
            return False, [
                f"Error: LORA_BASE_PATH must be an absolute path. Current value: {lora_base_path}"
            ]
    return True, []
