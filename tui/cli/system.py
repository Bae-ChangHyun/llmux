"""System info commands: GPU, HF model memory estimate, env validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from tui.cli._runtime import emit_json, emit_table, run_async

app = typer.Typer(help="System info & env validation.", no_args_is_help=True)


@app.command("gpu")
def gpu(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Print nvidia-smi summary (one row per GPU)."""
    from tui.common.docker import get_gpu_info

    gpus = run_async(get_gpu_info())
    rows = [asdict(g) for g in gpus]
    if json_out:
        emit_json(rows)
        return
    if not rows:
        print("(no GPUs detected — is nvidia-smi installed?)")
        return
    emit_table(
        rows,
        columns=["index", "name", "memory_used", "memory_total", "utilization", "temperature"],
    )


@app.command("mem-estimate")
def mem_estimate(
    model_id: str = typer.Argument(..., help="Hugging Face model id, e.g. Qwen/Qwen3-8B."),
) -> None:
    """Estimate VRAM footprint via `hf-mem` (the same engine the TUI uses)."""
    from tui.common.mem import estimate_model_memory

    result = run_async(estimate_model_memory(model_id))
    print(result)


@app.command("env-check")
def env_check(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Validate `.env.common` and report key paths."""
    from tui.common.profile_store import PROJECT_ROOT
    from tui.common.mem import _parse_env_file

    common = PROJECT_ROOT / ".env.common"
    findings = {
        "project_root": str(PROJECT_ROOT),
        "env_common_path": str(common),
        "env_common_exists": common.exists(),
        "issues": [],
    }
    if not common.exists():
        findings["issues"].append(
            "Missing .env.common — copy from .env.common.example and fill in HF_TOKEN/HF_CACHE_PATH."
        )
    else:
        env = _parse_env_file(common)
        for key in ("HF_TOKEN", "HF_CACHE_PATH", "VLLM_VERSION"):
            findings[key] = env.get(key, "")
            if not env.get(key):
                findings["issues"].append(f"{key} not set in .env.common")
        hf_cache = env.get("HF_CACHE_PATH", "")
        if hf_cache and not Path(hf_cache).is_absolute():
            findings["issues"].append(f"HF_CACHE_PATH must be absolute: {hf_cache}")

    findings["status"] = "ok" if not findings["issues"] else "error"

    if json_out:
        emit_json(findings)
        raise typer.Exit(code=0 if findings["status"] == "ok" else 1)

    print(f"Project root : {findings['project_root']}")
    print(f"Env common   : {findings['env_common_path']}")
    if not findings["env_common_exists"]:
        print("Status       : MISSING .env.common")
        raise typer.Exit(code=1)
    for k in ("HF_TOKEN", "HF_CACHE_PATH", "VLLM_VERSION"):
        v = findings.get(k, "") or "(unset)"
        # Mask token but reveal length so users can spot truncation issues.
        if k == "HF_TOKEN" and v and v != "(unset)":
            v = f"<set, {len(v)} chars>"
        print(f"  {k:<14}: {v}")
    if findings["issues"]:
        print("\nIssues:")
        for i in findings["issues"]:
            print(f"  - {i}")
        raise typer.Exit(code=1)
    print("\nStatus       : OK")
