"""System info commands: GPU, HF model memory estimate, env validation."""

from __future__ import annotations

from dataclasses import asdict

import typer

from tui.cli._runtime import emit_json, emit_table, run_async

app = typer.Typer(help="System info & env validation.", no_args_is_help=True)


@app.command("gpu")
def gpu(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Print nvidia-smi summary (one row per GPU)."""
    from tui.common.docker import get_gpu_info

    from tui.common.docker import gpu_probe_failed

    gpus = run_async(get_gpu_info())
    rows = [asdict(g) for g in gpus]
    # A CPU-only host legitimately has zero GPUs; only a present-but-broken
    # nvidia-smi is an error.
    failed = not rows and run_async(gpu_probe_failed())
    if json_out:
        emit_json(rows)
        raise typer.Exit(code=1 if failed else 0)
    if not rows:
        print(
            "(nvidia-smi is installed but failed — check the driver)" if failed
            else "(no GPUs detected — is nvidia-smi installed?)"
        )
        raise typer.Exit(code=1 if failed else 0)
    emit_table(
        rows,
        columns=["index", "name", "memory_used", "memory_total", "utilization", "temperature"],
    )


@app.command("mem-estimate")
def mem_estimate(
    model_id: str = typer.Argument(..., help="Hugging Face model id, e.g. Qwen/Qwen3-8B."),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit a JSON record with parsed est_gb + per-GPU ratios.",
    ),
) -> None:
    """Estimate VRAM footprint + per-GPU fit (mirrors the dashboard mem panel).

    The dashboard renders the same hf-mem estimate next to live GPU bars; this
    CLI command does the same calculation with text output so scripts/agents
    can see whether the model would fit on each available GPU under the
    implicit `TP = len(GPUs)` split the TUI uses.
    """
    import re as _re

    from tui.common.docker import get_gpu_info
    from tui.common.mem import estimate_model_memory

    async def _gather() -> tuple[str, list]:
        estimate = await estimate_model_memory(model_id)
        gpus = await get_gpu_info()
        return estimate, gpus

    estimate, gpus = run_async(_gather())

    match = _re.search(r"~([\d.]+)GB", estimate)
    est_gb = float(match.group(1)) if match else 0.0
    n_gpus = len(gpus)
    per_gpu_gb = est_gb / n_gpus if (n_gpus > 1 and est_gb > 0) else est_gb

    fit_rows: list[dict] = []
    any_over = False
    for g in gpus:
        try:
            total_gb = int(g.memory_total) / 1024
        except (TypeError, ValueError):
            total_gb = 0.0
        ratio = per_gpu_gb / total_gb if total_gb > 0 else 0.0
        over = ratio > 1.0
        if over:
            any_over = True
        fit_rows.append(
            {
                "index": g.index,
                "name": g.name,
                "total_gb": round(total_gb, 1),
                "per_gpu_gb": round(per_gpu_gb, 1),
                "ratio": round(ratio, 3),
                "over": over,
            }
        )

    if json_out:
        emit_json(
            {
                "model_id": model_id,
                "estimate": estimate,
                "est_gb": est_gb,
                "n_gpus": n_gpus,
                "per_gpu_gb": round(per_gpu_gb, 2),
                "gpus": fit_rows,
                "any_over": any_over,
                "estimated": est_gb > 0,
            }
        )
        # est_gb == 0 means the estimate never parsed (gated model, network
        # failure, ...). Reporting any_over=false there would read as "it fits".
        raise typer.Exit(code=0 if (est_gb > 0 and not any_over) else 1)

    print(estimate)
    if not gpus:
        print("(no GPUs detected — skipping per-GPU fit view)")
        return
    if est_gb <= 0:
        print("(no parseable size in estimate — skipping per-GPU fit view)")
        raise typer.Exit(code=1)
    if n_gpus > 1:
        print(f"TP={n_gpus}: {per_gpu_gb:.1f} GB/GPU")
    for r in fit_rows:
        if r["over"]:
            print(
                f"  GPU{r['index']}  OVER  {r['per_gpu_gb']:.1f}/{r['total_gb']:.0f} GB"
            )
        else:
            print(
                f"  GPU{r['index']}  {r['ratio'] * 100:3.0f}%  "
                f"{r['per_gpu_gb']:.1f}/{r['total_gb']:.0f} GB"
            )
    if any_over:
        raise typer.Exit(code=1)


@app.command("disk")
def disk(
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a human-readable summary."
    ),
) -> None:
    """Show llama.cpp model dir, GGUF inventory + sizes, and df -h usage.

    Mirrors the llama.cpp TUI System screen's "Disk / Model Dir" tab — useful
    for headless health-checks when GGUF downloads keep refilling a model
    cache. Reuses `_get_model_dir` and `get_disk_usage` from the llama.cpp
    backend so behavior stays in lockstep with the TUI panel.
    """
    from tui.backends.llamacpp.backend import (
        ROOT,
        _get_hf_cache_dir,
        _get_model_dir,
        get_disk_usage,
        list_cached_gguf,
    )

    model_dir = _get_model_dir()
    hf_cache_dir = _get_hf_cache_dir()
    cached = list_cached_gguf()
    cached_total_gb = round(sum(c["size_bytes"] for c in cached) / 1024**3, 1)
    files: list[dict] = []
    if model_dir.exists():
        for f in sorted(
            model_dir.glob("*.gguf"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        ):
            files.append(
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "size_gb": round(f.stat().st_size / 1024**3, 1),
                }
            )

    target = str(model_dir if model_dir.exists() else ROOT)
    used, avail, pct = run_async(get_disk_usage(target))

    total_gb = round(sum(f["size_bytes"] for f in files) / 1024**3, 1)

    if json_out:
        emit_json(
            {
                "project_root": str(ROOT),
                "model_dir": str(model_dir),
                "model_dir_exists": model_dir.exists(),
                "hf_cache_dir": str(hf_cache_dir),
                "df_target": target,
                "df_used": used,
                "df_avail": avail,
                "df_percent": pct,
                "gguf_files": files,
                "gguf_total_gb": total_gb,
                "hf_cache_gguf_files": cached,
                "hf_cache_gguf_total_gb": cached_total_gb,
            }
        )
        return

    print(f"Project root : {ROOT}")
    print(f"Model dir    : {model_dir}")
    if not model_dir.exists():
        print("(model dir does not exist yet — no GGUF files to list)")
    else:
        print(f"GGUF files   : {len(files)}  (total {total_gb:.1f} GB)")
        for f in files:
            print(f"  {f['name']}  {f['size_gb']:.1f} GB")
    # llama-server's `-hf` download lands here, not in MODEL_DIR.
    print(f"HF cache dir : {hf_cache_dir}")
    print(f"HF cache GGUF: {len(cached)}  (total {cached_total_gb:.1f} GB)")
    for c in cached:
        print(f"  {c['repo']}/{c['name']}  {c['size_gb']:.1f} GB")
    if used:
        print(f"df {target}: used {used}, avail {avail}  ({pct})")


@app.command("env-check")
def env_check(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Validate `.env.common` and report key paths."""
    from tui.common.profile_store import PROJECT_ROOT
    from tui.common.env import parse_env_file, validate_common_env

    common = PROJECT_ROOT / ".env.common"
    # Defer the verdict to the same validator the start path gates on, instead
    # of a second, stricter list. HF_TOKEN is optional (public repos need none)
    # and VLLM_VERSION isn't even in .env.common.example — the runtime injects
    # it into the compose env at start time — so requiring both here failed a
    # perfectly good default setup.
    ok, messages = validate_common_env(common)
    findings = {
        "project_root": str(PROJECT_ROOT),
        "env_common_path": str(common),
        "env_common_exists": common.exists(),
        "issues": [] if ok else list(messages),
    }
    env = parse_env_file(common) if common.exists() else {}
    for key in ("HF_TOKEN", "HF_CACHE_PATH"):
        findings[key] = env.get(key, "")

    findings["status"] = "ok" if ok else "error"

    if json_out:
        emit_json(findings)
        raise typer.Exit(code=0 if ok else 1)

    print(f"Project root : {findings['project_root']}")
    print(f"Env common   : {findings['env_common_path']}")
    if not findings["env_common_exists"]:
        print("Status       : MISSING .env.common")
        raise typer.Exit(code=1)

    token = findings["HF_TOKEN"]
    # Mask token but reveal length so users can spot truncation issues.
    print(
        f"  {'HF_TOKEN':<14}: "
        + (f"<set, {len(token)} chars>" if token else "(unset — optional, needed for gated models)")
    )
    print(f"  {'HF_CACHE_PATH':<14}: {findings['HF_CACHE_PATH'] or '(unset)'}")
    print(f"  {'VLLM_VERSION':<14}: (injected at start time)")

    if findings["issues"]:
        print("\nIssues:")
        for i in findings["issues"]:
            print(f"  - {i}")
        raise typer.Exit(code=1)
    print("\nStatus       : OK")
