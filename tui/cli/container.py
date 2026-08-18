"""Container lifecycle commands: up / down / logs / ps / render-env."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer

from tui.cli._runtime import (
    BACKENDS,
    detect_backend,
    docker_logs_follow,
    docker_logs_once,
    emit_json,
    emit_table,
    gather_conflict_warnings,
    run_async,
    stream_async,
)
from tui.common import profile_store

app = typer.Typer(help="Container lifecycle (start, stop, logs, status).", no_args_is_help=True)


class _StatusProbeFailed(RuntimeError):
    """`docker ps` failed, so container state is unknown rather than stopped."""



@app.command("up")
def up(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help=f"Force backend ({', '.join(BACKENDS)}); auto-detect if omitted."
    ),
    tag: str = typer.Option(
        "", "--tag", "-t",
        help="Image tag override. Without --dev: a full image ref "
        "(vLLM vllm/vllm-openai:<tag>, llama.cpp ghcr.io/foo/bar:v1). "
        "With --dev: pass just the dev tag (it's sanitized and resolved to "
        "<backend>-dev:<tag>). Empty = profile's pinned image_tag or backend default.",
    ),
    dev: bool = typer.Option(
        False, "--dev",
        help=(
            "Use the locally-built dev image (vllm-dev:<tag> for vLLM, "
            "llamacpp-dev:<tag> for llama.cpp). Triggers a one-off build if the "
            "tag is missing locally or was built from a different repo/branch."
        ),
    ),
    default_image: bool = typer.Option(
        False, "--default-image",
        help=(
            "Mirror the TUI's 'Default Image' selection — explicitly drop the "
            "profile's pinned image_tag for this run only and fall back to the "
            "compose default image. Useful when a profile has a stale dev tag "
            "pinned and you want to sanity-check against the upstream image "
            "without editing profiles.yaml."
        ),
    ),
    pull: bool = typer.Option(
        False, "--pull", help="vLLM only: force --pull always when bringing the container up."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url",
        help="--dev only: override the source repo URL (default from .env.common per backend).",
    ),
    branch: str = typer.Option(
        "", "--branch",
        help="--dev only: override the source branch (default from .env.common per backend).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Skip the cross-backend port/GPU conflict pre-flight (matches the "
             "TUI dashboard's pre-start gate). Required to start a container "
             "when llmux detects a conflict.",
    ),
) -> None:
    """Start a profile's container. Streams compose output to stdout."""
    # Flag sanity first — these are contradictions in the invocation itself and
    # don't depend on the profile resolving.
    if dev and default_image:
        typer.echo(
            "Error: --dev and --default-image are mutually exclusive (one forces a "
            "dev tag, the other clears any pinned image).",
            err=True,
        )
        raise typer.Exit(code=2)

    if tag and default_image:
        # Contradictory: one names an image, the other says "use the default".
        # Rejecting up-front also removes a cross-backend divergence — llama.cpp
        # ignored the tag here while vLLM honored it.
        typer.echo(
            "Error: --tag and --default-image are mutually exclusive (one names an "
            "image, the other falls back to the compose default).",
            err=True,
        )
        raise typer.Exit(code=2)

    bk = detect_backend(profile, override=backend)

    # Conflict pre-flight: mirror the TUI dashboard. The headless CLI defaults
    # to ABORT-on-conflict so scripts/agents don't silently start a container
    # that competes with another for the same port or GPU. `--force` opts out.
    if not force:
        warnings = run_async(gather_conflict_warnings(profile, bk))
        if warnings:
            typer.echo(
                f"Conflict pre-flight: cannot start '{profile}' ({bk}) — "
                f"{len(warnings)} issue(s) detected:",
                err=True,
            )
            for w in warnings:
                typer.echo(f"  • {w}", err=True)
            typer.echo(
                "Aborting. Re-run with --force to start anyway.", err=True
            )
            raise typer.Exit(code=1)

    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_up

        rc = stream_async(
            stream_container_up(
                profile,
                use_dev=dev,
                use_default_image=default_image,
                tag=tag,
                pull=pull,
                repo_url=repo_url,
                branch=branch,
            )
        )
    else:
        from tui.backends.llamacpp.backend_runtime import stream_container_up as lc_up

        if pull:
            typer.echo(
                "Warning: --pull is vLLM-only and ignored for llama.cpp",
                err=True,
            )
        rc = stream_async(
            lc_up(
                profile,
                use_dev=dev,
                use_default_image=default_image,
                tag=tag,
                repo_url=repo_url,
                branch=branch,
            )
        )
    raise typer.Exit(code=rc)



@app.command("prepare")
def prepare(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
) -> None:
    """Download the model and render the runtime files — without starting it.

    Does everything `up` does up to the point of launching the server: renders
    the profile env (and llama.cpp's command override), makes sure the image is
    on disk, then downloads the weights into the HF cache using a throwaway
    container. No server is started and no GPU is touched, so a later
    `llmux up <profile>` only has to load what is already there.
    """
    bk = detect_backend(profile, override=backend)
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_prepare
    else:
        from tui.backends.llamacpp.backend_runtime import stream_container_prepare

    rc = stream_async(stream_container_prepare(profile))
    raise typer.Exit(code=rc)


@app.command("down")
def down(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
) -> None:
    """Stop a profile's container."""
    bk = detect_backend(profile, override=backend)
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import container_down

        rc, msg = run_async(container_down(profile))
    else:
        from tui.backends.llamacpp.backend_runtime import container_down as lc_down

        rc, msg = run_async(lc_down(profile))
    if msg:
        print(msg.rstrip())
    raise typer.Exit(code=rc)



@app.command("logs")
def logs(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    tail: int = typer.Option(
        200, "--tail", "-n", help="Number of recent lines to show before following."
    ),
    follow: bool = typer.Option(
        True, "--follow/--no-follow", "-f/-F",
        help="Follow log output (default). Use --no-follow to print recent lines and exit.",
    ),
) -> None:
    """Stream container logs (Ctrl-C to stop following)."""
    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    container_name = sp.container_name or sp.name

    # Both paths go through the shared wrapper so the exit code is the child
    # `docker logs` status on either side of --follow.
    runner = docker_logs_follow if follow else docker_logs_once
    try:
        rc = run_async(runner(container_name, tail=tail))
    except KeyboardInterrupt:
        rc = 130
    raise typer.Exit(code=rc)



@app.command("benchmark")
def benchmark(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    prompt: str = typer.Option(
        "Explain the theory of relativity in about 150 words.", "--prompt",
        help="Prompt to send to /v1/chat/completions.",
    ),
    max_tokens: int = typer.Option(
        200, "--max-tokens", help="max_tokens for the bench request."
    ),
    runs: int = typer.Option(
        3, "--runs", help="Measured runs; the reported figure is their median."
    ),
    warmup: int = typer.Option(
        1, "--warmup", help="Warmup runs discarded before measuring."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit a JSON record instead of a human line."
    ),
) -> None:
    """Benchmark a running container (tok/s). Mirrors the TUI dashboard bench."""
    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    if not sp.port:
        typer.echo(f"profile '{profile}' has no port set", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        # Match the dashboard's per-backend dispatch: vLLM discovers the
        # served model via /v1/models, llama.cpp uses the config `alias` (or
        # the config name) since llama-server may not expose multiple models.
        from tui.common.http import list_served_models, run_bench

        if bk == "vllm":
            served = await list_served_models(sp.port)
            model = served[0] if served else (sp.model_id or "")
            if not model:
                raise RuntimeError(
                    "could not identify a served model "
                    "(/v1/models returned nothing and profile has no model_id)"
                )
        else:
            from tui.backends.llamacpp.backend import load_config as l_load_config

            cfg = l_load_config(sp.config_name or sp.name)
            model = str(cfg.get("alias", cfg.name))

        r = await run_bench(
            sp.port,
            model,
            prompt=prompt,
            max_tokens=max_tokens,
            runs=runs,
            warmup=warmup,
        )
        return {
            "profile": profile,
            "backend": bk,
            "port": sp.port,
            "warmup": warmup,
            **r,
        }

    try:
        result = run_async(_run())
    except Exception as exc:
        typer.echo(f"benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if json_out:
        emit_json(result)
        return
    head = f"{result['profile']} ({result['backend']}, port {result['port']})"
    print(f"{head}  [{result['model']}]  warmup={result['warmup']}")
    for i, r in enumerate(result["runs"], start=1):
        print(
            f"  run {i}: {r['tokens']} tok / {r['elapsed']:.2f}s "
            f"= {r['tps']:.1f} tok/s"
        )
    print(
        f"  median: {result['median_tps']:.1f} tok/s "
        f"({result['min_tps']:.1f}–{result['max_tps']:.1f})"
    )



@app.command("stats")
def stats(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help=f"Limit to one backend ({', '.join(BACKENDS)}); show all if omitted.",
    ),
    interval: float = typer.Option(
        2.0, "--interval", "-i", help="Seconds between /metrics samples."
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Take two samples one interval apart, print one tick, then exit.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one compact JSON line per tick (NDJSON)."
    ),
) -> None:
    """Live token throughput (tok/s) polled from each running container's /metrics.

    Rates are deltas between successive samples, so the first sample only
    establishes a baseline — the first line appears one `--interval` later.
    A profile whose server is down or was started without metrics exposed
    shows `n/a` and keeps being polled rather than aborting the run.
    """
    import json as _json
    import time

    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    if interval <= 0:
        # asyncio.sleep(0) would spin the poll loop as fast as the event loop
        # can schedule it, hammering /metrics and pinning a core.
        raise typer.BadParameter("interval must be > 0", param_hint="--interval")
    backends = [backend] if backend else list(BACKENDS)

    from tui.common.metrics import ThroughputTracker, fetch_token_counters

    tracker = ThroughputTracker()

    async def _running() -> list[dict]:
        out: list[dict] = []
        for bk in backends:
            if bk == "vllm":
                from tui.backends.vllm.backend_runtime import get_container_statuses
            else:
                from tui.backends.llamacpp.backend_runtime import (
                    get_container_statuses,
                )
            try:
                statuses = await get_container_statuses()
            except RuntimeError as exc:
                raise _StatusProbeFailed(f"{bk}: {exc}") from exc
            for s in statuses:
                if s.running and s.port:
                    out.append(
                        {"backend": bk, "profile": s.profile_name, "port": s.port}
                    )
        out.sort(key=lambda r: (r["backend"], r["profile"]))
        return out

    async def _sample() -> list[dict]:
        rows: list[dict] = []
        targets = await _running()
        # Fetch concurrently (same as the dashboard's _poll_throughput): polling
        # in turn let one slow/hanging /metrics endpoint delay every profile
        # behind it, skewing the deltas of the ones that answered promptly.
        samples = await asyncio.gather(
            *(fetch_token_counters(t["port"]) for t in targets),
            return_exceptions=True,
        )
        # One timestamp for the whole batch, taken after the fetches resolve.
        now = time.monotonic()
        for t, counters in zip(targets, samples):
            key = f"{t['backend']}:{t['profile']}"
            if isinstance(counters, BaseException):
                counters = None
            if counters is None:
                # Unreachable / no metrics — drop the baseline so a later
                # restart doesn't diff against a stale pre-restart counter.
                tracker.forget(key)
                rate = None
            else:
                rate = tracker.update(key, counters, now)
            rows.append(
                {
                    **t,
                    "prompt_tok_per_s": rate[0] if rate else None,
                    "generation_tok_per_s": rate[1] if rate else None,
                }
            )
        return rows

    def _emit(rows: list[dict]) -> None:
        if json_out:
            print(_json.dumps(rows, ensure_ascii=False, default=str), flush=True)
            return
        if not rows:
            print("(no running containers)", flush=True)
            return
        for r in rows:
            head = f"{r['profile']} ({r['backend']}, port {r['port']})"
            if r["generation_tok_per_s"] is None:
                print(f"{head}: n/a", flush=True)
            else:
                print(
                    f"{head}: prompt {r['prompt_tok_per_s']:.1f} tok/s"
                    f"  ·  gen {r['generation_tok_per_s']:.1f} tok/s",
                    flush=True,
                )

    async def _drive() -> None:
        baseline = True
        while True:
            rows = await _sample()
            if baseline:
                # First pass only seeds the tracker; with nothing running there
                # will never be a rate, so say so instead of hanging silently.
                baseline = False
                if not rows:
                    _emit(rows)
                    return
            else:
                _emit(rows)
                if once:
                    return
            await asyncio.sleep(interval)

    try:
        run_async(_drive())
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    except _StatusProbeFailed as exc:
        typer.echo(f"container status probe failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc



@app.command("ps")
def ps(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help=f"Limit to one backend ({', '.join(BACKENDS)}); show all if omitted.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    running_only: bool = typer.Option(
        False, "--running", "-r", help="Only show running containers."
    ),
) -> None:
    """List profiles and their container status across backends."""
    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    backends = [backend] if backend else list(BACKENDS)
    rows = []

    async def _collect():
        # Both backends expose the identical `get_container_statuses()`
        # contract (field-for-field mirror — see llamacpp.backend_runtime
        # ContainerStatus docstring), so dispatch once and reuse the same
        # row-builder for both.
        for bk in backends:
            if bk == "vllm":
                from tui.backends.vllm.backend_runtime import get_container_statuses
            else:
                from tui.backends.llamacpp.backend_runtime import (
                    get_container_statuses,
                )

            try:
                statuses = await get_container_statuses()
            except RuntimeError as exc:
                raise _StatusProbeFailed(f"{bk}: {exc}") from exc
            for s in statuses:
                rows.append(
                    {
                        "backend": bk,
                        "profile": s.profile_name,
                        "container": s.container_name,
                        "status": s.status_text,
                        "running": s.running,
                        "port": s.port,
                        "gpu": s.gpu_id,
                        "model": s.model,
                    }
                )

    try:
        run_async(_collect())
    except _StatusProbeFailed as exc:
        typer.echo(f"container status probe failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if running_only:
        rows = [r for r in rows if r["running"]]

    if json_out:
        emit_json(rows)
        return

    emit_table(
        rows,
        columns=["backend", "profile", "status", "port", "gpu", "container", "model"],
    )



@app.command("render-env")
def render_env(
    profile: Optional[str] = typer.Argument(
        None, help="Profile name to render. Omit to re-render all profiles."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Limit when rendering all (no PROFILE given)."
    ),
) -> None:
    """Re-render `.runtime/<backend>/<profile>.env` from `profiles.yaml`."""
    if backend and backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    if profile is None:
        # Render per-profile rather than via render_all(): a single unrenderable
        # value would otherwise abort the whole batch with a traceback and never
        # say which profile was at fault.
        failures: list[str] = []
        for bk in ([backend] if backend else list(BACKENDS)):
            for sp in profile_store.list_profiles(bk):
                try:
                    print(profile_store.render_env(sp))
                except ValueError as exc:
                    failures.append(f"{bk}/{sp.name}: {exc}")
        if failures:
            print("\nFailed to render:")
            for f in failures:
                print(f"  - {f}")
            raise typer.Exit(code=1)
        return

    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    try:
        out_path = profile_store.render_env(sp)
    except ValueError as exc:
        print(f"Error: cannot render {bk}/{profile}.env — {exc}")
        print(f"  Fix it with: llmux profile edit {profile} --unset <KEY>")
        raise typer.Exit(code=1)
    print(out_path)
