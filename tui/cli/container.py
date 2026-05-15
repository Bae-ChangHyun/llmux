"""Container lifecycle commands: up / down / logs / ps / render-env."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer

from tui.cli._runtime import (
    BACKENDS,
    detect_backend,
    emit_json,
    emit_table,
    run_async,
    stream_async,
)
from tui.common import profile_store

app = typer.Typer(help="Container lifecycle (start, stop, logs, status).", no_args_is_help=True)


# ---- up ---------------------------------------------------------------------

@app.command("up")
def up(
    profile: str = typer.Argument(..., help="Profile name (from profiles.yaml)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help=f"Force backend ({', '.join(BACKENDS)}); auto-detect if omitted."
    ),
    tag: str = typer.Option(
        "", "--tag", "-t",
        help="Image tag override. vLLM: vllm/vllm-openai:<tag> or vllm-dev:<tag> with --dev. "
        "Empty = use highest local versioned tag.",
    ),
    dev: bool = typer.Option(
        False, "--dev", help="vLLM only: use the locally-built vllm-dev:<tag> image."
    ),
    pull: bool = typer.Option(
        False, "--pull", help="vLLM only: force --pull always when bringing the container up."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url", help="vLLM dev image: override default vLLM source repo URL."
    ),
    branch: str = typer.Option(
        "", "--branch", help="vLLM dev image: override default vLLM source branch."
    ),
) -> None:
    """Start a profile's container. Streams compose output to stdout."""
    bk = detect_backend(profile, override=backend)
    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_up

        rc = stream_async(
            stream_container_up(
                profile,
                use_dev=dev,
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
                tag=tag,
                repo_url=repo_url,
                branch=branch,
            )
        )
    raise typer.Exit(code=rc)


# ---- down -------------------------------------------------------------------

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


# ---- logs -------------------------------------------------------------------

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

    if not follow:
        import subprocess

        rc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_name]
        ).returncode
        raise typer.Exit(code=rc)

    if bk == "vllm":
        from tui.backends.vllm.backend_runtime import stream_container_logs as _gen

        async def _drive():
            async for line in _gen(container_name):
                print(line, flush=True)
            return 0
    else:
        from tui.backends.llamacpp.backend import stream_logs as _gen

        async def _drive():
            async for evt in _gen(container_name, lines=tail):
                if isinstance(evt, tuple):
                    if evt[0] == "log":
                        print(evt[1], flush=True)
                else:
                    print(evt, flush=True)
            return 0

    try:
        rc = asyncio.run(_drive())
    except KeyboardInterrupt:
        rc = 130
    raise typer.Exit(code=rc)


# ---- ps ---------------------------------------------------------------------

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
    backends = [backend] if backend else list(BACKENDS)
    rows = []

    async def _collect():
        for bk in backends:
            if bk == "vllm":
                from tui.backends.vllm.backend_runtime import get_container_statuses

                statuses = await get_container_statuses()
                for s in statuses:
                    rows.append(
                        {
                            "backend": "vllm",
                            "profile": s.profile_name,
                            "container": s.container_name,
                            "status": s.status_text,
                            "running": s.running,
                            "port": s.port,
                            "gpu": s.gpu_id,
                            "model": s.model,
                        }
                    )
            else:
                # llama.cpp: derive status by querying docker for each profile.
                from tui.backends.llamacpp.backend import (
                    list_profile_names as lc_list,
                    load_profile as lc_load,
                    run_command,
                )

                names = lc_list()
                rc, out = await run_command(
                    "docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}", timeout=10
                )
                statusmap: dict[str, str] = {}
                if rc == 0:
                    for line in out.strip().splitlines():
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            statusmap[parts[0]] = parts[1]
                for n in names:
                    p = lc_load(n)
                    cn = p.container_name or p.name
                    raw = statusmap.get(cn, "")
                    status = "stopped"
                    running = False
                    if raw:
                        if "(healthy)" in raw:
                            status, running = "healthy", True
                        elif "(unhealthy)" in raw:
                            status, running = "unhealthy", True
                        elif "(health: starting)" in raw:
                            status, running = "starting", True
                        elif raw.startswith("Up "):
                            status, running = "running", True
                        elif raw.startswith("Exited "):
                            status = "exited"
                    rows.append(
                        {
                            "backend": "llamacpp",
                            "profile": n,
                            "container": cn,
                            "status": status,
                            "running": running,
                            "port": p.port,
                            "gpu": p.gpu_id,
                            "model": p.model_file or "",
                        }
                    )

    run_async(_collect())

    if running_only:
        rows = [r for r in rows if r["running"]]

    if json_out:
        emit_json(rows)
        return

    emit_table(
        rows,
        columns=["backend", "profile", "status", "port", "gpu", "container", "model"],
    )


# ---- render-env -------------------------------------------------------------

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
    if profile is None:
        paths = profile_store.render_all(backend=backend)
        for p in paths:
            print(p)
        return

    bk = detect_backend(profile, override=backend)
    sp = profile_store.load_profile(profile, bk)
    out_path = profile_store.render_env(sp)
    print(out_path)
