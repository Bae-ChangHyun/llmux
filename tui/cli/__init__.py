"""llmux CLI — headless interface mirroring TUI features.

The TUI is still the default: `llmux` with no arguments launches it.
Subcommands provide non-interactive access for scripts and agents.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="llmux",
    help="Terminal multiplexer for vLLM + llama.cpp. "
    "Run with no arguments to launch the TUI; use subcommands for headless control.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _interactive_session() -> bool:
    """True only for a genuine interactive terminal session.

    Returns False under CI, or when `LLMUX_NONINTERACTIVE` is set, even if a
    pseudo-TTY is allocated — so first-run onboarding and the update prompt
    never block an automated job that happens to run with a TTY attached
    (`docker run -t`, `ssh -t`, some CI runners).
    """
    import os
    import sys

    if os.environ.get("CI") or os.environ.get("LLMUX_NONINTERACTIVE"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _maybe_run_onboarding() -> None:
    """Run the first-run wizard once, when `.env.common` is missing and the
    session is interactive.

    Headless invocations (scripts, pipes, CI) are left untouched so they fall
    back to the normal validation error instead of blocking on a prompt.
    Onboarding is best-effort — any failure is swallowed so it never breaks
    startup.
    """
    try:
        from tui.common.onboarding import needs_onboarding, run_onboarding

        if not needs_onboarding():
            return
        if not _interactive_session():
            return
        run_onboarding()
    except Exception as exc:  # noqa: BLE001 — never fatal, but never silent
        import sys

        print(f"Warning: onboarding did not complete ({exc}).", file=sys.stderr)


def _maybe_check_for_update() -> None:
    """Check for a newer llmux release, in interactive sessions.

    Runs on every interactive invocation (only a failed lookup backs off), so
    a release published minutes ago shows up on the next command. Headless
    invocations skip it so scripts and CI never see a prompt or extra output.
    SystemExit (raised after a successful self-update so the user restarts on
    fresh code) is allowed to propagate; everything else is swallowed — a
    version check must never break startup.
    """
    if not _interactive_session():
        return
    try:
        from tui.common.version_check import check_for_update

        check_for_update()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — never fatal, but never silent
        import sys

        print(f"Warning: update check failed ({exc}).", file=sys.stderr)


def llmux_version() -> str:
    """Installed llmux version, from the package metadata.

    A source checkout that was never `pip install`ed has no metadata; fall back
    to reading pyproject.toml so `--version` still answers in a dev tree.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("llmux")
    except PackageNotFoundError:
        import tomllib

        from tui.common.profile_store import PROJECT_ROOT

        pyproject = PROJECT_ROOT / "pyproject.toml"
        if not pyproject.exists():
            return "unknown"
        data = tomllib.loads(pyproject.read_text())
        return data.get("project", {}).get("version", "unknown")


def _version_callback(value: bool) -> None:
    if not value:
        return
    print(f"llmux {llmux_version()}")
    raise typer.Exit()


def _configure_logging(level: str) -> None:
    """Attach a stderr handler so the modules' log calls are actually visible.

    Without this the root logger has no handler: DEBUG records are dropped
    entirely and WARNING records only reach stderr via logging.lastResort,
    which the TUI swallows.
    """
    import logging

    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise typer.BadParameter(
            f"unknown log level {level!r} (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            param_hint="--log-level",
        )
    import sys

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    # Scoped to llmux's own logger: a root handler at DEBUG would drown the
    # output in httpcore/urllib3 chatter.
    logger = logging.getLogger("tui")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the llmux version and exit.",
    ),
    log_level: str = typer.Option(
        "WARNING", "--log-level",
        envvar="LLMUX_LOG_LEVEL",
        help="Log verbosity on stderr (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
) -> None:
    """Default behavior: launch the TUI when no subcommand is given."""
    _configure_logging(log_level)
    _maybe_run_onboarding()
    _maybe_check_for_update()
    if ctx.invoked_subcommand is not None:
        return
    from tui.cli._launch_tui import launch_tui

    launch_tui()


@app.command("tui")
def tui_cmd() -> None:
    """Launch the interactive TUI explicitly."""
    from tui.cli._launch_tui import launch_tui

    launch_tui()


@app.command("top")
def top_cmd(
    profile: str = typer.Argument(
        None, help="Show only this running profile (default: every running model)."
    ),
) -> None:
    """Live btop-style system monitor in a plain terminal (no TUI).

    Always shows every GPU — util, memory, temperature, power, PCIe — whether or
    not anything is running, plus a panel per running model (throughput and KV
    graphs, cache hit, requests, latency percentiles). q or Ctrl+C to quit."""
    from tui.common.plain_monitor import run_cli

    raise typer.Exit(code=run_cli(profile))


@app.command("update")
def update_cmd(
    check: bool = typer.Option(
        False, "--check", help="Only report; never touch the checkout."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Update without asking for confirmation."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the status as JSON instead of a human line."
    ),
) -> None:
    """Check for a newer llmux release right now, and update to it.

    Unlike the startup check this ignores the failure back-off and runs even
    in a non-interactive session, so it always answers. Exits non-zero when
    the check could not be completed or the update failed.
    """
    from tui.cli._runtime import emit_json
    from tui.common import version_check as vc

    status = vc.resolve_status(respect_cooldown=False)

    if json_out:
        emit_json({
            "state": status.state,
            "latest_tag": status.tag,
            "local_version": status.local_version,
            "url": status.url,
            "detail": status.detail,
            "checkout": str(vc.PROJECT_ROOT),
        })
        if status.state == vc.UNKNOWN:
            raise typer.Exit(code=1)
        if status.state == vc.CURRENT or check:
            raise typer.Exit(code=0)
    else:
        if status.state == vc.UNKNOWN:
            typer.echo(f"could not check for updates — {status.detail}", err=True)
            raise typer.Exit(code=1)
        if status.state == vc.CURRENT:
            print(f"llmux is up to date ({status.tag or status.local_version}).")
            raise typer.Exit(code=0)
        print(f"llmux {status.tag} is available.  {status.url}".rstrip())
        if check:
            print(f"  local: {status.local_version or 'unknown'}  ({vc.PROJECT_ROOT})")
            raise typer.Exit(code=0)

    blocked = vc.update_blocked_reason()
    if blocked:
        typer.echo(
            f"refusing to update automatically — {blocked}. "
            f"Pull it yourself in {vc.PROJECT_ROOT} when ready.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not yes and not typer.confirm("Update now?", default=True):
        raise typer.Exit(code=0)

    ok, message = vc.apply_update(status.tag)
    if not ok:
        typer.echo(message, err=True)
        raise typer.Exit(code=1)
    print(message)


from tui.cli import config as _config  # noqa: E402
from tui.cli import container as _container  # noqa: E402
from tui.cli import image as _image  # noqa: E402
from tui.cli import profile as _profile  # noqa: E402
from tui.cli import system as _system  # noqa: E402

app.add_typer(_container.app, name="container", help="Container lifecycle commands.")
app.add_typer(_profile.app, name="profile", help="Profile CRUD + quick-setup.")
app.add_typer(_config.app, name="config", help="Config (YAML) CRUD.")
app.add_typer(_image.app, name="image", help="Docker image inventory + dev build.")
app.add_typer(_system.app, name="system", help="System info & env validation.")
# Top-level shortcuts for the most-used system commands.
app.command("gpu", help="Print nvidia-smi summary (alias for `system gpu`).")(_system.gpu)
app.command("env-check", help="Validate .env.common (alias for `system env-check`).")(
    _system.env_check
)
# Top-level shortcuts so agents can write `llmux up <profile>` directly.
app.command("up", help="Start a profile's container (alias for `container up`).")(
    _container.up
)
app.command(
    "prepare",
    help="Download the model + render runtime files without starting "
    "(alias for `container prepare`).",
)(_container.prepare)
app.command("down", help="Stop a profile's container (alias for `container down`).")(
    _container.down
)
app.command("logs", help="Stream container logs (alias for `container logs`).")(
    _container.logs
)
app.command("ps", help="List profiles and their container status (alias for `container ps`).")(
    _container.ps
)
app.command(
    "render-env",
    help="Re-render runtime env file from profiles.yaml (alias for `container render-env`).",
)(_container.render_env)
app.command(
    "bench",
    help="Benchmark a running container's tok/s (alias for `container benchmark`).",
)(_container.benchmark)
app.command(
    "stats",
    help="Live tok/s from running containers' /metrics (alias for `container stats`).",
)(_container.stats)


def main() -> None:
    """Console-script entrypoint (`llmux` in pyproject.scripts)."""
    app()


if __name__ == "__main__":
    main()
