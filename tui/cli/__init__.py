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
    except Exception:  # noqa: BLE001 — onboarding is best-effort, never fatal
        pass


def _maybe_check_for_update() -> None:
    """Check for a newer llmux release once per day, in interactive sessions.

    Headless invocations skip it so scripts and CI never see a prompt or
    extra output. SystemExit (raised after a successful self-update so the
    user restarts on fresh code) is allowed to propagate; everything else is
    swallowed — a version check must never break startup.
    """
    if not _interactive_session():
        return
    try:
        from tui.common.version_check import check_for_update

        check_for_update()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — best-effort, never fatal
        pass


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Default behavior: launch the TUI when no subcommand is given."""
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


# ── Sub-apps ─────────────────────────────────────────────────────────────────
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


def main() -> None:
    """Console-script entrypoint (`llmux` in pyproject.scripts)."""
    app()


if __name__ == "__main__":
    main()
