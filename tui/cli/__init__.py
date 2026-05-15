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


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Default behavior: launch the TUI when no subcommand is given."""
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
