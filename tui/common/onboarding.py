"""First-run onboarding: interactively create `.env.common`.

A fresh checkout has no `.env.common` — it is gitignored, only the tracked
`.env.common.example` template ships. Rather than failing later with a
"create it from the example" error, the CLI entry point runs this wizard once
to collect the few values that genuinely need a human (HF cache location,
model directory, optional HF token) and renders a complete `.env.common` from
the template, keeping every other key at its documented default.

The caller owns the TTY gate: a non-interactive invocation (script, pipe, CI)
must skip this so it falls back to the normal validation error instead of
blocking on a prompt.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from tui.common.env import validate_common_env
from tui.common.profile_store import PROJECT_ROOT

COMMON_ENV = PROJECT_ROOT / ".env.common"
COMMON_ENV_EXAMPLE = PROJECT_ROOT / ".env.common.example"
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
PROFILES_EXAMPLE = PROJECT_ROOT / "profiles.example.yaml"


def needs_onboarding() -> bool:
    """True when `.env.common` is absent — the one signal of a fresh checkout."""
    return not COMMON_ENV.exists()


def _render_env(overrides: dict[str, str]) -> str:
    """Render `.env.common` from the tracked example, applying `overrides`.

    Every comment and every unprompted key (LLAMACPP_*, TZ, LORA_BASE_PATH) is
    kept verbatim from the template, so the result stays self-documenting.
    """
    lines: list[str] = []
    for line in COMMON_ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in overrides:
                lines.append(f"{key}={overrides[key]}")
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def run_onboarding() -> bool:
    """Interactive wizard: collect a few values and write `.env.common`.

    Returns True when `.env.common` was written and validates; False if the
    user aborted or the template was missing. Never raises — the caller treats
    a False result as "fall through to normal validation".
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    console = Console()

    if not COMMON_ENV_EXAMPLE.exists():
        # No template to render from — let normal validation guide the user.
        return False

    console.print(
        Panel(
            "No [bold].env.common[/bold] yet — let's set up llmux.\n"
            "Press Enter to accept each default shown in [dim][brackets][/dim].",
            title="llmux onboarding",
            border_style="cyan",
        )
    )

    try:
        hf_cache = Prompt.ask(
            "HuggingFace cache directory",
            default=str(Path.home() / ".cache" / "huggingface"),
            console=console,
        )
        model_dir = Prompt.ask(
            "llama.cpp GGUF model directory",
            default=str(PROJECT_ROOT / "models"),
            console=console,
        )
        hf_token = Prompt.ask(
            "HuggingFace token (optional — needed for gated models)",
            default="",
            console=console,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Onboarding cancelled.[/yellow]")
        return False

    hf_cache_path = str(Path(hf_cache).expanduser())
    model_dir_path = str(Path(model_dir).expanduser())

    if not os.path.isabs(hf_cache_path):
        console.print(
            f"[red]HF_CACHE_PATH must be an absolute path:[/red] {hf_cache_path}"
        )
        return False
    if not os.path.isabs(model_dir_path):
        console.print(
            f"[red]MODEL_DIR must be an absolute path:[/red] {model_dir_path}"
        )
        return False

    try:
        COMMON_ENV.write_text(
            _render_env(
                {
                    "HF_CACHE_PATH": hf_cache_path,
                    "MODEL_DIR": model_dir_path,
                    "HF_TOKEN": hf_token.strip(),
                }
            )
        )
    except OSError as exc:
        console.print(f"[red]Could not write {COMMON_ENV}:[/red] {exc}")
        return False

    # Pre-create the bind-mount targets so docker does not create them as root.
    for path in (hf_cache_path, model_dir_path):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # non-fatal — surfaced later by validation if it matters

    # Offer to seed profiles.yaml so the first launch has something to show.
    if not PROFILES_YAML.exists() and PROFILES_EXAMPLE.exists():
        try:
            if Confirm.ask(
                "Create profiles.yaml from the template?",
                default=True,
                console=console,
            ):
                shutil.copyfile(PROFILES_EXAMPLE, PROFILES_YAML)
        except (KeyboardInterrupt, EOFError):
            pass

    ok, messages = validate_common_env(COMMON_ENV)
    if not ok:
        console.print("[red]" + "\n".join(messages) + "[/red]")
        return False

    console.print(
        Panel(
            f"[green]✓[/green] Wrote [bold]{COMMON_ENV}[/bold]\n"
            "Edit it any time to tweak LoRA paths, image tags, or timezone.",
            border_style="green",
        )
    )
    return True
