"""Config (YAML) management commands.

For vLLM: `config/vllm/<name>.yaml` — `model:` + `gpu-memory-utilization:` +
arbitrary `vllm serve` flags.

For llama.cpp: `config/llamacpp/<name>.yaml` — flat dict of `llama-server` flags.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import typer
import yaml

from tui.cli._runtime import BACKENDS, emit_json, emit_table

app = typer.Typer(help="Config (YAML) CRUD.", no_args_is_help=True)


def _config_dir(backend: str) -> Path:
    if backend == "vllm":
        from tui.backends.vllm.backend_common import CONFIG_DIR

        return CONFIG_DIR
    if backend == "llamacpp":
        from tui.backends.llamacpp.backend import CONFIG_DIR

        return CONFIG_DIR
    raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")


def _list_names(backend: str) -> list[str]:
    cdir = _config_dir(backend)
    if not cdir.exists():
        return []
    return sorted(p.stem for p in cdir.glob("*.yaml") if p.stem != "example")


def _load_yaml(backend: str, name: str) -> dict:
    path = _config_dir(backend) / f"{name}.yaml"
    if not path.exists():
        raise typer.BadParameter(f"config not found: {path}", param_hint="NAME")
    raw = yaml.safe_load(path.read_text()) or {}
    return raw if isinstance(raw, dict) else {}


def _save_yaml(backend: str, name: str, data: dict) -> Path:
    cdir = _config_dir(backend)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    )
    return path


def _parse_set_kv(items: list[str]) -> dict:
    """Parse repeated --set KEY=VALUE; YAML-load values for ints/bools/lists."""
    out: dict = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(
                f"--set value must be KEY=VALUE, got {raw!r}", param_hint="--set"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter("--set KEY may not be empty", param_hint="--set")
        if value == "":
            out[key] = True
        else:
            try:
                out[key] = yaml.safe_load(value)
            except yaml.YAMLError:
                out[key] = value
    return out


def _resolve_backend(backend: Optional[str], name: Optional[str]) -> str:
    """Pick a backend: explicit override, else find which backend has the config."""
    if backend:
        if backend not in BACKENDS:
            raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
        return backend
    if name is None:
        return "vllm"
    matches = [b for b in BACKENDS if (_config_dir(b) / f"{name}.yaml").exists()]
    if not matches:
        # Default to vllm for new configs.
        return "vllm"
    if len(matches) > 1:
        raise typer.BadParameter(
            f"config '{name}' exists in multiple backends ({', '.join(matches)}); "
            "disambiguate with --backend",
            param_hint="NAME",
        )
    return matches[0]


@app.command("list")
def list_configs(
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List config files."""
    backends = [backend] if backend else list(BACKENDS)
    rows = []
    for bk in backends:
        for n in _list_names(bk):
            data = _load_yaml(bk, n)
            rows.append(
                {
                    "backend": bk,
                    "name": n,
                    "model": data.get("model", ""),
                    "params": len(data),
                }
            )
    if json_out:
        emit_json(rows)
        return
    emit_table(rows, columns=["backend", "name", "model", "params"])


@app.command("show")
def show_config(
    name: str = typer.Argument(...),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Print a config YAML."""
    bk = _resolve_backend(backend, name)
    data = _load_yaml(bk, name)
    if json_out:
        emit_json(data)
        return
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False))


@app.command("new")
def new_config(
    name: str = typer.Argument(..., help="Config name (becomes <name>.yaml)."),
    backend: str = typer.Option("vllm", "--backend", "-b"),
    model: str = typer.Option("", "--model", "-m", help="vLLM model: model field."),
    gpu_memory_utilization: str = typer.Option(
        "0.9", "--gpu-mem", help="vLLM only: gpu-memory-utilization."
    ),
    set_kv: list[str] = typer.Option(
        [], "--set", help="Repeatable: KEY=VALUE entries. YAML-typed values."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite if file exists."),
) -> None:
    """Create a new config YAML."""
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", name):
        raise typer.BadParameter("invalid name", param_hint="NAME")
    path = _config_dir(backend) / f"{name}.yaml"
    if path.exists() and not overwrite:
        raise typer.BadParameter(f"config already exists: {path} (use --overwrite)")

    if backend == "vllm":
        data: dict = {"model": model, "gpu-memory-utilization": gpu_memory_utilization}
    else:
        data = {}
    data.update(_parse_set_kv(set_kv))
    saved = _save_yaml(backend, name, data)
    print(saved)


@app.command("edit")
def edit_config(
    name: str = typer.Argument(...),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    set_kv: list[str] = typer.Option([], "--set", help="Repeatable: KEY=VALUE."),
    unset: list[str] = typer.Option([], "--unset", help="Repeatable: KEY to remove."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    gpu_memory_utilization: Optional[str] = typer.Option(None, "--gpu-mem"),
) -> None:
    """Patch fields in an existing config."""
    bk = _resolve_backend(backend, name)
    data = _load_yaml(bk, name)
    if model is not None:
        data["model"] = model
    if gpu_memory_utilization is not None:
        data["gpu-memory-utilization"] = gpu_memory_utilization
    data.update(_parse_set_kv(set_kv))
    for k in unset:
        data.pop(k, None)
    saved = _save_yaml(bk, name, data)
    print(saved)


@app.command("delete")
def delete_config(
    name: str = typer.Argument(...),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete a config YAML."""
    bk = _resolve_backend(backend, name)
    path = _config_dir(bk) / f"{name}.yaml"
    if not path.exists():
        raise typer.BadParameter(f"config not found: {path}", param_hint="NAME")
    if not yes:
        if not typer.confirm(f"Delete {path}?"):
            raise typer.Exit(code=1)
    path.unlink()
    print(f"Deleted {path}")
