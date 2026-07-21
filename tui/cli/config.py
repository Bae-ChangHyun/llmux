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
from tui.cli.profile import _validate_gpu_mem
from tui.common import profile_store

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


def _config_path(backend: str, name: str) -> Path:
    """Return the canonical on-disk path for `<backend>/<name>.yaml`."""
    return _config_dir(backend) / f"{name}.yaml"


def _backend_save_config(
    backend: str, name: str, data: dict, disabled: dict | None = None
) -> Path:
    """Route a config write through the backend's canonical serializer.

    The vLLM backend hoists `model` / `gpu-memory-utilization` into typed
    fields and writes them in a fixed order; the llama.cpp backend writes a
    flat flag dict. Going through these instead of a local `yaml.safe_dump`
    keeps TUI and CLI round-trips byte-identical. `disabled`
    carries params kept as comment markers rather than active keys.
    """
    disabled = disabled or {}
    cdir = _config_dir(backend)
    cdir.mkdir(parents=True, exist_ok=True)
    if backend == "vllm":
        from tui.backends.vllm.backend_common import Config as VllmConfig
        from tui.backends.vllm.backend_storage import save_config as v_save

        extra = dict(data)
        model = str(extra.pop("model", ""))
        gpu_mem = str(extra.pop("gpu-memory-utilization", "0.9"))
        v_save(VllmConfig(
            name=name,
            model=model,
            gpu_memory_utilization=gpu_mem,
            extra_params=extra,
            disabled_params=dict(disabled),
        ))
    else:
        from tui.backends.llamacpp.backend import Config as LcppConfig
        from tui.backends.llamacpp.backend import save_config as l_save

        l_save(LcppConfig(name=name, params=dict(data), disabled_params=dict(disabled)))
    return _config_path(backend, name)


def _backend_load_disabled(backend: str, name: str) -> dict:
    """Disabled params of an existing config (empty when none / file absent)."""
    if backend == "vllm":
        from tui.backends.vllm.backend_storage import load_config as v_load

        return dict(v_load(name).disabled_params)
    from tui.backends.llamacpp.backend import load_config as l_load

    return dict(l_load(name).disabled_params)


def _backend_load_config(backend: str, name: str) -> dict:
    """Read a config through the backend's canonical loader, flattened to a
    `{key: value}` dict the existing --set/--unset CLI logic patches in place.

    For vLLM: the typed fields are re-merged into the flat dict under their
    on-disk key names (`model`, `gpu-memory-utilization`) so that subsequent
    `_backend_save_config` round-trips back through the same code path.
    """
    if backend == "vllm":
        from tui.backends.vllm.backend_storage import load_config as v_load

        cfg = v_load(name)
        data: dict = {"model": cfg.model, "gpu-memory-utilization": cfg.gpu_memory_utilization}
        data.update(cfg.extra_params)
        return data
    from tui.backends.llamacpp.backend import load_config as l_load

    cfg = l_load(name)
    return dict(cfg.params)


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


_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Creation rule — matches the TUI's ConfigFormScreen (lowercase only). Kept
# separate from _NAME_RE on purpose: the *reference* paths (show/edit/delete/
# clone-source) stay on the permissive rule so a user who already has an
# `Uppercase.yaml` on disk — from an older build or made by hand — can still
# inspect and remove it. Only new names are held to the strict rule.
_NEW_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_name(name: str) -> None:
    """Reject names that could escape `_config_dir(bk)` via path traversal."""
    if not _NAME_RE.match(name):
        raise typer.BadParameter(
            f"invalid config name {name!r}: must match {_NAME_RE.pattern}",
            param_hint="NAME",
        )


def _reject_vllm_only_config_options(
    backend: str, *, model_given: bool, gpu_mem_given: bool
) -> None:
    """Refuse `--model` / `--gpu-mem` on a llama.cpp config.

    Both are vLLM-shaped: `_backend_save_config` drops them for llama.cpp
    (`new`), or writes them through verbatim (`edit`) as `model` /
    `gpu-memory-utilization` keys — flags llama-server does not have, so the
    profile only breaks later, at start time.
    """
    if backend != "llamacpp":
        return
    offenders = [
        flag
        for flag, given in (("--model", model_given), ("--gpu-mem", gpu_mem_given))
        if given
    ]
    if not offenders:
        return
    verb = "is" if len(offenders) == 1 else "are"
    raise typer.BadParameter(
        f"{', '.join(offenders)} {verb} vLLM-only and not supported by backend "
        "'llamacpp'; use --set KEY=VALUE for llama-server flags.",
        param_hint=offenders[0],
    )


def _validate_gpu_mem_in_data(backend: str, data: dict) -> None:
    """Apply the --gpu-mem range rule to a value that arrived via --set.

    vLLM only: llama.cpp's `--set` is the raw llama-server flag namespace, where
    a key of the same spelling would just be a user-defined flag — not the vLLM
    field this rule governs.
    """
    if backend != "vllm":
        return
    raw = data.get("gpu-memory-utilization")
    if raw is None:
        return
    _validate_gpu_mem(str(raw), param_hint="--set gpu-memory-utilization")


def _reject_creating_example(name: str, *, param_hint: str) -> None:
    """`example.yaml` is the tracked template — recreating or overwriting it has
    no legitimate use, so `new`/`clone` refuse even with --overwrite.

    `edit`/`delete` stay available as escape hatches (someone may genuinely need
    to fix the template) but warn instead of acting silently.
    """
    if name == "example":
        raise typer.BadParameter(
            "'example' is the tracked template config and may not be created or "
            "overwritten; pick a different name.",
            param_hint=param_hint,
        )


def _warn_touching_example(backend: str, name: str, verb: str) -> None:
    if name == "example":
        print(
            f"warning: {verb} the tracked template config "
            f"config/{backend}/example.yaml"
        )


def _validate_new_name(name: str, *, param_hint: str = "NAME") -> None:
    """Name rule for configs we are about to create (TUI-equivalent)."""
    _validate_name(name)
    if not _NEW_NAME_RE.match(name):
        raise typer.BadParameter(
            f"invalid config name {name!r}: must be lowercase — match "
            f"{_NEW_NAME_RE.pattern} (start with [a-z0-9], then lowercase "
            "letters, digits, dashes, or underscores only).",
            param_hint=param_hint,
        )


def _resolve_backend_for_existing(backend: Optional[str], name: str) -> str:
    """Pick the backend that owns an existing `<name>.yaml`.

    Used by show/edit/delete (read-or-mutate paths). Raises if the config
    can't be located in either backend, so a typo never silently defaults
    to vllm and reports "config not found" only at the next read.
    """
    _validate_name(name)
    if backend:
        if backend not in BACKENDS:
            raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
        if not (_config_dir(backend) / f"{name}.yaml").exists():
            raise typer.BadParameter(
                f"config '{name}' not found in backend '{backend}'", param_hint="NAME"
            )
        return backend
    matches = [b for b in BACKENDS if (_config_dir(b) / f"{name}.yaml").exists()]
    if not matches:
        raise typer.BadParameter(
            f"config '{name}' not found in any backend", param_hint="NAME"
        )
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
    bk = _resolve_backend_for_existing(backend, name)
    data = _load_yaml(bk, name)
    disabled = _backend_load_disabled(bk, name)
    if json_out:
        # Structured (params + disabled) so a caller can tell the two apart —
        # the flat file dump can't express a disabled param (it's a comment).
        emit_json({"params": data, "disabled": disabled})
        return
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False))
    if disabled:
        print("# disabled:")
        print(
            yaml.safe_dump(
                disabled, sort_keys=False, allow_unicode=True, default_flow_style=False
            ).rstrip()
        )


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
    _validate_new_name(name)
    _reject_creating_example(name, param_hint="NAME")
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    # `new` has no None sentinel, so "given" = "differs from the default".
    _reject_vllm_only_config_options(
        backend,
        model_given=bool(model),
        gpu_mem_given=gpu_memory_utilization != "0.9",
    )
    if backend == "vllm":
        _validate_gpu_mem(gpu_memory_utilization)
    path = _config_dir(backend) / f"{name}.yaml"
    if path.exists() and not overwrite:
        raise typer.BadParameter(f"config already exists: {path} (use --overwrite)")

    if backend == "vllm":
        data: dict = {"model": model, "gpu-memory-utilization": gpu_memory_utilization}
    else:
        data = {}
    data.update(_parse_set_kv(set_kv))
    _validate_gpu_mem_in_data(backend, data)
    saved = _backend_save_config(backend, name, data)
    print(saved)


@app.command("edit")
def edit_config(
    name: str = typer.Argument(...),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    set_kv: list[str] = typer.Option([], "--set", help="Repeatable: KEY=VALUE."),
    unset: list[str] = typer.Option([], "--unset", help="Repeatable: KEY to remove (both active & disabled)."),
    disable: list[str] = typer.Option(
        [], "--disable", help="Repeatable: move an active KEY to disabled (kept, not served)."
    ),
    enable: list[str] = typer.Option(
        [], "--enable", help="Repeatable: move a disabled KEY back to active."
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    gpu_memory_utilization: Optional[str] = typer.Option(None, "--gpu-mem"),
) -> None:
    """Patch fields in an existing config."""
    bk = _resolve_backend_for_existing(backend, name)
    _warn_touching_example(bk, name, "modifying")
    _reject_vllm_only_config_options(
        bk,
        model_given=model is not None,
        gpu_mem_given=gpu_memory_utilization is not None,
    )
    data = _backend_load_config(bk, name)
    disabled = _backend_load_disabled(bk, name)
    if model is not None:
        data["model"] = model
    if gpu_memory_utilization is not None:
        _validate_gpu_mem(gpu_memory_utilization)
        data["gpu-memory-utilization"] = gpu_memory_utilization
    # --set re-activates a key that was disabled (and updates its value).
    for k, v in _parse_set_kv(set_kv).items():
        disabled.pop(k, None)
        data[k] = v
    # vLLM's model / gpu-memory-utilization are typed fields the serializer
    # always re-emits with a default, so a disabled marker for them would sit
    # alongside a live default — `up` would silently use the default, not the
    # value the user thought they'd toggled. Refuse to toggle them at all.
    if bk == "vllm":
        for flag, keys in (("--disable", disable), ("--enable", enable)):
            for k in keys:
                if k in ("model", "gpu-memory-utilization"):
                    raise typer.BadParameter(
                        f"{k} is a core vLLM field and cannot be toggled; use "
                        "--model / --gpu-mem to change it.",
                        param_hint=flag,
                    )
    # --disable: active -> disabled. Must currently be active.
    for k in disable:
        if k not in data:
            raise typer.BadParameter(
                f"cannot disable {k!r}: not an active param in config '{name}'.",
                param_hint="--disable",
            )
        disabled[k] = data.pop(k)
    # --enable: disabled -> active. Must currently be disabled.
    for k in enable:
        if k not in disabled:
            raise typer.BadParameter(
                f"cannot enable {k!r}: not a disabled param in config '{name}'.",
                param_hint="--enable",
            )
        data[k] = disabled.pop(k)
    # Validate AFTER enable so a bad gpu-memory-utilization brought back from
    # the disabled set (or set above) can't slip through unchecked.
    _validate_gpu_mem_in_data(bk, data)
    # --unset removes from wherever it lives.
    for k in unset:
        data.pop(k, None)
        disabled.pop(k, None)
    saved = _backend_save_config(bk, name, data, disabled)
    print(saved)


@app.command("clone")
def clone_config(
    src: str = typer.Argument(..., help="Source config name to copy from."),
    dst: str = typer.Argument(..., help="New config name to create."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help="Backend of the source config; required if SRC exists in both backends.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite DST if it already exists."
    ),
) -> None:
    """Clone SRC config to DST (same backend). Useful when quick-setup
    `--copy-from` isn't enough — e.g. cloning a tuned config to a new name
    without creating a profile in the same step.
    """
    bk = _resolve_backend_for_existing(backend, src)
    _validate_new_name(dst, param_hint="DST")
    _reject_creating_example(dst, param_hint="DST")
    dst_path = _config_dir(bk) / f"{dst}.yaml"
    if dst_path.exists() and not overwrite:
        raise typer.BadParameter(
            f"destination config already exists: {dst_path} (use --overwrite)",
            param_hint="DST",
        )
    data = _backend_load_config(bk, src)
    disabled = _backend_load_disabled(bk, src)
    saved = _backend_save_config(bk, dst, data, disabled)
    print(saved)


@app.command("rename")
def rename_config(
    old: str = typer.Argument(..., help="Existing config name."),
    new: str = typer.Argument(..., help="New config name."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help="Backend of the config; required if OLD exists in both backends.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Rename a config YAML and repoint every profile that referenced it."""
    from tui.common import config_store

    bk = _resolve_backend_for_existing(backend, old)
    _validate_new_name(new, param_hint="NEW")
    _reject_creating_example(new, param_hint="NEW")

    referencing = config_store.referencing_profiles(bk, old)
    if referencing:
        # A referencing profile whose container is up would keep serving the old
        # file until it restarts; renaming under it makes the running container
        # disagree with profiles.yaml.
        from tui.cli.profile import _require_stopped

        for p in referencing:
            _require_stopped(p, action="rename the config")

    if not yes:
        prompt = f"Rename config '{old}' → '{new}' (backend={bk})?"
        if referencing:
            listed = ", ".join(sorted(p.name for p in referencing))
            prompt = (
                f"Rename config '{old}' → '{new}' (backend={bk})? "
                f"(referenced by profiles: {listed} — they will be repointed)"
            )
        if not typer.confirm(prompt):
            raise typer.Exit(code=0)

    try:
        updated = config_store.rename_config(bk, old, new)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="NEW") from exc
    print(f"Renamed {_config_path(bk, old)} → {_config_path(bk, new)}")
    for name in updated:
        print(f"Repointed profile: {name}")


@app.command("from-recipe")
def config_from_recipe(
    model_id: str = typer.Argument(..., help="HuggingFace model id, e.g. Qwen/Qwen3-32B."),
    variant: str = typer.Option(
        "", "--variant", help="Precision variant (default/fp8/awq/…). Auto-picks a "
        "GPU-fitting one if omitted.",
    ),
    feature: list[str] = typer.Option(
        [], "--feature", help="Repeatable: opt-in feature to enable (e.g. reasoning)."
    ),
    name: str = typer.Option("", "--name", "-n", help="Config name (auto-derived if omitted)."),
    list_only: bool = typer.Option(
        False, "--list", help="List the recipe's variants + features and exit."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Preview the resulting config as JSON without writing."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite if it exists."),
) -> None:
    """Create a vLLM config from the official vllm-project/recipes recipe.

    The TUI shows a review window; headless callers pick a variant with
    --variant (or let it auto-pick the highest-quality one that fits the GPU).
    """
    from tui.cli._runtime import run_async
    from tui.common.recipes import build_config, fetch_recipe

    recipe = run_async(fetch_recipe(model_id))
    if recipe is None:
        raise typer.BadParameter(
            f"no vLLM recipe found for {model_id} (see recipes.vllm.ai)",
            param_hint="MODEL_ID",
        )

    if list_only:
        print(f"{recipe.model_id}  (min vLLM {recipe.min_vllm_version or '?'})")
        if recipe.variants:
            print("variants:")
            for v in recipe.variants:
                vram = f"≥{v.vram_minimum_gb:.0f}GB" if v.vram_minimum_gb else "?"
                print(f"  {v.name:<10} {v.precision or '':<6} {vram:<8} {v.description}")
        if recipe.features:
            print("features:")
            for f in recipe.features:
                print(f"  {f.name:<16} {f.description}")
        return

    chosen = _pick_recipe_variant(recipe, variant)
    model, params = build_config(recipe, chosen, list(feature))

    if json_out:
        emit_json({"model": model, "variant": chosen.name if chosen else "",
                   "params": params})
        return

    cfg_name = name or _derive_config_name(model)
    _validate_new_name(cfg_name, param_hint="--name")
    _reject_creating_example(cfg_name, param_hint="--name")
    path = _config_dir("vllm") / f"{cfg_name}.yaml"
    if path.exists() and not overwrite:
        raise typer.BadParameter(
            f"config already exists: {path} (use --overwrite or --name)",
            param_hint="--name",
        )
    data: dict = {"model": model, "gpu-memory-utilization": "0.9"}
    data.update(params)
    saved = _backend_save_config("vllm", cfg_name, data, {})
    print(saved)


def _pick_recipe_variant(recipe, variant_name: str):
    """Resolve --variant to a RecipeVariant, or auto-pick the best GPU fit."""
    if not recipe.variants:
        return None
    by_name = {v.name: v for v in recipe.variants}
    if variant_name:
        if variant_name not in by_name:
            raise typer.BadParameter(
                f"unknown variant {variant_name!r}; choose from "
                f"{', '.join(by_name)}", param_hint="--variant",
            )
        return by_name[variant_name]
    # Auto-pick: highest-quality (recipe order) variant that fits the GPU;
    # else the smallest-VRAM one. Mirrors the TUI review screen.
    from tui.cli._runtime import run_async
    from tui.common.docker import get_gpu_info

    gpus = run_async(get_gpu_info())
    gpu_gb = None
    if gpus:
        try:
            gpu_gb = max(int(g.memory_total) / 1024 for g in gpus)
        except (TypeError, ValueError):
            gpu_gb = None
    if gpu_gb is not None:
        for v in recipe.variants:
            if v.vram_minimum_gb is not None and v.vram_minimum_gb <= gpu_gb:
                return v
    return min(recipe.variants, key=lambda v: v.vram_minimum_gb or 1e9)


def _derive_config_name(model_id: str) -> str:
    # Drop `.` from the allowed set so the result matches `_NEW_NAME_RE` (which
    # forbids dots): a dotted model id like `Qwen/Qwen2.5-7B-Instruct` used to
    # derive `qwen2.5-7b-instruct` and then fail --name validation. Mirrors the
    # quick-setup derivation, which also collapses dots to dashes.
    tail = model_id.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9_-]+", "-", tail.lower()).strip("-") or "recipe-config"


@app.command("delete")
def delete_config(
    name: str = typer.Argument(...),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete a config YAML."""
    bk = _resolve_backend_for_existing(backend, name)
    path = _config_dir(bk) / f"{name}.yaml"
    if not path.exists():
        raise typer.BadParameter(f"config not found: {path}", param_hint="NAME")
    _warn_touching_example(bk, name, "deleting")
    # Profiles pointing at this config would keep a dangling config_name in
    # profiles.yaml once the YAML is gone; clear them like the TUI's
    # ConfirmDeleteConfigScreen does.
    referencing = [p for p in profile_store.list_profiles(bk) if p.config_name == name]
    if not yes:
        prompt = f"Delete {path}?"
        if referencing:
            listed = ", ".join(sorted(p.name for p in referencing))
            prompt = (
                f"Delete {path}? "
                f"(referenced by profiles: {listed} — their config_name will be cleared)"
            )
        if not typer.confirm(prompt):
            raise typer.Exit(code=0)
    path.unlink()
    print(f"Deleted {path}")
    for p in referencing:
        p.config_name = ""
        profile_store.save_profile(p)
        print(f"Cleared config_name on profile: {p.name}")
