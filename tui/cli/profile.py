"""Profile management commands.

Profiles live in `profiles.yaml` (gitignored). Each entry has a `backend`
(vllm | llamacpp). The CLI auto-detects backend from the name; pass --backend
to disambiguate when the same name exists in both.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

import typer

from tui.cli._runtime import (
    BACKENDS,
    detect_backend,
    emit_json,
    emit_table,
    run_async,
)
from tui.common import profile_store

app = typer.Typer(help="Profile CRUD + quick-setup.", no_args_is_help=True)


def _profile_to_row(p: profile_store.StoredProfile) -> dict:
    return {
        "backend": p.backend,
        "name": p.name,
        "port": p.port,
        "gpu_id": p.gpu_id,
        "config": p.config_name or p.name,
        "model": p.model_id or p.model_file or "",
    }


# ---- list / show -------------------------------------------------------------

@app.command("list")
def list_profiles(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help=f"Limit to one backend ({', '.join(BACKENDS)})."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List all profiles across (or within) backends."""
    backends = [backend] if backend else list(BACKENDS)
    rows = []
    for bk in backends:
        rows.extend(_profile_to_row(p) for p in profile_store.list_profiles(bk))

    if json_out:
        emit_json(rows)
        return
    emit_table(rows, columns=["backend", "name", "port", "gpu_id", "config", "model"])


@app.command("show")
def show_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Force backend; auto-detect if omitted."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML-ish text."),
) -> None:
    """Show a profile's full record."""
    bk = detect_backend(name, override=backend)
    sp = profile_store.load_profile(name, bk)
    data = asdict(sp)
    if json_out:
        emit_json(data)
        return
    for k, v in data.items():
        print(f"{k}: {v}")


# ---- new / edit / delete -----------------------------------------------------

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Profile name rule (unified across backends). docker compose project names are
# lowercase-only, and a cross-backend profile created with mixed-case here used
# to round-trip cleanly via vLLM but fail validation on the llama.cpp side, so
# both backends now share the lowercase rule.
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_profile_name(name: str, backend: str, *, param_hint: str = "NAME") -> None:
    """Reject names a backend's own runtime would reject at start time.

    Applied to both `new` and `edit` so an invalid name can't sneak in via
    a direct profiles.yaml edit and then get round-tripped through `edit`
    unchecked.
    """
    if not _PROFILE_NAME_RE.match(name):
        raise typer.BadParameter(
            f"Profile names must match {_PROFILE_NAME_RE.pattern} "
            "(lowercase: start with [a-z0-9], then [a-z0-9_-] only). "
            "docker compose project names are lowercase-only, so this rule "
            "is shared by both backends.",
            param_hint=param_hint,
        )


def _parse_set_kv(items: list[str], *, backend: str = "") -> dict[str, str]:
    """Parse repeated --set KEY=VALUE pairs into a dict.

    Validates KEY against the same regex profile_store uses for env-line
    rendering, so a bad name fails up-front with a clean usage error instead
    of a deep traceback at save_profile time. When `backend` is supplied, also
    rejects keys that profile_store treats as reserved (GPU_ID, VLLM_PORT,
    CONTAINER_NAME, …) — otherwise the conflict checker and the rendered .env
    would silently disagree about which value is in effect.
    """
    reserved = profile_store.reserved_env_keys(backend) if backend else frozenset()
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(
                f"--set value must be KEY=VALUE, got {raw!r}", param_hint="--set"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter("--set KEY may not be empty", param_hint="--set")
        if not _ENV_KEY_RE.match(key):
            raise typer.BadParameter(
                f"--set KEY {key!r} must match {_ENV_KEY_RE.pattern} "
                "(env-var rules: leading letter or underscore, then alphanumerics/underscores)",
                param_hint="--set",
            )
        if key in reserved:
            raise typer.BadParameter(
                f"--set KEY {key!r} is reserved by the {backend} backend; "
                f"use the dedicated profile field (e.g. --port, --gpu-id) instead.",
                param_hint="--set",
            )
        out[key] = value
    return out


@app.command("new")
def new_profile(
    name: str = typer.Argument(..., help="Profile name (alphanumeric, dash, underscore)."),
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help=f"Backend ({', '.join(BACKENDS)})."
    ),
    port: int = typer.Option(0, "--port", "-p", help="Host port (0 = backend default)."),
    gpu_id: str = typer.Option("0", "--gpu-id", "-g", help="GPU id(s), comma-separated."),
    model: str = typer.Option("", "--model", "-m", help="Hugging Face model id (vLLM)."),
    config_name: str = typer.Option(
        "", "--config", "-c", help="Linked config name (defaults to profile name)."
    ),
    container_name: str = typer.Option(
        "", "--container", help="Container name (defaults to profile name)."
    ),
    enable_lora: bool = typer.Option(False, "--lora/--no-lora", help="vLLM only: enable LoRA."),
    extra_pip: str = typer.Option(
        "", "--extra-pip", help="vLLM only: extra pip packages installed before serve."
    ),
    set_env: list[str] = typer.Option(
        [], "--set", help="Repeatable: KEY=VALUE entries appended to env_vars."
    ),
    # llama.cpp specific
    model_file: str = typer.Option("", "--model-file", help="llama.cpp: GGUF filename."),
    hf_repo: str = typer.Option("", "--hf-repo", help="llama.cpp: Hugging Face repo for download."),
    hf_file: str = typer.Option("", "--hf-file", help="llama.cpp: HF file for download."),
) -> None:
    """Create a new profile entry in profiles.yaml."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")
    _validate_profile_name(name, backend)
    if profile_store.load_profile(name, backend) is not None:
        raise typer.BadParameter(
            f"profile '{name}' already exists in backend '{backend}'", param_hint="NAME"
        )

    defaults = profile_store.DEFAULTS[backend]
    sp = profile_store.StoredProfile(
        name=name,
        backend=backend,
        container_name=container_name or name,
        port=port or int(defaults["port"]),
        gpu_id=gpu_id or str(defaults["gpu_id"]),
        config_name=config_name or name,
        tensor_parallel_size=len(gpu_id.split(",")) if gpu_id else 1,
        model_id=model,
        enable_lora=enable_lora,
        extra_pip_packages=extra_pip,
        env_vars=_parse_set_kv(set_env, backend=backend),
        model_file=model_file,
        hf_repo=hf_repo,
        hf_file=hf_file,
    )
    profile_store.save_profile(sp)
    print(f"Created profile '{name}' (backend={backend})")


@app.command("edit")
def edit_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    port: Optional[int] = typer.Option(None, "--port", "-p"),
    gpu_id: Optional[str] = typer.Option(None, "--gpu-id", "-g"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    config_name: Optional[str] = typer.Option(None, "--config", "-c"),
    container_name: Optional[str] = typer.Option(None, "--container"),
    enable_lora: Optional[bool] = typer.Option(None, "--lora/--no-lora"),
    extra_pip: Optional[str] = typer.Option(None, "--extra-pip"),
    set_env: list[str] = typer.Option(
        [], "--set", help="Repeatable: KEY=VALUE entries to add/override in env_vars."
    ),
    unset_env: list[str] = typer.Option(
        [], "--unset", help="Repeatable: KEY to remove from env_vars."
    ),
    model_file: Optional[str] = typer.Option(None, "--model-file"),
    hf_repo: Optional[str] = typer.Option(None, "--hf-repo"),
    hf_file: Optional[str] = typer.Option(None, "--hf-file"),
    image_tag: Optional[str] = typer.Option(
        None, "--image-tag",
        help="Docker image override (e.g. 'llamacpp-dev:mtp_main'). Pass empty string to clear.",
    ),
) -> None:
    """Edit fields of an existing profile (only specified options change)."""
    bk = detect_backend(name, override=backend)
    _validate_profile_name(name, bk)
    sp = profile_store.load_profile(name, bk)

    if port is not None:
        sp.port = port
    if gpu_id is not None:
        sp.gpu_id = gpu_id
        sp.tensor_parallel_size = len(gpu_id.split(",")) if gpu_id else 1
    if model is not None:
        sp.model_id = model
    if config_name is not None:
        sp.config_name = config_name
    if container_name is not None:
        sp.container_name = container_name
    if enable_lora is not None:
        sp.enable_lora = enable_lora
    if extra_pip is not None:
        sp.extra_pip_packages = extra_pip
    if model_file is not None:
        sp.model_file = model_file
    if hf_repo is not None:
        sp.hf_repo = hf_repo
    if hf_file is not None:
        sp.hf_file = hf_file
    if image_tag is not None:
        sp.image_tag = image_tag
    for k, v in _parse_set_kv(set_env, backend=bk).items():
        sp.env_vars[k] = v
    for k in unset_env:
        sp.env_vars.pop(k, None)

    profile_store.save_profile(sp)
    print(f"Updated profile '{name}' (backend={bk})")


@app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
    with_config: bool = typer.Option(
        False, "--with-config", help="Also delete the linked config YAML."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a profile (and optionally its linked config)."""
    bk = detect_backend(name, override=backend)
    if not yes:
        suffix = " AND its config YAML" if with_config else ""
        confirm = typer.confirm(f"Delete profile '{name}' (backend={bk}){suffix}?")
        if not confirm:
            raise typer.Exit(code=0)

    if with_config:
        if bk == "vllm":
            from tui.backends.vllm.backend_storage import delete_profile as _d

            _d(name, delete_config=True)
        else:
            from tui.backends.llamacpp.backend import delete_profile as _d

            _d(name, delete_config_too=True)
    else:
        profile_store.delete_profile(name, bk)
    print(f"Deleted profile '{name}' (backend={bk})")


# ---- quick-setup -------------------------------------------------------------

_LLAMACPP_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


def _normalize_hf_repo(raw: str) -> str:
    """Strip huggingface.co URL wrapping → bare 'org/name' (mirrors the TUI)."""
    s = raw.strip().rstrip("/")
    if not s:
        return ""
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]
        if s.startswith("api/models/"):
            s = s[len("api/models/") :]
    parts = s.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


@app.command("quick-setup")
def quick_setup(
    model: Optional[str] = typer.Argument(
        None,
        help="HF model id (vLLM only — required for --backend vllm, e.g. Qwen/Qwen3-8B). "
             "Ignored for --backend llamacpp; use --hf-repo + --hf-file there.",
    ),
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help=f"Backend ({', '.join(BACKENDS)})."
    ),
    name: str = typer.Option(
        "", "--name", "-n", help="Profile name (auto-derived from model/repo if omitted)."
    ),
    port: int = typer.Option(
        0, "--port", "-p",
        help="Host port (0 = backend default: vLLM 8000, llama.cpp 8080).",
    ),
    gpu_id: str = typer.Option("0", "--gpu-id", "-g", help="GPU id(s), comma-separated."),
    # vLLM-only
    gpu_memory_utilization: str = typer.Option(
        "0.9", "--gpu-mem", help="vLLM only: gpu-memory-utilization (0.0–1.0)."
    ),
    enable_lora: bool = typer.Option(False, "--lora/--no-lora", help="vLLM only."),
    # llama.cpp-only
    hf_repo: str = typer.Option(
        "", "--hf-repo",
        help="llama.cpp only: HuggingFace repo (e.g. unsloth/Qwen3-30B-A3B-GGUF).",
    ),
    hf_file: str = typer.Option(
        "", "--hf-file",
        help="llama.cpp only: GGUF filename inside the repo (validated against HF API when reachable).",
    ),
    ctx_size: str = typer.Option(
        "32768", "--ctx-size",
        help="llama.cpp only: --ctx-size (token context). Empty = backend default.",
    ),
    n_gpu_layers: str = typer.Option(
        "99", "--n-gpu-layers",
        help="llama.cpp only: --n-gpu-layers (99 = all). Empty = backend default.",
    ),
    cache_type_k: str = typer.Option(
        "bf16", "--cache-type-k",
        help="llama.cpp only: --cache-type-k (f16/bf16/q8_0/q4_0). Empty = backend default.",
    ),
    cache_type_v: str = typer.Option(
        "bf16", "--cache-type-v",
        help="llama.cpp only: --cache-type-v (f16/bf16/q8_0/q4_0). Empty = backend default.",
    ),
    flash_attn: bool = typer.Option(
        True, "--flash-attn/--no-flash-attn", help="llama.cpp only: --flash-attn."
    ),
    jinja: bool = typer.Option(
        True, "--jinja/--no-jinja",
        help="llama.cpp only: --jinja (required for /v1/chat/completions).",
    ),
    override_tensors: str = typer.Option(
        "", "--override-tensors",
        help="llama.cpp only: --override-tensors regex (MoE expert offload).",
    ),
    # Shared (now both backends)
    copy_config_from: str = typer.Option(
        "", "--copy-from",
        help="Copy extra_params (vLLM) / all params (llama.cpp) from an existing config name.",
    ),
) -> None:
    """Create a profile + config in one step (mirrors TUI Quick Setup, both backends)."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"unknown backend: {backend}", param_hint="--backend")

    if backend == "vllm":
        _quick_setup_vllm(
            model=model,
            name=name,
            port=port or 8000,
            gpu_id=gpu_id,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            copy_config_from=copy_config_from,
        )
        return

    # llama.cpp path: warn (don't fail) if the positional MODEL was supplied —
    # it's meaningless here (HF model id makes no sense for a GGUF profile) and
    # silently dropping it is a real footgun in scripts. Emit clearly on stderr
    # so a caller can see they probably meant `--name MODEL` (or know the
    # auto-derive from --hf-repo is what they'll actually get).
    if model and not name:
        typer.echo(
            f"Warning: positional MODEL '{model}' is ignored for --backend "
            "llamacpp; the profile name will be auto-derived from --hf-repo. "
            f"Pass --name {model!r} explicitly if you meant that as the name.",
            err=True,
        )
    elif model and name:
        typer.echo(
            f"Warning: positional MODEL '{model}' is ignored for --backend "
            f"llamacpp (using --name '{name}').",
            err=True,
        )

    _quick_setup_llamacpp(
        hf_repo=hf_repo,
        hf_file=hf_file,
        name=name,
        port=port or 8080,
        gpu_id=gpu_id,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        flash_attn=flash_attn,
        jinja=jinja,
        override_tensors=override_tensors,
        copy_config_from=copy_config_from,
    )


def _quick_setup_vllm(
    *,
    model: Optional[str],
    name: str,
    port: int,
    gpu_id: str,
    gpu_memory_utilization: str,
    enable_lora: bool,
    copy_config_from: str,
) -> None:
    if not model:
        raise typer.BadParameter(
            "vllm quick-setup requires a HF model id (positional MODEL).",
            param_hint="MODEL",
        )

    if not name:
        tail = model.rsplit("/", 1)[-1]
        derived = re.sub(r"[^a-z0-9-]", "-", tail.lower()).strip("-")
        if not derived:
            raise typer.BadParameter(
                "could not derive name from model; pass --name explicitly", param_hint="--name"
            )
        name = derived

    if not _PROFILE_NAME_RE.match(name):
        raise typer.BadParameter("derived name has invalid characters; pass --name explicitly")

    from tui.backends.vllm.backend_storage import (
        list_profile_names as v_list,
        list_config_names as v_clist,
        load_config,
        save_config,
    )
    from tui.backends.vllm.backend_common import Config

    existing_profiles = v_list()
    existing_configs = v_clist()
    final_name = name
    suffix = 0
    while final_name in existing_profiles or final_name in existing_configs:
        suffix += 1
        final_name = f"{name}-{suffix}"

    extra_params: dict = {}
    if copy_config_from:
        src = load_config(copy_config_from)
        extra_params = dict(src.extra_params)

    save_config(
        Config(
            name=final_name,
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_params=extra_params,
        )
    )

    profile_store.save_profile(
        profile_store.StoredProfile(
            name=final_name,
            backend="vllm",
            container_name=final_name,
            port=port,
            gpu_id=gpu_id,
            tensor_parallel_size=len(gpu_id.split(",")) if gpu_id else 1,
            config_name=final_name,
            model_id=model,
            enable_lora=enable_lora,
        )
    )
    print(f"Created profile + config: {final_name}")


def _quick_setup_llamacpp(
    *,
    hf_repo: str,
    hf_file: str,
    name: str,
    port: int,
    gpu_id: str,
    ctx_size: str,
    n_gpu_layers: str,
    cache_type_k: str,
    cache_type_v: str,
    flash_attn: bool,
    jinja: bool,
    override_tensors: str,
    copy_config_from: str,
) -> None:
    repo = _normalize_hf_repo(hf_repo)
    if not repo or not _LLAMACPP_REPO_RE.match(repo):
        raise typer.BadParameter(
            "llama.cpp quick-setup requires a valid HF repo (org/name).",
            param_hint="--hf-repo",
        )
    if not hf_file:
        raise typer.BadParameter(
            "llama.cpp quick-setup requires --hf-file (GGUF filename inside the repo).",
            param_hint="--hf-file",
        )
    if not 1024 <= int(port) <= 65535:
        raise typer.BadParameter(
            "port must be in 1024–65535", param_hint="--port"
        )
    if not re.fullmatch(r"[0-9](,[0-9])*", gpu_id):
        raise typer.BadParameter(
            "GPU id must be digits separated by commas (e.g. '0' or '0,1').",
            param_hint="--gpu-id",
        )

    from tui.backends.llamacpp.backend import (
        Config as LcppConfig,
        list_config_names as l_clist,
        list_hf_repo_files,
        list_profile_names as l_list,
        load_config as l_load_config,
        save_config as l_save_config,
        validate_name as l_validate_name,
    )

    # Validate the GGUF filename against the live repo listing when reachable.
    # If the HF API call fails (network down, private repo, rate-limit) we
    # don't hard-fail — the TUI also lets you proceed once you've selected a
    # file, and a headless caller may already know what's in the repo.
    files = run_async(list_hf_repo_files(repo))
    gguf_files = [
        str(f.get("path", ""))
        for f in files
        if isinstance(f, dict)
        and f.get("type") == "file"
        and str(f.get("path", "")).lower().endswith(".gguf")
    ]
    if gguf_files and hf_file not in gguf_files:
        available = "\n  ".join(gguf_files[:20])
        more = f"\n  ... ({len(gguf_files) - 20} more)" if len(gguf_files) > 20 else ""
        raise typer.BadParameter(
            f"'{hf_file}' not found in {repo}. Available GGUF files:\n  {available}{more}",
            param_hint="--hf-file",
        )

    # Auto-derive name: <repo-tail-without-GGUF>, lowercased + safe chars.
    if not name:
        base = repo.rsplit("/", 1)[-1]
        base = re.sub(r"[-_]?GGUF$", "", base, flags=re.I)
        name = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
    if not name:
        raise typer.BadParameter(
            "could not derive name from repo; pass --name explicitly", param_hint="--name"
        )
    if not l_validate_name(name):
        raise typer.BadParameter(
            "llama.cpp profile names must be lowercase: start with [a-z0-9], "
            "then [a-z0-9_-] only.",
            param_hint="--name",
        )

    existing = set(l_list()) | set(l_clist())
    final_name = name
    suffix = 0
    while final_name in existing:
        suffix += 1
        final_name = f"{name}-{suffix}"

    # --- Build llama.cpp config params (mirrors QuickSetupScreen.on_create) ---
    params: dict = {}
    if copy_config_from:
        src = l_load_config(copy_config_from)
        params.update(src.params)
    params["model-file"] = hf_file
    params.setdefault("alias", final_name)

    def _set_int(key: str, raw: str) -> None:
        if not raw:
            params.pop(key, None)
            return
        try:
            params[key] = int(raw)
        except ValueError:
            params[key] = raw

    _set_int("ctx-size", ctx_size)
    _set_int("n-gpu-layers", n_gpu_layers)
    if cache_type_k:
        params["cache-type-k"] = cache_type_k
    else:
        params.pop("cache-type-k", None)
    if cache_type_v:
        params["cache-type-v"] = cache_type_v
    else:
        params.pop("cache-type-v", None)

    if flash_attn:
        params["flash-attn"] = True
    else:
        params.pop("flash-attn", None)
    if jinja:
        params["jinja"] = True
    else:
        params.pop("jinja", None)
    if override_tensors:
        params["override-tensors"] = [override_tensors]
    else:
        params.pop("override-tensors", None)

    l_save_config(LcppConfig(name=final_name, params=params))

    profile_store.save_profile(
        profile_store.StoredProfile(
            name=final_name,
            backend="llamacpp",
            container_name=final_name,
            port=int(port),
            gpu_id=gpu_id,
            config_name=final_name,
            model_file=hf_file,
            hf_repo=repo,
            hf_file=hf_file,
        )
    )
    print(f"Created profile + config: {final_name}")
