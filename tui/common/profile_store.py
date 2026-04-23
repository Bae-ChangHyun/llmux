"""Unified YAML-based profile storage.

Profiles live in a single `profiles.yaml` at the repo root. At launch time,
each profile is rendered into `.runtime/<backend>/<name>.env` for
`docker compose --env-file` consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shlex
from pathlib import Path
from typing import Any, Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

Backend = Literal["vllm", "llamacpp"]

DEFAULTS: dict[str, dict[str, Any]] = {
    "vllm": {
        "port": 8000,
        "gpu_id": "0",
        "tensor_parallel_size": 1,
        "enable_lora": False,
    },
    "llamacpp": {
        "port": 8080,
        "gpu_id": "0",
    },
}

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class StoredProfile:
    """Superset profile record; fields not applicable to a backend stay default."""

    name: str
    backend: str
    container_name: str = ""
    port: int = 0
    gpu_id: str = "0"
    config_name: str = ""
    tensor_parallel_size: int = 1
    model_id: str = ""
    enable_lora: bool = False
    max_loras: int | None = None
    max_lora_rank: int | None = None
    lora_modules: str = ""
    extra_pip_packages: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    model_file: str = ""
    hf_repo: str = ""
    hf_file: str = ""


def _load_yaml() -> dict:
    if not PROFILES_YAML.exists():
        return {"version": 1, "defaults": DEFAULTS, "profiles": []}
    raw = yaml.safe_load(PROFILES_YAML.read_text()) or {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("version", 1)
    raw.setdefault("defaults", DEFAULTS)
    raw.setdefault("profiles", [])
    return raw


def _backend_defaults(data: dict, backend: str) -> dict[str, Any]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    defaults = dict(DEFAULTS[backend])
    user_defaults = data.get("defaults", {})
    if isinstance(user_defaults, dict) and isinstance(
        user_defaults.get(backend), dict
    ):
        defaults.update(user_defaults[backend])
    return defaults


def _write_yaml(data: dict) -> None:
    PROFILES_YAML.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=120,
        )
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_profile(entry: dict, defaults: dict[str, Any] | None = None) -> StoredProfile:
    backend = entry.get("backend")
    if backend not in ("vllm", "llamacpp"):
        raise ValueError(f"Invalid backend {backend!r} in profile {entry.get('name')}")
    defaults = defaults or DEFAULTS[backend]
    merged: dict[str, Any] = dict(defaults)
    merged.update(entry)
    name = merged["name"]
    return StoredProfile(
        name=name,
        backend=backend,
        container_name=merged.get("container_name", name),
        port=int(merged.get("port", defaults["port"])),
        gpu_id=str(merged.get("gpu_id", "0")),
        config_name=merged.get("config_name", name),
        tensor_parallel_size=int(merged.get("tensor_parallel_size", 1)),
        model_id=merged.get("model_id", ""),
        enable_lora=_parse_bool(merged.get("enable_lora", False)),
        max_loras=merged.get("max_loras"),
        max_lora_rank=merged.get("max_lora_rank"),
        lora_modules=merged.get("lora_modules", ""),
        extra_pip_packages=merged.get("extra_pip_packages", ""),
        env_vars=dict(merged.get("env_vars") or {}),
        model_file=merged.get("model_file", ""),
        hf_repo=merged.get("hf_repo", ""),
        hf_file=merged.get("hf_file", ""),
    )


def _profile_to_entry(
    profile: StoredProfile, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or DEFAULTS[profile.backend]
    out: dict[str, Any] = {"name": profile.name, "backend": profile.backend}
    if profile.container_name and profile.container_name != profile.name:
        out["container_name"] = profile.container_name
    if profile.port and profile.port != defaults.get("port"):
        out["port"] = profile.port
    if profile.gpu_id and profile.gpu_id != defaults.get("gpu_id", "0"):
        out["gpu_id"] = profile.gpu_id
    if profile.config_name and profile.config_name != profile.name:
        out["config_name"] = profile.config_name

    if profile.backend == "vllm":
        if profile.tensor_parallel_size != defaults.get("tensor_parallel_size", 1):
            out["tensor_parallel_size"] = profile.tensor_parallel_size
        if profile.model_id:
            out["model_id"] = profile.model_id
        if profile.enable_lora != defaults.get("enable_lora", False):
            out["enable_lora"] = profile.enable_lora
        if profile.max_loras is not None:
            out["max_loras"] = profile.max_loras
        if profile.max_lora_rank is not None:
            out["max_lora_rank"] = profile.max_lora_rank
        if profile.lora_modules:
            out["lora_modules"] = profile.lora_modules
        if profile.extra_pip_packages:
            out["extra_pip_packages"] = profile.extra_pip_packages
        if profile.env_vars:
            out["env_vars"] = dict(profile.env_vars)
    else:
        if profile.model_file:
            out["model_file"] = profile.model_file
        if profile.hf_repo:
            out["hf_repo"] = profile.hf_repo
        if profile.hf_file:
            out["hf_file"] = profile.hf_file
    return out


def list_profiles(backend: str) -> list[StoredProfile]:
    data = _load_yaml()
    defaults = _backend_defaults(data, backend)
    return [
        _to_profile(p, defaults)
        for p in data.get("profiles", [])
        if p.get("backend") == backend
    ]


def list_profile_names(backend: str) -> list[str]:
    return sorted(p.name for p in list_profiles(backend))


def load_profile(name: str, backend: str) -> StoredProfile | None:
    for p in list_profiles(backend):
        if p.name == name:
            return p
    return None


def save_profile(profile: StoredProfile) -> None:
    data = _load_yaml()
    profiles = data.get("profiles", [])
    _render_env_lines(profile)
    entry = _profile_to_entry(profile, _backend_defaults(data, profile.backend))
    for idx, existing in enumerate(profiles):
        if (
            existing.get("name") == profile.name
            and existing.get("backend") == profile.backend
        ):
            profiles[idx] = entry
            break
    else:
        profiles.append(entry)
    data["profiles"] = profiles
    _write_yaml(data)
    render_env(profile)


def delete_profile(name: str, backend: str) -> bool:
    data = _load_yaml()
    profiles = data.get("profiles", [])
    remaining = [
        p for p in profiles
        if not (p.get("name") == name and p.get("backend") == backend)
    ]
    if len(remaining) == len(profiles):
        return False
    data["profiles"] = remaining
    _write_yaml(data)
    rt = runtime_env_path(name, backend)
    if rt.exists():
        rt.unlink()
    return True


def runtime_env_path(name: str, backend: str) -> Path:
    return RUNTIME_DIR / backend / f"{name}.env"


def _env_line(key: str, value: Any) -> str:
    if not _ENV_KEY_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    return f"{key}={shlex.quote(str(value))}"


def _render_env_lines(profile: StoredProfile) -> list[str]:
    lines: list[str] = [
        "# Auto-rendered from profiles.yaml — do not edit directly.",
        f"# Profile: {profile.name} ({profile.backend})",
        "",
    ]
    if profile.backend == "vllm":
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("VLLM_PORT", profile.port),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("TENSOR_PARALLEL_SIZE", profile.tensor_parallel_size),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
            _env_line("MODEL_ID", profile.model_id),
            _env_line("ENABLE_LORA", "true" if profile.enable_lora else "false"),
        ]
        if profile.max_loras is not None:
            lines.append(_env_line("MAX_LORAS", profile.max_loras))
        if profile.max_lora_rank is not None:
            lines.append(_env_line("MAX_LORA_RANK", profile.max_lora_rank))
        if profile.lora_modules:
            lines.append(_env_line("LORA_MODULES", profile.lora_modules))
        if profile.extra_pip_packages:
            lines.append(_env_line("EXTRA_PIP_PACKAGES", profile.extra_pip_packages))
        for k, v in profile.env_vars.items():
            lines.append(_env_line(k, v))
    else:
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("LLAMA_PORT", profile.port),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
        ]
        if profile.model_file:
            lines.append(_env_line("MODEL_FILE", profile.model_file))
        if profile.hf_repo:
            lines.append(_env_line("HF_REPO", profile.hf_repo))
        if profile.hf_file:
            lines.append(_env_line("HF_FILE", profile.hf_file))
    lines.append("")
    return lines


def render_env(profile: StoredProfile) -> Path:
    out_path = runtime_env_path(profile.name, profile.backend)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _render_env_lines(profile)
    out_path.write_text("\n".join(lines))
    return out_path


def render_all(backend: str | None = None) -> list[Path]:
    backends = [backend] if backend else ["vllm", "llamacpp"]
    out: list[Path] = []
    for b in backends:
        for p in list_profiles(b):
            out.append(render_env(p))
    return out


def _cli() -> int:
    import sys

    argv = sys.argv[1:]
    if len(argv) == 3 and argv[0] == "render":
        backend, name = argv[1], argv[2]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        stored = load_profile(name, backend)
        if stored is None:
            print(f"Profile not found: {backend}/{name}", file=sys.stderr)
            return 1
        path = render_env(stored)
        print(path)
        return 0
    if len(argv) == 2 and argv[0] == "list":
        backend = argv[1]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        for name in list_profile_names(backend):
            print(name)
        return 0
    print(
        "Usage:\n"
        "  python -m tui.common.profile_store render <backend> <name>\n"
        "  python -m tui.common.profile_store list <backend>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
