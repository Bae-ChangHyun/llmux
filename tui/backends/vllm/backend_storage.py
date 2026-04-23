"""vLLM backend profile/config I/O — delegates profiles to the shared YAML store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tui.common import profile_store

from .backend_common import CONFIG_DIR, Config, Profile


def _parse_env_file(path: Path | str) -> dict[str, str]:
    """Parse a .env file into a dict (comments and blanks skipped). Missing file = {}."""
    path = Path(path)
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:
            if " #" in value:
                value = value[: value.index(" #")].rstrip()
        data[key.strip()] = value
    return data


def _to_profile(stored: profile_store.StoredProfile) -> Profile:
    return Profile(
        name=stored.name,
        container_name=stored.container_name or stored.name,
        port=str(stored.port),
        gpu_id=stored.gpu_id,
        tensor_parallel=str(stored.tensor_parallel_size),
        config_name=stored.config_name,
        model_id=stored.model_id,
        enable_lora="true" if stored.enable_lora else "false",
        max_loras=str(stored.max_loras) if stored.max_loras is not None else "",
        max_lora_rank=str(stored.max_lora_rank) if stored.max_lora_rank is not None else "",
        lora_modules=stored.lora_modules,
        env_vars={
            **({"EXTRA_PIP_PACKAGES": stored.extra_pip_packages} if stored.extra_pip_packages else {}),
            **stored.env_vars,
        },
    )


def _to_stored(profile: Profile) -> profile_store.StoredProfile:
    env_vars = dict(profile.env_vars)
    extra_pip = env_vars.pop("EXTRA_PIP_PACKAGES", "").strip()
    return profile_store.StoredProfile(
        name=profile.name,
        backend="vllm",
        container_name=profile.container_name or profile.name,
        port=int(profile.port or 8000),
        gpu_id=profile.gpu_id or "0",
        config_name=profile.config_name or profile.name,
        tensor_parallel_size=int(profile.tensor_parallel or 1),
        model_id=profile.model_id,
        enable_lora=(profile.enable_lora or "false").lower() == "true",
        max_loras=int(profile.max_loras) if str(profile.max_loras).strip() else None,
        max_lora_rank=int(profile.max_lora_rank) if str(profile.max_lora_rank).strip() else None,
        lora_modules=profile.lora_modules,
        extra_pip_packages=extra_pip,
        env_vars=env_vars,
    )


def load_profile(name: str) -> Profile:
    stored = profile_store.load_profile(name, "vllm")
    if stored is None:
        return Profile(name=name)
    profile_store.render_env(stored)
    return _to_profile(stored)


def save_profile(profile: Profile) -> None:
    profile_store.save_profile(_to_stored(profile))


def delete_profile(name: str, delete_config: bool = False) -> None:
    if delete_config:
        stored = profile_store.load_profile(name, "vllm")
        config_name = stored.config_name if stored else ""
        if config_name:
            other_refs = [
                n for n in profile_store.list_profile_names("vllm")
                if n != name and (profile_store.load_profile(n, "vllm") or None)
                and profile_store.load_profile(n, "vllm").config_name == config_name  # type: ignore[union-attr]
            ]
            if not other_refs:
                config_path = CONFIG_DIR / f"{config_name}.yaml"
                if config_path.exists():
                    config_path.unlink()
    profile_store.delete_profile(name, "vllm")


def list_profile_names() -> list[str]:
    return profile_store.list_profile_names("vllm")


def load_config(name: str) -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return Config(name=name)

    raw_data = yaml.safe_load(path.read_text())
    if raw_data is None:
        data: dict[str, Any] = {}
    elif isinstance(raw_data, dict):
        data = dict(raw_data)
    else:
        data = {}

    model = str(data.pop("model", ""))
    gpu_mem = str(data.pop("gpu-memory-utilization", "0.9"))
    extra = {str(key): value for key, value in data.items()}
    return Config(name=name, model=model, gpu_memory_utilization=gpu_mem, extra_params=extra)


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model": config.model,
        "gpu-memory-utilization": config.gpu_memory_utilization,
    }
    for key, value in config.extra_params.items():
        data[key] = True if value == "" else value
    config.path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    )


def parse_config_param_value(raw_value: str) -> Any:
    if raw_value == "":
        return True
    return yaml.safe_load(raw_value)


def format_config_param_value(value: Any) -> str:
    if value is True:
        return ""
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(
            value,
            default_flow_style=True,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
    return str(value)


def delete_config(name: str) -> None:
    path = CONFIG_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()


def list_config_names() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))
