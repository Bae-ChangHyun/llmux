"""Profile/config file parsing and persistence helpers."""

from __future__ import annotations

from typing import Any

import yaml

from .backend_common import CONFIG_DIR, PROFILES_DIR, Config, Profile


def _parse_env_file(path) -> dict[str, str]:
    """Parse a .env file into a dict, ignoring comments and blank lines."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                if " #" in value:
                    value = value[:value.index(" #")].rstrip()
            data[key.strip()] = value
    return data


def load_profile(name: str) -> Profile:
    path = PROFILES_DIR / f"{name}.env"
    data = _parse_env_file(path)
    known_keys = {
        "CONTAINER_NAME", "VLLM_PORT", "GPU_ID", "TENSOR_PARALLEL_SIZE",
        "CONFIG_NAME", "MODEL_ID", "ENABLE_LORA", "MAX_LORAS", "MAX_LORA_RANK", "LORA_MODULES",
    }
    env_vars = {k: v for k, v in data.items() if k not in known_keys}
    return Profile(
        name=name,
        container_name=data.get("CONTAINER_NAME", name),
        port=data.get("VLLM_PORT", "8000"),
        gpu_id=data.get("GPU_ID", "0"),
        tensor_parallel=data.get("TENSOR_PARALLEL_SIZE", "1"),
        config_name=data.get("CONFIG_NAME", ""),
        model_id=data.get("MODEL_ID", ""),
        enable_lora=data.get("ENABLE_LORA", "false"),
        max_loras=data.get("MAX_LORAS", ""),
        max_lora_rank=data.get("MAX_LORA_RANK", ""),
        lora_modules=data.get("LORA_MODULES", ""),
        env_vars=env_vars,
    )


def save_profile(profile: Profile) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Profile: {profile.name}",
        f"# GPU: {profile.gpu_id}, Port: {profile.port}",
        "",
        f"CONTAINER_NAME={profile.container_name}",
        f"VLLM_PORT={profile.port}",
        f"CONFIG_NAME={profile.config_name}",
        f"MODEL_ID={profile.model_id}",
        "",
        "# GPU Configuration",
        f"GPU_ID={profile.gpu_id}",
        f"TENSOR_PARALLEL_SIZE={profile.tensor_parallel}",
        "",
        "# LoRA Configuration",
        f"ENABLE_LORA={profile.enable_lora}",
    ]
    if profile.max_loras:
        lines.append(f"MAX_LORAS={profile.max_loras}")
    if profile.max_lora_rank:
        lines.append(f"MAX_LORA_RANK={profile.max_lora_rank}")
    if profile.lora_modules:
        lines.append(f"LORA_MODULES={profile.lora_modules}")
    if profile.env_vars:
        lines.append("")
        lines.append("# Extra environment variables")
        for key, value in profile.env_vars.items():
            lines.append(f"{key}={value}")
    lines.append("")
    profile.path.write_text("\n".join(lines))


def delete_profile(name: str, delete_config: bool = False) -> None:
    path = PROFILES_DIR / f"{name}.env"
    if delete_config and path.exists():
        data = _parse_env_file(path)
        config_name = data.get("CONFIG_NAME", "")
        if config_name:
            other_refs = [
                profile_name for profile_name in list_profile_names()
                if profile_name != name and load_profile(profile_name).config_name == config_name
            ]
            if not other_refs:
                config_path = CONFIG_DIR / f"{config_name}.yaml"
                if config_path.exists():
                    config_path.unlink()
    if path.exists():
        path.unlink()


def list_profile_names() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        path.stem for path in PROFILES_DIR.glob("*.env")
        if path.stem != "example"
    )


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
    """Parse a config parameter value from the UI into a YAML-safe Python value.

    Blank values preserve the existing shortcut semantics for boolean flags and are
    written back as ``true``.
    """
    if raw_value == "":
        return True
    return yaml.safe_load(raw_value)


def format_config_param_value(value: Any) -> str:
    """Format a config parameter value for editing in the UI."""
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
