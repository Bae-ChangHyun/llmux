"""Shared backend constants, validation, and dataclasses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = SCRIPT_DIR / "profiles"
CONFIG_DIR = SCRIPT_DIR / "config"
COMMON_ENV = SCRIPT_DIR / ".env.common"
VLLM_SRC_DIR = SCRIPT_DIR / ".vllm-src"
DEFAULT_VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"


def validate_name(name: str) -> bool:
    """Check that name contains only alphanumeric, dash, and underscore.
    Also prevents argument injection (names starting with -)."""
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name))


@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: str = "8000"
    gpu_id: str = "0"
    tensor_parallel: str = "1"
    config_name: str = ""
    model_id: str = ""
    enable_lora: str = "false"
    max_loras: str = ""
    max_lora_rank: str = ""
    lora_modules: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return PROFILES_DIR / f"{self.name}.env"


@dataclass
class Config:
    name: str
    model: str = ""
    gpu_memory_utilization: str = "0.9"
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.name}.yaml"


@dataclass
class ContainerStatus:
    profile_name: str
    container_name: str
    running: bool = False
    status_text: str = "stopped"
    health: str = ""
    port: str = ""
    gpu_id: str = ""
    image: str = ""
    model: str = ""
    lora: bool = False


@dataclass
class GpuInfo:
    index: str
    name: str
    memory_used: str
    memory_total: str
    utilization: str
    temperature: str


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str
