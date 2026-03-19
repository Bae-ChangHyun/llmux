"""Backend module: interfaces with run.sh CLI and system commands."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = SCRIPT_DIR / "profiles"
CONFIG_DIR = SCRIPT_DIR / "config"
RUN_SH = SCRIPT_DIR / "run.sh"


def validate_name(name: str) -> bool:
    """Check that name contains only alphanumeric, dash, and underscore.
    Also prevents argument injection (names starting with -)."""
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: str = "8000"
    gpu_id: str = "0"
    tensor_parallel: str = "1"
    config_name: str = ""
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
    extra_params: dict[str, str] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# Profile I/O
# ---------------------------------------------------------------------------

def _parse_env_file(path: Path) -> dict[str, str]:
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
            data[key.strip()] = value
    return data


def load_profile(name: str) -> Profile:
    path = PROFILES_DIR / f"{name}.env"
    data = _parse_env_file(path)
    known_keys = {
        "CONTAINER_NAME", "VLLM_PORT", "GPU_ID", "TENSOR_PARALLEL_SIZE",
        "CONFIG_NAME", "ENABLE_LORA", "MAX_LORAS", "MAX_LORA_RANK", "LORA_MODULES",
    }
    env_vars = {k: v for k, v in data.items() if k not in known_keys}
    return Profile(
        name=name,
        container_name=data.get("CONTAINER_NAME", name),
        port=data.get("VLLM_PORT", "8000"),
        gpu_id=data.get("GPU_ID", "0"),
        tensor_parallel=data.get("TENSOR_PARALLEL_SIZE", "1"),
        config_name=data.get("CONFIG_NAME", ""),
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
        for k, v in profile.env_vars.items():
            lines.append(f"{k}={v}")
    lines.append("")
    profile.path.write_text("\n".join(lines))


def delete_profile(name: str, delete_config: bool = False) -> None:
    path = PROFILES_DIR / f"{name}.env"
    if delete_config and path.exists():
        data = _parse_env_file(path)
        config_name = data.get("CONFIG_NAME", "")
        if config_name:
            config_path = CONFIG_DIR / f"{config_name}.yaml"
            if config_path.exists():
                config_path.unlink()
    if path.exists():
        path.unlink()


def list_profile_names() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        p.stem for p in PROFILES_DIR.glob("*.env")
        if p.stem != "example"
    )


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(name: str) -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return Config(name=name)
    data = yaml.safe_load(path.read_text()) or {}
    model = str(data.pop("model", ""))
    gpu_mem = str(data.pop("gpu-memory-utilization", "0.9"))
    extra = {str(k): str(v) for k, v in data.items()}
    return Config(name=name, model=model, gpu_memory_utilization=gpu_mem, extra_params=extra)


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"model: {config.model}",
        f"gpu-memory-utilization: {config.gpu_memory_utilization}",
    ]
    for k, v in config.extra_params.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    config.path.write_text("\n".join(lines))


def delete_config(name: str) -> None:
    path = CONFIG_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()


def list_config_names() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Docker / Container operations (async)
# ---------------------------------------------------------------------------

async def run_command(*args: str, timeout: float = 30) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def run_sh(*args: str, timeout: float = 60) -> tuple[int, str]:
    """Run ./run.sh with given arguments."""
    return await run_command(str(RUN_SH), *args, timeout=timeout)


async def is_container_running(container_name: str) -> bool:
    rc, out = await run_command("docker", "ps", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        return False
    return container_name in out.strip().splitlines()


async def get_container_statuses() -> list[ContainerStatus]:
    """Get status for all profiles, including health check status."""
    profiles = list_profile_names()
    # Single docker ps call to get running container names and health status
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Status}}",
        timeout=10,
    )
    running_info: dict[str, str] = {}
    if rc == 0:
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                running_info[parts[0]] = parts[1]

    statuses = []
    for name in profiles:
        p = load_profile(name)
        running = p.container_name in running_info
        health = ""
        status_text = "stopped"
        if running:
            docker_status = running_info[p.container_name]
            if "(healthy)" in docker_status:
                health = "healthy"
                status_text = "healthy"
            elif "(unhealthy)" in docker_status:
                health = "unhealthy"
                status_text = "unhealthy"
            elif "(health: starting)" in docker_status:
                health = "starting"
                status_text = "starting"
            else:
                status_text = "running"
        model = ""
        if p.config_name:
            c = load_config(p.config_name)
            model = c.model
        statuses.append(ContainerStatus(
            profile_name=name,
            container_name=p.container_name,
            running=running,
            status_text=status_text,
            health=health,
            port=p.port,
            gpu_id=p.gpu_id,
            model=model,
            lora=p.enable_lora == "true",
        ))
    return statuses


async def check_port_conflict(profile: Profile) -> str | None:
    """Check if the port is used by a *running* container from another profile."""
    rc, out = await run_command("docker", "ps", "--format", "{{.Names}}", timeout=10)
    running_names = set(out.strip().splitlines()) if rc == 0 else set()

    for name in list_profile_names():
        if name == profile.name:
            continue
        other = load_profile(name)
        if other.port == profile.port and other.container_name in running_names:
            return name
    return None


async def container_up(profile_name: str, use_dev: bool = False, tag: str = "") -> tuple[int, str]:
    args = [profile_name, "up"]
    if use_dev:
        args.append("--dev")
    if tag:
        args.extend(["--tag", tag])
    return await run_sh(*args, timeout=600)


async def stream_container_up(profile_name: str, use_dev: bool = False, tag: str = ""):
    """Async generator that streams run.sh output line by line, then yields return code."""
    args = [str(RUN_SH), profile_name, "up"]
    if use_dev:
        args.append("--dev")
    if tag:
        args.extend(["--tag", tag])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode(errors="replace").rstrip("\n"))
        await proc.wait()
        yield ("rc", proc.returncode or 0)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise


async def container_down(profile_name: str) -> tuple[int, str]:
    return await run_sh(profile_name, "down", timeout=30)


async def stream_container_logs(container_name: str):
    """Async generator that yields log lines."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", "100", container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip("\n")
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

async def get_gpu_info() -> list[GpuInfo]:
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
        timeout=10,
    )
    if rc != 0:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(GpuInfo(
                index=parts[0], name=parts[1],
                memory_used=parts[2], memory_total=parts[3],
                utilization=parts[4], temperature=parts[5],
            ))
    return gpus


async def get_docker_images(repo: str = "vllm/vllm-openai") -> list[DockerImage]:
    rc, out = await run_command(
        "docker", "images", repo,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        return []
    images = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            images.append(DockerImage(
                repository=parts[0], tag=parts[1],
                size=parts[2], created=parts[3],
            ))
    return images


async def get_dev_images() -> list[DockerImage]:
    return await get_docker_images("vllm-dev")


# ---------------------------------------------------------------------------
# Version info for container start screen
# ---------------------------------------------------------------------------

async def get_local_latest_tag() -> str:
    """Get the latest local vllm/vllm-openai image tag (excluding nightly)."""
    rc, out = await run_command(
        "docker", "images", "vllm/vllm-openai",
        "--format", "{{.Tag}}",
        timeout=10,
    )
    if rc != 0 or not out.strip():
        return "none"
    for tag in out.strip().splitlines():
        tag = tag.strip()
        if tag and tag != "<none>":
            return tag
    return "none"


async def get_dockerhub_release_version() -> str:
    """Get latest release version from Docker Hub (v0.x.x format)."""
    import json
    rc, out = await run_command(
        "curl", "-s", "--connect-timeout", "5", "--max-time", "10",
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return "unknown"
    try:
        data = json.loads(out)
        for result in data.get("results", []):
            name = result.get("name", "")
            if re.match(r"^v\d+\.\d+\.\d+", name):
                return name
    except (json.JSONDecodeError, KeyError):
        pass
    return "unknown"


async def get_dockerhub_nightly_date() -> str:
    """Get last updated date of the nightly tag from Docker Hub."""
    import json
    rc, out = await run_command(
        "curl", "-s", "--connect-timeout", "5", "--max-time", "10",
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/nightly",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return "unknown"
    try:
        data = json.loads(out)
        last_updated = data.get("last_updated", "")
        if last_updated:
            return last_updated.split("T")[0]
    except (json.JSONDecodeError, KeyError):
        pass
    return "unknown"
