"""Backend module: container/runtime operations for the TUI."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

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
            else:
                # Strip inline comments (KEY=value # comment)
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
            # Only delete config if no other profile references it
            other_refs = [
                n for n in list_profile_names()
                if n != name and load_profile(n).config_name == config_name
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
    extra = {str(k): "" if isinstance(v, bool) else str(v) for k, v in data.items()}
    return Config(name=name, model=model, gpu_memory_utilization=gpu_mem, extra_params=extra)


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {"model": config.model, "gpu-memory-utilization": config.gpu_memory_utilization}
    for k, v in config.extra_params.items():
        data[k] = True if v == "" else v
    config.path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))


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
        cwd=str(SCRIPT_DIR),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def run_command_with_options(
    *args: str,
    timeout: float = 30,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command with explicit cwd/env and return combined output."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd or SCRIPT_DIR),
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def stream_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    """Yield stdout/stderr lines followed by the final return code."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd or SCRIPT_DIR),
        env=env,
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


def _common_env() -> dict[str, str]:
    return _parse_env_file(COMMON_ENV)


def get_dev_build_defaults() -> tuple[str, str]:
    common = _common_env()
    repo_url = common.get("VLLM_REPO_URL", DEFAULT_VLLM_REPO_URL) or DEFAULT_VLLM_REPO_URL
    branch = common.get("VLLM_BRANCH", "main") or "main"
    return repo_url, branch


def _build_lora_options(profile: Profile) -> str:
    if profile.enable_lora != "true":
        return ""
    parts = ["--enable-lora"]
    if profile.max_loras:
        parts.extend(["--max-loras", profile.max_loras])
    if profile.max_lora_rank:
        parts.extend(["--max-lora-rank", profile.max_lora_rank])
    if profile.lora_modules:
        parts.extend(["--lora-modules", profile.lora_modules.replace(",", " ")])
    return " ".join(parts)


def _ensure_common_env(profile: Profile) -> tuple[bool, list[str]]:
    if not COMMON_ENV.exists():
        return False, [
            "Error: .env.common not found.",
            "Create it from .env.common.example before starting containers.",
        ]

    common = _common_env()
    hf_cache_path = common.get("HF_CACHE_PATH", "")
    if not hf_cache_path:
        return False, ["Error: HF_CACHE_PATH is not set in .env.common"]
    if not os.path.isabs(hf_cache_path):
        return False, [f"Error: HF_CACHE_PATH must be an absolute path. Current value: {hf_cache_path}"]

    if profile.enable_lora == "true":
        lora_base_path = common.get("LORA_BASE_PATH", "")
        if not lora_base_path:
            return False, ["Error: ENABLE_LORA=true but LORA_BASE_PATH is not set in .env.common"]
        if not os.path.isabs(lora_base_path):
            return False, [f"Error: LORA_BASE_PATH must be an absolute path. Current value: {lora_base_path}"]

    return True, []


def _ensure_profile_config(profile: Profile) -> tuple[bool, list[str]]:
    messages: list[str] = []
    if not profile.config_name:
        profile.config_name = profile.name
        save_profile(profile)
        messages.append(f"No config linked for '{profile.name}'. Auto-linked default config '{profile.config_name}'.")

    config = load_config(profile.config_name)
    if config.path.exists():
        model = config.model.strip()
        if model and model != "your-org/your-model":
            return True, messages
        messages.extend([
            f"Error: config/{profile.config_name}.yaml does not have a valid model configured yet.",
            f"Set the model field in config/{profile.config_name}.yaml or set MODEL_ID in profiles/{profile.name}.env, then start again.",
        ])
        return False, messages

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if profile.model_id:
        save_config(Config(
            name=profile.config_name,
            model=profile.model_id,
            gpu_memory_utilization="0.55",
        ))
        messages.append(f"Created default config: config/{profile.config_name}.yaml")
        return True, messages

    config.path.write_text(
        f"# Auto-generated default config for profile: {profile.name}\n"
        "# Set a valid Hugging Face model ID below, then start again.\n"
        "model: your-org/your-model\n"
        "gpu-memory-utilization: 0.55\n"
    )
    messages.extend([
        f"Created config/{profile.config_name}.yaml but MODEL_ID is not set for profile '{profile.name}'.",
        f"Edit the config model field or set MODEL_ID in profiles/{profile.name}.env, then start again.",
    ])
    return False, messages


def _compose_files(profile: Profile, use_dev: bool) -> list[str]:
    files = ["-f", "docker-compose.dev.yaml" if use_dev else "docker-compose.yaml"]
    if profile.enable_lora == "true":
        files.extend(["-f", "docker-compose.lora.yaml"])
    files.extend(["-f", "docker-compose.overrides.yaml"])
    return files


def _compose_env(
    profile: Profile,
    *,
    use_dev: bool,
    image_tag: str = "",
    version_tag: str = "",
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(_common_env())
    env.update(_parse_env_file(profile.path))
    env["PROFILE_PATH"] = str(profile.path)
    env["CONFIG_NAME"] = profile.config_name
    env["LORA_OPTIONS"] = _build_lora_options(profile)
    if use_dev:
        env["VLLM_DEV_TAG"] = image_tag
    else:
        env["VLLM_VERSION"] = version_tag
    return env


async def _container_exists(container_name: str) -> bool:
    rc, out = await run_command("docker", "ps", "-a", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        return False
    return container_name in out.strip().splitlines()


async def _gpu_conflict_messages(profile: Profile) -> list[str]:
    rc, out = await run_command("docker", "ps", "--format", "{{.Names}}", timeout=10)
    running_names = set(out.strip().splitlines()) if rc == 0 else set()
    messages: list[str] = []
    profile_gpu_ids = {gpu.strip() for gpu in profile.gpu_id.split(",") if gpu.strip()}
    for name in list_profile_names():
        if name == profile.name:
            continue
        other = load_profile(name)
        if other.container_name not in running_names:
            continue
        other_gpu_ids = {gpu.strip() for gpu in other.gpu_id.split(",") if gpu.strip()}
        for gpu_id in sorted(profile_gpu_ids & other_gpu_ids):
            messages.append(f"Warning: GPU {gpu_id} is also used by running container '{other.container_name}'")
    return messages


async def _detect_gpu_arch() -> str:
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
        timeout=10,
    )
    if rc != 0 or not out.strip():
        return ""
    return out.splitlines()[0].replace(".", "").strip()


async def _clone_or_update_vllm(repo_url: str, branch: str):
    if VLLM_SRC_DIR.joinpath(".git").exists():
        yield ("log", "Updating existing vLLM source...")
        rc, current_remote = await run_command_with_options(
            "git", "remote", "get-url", "origin",
            cwd=VLLM_SRC_DIR,
            timeout=30,
        )
        if rc != 0:
            yield ("log", current_remote.strip() or "Error: failed to inspect existing vLLM source")
            yield ("rc", 1)
            return
        if current_remote.strip() != repo_url:
            yield ("log", "Repository URL changed. Re-cloning...")
            shutil.rmtree(VLLM_SRC_DIR)

    if not VLLM_SRC_DIR.exists():
        async for event in stream_command(
            ["git", "clone", repo_url, str(VLLM_SRC_DIR)],
            cwd=SCRIPT_DIR,
        ):
            if event[0] == "rc":
                if event[1] != 0:
                    yield event
                    return
                continue
            yield event

    rc, out = await run_command_with_options("git", "fetch", "origin", cwd=VLLM_SRC_DIR, timeout=120)
    if rc != 0:
        yield ("log", out.strip() or "Error: git fetch failed")
        yield ("rc", rc)
        return

    rc, out = await run_command_with_options("git", "checkout", branch, cwd=VLLM_SRC_DIR, timeout=60)
    if rc != 0:
        rc, out = await run_command_with_options(
            "git", "checkout", "-b", branch, f"origin/{branch}",
            cwd=VLLM_SRC_DIR,
            timeout=60,
        )
        if rc != 0:
            yield ("log", out.strip() or f"Error: failed to checkout branch {branch}")
            yield ("rc", rc)
            return

    rc, out = await run_command_with_options("git", "pull", "origin", branch, cwd=VLLM_SRC_DIR, timeout=120)
    if rc != 0 and out.strip():
        yield ("log", out.strip())

    rc, commit_hash = await run_command_with_options(
        "git", "rev-parse", "--short", "HEAD",
        cwd=VLLM_SRC_DIR,
        timeout=30,
    )
    if rc != 0:
        yield ("log", commit_hash.strip() or "Error: failed to read commit hash")
        yield ("rc", rc)
        return
    yield ("commit", commit_hash.strip())


async def _stream_build_dev_image(
    branch: str,
    *,
    repo_url: str = DEFAULT_VLLM_REPO_URL,
    custom_tag: str = "",
    use_official: bool = False,
):
    gpu_arch = ""
    gpu_name = ""
    if not use_official:
        gpu_arch = await _detect_gpu_arch()
        if not gpu_arch:
            yield ("log", "Error: Could not detect GPU. Make sure nvidia-smi works.")
            yield ("rc", 1)
            return
        rc, gpu_name = await run_command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", timeout=10)
        gpu_name = gpu_name.splitlines()[0].strip() if rc == 0 and gpu_name.strip() else "unknown"

    from datetime import datetime, timezone

    main_tag = custom_tag or f"{branch}-{datetime.now().strftime('%Y%m%d')}"
    yield ("log", "Building vLLM from source")
    yield ("log", f"Repository: {repo_url}")
    yield ("log", f"Branch: {branch}")
    if use_official:
        yield ("log", "Using official vLLM Dockerfile (ALL architectures)")
    else:
        yield ("log", f"Detected GPU: {gpu_name} (sm_{gpu_arch})")
        yield ("log", "Building for your GPU only")
    yield ("log", f"Tag: vllm-dev:{main_tag}")

    commit_hash = ""
    async for event in _clone_or_update_vllm(repo_url, branch):
        if event[0] == "commit":
            commit_hash = event[1]
        else:
            yield event
            if event[0] == "rc" and event[1] != 0:
                return

    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = [
        "docker", "build",
        "-f", str(VLLM_SRC_DIR / "docker/Dockerfile"),
        "--build-arg", "RUN_WHEEL_CHECK=false",
        "--target", "vllm-openai",
        "--label", f"vllm.repo.url={repo_url}",
        "--label", f"vllm.repo.branch={branch}",
        "--label", f"vllm.commit.hash={commit_hash}",
        "--label", f"vllm.build.date={build_date}",
        "--label", f"vllm.build.type={'official' if use_official else 'fast'}",
        "-t", f"vllm-dev:{main_tag}",
        "-t", f"vllm-dev:{branch}",
    ]
    if not use_official:
        cmd.extend(["--build-arg", f"torch_cuda_arch_list={gpu_arch}"])
    cmd.append(str(VLLM_SRC_DIR))

    async for event in stream_command(cmd, cwd=SCRIPT_DIR):
        if event[0] == "rc":
            if event[1] != 0:
                yield event
            return
        yield event


async def _get_image_label(image_ref: str, label: str) -> str:
    rc, out = await run_command(
        "docker", "inspect", image_ref,
        f"--format={{{{index .Config.Labels {label!r}}}}}",
        timeout=20,
    )
    if rc != 0:
        return ""
    value = out.strip()
    return "" if value == "<no value>" else value


async def _dev_image_matches(image_tag: str, repo_url: str, branch: str) -> bool:
    image_ref = f"vllm-dev:{image_tag}"
    saved_repo = await _get_image_label(image_ref, "vllm.repo.url")
    saved_branch = await _get_image_label(image_ref, "vllm.repo.branch")
    if not saved_repo or not saved_branch:
        return False
    return saved_repo == repo_url and saved_branch == branch


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


async def container_up(
    profile_name: str,
    use_dev: bool = False,
    tag: str = "",
    repo_url: str = "",
    branch: str = "",
) -> tuple[int, str]:
    lines: list[str] = []
    rc = 0
    async for msg_type, data in stream_container_up(
        profile_name,
        use_dev=use_dev,
        tag=tag,
        repo_url=repo_url,
        branch=branch,
    ):
        if msg_type == "log":
            lines.append(data)
        elif msg_type == "rc":
            rc = int(data)
    return rc, "\n".join(lines)


async def stream_container_up(
    profile_name: str,
    use_dev: bool = False,
    tag: str = "",
    pull: bool = False,
    repo_url: str = "",
    branch: str = "",
):
    """Async generator that streams container startup output line by line."""
    profile = load_profile(profile_name)

    conflict = await check_port_conflict(profile)
    if conflict:
        yield ("log", f"Error: Port {profile.port} is already in use by profile '{conflict}'")
        yield ("rc", 1)
        return

    ok, messages = _ensure_common_env(profile)
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    ok, messages = _ensure_profile_config(profile)
    for message in messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    for message in await _gpu_conflict_messages(profile):
        yield ("log", message)

    extra_packages = profile.env_vars.get("EXTRA_PIP_PACKAGES", "").strip()
    if extra_packages:
        yield ("log", f"Extra pip packages: {extra_packages}")

    compose_files = _compose_files(profile, use_dev)

    if use_dev:
        default_repo_url, default_branch = get_dev_build_defaults()
        resolved_repo_url = repo_url.strip() or default_repo_url
        resolved_branch = branch.strip() or default_branch
        image_tag = tag or resolved_branch
        rc, _ = await run_command("docker", "image", "inspect", f"vllm-dev:{image_tag}", timeout=20)
        needs_build = rc != 0
        if not tag and not needs_build:
            needs_build = not await _dev_image_matches(image_tag, resolved_repo_url, resolved_branch)

        if needs_build:
            if tag:
                yield ("log", f"Error: Image vllm-dev:{image_tag} not found")
                rc, out = await run_command("docker", "images", "vllm-dev", "--format", "  {{.Tag}}", timeout=20)
                if rc == 0 and out.strip():
                    yield ("log", "Available images:")
                    for line in out.strip().splitlines():
                        yield ("log", line)
                yield ("rc", 1)
                return
            if rc == 0:
                yield ("log", "Existing dev image metadata does not match the requested repository/branch. Rebuilding...")
            else:
                yield ("log", "Dev image not found. Building first...")
            async for event in _stream_build_dev_image(resolved_branch, repo_url=resolved_repo_url):
                yield event
                if event[0] == "rc" and event[1] != 0:
                    return

        yield ("log", f"Using image: vllm-dev:{image_tag}")
        env = _compose_env(profile, use_dev=True, image_tag=image_tag)
        compose_cmd = [
            "docker", "compose", *compose_files,
            "--env-file", str(COMMON_ENV),
            "--env-file", str(profile.path),
            "-p", profile.name,
            "up", "-d",
        ]
    else:
        version_tag = tag or await get_local_latest_tag()
        if version_tag == "none":
            yield ("log", "Error: No local vllm/vllm-openai images found.")
            yield ("log", "Pull an image first: docker pull vllm/vllm-openai:latest")
            yield ("rc", 1)
            return

        yield ("log", f"Using image: vllm/vllm-openai:{version_tag}")
        env = _compose_env(profile, use_dev=False, version_tag=version_tag)
        compose_cmd = [
            "docker", "compose", *compose_files,
            "--env-file", str(COMMON_ENV),
            "--env-file", str(profile.path),
            "-p", profile.name,
            "up", "-d",
        ]
        if pull or version_tag in {"latest", "nightly"}:
            compose_cmd.extend(["--pull", "always"])

    async for event in stream_command(compose_cmd, cwd=SCRIPT_DIR, env=env):
        yield event
        if event[0] == "rc":
            if int(event[1]) == 0:
                yield ("log", f"{profile.name} started successfully!")
            return


async def container_down(profile_name: str) -> tuple[int, str]:
    profile = load_profile(profile_name)
    if not await _container_exists(profile.container_name):
        return 0, f"{profile_name} is not running"

    image_rc, image = await run_command(
        "docker", "inspect", profile.container_name,
        "--format={{.Config.Image}}",
        timeout=20,
    )
    use_dev = image_rc == 0 and image.strip().startswith("vllm-dev:")
    env = _compose_env(
        profile,
        use_dev=use_dev,
        image_tag=image.strip().split(":", 1)[1] if use_dev and ":" in image.strip() else "",
    )
    compose_cmd = [
        "docker", "compose", *_compose_files(profile, use_dev),
        "--env-file", str(COMMON_ENV),
        "--env-file", str(profile.path),
        "-p", profile.name,
        "down",
    ]
    rc, out = await run_command_with_options(*compose_cmd, cwd=SCRIPT_DIR, env=env, timeout=60)
    if rc == 0:
        return 0, f"{profile_name} stopped successfully!"

    stop_rc, stop_out = await run_command("docker", "stop", profile.container_name, timeout=30)
    if stop_rc != 0:
        return stop_rc, stop_out
    rm_rc, rm_out = await run_command("docker", "rm", profile.container_name, timeout=30)
    if rm_rc != 0:
        return rm_rc, rm_out
    return 0, f"{profile_name} stopped successfully!"


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
        except (ProcessLookupError, OSError):
            pass
        try:
            await proc.wait()
        except (asyncio.CancelledError, ProcessLookupError, OSError):
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


def format_gpu_bar(gpus: list[GpuInfo], bar_width: int = 8) -> str:
    """Format GPU info as a rich-text progress bar string."""
    if not gpus:
        return "[dim]GPU info unavailable[/dim]"
    parts = []
    for g in gpus:
        try:
            used = int(g.memory_used)
            total = int(g.memory_total)
        except (ValueError, TypeError):
            continue
        ratio = used / total if total > 0 else 0
        filled = round(ratio * bar_width)
        empty = bar_width - filled
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
        mem = f"{used / 1024:.1f}/{total / 1024:.1f}GB"
        parts.append(
            f"[bold]GPU{g.index}[/bold] {bar}  {mem}  {g.utilization}%  {g.temperature}°C"
        )
    return "  [dim]│[/dim]  ".join(parts)


# ---------------------------------------------------------------------------
# Version info for container start screen
# ---------------------------------------------------------------------------

async def estimate_model_memory(model_id: str) -> str:
    """Estimate GPU memory requirement for a HuggingFace model using hf-mem."""
    try:
        from hf_mem import arun
        # Pass HF token for gated models
        common_env = _parse_env_file(SCRIPT_DIR / ".env.common")
        token = common_env.get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")
        kwargs: dict = {"model_id": model_id, "experimental": True}
        if token and not token.startswith("your_"):
            kwargs["hf_token"] = token

        # Try default first, fallback to fp8 for models with unsupported quantization
        try:
            result = await arun(**kwargs)
        except RuntimeError as e:
            if "kv-cache-dtype" in str(e).lower() or "kv_cache_dtype" in str(e).lower():
                kwargs["kv_cache_dtype"] = "fp8"
                result = await arun(**kwargs)
            else:
                raise

        mem_bytes = getattr(result, 'memory', 0) or 0
        kv_bytes = getattr(result, 'kv_cache', 0) or 0
        total_bytes = getattr(result, 'total_memory', None) or (mem_bytes + kv_bytes)
        if not total_bytes:
            return "estimation failed (no data)"
        total_gb = total_bytes / (1024 ** 3)
        mem_gb = mem_bytes / (1024 ** 3)
        kv_gb = kv_bytes / (1024 ** 3)
        if kv_gb > 0:
            return f"~{total_gb:.1f}GB (model: {mem_gb:.1f}GB + KV: {kv_gb:.1f}GB)"
        return f"~{total_gb:.1f}GB"
    except Exception as e:
        err = str(e)
        if "403" in err:
            return "gated model - HF_TOKEN required"
        if "404" in err or "not found" in err.lower():
            return "model not found on HuggingFace"
        return f"estimation failed"


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


# ---------------------------------------------------------------------------
# vLLM parameter extraction from docker image
# ---------------------------------------------------------------------------

_VLLM_PARAMS_CACHE_DIR = CONFIG_DIR

_EXTRACT_SCRIPT = r'''
import re, os, json
vllm_path = __import__("vllm").__path__[0]
args = set()
scan_dirs = [
    os.path.join(vllm_path, "entrypoints"),
    os.path.join(vllm_path, "engine"),
    os.path.join(vllm_path, "config"),
]
for scan_dir in scan_dirs:
    if not os.path.isdir(scan_dir):
        continue
    for root, _, files in os.walk(scan_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, f)) as fh:
                    for line in fh:
                        for m in re.finditer(r"add_argument\(\s*[\"'](-{2}[a-zA-Z][a-zA-Z0-9_-]*)[\"']", line):
                            args.add(m.group(1)[2:])
            except Exception:
                pass
print(json.dumps(sorted(args)))
'''


async def extract_vllm_params(image_tag: str = "") -> set[str]:
    """Extract valid vllm serve parameters from a docker image.

    Scans the vllm Python source in the container to find all add_argument
    definitions. Results are cached per image tag in config/.vllm-params-{tag}.json.
    """
    import json

    if not image_tag:
        image_tag = await get_local_latest_tag()
        if image_tag == "none":
            return set()

    # Check cache
    cache_file = _VLLM_PARAMS_CACHE_DIR / f".vllm-params-{image_tag}.json"
    if cache_file.exists():
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            pass

    # Extract from container
    rc, out = await run_command(
        "docker", "run", "--rm", "--entrypoint", "python3",
        f"vllm/vllm-openai:{image_tag}",
        "-c", _EXTRACT_SCRIPT,
        timeout=30,
    )
    if rc != 0 or not out.strip():
        return set()

    # Parse last line (stdout may have warnings before JSON)
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                params = json.loads(line)
                _VLLM_PARAMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(params))
                return set(params)
            except json.JSONDecodeError:
                continue
    return set()
