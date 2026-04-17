"""Container orchestration and compose/runtime helpers."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket

from .backend_common import (
    COMMON_ENV,
    CONFIG_DIR,
    DEFAULT_VLLM_REPO_URL,
    SCRIPT_DIR,
    VLLM_SRC_DIR,
    Config,
    ContainerStatus,
    Profile,
)
from .backend_inspect import get_dockerhub_release_version, get_local_latest_tag
from .backend_process import run_command, run_command_with_options, stream_command
from .backend_storage import (
    _parse_env_file,
    list_profile_names,
    load_config,
    load_profile,
    save_config,
    save_profile,
)


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
        return False, [
            f"Error: HF_CACHE_PATH must be an absolute path. Current value: {hf_cache_path}"
        ]

    if profile.enable_lora == "true":
        lora_base_path = common.get("LORA_BASE_PATH", "")
        if not lora_base_path:
            return False, ["Error: ENABLE_LORA=true but LORA_BASE_PATH is not set in .env.common"]
        if not os.path.isabs(lora_base_path):
            return False, [
                f"Error: LORA_BASE_PATH must be an absolute path. Current value: {lora_base_path}"
            ]

    return True, []


def _ensure_profile_config(profile: Profile) -> tuple[bool, list[str]]:
    messages: list[str] = []
    if not profile.config_name:
        profile.config_name = profile.name
        save_profile(profile)
        messages.append(
            f"No config linked for '{profile.name}'. Auto-linked default config '{profile.config_name}'."
        )

    config = load_config(profile.config_name)
    if config.path.exists():
        model = config.model.strip()
        if model and model != "your-org/your-model":
            return True, messages
        messages.extend(
            [
                f"Error: config/{profile.config_name}.yaml does not have a valid model configured yet.",
                (
                    f"Set the model field in config/{profile.config_name}.yaml or set MODEL_ID "
                    f"in profiles/{profile.name}.env, then start again."
                ),
            ]
        )
        return False, messages

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if profile.model_id:
        save_config(
            Config(
                name=profile.config_name,
                model=profile.model_id,
                gpu_memory_utilization="0.55",
            )
        )
        messages.append(f"Created default config: config/{profile.config_name}.yaml")
        return True, messages

    config.path.write_text(
        f"# Auto-generated default config for profile: {profile.name}\n"
        "# Set a valid Hugging Face model ID below, then start again.\n"
        "model: your-org/your-model\n"
        "gpu-memory-utilization: 0.55\n"
    )
    messages.extend(
        [
            f"Created config/{profile.config_name}.yaml but MODEL_ID is not set for profile '{profile.name}'.",
            (
                f"Edit the config model field or set MODEL_ID in profiles/{profile.name}.env, "
                "then start again."
            ),
        ]
    )
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
            messages.append(
                f"Warning: GPU {gpu_id} is also used by running container '{other.container_name}'"
            )
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
            "git",
            "remote",
            "get-url",
            "origin",
            cwd=VLLM_SRC_DIR,
            timeout=30,
        )
        if rc != 0:
            yield ("log", current_remote.strip() or "Error: failed to inspect existing vLLM source")
            yield ("rc", 1)
            return
        if current_remote.strip() != repo_url:
            yield ("log", "Repository URL changed. Re-cloning...")
            await asyncio.to_thread(shutil.rmtree, VLLM_SRC_DIR)

    if not VLLM_SRC_DIR.exists():
        async for event in stream_command(["git", "clone", repo_url, str(VLLM_SRC_DIR)], cwd=SCRIPT_DIR):
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
            "git",
            "checkout",
            "-b",
            branch,
            f"origin/{branch}",
            cwd=VLLM_SRC_DIR,
            timeout=60,
        )
        if rc != 0:
            yield ("log", out.strip() or f"Error: failed to checkout branch {branch}")
            yield ("rc", rc)
            return

    rc, out = await run_command_with_options(
        "git", "pull", "origin", branch, cwd=VLLM_SRC_DIR, timeout=120
    )
    if rc != 0 and out.strip():
        yield ("log", out.strip())

    rc, commit_hash = await run_command_with_options(
        "git", "rev-parse", "--short", "HEAD", cwd=VLLM_SRC_DIR, timeout=30
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
        rc, gpu_name = await run_command(
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader", timeout=10
        )
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
        "docker",
        "build",
        "-f",
        str(VLLM_SRC_DIR / "docker/Dockerfile"),
        "--build-arg",
        "RUN_WHEEL_CHECK=false",
        "--target",
        "vllm-openai",
        "--label",
        f"vllm.repo.url={repo_url}",
        "--label",
        f"vllm.repo.branch={branch}",
        "--label",
        f"vllm.commit.hash={commit_hash}",
        "--label",
        f"vllm.build.date={build_date}",
        "--label",
        f"vllm.build.type={'official' if use_official else 'fast'}",
        "-t",
        f"vllm-dev:{main_tag}",
        "-t",
        f"vllm-dev:{branch}",
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
        "docker",
        "inspect",
        image_ref,
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
    rc, out = await run_command(
        "docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}", timeout=10
    )
    container_info: dict[str, str] = {}
    if rc == 0:
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                container_info[parts[0]] = parts[1]

    statuses = []
    for name in profiles:
        profile = load_profile(name)
        docker_status = container_info.get(profile.container_name, "")
        running = False
        health = ""
        status_text = "stopped"
        if docker_status:
            if "(healthy)" in docker_status:
                running = True
                health = "healthy"
                status_text = "healthy"
            elif "(unhealthy)" in docker_status:
                running = True
                health = "unhealthy"
                status_text = "unhealthy"
            elif "(health: starting)" in docker_status:
                running = True
                health = "starting"
                status_text = "starting"
            elif docker_status.startswith("Up "):
                running = True
                status_text = "running"
            elif docker_status.startswith("Exited ") or docker_status.startswith("Dead "):
                status_text = "exited"
            else:
                status_text = "created"
        model = ""
        if profile.config_name:
            config = load_config(profile.config_name)
            model = config.model
        statuses.append(
            ContainerStatus(
                profile_name=name,
                container_name=profile.container_name,
                running=running,
                status_text=status_text,
                health=health,
                port=profile.port,
                gpu_id=profile.gpu_id,
                model=model,
                lora=profile.enable_lora == "true",
            )
        )
    return statuses


async def check_port_conflict(profile: Profile) -> str | None:
    """Check whether the profile port is already occupied by a running container
    or local process. Static profile-to-profile overlap (both stopped) is ignored.

    Returns a short human-readable description when a conflict is found.
    """
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=10
    )
    if rc == 0:
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            container_name, ports = parts
            if container_name == profile.container_name:
                continue
            if re.search(rf"(^|[^\d]){re.escape(profile.port)}->", ports):
                for name in list_profile_names():
                    other = load_profile(name)
                    if other.container_name == container_name:
                        return f"profile '{name}'"
                return f"container '{container_name}'"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", int(profile.port)))
    except OSError:
        return f"another local process on 127.0.0.1:{profile.port}"
    finally:
        sock.close()
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
        yield ("log", f"Error: Port {profile.port} is already in use by {conflict}")
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
                rc, out = await run_command(
                    "docker", "images", "vllm-dev", "--format", "  {{.Tag}}", timeout=20
                )
                if rc == 0 and out.strip():
                    yield ("log", "Available images:")
                    for line in out.strip().splitlines():
                        yield ("log", line)
                yield ("rc", 1)
                return
            if rc == 0:
                yield (
                    "log",
                    "Existing dev image metadata does not match the requested repository/branch. Rebuilding...",
                )
            else:
                yield ("log", "Dev image not found. Building first...")
            async for event in _stream_build_dev_image(resolved_branch, repo_url=resolved_repo_url):
                yield event
                if event[0] == "rc" and event[1] != 0:
                    return

        yield ("log", f"Using image: vllm-dev:{image_tag}")
        env = _compose_env(profile, use_dev=True, image_tag=image_tag)
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            "--env-file",
            str(COMMON_ENV),
            "--env-file",
            str(profile.path),
            "-p",
            profile.name,
            "up",
            "-d",
        ]
    else:
        version_tag = tag or await get_local_latest_tag()
        if version_tag == "none":
            yield ("log", "Error: No local vllm/vllm-openai images found.")
            release_version = await get_dockerhub_release_version()
            if release_version != "unknown":
                yield (
                    "log",
                    f"Pull a stable version first: docker pull vllm/vllm-openai:{release_version}",
                )
            else:
                yield ("log", "Pull a specific version first, or choose Official Release in the UI.")
            yield ("rc", 1)
            return

        yield ("log", f"Using image: vllm/vllm-openai:{version_tag}")
        env = _compose_env(profile, use_dev=False, version_tag=version_tag)
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            "--env-file",
            str(COMMON_ENV),
            "--env-file",
            str(profile.path),
            "-p",
            profile.name,
            "up",
            "-d",
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
        "docker",
        "inspect",
        profile.container_name,
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
        "docker",
        "compose",
        *_compose_files(profile, use_dev),
        "--env-file",
        str(COMMON_ENV),
        "--env-file",
        str(profile.path),
        "-p",
        profile.name,
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
        "docker",
        "logs",
        "-f",
        "--tail",
        "100",
        container_name,
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
