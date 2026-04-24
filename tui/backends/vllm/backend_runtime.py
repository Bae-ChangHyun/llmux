"""Container orchestration and compose/runtime helpers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import urllib.request

from .backend_common import (
    COMMON_ENV,
    COMPOSE_DIR,
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


_BUILD_LOCK = asyncio.Lock()


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
                    f"in profiles.yaml for '{profile.name}', then start again."
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
                f"Edit the config model field or set MODEL_ID in profiles.yaml for '{profile.name}', "
                "then start again."
            ),
        ]
    )
    return False, messages


def _compose_files(profile: Profile, use_dev: bool) -> list[str]:
    base = "docker-compose.dev.yaml" if use_dev else "docker-compose.yaml"
    files = ["-f", str(COMPOSE_DIR / base)]
    if profile.enable_lora == "true":
        files.extend(["-f", str(COMPOSE_DIR / "docker-compose.lora.yaml")])
    files.extend(["-f", str(COMPOSE_DIR / "docker-compose.overrides.yaml")])
    files.extend(["--project-directory", str(SCRIPT_DIR)])
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
    """Return unique GPU compute capabilities as a space-separated list.

    Preserves dot form ('8.6 8.9') so PyTorch's TORCH_CUDA_ARCH_LIST and
    CMake recognize each arch. Covers mixed-SM setups rather than only GPU 0.
    """
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
        timeout=10,
    )
    if rc != 0 or not out.strip():
        return ""
    caps = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return " ".join(caps)


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
    if rc != 0:
        yield ("log", out.strip() or f"Error: git pull failed for branch {branch}")
        yield (
            "log",
            "Hint: stash or reset local changes in .vllm-src/, then retry.",
        )
        yield ("rc", rc)
        return

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
    if _BUILD_LOCK.locked():
        yield (
            "log",
            "Another dev build is already running. Waiting for it to finish...",
        )
    async with _BUILD_LOCK:
        async for event in _do_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            use_official=use_official,
        ):
            yield event


async def _do_build_dev_image(
    branch: str,
    *,
    repo_url: str,
    custom_tag: str,
    use_official: bool,
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
        yield ("log", f"Detected GPU: {gpu_name} (compute: {gpu_arch})")
        yield (
            "log",
            "Building with local GPU arch targets. "
            "Some upstream dependencies may still compile compatibility kernels.",
        )
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

    build_env = os.environ.copy()
    build_env.setdefault("DOCKER_BUILDKIT", "1")

    async for event in stream_command(cmd, cwd=SCRIPT_DIR, env=build_env):
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


async def _models_endpoint_ready(port: str, timeout: int = 3) -> bool:
    """Return True when /v1/models responds with at least one served model id."""
    loop = asyncio.get_running_loop()

    def _probe() -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=timeout
            ) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            data = payload.get("data", [])
            return any(isinstance(item, dict) and item.get("id") for item in data)
        except Exception:
            return False

    return await loop.run_in_executor(None, _probe)


async def _post_start_validation(
    profile: Profile,
    *,
    timeout: float = 45.0,
    poll_interval: float = 2.0,
) -> tuple[bool, list[str]]:
    """Validate container state right after `compose up -d`.

    Prevents false-positive success when compose starts a container that exits
    immediately (for example, GPU OOM during engine init).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout, poll_interval)

    while True:
        rc, state = await run_command(
            "docker",
            "inspect",
            profile.container_name,
            "--format",
            "{{.State.Status}}\t{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            timeout=10,
        )
        status = "unknown"
        health = "unknown"
        if rc == 0 and state.strip():
            parts = state.strip().split("\t", 1)
            status = parts[0].strip()
            health = parts[1].strip() if len(parts) == 2 else "unknown"

        if status in {"restarting", "exited", "dead"} or health == "unhealthy":
            _, tail = await run_command(
                "docker",
                "logs",
                "--tail",
                "80",
                profile.container_name,
                timeout=10,
            )
            reason = (
                f"container '{profile.container_name}' exited during startup ({status})"
                if status in {"restarting", "exited", "dead"}
                else f"container '{profile.container_name}' became unhealthy during startup"
            )
            messages = [f"Error: {reason}."]
            if tail.strip():
                messages.append("Recent logs:")
                messages.extend([f"  {line}" for line in tail.strip().splitlines()[-12:]])
            return False, messages

        if await _models_endpoint_ready(profile.port):
            return True, []

        if loop.time() >= deadline:
            return True, [
                "Warning: container started but /v1/models is not ready yet.",
                "Watch logs and retry benchmark once the model finishes loading.",
            ]

        await asyncio.sleep(poll_interval)


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
        if version_tag == "latest":
            # Refuse the `:latest` alias outright. It doesn't describe the image's
            # real contents and clicking "Local Latest" / "Official Release" should
            # always resolve to a specific semver tag before reaching here.
            yield (
                "log",
                "Error: `:latest` is an ambiguous alias and is not allowed. "
                "Pick Local Latest (resolves to your highest local versioned tag) "
                "or Official Release (pulls DockerHub's latest stable by explicit "
                "version), or enter a specific tag in Custom.",
            )
            yield ("rc", 1)
            return
        if version_tag == "none":
            yield (
                "log",
                "Error: no versioned vllm/vllm-openai image found locally. "
                "llmux refuses to start from `:latest` aliases because they don't "
                "describe their actual contents.",
            )
            release_version = await get_dockerhub_release_version()
            if release_version != "unknown":
                yield (
                    "log",
                    f"Pull a specific version first, e.g.: docker pull vllm/vllm-openai:{release_version}",
                )
            else:
                yield (
                    "log",
                    "Pull a specific version first (docker pull vllm/vllm-openai:<version>), "
                    "or choose Official Release in the UI.",
                )
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
        # Force-pull when:
        #   (a) the UI explicitly requested it (Official Release → pulls the
        #       resolved semver tag from DockerHub; the tag itself is a version,
        #       never the bare `:latest` alias), or
        #   (b) the tag is `:nightly` — it's an intentionally-rolling target
        #       with no versioned alternative, so we always want the freshest.
        # `:latest` is rejected above and never reaches this point.
        if pull or version_tag == "nightly":
            compose_cmd.extend(["--pull", "always"])

    async for event in stream_command(compose_cmd, cwd=SCRIPT_DIR, env=env):
        if event[0] != "rc":
            yield event
            continue

        rc = int(event[1])
        if rc != 0:
            yield ("rc", rc)
            return

        yield ("log", f"{profile.name} started successfully!")

        ok, messages = await _post_start_validation(profile)
        for message in messages:
            yield ("log", message)
        if not ok:
            yield ("rc", 1)
            return

        if not use_dev:
            async for evt in _verify_vllm_version(profile.container_name, version_tag):
                yield evt
        yield ("rc", 0)
        return


async def _verify_vllm_version(container_name: str, expected_tag: str):
    """Compare the tag we told docker to run against the vllm version actually
    running inside the container. Warn (but don't fail) on mismatch — tag names
    can lie about contents if someone ran `docker tag` by hand or pulled the
    same `latest` alias at different times.
    """
    from .backend_inspect import _parse_stable_version_tag

    expected = _parse_stable_version_tag(expected_tag)
    if expected is None:
        # Only verify for versioned tags — `latest`/`nightly` wouldn't be reached
        # under the new Local-Latest logic anyway.
        return

    # Query the running container. Give vllm a moment to print its banner, but
    # don't block the UI — we fall back silently on timeout.
    rc, out = await run_command(
        "docker",
        "exec",
        container_name,
        "python3",
        "-c",
        "import vllm; print(vllm.__version__)",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return

    actual_str = out.strip().splitlines()[-1].strip()
    actual = _parse_stable_version_tag("v" + actual_str if not actual_str.startswith("v") else actual_str)
    if actual is None:
        return

    if actual != expected:
        yield (
            "log",
            f"⚠ Warning: image tag says v{expected[0]}.{expected[1]}.{expected[2]}, "
            f"but the container reports vllm {actual_str}. The tag may have been "
            f"retagged or pulled at a different time — consider re-pulling the "
            f"specific version you want.",
        )


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
