"""Container orchestration and compose/runtime helpers."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from tui.common import dev_build
from tui.common.conflicts import gpu_conflict_messages as _shared_gpu_conflict_messages
from tui.common.env import validate_common_env


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
    return validate_common_env(
        COMMON_ENV,
        require_lora_base_path=(profile.enable_lora == "true"),
    )


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
        # Existing config still carries the placeholder. If the profile has a
        # concrete model_id, auto-promote it into the config (preserving any
        # other tuning the user set on the same file). Without this rewrite,
        # the placeholder kept winning despite the per-profile MODEL_ID being
        # set — and the error told the user to "set MODEL_ID" they had
        # already set.
        if profile.model_id:
            config.model = profile.model_id
            save_config(config)
            messages.append(
                f"Updated config/{profile.config_name}.yaml: model placeholder replaced "
                f"with profile MODEL_ID '{profile.model_id}'."
            )
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
    vllm_image: str = "",
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
    # Explicitly pin (or clear) VLLM_IMAGE. The rendered profile .env may
    # already carry VLLM_IMAGE when the profile pins an image_tag; setting it
    # here — to the desired value, or to "" for a UI/version-tag launch —
    # ensures it never leaks past the priority order. An empty value lets
    # compose's `${VLLM_IMAGE:-...}` fall back to the version-tag default.
    env["VLLM_IMAGE"] = vllm_image
    return env


def _env_file_args(profile: Profile) -> list[str]:
    """Return `--env-file` args, including each file only when it exists.

    `docker compose` aborts up-front if an `--env-file` path is missing; that
    would make `container_down` skip its clean network teardown and leak the
    compose network, forcing the docker stop/rm fallback.
    """
    args: list[str] = []
    if COMMON_ENV.exists():
        args.extend(["--env-file", str(COMMON_ENV)])
    if profile.path.exists():
        args.extend(["--env-file", str(profile.path)])
    return args


async def _container_exists(container_name: str) -> bool:
    rc, out = await run_command("docker", "ps", "-a", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        return False
    return container_name in out.strip().splitlines()


async def _gpu_conflict_messages(profile: Profile) -> list[str]:
    """Backend-local thin wrapper over the shared cross-backend helper. Both
    backends call the same `tui.common.conflicts.gpu_conflict_messages` so
    new GPU overlap rules propagate without per-backend drift."""
    return await _shared_gpu_conflict_messages(
        profile_name=profile.name,
        container_name=profile.container_name or profile.name,
        profile_gpu_id=profile.gpu_id,
        backend="vllm",
    )


async def _detect_gpu_arch() -> str:
    """Return PyTorch's TORCH_CUDA_ARCH_LIST format (space-separated dotted caps).

    Thin wrapper around the shared dev_build.detect_local_gpu_caps() helper.
    Kept here as a stable name for the rest of the vllm runtime to call.
    """
    return dev_build.format_arch_torch(await dev_build.detect_local_gpu_caps())


def _force_local_arch_for_deepep(dockerfile: os.PathLike[str] | str) -> tuple[bool, str]:
    """Patch upstream Dockerfile so DeepEP respects TORCH_CUDA_ARCH_LIST.

    Upstream sometimes hard-codes `9.0a 10.0a` for the DeepEP wheel build stage.
    We replace only that step with the already-exported TORCH_CUDA_ARCH_LIST value.
    """
    path = os.fspath(dockerfile)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return False, f"Error: failed to read upstream Dockerfile: {exc}"

    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if "/tmp/install_python_libraries.sh" not in line:
            continue
        if line.lstrip().startswith("COPY "):
            continue

        # The export is typically the immediately previous non-empty line.
        for prev in range(idx - 1, max(idx - 6, -1), -1):
            if "export TORCH_CUDA_ARCH_LIST=" not in lines[prev]:
                continue

            if (
                "$TORCH_CUDA_ARCH_LIST" in lines[prev]
                or "${TORCH_CUDA_ARCH_LIST}" in lines[prev]
            ):
                return True, "DeepEP already respects local TORCH_CUDA_ARCH_LIST."

            indent = lines[prev].split("export", 1)[0]
            lines[prev] = (
                f'{indent}export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" && \\\n'
            )
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("".join(lines))
            except OSError as exc:
                return False, f"Error: failed to patch upstream Dockerfile: {exc}"
            return True, "Patched DeepEP stage to use local TORCH_CUDA_ARCH_LIST."

        return False, "Error: could not locate DeepEP arch export line near install step."

    return True, "DeepEP install step not found; skipping DeepEP arch patch."


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


VLLM_DEV_SPEC = dev_build.DevBuildSpec(
    backend="vllm",
    image_prefix="vllm-dev",
    src_dir=VLLM_SRC_DIR,
    default_repo_url=DEFAULT_VLLM_REPO_URL,
    default_branch="main",
    dockerfile_relpath="docker/Dockerfile",
    target="vllm-openai",
    base_build_args=(("RUN_WHEEL_CHECK", "false"),),
    label_prefix="vllm",
)


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

    extra_log: list[str] = []
    if use_official:
        extra_log.append("Using official vLLM Dockerfile (ALL architectures)")
    else:
        extra_log.append(f"Detected GPU: {gpu_name} (compute: {gpu_arch})")
        extra_log.append("Building with local GPU arch targets.")

    extra_build_args: list[tuple[str, str]] = []
    if not use_official:
        extra_build_args.append(("torch_cuda_arch_list", gpu_arch))

    async def _patch():
        if use_official:
            return True, ""
        ok, msg = _force_local_arch_for_deepep(VLLM_SRC_DIR / "docker/Dockerfile")
        return ok, msg

    async for event in dev_build.stream_build(
        VLLM_DEV_SPEC,
        branch,
        repo_url=repo_url,
        custom_tag=custom_tag,
        extra_build_args=tuple(extra_build_args),
        extra_log_lines=tuple(extra_log),
        pre_build=_patch,
        extra_labels=(("vllm.build.type", "official" if use_official else "fast"),),
    ):
        yield event


async def _dev_image_matches(image_tag: str, repo_url: str, branch: str) -> bool:
    return await dev_build.image_matches(VLLM_DEV_SPEC, image_tag, repo_url, branch)


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
    # SO_REUSEADDR is required here: our own _post_start_validation hits
    # /v1/models on this same port after every successful `up`, which leaves
    # a TIME-WAIT entry on 127.0.0.1:<port> for ~60s. Without SO_REUSEADDR a
    # plain bind() refuses TIME-WAIT ports and we'd falsely report
    # "another local process on 127.0.0.1:<port>" on every `up→down→up`
    # cycle within that window. An actively LISTENING socket on the port
    # still fails the bind (that's the real conflict we care about), so
    # this only relaxes the spurious TIME-WAIT false positive — do not
    # remove without re-introducing the bug.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


async def _dir_size_bytes(path: str | None) -> int | None:
    """Disk usage of `path` in bytes via `du`, or None when unavailable.

    Backend-agnostic download-progress signal: both vLLM and llama.cpp stream
    model weights into the bind-mounted HF cache, so a growing cache directory
    means a download is still in flight. `-B1` reports actual disk blocks (not
    apparent size) so detection holds even if a downloader pre-allocates the
    target file. Mirrors llamacpp._dir_size_bytes — keep the two in sync.
    """
    if not path:
        return None
    rc, out = await run_command("du", "-s", "-B1", path, timeout=15)
    if rc != 0:
        return None
    head = out.strip().split(maxsplit=1)
    if not head or not head[0].isdigit():
        return None
    return int(head[0])


async def _post_start_validation(
    profile: Profile,
    *,
    timeout: float = 45.0,
    poll_interval: float = 3.0,
    max_wait: float = 1800.0,
    hf_cache_path: str | None = None,
):
    """Validate container state right after `compose up -d`.

    Async generator: yields ("log", str) progress lines while waiting, then a
    final ("result", ok: bool, messages: list[str]) tuple. Prevents
    false-positive success when compose starts a container that exits
    immediately (for example, GPU OOM during engine init).

    Readiness uses a *stall* deadline rather than a fixed wall-clock budget:
    the deadline is pushed back on every observed sign of progress — the HF
    cache growing (model still downloading) or the container log advancing
    (model still loading). A first-run download of a multi-GB model would
    otherwise blow a flat 45s budget and surface as a false "not ready" error
    even though the container comes up fine minutes later.

    `max_wait` is an absolute backstop: a container that only churns its log
    (e.g. an error-retry loop) would otherwise reset the stall deadline
    forever. The backstop is *not* applied while a download is actively in
    flight (cache still growing) — that is genuine progress and must never be
    capped. Mirrors llamacpp._post_start_validation — keep the two in sync.
    """
    loop = asyncio.get_running_loop()
    stall_timeout = max(timeout, poll_interval)
    deadline = loop.time() + stall_timeout
    start = loop.time()

    last_cache_size: int | None = None
    last_log_tail: str | None = None
    last_heartbeat = start
    downloaded_bytes = 0

    while True:
        rc, state = await run_command(
            "docker",
            "inspect",
            profile.container_name,
            "--format",
            "{{.State.Status}}\t{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            timeout=10,
        )
        if rc != 0:
            message = state.strip() or "docker inspect failed"
            yield (
                "result",
                False,
                [
                    f"Error: could not inspect container '{profile.container_name}' after startup.",
                    f"  {message}",
                ],
            )
            return
        status = "unknown"
        health = "unknown"
        if state.strip():
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
            yield ("result", False, messages)
            return

        if status not in {"running"}:
            yield (
                "result",
                False,
                [
                    f"Error: container '{profile.container_name}' is not running after startup ({status})."
                ],
            )
            return

        if await _models_endpoint_ready(profile.port):
            yield ("result", True, [])
            return

        # Progress detection: push the stall deadline back on any sign of life —
        # the HF cache growing (still downloading) or the log advancing (still
        # loading). Genuine stalls leave both flat and time out as before.
        progressed = False
        downloading = False

        cache_size = await _dir_size_bytes(hf_cache_path)
        if (
            last_cache_size is not None
            and cache_size is not None
            and cache_size > last_cache_size
        ):
            downloading = True
            progressed = True
            downloaded_bytes += cache_size - last_cache_size
        if cache_size is not None:
            last_cache_size = cache_size

        _, log_tail = await run_command(
            "docker", "logs", "--tail", "15", profile.container_name, timeout=10
        )
        if last_log_tail is not None and log_tail != last_log_tail:
            progressed = True
        last_log_tail = log_tail

        # Read the clock *after* the awaited probes above so the stall window
        # is measured from now, not from before up to ~25s of subprocess I/O.
        now = loop.time()
        if progressed:
            deadline = now + stall_timeout

        if progressed and now - last_heartbeat >= 12.0:
            if downloading:
                yield (
                    "log",
                    f"  ⏳ 모델 다운로드 중... "
                    f"({int(now - start)}s 경과, HF 캐시 +{downloaded_bytes / 1024**3:.1f} GB)",
                )
            else:
                yield ("log", f"  ⏳ 모델 로딩 중... ({int(now - start)}s 경과)")
            last_heartbeat = now

        if not downloading and now - start >= max_wait:
            # Absolute backstop: a download still in flight is never capped
            # (cache growth above keeps `downloading` True), but a container
            # that only churns its log without ever serving a model must not
            # hold the probe open forever.
            yield (
                "result",
                False,
                [
                    f"Error: container '{profile.container_name}' still not ready "
                    f"after {int(max_wait)}s.",
                    "  Model may be stuck — check `docker logs` and retry once ready.",
                ],
            )
            return

        if now >= deadline:
            # No progress for the full stall window: the container is running
            # but /v1/models still has no served entry, so we cannot truthfully
            # claim readiness. Returning False prevents the caller (CLI/TUI)
            # from chaining a benchmark or marking the start as a green success
            # when the model is in fact still loading; users can re-run `up`
            # (a no-op for a running container) once it finishes initializing.
            yield (
                "result",
                False,
                [
                    "Error: container started but /v1/models is not ready within timeout.",
                    "  Model is likely still loading — watch logs and retry once ready.",
                ],
            )
            return

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

    extra_packages = (profile.extra_pip_packages or "").strip()
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
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
            # Dev images are locally built — there is no remote to pull
            # them from, so disable compose's default missing-policy
            # fallback.
            "--pull",
            "never",
        ]
    elif not tag and profile.image_tag:
        # Per-profile pinned image. UI overrides (Dev Build / explicit Custom
        # Tag) take precedence and are handled above; absent those, honor the
        # profile's image_tag over the version-tag default. version_tag stays
        # empty so the shared block below skips version verification.
        version_tag = ""
        yield ("log", f"Using image: {profile.image_tag}")
        env = _compose_env(profile, use_dev=False, vllm_image=profile.image_tag)
        compose_cmd = [
            "docker",
            "compose",
            *compose_files,
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
            # Pinned image_tag implies the user already has the image
            # locally — `--pull never` blocks a surprise re-pull from
            # docker compose's default `missing` policy if the local
            # image somehow drops out of `docker images`.
            "--pull",
            "never",
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
            *_env_file_args(profile),
            "-p",
            profile.name,
            "up",
            "-d",
        ]
        # Pull policy — three distinct cases:
        #   * pull=True or `:nightly`
        #       → `--pull always`. Nightly is intentionally rolling; pull=True
        #         is reserved for any future "force re-pull" UI action.
        #   * Caller passed an explicit tag (Official Release resolves to a
        #     DockerHub semver here, or the user typed a Custom Tag)
        #       → `--pull missing`. If that tag is already local we reuse it
        #         silently; if it's not local, compose fetches it once. This
        #         matches the user's mental model of "pick the version" — not
        #         "force re-download". Critically, Official Release no longer
        #         re-pulls a manifest the user already has.
        #   * No explicit tag (Local Latest, resolved from local images)
        #       → `--pull never`. The user picked from images they already
        #         have, so a missing image is a real error, not an excuse to
        #         silently re-download.
        # `:latest` is rejected above and never reaches this point.
        if pull or version_tag == "nightly":
            compose_cmd.extend(["--pull", "always"])
        elif tag:
            compose_cmd.extend(["--pull", "missing"])
        else:
            compose_cmd.extend(["--pull", "never"])

    async for event in stream_command(compose_cmd, cwd=SCRIPT_DIR, env=env):
        if event[0] != "rc":
            yield event
            continue

        rc = int(event[1])
        if rc != 0:
            yield ("rc", rc)
            return

        yield ("log", f"{profile.name} started successfully!")

        ok = False
        val_messages: list[str] = []
        async for ev in _post_start_validation(
            profile, hf_cache_path=env.get("HF_CACHE_PATH") or None
        ):
            if ev[0] == "result":
                ok, val_messages = ev[1], ev[2]
            else:
                yield ev
        for message in val_messages:
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
        yield (
            "log",
            "Warning: could not verify vLLM version inside the container "
            f"({out.strip() or 'docker exec failed'}).",
        )
        return

    actual_str = out.strip().splitlines()[-1].strip()
    actual = _parse_stable_version_tag("v" + actual_str if not actual_str.startswith("v") else actual_str)
    if actual is None:
        yield (
            "log",
            f"Warning: could not parse vLLM version reported by container: {actual_str}",
        )
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
        *_env_file_args(profile),
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
    # Best-effort: tear down the compose network too. Without this, repeated
    # `compose down` failures leave `<profile>_default` networks accumulating
    # under `docker network ls`. Errors are ignored — the network may not
    # exist (compose never created it) or may still be in use by an external
    # container, and neither case is fatal for the stop operation.
    await run_command("docker", "network", "rm", f"{profile.name}_default", timeout=10)
    return 0, f"{profile_name} stopped successfully!"


async def stream_container_logs(container_name: str, *, tail: int = 100):
    """Async generator that yields log lines.

    `tail` controls how many recent lines `docker logs` prints before it
    starts following live output.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "logs",
        "-f",
        "--tail",
        str(tail),
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
