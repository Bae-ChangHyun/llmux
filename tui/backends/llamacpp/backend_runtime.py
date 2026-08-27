"""Python-native start/stop pipeline for llama.cpp profiles.

Mirrors the vllm backend pattern: `stream_container_up()` is an async
generator yielding ("log", str) / ("rc", int) events, `container_down()`
returns (rc, message).

Also hosts the llama.cpp side of the unified dev-build pipeline
(get_dev_build_defaults, _stream_build_dev_image, _dev_image_matches).
The mechanics live in tui.common.dev_build; this module supplies the
DevBuildSpec and lets the user override CUDA_DOCKER_ARCH for faster
builds.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tui.common import dev_build, prepare, profile_store
from tui.common.conflicts import gpu_conflict_messages as _shared_gpu_conflict_messages
from tui.common.docker import (
    run_command as _docker_run,
)
from tui.common.env import expand_env_values, validate_common_env

from .backend import (
    COMMON_ENV,
    LLAMACPP_OFFICIAL_IMAGE,
    CONFIG_DIR,
    PROJECT_ROOT,
    RUNTIME_DIR,
    SCRIPTS_DIR,
    Config,
    Profile,
    _parse_env_file,
    list_profile_names,
    load_config,
    load_profile,
    save_config,
)

COMPOSE_DIR = PROJECT_ROOT / "compose" / "llamacpp"
BASE_COMPOSE = COMPOSE_DIR / "docker-compose.yaml"
DEV_COMPOSE = COMPOSE_DIR / "docker-compose.dev.yaml"

LLAMACPP_SRC_DIR = PROJECT_ROOT / ".llamacpp-src"
DEFAULT_LLAMACPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

LLAMACPP_DEV_SPEC = dev_build.DevBuildSpec(
    backend="llamacpp",
    image_prefix="llamacpp-dev",
    src_dir=LLAMACPP_SRC_DIR,
    default_repo_url=DEFAULT_LLAMACPP_REPO_URL,
    default_branch="master",
    dockerfile_relpath=".devops/cuda.Dockerfile",
    target="server",
    label_prefix="llamacpp",
)


_BUILD_LOCK = asyncio.Lock()


@dataclass
class ContainerStatus:
    """Structured per-profile container status.

    Field-for-field mirror of tui.backends.vllm.backend_common.ContainerStatus
    so the CLI `ps` command can consume either backend's
    get_container_statuses() interchangeably.
    """

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


def get_dev_build_defaults() -> tuple[str, str]:
    """Return (repo_url, branch) honoring .env.common overrides."""
    env = _parse_env_file(COMMON_ENV) if COMMON_ENV.exists() else {}
    repo_url = env.get("LLAMACPP_REPO_URL", "").strip() or DEFAULT_LLAMACPP_REPO_URL
    branch = env.get("LLAMACPP_BRANCH", "").strip() or LLAMACPP_DEV_SPEC.default_branch
    return repo_url, branch


async def _stream_build_dev_image(
    branch: str,
    *,
    repo_url: str = "",
    custom_tag: str = "",
    cuda_arch: str = "",
    use_multi_arch: bool = False,
):
    """Lock-guarded wrapper around the unified builder."""
    if _BUILD_LOCK.locked():
        yield ("log", "Another llamacpp dev build is already running. Waiting...")
    async with _BUILD_LOCK:
        async for event in _do_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            cuda_arch=cuda_arch,
            use_multi_arch=use_multi_arch,
        ):
            yield event


async def _do_build_dev_image(
    branch: str,
    *,
    repo_url: str,
    custom_tag: str,
    cuda_arch: str,
    use_multi_arch: bool = False,
):
    """Build llamacpp-dev:<tag>.

    cuda_arch precedence:
      - explicit `cuda_arch` arg (e.g. "89" or "86;89") → use as-is
      - else auto-detect local GPU via nvidia-smi → CMake-formatted value
      - else (or use_multi_arch=True) fall back to "default" (Dockerfile's
        multi-arch path)
    """
    resolved_arch = cuda_arch.strip()
    log_lines: list[str] = []

    if not resolved_arch and not use_multi_arch:
        caps = await dev_build.detect_local_gpu_caps()
        if not caps:
            # Matches vLLM: a multi-arch build takes hours, so don't start one
            # as a consolation prize for a failed probe. --multi-arch opts in.
            yield ("log", "Error: could not detect GPU (nvidia-smi unavailable). "
                          "Pass --cuda-arch <arch> or --multi-arch to build anyway.")
            yield ("rc", 1)
            return
        resolved_arch = dev_build.format_arch_cmake(caps)
        log_lines.append(f"Detected GPUs (compute_cap): {', '.join(caps)}")
        log_lines.append(f"Building with CUDA_DOCKER_ARCH={resolved_arch} (fast)")

    extra_build_args: list[tuple[str, str]] = []
    if resolved_arch:
        extra_build_args.append(("CUDA_DOCKER_ARCH", resolved_arch))
    else:
        log_lines.append("Building with CUDA_DOCKER_ARCH=default (multi-arch, slower)")

    async for event in dev_build.stream_build(
        LLAMACPP_DEV_SPEC,
        branch,
        repo_url=repo_url,
        custom_tag=custom_tag,
        extra_build_args=tuple(extra_build_args),
        extra_log_lines=tuple(log_lines),
    ):
        yield event


async def _dev_image_matches(image_tag: str, repo_url: str, branch: str) -> bool:
    return await dev_build.image_matches(LLAMACPP_DEV_SPEC, image_tag, repo_url, branch)


async def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 60) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except FileNotFoundError:
        # `docker` (or whichever binary) is not on PATH. Surface a clear error
        # rather than letting the missing-executable exception bubble up and
        # crash container_down().
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (stdout or b"").decode(errors="replace")


async def _stream(args: list[str], *, env: dict[str, str] | None = None):
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except FileNotFoundError:
        yield ("log", f"✗ Executable not found: {args[0] if args else '<empty>'}")
        yield ("rc", -1)
        return
    if proc.stdout is None:
        yield ("rc", 1)
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode(errors="replace").rstrip("\n"))
        await proc.wait()
        rc = proc.returncode if proc.returncode is not None else -1
        yield ("rc", rc)
    except asyncio.CancelledError:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await proc.wait()
        except (asyncio.CancelledError, ProcessLookupError, OSError):
            pass
        raise


def _override_path(profile_name: str) -> Path:
    return RUNTIME_DIR / f"override-{profile_name}.yaml"


def _compose_files(profile: Profile) -> list[str]:
    """Compose -f arguments. Override file MUST exist (rendered by render-override.py).

    If the profile pins a `llamacpp-dev:*` image via `image_tag`, we layer
    `docker-compose.dev.yaml` between base and the per-profile override so the
    image: line gets swapped while command/volumes/env stay intact.
    """
    override = _override_path(profile.name)
    files = ["-f", str(BASE_COMPOSE)]
    if profile.image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
        files.extend(["-f", str(DEV_COMPOSE)])
    files.extend(["-f", str(override)])
    return files


def _compose_env(profile: Profile) -> dict[str, str]:
    """Merged env for docker compose: os.environ + .env.common + per-profile .env."""
    env = os.environ.copy()
    # Expand $VAR/~ only in the shared .env.common path values: process env
    # outranks --env-file in compose, so a raw `/home/$USER/...` would be
    # bind-mounted literally. The profile .env carries user env_vars, which must
    # stay literal — expanding them would make a deliberate `$VAR`/`~` value
    # impossible to pass through.
    if COMMON_ENV.exists():
        env.update(expand_env_values(_parse_env_file(COMMON_ENV)))
    if profile.path.exists():
        env.update(_parse_env_file(profile.path))
    env["PROFILE_PATH"] = str(profile.path)
    return env


def _compose_base_args(profile: Profile) -> list[str]:
    """docker compose base args. --env-file is only appended when the file
    actually exists — otherwise `docker compose` aborts up-front on a missing
    --env-file path, which would make `container_down` skip its clean
    network teardown and leak orphan resources."""
    args = [
        "docker",
        "compose",
        "-p",
        profile.name,
        *_compose_files(profile),
        "--project-directory",
        str(PROJECT_ROOT),
    ]
    if COMMON_ENV.exists():
        args.extend(["--env-file", str(COMMON_ENV)])
    if profile.path.exists():
        args.extend(["--env-file", str(profile.path)])
    return args


async def _render_override(profile_name: str) -> tuple[int, str]:
    """Re-render .runtime/llamacpp/override-<name>.yaml.

    Uses `sys.executable` so that under `uv run llmux` the venv's Python (and
    its installed PyYAML) is used — `"python3"` would resolve to the system
    interpreter, which usually lacks the dev dependencies and would fail the
    render step right before compose up.
    """
    return await _run(
        sys.executable,
        str(SCRIPTS_DIR / "render-override.py"),
        profile_name,
    )


async def _gpu_conflict_messages(profile: Profile) -> list[str]:
    """Thin wrapper over the shared cross-backend helper (see vllm side)."""
    return await _shared_gpu_conflict_messages(
        profile_name=profile.name,
        container_name=profile.container_name or profile.name,
        profile_gpu_id=profile.gpu_id,
        backend="llamacpp",
    )


async def check_port_conflict(profile: Profile) -> str | None:
    """Check whether the profile port is already occupied by a running
    container or local process. Mirrors vllm.check_port_conflict — static
    profile-to-profile overlap (both stopped) is ignored.

    Returns a short human-readable description when a conflict is found.
    """
    port = str(profile.port)
    rc, out = await _docker_run(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=10
    )
    if rc == 0:
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            container_name, ports = parts
            if container_name == profile.container_name:
                # Our own container is already up — see the identical guard in
                # vllm.check_port_conflict. The bind probe below cannot tell our
                # own docker-proxy from a foreign listener, so it blocked a
                # re-`up` (a compose no-op) with a bogus conflict.
                if re.search(rf"(^|[^\d]){re.escape(port)}->", ports):
                    return None
                continue
            if re.search(rf"(^|[^\d]){re.escape(port)}->", ports):
                for name in list_profile_names():
                    other = load_profile(name)
                    if other.container_name == container_name:
                        return f"profile '{name}'"
                return f"container '{container_name}'"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR: our own readiness probe leaves a TIME_WAIT entry on this
    # port for ~60s, and a plain bind() refuses those. A port a process is
    # actively LISTENING on still fails — that is the conflict we look for.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(port)))
    except OSError:
        return f"another local process on 127.0.0.1:{port}"
    finally:
        sock.close()
    return None


async def get_container_statuses() -> list[ContainerStatus]:
    """Status for every llama.cpp profile, including docker health state.

    Mirrors vllm.get_container_statuses() field-for-field (including the
    Dead/created/exited handling) so the CLI `ps` command can call either
    backend's version as a drop-in.
    """
    names = list_profile_names()
    rc, out = await _docker_run(
        "docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}", timeout=10
    )
    if rc != 0:
        raise RuntimeError(
            "docker ps failed or timed out — container state is unknown"
        )
    container_info: dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            container_info[parts[0]] = parts[1]

    statuses: list[ContainerStatus] = []
    for name in names:
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
        # Resolve a meaningful model label, matching vLLM's behavior of
        # surfacing the served alias rather than just the on-disk filename.
        # Priority: config.alias (what /v1/models actually returns) →
        # hf_repo (HF model id) → model_file (GGUF filename).
        model_label = ""
        if profile.config_name:
            cfg = load_config(profile.config_name)
            alias = cfg.get("alias")
            if isinstance(alias, str) and alias.strip():
                model_label = alias.strip()
        if not model_label:
            model_label = profile.hf_repo or profile.model_file or ""

        statuses.append(
            ContainerStatus(
                profile_name=name,
                container_name=profile.container_name,
                running=running,
                status_text=status_text,
                health=health,
                port=str(profile.port),
                gpu_id=profile.gpu_id,
                image=profile.image_tag,
                model=model_label,
                lora=False,
            )
        )
    return statuses


async def _models_endpoint_ready(port: str | int, timeout: int = 3) -> bool:
    """Return True when /v1/models responds with at least one served model id.

    Mirrors vllm._models_endpoint_ready — llama-server exposes the same
    OpenAI-compatible /v1/models endpoint, so listing the served model alias
    is a true readiness signal (vs /health, which only reports process
    liveness once the HTTP server is up).
    """
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
    target file. Mirrors vllm._dir_size_bytes — keep the two in sync.
    """
    if not path:
        return None
    rc, out = await _docker_run("du", "-s", "-B1", path, timeout=15)
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
    final ("result", ok: bool, messages: list[str]) tuple.

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
    capped. Mirrors vllm._post_start_validation — keep the two in sync.
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
        rc, state = await _docker_run(
            "docker",
            "inspect",
            profile.container_name,
            "--format",
            "{{.State.Status}}\t{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            timeout=10,
        )
        if rc != 0:
            yield (
                "result",
                False,
                [
                    f"Error: could not inspect container '{profile.container_name}' after startup.",
                    f"  {state.strip() or 'docker inspect failed'}",
                ],
            )
            return
        status, _, health = state.strip().partition("\t")
        status = status or "unknown"

        if status in {"restarting", "exited", "dead"} or health == "unhealthy":
            _, tail = await _docker_run(
                "docker", "logs", "--tail", "80", profile.container_name, timeout=10
            )
            reason = (
                f"container '{profile.container_name}' exited during startup ({status})"
                if status in {"restarting", "exited", "dead"}
                else f"container '{profile.container_name}' became unhealthy during startup"
            )
            msgs = [f"Error: {reason}."]
            if tail.strip():
                msgs.append("Recent logs:")
                msgs.extend([f"  {line}" for line in tail.strip().splitlines()[-12:]])
            yield ("result", False, msgs)
            return

        if status != "running":
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

        _, log_tail = await _docker_run(
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
            # No progress for the full stall window: container is running but
            # /v1/models has no served entry. Return False so a chained
            # benchmark or success banner does not fire on a half-ready model.
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


def _ensure_profile_config(
    stored: profile_store.StoredProfile, profile: Profile
) -> tuple[bool, list[str]]:
    """Link (and create) the profile's config so the override can render.

    Without this the Start screen's "a default config will be generated on
    start" promise was false on llama.cpp: an unlinked profile just failed in
    render-override.
    """
    messages: list[str] = []
    if not stored.config_name:
        stored.config_name = profile.name
        profile.config_name = profile.name
        try:
            profile_store.save_profile(stored)
        except Exception as exc:  # noqa: BLE001 — surface, don't half-start
            messages.append(f"✗ config 자동 링크 저장 실패: {exc}")
            return False, messages
        messages.append(f"▸ config 미링크 — '{profile.name}' 로 자동 링크")

    config_path = CONFIG_DIR / f"{stored.config_name}.yaml"
    if not config_path.exists():
        try:
            save_config(Config(
                name=stored.config_name,
                params={
                    "alias": stored.config_name,
                    "ctx-size": 8192,
                    "n-gpu-layers": 99,
                },
            ))
        except Exception as exc:  # noqa: BLE001
            messages.append(f"✗ 기본 config 생성 실패: {exc}")
            return False, messages
        messages.append(
            f"▸ 기본 config 생성: {config_path} (ctx-size/n-gpu-layers 기본값 — 확인 후 조정하세요)"
        )
        return True, messages

    existing = load_config(stored.config_name)
    if not existing.params:
        messages.append(f"✗ config '{stored.config_name}' 가 비어 있습니다 ({config_path}).")
        messages.append("  ctx-size / n-gpu-layers 등을 채운 뒤 다시 시작하세요.")
        return False, messages
    missing = [k for k in ("ctx-size", "n-gpu-layers") if k not in existing.params]
    if missing:
        messages.append(
            f"⚠ config '{stored.config_name}' 에 {', '.join(missing)} 없음 — llama-server 기본값이 적용됩니다."
        )
    return True, messages


async def stream_container_up(
    profile_name: str,
    *,
    use_dev: bool = False,
    use_default_image: bool = False,
    tag: str = "",
    repo_url: str = "",
    branch: str = "",
):
    """Mirrors vllm: render profile/override, optionally build dev image, compose up, validate.

    Parameters mirror tui.backends.vllm.backend_runtime.stream_container_up:
      - use_dev=True: TUI requested a `llamacpp-dev:<tag>` build/launch.
        repo_url/branch resolve via .env.common defaults when empty.
      - use_default_image=True: TUI's "Default Image" selection — explicitly
        drop any pinned profile.image_tag so base compose falls back to its
        built-in default. Without this signal the pinned tag would silently
        win (resolved_image_tag starts as profile.image_tag).
      - tag (non-dev path): user-supplied custom `<image>:<tag>`. When
        empty, fall back to the profile's image_tag or the default image
        from compose.
      - All-zero (default) keeps the previous behavior — profile.image_tag
        decides; default ghcr image otherwise.

    Priority order: UI override (dev / custom tag / default) > profile.image_tag
    > compose default.

    Yields ("log", str) lines and a final ("rc", int).
    """
    ok, env_messages = validate_common_env(COMMON_ENV)
    for message in env_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        yield ("log", f"✗ 프로필 없음: llamacpp/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)

    # Safety pre-flight (parity with vllm.stream_container_up): a hard port
    # conflict aborts; GPU id overlap is a non-fatal warning.
    conflict = await check_port_conflict(profile)
    if conflict:
        yield ("log", f"✗ 포트 {profile.port} 사용 중 — {conflict}")
        yield ("rc", 1)
        return
    for message in await _gpu_conflict_messages(profile):
        yield ("log", message)

    # Auto-link / auto-create the config (parity with vllm._ensure_profile_config).
    #
    # Must run BEFORE the image-override block: save_profile() persists
    # `stored`, and that block assigns a one-off tag onto it.
    ok, cfg_messages = _ensure_profile_config(stored, profile)
    for message in cfg_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    # Apply TUI-side image override (a Dev Build / Custom Tag / Default Image
    # selection trumps whatever's pinned on the profile). This is identical to
    # how the vllm screen passes use_dev/tag down to its stream_container_up.
    resolved_image_tag = profile.image_tag
    if use_default_image:
        # Explicit "Default Image" — clear any pinned image_tag so neither
        # LLAMACPP_IMAGE nor LLAMACPP_DEV_TAG is rendered into the per-profile
        # .env and base compose uses its built-in default image.
        resolved_image_tag = ""
    elif use_dev:
        # Resolve repo/branch defaults so callers can pass empty strings.
        default_repo, default_branch = get_dev_build_defaults()
        resolved_repo = (repo_url or default_repo).strip()
        resolved_branch = (branch or default_branch).strip()
        # Same sanitizer the builder uses — it tags `llamacpp-dev:<safe_branch>`,
        # so a raw `feat/foo` would be an invalid reference that never matches.
        dev_tag = dev_build.sanitize_docker_tag((tag or resolved_branch).strip())
        resolved_image_tag = f"{LLAMACPP_DEV_SPEC.image_prefix}:{dev_tag}"

        # Build the dev image on demand if missing OR if the locally cached
        # tag was built from a different repo/branch. The builder stamps
        # repo/branch labels on every image, so `<prefix>:master` from
        # ggml-org and `<prefix>:master` from a fork collide on tag name —
        # reusing blindly would silently start the wrong source.
        exists = await dev_build.image_exists_locally(LLAMACPP_DEV_SPEC, dev_tag)

        if tag and not exists:
            # An explicit --tag names an image the user expects to already
            # exist; silently building a *different* source under that name is
            # not what they asked for. vLLM errors out here — match it.
            yield ("log", f"Error: Image {resolved_image_tag} not found")
            rc, out = await _docker_run(
                "docker", "images", LLAMACPP_DEV_SPEC.image_prefix,
                "--format", "  {{.Tag}}", timeout=20,
            )
            if rc == 0 and out.strip():
                yield ("log", "Available images:")
                for line in out.strip().splitlines():
                    yield ("log", line)
            yield ("rc", 1)
            return

        # An explicit --tag names an image the user already has; the label check
        # exists only to catch a *branch-derived* tag whose cached image came
        # from another repo/branch. Applying it to an explicit tag would rebuild
        # over the user's own image. vLLM guards this the same way
        # (`if not tag and not needs_build`).
        matches = exists and (
            bool(tag)
            or await dev_build.image_matches(
                LLAMACPP_DEV_SPEC, dev_tag, resolved_repo, resolved_branch
            )
        )
        if not exists:
            yield ("log", f"▸ Dev image {resolved_image_tag} not found — building from {resolved_repo}@{resolved_branch}")
        elif not matches:
            yield ("log", f"▸ Dev image {resolved_image_tag} exists but was built from a different repo/branch — rebuilding from {resolved_repo}@{resolved_branch}")
        if not matches:
            async for event in _stream_build_dev_image(
                resolved_branch, repo_url=resolved_repo, custom_tag=tag
            ):
                yield event
                if event[0] == "rc" and event[1] != 0:
                    return
    elif tag:
        # Custom Tag: pass through as-is (e.g. `ghcr.io/foo/bar:v1`).
        resolved_image_tag = tag

    # Apply the resolved tag to BOTH objects:
    #  - `stored` (StoredProfile) feeds profile_store.render_env() → the
    #    per-profile .env that compose reads as LLAMACPP_DEV_TAG.
    #  - `profile` (backend.Profile) feeds _compose_files() → whether the
    #    dev compose override gets layered in.
    # Rendering in-process (not via a subprocess that re-reads profiles.yaml)
    # is what makes a one-off TUI "Dev Build" / "Custom Tag" selection
    # actually reach docker compose instead of silently falling back to the
    # saved profile's image.
    stored.image_tag = resolved_image_tag
    profile.image_tag = resolved_image_tag
    try:
        profile_store.render_env(stored)
    except Exception as exc:  # noqa: BLE001 — surface any render failure to the UI
        yield ("log", f"✗ profile env render 실패: {exc}")
        yield ("rc", 1)
        return

    # Sanity-check pinned dev images that we didn't build above (e.g. profile
    # was already pointed at a dev tag via `profile edit --image-tag`).
    if (
        not use_dev
        and resolved_image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:")
    ):
        dev_tag = resolved_image_tag.split(":", 1)[1]
        if not await dev_build.image_exists_locally(LLAMACPP_DEV_SPEC, dev_tag):
            yield ("log", f"✗ Dev image {resolved_image_tag} not found locally.")
            yield ("log", "  Build it first:")
            yield ("log", f"  uv run llmux image build-dev --backend llamacpp --tag {dev_tag}")
            yield ("rc", 1)
            return
        yield ("log", f"▸ Dev image: {resolved_image_tag}")
    elif not use_dev and resolved_image_tag:
        # Non-dev custom image reference (e.g. ghcr.io/foo/bar:v1). It was
        # rendered into the per-profile .env as LLAMACPP_IMAGE above and is
        # consumed verbatim by base compose's `image: ${LLAMACPP_IMAGE:-...}`.
        yield ("log", f"▸ Image: {resolved_image_tag}")

    yield ("log", "▸ command 렌더링")
    rc, out = await _render_override(profile_name)
    if rc != 0:
        yield ("log", out.strip() or "✗ override 렌더 실패")
        yield ("rc", rc)
        return
    # The renderer drops config keys llmux manages and warns on stderr; without
    # this the warning would never reach the user on the success path.
    for line in out.splitlines():
        if line.startswith("warning:"):
            yield ("log", f"⚠ {line[len('warning:'):].strip()}")

    yield ("log", f"▸ '{profile_name}' 프로필로 기동")
    env = _compose_env(profile)
    # Pin the resolved image directly in the *process* env — which outranks
    # --env-file in compose — so a concurrent llmux process re-rendering this
    # profile's .env (restoring the saved pin) can't swap the image out from
    # under a one-off Dev Build / Custom Tag / Default Image selection. Mirrors
    # vLLM's _compose_env VLLM_IMAGE handling; both vars are set explicitly so
    # the unused one is cleared rather than inherited stale.
    if resolved_image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
        env["LLAMACPP_DEV_TAG"] = resolved_image_tag.split(":", 1)[1]
        env["LLAMACPP_IMAGE"] = ""
    else:
        # "" lets base compose's `${LLAMACPP_IMAGE:-<default>}` fall through to
        # the default; a non-dev custom ref is used verbatim.
        env["LLAMACPP_IMAGE"] = resolved_image_tag
        env["LLAMACPP_DEV_TAG"] = ""
    # `--pull never` because the image is either:
    #   * a locally-built `llamacpp-dev:<tag>` (no remote to pull from), or
    #   * an explicitly-versioned ghcr/registry image the user already pulled
    #     via `llmux image pull` or a previous run.
    # Letting docker compose's default `missing` policy fire would surface as
    # a surprise "Pulling …" line that confuses users picking a known-local
    # image; if the image really is gone, the clear "image not found" error
    # is better than a silent re-download.
    cmd = [*_compose_base_args(profile), "up", "-d", "--pull", "never"]

    async for event in _stream(cmd, env=env):
        if event[0] != "rc":
            yield event
            continue
        rc = int(event[1])
        if rc != 0:
            yield ("rc", rc)
            return

        ok = False
        val_messages: list[str] = []
        async for ev in _post_start_validation(
            profile, hf_cache_path=env.get("HF_CACHE_PATH") or None
        ):
            if ev[0] == "result":
                ok, val_messages = ev[1], ev[2]
            else:
                yield ev
        for msg in val_messages:
            yield ("log", msg)
        if not ok:
            yield ("rc", 1)
            return

        try:
            (PROJECT_ROOT / ".current-profile.llamacpp").write_text(profile_name)
        except OSError as exc:
            yield ("log", f"⚠ .current-profile.llamacpp 기록 실패: {exc}")

        yield ("log", f"✓ 프로필 '{profile_name}' 활성화됨")
        yield ("log", f"  Endpoint: http://localhost:{profile.port}/v1")
        yield ("log", f"  Health:   curl http://localhost:{profile.port}/health")
        yield ("rc", 0)
        return


async def stream_container_prepare(profile_name: str):
    """Render runtime files, make sure the image is local, download the GGUF.

    Everything `up` does before `docker compose up`, plus the weight download —
    and nothing after it. Mirrors vllm.stream_container_prepare.
    """
    ok, env_messages = validate_common_env(COMMON_ENV)
    for message in env_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        yield ("log", f"✗ 프로필 없음: llamacpp/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)

    ok, cfg_messages = _ensure_profile_config(stored, profile)
    for message in cfg_messages:
        yield ("log", message)
    if not ok:
        yield ("rc", 1)
        return

    if not stored.hf_repo or not (stored.hf_file or stored.model_file):
        yield ("log", f"✗ '{profile_name}' 에 hf_repo / hf_file 이 없습니다 — 받을 GGUF 를 특정할 수 없습니다.")
        yield ("log", f"  llmux profile edit {profile_name} --hf-repo <org/repo> --hf-file <파일.gguf>")
        yield ("rc", 1)
        return

    try:
        env_path = profile_store.render_env(stored)
    except Exception as exc:  # noqa: BLE001 — surface any render failure
        yield ("log", f"✗ profile env render 실패: {exc}")
        yield ("rc", 1)
        return
    yield ("log", f"▸ runtime env 렌더: {env_path}")

    rc, out = await _render_override(profile_name)
    if rc != 0:
        yield ("log", out.strip() or "✗ override 렌더 실패")
        yield ("rc", rc)
        return
    for line in out.splitlines():
        if line.startswith("warning:"):
            yield ("log", f"⚠ {line[len('warning:'):].strip()}")
    yield ("log", f"▸ command 렌더: {_override_path(profile_name)}")

    image_ref = profile.image_tag or LLAMACPP_OFFICIAL_IMAGE
    if not await prepare.image_present(image_ref):
        if image_ref.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
            dev_tag = image_ref.split(":", 1)[1]
            yield ("log", f"✗ Dev image {image_ref} 가 로컬에 없습니다.")
            yield ("log", f"  먼저 빌드하세요: uv run llmux image build-dev --backend llamacpp --tag {dev_tag}")
            yield ("rc", 1)
            return
        async for event in prepare.stream_pull(image_ref):
            if event[0] == "rc":
                if int(event[1]) != 0:
                    yield ("log", f"✗ 이미지 pull 실패: {image_ref}")
                    yield ("rc", int(event[1]))
                    return
            else:
                yield event
    yield ("log", f"▸ Image: {image_ref}")

    cache_path = prepare.hf_cache_path()
    if not cache_path:
        yield ("log", "✗ .env.common 의 HF_CACHE_PATH 가 비어 있습니다 — 받을 위치가 없습니다.")
        yield ("rc", 1)
        return

    hf_file = stored.hf_file or stored.model_file

    yield ("log", f"▸ {stored.hf_repo} / {hf_file} 다운로드 → {cache_path}")
    rc = -1
    async for event in prepare.stream_llamacpp_download(
        hf_repo=stored.hf_repo,
        hf_file=hf_file,
        cache_path=cache_path,
        token=prepare.hf_token(),
        container_name=prepare.prepare_container_name(profile.container_name),
    ):
        if event[0] == "rc":
            rc = int(event[1])
        else:
            yield event
    if rc != 0:
        yield ("log", f"✗ 다운로드 실패 (rc={rc}).")
        yield ("rc", rc if rc != 0 else 1)
        return

    yield ("log", f"✓ '{profile_name}' 준비 완료 — 시작: llmux up {profile_name}")
    yield ("rc", 0)


async def _container_exists(container_name: str) -> bool | None:
    """True/False if the probe succeeded, None if `docker ps` failed/timed out."""
    rc, out = await _docker_run(
        "docker", "ps", "-a", "--format", "{{.Names}}", timeout=10
    )
    if rc != 0:
        return None
    return container_name in out.strip().splitlines()


async def container_down(profile_name: str) -> tuple[int, str]:
    """compose down (clean network teardown), falling back to docker rm -f for
    orphaned containers without an override file."""
    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        return 1, f"✗ 프로필 없음: llamacpp/{profile_name}"

    profile = load_profile(profile_name)

    render_warning = ""
    try:
        profile_store.render_env(stored)
    except Exception as exc:  # noqa: BLE001 — the rm -f path still works, but say so
        render_warning = f" (⚠ env 렌더 실패: {exc})"

    compose_err = ""
    override = _override_path(profile_name)
    if override.exists():
        env = _compose_env(profile)
        cmd = [*_compose_base_args(profile), "down"]
        rc, out = await _run(*cmd, env=env, timeout=120)
        if rc == 0:
            return 0, f"✓ '{profile_name}' 중지 완료{render_warning}"
        compose_err = next(
            (line for line in reversed(out.strip().splitlines()) if line.strip()),
            f"rc={rc}",
        )

    exists = await _container_exists(profile.container_name)
    if exists is None:
        return 1, (
            f"✗ '{profile_name}' 컨테이너 상태를 확인할 수 없음 "
            "(docker ps 실패 또는 타임아웃) — 중지 여부 불명"
        )

    if exists:
        rc_rm, out_rm = await _docker_run(
            "docker", "rm", "-f", profile.container_name, timeout=30
        )
        if rc_rm != 0:
            return rc_rm, out_rm.strip() or "✗ docker rm -f 실패"
        # Errors ignored — the network may not exist, or may still hold an
        # external container; neither is fatal for the stop itself.
        await _docker_run(
            "docker", "network", "rm", f"{profile.name}_default", timeout=10
        )
        detail = f" (compose down 실패: {compose_err})" if compose_err else ""
        return 0, f"✓ '{profile_name}' 중지 완료 (rm -f){detail}{render_warning}"

    if compose_err:
        return 1, (
            f"✗ '{profile_name}' compose down 실패 후 컨테이너를 찾을 수 없음: "
            f"{compose_err}{render_warning}"
        )
    return 0, f"({profile_name}) 컨테이너 없음 — 이미 중지됨{render_warning}"
