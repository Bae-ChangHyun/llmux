"""Python-native start/stop pipeline for llama.cpp profiles.

Mirrors the vllm backend pattern: `stream_container_up()` is an async
generator yielding ("log", str) / ("rc", int) events, `container_down()`
returns (rc, message). Replaces the legacy shell scripts in
scripts/llamacpp/ (switch.sh, stop.sh, _common.sh).

Also hosts the llama.cpp side of the unified dev-build pipeline
(get_dev_build_defaults, _stream_build_dev_image, _dev_image_matches).
The mechanics live in tui.common.dev_build; this module supplies the
DevBuildSpec and lets the user override CUDA_DOCKER_ARCH for faster
builds.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tui.common import dev_build, profile_store
from tui.common.docker import run_command as _docker_run

from .backend import (
    COMMON_ENV,
    PROJECT_ROOT,
    RUNTIME_DIR,
    SCRIPTS_DIR,
    Profile,
    _get_model_dir,
    _parse_env_file,
    load_profile,
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
        if caps:
            resolved_arch = dev_build.format_arch_cmake(caps)
            log_lines.append(f"Detected GPUs (compute_cap): {', '.join(caps)}")
            log_lines.append(f"Building with CUDA_DOCKER_ARCH={resolved_arch} (fast)")
        else:
            log_lines.append(
                "Could not auto-detect GPU (nvidia-smi unavailable) — falling back to multi-arch"
            )

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


# ---------------------------------------------------------------------------
# Subprocess helpers (kept local to avoid cross-backend coupling)
# ---------------------------------------------------------------------------


async def _run(*args: str, env: dict[str, str] | None = None, timeout: float = 60) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def _stream(args: list[str], *, env: dict[str, str] | None = None):
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
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
        yield ("rc", proc.returncode or 0)
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


# ---------------------------------------------------------------------------
# Compose argument builders
# ---------------------------------------------------------------------------


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
    if COMMON_ENV.exists():
        env.update(_parse_env_file(COMMON_ENV))
    if profile.path.exists():
        env.update(_parse_env_file(profile.path))
    return env


def _compose_base_args(profile: Profile) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        profile.name,
        *_compose_files(profile),
        "--project-directory",
        str(PROJECT_ROOT),
        "--env-file",
        str(COMMON_ENV),
        "--env-file",
        str(profile.path),
    ]


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


async def _render_profile_env(profile_name: str) -> tuple[int, str]:
    """Re-render .runtime/llamacpp/<name>.env via the unified profile_store CLI."""
    return await _run(
        "python3",
        "-m",
        "tui.common.profile_store",
        "render",
        "llamacpp",
        profile_name,
    )


async def _render_override(profile_name: str) -> tuple[int, str]:
    """Re-render .runtime/llamacpp/override-<name>.yaml."""
    return await _run(
        "python3",
        str(SCRIPTS_DIR / "render-override.py"),
        profile_name,
    )


async def _ensure_model_file(profile: Profile):
    """If GGUF missing, invoke pull-model.sh. Yields stream events."""
    if not profile.model_file:
        return
    model_path = _get_model_dir() / profile.model_file
    if model_path.exists():
        return
    yield ("log", f"▸ 모델 파일 없음 → 다운로드 시도: {profile.model_file}")
    pull_script = SCRIPTS_DIR / "pull-model.sh"
    if not pull_script.exists():
        yield ("log", f"✗ pull-model.sh 없음: {pull_script}")
        yield ("rc", 1)
        return
    async for event in _stream([str(pull_script), profile.name]):
        if event[0] == "rc" and event[1] != 0:
            yield event
            return
        yield event


# ---------------------------------------------------------------------------
# Post-start health validation (mirrors vllm _post_start_validation)
# ---------------------------------------------------------------------------


async def _post_start_validation(
    profile: Profile, *, timeout: float = 45.0, poll_interval: float = 2.0
) -> tuple[bool, list[str]]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout, poll_interval)

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
            return False, [
                f"Error: could not inspect container '{profile.container_name}' after startup.",
                f"  {state.strip() or 'docker inspect failed'}",
            ]
        status, _, health = state.strip().partition("\t")

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
            return False, msgs

        if status != "running":
            return False, [
                f"Error: container '{profile.container_name}' is not running after startup ({status})."
            ]

        rc_h, _ = await _docker_run(
            "curl", "-fsS", f"http://127.0.0.1:{profile.port}/health", timeout=5
        )
        if rc_h == 0:
            return True, []

        if loop.time() >= deadline:
            return True, [
                "Warning: container started but /health is not ready yet.",
                "Watch logs and retry once the model finishes loading.",
            ]

        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Public API: start
# ---------------------------------------------------------------------------


async def stream_container_up(profile_name: str):
    """Mirrors switch.sh: render profile/override, ensure model, compose up, validate.

    Yields ("log", str) lines and a final ("rc", int).
    """
    if not COMMON_ENV.exists():
        yield ("log", "✗ .env.common 없음. 'cp .env.common.example .env.common' 후 값 수정.")
        yield ("rc", 1)
        return

    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        yield ("log", f"✗ 프로필 없음: llamacpp/{profile_name} (profiles.yaml 확인)")
        yield ("rc", 1)
        return

    profile = load_profile(profile_name)

    rc, out = await _render_profile_env(profile_name)
    if rc != 0:
        yield ("log", out.strip() or f"✗ profile env render 실패: {profile_name}")
        yield ("rc", rc)
        return

    # If the profile pins a dev image, refuse to start until it exists locally.
    # Building it is opt-in via `llmux image build-dev` — auto-building during
    # a `up` call would silently kick off a 20+ minute job from inside the TUI.
    if profile.image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
        dev_tag = profile.image_tag.split(":", 1)[1]
        if not await dev_build.image_exists_locally(LLAMACPP_DEV_SPEC, dev_tag):
            yield ("log", f"✗ Dev image {profile.image_tag} not found locally.")
            yield ("log", "  Build it first:")
            yield ("log", f"  uv run llmux image build-dev --backend llamacpp --branch {dev_tag}")
            yield ("rc", 1)
            return
        yield ("log", f"▸ Dev image: {profile.image_tag}")

    async for event in _ensure_model_file(profile):
        if event[0] == "rc" and event[1] != 0:
            yield event
            return
        yield event

    yield ("log", "▸ command 렌더링")
    rc, out = await _render_override(profile_name)
    if rc != 0:
        yield ("log", out.strip() or "✗ override 렌더 실패")
        yield ("rc", rc)
        return

    yield ("log", f"▸ '{profile_name}' 프로필로 기동")
    env = _compose_env(profile)
    cmd = [*_compose_base_args(profile), "up", "-d"]

    async for event in _stream(cmd, env=env):
        if event[0] != "rc":
            yield event
            continue
        rc = int(event[1])
        if rc != 0:
            yield ("rc", rc)
            return

        ok, messages = await _post_start_validation(profile)
        for msg in messages:
            yield ("log", msg)
        if not ok:
            yield ("rc", 1)
            return

        # Persist last-active profile (parity with switch.sh).
        try:
            (PROJECT_ROOT / ".current-profile.llamacpp").write_text(profile_name)
        except OSError as exc:
            yield ("log", f"⚠ .current-profile.llamacpp 기록 실패: {exc}")

        yield ("log", f"✓ 프로필 '{profile_name}' 활성화됨")
        yield ("log", f"  Endpoint: http://localhost:{profile.port}/v1")
        yield ("log", f"  Health:   curl http://localhost:{profile.port}/health")
        yield ("rc", 0)
        return


# ---------------------------------------------------------------------------
# Public API: stop
# ---------------------------------------------------------------------------


async def container_down(profile_name: str) -> tuple[int, str]:
    """Mirrors stop.sh: prefer compose down (clean network teardown), fall back
    to docker rm -f for orphaned containers without an override file."""
    stored = profile_store.load_profile(profile_name, "llamacpp")
    if stored is None:
        return 1, f"✗ 프로필 없음: llamacpp/{profile_name}"

    profile = load_profile(profile_name)

    # Re-render env so compose has the right CONTAINER_NAME / LLAMA_PORT.
    await _render_profile_env(profile_name)

    override = _override_path(profile_name)
    if override.exists():
        env = _compose_env(profile)
        cmd = [*_compose_base_args(profile), "down"]
        rc, out = await _run(*cmd, env=env, timeout=120)
        if rc == 0:
            return 0, f"✓ '{profile_name}' 중지 완료"
        # fallthrough to docker rm -f if compose down fails

    rc_inspect, names = await _docker_run("docker", "ps", "-a", "--format", "{{.Names}}", timeout=10)
    if rc_inspect == 0 and profile.container_name in names.split():
        rc_rm, out_rm = await _docker_run("docker", "rm", "-f", profile.container_name, timeout=30)
        if rc_rm != 0:
            return rc_rm, out_rm.strip() or "✗ docker rm -f 실패"
        return 0, f"✓ '{profile_name}' 중지 완료 (rm -f)"

    return 0, f"({profile_name}) 컨테이너 없음 — 이미 중지됨"
