"""Python-native start/stop pipeline for llama.cpp profiles.

Mirrors the vllm backend pattern: `stream_container_up()` is an async
generator yielding ("log", str) / ("rc", int) events, `container_down()`
returns (rc, message). Replaces the legacy shell scripts in
scripts/llamacpp/ (switch.sh, stop.sh, _common.sh).

Stage 1 scope: behavior-identical replacement of the shell path. Dev-build
hooks land in Stage 2.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tui.common import profile_store
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
    """Compose -f arguments. Override file MUST exist (rendered by render-override.py)."""
    override = _override_path(profile.name)
    files = ["-f", str(BASE_COMPOSE), "-f", str(override)]
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
    """If GGUF missing, download via `hf` CLI directly. Yields stream events.

    Replaces the legacy scripts/llamacpp/pull-model.sh, which sourced
    _common.sh — itself moved to legacy/ in this refactor.
    """
    if not profile.model_file:
        return
    model_dir = _get_model_dir()
    model_path = model_dir / profile.model_file
    if model_path.exists():
        return

    hf_repo = profile.hf_repo
    hf_file = profile.hf_file or profile.model_file
    if not hf_repo:
        yield ("log", f"✗ hf_repo 미설정 (profiles.yaml 의 {profile.name} 확인)")
        yield ("rc", 1)
        return
    if not hf_file:
        yield ("log", f"✗ hf_file 미설정 (profiles.yaml 의 {profile.name} 확인)")
        yield ("rc", 1)
        return

    yield ("log", f"▸ 모델 파일 없음 → 다운로드 시도: {hf_repo} / {hf_file}")
    model_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if COMMON_ENV.exists():
        for k, v in _parse_env_file(COMMON_ENV).items():
            env[k] = v
    # The xet-core transport (huggingface_hub >= 0.27 default) fails DNS
    # resolution in some sandboxed/agent environments where the regular LFS
    # HTTPS path works fine. Force the classic transport unless the user
    # has explicitly opted in.
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    # Prefer `hf` (huggingface_hub >= 0.27); fall back to `huggingface-cli`.
    rc_which, _ = await _run("bash", "-lc", "command -v hf", timeout=5)
    cmd = (
        ["hf", "download", hf_repo, hf_file, "--local-dir", str(model_dir)]
        if rc_which == 0
        else ["huggingface-cli", "download", hf_repo, hf_file, "--local-dir", str(model_dir)]
    )
    async for event in _stream(cmd, env=env):
        if event[0] == "rc" and event[1] != 0:
            yield ("log", "✗ hf 다운로드 실패. `pip install -U huggingface_hub` 또는 `uv tool install huggingface_hub` 확인.")
            yield event
            return
        yield event
    yield ("log", f"✓ 완료: {model_path}")


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
