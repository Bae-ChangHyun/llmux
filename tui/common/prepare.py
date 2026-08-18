"""Model prefetch primitives shared by both backends' `prepare` flow.

`prepare` renders a profile's runtime files, makes sure its image is on disk and
downloads the weights — then stops. The server never starts, so nothing is
loaded onto a GPU.

Downloads run inside a throwaway container built from the same image the
profile would serve from: vLLM calls `huggingface_hub.snapshot_download`,
llama.cpp runs llama-server's own `-hf` path and is stopped as soon as the GGUF
lands in the host cache. Neither needs an `hf` CLI on the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path

from tui.common.env import host_expand, parse_env_file
from tui.common.i18n import t
from tui.common.profile_store import PROJECT_ROOT

COMMON_ENV = PROJECT_ROOT / ".env.common"

_PROGRESS_INTERVAL = 1.0
_POLL_INTERVAL = 2.0

# `original/` holds the Meta-format checkpoint of Llama-style repos — vLLM loads
# the HF-format weights next to it, so pulling both doubles the download.
_VLLM_IGNORE_PATTERNS = ["original/**"]

_VLLM_DOWNLOAD_SNIPPET = (
    "import os;"
    "from huggingface_hub import snapshot_download;"
    "ignore=os.environ['LLMUX_PREPARE_IGNORE'].split(',');"
    "print('skipping (vLLM does not load it):', ignore);"
    "print('snapshot:', snapshot_download("
    "os.environ['LLMUX_PREPARE_MODEL'], ignore_patterns=ignore))"
)


def _common() -> dict[str, str]:
    if not COMMON_ENV.exists():
        return {}
    return parse_env_file(COMMON_ENV)


def hf_cache_path() -> str:
    """Host HF cache directory from .env.common ("" when unset)."""
    raw = _common().get("HF_CACHE_PATH", "").strip()
    return host_expand(raw) if raw else ""


def hf_token() -> str:
    token = _common().get("HF_TOKEN", "").strip()
    return "" if token.startswith("your_") else token


def prepare_container_name(container_name: str) -> str:
    return f"{container_name}-prepare"


async def _run(*args: str, timeout: float = 30) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return -1, f"Executable not found: {args[0] if args else '<empty>'}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, (out or b"").decode(errors="replace")


async def stream_lines(args: list[str]):
    """Yield ("log", line) then ("rc", code), splitting on CR as well as LF.

    Download progress is redrawn with a bare CR, so a plain readline() would
    hold the whole transfer in one line; CR-terminated redraws are throttled to
    one per second so a multi-GB pull can't flood the log.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        yield ("log", f"✗ Executable not found: {args[0] if args else '<empty>'}")
        yield ("rc", -1)
        return

    buf = b""
    last_progress = 0.0
    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                match = re.search(rb"\r\n|\r|\n", buf)
                if match is None:
                    break
                line, delim = buf[: match.start()], match.group()
                buf = buf[match.end():]
                text = line.decode(errors="replace").rstrip()
                if not text:
                    continue
                if delim == b"\r":
                    now = time.monotonic()
                    if now - last_progress < _PROGRESS_INTERVAL:
                        continue
                    last_progress = now
                yield ("log", text)
        tail = buf.decode(errors="replace").rstrip()
        if tail:
            yield ("log", tail)
        await proc.wait()
        yield ("rc", proc.returncode if proc.returncode is not None else -1)
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(ProcessLookupError, OSError):
            await proc.wait()
        raise


async def image_present(image_ref: str) -> bool:
    rc, _ = await _run("docker", "image", "inspect", image_ref, timeout=20)
    return rc == 0


async def stream_pull(image_ref: str):
    """Pull `image_ref`, streaming docker's own progress output."""
    yield ("log", t(f"▸ Pulling image: {image_ref}", f"▸ 이미지 받는 중: {image_ref}"))
    async for event in stream_lines(["docker", "pull", image_ref]):
        yield event


async def stream_vllm_download(
    *,
    image_ref: str,
    model_id: str,
    cache_path: str,
    token: str,
    container_name: str,
):
    """Download a HF snapshot with the image's own huggingface_hub."""
    await _run("docker", "rm", "-f", container_name, timeout=30)
    args = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{cache_path}:/root/.cache/huggingface",
        "-e", f"LLMUX_PREPARE_MODEL={model_id}",
        "-e", f"LLMUX_PREPARE_IGNORE={','.join(_VLLM_IGNORE_PATTERNS)}",
    ]
    if token:
        args += ["-e", f"HF_TOKEN={token}", "-e", f"HUGGING_FACE_HUB_TOKEN={token}"]
    args += ["--entrypoint", "python3", image_ref, "-c", _VLLM_DOWNLOAD_SNIPPET]

    try:
        async for event in stream_lines(args):
            yield event
    except asyncio.CancelledError:
        await _run("docker", "rm", "-f", container_name, timeout=30)
        raise


def _repo_cache_dir(cache_path: str, hf_repo: str) -> Path:
    org, _, name = hf_repo.partition("/")
    return Path(cache_path) / "hub" / f"models--{org}--{name}"


def gguf_in_cache(cache_path: str, hf_repo: str, hf_file: str) -> Path | None:
    """The cached GGUF for `hf_repo`/`hf_file`, or None.

    llama.cpp downloads to `<blob>.downloadInProgress` and only links the
    snapshot entry once the transfer finished, so a hit here means complete.
    """
    repo_dir = _repo_cache_dir(cache_path, hf_repo)
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for match in sorted(snapshots.glob(f"*/{hf_file}")):
        if match.exists():
            return match
    return None


def _downloads_in_flight(cache_path: str, hf_repo: str) -> bool:
    repo_dir = _repo_cache_dir(cache_path, hf_repo)
    if not repo_dir.is_dir():
        return False
    return any(repo_dir.rglob("*.downloadInProgress"))


async def _container_running(container_name: str) -> bool | None:
    """True/False when the probe succeeded, None when `docker inspect` failed."""
    rc, out = await _run(
        "docker", "inspect", "-f", "{{.State.Running}}", container_name, timeout=20
    )
    if rc != 0:
        return None
    return out.strip().splitlines()[-1].strip() == "true"


async def stream_llamacpp_download(
    *,
    image_ref: str,
    hf_repo: str,
    hf_file: str,
    cache_path: str,
    token: str,
    container_name: str,
):
    """Run llama-server only long enough to fetch the GGUF, then stop it.

    llama.cpp has no download-only entrypoint: the transfer happens while
    arguments are parsed, before the model is loaded. So the throwaway
    container is stopped the moment the file appears in the host cache — the
    weights never reach a GPU.
    """
    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is not None:
        yield ("log", t(f"▸ Already in the HF cache: {cached}",
                        f"▸ 이미 HF 캐시에 있음: {cached}"))
        yield ("rc", 0)
        return

    await _run("docker", "rm", "-f", container_name, timeout=30)
    args = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{cache_path}:/root/.cache/huggingface",
    ]
    if token:
        args += ["-e", f"HF_TOKEN={token}"]
    args += [
        image_ref,
        "-hf", hf_repo,
        "-hff", hf_file,
        "--host", "127.0.0.1",
        "--port", "8080",
        "--no-webui",
    ]
    rc, out = await _run(*args, timeout=120)
    if rc != 0:
        yield ("log", out.strip() or t("✗ could not start the download container",
                                       "✗ 다운로드 컨테이너 기동 실패"))
        yield ("rc", rc if rc != 0 else 1)
        return

    stopped_by_us = False

    async def _watch() -> None:
        nonlocal stopped_by_us
        while True:
            if gguf_in_cache(cache_path, hf_repo, hf_file) is not None and not (
                _downloads_in_flight(cache_path, hf_repo)
            ):
                stopped_by_us = True
                await _run("docker", "stop", "-t", "5", container_name, timeout=60)
                return
            running = await _container_running(container_name)
            if running is None or not running:
                return
            await asyncio.sleep(_POLL_INTERVAL)

    watcher = asyncio.create_task(_watch())
    try:
        async for event in stream_lines(["docker", "logs", "-f", container_name]):
            if event[0] == "log":
                yield event
        await watcher
    except asyncio.CancelledError:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        await _run("docker", "rm", "-f", container_name, timeout=30)
        raise
    finally:
        if not watcher.done():
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    exit_rc, exit_out = await _run(
        "docker", "inspect", "-f", "{{.State.ExitCode}}", container_name, timeout=20
    )
    await _run("docker", "rm", "-f", container_name, timeout=30)

    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is not None:
        yield ("log", t(f"▸ Downloaded: {cached}", f"▸ 다운로드 완료: {cached}"))
        yield ("rc", 0)
        return

    if stopped_by_us:
        # The watcher only stops the container after the file is there, so a
        # missing file here means the cache was mutated underneath us.
        yield ("log", t("✗ the GGUF vanished from the cache after the download",
                        "✗ 다운로드 후 GGUF 가 캐시에서 사라졌습니다"))
        yield ("rc", 1)
        return

    detail = exit_out.strip() if exit_rc == 0 else t("unknown exit code", "종료 코드 불명")
    yield ("log", t(
        f"✗ llama-server stopped before the GGUF was cached (exit {detail})",
        f"✗ GGUF 다운로드 완료 전에 llama-server 가 종료됨 (exit {detail})",
    ))
    yield ("rc", 1)
