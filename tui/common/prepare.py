"""Model prefetch primitives shared by both backends' `prepare` flow.

`prepare` renders a profile's runtime files, makes sure its image is on disk and
downloads the weights — then stops. The server never starts, so nothing is
loaded onto a GPU.

Downloads run inside a throwaway container that calls
`huggingface_hub.snapshot_download`, so both backends pull over parallel
workers and neither needs an `hf` CLI on the host. llama.cpp's own `-hf` fetch
is a single stream, so its GGUFs come down through the downloader image named
by PREPARE_DOWNLOADER_IMAGE instead.
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

_GGUF_DOWNLOAD_SNIPPET = (
    "import os;"
    "from huggingface_hub import snapshot_download;"
    "allow=os.environ['LLMUX_PREPARE_ALLOW'].split(',');"
    "workers=os.environ.get('LLMUX_PREPARE_WORKERS');"
    "extra={'max_workers': int(workers)} if workers else {};"
    "print('fetching:', allow, extra);"
    "print('snapshot:', snapshot_download("
    "os.environ['LLMUX_PREPARE_MODEL'], allow_patterns=allow, **extra))"
)

_SPLIT_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$"
)


def gguf_shard_names(hf_file: str) -> list[str]:
    """Every shard of a split GGUF, or just the file when it is not split.

    Shards are far from equal — unsloth's first one can be 10MB against a 50GB
    sibling — so a profile's `hf_file` is only ever the entry point.
    """
    directory, _, name = hf_file.rpartition("/")
    match = _SPLIT_SHARD_RE.match(name)
    if match is None:
        return [hf_file]
    prefix = f"{directory}/" if directory else ""
    stem, total = match.group("stem"), int(match.group("total"))
    return [f"{prefix}{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def downloader_image() -> str:
    """Image whose python3 runs snapshot_download ("" when unset)."""
    return _common().get("PREPARE_DOWNLOADER_IMAGE", "").strip()


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
    """The cached entry point for `hf_repo`/`hf_file`, or None.

    Every shard has to be there, not just the one the profile names: a partial
    set would otherwise read as a hit and skip the rest of the download.
    snapshot_download links a snapshot entry only once its blob is complete, so
    a full set of links means the model is whole.
    """
    repo_dir = _repo_cache_dir(cache_path, hf_repo)
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    shards = gguf_shard_names(hf_file)
    for snapshot in sorted(snapshots.iterdir()):
        paths = [snapshot / shard for shard in shards]
        if all(path.exists() for path in paths):
            return paths[0]
    return None


async def stream_llamacpp_download(
    *,
    hf_repo: str,
    hf_file: str,
    cache_path: str,
    token: str,
    container_name: str,
    max_workers: int | None = None,
):
    """Fetch a GGUF — every shard of a split one — into the host HF cache.

    The transfer runs in the PREPARE_DOWNLOADER_IMAGE container, not the
    llama.cpp server image: the weights are pulled by huggingface_hub and never
    reach a GPU. HF caps a single connection at a few MB/s, so `max_workers` is
    the knob for how much of the line the download is allowed to take.
    """
    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is not None:
        yield ("log", t(f"▸ Already in the HF cache: {cached}",
                        f"▸ 이미 HF 캐시에 있음: {cached}"))
        yield ("rc", 0)
        return

    image_ref = downloader_image()
    if not image_ref:
        yield ("log", t(
            "✗ PREPARE_DOWNLOADER_IMAGE is unset in .env.common — no image to download with",
            "✗ .env.common 의 PREPARE_DOWNLOADER_IMAGE 가 비어 있습니다 — GGUF 를 받을 이미지가 없습니다.",
        ))
        yield ("rc", 1)
        return
    if not await image_present(image_ref):
        async for event in stream_pull(image_ref):
            if event[0] == "rc":
                if int(event[1]) != 0:
                    yield ("log", t(f"✗ could not pull {image_ref}",
                                    f"✗ downloader 이미지 pull 실패: {image_ref}"))
                    yield ("rc", int(event[1]))
                    return
            else:
                yield event

    await _run("docker", "rm", "-f", container_name, timeout=30)
    args = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{cache_path}:/root/.cache/huggingface",
        "-e", f"LLMUX_PREPARE_MODEL={hf_repo}",
        "-e", f"LLMUX_PREPARE_ALLOW={','.join(gguf_shard_names(hf_file))}",
    ]
    if max_workers is not None:
        args += ["-e", f"LLMUX_PREPARE_WORKERS={max_workers}"]
    if token:
        args += ["-e", f"HF_TOKEN={token}", "-e", f"HUGGING_FACE_HUB_TOKEN={token}"]
    args += ["--entrypoint", "python3", image_ref, "-c", _GGUF_DOWNLOAD_SNIPPET]

    rc = -1
    try:
        async for event in stream_lines(args):
            if event[0] == "rc":
                rc = int(event[1])
            else:
                yield event
    except asyncio.CancelledError:
        await _run("docker", "rm", "-f", container_name, timeout=30)
        raise

    if rc != 0:
        yield ("rc", rc)
        return

    cached = gguf_in_cache(cache_path, hf_repo, hf_file)
    if cached is None:
        yield ("log", t(f"✗ {hf_file} is not in the cache after the download",
                        f"✗ 다운로드 후에도 캐시에 {hf_file} 이 없습니다"))
        yield ("rc", 1)
        return

    yield ("log", t(f"▸ Downloaded: {cached}", f"▸ 다운로드 완료: {cached}"))
    yield ("rc", 0)
