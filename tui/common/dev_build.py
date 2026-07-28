"""Backend-agnostic dev image builder.

vllm and llama.cpp share the same shape:
  1. git clone/update a source checkout
  2. docker build -f <dockerfile> --target <stage> -t <prefix>:<tag>
  3. label the image with repo/branch/commit so we can identify it later

The mechanics (clone, fetch, checkout, pull, rev-parse, docker build) are
identical. The backend-specific pieces are:
  - which directory to check out into (`.vllm-src` vs `.llamacpp-src`)
  - default repo URL, image prefix
  - Dockerfile path within the checkout
  - whether to detect local GPU arch and pass it as a build-arg
  - optional Dockerfile patches (vllm's DeepEP arch fixup)

The backend hands us a DevBuildSpec and any extra build args; this module
handles the rest, yielding ("log", str) / ("commit", sha) / ("rc", int)
events compatible with the existing AsyncGenerator API.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DevBuildSpec:
    """Backend-specific knobs for the unified dev build pipeline."""

    backend: str                       # "vllm" or "llamacpp"
    image_prefix: str                  # "vllm-dev" or "llamacpp-dev"
    src_dir: Path                      # absolute path; module-level constant on each backend
    default_repo_url: str
    default_branch: str = "main"
    dockerfile_relpath: str = ""       # relative to src_dir, e.g. "docker/Dockerfile"
    target: str = ""                   # docker build --target value; "" disables
    base_build_args: tuple[tuple[str, str], ...] = ()
    label_prefix: str = ""             # defaults to backend


def _label(spec: DevBuildSpec) -> str:
    return spec.label_prefix or spec.backend


_TAG_INVALID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_docker_tag(name: str) -> str:
    """Replace docker-tag-invalid characters with `-`.

    Docker image tags accept `[A-Za-z0-9_.-]` and must not start with `.` or `-`.
    Branch names like `releases/v0.21.0` or `feat/foo` are common, so map every
    invalid character to `-` and trim leading punctuation. The branch's original
    spelling is still preserved in the `<prefix>.repo.branch` label.

    Public because the *runtime* paths must derive the exact same tag the
    builder stamps (`<prefix>:<safe_branch>`) — a slash in a branch like
    `feat/foo` is not a valid docker tag reference.
    """
    sanitized = _TAG_INVALID_CHARS.sub("-", name).lstrip(".-")
    return sanitized or "branch"


# Back-compat alias for existing internal callers.
_sanitize_docker_tag = sanitize_docker_tag


_DEV_IMAGE_PREFIXES = ("vllm-dev:", "llamacpp-dev:")
_TAG_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")


def image_tag_error(value: str) -> str:
    """Why `value` is an unusable image reference, or "" if it's fine.

    Empty is allowed (it clears the pin). For our own dev images the tag part
    must already equal `sanitize_docker_tag(tag)` — otherwise a pin like
    `vllm-dev:feat/foo` would name an image the builder/runtime (which sanitize
    to `feat-foo`) never produce. For any other reference the tag part just has
    to be a legal docker tag.
    """
    value = value.strip()
    if not value:
        return ""
    for prefix in _DEV_IMAGE_PREFIXES:
        if value.startswith(prefix):
            tag = value[len(prefix):]
            if not tag:
                return f"{prefix.rstrip(':')} image needs a tag after ':'"
            safe = sanitize_docker_tag(tag)
            if tag != safe:
                return (
                    f"invalid dev image tag {tag!r}; docker tags can't contain "
                    f"'/' or other specials — did you mean {prefix}{safe}?"
                )
            return ""
    # Generic reference: the tag is the segment after the last ':' *unless* that
    # colon belongs to a registry host:port (i.e. sits before the last '/').
    has_tag = ":" in value and value.rfind(":") > value.rfind("/")
    if not has_tag:
        # An untagged ref resolves to `:latest` at pull time — same ambiguous
        # alias the Custom Tag field hard-rejects, so refuse it here too.
        return (
            f"image reference {value!r} has no tag and would resolve to "
            "`:latest`; pin a specific version tag."
        )
    tag = value.rsplit(":", 1)[1]
    if tag == "latest":
        return (
            "`:latest` is an ambiguous alias and is not allowed; pin a specific "
            "version tag."
        )
    if not _TAG_PART_RE.match(tag):
        return (
            f"invalid image tag {tag!r}; must match {_TAG_PART_RE.pattern} "
            "(start with a letter/digit/underscore, then letters/digits/._-)"
        )
    return ""


# ---------------------------------------------------------------------------
# subprocess helpers (small, local — avoids cross-backend coupling)
# ---------------------------------------------------------------------------


async def _run(*args: str, cwd: Path | None = None, timeout: float = 30) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError:
        # Binary not on PATH (e.g. nvidia-smi on a CPU-only host or CI).
        # Callers treat a non-zero rc as "command unavailable" and degrade
        # gracefully — e.g. detect_local_gpu_caps() returns [] → multi-arch.
        return -1, f"command not found: {args[0]}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def _stream(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
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
# git clone/update
# ---------------------------------------------------------------------------


async def clone_or_update(spec: DevBuildSpec, repo_url: str, branch: str):
    """Yield ('log', str) / ('commit', sha) / ('rc', int).

    Refuses to silently switch remotes if .git/ already points at a different
    repository URL — the caller can rm -rf the checkout if they want to swap.
    """
    if spec.src_dir.joinpath(".git").exists():
        yield ("log", f"Updating existing {spec.backend} source...")
        rc, current = await _run("git", "remote", "get-url", "origin", cwd=spec.src_dir, timeout=30)
        if rc != 0:
            yield ("log", current.strip() or f"Error: failed to inspect existing {spec.backend} source")
            yield ("rc", 1)
            return
        if current.strip() != repo_url:
            yield ("log", f"Error: existing {spec.src_dir.name} remote URL differs from the requested repository.")
            yield ("log", f"Existing: {current.strip()}")
            yield ("log", f"Requested: {repo_url}")
            yield ("log", f"Move or delete {spec.src_dir.name} yourself if you want to replace the checkout.")
            yield ("rc", 1)
            return

    if not spec.src_dir.exists():
        async for event in _stream(["git", "clone", repo_url, str(spec.src_dir)]):
            if event[0] == "rc":
                if event[1] != 0:
                    yield event
                    return
                continue
            yield event

    rc, out = await _run("git", "fetch", "origin", cwd=spec.src_dir, timeout=120)
    if rc != 0:
        yield ("log", out.strip() or "Error: git fetch failed")
        yield ("rc", rc)
        return

    rc, out = await _run("git", "checkout", branch, cwd=spec.src_dir, timeout=60)
    if rc != 0:
        rc, out = await _run(
            "git", "checkout", "-b", branch, f"origin/{branch}", cwd=spec.src_dir, timeout=60
        )
        if rc != 0:
            yield ("log", out.strip() or f"Error: failed to checkout branch {branch}")
            yield ("rc", rc)
            return

    rc, out = await _run("git", "pull", "origin", branch, cwd=spec.src_dir, timeout=120)
    if rc != 0:
        yield ("log", out.strip() or f"Error: git pull failed for branch {branch}")
        yield ("log", f"Hint: stash or reset local changes in {spec.src_dir.name}/, then retry.")
        yield ("rc", rc)
        return

    rc, sha = await _run("git", "rev-parse", "--short", "HEAD", cwd=spec.src_dir, timeout=30)
    if rc != 0:
        yield ("log", sha.strip() or "Error: failed to read commit hash")
        yield ("rc", rc)
        return
    yield ("commit", sha.strip())


# ---------------------------------------------------------------------------
# docker build
# ---------------------------------------------------------------------------


async def stream_build(
    spec: DevBuildSpec,
    branch: str,
    *,
    repo_url: str = "",
    custom_tag: str = "",
    extra_build_args: tuple[tuple[str, str], ...] = (),
    extra_log_lines: tuple[str, ...] = (),
    pre_build=None,  # async callable() -> (ok: bool, message: str) or None
    extra_labels: tuple[tuple[str, str], ...] = (),
):
    """Clone/update, then docker build with backend-prefixed tags + labels.

    Tags applied:
      - <prefix>:<custom_tag or branch-YYYYMMDD>  (the unique tag)
      - <prefix>:<branch>                         (stable alias to the latest build)

    `pre_build` runs between clone_or_update and docker build — backends use
    this to patch the freshly-checked-out Dockerfile (vllm's DeepEP arch fix).
    `extra_build_args` lets the backend pass through things like
    `--build-arg torch_cuda_arch_list=...`.
    """
    resolved_repo = repo_url or spec.default_repo_url
    resolved_branch = branch or spec.default_branch
    # docker tags reject `/` and a few other characters; branch names like
    # `releases/v0.21.0` are common so sanitize for the tag while keeping
    # the original string in the `.repo.branch` label.
    safe_branch = _sanitize_docker_tag(resolved_branch)
    # custom_tag goes through the same sanitizer: the start path
    # (`up --dev --tag feat/foo`) sanitizes before looking the image up, so a
    # raw tag here would either be an invalid docker reference or — worse —
    # build under a name the runtime never resolves to.
    main_tag = (
        _sanitize_docker_tag(custom_tag)
        if custom_tag
        else f"{safe_branch}-{datetime.now().strftime('%Y%m%d')}"
    )

    yield ("log", f"Building {spec.backend} from source")
    yield ("log", f"Repository: {resolved_repo}")
    yield ("log", f"Branch: {resolved_branch}")
    for extra in extra_log_lines:
        yield ("log", extra)
    yield ("log", f"Tag: {spec.image_prefix}:{main_tag}")

    commit_hash = ""
    async for event in clone_or_update(spec, resolved_repo, resolved_branch):
        if event[0] == "commit":
            commit_hash = event[1]
        else:
            yield event
            if event[0] == "rc" and event[1] != 0:
                return

    dockerfile_path = spec.src_dir / spec.dockerfile_relpath if spec.dockerfile_relpath else None
    if dockerfile_path and not dockerfile_path.exists():
        yield ("log", f"Error: Dockerfile not found at {dockerfile_path}")
        yield ("rc", 1)
        return

    if pre_build is not None:
        ok, msg = await pre_build()
        if msg:
            yield ("log", msg)
        if not ok:
            yield ("rc", 1)
            return

    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    label_prefix = _label(spec)
    cmd: list[str] = ["docker", "build"]
    if dockerfile_path:
        cmd.extend(["-f", str(dockerfile_path)])
    if spec.target:
        cmd.extend(["--target", spec.target])
    for arg_k, arg_v in spec.base_build_args:
        cmd.extend(["--build-arg", f"{arg_k}={arg_v}"])
    for arg_k, arg_v in extra_build_args:
        cmd.extend(["--build-arg", f"{arg_k}={arg_v}"])
    cmd.extend([
        "--label", f"{label_prefix}.repo.url={resolved_repo}",
        "--label", f"{label_prefix}.repo.branch={resolved_branch}",
        "--label", f"{label_prefix}.commit.hash={commit_hash}",
        "--label", f"{label_prefix}.build.date={build_date}",
    ])
    for lk, lv in extra_labels:
        cmd.extend(["--label", f"{lk}={lv}"])
    cmd.extend([
        "-t", f"{spec.image_prefix}:{main_tag}",
        "-t", f"{spec.image_prefix}:{safe_branch}",
    ])
    cmd.append(str(spec.src_dir))

    build_env = os.environ.copy()
    build_env.setdefault("DOCKER_BUILDKIT", "1")
    async for event in _stream(cmd, env=build_env):
        if event[0] == "rc" and event[1] != 0:
            yield event
            return
        yield event


# ---------------------------------------------------------------------------
# image inspection helpers
# ---------------------------------------------------------------------------


async def detect_local_gpu_caps() -> list[str]:
    """Return unique compute_cap strings from `nvidia-smi`, sorted.

    Examples: ["8.9"] for a single RTX 4080 SUPER, ["8.6", "8.9"] for a
    mixed-SM machine. Empty list if nvidia-smi fails — caller decides
    whether to fall back to a multi-arch build.
    """
    rc, out = await _run(
        "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", timeout=10
    )
    if rc != 0 or not out.strip():
        return []
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def format_arch_torch(caps: list[str]) -> str:
    """vLLM / PyTorch convention: dotted, space-separated (e.g. '8.6 8.9')."""
    return " ".join(caps)


def format_arch_cmake(caps: list[str]) -> str:
    """CMake CUDA_ARCHITECTURES / llama.cpp CUDA_DOCKER_ARCH convention:
    no dot, semicolon-separated (e.g. '86;89')."""
    return ";".join(c.replace(".", "") for c in caps)


async def get_image_label(image_ref: str, label: str) -> str:
    # The label key must be a Go *string* literal — double-quoted. Single
    # quotes are parsed as a rune literal and `docker inspect` fails with
    # rc=64.
    rc, out = await _run(
        "docker",
        "inspect",
        image_ref,
        '--format={{index .Config.Labels "' + label + '"}}',
        timeout=20,
    )
    if rc != 0:
        return ""
    value = out.strip()
    return "" if value == "<no value>" else value


async def image_matches(
    spec: DevBuildSpec, image_tag: str, repo_url: str, branch: str
) -> bool:
    """Return True iff <prefix>:<image_tag> was built from this repo+branch."""
    label_prefix = _label(spec)
    image_ref = f"{spec.image_prefix}:{image_tag}"
    saved_repo = await get_image_label(image_ref, f"{label_prefix}.repo.url")
    saved_branch = await get_image_label(image_ref, f"{label_prefix}.repo.branch")
    if not saved_repo or not saved_branch:
        return False
    return saved_repo == repo_url and saved_branch == branch


@dataclass
class DevImage:
    repository: str
    tag: str
    size: str
    created: str


async def list_local_dev_images(spec: DevBuildSpec) -> list[DevImage]:
    """Local dev images. Raises if the probe fails — an empty list must mean
    "none built", never "docker could not be reached"."""
    rc, out = await _run(
        "docker", "images", spec.image_prefix,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images {spec.image_prefix} failed or timed out: "
            f"{out.strip() or 'no output'}"
        )
    images: list[DevImage] = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] != "<none>":
            images.append(DevImage(repository=parts[0], tag=parts[1], size=parts[2], created=parts[3]))
    return images


async def image_exists_locally(spec: DevBuildSpec, image_tag: str) -> bool:
    rc, _ = await _run(
        "docker", "image", "inspect", f"{spec.image_prefix}:{image_tag}", timeout=20
    )
    return rc == 0
