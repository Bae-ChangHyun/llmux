"""System inspection, version lookup, and vLLM metadata helpers.

get_gpu_info / format_gpu_bar / estimate_model_memory 은 tui.common 으로
이동됨. 하위 호환을 위해 이 모듈에서도 re-export.
"""

from __future__ import annotations

import re
from datetime import datetime

from tui.common.docker import GpuInfo, format_gpu_bar, get_gpu_info  # re-export
from tui.common.mem import estimate_model_memory  # re-export

from .backend_common import CONFIG_DIR, DockerImage
from .backend_process import run_command


async def get_docker_images(repo: str = "vllm/vllm-openai") -> list[DockerImage]:
    rc, out = await run_command(
        "docker",
        "images",
        repo,
        "--format",
        "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        return []
    images = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            images.append(
                DockerImage(
                    repository=parts[0],
                    tag=parts[1],
                    size=parts[2],
                    created=parts[3],
                )
            )
    return images


async def get_dev_images() -> list[DockerImage]:
    return await get_docker_images("vllm-dev")


def _parse_stable_version_tag(tag: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _pick_preferred_tag(tags: list[str]) -> str | None:
    """Pick the highest semantic-version tag (e.g. v0.19.1). Returns None if the
    image has only moving tags like `latest` / `nightly` / `<none>`.

    We deliberately ignore `latest` and `nightly` because they don't describe
    the actual image contents — they're just aliases that upstream rewrites.
    """
    stable_tags = [
        (version, tag)
        for tag in tags
        if (version := _parse_stable_version_tag(tag)) is not None
    ]
    if stable_tags:
        return max(stable_tags)[1]
    return None


async def get_local_latest_tag() -> str:
    """Return the highest-version local vllm/vllm-openai tag.

    Only semver-style tags (e.g. `v0.19.1`) are considered. `latest` and
    `nightly` are skipped because they don't self-describe. If no versioned
    tag exists locally, returns "none" so the UI can prompt the user to pull
    a specific version.
    """
    rc, out = await run_command(
        "docker",
        "images",
        "vllm/vllm-openai",
        "--format",
        "{{.ID}}\t{{.Tag}}",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return "none"

    image_tags: dict[str, list[str]] = {}
    for line in out.strip().splitlines():
        image_id, _, tag = line.partition("\t")
        image_id = image_id.strip()
        tag = tag.strip()
        if not image_id or not tag or tag == "<none>":
            continue
        image_tags.setdefault(image_id, []).append(tag)

    best_version: tuple[int, int, int] | None = None
    best_tag = "none"
    for tags in image_tags.values():
        preferred = _pick_preferred_tag(tags)
        if preferred is None:
            continue
        version = _parse_stable_version_tag(preferred)
        if version is None:
            continue
        if best_version is None or version > best_version:
            best_version = version
            best_tag = preferred

    return best_tag


async def get_dockerhub_release_version() -> str:
    """Get the latest exact stable release version from Docker Hub."""
    import json

    rc, out = await run_command(
        "curl",
        "-s",
        "--connect-timeout",
        "5",
        "--max-time",
        "10",
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return "unknown"
    try:
        data = json.loads(out)
        stable_tags = [
            (version, name)
            for result in data.get("results", [])
            if (name := result.get("name", ""))
            if (version := _parse_stable_version_tag(name)) is not None
        ]
        if stable_tags:
            return max(stable_tags)[1]
    except (json.JSONDecodeError, KeyError):
        pass
    return "unknown"


async def get_dockerhub_nightly_date() -> str:
    """Get last updated date of the nightly tag from Docker Hub."""
    import json

    rc, out = await run_command(
        "curl",
        "-s",
        "--connect-timeout",
        "5",
        "--max-time",
        "10",
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
    """Extract valid vllm serve parameters from a docker image."""
    import json

    if not image_tag:
        image_tag = await get_local_latest_tag()
        if image_tag == "none":
            return set()

    cache_file = _VLLM_PARAMS_CACHE_DIR / f".vllm-params-{image_tag}.json"
    if cache_file.exists():
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            pass

    rc, out = await run_command(
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        f"vllm/vllm-openai:{image_tag}",
        "-c",
        _EXTRACT_SCRIPT,
        timeout=30,
    )
    if rc != 0 or not out.strip():
        return set()

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
