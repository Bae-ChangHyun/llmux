"""System inspection, version lookup, and vLLM metadata helpers.

get_gpu_info / format_gpu_bar / estimate_model_memory 은 tui.common 으로
이동됨. 하위 호환을 위해 이 모듈에서도 re-export.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import urllib.request
from datetime import datetime

from tui.common.docker import GpuInfo, format_gpu_bar, get_gpu_info  # re-export
from tui.common.mem import estimate_model_memory  # re-export

from .backend_common import CONFIG_DIR, DockerImage
from .backend_process import run_command

logger = logging.getLogger(__name__)


# CA bundle locations checked in order. Some Python builds (e.g. uv-managed
# CPython on RHEL/CentOS) ship with an OpenSSL default cafile of
# `/etc/ssl/cert.pem` that does not exist on the host, which causes urllib
# requests to Docker Hub to fail with `CERTIFICATE_VERIFY_FAILED`.
_SYSTEM_CA_CANDIDATES = (
    "/etc/pki/tls/certs/ca-bundle.crt",     # RHEL / CentOS / Fedora
    "/etc/ssl/certs/ca-certificates.crt",   # Debian / Ubuntu / Alpine
    "/etc/ssl/cert.pem",                    # macOS / OpenSSL default
)

_ssl_context: ssl.SSLContext | None = None


def _get_ssl_context() -> ssl.SSLContext:
    global _ssl_context
    if _ssl_context is not None:
        return _ssl_context

    candidates: list[str] = []
    try:
        import certifi  # type: ignore[import-not-found]
        candidates.append(certifi.where())
    except Exception:
        pass
    candidates.extend(_SYSTEM_CA_CANDIDATES)

    for cafile in candidates:
        if not cafile or not os.path.exists(cafile):
            continue
        try:
            _ssl_context = ssl.create_default_context(cafile=cafile)
            return _ssl_context
        except Exception:
            continue

    _ssl_context = ssl.create_default_context()
    return _ssl_context


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


async def _fetch_json_url(
    url: str,
    timeout: float = 5.0,
    *,
    headers: dict[str, str] | None = None,
) -> dict | None:
    """Fetch JSON from URL in a thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()

    def _fetch() -> dict | None:
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "llmux/1.0 (+https://github.com/Bae-ChangHyun/llmux)",
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, headers=request_headers)
        context = _get_ssl_context()
        first_error: Exception | None = None
        for target in (request, url):
            try:
                with urllib.request.urlopen(
                    target, timeout=timeout, context=context
                ) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                data = json.loads(payload)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                continue
        if first_error is not None:
            logger.debug("fetch %s failed: %s", url, first_error)
        return None

    return await loop.run_in_executor(None, _fetch)


async def _fetch_docker_registry_tags() -> list[str]:
    token_url = (
        "https://auth.docker.io/token?"
        "service=registry.docker.io&scope=repository:vllm/vllm-openai:pull"
    )
    token_payload = await _fetch_json_url(token_url, timeout=5.0)
    token = str((token_payload or {}).get("token", "")).strip()
    if not token:
        return []

    tags_payload = await _fetch_json_url(
        "https://registry-1.docker.io/v2/vllm/vllm-openai/tags/list?n=1000",
        timeout=5.0,
        headers={"Authorization": f"Bearer {token}"},
    )
    tags = (tags_payload or {}).get("tags", [])
    return [str(tag) for tag in tags if tag]


async def get_dockerhub_release_version() -> str:
    """Get the latest exact stable release version from Docker Hub."""
    base_urls = [
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
        "https://registry.hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100",
    ]
    for attempt in range(3):
        for base_url in base_urls:
            url = base_url
            pages_checked = 0
            while url and pages_checked < 5:
                data = await _fetch_json_url(url, timeout=5.0)
                if not data:
                    break
                stable_tags = [
                    (version, name)
                    for result in data.get("results", [])
                    if isinstance(result, dict)
                    if (name := str(result.get("name", "")))
                    if (version := _parse_stable_version_tag(name)) is not None
                ]
                if stable_tags:
                    return max(stable_tags)[1]
                next_url = data.get("next", "")
                url = str(next_url) if next_url else ""
                pages_checked += 1
        if attempt < 2:
            await asyncio.sleep(0.5)
    registry_preferred = _pick_preferred_tag(await _fetch_docker_registry_tags())
    if registry_preferred:
        return registry_preferred
    return "unknown"


async def get_dockerhub_nightly_date() -> str:
    """Get last updated date of the nightly tag from Docker Hub."""
    urls = [
        "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/nightly",
        "https://registry.hub.docker.com/v2/repositories/vllm/vllm-openai/tags/nightly",
    ]
    for attempt in range(3):
        for url in urls:
            data = await _fetch_json_url(url, timeout=5.0)
            if not data:
                continue
            last_updated = str(data.get("last_updated", "")).strip()
            if last_updated:
                return last_updated.split("T")[0]
        if attempt < 2:
            await asyncio.sleep(0.5)
    registry_tags = await _fetch_docker_registry_tags()
    return "available" if "nightly" in registry_tags else "unknown"


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
