"""Backend: 프로필/컨테이너 상태 스캔, 스크립트 래핑, config/profile CRUD."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tui.common import profile_store
from tui.common.config_markers import (
    dump_active_config,
    parse_disabled_markers,
    render_disabled_markers,
)
from tui.common.env import parse_env_file as _parse_env_file  # noqa: F401 — re-exported for callers

log = logging.getLogger(__name__)


def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "profiles.example.yaml").exists() and (cwd / "compose").is_dir():
        return cwd
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _resolve_project_root()
ROOT = PROJECT_ROOT
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "llamacpp"
CONFIG_DIR = PROJECT_ROOT / "config" / "llamacpp"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "llamacpp"
COMMON_ENV = PROJECT_ROOT / ".env.common"
CURRENT_PROFILE_FILE = PROJECT_ROOT / ".current-profile.llamacpp"

LLAMACPP_OFFICIAL_REPO = "ghcr.io/ggml-org/llama.cpp"
LLAMACPP_OFFICIAL_IMAGE = f"{LLAMACPP_OFFICIAL_REPO}:server-cuda"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_name(name: str) -> bool:
    """compose-safe lowercase name. Also prevents argv/path injection."""
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


# ---------------------------------------------------------------------------
# .env / YAML helpers
# ---------------------------------------------------------------------------


def _host_expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _get_model_dir() -> Path:
    env_common = ROOT / ".env.common"
    default = ROOT / "models"
    if not env_common.exists():
        return default
    env = _parse_env_file(env_common)
    raw = env.get("MODEL_DIR")
    if not raw:
        return default
    return Path(_host_expand(raw))


def hf_cache_setting() -> str:
    """Raw HF_CACHE_PATH from .env.common, or "" when unset.

    Callers that display cache contents must say so: an unset value silently
    points every lookup at ~/.cache/huggingface, which makes a typo'd key look
    like "no models downloaded".
    """
    env_common = ROOT / ".env.common"
    if not env_common.exists():
        return ""
    return _parse_env_file(env_common).get("HF_CACHE_PATH", "").strip()


def _get_hf_cache_dir() -> Path:
    raw = hf_cache_setting()
    if not raw:
        return Path.home() / ".cache" / "huggingface"
    return Path(_host_expand(raw))


def find_cached_gguf(hf_repo: str, filename: str) -> Path | None:
    """Locate a GGUF that llama-server already pulled via `-hf`.

    `-hf <repo> -hff <file>` downloads into the HF hub cache layout
    (`hub/models--{org}--{name}/snapshots/{rev}/{file}`), not MODEL_DIR — so
    the legacy MODEL_DIR probe reports "not downloaded" for every profile that
    uses the (now default) in-container download path.
    """
    if not hf_repo or not filename:
        return None
    org, _, name = hf_repo.partition("/")
    if not org or not name:
        return None
    snapshots = (
        _get_hf_cache_dir() / "hub" / f"models--{org}--{name}" / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for match in sorted(snapshots.glob(f"*/{filename}")):
        if match.exists():
            return match
    return None


def list_cached_gguf() -> list[dict[str, Any]]:
    """Every GGUF present in the HF hub cache, largest first.

    Shared by the TUI System/Disk tab and `llmux system disk` so both report
    the same inventory.
    """
    hub = _get_hf_cache_dir() / "hub"
    if not hub.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for repo_dir in sorted(hub.glob("models--*")):
        # Inverse of huggingface_hub's repo_folder_name(): "/" is encoded "--".
        repo = repo_dir.name[len("models--"):].replace("--", "/")
        # rglob, not glob("*/*.gguf") — `-hff subdir/file.gguf` nests the GGUF
        # under the snapshot dir, and sharded models split across a subfolder.
        for path in (repo_dir / "snapshots").rglob("*.gguf"):
            if not path.exists():
                # Snapshot entries are symlinks into blobs/; a dangling link
                # means a half-evicted cache entry, not a usable model.
                continue
            size = path.stat().st_size
            out.append(
                {
                    "repo": repo,
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": size,
                    "size_gb": round(size / 1024**3, 1),
                }
            )
    out.sort(key=lambda d: d["size_bytes"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: int = 8080
    gpu_id: str = "0"
    config_name: str = ""
    model_file: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    image_tag: str = ""
    # profiles.yaml 의 env_vars. 여기 없으면 _to_profile/_to_stored 왕복에서
    # 유실돼, CLI 로 --set 한 값이 TUI 저장 한 번에 조용히 지워진다.
    env_vars: dict[str, str] = field(default_factory=dict)
    # 런타임 상태
    downloaded: bool = False
    model_size_gb: float | None = None
    running: bool = False
    is_current: bool = False

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def path(self) -> Path:
        """Runtime .env path rendered from profiles.yaml."""
        return RUNTIME_DIR / f"{self.name}.env"


@dataclass
class Config:
    """YAML config = llama-server flag 목록."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    # Params kept but not passed to llama-server — stored as comment markers.
    disabled_params: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.name}.yaml"

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


# ---------------------------------------------------------------------------
# Profile CRUD + scanning
# ---------------------------------------------------------------------------


def _to_profile(stored: profile_store.StoredProfile) -> Profile:
    return Profile(
        name=stored.name,
        container_name=stored.container_name or stored.name,
        port=stored.port,
        gpu_id=stored.gpu_id,
        config_name=stored.config_name,
        model_file=stored.model_file,
        hf_repo=stored.hf_repo,
        hf_file=stored.hf_file,
        image_tag=stored.image_tag,
        env_vars=dict(stored.env_vars),
    )


def _to_stored(profile: Profile) -> profile_store.StoredProfile:
    return profile_store.StoredProfile(
        name=profile.name,
        backend="llamacpp",
        container_name=profile.container_name or profile.name,
        port=int(profile.port),
        gpu_id=profile.gpu_id or "0",
        config_name=profile.config_name,
        model_file=profile.model_file,
        hf_repo=profile.hf_repo,
        hf_file=profile.hf_file,
        image_tag=profile.image_tag,
        env_vars=dict(profile.env_vars),
    )


def list_profile_names() -> list[str]:
    return profile_store.list_profile_names("llamacpp")


def load_profile(name: str) -> Profile:
    stored = profile_store.load_profile(name, "llamacpp")
    if stored is None:
        return Profile(name=name)
    try:
        profile_store.render_env(stored)
    except ValueError:
        # A value the .env renderer refuses (quote/newline/control char, e.g.
        # from a hand-edited profiles.yaml) must not take down every read path:
        # this same load_profile backs `ps`, the dashboard scan, and the
        # pre-flight of *other* profiles. Skip the .env refresh and let the
        # profile load; stream_container_up re-renders and fails loudly for the
        # one profile that is actually broken.
        pass
    return _to_profile(stored)


def save_profile(profile: Profile) -> None:
    profile_store.save_profile(_to_stored(profile))


def delete_profile(name: str, delete_config_too: bool = False) -> None:
    if delete_config_too:
        stored = profile_store.load_profile(name, "llamacpp")
        # example.yaml is the tracked template — a cascade delete would remove
        # it from the working tree. The profile itself still goes.
        if stored and stored.config_name and stored.config_name != "example":
            other_refs = [
                n for n in profile_store.list_profile_names("llamacpp")
                if n != name
                and (other := profile_store.load_profile(n, "llamacpp"))
                and other.config_name == stored.config_name
            ]
            if not other_refs:
                cfg_path = CONFIG_DIR / f"{stored.config_name}.yaml"
                if cfg_path.exists():
                    cfg_path.unlink()
    profile_store.delete_profile(name, "llamacpp")


def list_profiles(running: set[str] | None = None) -> list[Profile]:
    """스캔: 실행 상태 + 다운로드 여부까지 조립.

    `running` 은 호출자가 주입한 실행 중 컨테이너 이름 집합. 이벤트 루프 블로킹을
    피하기 위해 동기 subprocess 를 내부에서 호출하지 않는다. None 이면 빈 집합
    으로 취급하므로 TUI 는 Phase 마다 `tui.common.docker.running_container_names`
    를 한 번 await 해서 넘겨야 한다."""
    model_dir = _get_model_dir()
    current = read_current_profile()
    running_containers: set[str] = running or set()

    result: list[Profile] = []
    for name in list_profile_names():
        p = load_profile(name)
        # Only honor the legacy MODEL_DIR probe for profiles that can actually
        # start: compose doesn't mount MODEL_DIR and render-override requires
        # hf_repo, so a model_file-only profile whose GGUF sits in ./models
        # would `up`-fail every time — marking it "downloaded" was a false ready.
        if p.model_file and p.hf_repo:
            model_path = model_dir / p.model_file
            if model_path.exists():
                p.downloaded = True
                p.model_size_gb = round(model_path.stat().st_size / 1024**3, 1)
        if not p.downloaded:
            # MODEL_DIR is the legacy host-side layout; the live `-hf` path
            # lands in the HF hub cache instead.
            cached = find_cached_gguf(p.hf_repo, p.hf_file or p.model_file)
            if cached is not None:
                p.downloaded = True
                p.model_size_gb = round(cached.stat().st_size / 1024**3, 1)
        p.running = p.container_name in running_containers
        p.is_current = name == current
        result.append(p)
    return result


def read_current_profile() -> str | None:
    if not CURRENT_PROFILE_FILE.exists():
        return None
    return CURRENT_PROFILE_FILE.read_text().strip() or None


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


def list_config_names() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.stem for path in CONFIG_DIR.glob("*.yaml") if path.stem != "example"
    )


def load_config(name: str) -> Config:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return Config(name=name)
    text = path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raw = {}
    params = {str(k): v for k, v in raw.items()}
    # A disabled marker whose key is also active is ignored — active wins.
    disabled = {
        k: v for k, v in parse_disabled_markers(text).items() if k not in params
    }
    return Config(name=name, params=params, disabled_params=disabled)


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = config.path.read_text() if config.path.exists() else None
    text = dump_active_config(existing, config.params)
    config.path.write_text(text + render_disabled_markers(config.disabled_params))


def delete_config(name: str) -> None:
    path = CONFIG_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()


def parse_config_param_value(raw: str) -> Any:
    """UI 입력 → YAML-safe Python 값. 빈 값은 True (boolean flag)."""
    if raw == "":
        return True
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def format_config_param_value(value: Any) -> str:
    """YAML 값 → UI 편집 가능한 문자열."""
    if value is True:
        return ""
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(
            value, default_flow_style=True, allow_unicode=True, sort_keys=False
        ).strip()
    return str(value)


# ---------------------------------------------------------------------------
# llama-server flag discovery (선택적: docker run llama-server --help)
# ---------------------------------------------------------------------------


async def extract_llama_server_flags() -> set[str]:
    """llama-server --help 를 docker 로 실행해 --foo-bar 플래그들 파싱.
    실패 시 빈 set 반환."""
    env = _parse_env_file(ROOT / ".env.common")
    image = env.get("LLAMACPP_IMAGE", "") or LLAMACPP_OFFICIAL_IMAGE
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--entrypoint", "/app/llama-server",
            image, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        # Don't leave the `docker run` orphaned when --help hangs — reap it like
        # the other subprocess helpers do.
        if proc is not None:
            proc.kill()
            await proc.wait()
        return set()
    except FileNotFoundError:
        return set()
    if proc.returncode not in (0, 1):
        return set()
    text = stdout.decode("utf-8", errors="replace")
    flags: set[str] = set()
    for match in re.finditer(r"--([a-zA-Z][a-zA-Z0-9-]+)", text):
        flag = match.group(1)
        if 2 <= len(flag) <= 40:
            flags.add(flag)
    return flags


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------


async def stream_logs(container_name: str, *, tail: int = 100):
    """docker logs 를 async 로 스트리밍. 라인 단위 yield.

    Signature is keyword-only `tail` for parity with
    `tui.backends.vllm.backend_runtime.stream_container_logs` — both backends
    expose `(container_name, *, tail: int = 100)` so the CLI follow path can
    call either interchangeably."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", str(tail), container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is None:
        return
    try:
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            yield chunk.decode("utf-8", errors="replace").rstrip()
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


from tui.common.http import chat_completion_bench as chat_completion  # noqa: F401 — re-exported


# ---------------------------------------------------------------------------
# GPU / Docker image inspection
# ---------------------------------------------------------------------------


from tui.common.docker import (  # noqa: F401 — re-exported for backward compat
    GpuInfo,
    format_gpu_bar,
    get_disk_usage,
    get_gpu_info,
    run_command,
)


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str


async def get_docker_images(repo: str = LLAMACPP_OFFICIAL_REPO) -> list[DockerImage]:
    """Local images for `repo`. Raises if the probe fails — an empty list must
    mean "no such image", never "docker could not be reached"."""
    rc, out = await run_command(
        "docker", "images", repo,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        raise RuntimeError(
            f"docker images {repo} failed or timed out: {out.strip() or 'no output'}"
        )
    images: list[DockerImage] = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] != "<none>":
            images.append(DockerImage(*parts[:4]))
    return images


# ---------------------------------------------------------------------------
# HuggingFace repo helpers (QuickSetup 용)
# ---------------------------------------------------------------------------


_HF_TREE_PAGE_CAP = 10


def _parse_link_next(header: str) -> str:
    """Pull the rel="next" URL out of an RFC-5988 Link header ("" when absent)."""
    for part in header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url = segments[0].strip()
        if not (url.startswith("<") and url.endswith(">")):
            continue
        for attr in segments[1:]:
            key, _, value = attr.partition("=")
            if key.strip() == "rel" and value.strip().strip('"') == "next":
                return url[1:-1]
    return ""


class HfListingUnavailable(RuntimeError):
    """The HF file listing could not be fetched — distinct from "no files"."""


async def list_hf_repo_files(repo: str) -> list[dict]:
    """HF API 로 repo 의 파일 목록 가져오기. GGUF 파일만 필터링하지는 않음.

    `recursive=true` 필수: 대형 sharded 모델은 quant 별 하위폴더(`Q4_K_M/...`)에
    GGUF 를 두는데, 비재귀 호출은 그런 폴더를 `type: "directory"` 항목 하나로만
    돌려줘서 QuickSetup 이 "GGUF 없음" 으로 오판한다. tree API 는 1000개 단위로
    페이지네이션되므로 Link: rel="next" 를 따라가며 병합한다.
    """
    import urllib.request

    loop = asyncio.get_running_loop()

    def _do():
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/api/models/{repo}/tree/main?recursive=true"
        headers = {"User-Agent": "llmux"}
        token = _parse_env_file(ROOT / ".env.common").get("HF_TOKEN", "").strip()
        if token and not token.startswith("your_"):
            headers["Authorization"] = f"Bearer {token}"

        entries: list[dict] = []
        # 무한 루프 방지 캡. 1000 * 10 = 10k 파일이면 어떤 GGUF 리포에도 충분하다.
        for _ in range(_HF_TREE_PAGE_CAP):
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                page = json.loads(r.read().decode())
                link = r.headers.get("Link", "") or ""
            if isinstance(page, list):
                entries.extend(page)
            url = _parse_link_next(link)
            if not url:
                break
        else:
            # Cap hit with a next link still pending — the listing is truncated,
            # so a GGUF the user asked for may look missing. Say so instead of
            # returning a silently-short list.
            if url:
                log.warning(
                    "HF tree listing for %s truncated at %d pages (%d entries); "
                    "files beyond that are not listed.",
                    repo, _HF_TREE_PAGE_CAP, len(entries),
                )
        return entries

    try:
        return await loop.run_in_executor(None, _do)
    except Exception as exc:
        raise HfListingUnavailable(f"{repo}: {exc}") from exc
