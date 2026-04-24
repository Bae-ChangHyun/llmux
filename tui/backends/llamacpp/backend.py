"""Backend: 프로필/컨테이너 상태 스캔, 스크립트 래핑, config/profile CRUD."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tui.common import profile_store

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_name(name: str) -> bool:
    """compose-safe lowercase name. Also prevents argv/path injection."""
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


# ---------------------------------------------------------------------------
# .env / YAML helpers
# ---------------------------------------------------------------------------


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        env[k.strip()] = v
    return env


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
    )


def list_profile_names() -> list[str]:
    return profile_store.list_profile_names("llamacpp")


def load_profile(name: str) -> Profile:
    stored = profile_store.load_profile(name, "llamacpp")
    if stored is None:
        return Profile(name=name)
    profile_store.render_env(stored)
    return _to_profile(stored)


def save_profile(profile: Profile) -> None:
    profile_store.save_profile(_to_stored(profile))


def delete_profile(name: str, delete_config_too: bool = False) -> None:
    if delete_config_too:
        stored = profile_store.load_profile(name, "llamacpp")
        if stored and stored.config_name:
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
        if p.model_file:
            model_path = model_dir / p.model_file
            if model_path.exists():
                p.downloaded = True
                p.model_size_gb = round(model_path.stat().st_size / 1024**3, 1)
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
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raw = {}
    return Config(name=name, params={str(k): v for k, v in raw.items()})


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.path.write_text(
        yaml.dump(
            config.params,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    )


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
    image = env.get("LLAMACPP_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--entrypoint", "/app/llama-server",
            image, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (asyncio.TimeoutError, FileNotFoundError):
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
# Script / subprocess wrappers
# ---------------------------------------------------------------------------


async def run_script(
    script: str, *args: str, timeout: float | None = 1800.0
) -> tuple[int, str]:
    """스크립트 실행, (exitcode, combined_output) 반환.

    기본 timeout 30분 — HF GGUF 다운로드는 수십 GB 규모라 긴 여유를 둔다.
    타임아웃 시 프로세스를 kill 하고 (124, '<stdout+...TIMED OUT>') 를 돌려준다.
    """
    proc = await asyncio.create_subprocess_exec(
        str(SCRIPTS_DIR / script),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    try:
        if timeout is None:
            stdout, _ = await proc.communicate()
        else:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        raise
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return 124, f"✗ '{script}' timed out after {timeout}s"
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def stream_script(
    script: str, *args: str
):
    """스크립트 실행을 라인 단위로 스트리밍."""
    proc = await asyncio.create_subprocess_exec(
        str(SCRIPTS_DIR / script),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    if proc.stdout is None:
        yield ("rc", 1)
        return
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode("utf-8", errors="replace").rstrip())
        await proc.wait()
        yield ("rc", proc.returncode or 0)
    except asyncio.CancelledError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        raise


async def stream_logs(container_name: str, lines: int = 200):
    """docker logs 를 async 로 스트리밍. 라인 단위 yield."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", str(lines), container_name,
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


from tui.common.http import chat_completion_bench as chat_completion  # re-export


# ---------------------------------------------------------------------------
# GPU / Docker image inspection
# ---------------------------------------------------------------------------


from tui.common.docker import (  # re-export — 공통 구현 사용
    GpuInfo,
    format_gpu_bar,
    get_gpu_info,
    run_command,
)


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str


async def get_docker_images(repo: str = "ghcr.io/ggml-org/llama.cpp") -> list[DockerImage]:
    rc, out = await run_command(
        "docker", "images", repo,
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}",
        timeout=10,
    )
    if rc != 0:
        return []
    images: list[DockerImage] = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[1] != "<none>":
            images.append(DockerImage(*parts[:4]))
    return images


async def get_disk_usage(path: str) -> tuple[str, str, str]:
    """return (used, avail, percent) for the filesystem containing `path`."""
    rc, out = await run_command("df", "-h", path, timeout=5)
    if rc != 0:
        return "", "", ""
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return "", "", ""
    parts = lines[1].split()
    # Filesystem  Size  Used Avail Use% Mounted
    if len(parts) < 5:
        return "", "", ""
    return parts[2], parts[3], parts[4]


# ---------------------------------------------------------------------------
# HuggingFace repo helpers (QuickSetup 용)
# ---------------------------------------------------------------------------


async def list_hf_repo_files(repo: str) -> list[dict]:
    """HF API 로 repo 의 파일 목록 가져오기. GGUF 파일만 필터링하지는 않음."""
    import urllib.request

    loop = asyncio.get_running_loop()

    def _do():
        url = f"https://huggingface.co/api/models/{repo}/tree/main"
        headers = {"User-Agent": "llmux"}
        token = _parse_env_file(ROOT / ".env.common").get("HF_TOKEN", "").strip()
        if token and not token.startswith("your_"):
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    try:
        return await loop.run_in_executor(None, _do)
    except Exception:
        return []
