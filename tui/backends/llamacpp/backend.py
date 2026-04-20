"""Backend: 프로필/컨테이너 상태 스캔, 스크립트 래핑, config/profile CRUD."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT
PROFILES_DIR = PROJECT_ROOT / "profiles" / "llamacpp"
CONFIG_DIR = PROJECT_ROOT / "config" / "llamacpp"
COMPOSE_DIR = PROJECT_ROOT / "compose" / "llamacpp"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "llamacpp"
COMMON_ENV = PROJECT_ROOT / ".env.common"
CURRENT_PROFILE_FILE = PROJECT_ROOT / ".current-profile.llamacpp"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_name(name: str) -> bool:
    """alphanumeric + dash/underscore. - 시작 금지 (argv injection 방지)."""
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name))


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
        return PROFILES_DIR / f"{self.name}.env"


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


def list_profile_names() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        path.stem for path in PROFILES_DIR.glob("*.env") if path.stem != "example"
    )


def load_profile(name: str) -> Profile:
    path = PROFILES_DIR / f"{name}.env"
    env = _parse_env_file(path)
    return Profile(
        name=name,
        container_name=env.get("CONTAINER_NAME", name),
        port=int(env.get("LLAMA_PORT", "8080") or "8080"),
        gpu_id=env.get("GPU_ID", "0"),
        config_name=env.get("CONFIG_NAME", name),
        model_file=env.get("MODEL_FILE", ""),
        hf_repo=env.get("HF_REPO", ""),
        hf_file=env.get("HF_FILE", ""),
    )


def save_profile(profile: Profile) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Profile: {profile.name}",
        "",
        f"CONTAINER_NAME={profile.container_name or profile.name}",
        f"LLAMA_PORT={profile.port}",
        f"CONFIG_NAME={profile.config_name or profile.name}",
        f"GPU_ID={profile.gpu_id}",
    ]
    if profile.model_file or profile.hf_repo or profile.hf_file:
        lines += [
            "",
            "# 자동 다운로드 (pull-model.sh 가 사용)",
        ]
        if profile.model_file:
            lines.append(f"MODEL_FILE={profile.model_file}")
        if profile.hf_repo:
            lines.append(f"HF_REPO={profile.hf_repo}")
        if profile.hf_file:
            lines.append(f"HF_FILE={profile.hf_file}")
    lines.append("")
    profile.path.write_text("\n".join(lines))


def delete_profile(name: str, delete_config_too: bool = False) -> None:
    path = PROFILES_DIR / f"{name}.env"
    if delete_config_too and path.exists():
        config_name = load_profile(name).config_name
        if config_name:
            other_refs = [
                n for n in list_profile_names()
                if n != name and load_profile(n).config_name == config_name
            ]
            if not other_refs:
                cfg_path = CONFIG_DIR / f"{config_name}.yaml"
                if cfg_path.exists():
                    cfg_path.unlink()
    if path.exists():
        path.unlink()


def list_profiles() -> list[Profile]:
    """스캔: 실행 상태 + 다운로드 여부까지 조립."""
    if not PROFILES_DIR.is_dir():
        return []
    model_dir = _get_model_dir()
    current = read_current_profile()
    running_containers = _running_container_names()

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


def _running_container_names() -> set[str]:
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    if out.returncode != 0:
        return set()
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


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


async def run_script(script: str, *args: str) -> tuple[int, str]:
    """스크립트 실행, (exitcode, combined_output) 반환."""
    proc = await asyncio.create_subprocess_exec(
        str(SCRIPTS_DIR / script),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    stdout, _ = await proc.communicate()
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def stream_logs(container_name: str, lines: int = 200):
    """docker logs 를 async 로 스트리밍. 라인 단위 yield."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", str(lines), container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


async def quick_health(port: int) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sf", "--max-time", "2",
            f"http://localhost:{port}/health",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        return False


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

    loop = asyncio.get_event_loop()

    def _do():
        url = f"https://huggingface.co/api/models/{repo}/tree/main"
        req = urllib.request.Request(url, headers={"User-Agent": "llamacpp-compose"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    try:
        return await loop.run_in_executor(None, _do)
    except Exception:
        return []
