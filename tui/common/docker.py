"""Docker / nvidia-smi 공통 헬퍼 — 두 backend 가 동일 API 사용."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: str
    name: str
    memory_used: str   # MiB
    memory_total: str  # MiB
    utilization: str
    temperature: str
    power: str = ""    # W (may be "[N/A]" on GPUs that don't report it)


async def run_command(*args: str, timeout: float = 10) -> tuple[int, str]:
    """subprocess 실행. (exitcode, combined_stdout_stderr) 반환. 타임아웃/부재 시 (1, '')."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return 1, ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # Timed-out probe: kill and reap the child so it doesn't leak.
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await proc.wait()
        except (asyncio.CancelledError, ProcessLookupError, OSError):
            pass
        return 1, ""
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def gpu_probe_failed() -> bool:
    """True only when nvidia-smi exists but cannot be queried.

    A host without nvidia-smi is CPU-only — an empty GPU list is the correct
    answer there. A host that *has* the binary but errors out has a broken
    driver/toolkit, and callers must not report that as "0 GPUs".
    """
    import shutil

    if shutil.which("nvidia-smi") is None:
        return False
    rc, _ = await run_command("nvidia-smi", "--list-gpus", timeout=5)
    return rc != 0


async def get_gpu_info() -> list[GpuInfo]:
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        timeout=5,
    )
    if rc != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append(GpuInfo(*parts[:7]))
        elif len(parts) >= 6:
            gpus.append(GpuInfo(*parts[:6]))
    return gpus


async def get_pcie_stats() -> dict[str, tuple[float, float]]:
    """gpu index → (rx MB/s, tx MB/s) via `nvidia-smi dmon -c 1 -s t`.

    Empty dict when nvidia-smi is absent or dmon isn't supported. dmon prints a
    two-line comment header then one row per GPU: `idx rxpci txpci` in MB/s.
    """
    rc, out = await run_command(
        "nvidia-smi", "dmon", "-c", "1", "-s", "t", timeout=5
    )
    if rc != 0:
        return {}
    stats: dict[str, tuple[float, float]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        idx = parts[0]
        try:
            rx, tx = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        stats[idx] = (rx, tx)
    return stats


async def running_container_names() -> set[str]:
    """현재 실행 중인 모든 docker container 이름 (backend 무관)."""
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}", timeout=5
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "docker ps failed")
    return {line.strip() for line in out.splitlines() if line.strip()}


async def running_container_ports() -> dict[str, str]:
    """container_name → ports 문자열 맵 (docker ps). 포트 충돌 선제 검사용."""
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=5
    )
    if rc != 0:
        raise RuntimeError(out.strip() or "docker ps failed")
    result: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def format_gpu_bar(gpus: list[GpuInfo], bar_width: int = 8) -> str:
    """GPU info 를 rich markup progress bar 로. 두 backend 가 동일 표기."""
    if not gpus:
        return "[dim]GPU info unavailable[/dim]"
    parts: list[str] = []
    for g in gpus:
        try:
            used = int(g.memory_used)
            total = int(g.memory_total)
        except (ValueError, TypeError):
            continue
        ratio = used / total if total > 0 else 0.0
        filled = round(ratio * bar_width)
        empty = bar_width - filled
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
        mem = f"{used / 1024:.1f}/{total / 1024:.1f}GB"
        parts.append(
            f"[bold]GPU{g.index}[/bold] {bar}  {mem}  {g.utilization}%  {g.temperature}°C"
        )
    return "  [dim]│[/dim]  ".join(parts)


GPU_WILDCARD = "*"
"""모든 GPU 를 의미하는 토큰. `all`, `-1` 같은 특수값 정규화 시 사용."""


def parse_gpu_ids(raw: str) -> set[str]:
    """GPU 지정 문자열 → 정규화된 set.

    - '0,1', '0' → {'0'} / {'0','1'}
    - '' (공백/None) → set() — '지정 없음'. 충돌 검사에서 conservative(검출 안 함) 로 동작.
    - 'all' / '-1' (대소문자 무관) → {GPU_WILDCARD}. 다른 어떤 GPU 사용과도 교집합 있다고 본다.
    """
    if not raw:
        return set()
    out: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        lower = token.lower()
        if lower in {"all", "-1"}:
            out.add(GPU_WILDCARD)
        else:
            out.add(token)
    return out


def gpu_sets_overlap(a: set[str], b: set[str]) -> set[str]:
    """두 GPU 셋의 교집합. wildcard 가 한쪽이라도 있으면 상대 셋 전체를 돌려준다.
    양쪽 모두 wildcard 면 {GPU_WILDCARD} 반환."""
    if GPU_WILDCARD in a and GPU_WILDCARD in b:
        return {GPU_WILDCARD}
    if GPU_WILDCARD in a:
        return set(b) if b else {GPU_WILDCARD}
    if GPU_WILDCARD in b:
        return set(a) if a else {GPU_WILDCARD}
    return a & b


async def get_disk_usage(path: str) -> tuple[str, str, str]:
    """`df -h <path>` → (used, avail, percent). Shared by both backends'
    System / Disk view so the same parsing rules apply."""
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
