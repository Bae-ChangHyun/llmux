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


async def run_command(*args: str, timeout: float = 10) -> tuple[int, str]:
    """subprocess 실행. (exitcode, combined_stdout_stderr) 반환. 타임아웃/부재 시 (1, '')."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError):
        return 1, ""
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def get_gpu_info() -> list[GpuInfo]:
    rc, out = await run_command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
        timeout=5,
    )
    if rc != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(GpuInfo(*parts[:6]))
    return gpus


async def running_container_names() -> set[str]:
    """현재 실행 중인 모든 docker container 이름 (backend 무관)."""
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}", timeout=5
    )
    if rc != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


async def running_container_ports() -> dict[str, str]:
    """container_name → ports 문자열 맵 (docker ps). 포트 충돌 선제 검사용."""
    rc, out = await run_command(
        "docker", "ps", "--format", "{{.Names}}\t{{.Ports}}", timeout=5
    )
    if rc != 0:
        return {}
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


def parse_gpu_ids(raw: str) -> set[str]:
    """'0,1' or '0' or '' → set of GPU index strings."""
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}
