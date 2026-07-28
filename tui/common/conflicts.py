"""Cross-backend 포트 / GPU 충돌 검사."""

from __future__ import annotations

import re

from tui.common.adapter import DashboardRow
from tui.common.docker import (
    GPU_WILDCARD,
    gpu_sets_overlap,
    parse_gpu_ids,
    run_command,
)


def _format_gpu_label(gpu_id: str) -> str:
    return "all GPUs" if gpu_id == GPU_WILDCARD else f"GPU {gpu_id}"


async def gpu_conflict_messages(
    *,
    profile_name: str,
    container_name: str,
    profile_gpu_id: str,
    backend: str,
) -> list[str]:
    """Cross-backend warnings when the target profile's GPUs overlap any
    currently-running container.

    Both backends call this with their own (name, container_name, gpu_id) and
    read off the same list of warnings.

    Never raises — a `docker ps` failure becomes a single inspection warning
    the caller logs alongside genuine overlap warnings, so it can't abort a
    startup.
    """
    # Lazy import: profile_store pulls in yaml at import time.
    from tui.common import profile_store

    rc, out = await run_command("docker", "ps", "--format", "{{.Names}}", timeout=10)
    if rc != 0:
        return [
            "Warning: could not inspect running containers for GPU overlap "
            f"({out.strip() or 'docker ps failed'})."
        ]
    running_names = {line.strip() for line in out.splitlines() if line.strip()}
    profile_gpu_ids = parse_gpu_ids(profile_gpu_id)
    # No GPUs requested → nothing to overlap. Without this, an empty set against
    # a wildcard-GPU container (`gpu_sets_overlap(∅, {*}) == {*}`) produced a
    # false "all GPUs" warning — mirror the sync gpu_conflicts() guard.
    if not profile_gpu_ids:
        return []
    messages: list[str] = []
    for backend_name in ("vllm", "llamacpp"):
        for other in profile_store.list_profiles(backend_name):
            if backend_name == backend and other.name == profile_name:
                continue
            other_container = other.container_name or other.name
            if other_container not in running_names:
                continue
            other_gpu_ids = parse_gpu_ids(other.gpu_id)
            for gpu_id in sorted(gpu_sets_overlap(profile_gpu_ids, other_gpu_ids)):
                friendly = "vLLM" if backend_name == "vllm" else "llama.cpp"
                messages.append(
                    f"Warning: {_format_gpu_label(gpu_id)} is also used by running "
                    f"{friendly} container '{other_container}'"
                )
    return messages


def _row_gpu_ids(row: DashboardRow) -> set[str]:
    gpu_raw = getattr(row, "gpu_id", "") or ""
    if not gpu_raw and row.raw is not None:
        gpu_raw = getattr(row.raw, "gpu_id", "") or ""
    return parse_gpu_ids(str(gpu_raw))


def port_conflicts(target: DashboardRow, rows: list[DashboardRow]) -> list[str]:
    """running 상태 row 중 port 가 target 과 같은 것. target 자신은 제외."""
    if target.port is None:
        return []
    msgs: list[str] = []
    for r in rows:
        if r is target:
            continue
        if r.backend == target.backend and r.profile_name == target.profile_name:
            continue
        if not r.running or r.port is None:
            continue
        if r.port == target.port:
            label = _format_backend(r.backend)
            msgs.append(
                f"Port {target.port} is occupied by {label} profile "
                f"'{r.profile_name}' (container '{r.container_name}')"
            )
    return msgs


def external_port_conflicts(
    target: DashboardRow,
    rows: list[DashboardRow],
    external_ports: dict[str, str],
) -> list[str]:
    """llmux 가 관리하지 않는 외부 컨테이너가 target.port 를 점유 중인지 감지.

    Parameters
    ----------
    target : DashboardRow
        새로 기동하려는 프로필.
    rows : list[DashboardRow]
        현재 llmux 가 아는 전체 row (external 제외용 화이트리스트).
    external_ports : dict[str, str]
        docker ps 결과 — container_name → "0.0.0.0:8080->8080/tcp" 식 ports 문자열.
    """
    if target.port is None:
        return []
    known_containers = {r.container_name for r in rows if r.container_name}
    msgs: list[str] = []
    pat = re.compile(r":(\d+)->")
    for cname, ports in external_ports.items():
        if cname in known_containers:
            continue
        for match in pat.finditer(ports):
            try:
                host_port = int(match.group(1))
            except ValueError:
                continue
            if host_port == target.port:
                msgs.append(
                    f"Port {target.port} is occupied by external container "
                    f"'{cname}' (not managed by llmux)"
                )
                break
    return msgs


def gpu_conflicts(target: DashboardRow, rows: list[DashboardRow]) -> list[str]:
    """running 상태 row 중 GPU index 가 교집합 있는 것. `all`/`-1` wildcard 지원."""
    target_gpus = _row_gpu_ids(target)
    if not target_gpus:
        return []
    msgs: list[str] = []
    for r in rows:
        if r is target:
            continue
        if r.backend == target.backend and r.profile_name == target.profile_name:
            continue
        if not r.running:
            continue
        other_gpus = _row_gpu_ids(r)
        if not other_gpus:
            continue
        shared = gpu_sets_overlap(target_gpus, other_gpus)
        if not shared:
            continue
        label = _format_backend(r.backend)
        for gpu in sorted(shared):
            display = "all GPUs" if gpu == GPU_WILDCARD else f"GPU {gpu}"
            msgs.append(
                f"{display} is used by {label} profile '{r.profile_name}' "
                f"(container '{r.container_name}')"
            )
    return msgs


def _format_backend(name: str) -> str:
    return {"vllm": "vLLM", "llamacpp": "llama.cpp"}.get(name, name)
