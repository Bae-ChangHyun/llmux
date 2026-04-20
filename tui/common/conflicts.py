"""Cross-backend 포트 / GPU 충돌 검사."""

from __future__ import annotations

from tui.common.adapter import DashboardRow
from tui.common.docker import parse_gpu_ids


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


def gpu_conflicts(target: DashboardRow, rows: list[DashboardRow]) -> list[str]:
    """running 상태 row 중 GPU index 가 교집합 있는 것."""
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
        shared = sorted(target_gpus & other_gpus)
        if not shared:
            continue
        label = _format_backend(r.backend)
        for gpu in shared:
            msgs.append(
                f"GPU {gpu} is used by {label} profile '{r.profile_name}' "
                f"(container '{r.container_name}')"
            )
    return msgs


def _format_backend(name: str) -> str:
    return {"vllm": "vLLM", "llamacpp": "llama.cpp"}.get(name, name)
