"""VllmAdapter — vllm backend 를 통합 Dashboard 규격으로 노출."""

from __future__ import annotations

from tui.backends.vllm import backend as vbackend
from tui.common.adapter import DashboardRow


class VllmAdapter:
    name = "vllm"
    display_name = "vLLM"
    accent_color = "#7c3aed"   # purple-600

    def rows(self, running: set[str]) -> list[DashboardRow]:
        out: list[DashboardRow] = []
        for name in vbackend.list_profile_names():
            p = vbackend.load_profile(name)
            if p is None:
                continue
            port = _parse_port(getattr(p, "port", "") or "")
            container = getattr(p, "container_name", "") or name
            model = getattr(p, "config_name", "") or ""
            detail_parts: list[str] = []
            tp = getattr(p, "tensor_parallel", "") or ""
            if tp and tp != "1":
                detail_parts.append(f"tp={tp}")
            if (getattr(p, "enable_lora", "") or "").lower() == "true":
                detail_parts.append("lora")
            detail = " ".join(detail_parts)
            out.append(
                DashboardRow(
                    backend=self.name,
                    profile_name=name,
                    container_name=container,
                    port=port,
                    running=container in running,
                    model=model,
                    detail=detail,
                    gpu_id=getattr(p, "gpu_id", "") or "",
                    raw=p,
                )
            )
        return out


def _parse_port(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
