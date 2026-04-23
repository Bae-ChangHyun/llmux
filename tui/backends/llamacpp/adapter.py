"""LlamacppAdapter — llamacpp backend 를 통합 Dashboard 규격으로 노출."""

from __future__ import annotations

from tui.backends.llamacpp import backend as lbackend
from tui.common.adapter import DashboardRow


class LlamacppAdapter:
    name = "llamacpp"
    display_name = "llama.cpp"
    accent_color = "#16a34a"   # green-600

    def rows(self, running: set[str]) -> list[DashboardRow]:
        out: list[DashboardRow] = []
        profiles = lbackend.list_profiles(running=running)
        for p in profiles:
            detail = ""
            if p.model_size_gb is not None:
                detail = f"{p.model_size_gb:.1f} GB"
            elif p.hf_file:
                detail = p.hf_file.split("/")[-1]
            out.append(
                DashboardRow(
                    backend=self.name,
                    profile_name=p.name,
                    container_name=p.container_name or p.name,
                    port=p.port or None,
                    running=p.running,
                    model=p.config_name or "",
                    detail=detail,
                    gpu_id=p.gpu_id or "",
                    raw=p,
                )
            )
        return out
