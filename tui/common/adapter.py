"""Backend adapter protocol — Dashboard 가 두 backend 를 동일한 방식으로 조회."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class DashboardRow:
    """통합 대시보드 한 행을 구성하는 read-only snapshot."""

    backend: str              # "vllm" | "llamacpp"
    profile_name: str         # 프로필 식별자 (파일명 stem)
    container_name: str       # docker container name (없을 수 있음)
    port: int | None          # 서비스 포트 (없을 수 있음)
    running: bool             # 컨테이너 실행 중 여부
    model: str                # 짧은 모델 설명 (config name 등)
    detail: str               # 추가 상세 (ex: "tp=2 lora" 또는 "35B Q4_K_XL")
    gpu_id: str = ""          # '0' / '0,1' — 충돌 검사용
    raw: Any = None           # 원본 Profile 객체 (action 시 backend 로 되돌릴 때 참조)


@runtime_checkable
class BackendAdapter(Protocol):
    """각 backend 가 구현하는 최소 인터페이스.

    Dashboard 는 `rows()` 만 사용. Profile/Config/Action 등 backend-specific 화면 은
    기존 backend 의 screens 를 그대로 push 하므로 adapter 가 관여하지 않는다.
    """

    name: str                 # "vllm" | "llamacpp"
    display_name: str         # "vLLM" | "llama.cpp"
    accent_color: str         # CSS color — backend 컬럼 강조용

    def rows(self, running: set[str]) -> list[DashboardRow]:
        """모든 프로필을 DashboardRow 로 변환. `running` 은 현재 실행 중인
        컨테이너 이름 집합 — 이벤트 루프 블로킹 방지를 위해 호출자가 주입."""
        ...

    def resolve_container(self, profile_name: str) -> str:
        """profile_name 의 container_name 반환 (docker logs / exec 등에 사용)."""
        ...
