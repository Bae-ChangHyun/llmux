"""DashboardRow — Dashboard 가 두 backend 를 동일한 방식으로 조회."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
