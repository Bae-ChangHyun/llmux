from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DashboardRow:
    backend: str
    profile_name: str
    container_name: str
    port: int | None
    running: bool
    model: str
    detail: str
    gpu_id: str = ""
    raw: Any = None
