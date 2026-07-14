#!/usr/bin/env python3
"""
render-override.py — profile + config 를 읽어 docker-compose.override.yaml 을 생성.

입력:
  - profiles.yaml 의 llamacpp/<profile> (CONFIG_NAME, MODEL_FILE 등)
  - config/llamacpp/<CONFIG_NAME>.yaml  (llama-server 플래그)

출력:
  - docker-compose.override.yaml  (services.llama-server.command 블록)

사용:
  python3 scripts/llamacpp/render-override.py <profile-name>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "llamacpp"
COMPOSE_DIR = ROOT / "compose" / "llamacpp"
RUNTIME_DIR = ROOT / ".runtime" / "llamacpp"

sys.path.insert(0, str(ROOT))
from tui.common import profile_store  # noqa: E402

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_name(name: str, kind: str) -> str:
    """filename stem 과 compose project name 에 안전한 값만 허용."""
    if not name or not _SAFE_NAME.match(name) or ".." in name:
        raise SystemExit(
            f"잘못된 {kind} 이름: {name!r} (허용: 소문자, 숫자, _, -)"
        )
    return name


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        # 따옴표 제거
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    return env


def render_command(
    cfg: dict,
    *,
    model_file: str = "",
    hf_repo: str = "",
    hf_file: str = "",
) -> list[str]:
    """config yaml + profile → llama-server CLI 인자 리스트.

    Model source resolution mirrors the vllm flow — `-hf <repo> -hff <file>`
    so llama-server downloads into the HF cache mounted from the host.
    """
    # `--metrics` is forced on so the dashboard's live tok/s poll has a
    # /metrics endpoint to read (llama-server does not expose it by default).
    args: list[str] = [
        "--host", "0.0.0.0", "--port", "8080", "--no-webui", "--metrics",
    ]

    # 모델 식별: profile 의 hf_repo/hf_file 이 1순위. config 의 model-file 은
    # display 용 fallback (예: HF cache 에 이미 받아둔 파일명을 의도).
    resolved_file = (hf_file or str(cfg.pop("model-file", "") or model_file)).strip()
    resolved_repo = hf_repo.strip()
    if not resolved_repo:
        raise ValueError(
            "hf_repo is empty — set profile.hf_repo so llama-server can `-hf` download"
        )
    if not resolved_file:
        raise ValueError("hf_file / model_file is empty — need a specific .gguf filename")
    args.extend(["-hf", resolved_repo])
    args.extend(["-hff", resolved_file])
    # 옛 model-file 키가 cfg 에 남아 있으면 무시 (위에서 이미 pop 했지만 안전).
    cfg.pop("model-file", None)

    # WebUI 는 강제 off. 사용자가 webui 활성화하려 해도 무시.
    cfg.pop("no-webui", None)
    cfg.pop("webui", None)
    # --metrics 는 위에서 이미 강제 주입 — config 에 남아 있으면 중복 플래그가 된다.
    cfg.pop("metrics", None)

    override_tensors = cfg.pop("override-tensors", None) or []
    extra_args = cfg.pop("extra-args", None) or []

    # Modern llama-server requires an explicit value for --flash-attn
    # (`on` / `off` / `auto`); bare `--flash-attn` consumes the next CLI arg
    # as the value and dies with "unknown value for --flash-attn: '--jinja'".
    # Keep this as a narrow whitelist — other boolean keys (--jinja,
    # --cont-batching, --metrics, --no-mmap, --mlock, …) are still bare flags.
    _BOOL_VALUE_FLAGS = {"flash-attn"}

    for key, value in cfg.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in _BOOL_VALUE_FLAGS:
                args.extend([flag, "on" if value else "off"])
            elif value:
                args.append(flag)
            # value-less bool: False 는 무시
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
        elif value is None:
            continue
        else:
            args.extend([flag, str(value)])

    for pattern in override_tensors:
        args.extend(["-ot", str(pattern)])

    args.extend(str(a) for a in extra_args)
    return args


def main() -> int:
    if len(sys.argv) != 2:
        print(f"사용법: {sys.argv[0]} <profile-name>", file=sys.stderr)
        return 2

    profile = _validate_name(sys.argv[1], "profile")
    stored = profile_store.load_profile(profile, "llamacpp")
    if stored is None:
        print(f"프로필 없음: llamacpp/{profile} (profiles.yaml 확인)", file=sys.stderr)
        return 1

    config_name = _validate_name(stored.config_name or profile, "config")
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"config 파일 없음: {config_path}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(cfg, dict):
        print(f"config 형식 오류: mapping YAML 이 필요합니다: {config_path}", file=sys.stderr)
        return 1
    try:
        command = render_command(
            cfg,
            model_file=stored.model_file,
            hf_repo=stored.hf_repo,
            hf_file=stored.hf_file,
        )
    except ValueError as exc:
        print(f"config 렌더 실패: {exc}", file=sys.stderr)
        return 1

    override = {
        "services": {
            "llama-server": {
                "command": command,
            },
        },
    }

    # Per-profile override file so multiple llamacpp profiles can run
    # concurrently without clobbering each other's command.
    out_path = RUNTIME_DIR / f"override-{profile}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AUTO-GENERATED by scripts/llamacpp/render-override.py — 직접 수정 금지.\n"
        f"# Profile: {profile}  Config: {config_name}\n"
    )
    out_path.write_text(header + yaml.safe_dump(override, sort_keys=False, width=200))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
