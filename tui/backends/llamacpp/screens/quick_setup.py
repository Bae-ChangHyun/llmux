"""Quick Setup — HF repo URL → GGUF 선택 → profile + config 자동 생성."""

from __future__ import annotations

import re
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    Switch,
)

from tui.backends.llamacpp.backend import (
    Config,
    Profile,
    list_config_names,
    list_hf_repo_files,
    list_profile_names,
    load_config,
    save_config,
    save_profile,
)


_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_MOE_PATTERN = re.compile(r"[Aa][0-9]+B")
_DEFAULT_OT = ".ffn_.*_exps.=CPU"


def _normalize_repo(raw: str) -> str:
    """huggingface.co URL 이면 repo 경로만 추출."""
    s = raw.strip()
    if not s:
        return ""
    s = s.rstrip("/")
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]
        if s.startswith("api/models/"):
            s = s[len("api/models/"):]
    parts = s.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


class QuickSetupScreen(ModalScreen[str]):
    """HF repo + GGUF 파일로 profile/config 자동 생성 modal."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    QuickSetupScreen { align: center middle; }
    QuickSetupScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 82;
        min-width: 60;
        height: 95%;
        max-height: 42;
        min-height: 22;
    }
    QuickSetupScreen .title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }
    QuickSetupScreen VerticalScroll { height: 1fr; min-height: 5; }
    QuickSetupScreen Label { margin-top: 1; color: $text-muted; }
    QuickSetupScreen #gguf-info {
        height: auto;
        min-height: 1;
        color: $text-muted;
        margin-top: 0;
    }
    QuickSetupScreen #moe-hint {
        height: auto;
        min-height: 1;
        color: $accent;
        margin-top: 1;
    }
    QuickSetupScreen #fetch-btn {
        width: 100%;
        margin-top: 1;
    }
    QuickSetupScreen .switch-row {
        height: 3;
        margin-top: 1;
        padding: 0;
    }
    QuickSetupScreen .switch-row Label {
        width: 1fr;
        margin-top: 1;
    }
    QuickSetupScreen .switch-row Switch {
        width: auto;
    }
    QuickSetupScreen Collapsible {
        margin-top: 1;
        border-top: solid $primary 30%;
    }
    QuickSetupScreen Collapsible CollapsibleTitle {
        color: $text;
    }
    QuickSetupScreen .section-help {
        color: $text-muted;
        margin: 0 0 1 0;
        height: auto;
    }
    QuickSetupScreen .buttons {
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_repo = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Quick Setup — HF repo → profile + config", classes="title")
            with VerticalScroll():
                # --- HF repo ---
                yield Label("HuggingFace repo (예: unsloth/Qwen3-30B-A3B-GGUF)")
                yield Input(placeholder="org/repo-GGUF 또는 URL", id="repo-input")
                yield Button("Fetch files", id="fetch-btn")
                yield Static("", id="gguf-info")

                yield Label("GGUF 파일")
                yield Select(
                    [("(repo 입력 후 Fetch)", "__none__")],
                    allow_blank=False,
                    id="gguf-select",
                )
                yield Static("", id="moe-hint")

                # --- 기본 ---
                yield Label("Profile 이름 (비우면 파일명에서 자동 생성)")
                yield Input(placeholder="auto", id="name-input")

                yield Label("Port")
                yield Input(placeholder="8080", value="8080", id="port-input")

                yield Label("GPU ID (예: 0 또는 0,1)")
                yield Input(placeholder="0", value="0", id="gpu-input")

                # --- llama.cpp 공통 파라미터 ---
                with Collapsible(
                    title="llama.cpp 공통 파라미터",
                    collapsed=False,
                    id="common-params",
                ):
                    yield Static(
                        "[dim]빈 칸은 config 에 기록되지 않고 llama.cpp 기본값을 사용합니다. "
                        "값이 있으면 config YAML 에 저장됩니다.[/dim]",
                        classes="section-help",
                    )

                    yield Label("Ctx size (컨텍스트 길이, tokens)")
                    yield Input(placeholder="32768", value="32768", id="ctx-input")

                    yield Label("N-GPU-Layers (GPU 에 올릴 레이어, 99=전체)")
                    yield Input(placeholder="99", value="99", id="ngl-input")

                    yield Label(
                        "KV cache K 정밀도 (f16 / bf16 / q8_0 / q4_0 — 낮출수록 VRAM ↓)"
                    )
                    yield Input(placeholder="bf16", value="bf16", id="ctk-input")

                    yield Label("KV cache V 정밀도")
                    yield Input(placeholder="bf16", value="bf16", id="ctv-input")

                    yield Label("Batch size (prompt eval 단위, 비우면 기본)")
                    yield Input(placeholder="기본값 사용", id="batch-input")

                    with Horizontal(classes="switch-row"):
                        yield Label("Flash Attention (속도↑, 정확도는 동일)")
                        yield Switch(value=True, id="flash-attn-switch")

                    with Horizontal(classes="switch-row"):
                        yield Label("Jinja chat template (/v1/chat/completions 에 필요)")
                        yield Switch(value=True, id="jinja-switch")

                # --- MoE Expert Offload ---
                with Collapsible(
                    title="MoE Expert Offload (선택)",
                    collapsed=True,
                    id="moe-collapsible",
                ):
                    yield Static(
                        "[dim]MoE 모델 (Qwen3-A3B, Mixtral 등) 은 전체 파라미터 중 일부만 "
                        "매 토큰 활성화됩니다. [b]Expert 가중치를 CPU RAM 에 두고 "
                        "활성 expert 만 GPU 로 스트리밍[/b] 하면 16GB GPU 에 35B MoE 도 올라갑니다.\n"
                        "예: [b]qwen3.6-35b-a3b UD-Q4_K_XL[/b] → VRAM 7.5GB + RAM 15GB 로 36 tok/s.\n\n"
                        "파일명에 'A3B', 'A7B' 패턴이 있으면 자동 감지 후 기본 정규식을 채워드립니다. "
                        "비우면 expert offload 안 함 (Dense 모델은 보통 필요 없음).[/dim]",
                        classes="section-help",
                    )
                    yield Label("override-tensors 정규식 (비우면 미적용)")
                    yield Input(
                        placeholder=_DEFAULT_OT,
                        id="ot-input",
                    )

                # --- 기존 config 복사 ---
                yield Label("기존 config 에서 추가 파라미터 복사 (선택)")
                yield Select(
                    self._build_config_options(),
                    prompt="None",
                    allow_blank=True,
                    id="copy-config-select",
                )

            with Horizontal(classes="buttons"):
                yield Button("Create", variant="primary", id="create-btn")
                yield Button("Cancel", id="cancel-btn")

    def _build_config_options(self) -> list[tuple[str, str]]:
        return [
            (f"{name} ({len(load_config(name).params)} params)", name)
            for name in list_config_names()
        ]

    # ----- Fetch GGUF files -----

    @on(Button.Pressed, "#fetch-btn")
    def _on_fetch(self) -> None:
        repo_raw = self.query_one("#repo-input", Input).value
        repo = _normalize_repo(repo_raw)
        if not repo or not _REPO_RE.match(repo):
            self.notify("유효한 HF repo 경로가 아님 (org/name)", severity="error")
            return
        self._last_repo = repo
        info = self.query_one("#gguf-info", Static)
        info.update(f"[dim]{repo} 파일 목록 가져오는 중...[/dim]")
        self._fetch_files(repo)

    @work(exclusive=True, group="hf-fetch")
    async def _fetch_files(self, repo: str) -> None:
        files = await list_hf_repo_files(repo)
        gguf_items = [
            f for f in files
            if isinstance(f, dict)
            and f.get("type") == "file"
            and str(f.get("path", "")).lower().endswith(".gguf")
        ]
        info = self.query_one("#gguf-info", Static)
        select = self.query_one("#gguf-select", Select)
        if not gguf_items:
            info.update("[red]GGUF 파일 없음 (또는 private repo — HF_TOKEN 확인)[/red]")
            select.set_options([("(없음)", "__none__")])
            return

        opts: list[tuple[str, str]] = []
        for f in sorted(gguf_items, key=lambda x: str(x.get("path", ""))):
            path = str(f.get("path", ""))
            size = f.get("size") or 0
            size_gb = size / 1024**3 if isinstance(size, (int, float)) else 0
            label = f"{path}  ({size_gb:.1f} GB)" if size_gb else path
            opts.append((label, path))
        select.set_options(opts)
        select.value = opts[0][1]
        info.update(f"[green]{len(opts)} 개 GGUF 파일[/green]")
        # 첫 파일로 MoE 감지
        self._update_moe_hint(opts[0][1])

    # ----- GGUF 선택 변화 시 MoE 감지 -----

    @on(Select.Changed, "#gguf-select")
    def _on_gguf_changed(self, event: Select.Changed) -> None:
        if event.value in (Select.BLANK, "__none__", None):
            return
        self._update_moe_hint(str(event.value))

    def _update_moe_hint(self, gguf_file: str) -> None:
        hint = self.query_one("#moe-hint", Static)
        ot_input = self.query_one("#ot-input", Input)
        moe_collapsible = self.query_one("#moe-collapsible", Collapsible)

        if _MOE_PATTERN.search(gguf_file):
            hint.update(
                f"[accent]⚠ MoE 감지[/accent]: [dim]'{gguf_file}' → "
                f"expert offload 권장 (아래 'MoE Expert Offload' 섹션 참고)[/dim]"
            )
            # 사용자가 수동으로 값을 지운 게 아니면 기본값 채움
            if not ot_input.value.strip():
                ot_input.value = _DEFAULT_OT
            moe_collapsible.collapsed = False
        else:
            hint.update(
                "[dim]Dense 모델 — expert offload 불필요 (전체 VRAM 적재)[/dim]"
            )

    # ----- Create -----

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    def _get(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        repo = _normalize_repo(self.query_one("#repo-input", Input).value)
        gguf_select = self.query_one("#gguf-select", Select)
        gguf_file = (
            str(gguf_select.value)
            if gguf_select.value not in (Select.BLANK, "__none__", None)
            else ""
        )
        name_raw = self._get("name-input")
        port_raw = self._get("port-input")
        gpu = self._get("gpu-input") or "0"
        ctx = self._get("ctx-input")
        ngl = self._get("ngl-input")
        ctk = self._get("ctk-input")
        ctv = self._get("ctv-input")
        batch = self._get("batch-input")
        ot = self._get("ot-input")
        flash_attn = self.query_one("#flash-attn-switch", Switch).value
        jinja = self.query_one("#jinja-switch", Switch).value

        # --- 필수 검증 ---
        if not repo or not _REPO_RE.match(repo):
            self.notify("유효한 HF repo 필요", severity="error")
            return
        if not gguf_file:
            self.notify("GGUF 파일 선택 필요 (Fetch 후)", severity="error")
            return
        try:
            port_num = int(port_raw or "8080")
            if not 1024 <= port_num <= 65535:
                raise ValueError
        except ValueError:
            self.notify("Port 는 1024–65535", severity="error")
            return

        # --- 이름 자동 생성 ---
        if not name_raw:
            base = repo.rsplit("/", 1)[-1]
            base = re.sub(r"[-_]?GGUF$", "", base, flags=re.I)
            name_raw = re.sub(r"[^A-Za-z0-9.-]+", "-", base).strip("-").lower()
        if not name_raw:
            self.notify("이름 생성 실패", severity="error")
            return
        # 파일명 stem 으로 직접 쓰이므로 path traversal / 특수문자 차단.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name_raw) or ".." in name_raw:
            self.notify(
                "이름 은 A-Z a-z 0-9 . _ - 만 가능 (.. 금지)",
                severity="error",
            )
            return

        # 충돌 시 suffix
        existing = set(list_profile_names()) | set(list_config_names())
        final_name = name_raw
        i = 0
        while final_name in existing:
            i += 1
            final_name = f"{name_raw}-{i}"

        # --- 추가 파라미터 복사 (베이스) ---
        params: dict[str, Any] = {}
        copy_sel = self.query_one("#copy-config-select", Select)
        if copy_sel.value and copy_sel.value != Select.BLANK:
            src = load_config(str(copy_sel.value))
            params.update(src.params)

        # --- 핵심 플래그 강제 설정 ---
        params["model-file"] = gguf_file
        params.setdefault("alias", final_name)

        # --- 선택 파라미터: 빈값이면 저장하지 않음 ---
        def _set_int(key: str, raw: str) -> None:
            if not raw:
                params.pop(key, None)
                return
            try:
                params[key] = int(raw)
            except ValueError:
                params[key] = raw

        _set_int("ctx-size", ctx)
        _set_int("n-gpu-layers", ngl)
        if ctk:
            params["cache-type-k"] = ctk
        else:
            params.pop("cache-type-k", None)
        if ctv:
            params["cache-type-v"] = ctv
        else:
            params.pop("cache-type-v", None)
        if batch:
            _set_int("batch-size", batch)
        else:
            params.pop("batch-size", None)

        # boolean 스위치는 켜져 있을 때만 저장 (꺼진 상태 = llama.cpp 기본)
        if flash_attn:
            params["flash-attn"] = True
        else:
            params.pop("flash-attn", None)
        if jinja:
            params["jinja"] = True
        else:
            params.pop("jinja", None)

        # MoE offload
        if ot:
            params["override-tensors"] = [ot]
        else:
            params.pop("override-tensors", None)

        save_config(Config(name=final_name, params=params))

        save_profile(
            Profile(
                name=final_name,
                container_name=final_name,
                port=port_num,
                gpu_id=gpu,
                config_name=final_name,
                model_file=gguf_file,
                hf_repo=repo,
                hf_file=gguf_file,
            )
        )

        self.notify(
            f"✓ 생성: {final_name}  (다음: 'u' 로 시작 — 처음이면 GGUF 자동 다운로드)",
            severity="information",
            timeout=8,
        )
        self.dismiss(final_name)
