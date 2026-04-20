"""Config 편집 화면 — YAML config (llama-server 플래그) CRUD."""

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from tui.backends.llamacpp import backend
from tui.backends.llamacpp.backend import (
    Config,
    delete_config,
    extract_llama_server_flags,
    format_config_param_value,
    list_config_names,
    list_profile_names,
    load_config,
    load_profile,
    parse_config_param_value,
    save_config,
    validate_name,
)


# ---------------------------------------------------------------------------
# llama-server 주요 플래그 참조표 (자동추출 실패/보완 시 fallback)
# key → (한줄설명, 예시값)
# ---------------------------------------------------------------------------

LLAMA_SERVER_FLAGS: dict[str, tuple[str, str]] = {
    # 필수/핵심
    "model-file":       ("GGUF 파일명 (MODEL_DIR 기준 상대경로)", "my-model-Q4_K_M.gguf"),
    "alias":            ("/v1/models 에 노출될 모델 이름",       "my-model"),
    # 컨텍스트 / 레이어
    "ctx-size":         ("컨텍스트 길이 (tokens)",                "32768"),
    "n-gpu-layers":     ("GPU 에 올릴 레이어 수 (99=전체)",       "99"),
    "n-predict":        ("기본 최대 생성 토큰",                   "-1"),
    "parallel":         ("동시 디코딩 슬롯 수",                   "1"),
    "batch-size":       ("physical batch (prompt eval)",         "2048"),
    "ubatch-size":      ("micro-batch size",                     "512"),
    "threads":          ("CPU 스레드 수",                        "8"),
    "threads-batch":    ("배치용 CPU 스레드 수",                  "8"),
    # KV 캐시
    "cache-type-k":     ("KV 캐시 K 정밀도 (f16/bf16/q8_0/q4_0)", "bf16"),
    "cache-type-v":     ("KV 캐시 V 정밀도",                     "bf16"),
    "no-kv-offload":    ("KV 캐시 GPU offload 끄기 (boolean)",    ""),
    # MoE / tensor 관리
    "override-tensors": ("tensor→device 라우팅 (-ot 리스트)",     ".ffn_.*_exps.=CPU"),
    "split-mode":       ("multi-GPU split 전략 (none/layer/row)", "layer"),
    "tensor-split":     ("GPU 간 tensor 비율 (예: 0.5,0.5)",      "0.5,0.5"),
    "main-gpu":         ("primary GPU index",                    "0"),
    # 최적화
    "flash-attn":       ("Flash Attention 켜기 (boolean)",         ""),
    "no-mmap":          ("mmap 비활성 (boolean)",                  ""),
    "mlock":            ("메모리 lock (boolean)",                  ""),
    "cont-batching":    ("continuous batching (boolean, 기본 on)", ""),
    "rope-scaling":     ("RoPE scaling type (none/linear/yarn)",  "yarn"),
    "rope-scale":       ("RoPE 스케일 factor",                    "2.0"),
    "rope-freq-base":   ("RoPE base freq override",               "10000"),
    # 샘플링 기본값
    "temp":             ("샘플링 temperature",                    "0.7"),
    "top-p":            ("top-p nucleus",                        "0.95"),
    "top-k":            ("top-k",                                "40"),
    "min-p":            ("min-p cutoff",                         "0.05"),
    "repeat-penalty":   ("repetition penalty",                   "1.1"),
    "seed":             ("RNG seed (-1=random)",                 "-1"),
    # Chat template / reasoning
    "jinja":            ("jinja chat-template 사용 (boolean)",     ""),
    "chat-template":    ("chat template 이름 오버라이드",          "chatml"),
    "chat-template-file": ("chat template 파일 경로",              "/models/tpl.jinja"),
    "reasoning-format": ("reasoning 출력 포맷 (none/deepseek)",    "deepseek"),
    "reasoning-budget": ("reasoning token budget",                "-1"),
    # 네트워크 / 보안
    "api-key":          ("API key (요청 헤더 인증)",               "sk-secret"),
    "api-key-file":     ("API key 파일 경로",                      "/run/api-keys"),
    "timeout":          ("요청 timeout 초",                        "600"),
    "metrics":          ("/metrics Prometheus 노출 (boolean)",      ""),
    "slots":            ("/slots endpoint (boolean)",              ""),
    "props":            ("/props endpoint (boolean)",              ""),
    "verbose":          ("verbose 로그 (boolean)",                 ""),
    "log-disable":      ("로그 파일 출력 끄기 (boolean)",           ""),
    # 멀티모달 / 프로젝터
    "mmproj":           ("multimodal projector GGUF 경로",         "/models/mmproj.gguf"),
    # 임베딩
    "embedding":        ("embedding 모드 (boolean)",               ""),
    "pooling":          ("embedding pooling (none/mean/cls/last)", "mean"),
    # 기타 고급
    "host":             ("bind host (docker override 에 의해 강제됨)", "0.0.0.0"),
    "port":             ("bind port (docker override 에 의해 강제됨)", "8080"),
}


# 런타임에 갱신되는 suggester
_KNOWN_FLAGS: set[str] = set(LLAMA_SERVER_FLAGS.keys())
_FLAG_SUGGESTER = SuggestFromList(sorted(_KNOWN_FLAGS), case_sensitive=False)


# ---------------------------------------------------------------------------
# ConfigFormScreen
# ---------------------------------------------------------------------------


class ConfigFormScreen(ModalScreen[str | None]):
    """config 생성/편집 modal."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfigFormScreen { align: center middle; }
    ConfigFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 90;
        min-width: 60;
        height: 95%;
        max-height: 45;
        min-height: 20;
    }
    ConfigFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row Label {
        width: 20;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ConfigFormScreen #params-title {
        margin-top: 1;
        text-style: bold;
        color: $text;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ConfigFormScreen #params-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    ConfigFormScreen #flag-help {
        color: $accent;
        margin-bottom: 1;
        padding: 0 1;
        background: $boost;
        min-height: 1;
    }
    ConfigFormScreen #add-param-row {
        height: 3;
        margin-bottom: 1;
        align-horizontal: left;
    }
    ConfigFormScreen #add-param-row Button {
        width: auto;
        min-width: 20;
    }
    ConfigFormScreen .param-row {
        height: auto;
        margin-bottom: 0;
    }
    ConfigFormScreen .param-row .param-key {
        width: 28;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-value {
        width: 1fr;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-remove {
        min-width: 5;
        width: 5;
        background: $error 20%;
        border: none;
        color: $error;
    }
    ConfigFormScreen VerticalScroll {
        height: 1fr;
        min-height: 5;
    }
    ConfigFormScreen .form-buttons {
        dock: bottom;
        height: 3;
        padding-top: 1;
        align: center middle;
        background: $surface;
    }
    """

    def __init__(self, config_name: str = "") -> None:
        super().__init__()
        self._config_name = config_name
        self._edit_mode = bool(config_name)
        self._param_counter = 0
        self._initial_config: Config | None = None
        self._saved_name: str | None = None

    def compose(self) -> ComposeResult:
        if self._edit_mode:
            self._initial_config = load_config(self._config_name)
        title = (
            f"Edit Config: {self._config_name}"
            if self._edit_mode
            else "New Config"
        )
        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")
            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label("Config Name")
                    yield Input(
                        value=self._config_name,
                        placeholder="my-model",
                        id="name-input",
                        disabled=self._edit_mode,
                    )
                yield Static("llama-server Parameters", id="params-title")
                yield Static(
                    "[dim]Tab/→: 자동완성  ·  boolean 플래그는 값 비우면 true[/dim]",
                    id="params-hint",
                )
                yield Static("", id="flag-help")
                with Horizontal(id="add-param-row"):
                    yield Button("+ Add Parameter", id="add-param-btn")
                yield Vertical(id="params-container")
            with Horizontal(classes="form-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Close", id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        if self._edit_mode and self._initial_config:
            for key, value in self._initial_config.params.items():
                self._add_param_row(key, format_config_param_value(value))
        else:
            # 새 config: 필수 핵심 플래그 3개 선제공
            for key in ("model-file", "ctx-size", "n-gpu-layers"):
                ex = LLAMA_SERVER_FLAGS.get(key, ("", ""))[1]
                self._add_param_row(key, ex)
        self._load_server_flags()

    @work(exclusive=False)
    async def _load_server_flags(self) -> None:
        global _KNOWN_FLAGS, _FLAG_SUGGESTER
        extracted = await extract_llama_server_flags()
        if extracted:
            _KNOWN_FLAGS.update(extracted)
            _FLAG_SUGGESTER = SuggestFromList(
                sorted(_KNOWN_FLAGS), case_sensitive=False
            )
            for inp in self.query(".param-key"):
                inp.suggester = _FLAG_SUGGESTER

    def _add_param_row(
        self, key: str = "", value: str = "", *, focus: bool = False
    ) -> None:
        container = self.query_one("#params-container", Vertical)
        row_id = f"param-row-{self._param_counter}"
        self._param_counter += 1
        key_input = Input(
            value=key,
            placeholder="flag-name (Tab: 자동완성)",
            suggester=_FLAG_SUGGESTER,
            classes="param-key",
        )
        row = Horizontal(
            key_input,
            Input(value=value, placeholder="value (비우면 true)", classes="param-value"),
            Button("x", classes="param-remove"),
            id=row_id,
            classes="param-row",
        )
        container.mount(row)

        def _after() -> None:
            try:
                row.scroll_visible(animate=False)
                if focus:
                    key_input.focus()
            except Exception:
                pass

        self.call_after_refresh(_after)

    @on(Input.Changed, ".param-key")
    def _on_key_changed(self, event: Input.Changed) -> None:
        """플래그 이름 변경 시 하단 헬프 영역 업데이트."""
        key = event.value.strip()
        help_widget = self.query_one("#flag-help", Static)
        info = LLAMA_SERVER_FLAGS.get(key)
        if info:
            desc, example = info
            ex_suffix = f"  [dim](예: {example})[/dim]" if example else ""
            help_widget.update(f"[b]{key}[/b]: {desc}{ex_suffix}")
        elif key in _KNOWN_FLAGS:
            help_widget.update(f"[b]{key}[/b]: [dim](llama-server --help 에 존재)[/dim]")
        else:
            help_widget.update("")

    @on(Button.Pressed, "#add-param-btn")
    def _on_add_param(self, event: Button.Pressed) -> None:
        event.stop()
        self._add_param_row(focus=True)
        self.notify("파라미터 추가됨 — flag 이름 입력 (Tab 자동완성)", timeout=3)

    @on(Button.Pressed, ".param-remove")
    def _on_remove_param(self, event: Button.Pressed) -> None:
        widget = event.button.parent
        while widget is not None:
            if hasattr(widget, "classes") and "param-row" in widget.classes:
                widget.remove()
                return
            widget = widget.parent

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()

        if not name:
            self.notify("Config 이름 필수", severity="error")
            return
        if not validate_name(name):
            self.notify(
                "이름은 영숫자/대시/언더스코어만 가능 ('-' 시작 금지)", severity="error"
            )
            return
        if not self._edit_mode and name in list_config_names():
            self.notify(f"Config '{name}' 이미 존재", severity="error")
            return

        params: dict[str, Any] = {}
        seen: set[str] = set()
        for row in self.query(".param-row"):
            key = row.query_one(".param-key", Input).value.strip()
            val = row.query_one(".param-value", Input).value.strip()
            if not key:
                continue
            if key in seen:
                self.notify(f"중복 플래그: {key}", severity="error")
                return
            seen.add(key)
            try:
                params[key] = parse_config_param_value(val)
            except Exception as exc:
                self.notify(f"'{key}' 값 파싱 실패: {exc}", severity="error")
                return

        unknown = [k for k in params if k not in _KNOWN_FLAGS]
        if unknown:
            self.notify(
                f"알 수 없는 플래그 (llama-server 버전에 따라 유효할 수 있음): {', '.join(unknown)}",
                severity="warning",
                timeout=6,
            )

        cfg = Config(name=name, params=params)
        save_config(cfg)
        self.notify(f"저장: {name}", severity="information")
        self._saved_name = name

        if not self._edit_mode:
            self._edit_mode = True
            self._config_name = name
            self.query_one("#name-input", Input).disabled = True
            self.query_one("#form-title", Static).update(
                f"[b]Edit Config: {name}[/b]"
            )

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)


# ---------------------------------------------------------------------------
# ConfirmDeleteConfigScreen
# ---------------------------------------------------------------------------


class ConfirmDeleteConfigScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfirmDeleteConfigScreen { align: center middle; }
    ConfirmDeleteConfigScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    ConfirmDeleteConfigScreen #confirm-title {
        text-style: bold;
        color: $error;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-msg,
    ConfirmDeleteConfigScreen #confirm-warn {
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-warn { color: $warning; }
    ConfirmDeleteConfigScreen .confirm-buttons {
        height: auto;
        align: center middle;
    }
    ConfirmDeleteConfigScreen .confirm-buttons Button { margin: 0 1; }
    """

    def __init__(self, config_name: str, referencing_profiles: list[str]) -> None:
        super().__init__()
        self._config_name = config_name
        self._referencing = referencing_profiles

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Delete Config", id="confirm-title")
            yield Static(
                f"[b]{self._config_name}[/b] 를 삭제할까요?",
                id="confirm-msg",
            )
            if self._referencing:
                yield Static(
                    f"[b]경고:[/b] 사용 중인 프로필: {', '.join(self._referencing)}",
                    id="confirm-warn",
                )
            with Horizontal(classes="confirm-buttons"):
                yield Button("Delete", id="confirm-yes", variant="error")
                yield Button("Cancel", id="confirm-no", variant="default")

    @on(Button.Pressed, "#confirm-yes")
    def _on_yes(self, event: Button.Pressed) -> None:
        delete_config(self._config_name)
        # 참조 프로필의 CONFIG_NAME 을 빈값으로 (단순 clear)
        from tui.backends.llamacpp.backend import save_profile
        for profile_name in self._referencing:
            p = load_profile(profile_name)
            p.config_name = ""
            save_profile(p)
        self.app.notify(f"삭제됨: {self._config_name}")
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _on_no(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# ConfigListScreen
# ---------------------------------------------------------------------------


class ConfigListScreen(Screen):
    """config/*.yaml 전체 목록."""

    BINDINGS = [
        Binding("n", "new_config", "New"),
        Binding("e,enter", "edit_config", "Edit"),
        Binding("delete,x", "delete_config", "Delete"),
        Binding("escape,backspace", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[b]Configs (llama-server YAML)[/b]", id="config-title")
        yield DataTable(id="config-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.add_columns("Name", "Model File", "Ctx", "N-GPU-Layers", "Params")
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        for name in list_config_names():
            cfg = load_config(name)
            model_file = str(cfg.params.get("model-file", "—"))
            ctx = str(cfg.params.get("ctx-size", "—"))
            ngl = str(cfg.params.get("n-gpu-layers", "—"))
            table.add_row(
                cfg.name, model_file, ctx, ngl, str(len(cfg.params)), key=cfg.name
            )

    def _get_selected(self) -> str | None:
        table = self.query_one("#config-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            return row_key.value if row_key else None
        except Exception:
            return None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_edit_config()

    def action_new_config(self) -> None:
        self.app.push_screen(ConfigFormScreen(), self._on_form_closed)

    def action_edit_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify("선택된 config 없음", severity="warning")
            return
        self.app.push_screen(ConfigFormScreen(name), self._on_form_closed)

    def action_delete_config(self) -> None:
        name = self._get_selected()
        if not name:
            self.notify("선택된 config 없음", severity="warning")
            return
        referencing = [
            p for p in list_profile_names() if load_profile(p).config_name == name
        ]
        self.app.push_screen(
            ConfirmDeleteConfigScreen(name, referencing), self._on_delete_confirmed
        )

    def action_refresh(self) -> None:
        self._refresh_table()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _on_form_closed(self, result: str | None = None) -> None:
        self._refresh_table()

    def _on_delete_confirmed(self, result: bool) -> None:
        if result:
            self._refresh_table()
