"""Dashboard screen — 프로필 목록 + 빠른 액션."""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from tui.backends.llamacpp import backend


def _status_cell(p: backend.Profile) -> str:
    if p.running:
        return "● running"
    return "○ stopped"


def _model_cell(p: backend.Profile) -> str:
    if not p.model_file:
        return "—"
    if p.downloaded:
        return f"✓ {p.model_file} ({p.model_size_gb} GB)"
    return f"✗ {p.model_file} (미다운)"


class ActionModal(ModalScreen[str]):
    """프로필 액션 선택 모달. vLLM ProfileActionScreen 과 동일한 OptionList 스타일."""

    BINDINGS = [Binding("escape", "cancel", show=False)]

    DEFAULT_CSS = """
    ActionModal {
        align: center middle;
    }
    ActionModal > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 42;
        height: auto;
    }
    ActionModal #action-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ActionModal OptionList {
        height: auto;
        max-height: 14;
    }
    """

    def __init__(self, profile: backend.Profile) -> None:
        super().__init__()
        self.profile = profile

    def compose(self) -> ComposeResult:
        p = self.profile
        running = p.running
        downloaded = p.downloaded

        if running:
            status = "[green]● running[/]"
        else:
            status = "[dim]○ stopped[/]"

        options: list[Option] = []
        if running:
            options.append(Option("■ Stop Container", id="stop"))
            options.append(Option("◉ View Logs", id="logs"))
            options.append(Option("⚡ Benchmark", id="benchmark"))
        else:
            options.append(Option("▶ Start Container", id="start"))
        options.append(Option("✎ Edit Profile", id="edit-profile"))
        options.append(Option("⚙ Edit Config", id="edit-config"))
        if not running:
            options.append(Option("✗ Delete Profile", id="delete-profile"))

        with Vertical():
            yield Static(f"{p.name}  {status}", id="action-title")
            yield OptionList(*options, id="action-list")

    @on(OptionList.OptionSelected, "#action-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_cancel(self) -> None:
        self.dismiss("")


class LogViewer(ModalScreen[None]):
    """docker logs -f 실시간 표시. RichLog 로 스크롤/자동 follow 지원."""

    BINDINGS = [
        Binding("escape,q", "close", "Close", show=True),
        Binding("f", "toggle_follow", "Follow on/off"),
    ]

    def __init__(self, container_name: str) -> None:
        super().__init__()
        self.container_name = container_name

    def compose(self) -> ComposeResult:
        with Vertical(id="log-container"):
            yield Label(
                f"  Logs — {self.container_name}   "
                "[dim](esc:close  f:auto-follow  ↑↓/PgUp/PgDn:scroll)[/dim]",
                id="log-title",
            )
            yield RichLog(
                id="log-body",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=True,
                max_lines=5000,
            )

    def on_mount(self) -> None:
        self._stream()

    @work(exclusive=True, group="logviewer-stream")
    async def _stream(self) -> None:
        log = self.query_one("#log-body", RichLog)
        try:
            async for line in backend.stream_logs(self.container_name, lines=200):
                log.write(backend.strip_ansi(line))
        except Exception as exc:  # pragma: no cover
            log.write(f"[로그 스트림 오류] {exc}")

    def action_toggle_follow(self) -> None:
        log = self.query_one("#log-body", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)
        self.notify(f"auto-follow: {'ON' if log.auto_scroll else 'OFF'}", timeout=2)

    def action_close(self) -> None:
        self.workers.cancel_group(self, "logviewer-stream")
        self.dismiss(None)


class DashboardScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        # Enter 는 DataTable 이 RowSelected 메시지로 소비 → on_data_table_row_selected 에서 처리.
        Binding("a,space", "open_actions", "Action"),
        Binding("c", "open_configs", "Configs"),
        Binding("s", "open_system", "System"),
        Binding("p", "new_profile", "New Profile"),
        Binding("N", "quick_setup", "Quick Setup (HF)"),
        Binding("question_mark", "help", "More keys"),
        # 아래는 액션 메뉴 안에도 있으므로 Footer 에서 숨김 (터미널 폭 절약)
        Binding("u", "quick_start", "Start/Switch", show=False),
        Binding("d", "quick_stop", "Stop", show=False),
        Binding("l", "quick_logs", "Logs", show=False),
        Binding("b", "quick_bench", "Benchmark", show=False),
        Binding("E", "edit_profile", "Edit Profile", show=False),
        Binding("X", "delete_profile", "Delete Profile", show=False),
    ]

    profiles: reactive[list[backend.Profile]] = reactive([], init=False)
    current_profile: reactive[str | None] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="profile-table", cursor_type="row")
        yield Static("", id="gpu-bar")
        with Horizontal(id="status-bar"):
            yield Static("", id="current-status")
            yield Static("", id="endpoint-status")
            yield Static("", id="hint-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Profile", "Model (GGUF)", "Port", "GPU", "Status")
        self.refresh_profiles()
        self._poll_gpu()
        self._gpu_timer = self.set_interval(3.0, self._poll_gpu)

    def on_screen_suspend(self) -> None:
        if getattr(self, "_gpu_timer", None) is not None:
            self._gpu_timer.pause()

    def on_screen_resume(self) -> None:
        if getattr(self, "_gpu_timer", None) is not None:
            self._gpu_timer.resume()
        self._poll_gpu()

    @work(exclusive=True, group="dashboard-gpu")
    async def _poll_gpu(self) -> None:
        gpus = await backend.get_gpu_info()
        try:
            self.query_one("#gpu-bar", Static).update(backend.format_gpu_bar(gpus))
        except Exception:
            pass

    def refresh_profiles(self) -> None:
        profiles = backend.list_profiles()
        self.profiles = profiles
        self.current_profile = backend.read_current_profile()
        table = self.query_one(DataTable)
        # 커서 위치 보존
        cursor_row = table.cursor_row
        table.clear()
        for p in profiles:
            name = p.name
            if p.is_current:
                name = f"★ {name}"
            table.add_row(
                name,
                _model_cell(p),
                str(p.port),
                p.gpu_id,
                _status_cell(p),
                key=p.name,
            )
        if profiles:
            try:
                table.move_cursor(row=min(cursor_row, len(profiles) - 1))
            except Exception:
                pass
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        current = self.current_profile
        cur = self.query_one("#current-status", Static)
        ep = self.query_one("#endpoint-status", Static)
        hint = self.query_one("#hint-status", Static)

        if current:
            match = next((p for p in self.profiles if p.name == current), None)
            if match and match.running:
                cur.update(f"[b]활성:[/b] {current}  [green]●[/green]")
                ep.update(f"[b]endpoint:[/b] {match.endpoint}")
            else:
                cur.update(f"[b]활성:[/b] {current}  [dim](중지됨)[/dim]")
                ep.update("")
        else:
            cur.update("[dim]활성 프로필 없음[/dim]")
            ep.update("")
        hint.update(
            "[dim]Enter:액션 u:시작 d:중지 l:로그 b:벤치 c:Configs s:System p/E/X:Profile N:HF-New r:↻ q:종료[/dim]"
        )

    def _selected(self) -> backend.Profile | None:
        table = self.query_one(DataTable)
        if not self.profiles:
            return None
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            key_val = row_key.value if row_key is not None else None
            if key_val is not None:
                for p in self.profiles:
                    if p.name == key_val:
                        return p
        except Exception:
            pass
        try:
            return self.profiles[table.cursor_row]
        except IndexError:
            return None

    def action_refresh(self) -> None:
        self.refresh_profiles()
        self.notify("새로고침 완료")

    def action_help(self) -> None:
        self.notify(
            "Enter/a/Space: 액션 메뉴   u: 시작/전환   d: 중지\n"
            "l: 로그   b: 벤치마크   r: 새로고침   q: 종료",
            title="키 도움말",
            timeout=8,
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter 키 이벤트 (DataTable 이 소비하는 Enter)."""
        event.stop()
        self.action_open_actions()

    def action_open_actions(self) -> None:
        p = self._selected()
        if not p:
            return

        def _on_action(result: str | None) -> None:
            if result is None:
                return
            self._dispatch_action(p, result)

        self.app.push_screen(ActionModal(p), _on_action)

    def action_quick_start(self) -> None:
        p = self._selected()
        if p and not p.running:
            self._dispatch_action(p, "start")

    def action_quick_stop(self) -> None:
        p = self._selected()
        if p and p.running:
            self._dispatch_action(p, "stop")

    def action_quick_logs(self) -> None:
        p = self._selected()
        if p and p.running:
            self._dispatch_action(p, "logs")

    def action_quick_bench(self) -> None:
        p = self._selected()
        if p and p.running:
            self._dispatch_action(p, "benchmark")

    def action_open_configs(self) -> None:
        self.app.push_screen("configs")

    def action_open_system(self) -> None:
        self.app.push_screen("system")

    def action_quick_setup(self) -> None:
        from tui.backends.llamacpp.screens.quick_setup import QuickSetupScreen

        def _after(result: str | None) -> None:
            if result:
                self.refresh_profiles()

        self.app.push_screen(QuickSetupScreen(), _after)

    def action_new_profile(self) -> None:
        from tui.backends.llamacpp.screens.profile import ProfileFormScreen

        def _after(result: str | None) -> None:
            if result:
                self.refresh_profiles()

        self.app.push_screen(ProfileFormScreen(), _after)

    def action_edit_profile(self) -> None:
        from tui.backends.llamacpp.screens.profile import ProfileFormScreen

        p = self._selected()
        if not p:
            self.notify("선택된 프로필 없음", severity="warning")
            return

        def _after(result: str | None) -> None:
            if result:
                self.refresh_profiles()

        self.app.push_screen(ProfileFormScreen(p), _after)

    def action_delete_profile(self) -> None:
        from tui.backends.llamacpp.screens.profile import ProfileDeleteScreen

        p = self._selected()
        if not p:
            self.notify("선택된 프로필 없음", severity="warning")
            return
        if p.running:
            self.notify("실행 중인 프로필은 먼저 중지 (d)", severity="warning")
            return

        def _after(result: bool) -> None:
            if result:
                self.refresh_profiles()

        self.app.push_screen(ProfileDeleteScreen(p.name), _after)

    def _dispatch_action(self, p: backend.Profile, action: str) -> None:
        if action == "start":
            self.notify(f"'{p.name}' 기동 중 (최대 수 분)...", timeout=6)
            self.run_worker(self._do_switch(p.name), exclusive=True)
        elif action == "stop":
            self.run_worker(self._do_stop(p.name), exclusive=True)
        elif action == "logs":
            self.app.push_screen(LogViewer(p.container_name))
        elif action == "pull":
            self.notify(f"'{p.name}' 모델 다운로드 (수십 GB, 시간 소요)...", timeout=8)
            self.run_worker(self._do_pull(p.name), exclusive=True)
        elif action == "benchmark":
            self.run_worker(self._do_benchmark(p), exclusive=True)
        elif action == "edit-config":
            from tui.backends.llamacpp.screens.config import ConfigFormScreen

            def _after_cfg(result: str | None) -> None:
                if result:
                    self.refresh_profiles()

            self.app.push_screen(ConfigFormScreen(p.config_name), _after_cfg)
        elif action == "edit-profile":
            from tui.backends.llamacpp.screens.profile import ProfileFormScreen

            def _after_prof(result: str | None) -> None:
                if result:
                    self.refresh_profiles()

            self.app.push_screen(ProfileFormScreen(p), _after_prof)
        elif action == "delete-profile":
            from tui.backends.llamacpp.screens.profile import ProfileDeleteScreen

            def _after_del(result: bool) -> None:
                if result:
                    self.refresh_profiles()

            self.app.push_screen(ProfileDeleteScreen(p.name), _after_del)

    async def _do_switch(self, name: str) -> None:
        code, out = await backend.run_script("switch.sh", name)
        if code == 0:
            self.notify(f"✓ '{name}' 활성화", severity="information")
        else:
            self.notify(f"✗ switch 실패 ({code})", severity="error")
        self.refresh_profiles()

    async def _do_stop(self, name: str) -> None:
        code, _ = await backend.run_script("stop.sh")
        if code == 0:
            self.notify(f"✓ '{name}' 중지")
        else:
            self.notify(f"✗ stop 실패 ({code})", severity="error")
        self.refresh_profiles()

    async def _do_pull(self, name: str) -> None:
        code, out = await backend.run_script("pull-model.sh", name)
        if code == 0:
            self.notify(f"✓ '{name}' GGUF 다운로드 완료")
        else:
            tail = out.splitlines()[-3:] if out else []
            self.notify("✗ 다운로드 실패: " + " / ".join(tail), severity="error")
        self.refresh_profiles()

    async def _do_benchmark(self, p: backend.Profile) -> None:
        cfg = backend.load_config(p.config_name)
        alias = cfg.get("alias", p.config_name)
        self.notify(f"벤치마크 실행 ({alias})...")
        try:
            r = await backend.chat_completion(
                p.port, alias, "Explain the theory of relativity in about 150 words.", 200
            )
            u = r["usage"]
            ct = u.get("completion_tokens", 0)
            elapsed = r["elapsed"]
            tps = ct / elapsed if elapsed > 0 else 0
            self.notify(
                f"✓ {ct} tok / {elapsed:.2f}s = [b]{tps:.1f} tok/s[/b]",
                title=p.name,
                timeout=10,
            )
        except Exception as exc:
            self.notify(f"✗ 벤치마크 실패: {exc}", severity="error")
