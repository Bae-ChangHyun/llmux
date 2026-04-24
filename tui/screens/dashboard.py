"""Unified Dashboard — vLLM + llama.cpp 프로필을 한 DataTable 에 통합."""

from __future__ import annotations

import re

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from tui.backends.llamacpp import backend as lbackend
from tui.backends.llamacpp.adapter import LlamacppAdapter
from tui.backends.vllm import backend as vbackend
from tui.backends.vllm.adapter import VllmAdapter
from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.conflicts import (
    external_port_conflicts,
    gpu_conflicts,
    port_conflicts,
)
from tui.common.http import chat_completion_bench, list_served_models
from tui.common.mem import estimate_model_memory
from tui.common.widgets import BackendPickerModal, ConfirmModal


# ---------------------------------------------------------------------------
# UnifiedDashboard
# ---------------------------------------------------------------------------


class DashboardScreen(Screen):
    """두 backend 프로필을 단일 DataTable 로 통합 표시."""

    BINDINGS = [
        # 시각적 Footer
        Binding("enter", "action_menu", "Action"),
        Binding("n", "new_profile", "New"),
        Binding("m", "mem_estimate", "Memory"),
        Binding("s", "system_info", "System"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        # Power-user: footer 에서 숨김
        Binding("u", "start_container", show=False),
        Binding("d", "stop_container", show=False),
        Binding("l", "view_logs", show=False),
        Binding("e", "edit_profile", show=False),
        Binding("c", "edit_config", show=False),
        Binding("x", "delete_profile", show=False),
        Binding("question_mark", "help", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._vllm = VllmAdapter()
        self._llamacpp = LlamacppAdapter()
        self._rows: list[DashboardRow] = []
        self._gpus = []
        self._refresh_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status-bar")
        yield DataTable(id="profile-table", cursor_type="row")
        yield Static(
            "\n  No profiles yet — press [b]n[/b] to create one\n",
            id="empty-state",
        )
        yield Static("", id="gpu-bar")
        with Horizontal(id="mem-search-area"):
            yield Static(" 🔍 ", id="search-icon")
            yield Input(
                placeholder="Estimate HF model memory (press m then type, Enter to run)",
                id="mem-search-input",
            )
        yield Static("", id="mem-result-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#profile-table", DataTable)
        table.add_columns("Backend", "Profile", "Status", "Port", "Model", "Detail")
        self._reload()
        self._refresh_timer = self.set_interval(5.0, lambda: self._reload())
        self._poll_gpu()

    def on_screen_suspend(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.pause()

    def on_screen_resume(self) -> None:
        self._reload()
        if self._refresh_timer is not None:
            self._refresh_timer.resume()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    @work(exclusive=True, group="dashboard-reload")
    async def _reload(self) -> None:
        """모든 backend 프로필 재스캔. `docker ps` 는 단 한 번만 호출해 두 adapter 에 주입.

        Adapter 하나가 실패해도 다른 쪽 상태가 유지되도록 예외를 분리한다."""
        try:
            running = await common_docker.running_container_names()
        except Exception:
            running = set()

        rows: list[DashboardRow] = []
        try:
            rows.extend(self._vllm.rows(running))
        except Exception as exc:
            self.notify(f"vLLM scan failed: {exc}", severity="error")
        try:
            rows.extend(self._llamacpp.rows(running))
        except Exception as exc:
            self.notify(f"llama.cpp scan failed: {exc}", severity="error")
        rows.sort(key=lambda r: (r.backend, r.profile_name))
        self._rows = rows
        self._render_rows(rows)

    def _render_rows(self, rows: list[DashboardRow]) -> None:
        table = self.query_one("#profile-table", DataTable)
        empty = self.query_one("#empty-state")
        status_bar = self.query_one("#status-bar", Static)

        prev_key: str | None = None
        if table.row_count > 0:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                prev_key = str(row_key.value)
            except (KeyError, IndexError):
                pass

        table.clear()

        if not rows:
            table.styles.display = "none"
            empty.styles.display = "block"
            status_bar.update("")
            return

        table.styles.display = "block"
        empty.styles.display = "none"

        v_run = sum(1 for r in rows if r.backend == "vllm" and r.running)
        l_run = sum(1 for r in rows if r.backend == "llamacpp" and r.running)
        v_total = sum(1 for r in rows if r.backend == "vllm")
        l_total = sum(1 for r in rows if r.backend == "llamacpp")
        status_bar.update(
            f" [magenta]vLLM[/] {v_run}/{v_total}  ·  "
            f"[green]llama.cpp[/] {l_run}/{l_total}  ·  "
            "[dim]Enter = actions[/dim]"
        )

        for r in rows:
            backend_cell = (
                "[magenta]vLLM[/]" if r.backend == "vllm" else "[green]llama.cpp[/]"
            )
            if r.running:
                status_cell = "[green]● running[/]"
            else:
                status_cell = "[dim]○ stopped[/]"
            port_cell = str(r.port) if r.port is not None else "—"
            model_short = r.model.split("/")[-1] if "/" in r.model else (r.model or "—")
            detail = r.detail or "—"
            table.add_row(
                backend_cell,
                r.profile_name,
                status_cell,
                port_cell,
                model_short,
                detail,
                key=f"{r.backend}:{r.profile_name}",
            )

        if prev_key is not None:
            for idx, r in enumerate(rows):
                if f"{r.backend}:{r.profile_name}" == prev_key:
                    try:
                        table.move_cursor(row=idx)
                    except Exception:
                        pass
                    break

    @work(exclusive=True, group="dashboard-gpu")
    async def _poll_gpu(self) -> None:
        self._gpus = await common_docker.get_gpu_info()
        try:
            self.query_one("#gpu-bar", Static).update(
                common_docker.format_gpu_bar(self._gpus)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Row selection helpers
    # ------------------------------------------------------------------

    def _selected_row(self) -> DashboardRow | None:
        table = self.query_one("#profile-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            key_val = str(row_key.value) if row_key is not None else None
        except (KeyError, IndexError):
            return None
        if not key_val:
            return None
        for r in self._rows:
            if f"{r.backend}:{r.profile_name}" == key_val:
                return r
        return None

    # ------------------------------------------------------------------
    # Action dispatch (Enter or explicit key)
    # ------------------------------------------------------------------

    @on(DataTable.RowSelected, "#profile-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_action_menu()

    def action_action_menu(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._push_vllm_action(row)
        else:
            self._push_llamacpp_action(row)

    def _push_vllm_action(self, row: DashboardRow) -> None:
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        def after(action: str | None) -> None:
            if action:
                self._dispatch_vllm(action, row)

        self.app.push_screen(ProfileActionScreen(row.profile_name, row.running), after)

    def _push_llamacpp_action(self, row: DashboardRow) -> None:
        from tui.backends.llamacpp.screens.dashboard import ActionModal

        profile = lbackend.load_profile(row.profile_name)
        profile.running = row.running

        def after(action: str | None) -> None:
            if action:
                self._dispatch_llamacpp(action, row, profile)

        self.app.push_screen(ActionModal(profile), after)

    # ------------------------------------------------------------------
    # Cross-backend conflict gate (port + GPU, 양 backend 교차)
    # ------------------------------------------------------------------

    def _confirm_conflicts_before_start(self, row: DashboardRow, on_ok) -> None:
        """start 실행 전 다른 backend 포함 port/gpu 충돌 체크 (비동기 external 감지 포함)."""
        self._check_and_confirm(row, on_ok)

    @work(exclusive=False, group="conflict-check")
    async def _check_and_confirm(self, row: DashboardRow, on_ok) -> None:
        port_msgs = port_conflicts(row, self._rows)
        gpu_msgs = gpu_conflicts(row, self._rows)
        probe_msgs: list[str] = []
        try:
            ext_ports = await common_docker.running_container_ports()
        except Exception as exc:
            ext_ports = {}
            probe_msgs.append(
                "Could not inspect running Docker container ports. "
                f"Runtime port check will still run before start. ({exc})"
            )
        ext_msgs = external_port_conflicts(row, self._rows, ext_ports)

        if not port_msgs and not gpu_msgs and not ext_msgs and not probe_msgs:
            on_ok()
            return

        lines: list[str] = []
        if probe_msgs:
            lines.append("[b]Port probe warning:[/b]")
            lines += [f"  • {m}" for m in probe_msgs]
        if port_msgs:
            if lines:
                lines.append("")
            lines.append("[b]Port conflict (llmux):[/b]")
            lines += [f"  • {m}" for m in port_msgs]
        if ext_msgs:
            if lines:
                lines.append("")
            lines.append("[b]Port conflict (external):[/b]")
            lines += [f"  • {m}" for m in ext_msgs]
        if gpu_msgs:
            if lines:
                lines.append("")
            lines.append("[b]GPU conflict:[/b]")
            lines += [f"  • {m}" for m in gpu_msgs]
        lines.append("")
        lines.append("Proceed anyway?")
        message = "\n".join(lines)

        def after(proceed: bool) -> None:
            if proceed:
                on_ok()

        self.app.push_screen(
            ConfirmModal(message, confirm_label="Start anyway", variant="warning"),
            after,
        )

    # ----- vLLM dispatch -----

    def _dispatch_vllm(self, action: str, row: DashboardRow) -> None:
        name = row.profile_name
        if action == "start":
            from tui.backends.vllm.screens.container import ContainerUpScreen

            def launch() -> None:
                self.app.push_screen(
                    ContainerUpScreen(name),
                    self._after_mutation,
                )

            self._confirm_conflicts_before_start(row, launch)
            return
        elif action == "stop":
            self._confirm_vllm_stop(name)
        elif action == "logs":
            from tui.backends.vllm.screens.container import LogScreen

            p = vbackend.load_profile(name)
            self.app.push_screen(LogScreen(p.container_name))
        elif action == "benchmark":
            self._run_vllm_bench(row)
        elif action == "edit_profile":
            from tui.backends.vllm.screens.profile import ProfileFormScreen

            p = vbackend.load_profile(name)
            self.app.push_screen(ProfileFormScreen(p), self._after_mutation)
        elif action == "edit_config":
            from tui.backends.vllm.screens.config import ConfigFormScreen

            p = vbackend.load_profile(name)
            cfg = p.config_name or name
            self.app.push_screen(
                ConfigFormScreen(config_name=cfg), self._after_mutation
            )
        elif action == "delete":
            if row.running:
                self.notify(
                    "Cannot delete: container is running. Stop it first.",
                    severity="error",
                )
                return
            from tui.backends.vllm.screens.profile import ProfileDeleteScreen

            self.app.push_screen(ProfileDeleteScreen(name), self._after_mutation)

    def _confirm_vllm_stop(self, name: str) -> None:
        def on_ok(ok: bool) -> None:
            if ok:
                self._run_vllm_stop(name)

        self.app.push_screen(
            ConfirmModal(
                f"Stop vLLM container [b]{name}[/b]?",
                confirm_label="Yes, stop",
            ),
            on_ok,
        )

    @work(exclusive=True)
    async def _run_vllm_bench(self, row: DashboardRow) -> None:
        """vLLM /v1/chat/completions 한 번 쏴서 tok/s 측정."""
        if not row.port:
            self.notify("포트 정보 없음", severity="error")
            return
        models = await list_served_models(row.port)
        model = models[0] if models else (row.model or "")
        if not model:
            self.notify("서빙 모델 식별 실패 (/v1/models 응답 없음)", severity="error")
            return
        self.notify(f"벤치마크 실행 ({model})...")
        try:
            r = await chat_completion_bench(row.port, model)
            u = r["usage"]
            ct = u.get("completion_tokens", 0)
            elapsed = r["elapsed"]
            tps = ct / elapsed if elapsed > 0 else 0
            self.notify(
                f"✓ {ct} tok / {elapsed:.2f}s = [b]{tps:.1f} tok/s[/b]",
                title=row.profile_name,
                timeout=10,
            )
        except Exception as exc:
            self.notify(f"✗ 벤치마크 실패: {exc}", severity="error")

    @work(exclusive=False)
    async def _run_vllm_stop(self, name: str) -> None:
        self.notify(f"Stopping {name}...")
        rc, output = await vbackend.container_down(name)
        if rc == 0:
            self.notify(f"Stopped {name}.")
        else:
            self.notify(f"Error stopping {name}: {output}", severity="error")
        self._reload()

    # ----- llama.cpp dispatch -----

    def _dispatch_llamacpp(
        self, action: str, row: DashboardRow, profile
    ) -> None:
        name = row.profile_name
        if action == "start":
            from tui.backends.llamacpp.screens.dashboard import StartScreen

            def launch() -> None:
                self.app.push_screen(StartScreen(name), self._after_mutation)

            self._confirm_conflicts_before_start(row, launch)
            return
        elif action == "stop":
            self._confirm_llamacpp_stop(name)
        elif action == "logs":
            from tui.backends.llamacpp.screens.dashboard import LogViewer

            self.app.push_screen(LogViewer(profile.container_name))
        elif action == "benchmark":
            self._run_llamacpp_bench(profile)
        elif action == "edit-config":
            from tui.backends.llamacpp.screens.config import ConfigFormScreen

            self.app.push_screen(
                ConfigFormScreen(profile.config_name), self._after_mutation
            )
        elif action == "edit-profile":
            from tui.backends.llamacpp.screens.profile import ProfileFormScreen

            self.app.push_screen(ProfileFormScreen(profile), self._after_mutation)
        elif action == "delete-profile":
            from tui.backends.llamacpp.screens.profile import ProfileDeleteScreen

            self.app.push_screen(ProfileDeleteScreen(name), self._after_mutation)

    @work(exclusive=True)
    async def _run_llamacpp_switch(self, name: str) -> None:
        code, out = await lbackend.run_script("switch.sh", name)
        if code == 0:
            self.notify(f"✓ '{name}' 활성화")
        else:
            tail = out.splitlines()[-3:] if out else []
            msg = " / ".join(tail) if tail else f"code={code}"
            self.notify(f"✗ switch 실패: {msg}", severity="error")
        self._reload()

    def _confirm_llamacpp_stop(self, name: str) -> None:
        def on_ok(ok: bool) -> None:
            if ok:
                self._run_llamacpp_stop(name)

        self.app.push_screen(
            ConfirmModal(
                f"Stop llama.cpp container [b]{name}[/b]?",
                confirm_label="Yes, stop",
            ),
            on_ok,
        )

    @work(exclusive=False)
    async def _run_llamacpp_stop(self, name: str) -> None:
        self.notify(f"Stopping {name}...")
        code, out = await lbackend.run_script("stop.sh", name)
        if code == 0:
            self.notify(f"✓ '{name}' 중지")
        else:
            tail = out.splitlines()[-3:] if out else []
            msg = " / ".join(tail) if tail else f"code={code}"
            self.notify(f"✗ stop 실패: {msg}", severity="error")
        self._reload()

    @work(exclusive=True)
    async def _run_llamacpp_bench(self, profile) -> None:
        cfg = lbackend.load_config(profile.config_name)
        alias = cfg.get("alias", profile.config_name)
        self.notify(f"벤치마크 실행 ({alias})...")
        try:
            r = await lbackend.chat_completion(
                profile.port,
                alias,
                "Explain the theory of relativity in about 150 words.",
                200,
            )
            u = r["usage"]
            ct = u.get("completion_tokens", 0)
            elapsed = r["elapsed"]
            tps = ct / elapsed if elapsed > 0 else 0
            self.notify(
                f"✓ {ct} tok / {elapsed:.2f}s = [b]{tps:.1f} tok/s[/b]",
                title=profile.name,
                timeout=10,
            )
        except Exception as exc:
            self.notify(f"✗ 벤치마크 실패: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Quick shortcuts (u / d / l / e / c / x)
    # ------------------------------------------------------------------

    def action_start_container(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.running:
            self.notify("Container already running.", severity="warning", timeout=3)
            return
        if row.backend == "vllm":
            self._dispatch_vllm("start", row)
        else:
            self._dispatch_llamacpp(
                "start", row, lbackend.load_profile(row.profile_name)
            )

    def action_stop_container(self) -> None:
        row = self._selected_row()
        if row is None or not row.running:
            return
        if row.backend == "vllm":
            self._confirm_vllm_stop(row.profile_name)
        else:
            self._confirm_llamacpp_stop(row.profile_name)

    def action_view_logs(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if not row.running:
            self.notify("Logs are available only for running containers.", severity="warning")
            return
        if row.backend == "vllm":
            self._dispatch_vllm("logs", row)
        else:
            self._dispatch_llamacpp(
                "logs", row, lbackend.load_profile(row.profile_name)
            )

    def action_edit_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._dispatch_vllm("edit_profile", row)
        else:
            self._dispatch_llamacpp(
                "edit-profile", row, lbackend.load_profile(row.profile_name)
            )

    def action_edit_config(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.backend == "vllm":
            self._dispatch_vllm("edit_config", row)
        else:
            self._dispatch_llamacpp(
                "edit-config", row, lbackend.load_profile(row.profile_name)
            )

    def action_delete_profile(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.running:
            self.notify(
                "Cannot delete: container running. Stop first.", severity="error"
            )
            return
        if row.backend == "vllm":
            self._dispatch_vllm("delete", row)
        else:
            self._dispatch_llamacpp(
                "delete-profile", row, lbackend.load_profile(row.profile_name)
            )

    # ------------------------------------------------------------------
    # New profile / system / help / refresh
    # ------------------------------------------------------------------

    def action_refresh(self) -> None:
        self._reload()

    def action_new_profile(self) -> None:
        def after(backend_name: str) -> None:
            if backend_name == "vllm":
                from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

                self.app.push_screen(QuickSetupScreen(), self._after_mutation)
            elif backend_name == "llamacpp":
                from tui.backends.llamacpp.screens.quick_setup import QuickSetupScreen

                self.app.push_screen(QuickSetupScreen(), self._after_mutation)

        self.app.push_screen(BackendPickerModal(), after)

    def action_system_info(self) -> None:
        """현재 커서 위치의 backend 에 맞는 System 화면으로 이동.
        선택된 row 가 없으면 backend picker 로 사용자 선택."""
        row = self._selected_row()
        if row is not None:
            screen_id = "vllm_system" if row.backend == "vllm" else "llamacpp_system"
            self.app.push_screen(screen_id)
            return

        def after(backend_name: str) -> None:
            if backend_name == "vllm":
                self.app.push_screen("vllm_system")
            elif backend_name == "llamacpp":
                self.app.push_screen("llamacpp_system")

        self.app.push_screen(BackendPickerModal(), after)

    def action_help(self) -> None:
        self.notify(
            "[b]Dashboard[/b]\n"
            "  Enter   action menu\n"
            "  u/d/l   start/stop/logs\n"
            "  e/c/x   edit profile/config, delete\n"
            "  n s r q new/system/refresh/quit",
            title="Keys",
            timeout=10,
        )

    def _after_mutation(self, result: object = None) -> None:
        self._reload()

    # ------------------------------------------------------------------
    # HF model memory estimation (common feature on main dashboard)
    # ------------------------------------------------------------------

    def action_mem_estimate(self) -> None:
        self.query_one("#mem-search-input", Input).focus()

    @on(Input.Submitted, "#mem-search-input")
    def _on_mem_search(self, event: Input.Submitted) -> None:
        model = event.value.strip()
        if model:
            self._do_mem_estimate(model)

    @work(exclusive=True, group="mem-estimate")
    async def _do_mem_estimate(self, model_id: str) -> None:
        try:
            result_bar = self.query_one("#mem-result-bar", Static)
            result_bar.update(f"  [dim]⏳ Estimating {model_id}...[/dim]")
        except Exception:
            return

        result = await estimate_model_memory(model_id)

        match = re.search(r"~([\d.]+)GB", result)
        est_gb = float(match.group(1)) if match else 0

        try:
            model_short = model_id.split("/")[-1] if "/" in model_id else model_id
            if est_gb > 0 and self._gpus:
                n_gpus = len(self._gpus)
                per_gpu_gb = est_gb / n_gpus if n_gpus > 1 else est_gb
                tp_note = (
                    f"  [dim]TP={n_gpus}: {per_gpu_gb:.1f}GB/GPU[/dim]"
                    if n_gpus > 1
                    else ""
                )

                parts = []
                for g in self._gpus:
                    total_gb = int(g.memory_total) / 1024
                    ratio = per_gpu_gb / total_gb if total_gb > 0 else 0
                    bar_w = 12
                    if ratio > 1.0:
                        bar = f"[red bold]{'✗' * bar_w}[/red bold]"
                        label = (
                            f"[red bold]OVER[/red bold] "
                            f"{per_gpu_gb:.1f}/{total_gb:.0f}GB"
                        )
                    else:
                        filled = round(ratio * bar_w)
                        empty = bar_w - filled
                        color = (
                            "green"
                            if ratio < 0.7
                            else ("yellow" if ratio < 0.9 else "red")
                        )
                        bar = (
                            f"[{color}]{'━' * filled}[/{color}]"
                            f"[dim]{'╌' * empty}[/dim]"
                        )
                        label = (
                            f"[{color}]{ratio * 100:.0f}%[/{color}] "
                            f"{per_gpu_gb:.1f}/{total_gb:.0f}GB"
                        )
                    parts.append(f"GPU{g.index} {bar} {label}")
                gpu_line = " [dim]│[/dim] ".join(parts)
                result_bar.update(
                    f"  📦 [bold]{model_short}[/bold] {result}{tp_note}\n"
                    f"     {gpu_line}"
                )
            else:
                result_bar.update(f"  📦 [bold]{model_short}[/bold]  {result}")
        except Exception:
            pass
