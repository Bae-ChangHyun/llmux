from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Static,
    TabbedContent,
    TabPane,
    RichLog,
    Button,
    Header,
    Footer,
)
from textual import work

from tui.backends.vllm.backend import (
    COMMON_ENV,
    get_gpu_info,
    GpuInfo,
    get_docker_images,
    get_dev_images,
    DockerImage,
    list_profile_names,
    load_profile,
    get_dev_build_defaults,
)
from tui.common import profile_store, system_operations
from tui.common.dev_build import image_tag_error
from tui.common.docker import (
    GpuReadingLevel,
    container_snapshots,
    get_disk_usage,
    gpu_temperature_level,
    gpu_utilization_level,
    parse_gpu_reading,
)
from tui.common.env import host_expand, parse_env_file
from tui.common.i18n import t
from tui.common.widgets import ConfirmModal, DevBuildPromptModal, TextPromptModal


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while True:
        try:
            candidate.stat()
            return candidate
        except FileNotFoundError:
            parent = candidate.parent
            if parent == candidate:
                raise
            candidate = parent


class SystemScreen(Screen):
    BINDINGS = [
        Binding("escape,backspace,s", "go_back", t("Back", "뒤로"), show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "refresh_all", t("Refresh All", "전체 새로고침"), show=True),
    ]

    DEFAULT_CSS = """
    SystemScreen {
        layout: vertical;
    }
    #refresh-bar {
        height: auto;
        dock: bottom;
        padding: 0 2;
        align: right middle;
    }
    .section-title {
        text-style: bold;
        color: $primary;
        margin: 1 0 0 1;
    }
    #images-scroll {
        height: 1fr;
    }
    #images-scroll DataTable {
        max-height: 12;
    }
    #image-action-log {
        height: 8;
    }
    #environment-actions {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gpu_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane(t("GPU Status", "GPU 상태"), id="gpu-tab"):
                yield DataTable(id="gpu-table")
            with TabPane(t("Docker Images", "Docker 이미지"), id="images-tab"):
                with VerticalScroll(id="images-scroll"):
                    yield Static(t("Official Images (vllm/vllm-openai)", "공식 이미지 (vllm/vllm-openai)"), classes="section-title")
                    yield DataTable(id="official-images")
                    yield Static(t("Dev Images (vllm-dev)", "개발 이미지 (vllm-dev)"), classes="section-title")
                    yield DataTable(id="dev-images")
                yield Horizontal(
                    Button(t("Refresh Images", "이미지 새로고침"), id="btn-refresh-images", variant="primary"),
                    Button(t("Pull", "받기"), id="btn-pull-image"),
                    Button(t("Remove", "삭제"), id="btn-remove-image"),
                    Button(t("Build Dev", "개발 이미지 빌드"), id="btn-build-image"),
                    id="refresh-bar",
                )
                yield RichLog(id="image-action-log", highlight=True)
            with TabPane(t("Containers", "컨테이너"), id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
            with TabPane(t("Disk / HF Cache", "디스크 / HF 캐시"), id="disk-tab"):
                yield RichLog(id="disk-info", highlight=True)
            with TabPane(t("Environment", "환경"), id="environment-tab"):
                yield Horizontal(
                    Button(t("Validate", "검증"), id="btn-validate-env", variant="primary"),
                    Button(t("Render Runtime Envs", "런타임 환경 렌더링"), id="btn-render-envs"),
                    id="environment-actions",
                )
                yield RichLog(id="environment-log", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        gpu_table = self.query_one("#gpu-table", DataTable)
        gpu_table.add_columns("GPU", "Name", "Memory Used", "Memory Total", "Utilization", "Temperature")

        official_table = self.query_one("#official-images", DataTable)
        official_table.add_columns("Tag", "Size", "Created")

        dev_table = self.query_one("#dev-images", DataTable)
        dev_table.add_columns("Tag", "Size", "Created")

        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()

        self._gpu_timer = self.set_interval(3, self._refresh_gpu)

    def on_screen_suspend(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.pause()

    def on_screen_resume(self) -> None:
        self._refresh_gpu()
        if self._gpu_timer is not None:
            self._gpu_timer.resume()


    @work(exclusive=True, group="gpu")
    async def _refresh_gpu(self) -> None:
        try:
            gpus = await get_gpu_info()
        except RuntimeError as exc:
            self._update_gpu_error(exc)
            return
        self._update_gpu_table(gpus)

    def _update_gpu_error(self, exc: RuntimeError) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        table.add_row("--", t("GPU query failed", "GPU 조회 실패"), "--", "--", "--", "--")
        self.notify(
            t(f"GPU query failed: {exc}", f"GPU 조회 실패: {exc}"),
            severity="error",
            timeout=8,
        )

    def _update_gpu_table(self, gpus: list[GpuInfo]) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if not gpus:
            table.add_row("--", "No GPUs detected", "--", "--", "--", "--")
            return
        for gpu in gpus:
            try:
                util_level = gpu_utilization_level(gpu.utilization)
                util_color = {
                    GpuReadingLevel.NORMAL: "green",
                    GpuReadingLevel.WARNING: "yellow",
                    GpuReadingLevel.CRITICAL: "red",
                }[util_level]
                util_display = f"[{util_color}]{gpu.utilization}%[/]"
            except ValueError:
                util_display = "--"

            try:
                temp_level = gpu_temperature_level(gpu.temperature)
                temp_color = {
                    GpuReadingLevel.NORMAL: "green",
                    GpuReadingLevel.WARNING: "yellow",
                    GpuReadingLevel.CRITICAL: "red",
                }[temp_level]
                temp_display = f"[{temp_color}]{gpu.temperature}°C[/]"
            except ValueError:
                temp_display = "--"

            try:
                memory_used = parse_gpu_reading(gpu.memory_used, "memory used")
                memory_used_display = f"{memory_used:g} MiB"
            except ValueError:
                memory_used_display = "--"
            try:
                memory_total = parse_gpu_reading(gpu.memory_total, "memory total")
                if memory_total <= 0:
                    raise ValueError
                memory_total_display = f"{memory_total:g} MiB"
            except ValueError:
                memory_total_display = "--"

            table.add_row(
                gpu.index,
                gpu.name,
                memory_used_display,
                memory_total_display,
                util_display,
                temp_display,
            )


    @work(exclusive=True, group="images")
    async def _refresh_images(self) -> None:
        for table_id, fetch in (
            ("#official-images", get_docker_images),
            ("#dev-images", get_dev_images),
        ):
            try:
                self._update_image_table(table_id, await fetch())
            except Exception as exc:
                self._image_table_error(table_id, exc)

    def _update_image_table(self, table_id: str, images: list[DockerImage]) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        if not images:
            table.add_row("(none)", "--", "--")
            return
        for img in images:
            table.add_row(img.tag, img.size, img.created)

    def _image_table_error(self, table_id: str, exc: Exception) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        table.add_row(t("[red](query failed)[/]", "[red](조회 실패)[/]"), "--", "--")
        self.notify(t(f"Image list failed: {exc}", f"이미지 조회 실패: {exc}"),
                    severity="error", timeout=8)


    @work(exclusive=True, group="containers")
    async def _refresh_containers(self) -> None:
        log = self.query_one("#container-info", RichLog)
        log.clear()
        try:
            known_names = {
                load_profile(name).container_name
                for name in list_profile_names()
            }
            for stored in profile_store.list_profiles("llamacpp"):
                known_names.add(stored.container_name or stored.name)
            snapshots = await container_snapshots(include_stopped=True)
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(t("[red]Failed to get container info[/]", "[red]컨테이너 정보를 가져오지 못했습니다[/]"))
            log.write(str(exc))
            self.notify(
                t(
                    f"Container inventory failed: {exc}",
                    f"컨테이너 목록 조회 실패: {exc}",
                ),
                severity="error",
                timeout=8,
            )
            return
        selected = [snapshots[name] for name in sorted(known_names & snapshots.keys())]
        if not selected:
            log.write(t("[dim]No profile containers found.[/]", "[dim]프로필 컨테이너가 없습니다.[/]"))
            return
        log.write("NAME\tSTATE\tHEALTH\tSTATUS")
        for snapshot in selected:
            health = snapshot.health.value if snapshot.health.value != "none" else "--"
            log.write(
                f"{snapshot.name}\t{snapshot.raw_state}\t{health}\t{snapshot.raw_status}"
            )


    @work(exclusive=True, group="disk")
    async def _refresh_disk(self) -> None:
        log = self.query_one("#disk-info", RichLog)
        log.clear()
        try:
            env = parse_env_file(COMMON_ENV)
            raw_hf_cache = env.get("HF_CACHE_PATH", "").strip()
            if not raw_hf_cache:
                raise ValueError("HF_CACHE_PATH is not set in .env.common")
            hf_cache = host_expand(raw_hf_cache)
            probe_path = _nearest_existing_path(Path(hf_cache))
            used, avail, pct = await get_disk_usage(str(probe_path))
        except Exception as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(
                t(f"Disk inventory failed: {exc}", f"디스크 조회 실패: {exc}"),
                severity="error",
                timeout=8,
            )
            return

        log.write(t(f"[b]HF cache path:[/b] {hf_cache}", f"[b]HF 캐시 경로:[/b] {hf_cache}"))
        log.write("")
        log.write(t("[b]Filesystem usage[/b]", "[b]파일시스템 사용량[/b]"))
        log.write(t(f"  Used: {used}  Available: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))


    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self.notify(t("Refreshing all system info...", "모든 시스템 정보 새로고침 중..."))

    def _prompt_pull_image(self) -> None:
        configured = parse_env_file(COMMON_ENV).get("VLLM_IMAGE", "").strip()

        def after(image_ref: str | None) -> None:
            if image_ref:
                self._pull_image(image_ref)

        self.app.push_screen(
            TextPromptModal(
                t("Image reference to pull", "받을 이미지 참조"),
                default=configured,
                placeholder="vllm/vllm-openai:v0.20.1",
            ),
            after,
        )

    def _prompt_remove_image(self) -> None:
        def after_ref(image_ref: str | None) -> None:
            if not image_ref:
                return

            def after_confirm(confirmed: bool) -> None:
                if confirmed:
                    self._remove_image(image_ref)

            self.app.push_screen(
                ConfirmModal(
                    t(
                        f"Remove local image [b]{image_ref}[/b]?",
                        f"로컬 이미지 [b]{image_ref}[/b] 을 삭제할까요?",
                    ),
                    confirm_label=t("Remove", "삭제"),
                ),
                after_confirm,
            )

        self.app.push_screen(
            TextPromptModal(
                t("Image reference or ID to remove", "삭제할 이미지 참조 또는 ID"),
                placeholder="vllm/vllm-openai:v0.20.1",
            ),
            after_ref,
        )

    def _prompt_build_image(self) -> None:
        repo_url, branch = get_dev_build_defaults()

        def after(options: dict[str, object] | None) -> None:
            if options:
                self._build_image(options)

        self.app.push_screen(
            DevBuildPromptModal("vllm", repo_url, branch),
            after,
        )

    @work(exclusive=True, group="image-action")
    async def _pull_image(self, image_ref: str) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        error = image_tag_error(image_ref)
        if error:
            log.write(f"[red]{error}[/]")
            self.notify(error, severity="error", timeout=8)
            return
        log.write(t(f"Pulling {image_ref}", f"{image_ref} 받는 중"))
        try:
            rc, lines = await system_operations.pull_image(image_ref)
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for line in lines:
            log.write(line)
        if rc != 0:
            self.notify(t("Image pull failed", "이미지 받기 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Image pulled", "이미지를 받았습니다"))

    @work(exclusive=True, group="image-action")
    async def _remove_image(self, image_ref: str) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        rc, output = await system_operations.remove_image(image_ref)
        if output.strip():
            log.write(output)
        if rc != 0:
            self.notify(t("Image removal failed", "이미지 삭제 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Image removed", "이미지를 삭제했습니다"))

    @work(exclusive=True, group="image-action")
    async def _build_image(self, options: dict[str, object]) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        try:
            rc, lines = await system_operations.build_dev_image(
                "vllm",
                repo_url=str(options["repo_url"]),
                branch=str(options["branch"]),
                custom_tag=str(options["custom_tag"]),
                official=bool(options["official"]),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for line in lines:
            log.write(line)
        if rc != 0:
            self.notify(t("Development image build failed", "개발 이미지 빌드 실패"), severity="error", timeout=8)
            return
        self._refresh_images()
        self.notify(t("Development image built", "개발 이미지를 빌드했습니다"))

    def _validate_environment(self) -> None:
        log = self.query_one("#environment-log", RichLog)
        log.clear()
        ok, messages = system_operations.environment_status(COMMON_ENV)
        for message in messages:
            log.write(message)
        if ok:
            log.write(t("[green]Status: OK[/]", "[green]상태: 정상[/]"))
            self.notify(t("Environment is valid", "환경 설정이 유효합니다"))
        else:
            self.notify(t("Environment validation failed", "환경 설정 검증 실패"), severity="error", timeout=8)

    def _render_environment(self) -> None:
        log = self.query_one("#environment-log", RichLog)
        log.clear()
        try:
            rendered, failures = system_operations.render_backend_envs("vllm")
        except (OSError, RuntimeError, ValueError) as exc:
            log.write(f"[red]{exc}[/]")
            self.notify(str(exc), severity="error", timeout=8)
            return
        for path in rendered:
            log.write(t(f"Rendered {path}", f"렌더링 완료: {path}"))
        for failure in failures:
            log.write(f"[red]{failure}[/]")
        if failures:
            self.notify(t("Some runtime envs failed", "일부 런타임 환경 렌더링 실패"), severity="error", timeout=8)
        else:
            self.notify(t("Runtime envs rendered", "런타임 환경을 렌더링했습니다"))


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify(t("Refreshing Docker images...", "Docker 이미지 새로고침 중..."))
        elif event.button.id == "btn-pull-image":
            self._prompt_pull_image()
        elif event.button.id == "btn-remove-image":
            self._prompt_remove_image()
        elif event.button.id == "btn-build-image":
            self._prompt_build_image()
        elif event.button.id == "btn-validate-env":
            self._validate_environment()
        elif event.button.id == "btn-render-envs":
            self._render_environment()
