"""System information screen - GPU status, Docker images, running containers."""

from __future__ import annotations

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
    run_command,
    list_profile_names,
    load_profile,
    get_dev_build_defaults,
)
from tui.common import profile_store, system_operations
from tui.common.docker import get_disk_usage
from tui.common.env import host_expand, parse_env_file
from tui.common.i18n import t
from tui.common.widgets import ConfirmModal, DevBuildPromptModal, TextPromptModal


class SystemScreen(Screen):
    """Full screen with tabbed content showing GPU, Docker images, and containers."""

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
        /* Cap each table at a sane row count so the bottom one is
           visible by default; the outer VerticalScroll handles overflow. */
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
                # Wrap in VerticalScroll so the two stacked DataTables
                # remain reachable on short terminals — without this,
                # both tables shrink to ~1–2 rows each and the bottom
                # one can disappear entirely.
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
        gpus = await get_gpu_info()
        self._update_gpu_table(gpus)

    def _update_gpu_table(self, gpus: list[GpuInfo]) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if not gpus:
            table.add_row("--", "No GPU info available", "--", "--", "--", "--")
            return
        for gpu in gpus:
            try:
                util_val = int(gpu.utilization)
            except (ValueError, TypeError):
                util_val = 0
            if util_val > 80:
                util_display = f"[red]{gpu.utilization}%[/]"
            elif util_val > 50:
                util_display = f"[yellow]{gpu.utilization}%[/]"
            else:
                util_display = f"[green]{gpu.utilization}%[/]"

            try:
                temp_val = int(gpu.temperature)
            except (ValueError, TypeError):
                temp_val = 0
            if temp_val > 80:
                temp_display = f"[red]{gpu.temperature}°C[/]"
            elif temp_val > 60:
                temp_display = f"[yellow]{gpu.temperature}°C[/]"
            else:
                temp_display = f"[green]{gpu.temperature}°C[/]"

            table.add_row(
                gpu.index,
                gpu.name,
                f"{gpu.memory_used} MiB",
                f"{gpu.memory_total} MiB",
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
        """Show every llmux-managed container, across BOTH backends — matching
        the unified Dashboard and the CLI `container ps`."""
        known_names = {
            load_profile(name).container_name
            for name in list_profile_names()
        }
        for stored in profile_store.list_profiles("llamacpp"):
            known_names.add(stored.container_name or stored.name)
        rc, output = await run_command(
            "docker", "ps", "-a",
            "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
            timeout=10,
        )
        log = self.query_one("#container-info", RichLog)
        log.clear()
        if rc != 0:
            log.write(t("[red]Failed to get container info[/]", "[red]컨테이너 정보를 가져오지 못했습니다[/]"))
            log.write(output)
        elif not output.strip():
            log.write(t("[dim]No containers running.[/]", "[dim]실행 중인 컨테이너가 없습니다.[/]"))
        else:
            lines = output.strip().splitlines()
            filtered = [lines[0]]
            filtered.extend(
                line for line in lines[1:]
                if line.split()[0] in known_names
            )
            if len(filtered) == 1:
                log.write(t("[dim]No profile containers found.[/]", "[dim]프로필 컨테이너가 없습니다.[/]"))
                return
            for line in filtered:
                log.write(line)


    @work(exclusive=True, group="disk")
    async def _refresh_disk(self) -> None:
        """Show the host HF cache directory + its filesystem usage.

        vLLM streams every model through the host HF cache (mounted via
        compose as `${HF_CACHE_PATH}:/root/.cache/huggingface`), so the
        equivalent of llama.cpp's GGUF model directory is that path.
        """
        log = self.query_one("#disk-info", RichLog)
        log.clear()
        env = parse_env_file(COMMON_ENV)
        hf_cache = env.get("HF_CACHE_PATH", "")
        if not hf_cache:
            log.write(t("[yellow]HF_CACHE_PATH is not set in .env.common[/]", "[yellow].env.common 에 HF_CACHE_PATH 가 설정되지 않았습니다[/]"))
            return
        # The template default is `/home/$USER/...`; df needs the expanded path
        # or it always reports "Could not stat" (parity with the CLI/llama.cpp).
        hf_cache = host_expand(hf_cache)
        log.write(t(f"[b]HF cache path:[/b] {hf_cache}", f"[b]HF 캐시 경로:[/b] {hf_cache}"))
        log.write("")
        try:
            used, avail, pct = await get_disk_usage(hf_cache)
        except RuntimeError as exc:
            log.write(f"[red]{exc}[/]")
            return
        log.write(t("[b]Filesystem usage[/b]", "[b]파일시스템 사용량[/b]"))
        log.write(t(f"  Used: {used}  Available: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))


    def action_go_back(self) -> None:
        # pop_screen matches the llama.cpp side and the rest of the modal
        # navigation in this app; switch_screen would reset the screen stack
        # and lose any in-flight context the caller had pushed.
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
