"""System 정보 화면 — GPU / Docker 이미지 / 디스크."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from tui.backends.llamacpp.backend import (
    COMMON_ENV,
    GpuInfo,
    ROOT,
    _get_hf_cache_dir,
    hf_cache_setting,
    _get_model_dir,
    get_disk_usage,
    get_docker_images,
    get_gpu_info,
    list_cached_gguf,
    list_profile_names,
    load_profile,
    run_command,
)
from tui.backends.llamacpp.backend_runtime import (
    LLAMACPP_DEV_SPEC,
    get_dev_build_defaults,
)
from tui.common import profile_store, system_operations
from tui.backends.llamacpp.backend import LLAMACPP_OFFICIAL_REPO
from tui.common.dev_build import list_local_dev_images
from tui.common.env import parse_env_file
from tui.common.i18n import t
from tui.common.widgets import ConfirmModal, DevBuildPromptModal, TextPromptModal


class SystemScreen(Screen):
    """탭: GPU / Docker Images / Containers / Disk."""

    BINDINGS = [
        Binding("escape,backspace,s", "go_back", t("Back", "뒤로"), show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "refresh_all", t("Refresh", "새로고침"), show=True),
    ]

    DEFAULT_CSS = """
    SystemScreen { layout: vertical; }
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

    def __init__(self) -> None:
        super().__init__()
        self._gpu_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("GPU", id="gpu-tab"):
                yield DataTable(id="gpu-table")
            with TabPane(t("Docker Images", "Docker 이미지"), id="images-tab"):
                with VerticalScroll(id="images-scroll"):
                    yield Static(
                        t(f"Official Images ({LLAMACPP_OFFICIAL_REPO})", f"공식 이미지 ({LLAMACPP_OFFICIAL_REPO})"),
                        classes="section-title",
                    )
                    yield DataTable(id="llama-images")
                    yield Static(t("Dev Images (llamacpp-dev)", "개발 이미지 (llamacpp-dev)"), classes="section-title")
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
            with TabPane(t("Disk / Model Dir", "디스크 / 모델 폴더"), id="disk-tab"):
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
        gpu_table.add_columns("GPU", "Name", "Mem Used", "Mem Total", "Util", "Temp")
        images_table = self.query_one("#llama-images", DataTable)
        images_table.add_columns("Tag", "Size", "Created")
        dev_images_table = self.query_one("#dev-images", DataTable)
        dev_images_table.add_columns("Tag", "Size", "Created")

        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self._gpu_timer = self.set_interval(3, self._refresh_gpu)

    def on_screen_suspend(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.pause()

    def on_screen_resume(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.resume()
        self._refresh_gpu()


    @work(exclusive=True, group="sys-gpu")
    async def _refresh_gpu(self) -> None:
        gpus = await get_gpu_info()
        self._update_gpu_table(gpus)

    def _update_gpu_table(self, gpus: list[GpuInfo]) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if not gpus:
            table.add_row("--", "nvidia-smi 미감지", "--", "--", "--", "--")
            return
        for g in gpus:
            try:
                u = int(g.utilization)
            except ValueError:
                u = 0
            util = (
                f"[red]{g.utilization}%[/]" if u > 80
                else f"[yellow]{g.utilization}%[/]" if u > 50
                else f"[green]{g.utilization}%[/]"
            )
            try:
                temp_c = int(g.temperature)
            except ValueError:
                temp_c = 0
            temp = (
                f"[red]{g.temperature}°C[/]" if temp_c > 80
                else f"[yellow]{g.temperature}°C[/]" if temp_c > 60
                else f"[green]{g.temperature}°C[/]"
            )
            try:
                used_gb = int(g.memory_used) / 1024
                total_gb = int(g.memory_total) / 1024
                mem_used = f"{used_gb:.1f} GB"
                mem_total = f"{total_gb:.1f} GB"
            except ValueError:
                mem_used = f"{g.memory_used} MiB"
                mem_total = f"{g.memory_total} MiB"
            table.add_row(g.index, g.name, mem_used, mem_total, util, temp)


    @work(exclusive=True, group="sys-images")
    async def _refresh_images(self) -> None:
        # `DevImage` from tui.common.dev_build has matching .tag/.size/.created
        # attributes, so the same fill helper works for both tables.
        for table_id, fetch in (
            ("#llama-images",
             lambda: get_docker_images(LLAMACPP_OFFICIAL_REPO)),
            ("#dev-images",
             lambda: list_local_dev_images(LLAMACPP_DEV_SPEC)),
        ):
            try:
                self._update_image_table(table_id, await fetch())
            except Exception as exc:
                self._image_table_error(table_id, exc)

    def _update_image_table(self, table_id: str, images) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        if not images:
            table.add_row("(없음)", "--", "--")
            return
        for img in images:
            table.add_row(img.tag, img.size, img.created)

    def _image_table_error(self, table_id: str, exc: Exception) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        table.add_row(t("[red](query failed)[/]", "[red](조회 실패)[/]"), "--", "--")
        self.notify(t(f"Image list failed: {exc}", f"이미지 조회 실패: {exc}"),
                    severity="error", timeout=8)


    @work(exclusive=True, group="sys-containers")
    async def _refresh_containers(self) -> None:
        """Show every llmux-managed container, across BOTH backends. Mirrors
        the vLLM System screen."""
        known = {load_profile(n).container_name for n in list_profile_names()}
        for stored in profile_store.list_profiles("vllm"):
            known.add(stored.container_name or stored.name)
        rc, out = await run_command(
            "docker", "ps", "-a",
            "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
            timeout=10,
        )
        log = self.query_one("#container-info", RichLog)
        log.clear()
        if rc != 0:
            log.write(t("[red]docker ps failed[/]", "[red]docker ps 실패[/]"))
            if out.strip():
                log.write(out)
            return
        lines = out.strip().splitlines()
        if len(lines) < 2:
            log.write(t("[dim]No containers[/]", "[dim]컨테이너 없음[/]"))
            return
        header = lines[0]
        rows = [line for line in lines[1:] if line.split()[0] in known]
        log.write(header)
        if rows:
            for line in rows:
                log.write(line)
        else:
            log.write(t("[dim](no profile containers for this project)[/]", "[dim](이 프로젝트의 프로필 컨테이너 없음)[/]"))


    @work(exclusive=True, group="sys-disk")
    async def _refresh_disk(self) -> None:
        model_dir = _get_model_dir()
        log = self.query_one("#disk-info", RichLog)
        log.clear()
        log.write(f"[b]Project root:[/b] {ROOT}")
        log.write(f"[b]Model dir:[/b] {model_dir}")
        log.write("")

        if model_dir.exists():
            files = sorted(
                (f for f in model_dir.glob("*.gguf")),
                key=lambda f: f.stat().st_size,
                reverse=True,
            )
            log.write(t(f"[b]GGUF files ({len(files)})[/b]", f"[b]GGUF 파일 ({len(files)} 개)[/b]"))
            total = 0
            for f in files:
                sz = f.stat().st_size
                total += sz
                log.write(f"  {f.name}  [dim]{sz / 1024**3:.1f} GB[/dim]")
            log.write(t(f"  [dim]Total: {total / 1024**3:.1f} GB[/dim]", f"  [dim]합계: {total / 1024**3:.1f} GB[/dim]"))
            log.write("")
        else:
            log.write(t(f"[yellow]Model directory does not exist: {model_dir}[/]", f"[yellow]모델 디렉토리 존재하지 않음: {model_dir}[/]"))
            log.write("")

        # HF hub 캐시 — llama-server 가 `-hf` 로 받는 실제 저장 위치
        if not hf_cache_setting():
            log.write(t(
                "[yellow]HF_CACHE_PATH is not set in .env.common — showing the "
                "default location, which may not be where models were downloaded.[/]",
                "[yellow].env.common 에 HF_CACHE_PATH 가 없습니다 — 기본 경로를 "
                "표시하며, 실제 다운로드 위치와 다를 수 있습니다.[/]",
            ))
        cached = list_cached_gguf()
        log.write(t(f"[b]HF cache GGUF ({len(cached)})[/b]  [dim]{_get_hf_cache_dir()}[/dim]", f"[b]HF cache GGUF ({len(cached)} 개)[/b]  [dim]{_get_hf_cache_dir()}[/dim]"))
        if cached:
            for c in cached:
                log.write(
                    f"  {c['repo']}/{c['name']}  [dim]{c['size_gb']:.1f} GB[/dim]"
                )
            total_gb = sum(c["size_bytes"] for c in cached) / 1024**3
            log.write(t(f"  [dim]Total: {total_gb:.1f} GB[/dim]", f"  [dim]합계: {total_gb:.1f} GB[/dim]"))
        else:
            log.write(t("  [dim](no GGUF downloaded in HF cache)[/dim]", "  [dim](HF 캐시에 받아둔 GGUF 없음)[/dim]"))
        log.write("")

        probe_path = model_dir if model_dir.exists() else ROOT
        try:
            used, avail, pct = await get_disk_usage(str(probe_path))
        except RuntimeError as exc:
            log.write(f"[red]{exc}[/]")
            return
        log.write(t("[b]Disk usage[/b]", "[b]디스크 사용량[/b]"))
        log.write(t(f"  Used: {used}  Free: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))


    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self.notify(t("Refreshing", "새로고침"))

    def _prompt_pull_image(self) -> None:
        configured = parse_env_file(COMMON_ENV).get("LLAMACPP_IMAGE", "").strip()
        default = configured or f"{LLAMACPP_OFFICIAL_REPO}:server-cuda"

        def after(image_ref: str | None) -> None:
            if image_ref:
                self._pull_image(image_ref)

        self.app.push_screen(
            TextPromptModal(
                t("Image reference to pull", "받을 이미지 참조"),
                default=default,
                placeholder=f"{LLAMACPP_OFFICIAL_REPO}:server-cuda",
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
                placeholder=f"{LLAMACPP_OFFICIAL_REPO}:server-cuda",
            ),
            after_ref,
        )

    def _prompt_build_image(self) -> None:
        repo_url, branch = get_dev_build_defaults()

        def after(options: dict[str, object] | None) -> None:
            if options:
                self._build_image(options)

        self.app.push_screen(
            DevBuildPromptModal("llamacpp", repo_url, branch),
            after,
        )

    @work(exclusive=True, group="sys-image-action")
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

    @work(exclusive=True, group="sys-image-action")
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

    @work(exclusive=True, group="sys-image-action")
    async def _build_image(self, options: dict[str, object]) -> None:
        log = self.query_one("#image-action-log", RichLog)
        log.clear()
        try:
            rc, lines = await system_operations.build_dev_image(
                "llamacpp",
                repo_url=str(options["repo_url"]),
                branch=str(options["branch"]),
                custom_tag=str(options["custom_tag"]),
                cuda_arch=str(options["cuda_arch"]),
                multi_arch=bool(options["multi_arch"]),
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
            rendered, failures = system_operations.render_backend_envs("llamacpp")
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
            self.notify(t("Refreshing image list", "이미지 목록 새로고침"))
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
