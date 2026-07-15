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
)
from tui.common import profile_store
from tui.common.docker import get_disk_usage
from tui.common.env import host_expand, parse_env_file
from tui.common.i18n import t


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
                    id="refresh-bar",
                )
            with TabPane(t("Containers", "컨테이너"), id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
            with TabPane(t("Disk / HF Cache", "디스크 / HF 캐시"), id="disk-tab"):
                yield RichLog(id="disk-info", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        # GPU table columns
        gpu_table = self.query_one("#gpu-table", DataTable)
        gpu_table.add_columns("GPU", "Name", "Memory Used", "Memory Total", "Utilization", "Temperature")

        # Official images table columns
        official_table = self.query_one("#official-images", DataTable)
        official_table.add_columns("Tag", "Size", "Created")

        # Dev images table columns
        dev_table = self.query_one("#dev-images", DataTable)
        dev_table.add_columns("Tag", "Size", "Created")

        # Initial data load
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()

        # Auto-refresh GPU every 3 seconds
        self._gpu_timer = self.set_interval(3, self._refresh_gpu)

    def on_screen_suspend(self) -> None:
        if self._gpu_timer is not None:
            self._gpu_timer.pause()

    def on_screen_resume(self) -> None:
        self._refresh_gpu()
        if self._gpu_timer is not None:
            self._gpu_timer.resume()

    # ----- GPU Tab -----

    @work(exclusive=True, group="gpu")
    async def _refresh_gpu(self) -> None:
        """Fetch GPU info and update the GPU table."""
        gpus = await get_gpu_info()
        self._update_gpu_table(gpus)

    def _update_gpu_table(self, gpus: list[GpuInfo]) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if not gpus:
            table.add_row("--", "No GPU info available", "--", "--", "--", "--")
            return
        for gpu in gpus:
            # Color code utilization
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

            # Color code temperature
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

    # ----- Docker Images Tab -----

    @work(exclusive=True, group="images")
    async def _refresh_images(self) -> None:
        """Fetch Docker images and update both tables."""
        official = await get_docker_images()
        dev = await get_dev_images()
        self._update_image_table("#official-images", official)
        self._update_image_table("#dev-images", dev)

    def _update_image_table(self, table_id: str, images: list[DockerImage]) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        if not images:
            table.add_row("(none)", "--", "--")
            return
        for img in images:
            table.add_row(img.tag, img.size, img.created)

    # ----- Containers Tab -----

    @work(exclusive=True, group="containers")
    async def _refresh_containers(self) -> None:
        """Show every llmux-managed container, across BOTH backends.

        The unified Dashboard already lists profiles from both backends; this
        view used to filter to just the vLLM side, which made the System
        screen claim 'no profile containers found' when only llama.cpp
        containers were running. The CLI `container ps` returns both — this
        now matches.
        """
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

    # ----- Disk Tab -----

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
        used, avail, pct = await get_disk_usage(hf_cache)
        if used:
            log.write(t("[b]Filesystem usage[/b]", "[b]파일시스템 사용량[/b]"))
            log.write(t(f"  Used: {used}  Available: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))
        else:
            log.write(t(f"[yellow]Could not stat {hf_cache} (does it exist?)[/]", f"[yellow]{hf_cache} 를 확인할 수 없습니다 (존재하나요?)[/]"))

    # ----- Actions -----

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

    # ----- Button handlers -----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify(t("Refreshing Docker images...", "Docker 이미지 새로고침 중..."))
