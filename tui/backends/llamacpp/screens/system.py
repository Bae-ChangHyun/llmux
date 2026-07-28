"""System 정보 화면 — GPU / Docker 이미지 / 디스크."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
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
from tui.backends.llamacpp.backend_runtime import LLAMACPP_DEV_SPEC
from tui.common import profile_store
from tui.backends.llamacpp.backend import LLAMACPP_OFFICIAL_REPO
from tui.common.dev_build import list_local_dev_images
from tui.common.i18n import t


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
                yield Button(t("Refresh Images", "이미지 새로고침"), id="btn-refresh-images", variant="primary")
            with TabPane(t("Containers", "컨테이너"), id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
            with TabPane(t("Disk / Model Dir", "디스크 / 모델 폴더"), id="disk-tab"):
                yield RichLog(id="disk-info", highlight=True)
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

    # ----- GPU -----

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

    # ----- Docker images -----

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

    # ----- Containers -----

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

    # ----- Disk -----

    @work(exclusive=True, group="sys-disk")
    async def _refresh_disk(self) -> None:
        model_dir = _get_model_dir()
        log = self.query_one("#disk-info", RichLog)
        log.clear()
        log.write(f"[b]Project root:[/b] {ROOT}")
        log.write(f"[b]Model dir:[/b] {model_dir}")
        log.write("")

        # 모델 파일 목록
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

        # df -h
        probe_path = model_dir if model_dir.exists() else ROOT
        used, avail, pct = await get_disk_usage(str(probe_path))
        log.write(t("[b]Disk usage[/b]", "[b]디스크 사용량[/b]"))
        if used:
            log.write(t(f"  Used: {used}  Free: {avail}  ({pct})", f"  사용: {used}  남음: {avail}  ({pct})"))
        else:
            log.write(t(f"  [yellow]Could not stat {probe_path} (does it exist?)[/]",
                        f"  [yellow]{probe_path} 조회 실패 (경로가 존재하나요?)[/]"))

    # ----- Actions -----

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self.notify(t("Refreshing", "새로고침"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify(t("Refreshing image list", "이미지 목록 새로고침"))
