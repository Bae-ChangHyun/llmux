"""System 정보 화면 — GPU / Docker 이미지 / 디스크."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
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
    DockerImage,
    GpuInfo,
    ROOT,
    _get_model_dir,
    get_disk_usage,
    get_docker_images,
    get_gpu_info,
    list_profile_names,
    load_profile,
    run_command,
)


class SystemScreen(Screen):
    """탭: GPU / Docker Images / Containers / Disk."""

    BINDINGS = [
        Binding("escape,backspace,s", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "refresh_all", "Refresh", show=True),
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
    """

    def __init__(self) -> None:
        super().__init__()
        self._gpu_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("GPU", id="gpu-tab"):
                yield DataTable(id="gpu-table")
            with TabPane("Docker Images", id="images-tab"):
                yield Static("llama.cpp images", classes="section-title")
                yield DataTable(id="llama-images")
                yield Button("Refresh Images", id="btn-refresh-images", variant="primary")
            with TabPane("Containers", id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
            with TabPane("Disk / Model Dir", id="disk-tab"):
                yield RichLog(id="disk-info", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        gpu_table = self.query_one("#gpu-table", DataTable)
        gpu_table.add_columns("GPU", "Name", "Mem Used", "Mem Total", "Util", "Temp")
        images_table = self.query_one("#llama-images", DataTable)
        images_table.add_columns("Tag", "Size", "Created")

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
                t = int(g.temperature)
            except ValueError:
                t = 0
            temp = (
                f"[red]{g.temperature}°C[/]" if t > 80
                else f"[yellow]{g.temperature}°C[/]" if t > 60
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
        imgs = await get_docker_images("ghcr.io/ggml-org/llama.cpp")
        self._update_images_table(imgs)

    def _update_images_table(self, images: list[DockerImage]) -> None:
        table = self.query_one("#llama-images", DataTable)
        table.clear()
        if not images:
            table.add_row("(없음)", "--", "--")
            return
        for img in images:
            table.add_row(img.tag, img.size, img.created)

    # ----- Containers -----

    @work(exclusive=True, group="sys-containers")
    async def _refresh_containers(self) -> None:
        known = {load_profile(n).container_name for n in list_profile_names()}
        rc, out = await run_command(
            "docker", "ps", "-a",
            "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
            timeout=10,
        )
        log = self.query_one("#container-info", RichLog)
        log.clear()
        if rc != 0:
            log.write("[red]docker ps 실패[/]")
            return
        lines = out.strip().splitlines()
        if len(lines) < 2:
            log.write("[dim]컨테이너 없음[/]")
            return
        header = lines[0]
        rows = [line for line in lines[1:] if line.split()[0] in known]
        log.write(header)
        if rows:
            for line in rows:
                log.write(line)
        else:
            log.write("[dim](이 프로젝트의 프로필 컨테이너 없음)[/]")

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
            log.write(f"[b]GGUF 파일 ({len(files)} 개)[/b]")
            total = 0
            for f in files:
                sz = f.stat().st_size
                total += sz
                log.write(f"  {f.name}  [dim]{sz / 1024**3:.1f} GB[/dim]")
            log.write(f"  [dim]합계: {total / 1024**3:.1f} GB[/dim]")
            log.write("")
        else:
            log.write(f"[yellow]모델 디렉토리 존재하지 않음: {model_dir}[/]")
            log.write("")

        # df -h
        used, avail, pct = await get_disk_usage(str(model_dir if model_dir.exists() else ROOT))
        if used:
            log.write("[b]디스크 사용량[/b]")
            log.write(f"  사용: {used}  남음: {avail}  ({pct})")

    # ----- Actions -----

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self._refresh_disk()
        self.notify("새로고침")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify("이미지 목록 새로고침")
