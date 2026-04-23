"""System information screen - GPU status, Docker images, running containers."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
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
    get_gpu_info,
    GpuInfo,
    get_docker_images,
    get_dev_images,
    DockerImage,
    run_command,
    list_profile_names,
    load_profile,
)


class SystemScreen(Screen):
    """Full screen with tabbed content showing GPU, Docker images, and containers."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "refresh_all", "Refresh All", show=True),
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
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gpu_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("GPU Status", id="gpu-tab"):
                yield DataTable(id="gpu-table")
            with TabPane("Docker Images", id="images-tab"):
                yield Static("Official Images (vllm/vllm-openai)", classes="section-title")
                yield DataTable(id="official-images")
                yield Static("Dev Images (vllm-dev)", classes="section-title")
                yield DataTable(id="dev-images")
                yield Horizontal(
                    Button("Refresh Images", id="btn-refresh-images", variant="primary"),
                    id="refresh-bar",
                )
            with TabPane("Containers", id="containers-tab"):
                yield RichLog(id="container-info", highlight=True)
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
        """Fetch known profile containers and display them."""
        known_names = {
            load_profile(name).container_name
            for name in list_profile_names()
        }
        rc, output = await run_command(
            "docker", "ps", "-a",
            "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
            timeout=10,
        )
        log = self.query_one("#container-info", RichLog)
        log.clear()
        if rc != 0:
            log.write("[red]Failed to get container info[/]")
            log.write(output)
        elif not output.strip():
            log.write("[dim]No vLLM containers running.[/]")
        else:
            lines = output.strip().splitlines()
            filtered = [lines[0]]
            filtered.extend(
                line for line in lines[1:]
                if line.split()[0] in known_names
            )
            if len(filtered) == 1:
                log.write("[dim]No profile containers found.[/]")
                return
            for line in filtered:
                log.write(line)

    # ----- Actions -----

    def action_go_back(self) -> None:
        self.app.switch_screen("dashboard")

    def action_refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_images()
        self._refresh_containers()
        self.notify("Refreshing all system info...")

    # ----- Button handlers -----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-images":
            self._refresh_images()
            self.notify("Refreshing Docker images...")
