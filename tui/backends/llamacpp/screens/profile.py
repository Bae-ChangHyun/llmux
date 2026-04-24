"""Profile CRUD — .env 파일 편집 modal."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from tui.backends.llamacpp.backend import (
    Profile,
    delete_profile,
    list_config_names,
    list_profile_names,
    load_profile,
    save_profile,
    validate_name,
)


# ---------------------------------------------------------------------------
# ProfileFormScreen
# ---------------------------------------------------------------------------


class ProfileFormScreen(ModalScreen[str | None]):
    """Profile 생성/편집 modal."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ProfileFormScreen { align: center middle; }
    ProfileFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 78;
        min-width: 55;
        height: 95%;
        max-height: 38;
        min-height: 18;
    }
    ProfileFormScreen VerticalScroll { height: 1fr; min-height: 5; }
    ProfileFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row { height: auto; margin-bottom: 1; }
    ProfileFormScreen .form-row Label {
        width: 22;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ProfileFormScreen #section-title {
        margin-top: 1;
        text-style: bold;
        color: $text;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ProfileFormScreen #section-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-buttons {
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self, profile: Profile | None = None) -> None:
        super().__init__()
        self._profile = profile
        self._edit_mode = profile is not None
        self._saved_name: str | None = None

    def compose(self) -> ComposeResult:
        p = self._profile
        title = f"Edit Profile: {p.name}" if p else "New Profile"

        configs = list_config_names()
        config_options: list[tuple[str, str]] = [(name, name) for name in configs]

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label("Profile Name")
                    yield Input(
                        value=p.name if p else "",
                        placeholder="my-profile",
                        id="name-input",
                        disabled=self._edit_mode,
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Container Name")
                    yield Input(
                        value=p.container_name if p else "",
                        placeholder="(기본: profile name)",
                        id="container-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Port")
                    yield Input(
                        value=str(p.port) if p else "8080",
                        placeholder="8080",
                        id="port-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("GPU ID")
                    yield Input(
                        value=p.gpu_id if p else "0",
                        placeholder="0",
                        id="gpu-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Config")
                    select_kwargs: dict = dict(
                        prompt="Select config",
                        allow_blank=True,
                        id="config-select",
                    )
                    if p and p.config_name and p.config_name in configs:
                        select_kwargs["value"] = p.config_name
                    yield Select(config_options, **select_kwargs)

            with Horizontal(classes="form-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Close", id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        container = self.query_one("#container-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_id = self.query_one("#gpu-input", Input).value.strip()

        config_select = self.query_one("#config-select", Select)
        config_name = (
            str(config_select.value) if config_select.value != Select.BLANK else ""
        )

        # --- Validation ---
        if not name:
            self.notify("Profile 이름 필수", severity="error")
            return
        if not validate_name(name):
            self.notify(
                "이름은 영숫자/대시/언더스코어 ('-' 시작 금지)", severity="error"
            )
            return
        if not self._edit_mode and name in list_profile_names():
            self.notify(f"Profile '{name}' 이미 존재", severity="error")
            return
        if container and not validate_name(container):
            self.notify("컨테이너 이름 규칙 위반", severity="error")
            return
        try:
            port_int = int(port or "8080")
            if not (1024 <= port_int <= 65535):
                raise ValueError
        except ValueError:
            self.notify("Port 는 1024–65535 정수", severity="error")
            return
        if gpu_id and not re.match(r"^[0-9](,[0-9])*$", gpu_id):
            self.notify("GPU ID 는 숫자/콤마 (예: 0 또는 0,1)", severity="error")
            return

        # --- Build --- (HF 필드는 quick_setup 에서 설정, 편집 시 보존)
        if self._edit_mode and self._profile is not None:
            p = self._profile
            p.container_name = container or name
            p.port = port_int
            p.gpu_id = gpu_id or "0"
            p.config_name = config_name or name
        else:
            p = Profile(
                name=name,
                container_name=container or name,
                port=port_int,
                gpu_id=gpu_id or "0",
                config_name=config_name or name,
            )

        save_profile(p)
        self.notify(f"저장: {name}", severity="information")
        self._saved_name = name

        if not self._edit_mode:
            self._edit_mode = True
            self._profile = p
            self.query_one("#name-input", Input).disabled = True
            self.query_one("#form-title", Static).update(
                f"[b]Edit Profile: {name}[/b]"
            )

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)


# ---------------------------------------------------------------------------
# ProfileDeleteScreen
# ---------------------------------------------------------------------------


class ProfileDeleteScreen(ModalScreen[bool]):
    """Profile 삭제 확인."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ProfileDeleteScreen { align: center middle; }
    ProfileDeleteScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    ProfileDeleteScreen #delete-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileDeleteScreen .form-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    ProfileDeleteScreen .form-buttons Button { margin: 0 1; }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name
        self._profile = load_profile(profile_name)

    def compose(self) -> ComposeResult:
        cfg = self._profile.config_name
        other_refs = [
            n for n in list_profile_names()
            if n != self._profile_name and load_profile(n).config_name == cfg
        ] if cfg else []

        with Vertical():
            if cfg and not other_refs:
                yield Static(
                    f"[b]{self._profile_name}[/b] 삭제?\n"
                    f"[dim](연결된 config '{cfg}' 도 함께 삭제됨 — 다른 프로필 참조 없음)[/dim]",
                    id="delete-message",
                )
            elif cfg:
                yield Static(
                    f"[b]{self._profile_name}[/b] 삭제?\n"
                    f"[dim](config '{cfg}' 는 {', '.join(other_refs)} 도 사용 중 → 유지)[/dim]",
                    id="delete-message",
                )
            else:
                yield Static(
                    f"[b]{self._profile_name}[/b] 삭제?",
                    id="delete-message",
                )
            with Horizontal(classes="form-buttons"):
                yield Button("Delete", id="delete-btn", variant="error")
                yield Button("Cancel", id="cancel-btn", variant="default")

    @on(Button.Pressed, "#delete-btn")
    def _on_delete(self, event: Button.Pressed) -> None:
        cfg = self._profile.config_name
        other_refs = [
            n for n in list_profile_names()
            if n != self._profile_name and load_profile(n).config_name == cfg
        ] if cfg else []
        delete_config_too = bool(cfg) and not other_refs
        delete_profile(self._profile_name, delete_config_too=delete_config_too)
        if delete_config_too:
            self.app.notify(f"삭제: {self._profile_name} + config '{cfg}'")
        else:
            self.app.notify(f"삭제: {self._profile_name}")
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
