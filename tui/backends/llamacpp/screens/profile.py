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
from tui.common import profile_store
from tui.common.i18n import t


# ---------------------------------------------------------------------------
# ProfileFormScreen
# ---------------------------------------------------------------------------


def _default_port() -> str:
    """Backend default port, honoring a `defaults:` override in profiles.yaml."""
    from tui.common import profile_store

    return str(profile_store.effective_defaults("llamacpp")["port"])


class ProfileFormScreen(ModalScreen[str | None]):
    """Profile 생성/편집 modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    ProfileFormScreen { align: center middle; }
    ProfileFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 78;
        min-width: 55;
        /* No max-height — see vLLM ProfileFormScreen for rationale. */
        height: 95%;
        min-height: 12;
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
        title = (
            t(f"Edit Profile: {p.name}", f"프로필 편집: {p.name}")
            if p
            else t("New Profile", "새 프로필")
        )

        configs = list_config_names()
        config_options: list[tuple[str, str]] = [(name, name) for name in configs]

        # A dangling link (config YAML deleted out from under the profile) has
        # no matching option, so the Select would fall back to BLANK and saving
        # would silently rewrite config_name. Surface it instead.
        if p and p.config_name and p.config_name not in configs:
            config_options.insert(
                0, (t(f"{p.config_name} (missing)", f"{p.config_name} (없음)"), p.config_name)
            )
            configs = [*configs, p.config_name]

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label(t("Profile Name", "프로필 이름"))
                    yield Input(
                        value=p.name if p else "",
                        placeholder="my-profile",
                        id="name-input",
                        disabled=self._edit_mode,
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Container Name", "컨테이너 이름"))
                    yield Input(
                        value=p.container_name if p else "",
                        placeholder=t("(default: profile name)", "(기본: profile name)"),
                        id="container-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Port")
                    _port = _default_port()
                    yield Input(
                        value=str(p.port) if p else _port,
                        placeholder=_port,
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
                        prompt=t("Select config", "config 선택"),
                        allow_blank=True,
                        id="config-select",
                    )
                    if p and p.config_name and p.config_name in configs:
                        select_kwargs["value"] = p.config_name
                    yield Select(config_options, **select_kwargs)

                with Horizontal(classes="form-row"):
                    yield Label("HF Repo")
                    yield Input(
                        value=p.hf_repo if p else "",
                        placeholder=t(
                            "org/Model-GGUF — required (in-container -hf download source)",
                            "org/Model-GGUF — 필수 (컨테이너 내 -hf 다운로드 소스)",
                        ),
                        id="hf-repo-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("HF File")
                    yield Input(
                        value=p.hf_file if p else "",
                        placeholder=t("(optional) model-Q4_K_M.gguf", "(선택) model-Q4_K_M.gguf"),
                        id="hf-file-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Model File")
                    yield Input(
                        value=p.model_file if p else "",
                        placeholder=t(
                            "(optional) GGUF filename — only when it differs from HF File",
                            "(선택) GGUF 파일명 — HF File과 다를 때만",
                        ),
                        id="model-file-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Image Tag", "이미지 태그"))
                    yield Input(
                        value=p.image_tag if p else "",
                        placeholder=t(
                            "(blank = default image) e.g. llamacpp-dev:mtp-clean",
                            "(비우면 기본 이미지) 예: llamacpp-dev:mtp-clean",
                        ),
                        id="image-tag-input",
                    )

            with Horizontal(classes="form-buttons"):
                yield Button(t("Save", "저장"), id="save-btn", variant="primary")
                yield Button(t("Close", "닫기"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        container = self.query_one("#container-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_id = self.query_one("#gpu-input", Input).value.strip()
        image_tag = self.query_one("#image-tag-input", Input).value.strip()
        hf_repo = self.query_one("#hf-repo-input", Input).value.strip()
        hf_file = self.query_one("#hf-file-input", Input).value.strip()
        model_file = self.query_one("#model-file-input", Input).value.strip()

        config_select = self.query_one("#config-select", Select)
        config_name = (
            str(config_select.value) if config_select.value != Select.BLANK else ""
        )

        # --- Validation ---
        if not name:
            self.notify(t("Profile name is required", "Profile 이름 필수"), severity="error")
            return
        if not validate_name(name):
            self.notify(
                t("Name must be lowercase letters/digits/dashes/underscores", "이름은 소문자/숫자/대시/언더스코어"),
                severity="error",
            )
            return
        # Profile names must be globally unique across both backends (see
        # profile_store.find_name_owner) — container_name defaults to the name.
        if not self._edit_mode:
            owner = profile_store.find_name_owner(name)
            if owner is not None:
                self.notify(
                    t(
                        f"Profile '{name}' already exists (backend {owner})",
                        f"Profile '{name}' 이미 존재 (backend {owner})",
                    ),
                    severity="error",
                )
                return

        if not self._edit_mode and name == "example":
            # config 링크 기본값이 프로필 이름이라, 'example' 프로필은 tracked
            # example.yaml 에 파라미터를 써버린다.
            self.notify(
                t(
                    "'example' is the tracked template config name — use another name",
                    "'example' 은 tracked 템플릿 config 이름 — 다른 이름을 쓸 것",
                ),
                severity="error",
            )
            return
        if container and not validate_name(container):
            self.notify(t("Container name violates the naming rule", "컨테이너 이름 규칙 위반"), severity="error")
            return
        try:
            port_int = int(port or _default_port())
            if not (1024 <= port_int <= 65535):
                raise ValueError
        except ValueError:
            self.notify(t("Port must be an integer 1024–65535", "Port 는 1024–65535 정수"), severity="error")
            return
        if gpu_id and not re.match(r"^[0-9]+(,[0-9]+)*$", gpu_id):
            self.notify(t("GPU ID must be digits/commas (e.g. 0 or 0,1)", "GPU ID 는 숫자/콤마 (예: 0 또는 0,1)"), severity="error")
            return
        if hf_repo and not re.match(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$", hf_repo):
            self.notify(t("HF Repo format: org/name (e.g. unsloth/Qwen3-8B-GGUF)", "HF Repo 형식: org/name (예: unsloth/Qwen3-8B-GGUF)"), severity="error")
            return
        from tui.common.dev_build import image_tag_error

        tag_err = image_tag_error(image_tag)
        if tag_err:
            self.notify(tag_err, severity="error")
            return

        # --- Build ---
        # model_file defaults to hf_file when only the HF coordinates were
        # provided — mirrors what Quick Setup does so a hand-built profile
        # doesn't have to repeat the filename.
        effective_model_file = model_file or hf_file

        if self._edit_mode and self._profile is not None:
            p = self._profile
            p.container_name = container or name
            p.port = port_int
            p.gpu_id = gpu_id or "0"
            p.config_name = config_name or name
            p.image_tag = image_tag
            p.hf_repo = hf_repo
            p.hf_file = hf_file
            p.model_file = effective_model_file
        else:
            p = Profile(
                name=name,
                container_name=container or name,
                port=port_int,
                gpu_id=gpu_id or "0",
                config_name=config_name or name,
                image_tag=image_tag,
                hf_repo=hf_repo,
                hf_file=hf_file,
                model_file=effective_model_file,
            )

        save_profile(p)
        self.notify(t(f"Saved: {name}", f"저장: {name}"), severity="information")
        if not hf_repo:
            self.notify(
                t(
                    "Cannot start container without HF Repo (-hf download source required)",
                    "HF Repo 없이는 컨테이너 시작 불가 (-hf 다운로드 소스 필요)",
                ),
                severity="warning",
            )
        if not effective_model_file:
            self.notify(
                t(
                    "No HF File/Model File — the linked config must have model-file to start",
                    "HF File/Model File 없음 — 링크된 config 에 model-file 이 있어야 시작 가능",
                ),
                severity="warning",
            )
        self._saved_name = name

        if not self._edit_mode:
            self._edit_mode = True
            self._profile = p
            self.query_one("#name-input", Input).disabled = True
            self.query_one("#form-title", Static).update(
                t(f"[b]Edit Profile: {name}[/b]", f"[b]프로필 편집: {name}[/b]")
            )

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)

    def action_scroll_form(self, direction: str) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
        except Exception:
            return
        if direction == "up":
            scroll.scroll_page_up()
        elif direction == "down":
            scroll.scroll_page_down()
        elif direction == "home":
            scroll.scroll_home()
        elif direction == "end":
            scroll.scroll_end()


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
            if cfg == "example":
                # delete_profile() 이 tracked 템플릿은 건너뛴다 — 삭제된다고
                # 말하면 거짓 안내가 된다.
                yield Static(
                    t(
                        f"Delete [b]{self._profile_name}[/b]?\n"
                        f"[dim](config 'example' is the tracked template — kept)[/dim]",
                        f"[b]{self._profile_name}[/b] 삭제?\n"
                        f"[dim](config 'example' 은 tracked 템플릿이라 유지됨)[/dim]",
                    ),
                    id="delete-message",
                )
            elif cfg and not other_refs:
                yield Static(
                    t(
                        f"Delete [b]{self._profile_name}[/b]?\n"
                        f"[dim](linked config '{cfg}' is deleted too — no other profile references it)[/dim]",
                        f"[b]{self._profile_name}[/b] 삭제?\n"
                        f"[dim](연결된 config '{cfg}' 도 함께 삭제됨 — 다른 프로필 참조 없음)[/dim]",
                    ),
                    id="delete-message",
                )
            elif cfg:
                yield Static(
                    t(
                        f"Delete [b]{self._profile_name}[/b]?\n"
                        f"[dim](config '{cfg}' is also used by {', '.join(other_refs)} → kept)[/dim]",
                        f"[b]{self._profile_name}[/b] 삭제?\n"
                        f"[dim](config '{cfg}' 는 {', '.join(other_refs)} 도 사용 중 → 유지)[/dim]",
                    ),
                    id="delete-message",
                )
            else:
                yield Static(
                    t(f"Delete [b]{self._profile_name}[/b]?", f"[b]{self._profile_name}[/b] 삭제?"),
                    id="delete-message",
                )
            with Horizontal(classes="form-buttons"):
                yield Button(t("Delete", "삭제"), id="delete-btn", variant="error")
                yield Button(t("Cancel", "취소"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#delete-btn")
    def _on_delete(self, event: Button.Pressed) -> None:
        cfg = self._profile.config_name
        other_refs = [
            n for n in list_profile_names()
            if n != self._profile_name and load_profile(n).config_name == cfg
        ] if cfg else []
        delete_config_too = bool(cfg) and not other_refs
        delete_profile(self._profile_name, delete_config_too=delete_config_too)
        # backend 가 example.yaml 삭제를 스킵하므로 notify 도 맞춰야 한다.
        if delete_config_too and cfg != "example":
            self.app.notify(t(f"Deleted: {self._profile_name} + config '{cfg}'", f"삭제: {self._profile_name} + config '{cfg}'"))
        elif cfg == "example":
            self.app.notify(
                t(
                    f"Deleted: {self._profile_name} (tracked template config 'example' kept)",
                    f"삭제: {self._profile_name} (tracked 템플릿 config 'example' 유지)",
                )
            )
        else:
            self.app.notify(t(f"Deleted: {self._profile_name}", f"삭제: {self._profile_name}"))
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
