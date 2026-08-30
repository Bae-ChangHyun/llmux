"""Profile management screens - form for create/edit and delete confirmation."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Label, Input, Select, Switch, TextArea
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on

from tui.backends.vllm.backend import (
    Profile,
    load_profile,
    save_profile,
    delete_profile,
    list_config_names,
    list_profile_names,
    validate_name as _validate_name,
)
from tui.common import profile_store
from tui.common.i18n import t


def _default_port() -> str:
    """Backend default port, honoring a `defaults:` override in profiles.yaml."""
    from tui.common import profile_store

    return str(profile_store.effective_defaults("vllm")["port"])


class ProfileFormScreen(ModalScreen[str | None]):
    """Modal form for creating or editing a profile.

    Pass a Profile object to edit it; omit for a new blank form.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    ProfileFormScreen {
        align: center middle;
    }
    ProfileFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 70;
        min-width: 45;
        /* No max-height — short terminals need the modal to grow so the
           inner VerticalScroll can actually scroll the form. */
        height: 95%;
        min-height: 12;
    }
    ProfileFormScreen VerticalScroll {
        height: 1fr;
        min-height: 3;
    }
    ProfileFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row Label {
        width: 22;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ProfileFormScreen #env-vars-section {
        margin-top: 1;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ProfileFormScreen #env-vars-title {
        text-style: bold;
        color: $text;
    }
    ProfileFormScreen .env-var-line {
        color: $text-muted;
        height: auto;
        margin-left: 2;
    }
    ProfileFormScreen #env-vars-input {
        height: 6;
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
        # In edit mode the name field stays editable: a changed name means a
        # rename, and the original is what tells the two apart at save time.
        self._original_name = profile.name if profile is not None else ""

    def compose(self) -> ComposeResult:
        p = self._profile
        title = (
            t(f"Edit Profile: {p.name}", f"프로필 편집: {p.name}")
            if self._edit_mode
            else t("New Profile", "새 프로필")
        )

        configs = list_config_names()
        config_options: list[tuple[str, str]] = [(name, name) for name in configs]

        # A dangling link (config YAML deleted out from under the profile) has
        # no matching option, so the Select would fall back to BLANK and saving
        # would silently rewrite config_name to "". Surface it instead.
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
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Container Name", "컨테이너 이름"))
                    yield Input(
                        value=p.container_name if p else "",
                        placeholder="container-name",
                        id="container-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Port")
                    yield Input(
                        value=p.port if p else "",
                        placeholder=_default_port(),
                        id="port-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("GPU ID")
                    yield Input(
                        value=p.gpu_id if p else "",
                        placeholder="0",
                        id="gpu-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Tensor Parallel", "텐서 병렬"))
                    yield Input(
                        value=p.tensor_parallel if p else "",
                        placeholder="1",
                        id="tp-input",
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
                    yield Label("Model ID")
                    yield Input(
                        value=(p.model_id if p else ""),
                        placeholder=t(
                            "org/model-name (used for auto config)",
                            "org/model-name (자동 config 생성에 사용)",
                        ),
                        id="model-id-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Enable LoRA", "LoRA 사용"))
                    yield Switch(
                        value=(p.enable_lora == "true") if p else False,
                        id="lora-switch",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Extra Pip Packages", "추가 Pip 패키지"))
                    yield Input(
                        value=(p.extra_pip_packages if p else ""),
                        placeholder=t("e.g. flash-attn bitsandbytes", "예: flash-attn bitsandbytes"),
                        id="extra-pip-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label(t("Image Tag", "이미지 태그"))
                    yield Input(
                        value=p.image_tag if p else "",
                        placeholder=t(
                            "(blank = default image) e.g. myregistry/vllm:custom",
                            "(비우면 기본 이미지) 예: myregistry/vllm:custom",
                        ),
                        id="image-tag-input",
                    )

                yield Label(
                    t("Environment Variables (one KEY=VALUE per line)", "환경 변수 (한 줄에 KEY=VALUE 하나)"),
                    id="env-vars-title",
                )
                yield TextArea(
                    profile_store.format_env_vars_text(p.env_vars if p else {}),
                    id="env-vars-input",
                )

            with Horizontal(classes="form-buttons"):
                yield Button(t("Save", "저장"), id="save-btn", variant="primary")
                yield Button(t("Close", "닫기"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    async def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        container = self.query_one("#container-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_id = self.query_one("#gpu-input", Input).value.strip()
        tp = self.query_one("#tp-input", Input).value.strip()
        lora = self.query_one("#lora-switch", Switch).value

        original_gpu = self._profile.gpu_id if self._profile is not None else "0"
        original_tp = self._profile.tensor_parallel if self._profile is not None else "1"
        if (gpu_id or "0") != original_gpu and (tp or "1") == original_tp:
            tp = str(len((gpu_id or "0").split(",")))

        config_select = self.query_one("#config-select", Select)
        config_name = str(config_select.value) if config_select.value != Select.BLANK else ""
        model_id = self.query_one("#model-id-input", Input).value.strip()

        if not name:
            self.notify(t("Profile name is required.", "프로필 이름은 필수입니다."),
                        severity="error")
            return
        if not _validate_name(name):
            self.notify(
                t(
                    "Name must be lowercase: start with [a-z0-9], then lowercase letters, digits, dashes, or underscores only.",
                    "이름은 소문자여야 합니다: [a-z0-9] 로 시작하고 소문자·숫자·대시·언더스코어만 사용하세요.",
                ),
                severity="error",
            )
            return

        # Profile names must be globally unique across both backends (see
        # profile_store.find_name_owner) — container_name defaults to the name.
        renaming = self._edit_mode and name != self._original_name
        if not self._edit_mode or renaming:
            owner = profile_store.find_name_owner(name)
            if owner is not None:
                self.notify(
                    t(
                        f"Profile '{name}' already exists (backend {owner}).",
                        f"프로필 '{name}' 이(가) 이미 존재합니다 (backend {owner}).",
                    ),
                    severity="error",
                )
                return

        if (not self._edit_mode or renaming) and name == "example":
            # The config link defaults to the profile name, so an 'example'
            # profile would write its params into the tracked example.yaml.
            self.notify(
                t(
                    "'example' is the tracked template config name — pick another.",
                    "'example' 은 추적되는 템플릿 config 이름입니다 — 다른 이름을 쓰세요.",
                ),
                severity="error",
            )
            return

        if container and not _validate_name(container):
            self.notify(
                t(
                    "Container name must be lowercase: start with [a-z0-9], then lowercase letters, digits, dashes, or underscores only.",
                    "컨테이너 이름은 소문자여야 합니다: [a-z0-9] 로 시작하고 소문자·숫자·대시·언더스코어만 사용하세요.",
                ),
                severity="error",
            )
            return

        if port:
            try:
                port_int = int(port)
                if not (1024 <= port_int <= 65535):
                    raise ValueError
            except ValueError:
                self.notify(t("Port must be a number between 1024 and 65535.", "Port 는 1024 ~ 65535 사이의 숫자여야 합니다."),
                            severity="error")
                return

        if gpu_id and not re.match(r"^[0-9]+(,[0-9]+)*$", gpu_id):
            self.notify(t("GPU ID must contain only digits and commas.", "GPU ID 는 숫자와 콤마만 사용할 수 있습니다."),
                        severity="error")
            return

        if tp:
            try:
                tp_int = int(tp)
                if tp_int < 1:
                    raise ValueError
            except ValueError:
                self.notify(t("Tensor Parallel must be a positive integer.", "텐서 병렬은 양의 정수여야 합니다."),
                            severity="error")
                return

        extra_pip = self.query_one("#extra-pip-input", Input).value.strip()
        image_tag = self.query_one("#image-tag-input", Input).value.strip()
        try:
            env_vars = profile_store.parse_env_vars_text(
                self.query_one("#env-vars-input", TextArea).text,
                "vllm",
            )
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        from tui.common.dev_build import image_tag_error

        tag_err = image_tag_error(image_tag)
        if tag_err:
            self.notify(tag_err, severity="error")
            return

        if renaming and not await self._rename_to(name):
            return

        if self._edit_mode and self._profile is not None:
            profile = self._profile
            profile.container_name = container or name
            profile.port = port or _default_port()
            profile.gpu_id = gpu_id or "0"
            profile.tensor_parallel = tp or "1"
            profile.config_name = config_name
            profile.model_id = model_id
            profile.enable_lora = "true" if lora else "false"
            profile.extra_pip_packages = extra_pip
            profile.image_tag = image_tag
            profile.env_vars = env_vars
        else:
            profile = Profile(
                name=name,
                container_name=container or name,
                port=port or _default_port(),
                gpu_id=gpu_id or "0",
                tensor_parallel=tp or "1",
                config_name=config_name,
                model_id=model_id,
                enable_lora="true" if lora else "false",
                extra_pip_packages=extra_pip,
                image_tag=image_tag,
                env_vars=env_vars,
            )

        try:
            save_profile(profile)
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(t(f"Saved: {name}", f"저장됨: {name}"), severity="information")
        self._saved_name = name

        if not self._edit_mode:
            self._edit_mode = True
            self._profile = profile
            self._original_name = name
            self.query_one("#form-title", Static).update(
                t(f"[b]Edit Profile: {name}[/b]", f"[b]프로필 편집: {name}[/b]")
            )

    async def _rename_to(self, new_name: str) -> bool:
        """Rename the profile being edited. False = refused, caller must abort.

        The container is named after the profile, so renaming under a running
        one would orphan it — same rule the dashboard's `R` key enforces.
        """
        from tui.common import docker as common_docker

        old_name = self._original_name
        container = self._profile.container_name or old_name
        try:
            running = await common_docker.running_container_names()
        except Exception as exc:
            self.notify(
                t(
                    f"Could not check running containers ({exc}); rename aborted.",
                    f"실행 중 컨테이너를 확인할 수 없습니다 ({exc}). 이름 변경을 중단합니다.",
                ),
                severity="error",
            )
            return False
        if container in running:
            self.notify(
                t(
                    f"Container '{container}' is running; stop it before renaming.",
                    f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 이름을 변경하세요.",
                ),
                severity="error",
            )
            return False
        try:
            profile_store.rename_profile(old_name, new_name, "vllm")
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return False

        self._original_name = new_name
        self._profile.name = new_name
        self.query_one("#form-title", Static).update(
            t(f"[b]Edit Profile: {new_name}[/b]", f"[b]프로필 편집: {new_name}[/b]")
        )
        return True

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


class ProfileDeleteScreen(ModalScreen[bool]):
    """Simple confirmation modal for deleting a profile."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ProfileDeleteScreen {
        align: center middle;
    }
    ProfileDeleteScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    ProfileDeleteScreen #delete-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileDeleteScreen .form-buttons {
        height: 1;
        margin-top: 1;
        align: center middle;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name
        self._profile = load_profile(profile_name)

    def compose(self) -> ComposeResult:
        with Vertical():
            if self._profile.config_name:
                detail = t(
                    f"(profile only; linked config kept: {self._profile.config_name})",
                    f"(프로필만 삭제; 연결된 config 유지: {self._profile.config_name})",
                )
                yield Static(
                    t(
                        f"Delete [b]{self._profile_name}[/b]?\n{detail}",
                        f"[b]{self._profile_name}[/b] 삭제?\n{detail}",
                    ),
                    id="delete-message",
                )
            else:
                yield Static(
                    t(
                        f"Delete profile [b]{self._profile_name}[/b]?",
                        f"프로필 [b]{self._profile_name}[/b] 삭제?",
                    ),
                    id="delete-message",
                )
            with Horizontal(classes="form-buttons"):
                yield Button(t("Delete", "삭제"), id="delete-btn", variant="error")
                yield Button(t("Cancel", "취소"), id="cancel-btn", variant="default")

    @on(Button.Pressed, "#delete-btn")
    async def _on_delete(self, event: Button.Pressed) -> None:
        from tui.common import docker as common_docker

        container = self._profile.container_name or self._profile_name
        try:
            running = await common_docker.running_container_names()
        except Exception as exc:
            self.app.notify(
                t(
                    f"Could not verify container state ({exc}); deletion aborted.",
                    f"컨테이너 상태를 확인할 수 없습니다 ({exc}). 삭제를 중단합니다.",
                ),
                severity="error",
            )
            return
        if container in running:
            self.app.notify(
                t(
                    f"Container '{container}' is running; stop it before deleting.",
                    f"컨테이너 '{container}' 가 실행 중입니다. 중지 후 삭제하세요.",
                ),
                severity="error",
            )
            return
        delete_profile(self._profile_name, delete_config=False)
        suffix = (
            t(
                f"; linked config kept: {self._profile.config_name}",
                f"; 연결된 config 유지: {self._profile.config_name}",
            )
            if self._profile.config_name
            else ""
        )
        self.app.notify(
            t(f"Deleted profile: {self._profile_name}", f"프로필 삭제됨: {self._profile_name}")
            + suffix
        )
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
