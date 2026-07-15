"""공통 위젯 — 두 backend 가 공유할 modal 패턴."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tui.common.i18n import t


class BackendPickerModal(ModalScreen[str]):
    """새 프로필 생성 시 backend 선택 모달. 반환값: 'vllm' | 'llamacpp' | ''."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
        Binding("1", "pick('vllm')", show=False),
        Binding("2", "pick('llamacpp')", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Select backend", "백엔드 선택"), id="picker-title")
            yield OptionList(
                Option("vLLM", id="vllm"),
                Option("llama.cpp", id="llamacpp"),
                id="picker-list",
            )
            yield Static(
                t(
                    "[dim]1/2 or ↑↓ + Enter · esc to cancel[/dim]",
                    "[dim]1/2 또는 ↑↓ + Enter · esc 취소[/dim]",
                ),
                id="picker-foot",
            )

    def on_mount(self) -> None:
        self.query_one("#picker-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_pick(self, backend: str) -> None:
        self.dismiss(backend)

    def action_cancel(self) -> None:
        self.dismiss("")


class ConfirmModal(ModalScreen[bool]):
    """공용 확인 다이얼로그. 두 backend 의 destructive 액션에 재사용."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("y", "confirm", "Yes", show=False),
        Binding("n", "cancel", "No", show=False),
    ]

    def __init__(
        self,
        message: str,
        *,
        confirm_label: str | None = None,
        cancel_label: str | None = None,
        variant: str = "error",
    ) -> None:
        super().__init__()
        self._message = message
        # Resolve the default labels lazily in compose(), not here — a t()
        # default argument would be evaluated once at import time and lock the
        # language, defeating the fresh-per-call resolution i18n relies on.
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._variant = variant

    def compose(self) -> ComposeResult:
        confirm = self._confirm_label if self._confirm_label is not None else t("Yes", "네")
        cancel = self._cancel_label if self._cancel_label is not None else t("Cancel", "취소")
        yield Vertical(
            Static(self._message, id="confirm-message"),
            Horizontal(
                Button(confirm, id="confirm-yes", variant=self._variant),  # type: ignore[arg-type]
                Button(cancel, id="confirm-no", variant="default"),
                classes="form-buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class TextPromptModal(ModalScreen[str | None]):
    """공용 텍스트 입력 다이얼로그. 취소 시 None, 확인 시 입력 문자열."""

    # Without an explicit height the generic `ModalScreen > Vertical` rule in
    # app.tcss leaves Vertical at its default `height: 1fr`, so the dialog
    # stretched to nearly the full terminal around three rows of content.
    DEFAULT_CSS = """
    TextPromptModal > Vertical {
        width: 52;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $primary;
    }
    TextPromptModal #prompt-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        message: str,
        *,
        default: str = "",
        placeholder: str = "",
        confirm_label: str | None = None,
        cancel_label: str | None = None,
    ) -> None:
        super().__init__()
        self._message = message
        self._default = default
        self._placeholder = placeholder
        # See ConfirmModal — resolve default labels in compose(), not as t()
        # default arguments (which would freeze the language at import time).
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        confirm = self._confirm_label if self._confirm_label is not None else t("OK", "확인")
        cancel = self._cancel_label if self._cancel_label is not None else t("Cancel", "취소")
        yield Vertical(
            Static(self._message, id="prompt-message"),
            Input(
                value=self._default,
                placeholder=self._placeholder,
                id="prompt-input",
            ),
            Horizontal(
                Button(confirm, id="prompt-ok", variant="primary"),
                Button(cancel, id="prompt-cancel", variant="default"),
                classes="form-buttons",
            ),
        )

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prompt-ok":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        value = self.query_one("#prompt-input", Input).value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
