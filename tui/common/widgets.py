"""공통 위젯 — 두 backend 가 공유할 modal 패턴."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


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
            yield Static("Select backend", id="picker-title")
            yield OptionList(
                Option("vLLM", id="vllm"),
                Option("llama.cpp", id="llamacpp"),
                id="picker-list",
            )
            yield Static(
                "[dim]1/2 or ↑↓ + Enter · esc to cancel[/dim]",
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
        confirm_label: str = "Yes",
        cancel_label: str = "Cancel",
        variant: str = "error",
    ) -> None:
        super().__init__()
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._variant = variant

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._message, id="confirm-message"),
            Horizontal(
                Button(self._confirm_label, id="confirm-yes", variant=self._variant),  # type: ignore[arg-type]
                Button(self._cancel_label, id="confirm-no", variant="default"),
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

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        message: str,
        *,
        default: str = "",
        placeholder: str = "",
        confirm_label: str = "OK",
        cancel_label: str = "Cancel",
    ) -> None:
        super().__init__()
        self._message = message
        self._default = default
        self._placeholder = placeholder
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._message, id="confirm-message"),
            Input(
                value=self._default,
                placeholder=self._placeholder,
                id="prompt-input",
            ),
            Horizontal(
                Button(self._confirm_label, id="prompt-ok", variant="primary"),
                Button(self._cancel_label, id="prompt-cancel", variant="default"),
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
