"""공통 위젯 — 두 backend 가 공유할 modal 패턴."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option


class BackendPickerModal(ModalScreen[str]):
    """새 프로필 생성 시 backend 선택 모달. 반환값: 'vllm' | 'llamacpp' | ''."""

    BINDINGS = [
        Binding("escape,q", "cancel", "Cancel", show=False),
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
