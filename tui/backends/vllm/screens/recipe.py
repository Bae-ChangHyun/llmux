"""Recipe review modal — shown after a vLLM recipe is fetched, before it's applied.

The recipe carries several precision variants (bf16 / fp8 / awq …) with different
VRAM floors, and the user's GPU may not match the one the recipe was verified on
— so we don't silently pick one. This screen lays out every variant against the
detected GPU, lets the user toggle opt-in features, previews the exact flags that
will be written, and only then creates the profile + config.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RadioButton, RadioSet, Static, Switch

from tui.common.i18n import t
from tui.common.recipes import Recipe, RecipeVariant, build_config


class RecipeReviewScreen(ModalScreen[dict | None]):
    """Review a fetched recipe and pick a GPU-appropriate variant.

    Dismisses with ``{"model_id", "params", "variant", "features"}`` on Create,
    or ``None`` on cancel.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
    ]

    DEFAULT_CSS = """
    RecipeReviewScreen { align: center middle; }
    RecipeReviewScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 84;
        min-width: 55;
        height: 90%;
        min-height: 14;
    }
    RecipeReviewScreen .title {
        text-style: bold; color: $primary; text-align: center;
        width: 100%; margin-bottom: 1;
    }
    RecipeReviewScreen VerticalScroll { height: 1fr; min-height: 5; }
    RecipeReviewScreen Label { margin-top: 1; color: $text-muted; }
    RecipeReviewScreen #recipe-meta { color: $text-muted; margin-bottom: 1; }
    RecipeReviewScreen #version-warn { color: $warning; margin-top: 1; }
    RecipeReviewScreen .feature-row { height: 3; }
    RecipeReviewScreen .feature-row Label { width: 1fr; margin-top: 1; }
    RecipeReviewScreen .feature-row Switch { width: auto; }
    RecipeReviewScreen #preview {
        color: $text; background: $boost; padding: 1;
        margin-top: 1; border: round $primary 40%; height: auto;
    }
    RecipeReviewScreen .buttons {
        height: auto; min-height: 3; margin-top: 1; padding-top: 1;
        align: center middle; background: $surface; border-top: solid $primary 30%;
    }
    """

    def __init__(
        self,
        recipe: Recipe,
        *,
        gpu_total_gb: float | None = None,
        local_vllm_version: str = "",
    ) -> None:
        super().__init__()
        self._recipe = recipe
        self._gpu_total_gb = gpu_total_gb
        self._local_vllm_version = local_vllm_version

    # ----- variant helpers -----

    def _fits(self, v: RecipeVariant) -> bool | None:
        if self._gpu_total_gb is None or v.vram_minimum_gb is None:
            return None
        return v.vram_minimum_gb <= self._gpu_total_gb

    def _default_variant_index(self) -> int:
        """First variant (recipe order = quality-descending) that fits the GPU;
        if none fit, the smallest-VRAM one so the user at least sees a runnable
        option."""
        variants = self._recipe.variants
        for i, v in enumerate(variants):
            if self._fits(v):
                return i
        if variants:
            sized = [(v.vram_minimum_gb or 1e9, i) for i, v in enumerate(variants)]
            return min(sized)[1]
        return 0

    def _variant_label(self, v: RecipeVariant) -> str:
        parts = [v.name]
        if v.precision:
            parts.append(v.precision)
        if v.vram_minimum_gb is not None:
            parts.append(t(f"needs ≥{v.vram_minimum_gb:.0f} GB",
                           f"≥{v.vram_minimum_gb:.0f} GB 필요"))
        fits = self._fits(v)
        if fits is True:
            parts.append("[green]✓[/green]")
        elif fits is False:
            parts.append("[red]✗[/red]")
        label = "  ·  ".join(parts)
        if v.description:
            label += f"\n[dim]{v.description}[/dim]"
        return label

    # ----- compose -----

    def compose(self) -> ComposeResult:
        r = self._recipe
        with Vertical():
            yield Static(
                t(f"vLLM recipe · {r.title or r.model_id}",
                  f"vLLM 레시피 · {r.title or r.model_id}"),
                classes="title",
            )
            with VerticalScroll():
                meta = f"[b]{r.model_id}[/b]"
                if r.description:
                    meta += f"\n[dim]{r.description}[/dim]"
                yield Static(meta, id="recipe-meta")

                if self._gpu_total_gb is not None:
                    yield Static(
                        t(f"[dim]Detected GPU: {self._gpu_total_gb:.0f} GB[/dim]",
                          f"[dim]감지된 GPU: {self._gpu_total_gb:.0f} GB[/dim]"),
                    )

                if r.variants:
                    yield Label(t("Precision variant", "정밀도 변형"))
                    with RadioSet(id="variant-radio"):
                        default_idx = self._default_variant_index()
                        for i, v in enumerate(r.variants):
                            yield RadioButton(
                                self._variant_label(v),
                                id=f"variant-{i}",
                                value=(i == default_idx),
                            )

                if r.features:
                    yield Label(t("Optional features", "선택 기능"))
                    for f in r.features:
                        with Horizontal(classes="feature-row"):
                            desc = f": {f.description}" if f.description else ""
                            yield Label(f"{f.name}[dim]{desc}[/dim]")
                            yield Switch(value=False, id=f"feat-{f.name}")

                if r.min_vllm_version:
                    warn = self._version_warning()
                    if warn:
                        yield Static(warn, id="version-warn")

                yield Label(t("Resulting config", "생성될 config"))
                yield Static("", id="preview")

            with Horizontal(classes="buttons"):
                yield Button(t("Create", "생성"), variant="primary", id="create-btn")
                yield Button(t("Cancel", "취소"), id="cancel-btn")

    def on_mount(self) -> None:
        self._refresh_preview()

    def _version_warning(self) -> str:
        # Purely informational — recipe min-version vs the newest local image.
        rv, lv = self._recipe.min_vllm_version, self._local_vllm_version
        if not rv or not lv:
            return t(f"[dim]Recipe needs vLLM ≥ {rv}[/dim]",
                     f"[dim]레시피 요구 vLLM ≥ {rv}[/dim]") if rv else ""
        return t(
            f"Recipe needs vLLM ≥ {rv} · your latest local image is {lv}",
            f"레시피 요구 vLLM ≥ {rv} · 로컬 최신 이미지는 {lv}",
        )

    # ----- live preview -----

    def _selected_variant(self) -> RecipeVariant | None:
        if not self._recipe.variants:
            return None
        try:
            pressed = self.query_one("#variant-radio", RadioSet).pressed_button
        except Exception:
            return self._recipe.variants[self._default_variant_index()]
        if pressed and pressed.id and pressed.id.startswith("variant-"):
            return self._recipe.variants[int(pressed.id.split("-")[1])]
        return self._recipe.variants[self._default_variant_index()]

    def _enabled_features(self) -> list[str]:
        out = []
        for f in self._recipe.features:
            try:
                if self.query_one(f"#feat-{f.name}", Switch).value:
                    out.append(f.name)
            except Exception:
                pass
        return out

    def _current(self) -> tuple[str, dict]:
        return build_config(self._recipe, self._selected_variant(), self._enabled_features())

    def _refresh_preview(self) -> None:
        try:
            preview = self.query_one("#preview", Static)
        except Exception:
            return
        model_id, params = self._current()
        lines = [f"model: {model_id}"]
        lines += [f"{k}: {_fmt(v)}" for k, v in params.items()]
        preview.update("\n".join(lines))

    @on(RadioSet.Changed, "#variant-radio")
    def _on_variant(self, event: RadioSet.Changed) -> None:
        self._refresh_preview()

    @on(Switch.Changed)
    def _on_feature(self, event: Switch.Changed) -> None:
        self._refresh_preview()

    # ----- actions -----

    @on(Button.Pressed, "#create-btn")
    def _on_create(self, event: Button.Pressed) -> None:
        model_id, params = self._current()
        variant = self._selected_variant()
        self.dismiss({
            "model_id": model_id,
            "params": params,
            "variant": variant.name if variant else "",
            "features": self._enabled_features(),
        })

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_scroll_form(self, direction: str) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
        except Exception:
            return
        if direction == "up":
            scroll.scroll_page_up()
        else:
            scroll.scroll_page_down()


def _fmt(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)
