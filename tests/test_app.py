"""Width-guard (TooNarrowScreen) behaviour, driven through Textual's pilot.

Textual delivers the initial Resize *before* on_mount, so a terminal that
starts out narrow used to bury the guard underneath the dashboard: the stack
came out as [_default, TooNarrow, Dashboard], the guard was invisible, and F1
was a permanent no-op for the rest of the session.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tui.app import LlmuxApp
from tui.screens.dashboard import DashboardScreen
from tui.screens.too_narrow import TooNarrowScreen

NARROW = (60, 30)
WIDE = (120, 30)


class WidthGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_narrow_launch_puts_guard_on_top(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, TooNarrowScreen)
            # Exactly one guard, and it sits above the dashboard.
            guards = [s for s in app.screen_stack if isinstance(s, TooNarrowScreen)]
            self.assertEqual(len(guards), 1)
            self.assertTrue(
                any(isinstance(s, DashboardScreen) for s in app.screen_stack)
            )

    async def test_widening_after_narrow_launch_releases_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, TooNarrowScreen)

            await pilot.resize_terminal(*WIDE)
            await pilot.pause()

            self.assertIsInstance(app.screen, DashboardScreen)
            self.assertIsNone(app._too_narrow)
            self.assertFalse(
                any(isinstance(s, TooNarrowScreen) for s in app.screen_stack)
            )

    async def test_f1_works_after_narrow_launch_and_widen(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(*WIDE)
            await pilot.pause()

            # F1 used to be dead for the whole session because `_too_narrow`
            # stayed set on a guard that could never surface.
            app.push_screen("vllm_configs")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, DashboardScreen)

            await pilot.press("f1")
            await pilot.pause()
            self.assertIsInstance(app.screen, DashboardScreen)

    async def test_renarrow_does_not_stack_a_second_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(*WIDE)
            await pilot.pause()
            await pilot.resize_terminal(*NARROW)
            await pilot.pause()

            self.assertIsInstance(app.screen, TooNarrowScreen)
            guards = [s for s in app.screen_stack if isinstance(s, TooNarrowScreen)]
            self.assertEqual(len(guards), 1)

    async def test_wide_launch_has_no_guard(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, DashboardScreen)
            self.assertIsNone(app._too_narrow)

    async def test_f1_leaves_guard_in_place_while_narrow(self) -> None:
        app = LlmuxApp()
        async with app.run_test(size=NARROW) as pilot:
            await pilot.pause()
            await pilot.press("f1")
            await pilot.pause()
            # The guard outranks F1 — popping it would expose the clipped
            # screens it exists to hide.
            self.assertIsInstance(app.screen, TooNarrowScreen)


@contextlib.contextmanager
def _patched_vllm_config_dir(path: Path):
    """Point every module that holds a vLLM CONFIG_DIR reference at `path`."""
    from tui.backends.vllm import backend, backend_common, backend_storage
    from tui.backends.vllm.screens import config as config_screen

    with contextlib.ExitStack() as stack:
        for mod in (backend, backend_common, backend_storage, config_screen):
            stack.enter_context(patch.object(mod, "CONFIG_DIR", path))
        yield


class ConfigFormDisableSwitchTests(unittest.IsolatedAsyncioTestCase):
    """The config form's per-row Switch decides active vs disabled; toggling it
    off and saving must write a disabled marker, and re-opening must restore the
    off state."""

    async def test_toggle_off_saves_marker_and_restores(self) -> None:
        from textual.widgets import Input, Switch
        from tui.backends.vllm.backend_common import CONFIG_DIR as _real  # noqa: F401
        from tui.backends.vllm.screens.config import ConfigFormScreen
        from tui.backends.vllm import backend_storage as vs

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            with _patched_vllm_config_dir(cdir):
                # Seed a config with two active params.
                (cdir / "cfg.yaml").write_text(
                    "model: m/x\ngpu-memory-utilization: '0.9'\nmax-model-len: 4096\n"
                )

                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    # Find the max-model-len row's switch and turn it off.
                    target = None
                    for row in screen.query(".param-row"):
                        if row.query_one(".param-key", Input).value == "max-model-len":
                            target = row.query_one(".param-switch", Switch)
                    self.assertIsNotNone(target)
                    target.value = False
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()

                text = (cdir / "cfg.yaml").read_text()
                self.assertIn("# llmux:disabled max-model-len", text)

                # Reload: the param is disabled, so a fresh form shows it off.
                cfg = vs.load_config("cfg")
                self.assertIn("max-model-len", cfg.disabled_params)
                self.assertNotIn("max-model-len", cfg.extra_params)


if __name__ == "__main__":
    unittest.main()
