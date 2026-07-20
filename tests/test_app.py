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
from unittest.mock import AsyncMock, patch

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


class ConfigListRenameTests(unittest.IsolatedAsyncioTestCase):
    """`R` on the config list renames the YAML and repoints referencing
    profiles — and refuses while a referencing profile's container is up."""

    async def test_rename_moves_file_and_notifies(self) -> None:
        from tui.backends.vllm.screens.config import ConfigListScreen

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\n")
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults: {}\nprofiles:\n"
                "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )

            with _patched_vllm_config_dir(cdir), \
                patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
                patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
                patch("tui.common.config_store.config_dir", lambda b: cdir), \
                patch(
                    "tui.common.docker.running_container_names",
                    AsyncMock(return_value=set()),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigListScreen()
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._rename_config("cfg", "cfg2")
                    await pilot.pause()
                    await pilot.pause()

                self.assertFalse((cdir / "cfg.yaml").exists())
                self.assertTrue((cdir / "cfg2.yaml").exists())
                from tui.common import profile_store

                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg2"
                )

    async def test_rename_refuses_while_referencing_container_runs(self) -> None:
        from tui.backends.vllm.screens.config import ConfigListScreen

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\n")
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\ndefaults: {}\nprofiles:\n"
                "- name: p\n  backend: vllm\n  config_name: cfg\n"
            )

            with _patched_vllm_config_dir(cdir), \
                patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
                patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
                patch("tui.common.config_store.config_dir", lambda b: cdir), \
                patch(
                    "tui.common.docker.running_container_names",
                    AsyncMock(return_value={"p"}),
                ):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigListScreen()
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._rename_config("cfg", "cfg2")
                    await pilot.pause()
                    await pilot.pause()

                self.assertTrue((cdir / "cfg.yaml").exists())
                self.assertFalse((cdir / "cfg2.yaml").exists())


class DashboardRenameProfileTests(unittest.IsolatedAsyncioTestCase):
    """`R` on the dashboard renames a stopped profile; a running one is refused."""

    @contextlib.contextmanager
    def _env(self, root: Path, running: set[str]):
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text(
            "version: 1\ndefaults: {}\nprofiles:\n- name: solo\n  backend: vllm\n"
        )
        with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
            patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value=running),
            ), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])):
            yield

    async def test_r_opens_prompt_and_rename_applies(self) -> None:
        from tui.common import profile_store
        from tui.common.widgets import TextPromptModal

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, set()):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    await pilot.press("R")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, TextPromptModal)

                    app.screen.dismiss("renamed")
                    await pilot.pause()
                    await pilot.pause()

                self.assertIsNone(profile_store.load_profile("solo", "vllm"))
                moved = profile_store.load_profile("renamed", "vllm")
                self.assertIsNotNone(moved)
                # An unset config link is pinned to the old name so the profile
                # keeps resolving to the config file it already used.
                self.assertEqual(moved.config_name, "solo")

    async def test_r_refused_while_container_runs(self) -> None:
        from tui.common import profile_store
        from tui.common.widgets import TextPromptModal

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, {"solo"}):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    await pilot.press("R")
                    await pilot.pause()
                    # No prompt at all — the guard fires before any input.
                    self.assertNotIsInstance(app.screen, TextPromptModal)

                self.assertIsNotNone(profile_store.load_profile("solo", "vllm"))


if __name__ == "__main__":
    unittest.main()
