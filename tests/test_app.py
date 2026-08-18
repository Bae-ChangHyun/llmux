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


class ProfileFormRenameTests(unittest.IsolatedAsyncioTestCase):
    """The edit form's Name field is editable: changing it renames the profile
    instead of creating a second one."""

    @contextlib.contextmanager
    def _env(self, root: Path, running: set[str]):
        profiles_yaml = root / "profiles.yaml"
        profiles_yaml.write_text(
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: solo\n  backend: vllm\n  port: 8123\n"
        )
        with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), \
            patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"), \
            patch(
                "tui.common.docker.running_container_names",
                AsyncMock(return_value=running),
            ):
            yield

    async def test_editing_name_renames_in_place(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.backend import load_profile
        from tui.backends.vllm.screens.profile import ProfileFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, set()):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ProfileFormScreen(load_profile("solo"))
                    await app.push_screen(screen)
                    await pilot.pause()

                    name_input = screen.query_one("#name-input", Input)
                    self.assertFalse(name_input.disabled)  # the whole point
                    name_input.value = "renamed"
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Renamed, not duplicated.
                self.assertIsNone(profile_store.load_profile("solo", "vllm"))
                moved = profile_store.load_profile("renamed", "vllm")
                self.assertIsNotNone(moved)
                self.assertEqual(moved.port, 8123)
                self.assertEqual(
                    len(profile_store.list_profiles("vllm")), 1
                )

    async def test_editing_name_refused_while_running(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.backend import load_profile
        from tui.backends.vllm.screens.profile import ProfileFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._env(root, {"solo"}):
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ProfileFormScreen(load_profile("solo"))
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen.query_one("#name-input", Input).value = "renamed"
                    await pilot.pause()
                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Nothing moved, and no stray second profile was written.
                self.assertIsNotNone(profile_store.load_profile("solo", "vllm"))
                self.assertIsNone(profile_store.load_profile("renamed", "vllm"))
                self.assertEqual(len(profile_store.list_profiles("vllm")), 1)


class ConfigFormRenameTests(unittest.IsolatedAsyncioTestCase):
    """The config form's Name field is editable too: changing it renames the
    YAML and repoints referencing profiles, instead of forking a copy."""

    async def test_editing_name_renames_and_repoints(self) -> None:
        from textual.widgets import Input
        from tui.backends.vllm.screens.config import ConfigFormScreen
        from tui.common import profile_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "config"
            cdir.mkdir()
            (cdir / "cfg.yaml").write_text("model: m/x\ngpu-memory-utilization: '0.9'\n")
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
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    name_input = screen.query_one("#name-input", Input)
                    self.assertFalse(name_input.disabled)
                    name_input.value = "cfg2"
                    await pilot.pause()

                    screen.query_one("#save-btn").press()
                    await pilot.pause()
                    await pilot.pause()

                # Renamed, not duplicated, and the profile follows.
                self.assertFalse((cdir / "cfg.yaml").exists())
                self.assertTrue((cdir / "cfg2.yaml").exists())
                self.assertEqual(
                    profile_store.load_profile("p", "vllm").config_name, "cfg2"
                )


class MonitorScreenTests(unittest.IsolatedAsyncioTestCase):
    """The live monitor renders real metric values from /metrics + nvidia-smi."""

    def _row(self, backend="vllm"):
        from tui.common.adapter import DashboardRow
        return DashboardRow(
            backend=backend, profile_name="m1", container_name="m1",
            port=8000, running=True, model="m1", detail="", gpu_id="0",
        )

    async def test_renders_metrics_and_gpu(self) -> None:
        import io
        from rich.console import Console
        from tui.common.metrics import MetricsSnapshot
        from tui.common.docker import GpuInfo
        from tui.screens.monitor import MonitorScreen
        from textual.widgets import Static

        snap = MetricsSnapshot(
            backend="vllm", prompt_tokens=100.0, generation_tokens=200.0,
            requests_running=3.0, requests_waiting=1.0, kv_cache_usage=0.34,
        )
        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]

        with patch("tui.common.plain_monitor.fetch_snapshot",
                   AsyncMock(return_value=snap)), \
            patch("tui.common.plain_monitor._running_rows",
                  AsyncMock(return_value=([self._row()], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=gpus)), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(self._row())
                await app.push_screen(screen)
                for _ in range(30):
                    await pilot.pause()
                    if screen._rendered is not None:
                        break

                con = Console(width=120, file=io.StringIO())
                con.print(screen._rendered)
                out = con.file.getvalue()
                self.assertIn("34%", out)
                self.assertIn("GPU0", out)
                self.assertIn("210W", out)
                self.assertIn("REQUESTS", out)

    async def test_unreachable_server_does_not_crash(self) -> None:
        from tui.screens.monitor import MonitorScreen
        from textual.widgets import Static

        with patch("tui.common.plain_monitor.fetch_snapshot",
                   AsyncMock(return_value=None)), \
            patch("tui.common.plain_monitor._running_rows",
                  AsyncMock(return_value=([self._row()], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(self._row())
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()
                # Renders (with unreachable server) rather than raising.
                self.assertIsInstance(screen.query_one("#mon", Static), Static)

    async def test_opens_with_nothing_running_and_shows_gpu(self) -> None:
        """`v` must not be gated on a running container — the GPU panel is the
        whole point of opening the monitor when nothing is up."""
        import io
        from rich.console import Console
        from tui.common.docker import GpuInfo
        from tui.screens.monitor import MonitorScreen

        gpus = [GpuInfo("0", "RTX", "8000", "16000", "78", "71", "210")]
        with patch("tui.common.plain_monitor._running_rows", AsyncMock(return_value=([], []))), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=gpus)), \
            patch("tui.common.docker.get_pcie_stats", AsyncMock(return_value={})):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = MonitorScreen(None)
                await app.push_screen(screen)
                for _ in range(30):
                    await pilot.pause()
                    if screen._rendered is not None:
                        break
                con = Console(width=120, file=io.StringIO())
                con.print(screen._rendered)
                out = con.file.getvalue()
                self.assertIn("GPU0", out)
                self.assertIn("210W", out)


class VersionScreenEnterTests(unittest.IsolatedAsyncioTestCase):
    """Enter on the version picker commits the highlighted option and starts,
    instead of only selecting the radio."""

    def _profile(self):
        import types
        return types.SimpleNamespace(
            name="p", image_tag="", port=8000, container_name="p",
            config_name="p", gpu_id="0", model_id="", enable_lora="false",
            tensor_parallel="1", extra_pip_packages="",
        )

    async def test_enter_commits_selection_and_starts(self) -> None:
        from tui.backends.vllm.screens import container as mod
        from tui.backends.vllm.screens.container import ContainerUpScreen, VER_OFFICIAL
        from textual.widgets import RadioSet, RadioButton

        with patch.object(mod, "load_profile", lambda n: self._profile()), \
            patch.object(mod, "get_local_latest_tag", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_release_version", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_nightly_date", AsyncMock(return_value="available")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()

                rs = screen.query_one("#version-radio", RadioSet)
                rs.query_one(f"#{VER_OFFICIAL}", RadioButton).value = True
                await pilot.pause()

                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                self.assertEqual(called, [True])

    async def test_enter_on_empty_custom_focuses_input_not_start(self) -> None:
        from tui.backends.vllm.screens import container as mod
        from tui.backends.vllm.screens.container import ContainerUpScreen, VER_CUSTOM
        from textual.widgets import RadioSet, RadioButton, Input

        with patch.object(mod, "load_profile", lambda n: self._profile()), \
            patch.object(mod, "get_local_latest_tag", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_release_version", AsyncMock(return_value="v0.1.0")), \
            patch.object(mod, "get_dockerhub_nightly_date", AsyncMock(return_value="available")), \
            patch.object(mod, "get_gpu_info", AsyncMock(return_value=[])):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                screen = ContainerUpScreen("p")
                await app.push_screen(screen)
                await pilot.pause()
                await pilot.pause()

                rs = screen.query_one("#version-radio", RadioSet)
                rs.query_one(f"#{VER_CUSTOM}", RadioButton).value = True
                await pilot.pause()

                called = []
                screen._do_start = lambda: called.append(True)
                screen.action_confirm_start()
                # Empty custom tag → don't start, focus the input.
                self.assertEqual(called, [])
                self.assertIs(app.focused, screen.query_one("#custom-tag-input", Input))


class PlainModeToggleTests(unittest.IsolatedAsyncioTestCase):
    """The `t` key suspends the TUI, runs the plain dashboard, then reloads."""

    async def test_plain_mode_suspends_runs_monitor_and_reloads(self) -> None:
        import contextlib
        from unittest.mock import AsyncMock, MagicMock, patch
        from tui.screens.dashboard import DashboardScreen
        from tui.common.adapter import DashboardRow

        row = DashboardRow(backend="vllm", profile_name="m", container_name="m",
                           port=8000, running=True, model="m", detail="", gpu_id="0")
        with patch("tui.common.docker.running_container_names", AsyncMock(return_value=set())), \
            patch("tui.common.docker.get_gpu_info", AsyncMock(return_value=[])), \
            patch("tui.common.plain_monitor.run_plain_monitor", AsyncMock()) as run, \
            patch("tui.app.LlmuxApp.suspend", lambda self: contextlib.nullcontext()):
            app = LlmuxApp()
            async with app.run_test(size=WIDE) as pilot:
                await pilot.pause()
                dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
                dash._selected_row = MagicMock(return_value=row)
                dash._reload = MagicMock()
                await dash.action_plain_mode()
                run.assert_awaited_once()
                dash._reload.assert_called_once()


class ConfigFormRecipeMergeTests(unittest.IsolatedAsyncioTestCase):
    """Edit Config can pull a recipe: its params merge into the open form,
    updating rows that already exist instead of duplicating them."""

    async def test_apply_recipe_updates_and_adds_rows(self) -> None:
        from textual.widgets import Input, Switch
        from tui.backends.vllm.screens.config import ConfigFormScreen

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            with _patched_vllm_config_dir(cdir):
                (cdir / "cfg.yaml").write_text(
                    "model: m/x\ngpu-memory-utilization: '0.9'\nmax-model-len: 4096\n"
                )
                app = LlmuxApp()
                async with app.run_test(size=WIDE) as pilot:
                    await pilot.pause()
                    screen = ConfigFormScreen("cfg")
                    await app.push_screen(screen)
                    await pilot.pause()

                    screen._apply_recipe(
                        "org/other", {"max-model-len": 8192, "quantization": "awq"}
                    )
                    await pilot.pause()

                    rows = {
                        row.query_one(".param-key", Input).value: row
                        for row in screen.query(".param-row")
                    }
                    self.assertEqual(
                        rows["max-model-len"].query_one(".param-value", Input).value,
                        "8192",
                    )
                    self.assertTrue(
                        rows["max-model-len"].query_one(".param-switch", Switch).value
                    )
                    self.assertIn("quantization", rows)
                    self.assertEqual(
                        screen.query_one("#model-input", Input).value, "org/other"
                    )


class ActionMenuPrepareTests(unittest.IsolatedAsyncioTestCase):
    """`prepare` is offered only while the container is stopped — there is
    nothing to pre-download for a profile that is already serving."""

    async def test_vllm_menu_offers_prepare_only_when_stopped(self) -> None:
        from tui.backends.vllm.screens.dashboard import ProfileActionScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            for running, expected in ((False, True), (True, False)):
                screen = ProfileActionScreen("p", running)
                await app.push_screen(screen)
                await pilot.pause()
                ids = {o.id for o in screen.query_one("#action-list").options}
                self.assertEqual("prepare" in ids, expected)
                app.pop_screen()
                await pilot.pause()

    async def test_llamacpp_menu_offers_prepare_only_when_stopped(self) -> None:
        from tui.backends.llamacpp import backend as lbackend
        from tui.backends.llamacpp.screens.dashboard import ActionModal

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            for running, expected in ((False, True), (True, False)):
                profile = lbackend.Profile(name="p", container_name="p", port=8080)
                profile.running = running
                screen = ActionModal(profile)
                await app.push_screen(screen)
                await pilot.pause()
                ids = {o.id for o in screen.query_one("#action-list").options}
                self.assertEqual("prepare" in ids, expected)
                app.pop_screen()
                await pilot.pause()


class QuickSetupRecipeSourceTests(unittest.IsolatedAsyncioTestCase):
    """Quick Setup can pull the recipe of a different model than the one being
    configured (a quantized checkpoint borrowing its base model's recipe)."""

    async def test_source_input_drives_the_fetch(self) -> None:
        from unittest.mock import MagicMock

        from textual.widgets import Input
        from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            screen = QuickSetupScreen()
            await app.push_screen(screen)
            await pilot.pause()

            screen.query_one("#model-input", Input).value = "cpatonn/Qwen3-32B-AWQ"
            screen.query_one("#recipe-model-input", Input).value = "Qwen/Qwen3-32B"
            screen._fetch_recipe = MagicMock()
            screen.query_one("#fetch-recipe-btn").press()
            await pilot.pause()

            screen._fetch_recipe.assert_called_once_with(
                "Qwen/Qwen3-32B", "cpatonn/Qwen3-32B-AWQ"
            )

    async def test_blank_source_falls_back_to_the_model(self) -> None:
        from unittest.mock import MagicMock

        from textual.widgets import Input
        from tui.backends.vllm.screens.quick_setup import QuickSetupScreen

        app = LlmuxApp()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.pause()
            screen = QuickSetupScreen()
            await app.push_screen(screen)
            await pilot.pause()

            screen.query_one("#model-input", Input).value = "Qwen/Qwen3-32B"
            screen._fetch_recipe = MagicMock()
            screen.query_one("#fetch-recipe-btn").press()
            await pilot.pause()

            screen._fetch_recipe.assert_called_once_with(
                "Qwen/Qwen3-32B", "Qwen/Qwen3-32B"
            )
