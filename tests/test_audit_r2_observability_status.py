from __future__ import annotations

import asyncio
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import urllib.error


class MetricsContractTests(unittest.TestCase):
    def test_nonfinite_samples_are_unavailable(self) -> None:
        from tui.common.metrics import parse_snapshot, parse_token_counters

        snapshot = parse_snapshot(
            "llamacpp:predicted_tokens_seconds NaN\n"
            "llamacpp:kv_cache_usage_ratio +Inf\n"
        )

        self.assertIsNone(snapshot.gen_tps_gauge)
        self.assertIsNone(snapshot.kv_cache_usage)
        self.assertIsNone(
            parse_token_counters(
                "vllm:prompt_tokens_total 1\n"
                "vllm:generation_tokens_total NaN\n"
            )
        )

    def test_token_counter_names_match_exactly(self) -> None:
        from tui.common.metrics import parse_token_counters

        self.assertIsNone(
            parse_token_counters(
                "vllm:prompt_tokens_total_shadow 10\n"
                "vllm:generation_tokens_total_shadow 20\n"
            )
        )

    def test_alias_precedence_is_independent_of_exposition_order(self) -> None:
        from tui.common.metrics import parse_snapshot

        snapshot = parse_snapshot(
            "vllm:gpu_cache_usage_perc 0.9\n"
            "vllm:kv_cache_usage_perc 0.2\n"
            "vllm:time_per_output_token_seconds_sum 100\n"
            "vllm:time_per_output_token_seconds_count 1\n"
            "vllm:inter_token_latency_seconds_sum 1\n"
            "vllm:inter_token_latency_seconds_count 1\n"
        )

        self.assertEqual(snapshot.kv_cache_usage, 0.2)
        self.assertIsNotNone(snapshot.tpot)
        assert snapshot.tpot is not None
        self.assertEqual(snapshot.tpot.sum, 1.0)
        self.assertEqual(snapshot.tpot.count, 1.0)

    def test_malformed_preferred_alias_does_not_fall_back_to_legacy(self) -> None:
        from tui.common.metrics import parse_snapshot

        snapshot = parse_snapshot(
            "vllm:gpu_cache_usage_perc 0.9\n"
            "vllm:kv_cache_usage_perc NaN\n"
        )

        self.assertIsNone(snapshot.kv_cache_usage)

    def test_partial_histogram_does_not_synthesize_zero_average(self) -> None:
        from tui.common.metrics import parse_snapshot

        snapshot = parse_snapshot("vllm:request_prefill_time_seconds_count 4\n")

        self.assertIsNotNone(snapshot.prefill)
        assert snapshot.prefill is not None
        self.assertIsNone(snapshot.prefill.sum)
        self.assertEqual(snapshot.prefill.count, 4.0)
        self.assertIsNone(snapshot.prefill.avg())

    def test_missing_external_hits_render_as_unavailable(self) -> None:
        from rich.console import Console

        from tui.common.metrics import MetricsSnapshot
        from tui.common.monitor_render import Derived, _cache_panel

        output = io.StringIO()
        Console(file=output, width=80).print(
            _cache_panel(
                MetricsSnapshot(ext_prefix_queries=10.0, ext_prefix_hits=None),
                Derived(),
            )
        )

        external_line = next(
            line for line in output.getvalue().splitlines() if "external" in line
        )
        self.assertIn("—", external_line)
        self.assertNotIn("0%", external_line)


class MetricsFetchTests(unittest.IsolatedAsyncioTestCase):
    class _InlineLoop:
        async def _call(self, function):
            return function()

        def run_in_executor(self, _executor, function):
            return self._call(function)

    class _Response:
        def __init__(self, body: str) -> None:
            self._body = body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    async def test_http_failure_is_not_normal_metric_absence(self) -> None:
        from tui.common.metrics import MetricsUnavailableError, fetch_token_counters

        with patch(
            "tui.common.metrics.asyncio.get_running_loop",
            return_value=self._InlineLoop(),
        ), patch(
            "tui.common.metrics.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(MetricsUnavailableError):
                await fetch_token_counters(8000)

    async def test_successful_response_without_token_metrics_returns_none(self) -> None:
        from tui.common.metrics import fetch_token_counters

        with patch(
            "tui.common.metrics.asyncio.get_running_loop",
            return_value=self._InlineLoop(),
        ), patch(
            "tui.common.metrics.urllib.request.urlopen",
            return_value=self._Response("other_metric 0\n"),
        ):
            result = await fetch_token_counters(8000)

        self.assertIsNone(result)


class DockerSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def test_lifecycle_health_and_raw_status_are_preserved(self) -> None:
        from tui.common.docker import (
            ContainerHealth,
            ContainerLifecycle,
            parse_container_snapshots,
        )

        snapshots = parse_container_snapshots(
            "healthy\trunning\tUp 10 seconds (healthy)\n"
            "starting\trunning\tUp 2 seconds (health: starting)\n"
            "restart\trestarting\tRestarting (1) 1 second ago\n"
            "mystery\twarming\tWarming up\n"
        )

        self.assertEqual(snapshots["healthy"].lifecycle, ContainerLifecycle.RUNNING)
        self.assertEqual(snapshots["healthy"].health, ContainerHealth.HEALTHY)
        self.assertTrue(snapshots["healthy"].running)
        self.assertEqual(snapshots["starting"].health, ContainerHealth.STARTING)
        self.assertEqual(snapshots["restart"].lifecycle, ContainerLifecycle.RESTARTING)
        self.assertTrue(snapshots["restart"].running)
        self.assertEqual(snapshots["mystery"].lifecycle, ContainerLifecycle.UNKNOWN)
        self.assertEqual(snapshots["mystery"].raw_state, "warming")
        self.assertEqual(snapshots["mystery"].raw_status, "Warming up")

    def test_empty_is_normal_but_malformed_row_fails(self) -> None:
        from tui.common.docker import parse_container_snapshots

        self.assertEqual(parse_container_snapshots(""), {})
        with self.assertRaisesRegex(RuntimeError, "unexpected docker ps output"):
            parse_container_snapshots("missing-tabs\n")

    async def test_container_snapshots_requests_all_when_required(self) -> None:
        from tui.common.docker import container_snapshots

        with patch(
            "tui.common.docker.run_command",
            AsyncMock(return_value=(0, "c\texited\tExited (0) 1 second ago\n")),
        ) as run:
            snapshots = await container_snapshots(include_stopped=True)

        self.assertIn("c", snapshots)
        self.assertIn("--all", run.await_args.args)


class GpuReadingContractTests(unittest.TestCase):
    def test_shared_gpu_thresholds_and_malformed_values(self) -> None:
        from tui.common.docker import (
            GpuReadingLevel,
            gpu_temperature_level,
            gpu_utilization_level,
        )

        self.assertEqual(gpu_utilization_level("49.9"), GpuReadingLevel.NORMAL)
        self.assertEqual(gpu_utilization_level("50"), GpuReadingLevel.WARNING)
        self.assertEqual(gpu_utilization_level("80"), GpuReadingLevel.CRITICAL)
        self.assertEqual(gpu_temperature_level("60"), GpuReadingLevel.WARNING)
        self.assertEqual(gpu_temperature_level("80"), GpuReadingLevel.CRITICAL)
        with self.assertRaises(ValueError):
            gpu_utilization_level("N/A")
        with self.assertRaises(ValueError):
            gpu_temperature_level("NaN")


class AdapterLabelTests(unittest.TestCase):
    def test_vllm_adapter_uses_config_model_and_typed_running_state(self) -> None:
        from tui.backends.vllm import backend
        from tui.backends.vllm.adapter import VllmAdapter
        from tui.common.docker import RunningContainerNames, parse_container_snapshots

        profile = backend.Profile(
            name="p", container_name="c", config_name="config-label", model_id="fallback"
        )
        running = RunningContainerNames(
            parse_container_snapshots("c\trestarting\tRestarting (1) 1 second ago\n")
        )
        with patch.object(backend, "list_profile_names", return_value=["p"]), patch.object(
            backend, "load_profile", return_value=profile
        ), patch.object(
            backend,
            "load_config",
            return_value=backend.Config(name="config-label", model="org/actual-model"),
        ):
            row = VllmAdapter().rows(running)[0]

        self.assertEqual(row.model, "org/actual-model")
        self.assertTrue(row.running)

    def test_llamacpp_adapter_prefers_alias_then_hf_repo(self) -> None:
        from tui.backends.llamacpp import backend
        from tui.backends.llamacpp.adapter import LlamacppAdapter

        profile = backend.Profile(
            name="p",
            container_name="c",
            config_name="config-label",
            hf_repo="org/actual-model",
            running=True,
        )
        with patch.object(backend, "list_profiles", return_value=[profile]), patch.object(
            backend,
            "load_config",
            return_value=backend.Config(name="config-label", params={"alias": "served"}),
        ):
            row = LlamacppAdapter().rows({"c"})[0]
        self.assertEqual(row.model, "served")

        with patch.object(backend, "list_profiles", return_value=[profile]), patch.object(
            backend,
            "load_config",
            return_value=backend.Config(name="config-label"),
        ):
            row = LlamacppAdapter().rows({"c"})[0]
        self.assertEqual(row.model, "org/actual-model")


class DashboardStatusTests(unittest.TestCase):
    def test_dashboard_renders_typed_restarting_status(self) -> None:
        from tui.common.adapter import DashboardRow
        from tui.common.docker import parse_container_snapshots
        from tui.screens.dashboard import DashboardScreen

        screen = SimpleNamespace(
            _container_snapshots=parse_container_snapshots(
                "c\trestarting\tRestarting (1) 1 second ago\n"
            )
        )
        row = DashboardRow("vllm", "p", "c", 8000, True, "org/model", "")

        rendered = DashboardScreen._container_status_cell(screen, row)

        self.assertIn("restarting", rendered)
        self.assertNotIn("running", rendered)
        self.assertNotIn("stopped", rendered)


class DashboardRenderEnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_pilot_renders_latest_profile_under_storage_lock(self) -> None:
        from tui.app import LlmuxApp
        from tui.screens.dashboard import DashboardScreen

        profile = SimpleNamespace(name="selected", backend="vllm")
        rendered = "/tmp/selected.env"
        with patch(
            "tui.common.docker.running_container_names",
            AsyncMock(return_value=set()),
        ), patch(
            "tui.common.docker.get_gpu_info", AsyncMock(return_value=[])
        ), patch(
            "tui.screens.dashboard.VllmAdapter.rows", return_value=[]
        ), patch(
            "tui.screens.dashboard.LlamacppAdapter.rows", return_value=[]
        ), patch(
            "tui.screens.dashboard.load_profile", return_value=profile
        ), patch(
            "tui.screens.dashboard.render_env_for_profile",
            return_value=rendered,
            create=True,
        ) as render:
            app = LlmuxApp()
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                dashboard = next(
                    screen
                    for screen in app.screen_stack
                    if isinstance(screen, DashboardScreen)
                )
                dashboard.notify = MagicMock()
                dashboard._render_profile_env("selected", "vllm")

        render.assert_called_once_with(profile.name, profile.backend)
        self.assertIn("Rendered", dashboard.notify.call_args.args[0])


class MonitorPollingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _row(name: str, backend: str = "vllm"):
        from tui.common.adapter import DashboardRow

        return DashboardRow(
            backend=backend,
            profile_name=name,
            container_name=name,
            port=8000,
            running=True,
            model="org/model",
            detail="",
        )

    async def test_focus_runs_when_target_is_confirmed_despite_other_scan_error(self) -> None:
        from tui.common import plain_monitor

        with patch.object(
            plain_monitor,
            "_running_rows",
            AsyncMock(return_value=([self._row("target")], ["llama.cpp scan failed"])),
        ), patch.object(
            plain_monitor, "run_plain_monitor", AsyncMock()
        ) as run:
            rc = await plain_monitor._resolve_and_run("target")

        self.assertEqual(rc, 0)
        run.assert_awaited_once_with("target")

    async def test_endpoint_polling_is_concurrent_and_bounded(self) -> None:
        from tui.common import plain_monitor

        active = 0
        peak = 0

        async def fetch(_port):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return None

        rows = [self._row(f"p{i}") for i in range(10)]
        with patch.object(
            plain_monitor, "_running_rows", AsyncMock(return_value=(rows, []))
        ), patch.object(plain_monitor, "fetch_snapshot", side_effect=fetch):
            await plain_monitor.sample_entries(None, {}, 0.0, 0.0)

        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, plain_monitor.METRICS_POLL_CONCURRENCY)

    async def test_metric_fetch_failure_becomes_profile_notice(self) -> None:
        from tui.common import plain_monitor
        from tui.common.metrics import MetricsUnavailableError

        with patch.object(
            plain_monitor,
            "_running_rows",
            AsyncMock(return_value=([self._row("p")], [])),
        ), patch.object(
            plain_monitor,
            "fetch_snapshot",
            AsyncMock(side_effect=MetricsUnavailableError("request failed")),
        ):
            entries, notices = await plain_monitor.sample_entries(None, {}, 0.0, 0.0)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].snap)
        self.assertTrue(any("p" in notice and "request failed" in notice for notice in notices))

    async def test_poll_lag_includes_metric_collection(self) -> None:
        from tui.common import plain_monitor

        async def delayed_entries(*_args):
            await asyncio.sleep(0.02)
            return [], []

        with patch.object(
            plain_monitor.common_docker, "get_gpu_info", AsyncMock(return_value=[])
        ), patch.object(
            plain_monitor.common_docker, "get_pcie_stats", AsyncMock(return_value={})
        ), patch.object(plain_monitor, "sample_entries", side_effect=delayed_entries):
            result = await plain_monitor.poll_monitor(None, {}, [])

        self.assertGreaterEqual(result.lag_ms, 15)


class TerminalSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_tty_setup_failure_is_explicit_and_restores_attrs(self) -> None:
        from tui.common import plain_monitor

        restored: list[tuple] = []

        def fail_setcbreak(_fd) -> None:
            raise PermissionError("denied")

        termios = SimpleNamespace(
            TCSADRAIN=1,
            tcgetattr=lambda _fd: ["saved"],
            tcsetattr=lambda *args: restored.append(args),
        )
        tty = SimpleNamespace(setcbreak=fail_setcbreak)
        with patch.dict(sys.modules, {"termios": termios, "tty": tty}), patch.object(
            sys.stdin, "fileno", return_value=7
        ):
            with self.assertRaisesRegex(plain_monitor.TerminalSetupError, "denied"):
                await plain_monitor.run_plain_monitor()

        self.assertEqual(restored, [(7, 1, ["saved"])])


class MonitorStateTests(unittest.TestCase):
    def test_partial_histogram_breaks_window_baseline(self) -> None:
        from tui.common.metrics import Hist, MetricsSnapshot
        from tui.common.monitor_render import MonitorState

        state = MonitorState()
        first = state.update(
            MetricsSnapshot(prefill=Hist(sum=10.0, count=2.0)), 1.0, 0.0
        )
        partial = state.update(
            MetricsSnapshot(prefill=Hist(sum=None, count=3.0)), 2.0, 0.0
        )
        resumed = state.update(
            MetricsSnapshot(prefill=Hist(sum=20.0, count=4.0)), 3.0, 0.0
        )

        self.assertEqual(first.lat["prefill"], 5.0)
        self.assertIsNone(partial.lat["prefill"])
        self.assertEqual(resumed.lat["prefill"], 5.0)
        self.assertFalse(hasattr(state, "last_lag_ms"))
        self.assertFalse(hasattr(state, "samples"))


if __name__ == "__main__":
    unittest.main()
