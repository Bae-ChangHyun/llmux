import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tui import backend


class LoadConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._created: list[Path] = []

    def tearDown(self) -> None:
        for path in self._created:
            path.unlink(missing_ok=True)

    def _write_config(self, name: str, content: str) -> None:
        path = backend.CONFIG_DIR / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._created.append(path)

    def test_load_config_reads_mapping_yaml(self) -> None:
        self._write_config(
            "__test_valid_config__",
            "model: org/model\ngpu-memory-utilization: 0.8\nmax-model-len: 4096\n",
        )

        config = backend.load_config("__test_valid_config__")

        self.assertEqual(config.name, "__test_valid_config__")
        self.assertEqual(config.model, "org/model")
        self.assertEqual(config.gpu_memory_utilization, "0.8")
        self.assertEqual(config.extra_params, {"max-model-len": 4096})

    def test_load_config_ignores_non_mapping_yaml(self) -> None:
        self._write_config("__test_invalid_config__", "- not\n- a\n- mapping\n")

        config = backend.load_config("__test_invalid_config__")

        self.assertEqual(config.name, "__test_invalid_config__")
        self.assertEqual(config.model, "")
        self.assertEqual(config.gpu_memory_utilization, "0.9")
        self.assertEqual(config.extra_params, {})


class CheckPortConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_port_conflict_returns_readable_profile_reference(self) -> None:
        profile = backend.Profile(name="current", container_name="current", port="8000")
        other = backend.Profile(name="other", container_name="other-container", port="8000")

        globals_dict = backend.check_port_conflict.__globals__

        async def fake_run_command(*args, **kwargs):
            return 0, "other-container\t127.0.0.1:8000->8000/tcp\n"

        def fake_load_profile(name: str):
            return {"current": profile, "other": other}[name]

        with patch.dict(
            globals_dict,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current", "other"],
                "load_profile": fake_load_profile,
                "is_container_running": AsyncMock(return_value=False),
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertEqual(conflict, "profile 'other'")


if __name__ == "__main__":
    unittest.main()
