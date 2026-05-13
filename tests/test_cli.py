"""Smoke tests for the typer CLI surface.

CLI tests run via subprocess so each test gets a fresh interpreter with its
own LLMUX_ROOT — this avoids the profile_store PROJECT_ROOT being captured at
first import (which would otherwise pin to the real repo when other test
modules load first).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_temp_project() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="llmux-cli-test-"))
    (tmp / "compose").mkdir()
    (tmp / "profiles.example.yaml").write_text("version: 1\n")
    (tmp / "config" / "vllm").mkdir(parents=True)
    (tmp / "config" / "llamacpp").mkdir(parents=True)
    (tmp / ".runtime").mkdir()
    (tmp / "profiles.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "defaults": {
                    "vllm": {
                        "port": 8000, "gpu_id": "0",
                        "tensor_parallel_size": 1, "enable_lora": False,
                    },
                    "llamacpp": {"port": 8080, "gpu_id": "0"},
                },
                "profiles": [
                    {"name": "alpha", "backend": "vllm", "port": 8001,
                     "gpu_id": "1", "model_id": "Qwen/Qwen3-0.6B"},
                    {"name": "bravo", "backend": "vllm", "port": 8002, "gpu_id": "0"},
                    {"name": "lcpp", "backend": "llamacpp", "port": 8080, "gpu_id": "0"},
                ],
            },
            sort_keys=False,
        )
    )
    (tmp / "config" / "vllm" / "alpha.yaml").write_text(
        yaml.safe_dump(
            {"model": "Qwen/Qwen3-0.6B", "gpu-memory-utilization": "0.85", "max-model-len": 4096},
            sort_keys=False,
        )
    )
    return tmp


def _run_cli(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LLMUX_ROOT": str(tmp), "NO_COLOR": "1"}
    return subprocess.run(
        [sys.executable, "-m", "tui", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class CliSmokeTests(unittest.TestCase):
    """Side-effect-free dispatch + read-only command coverage."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = _make_temp_project()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # --- dispatch / help ----------------------------------------------------

    def test_help_lists_top_commands(self):
        r = _run_cli(self.tmp, "--help")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for cmd in ["tui", "up", "down", "logs", "ps", "render-env",
                    "container", "profile", "config", "image", "system",
                    "gpu", "env-check"]:
            self.assertIn(cmd, r.stdout)

    def test_subcommand_help(self):
        for sub in ["container", "profile", "config", "image", "system"]:
            r = _run_cli(self.tmp, sub, "--help")
            self.assertEqual(r.returncode, 0, f"{sub}: {r.stderr or r.stdout}")

    # --- profile read paths ------------------------------------------------

    def test_profile_list_table(self):
        r = _run_cli(self.tmp, "profile", "list")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for n in ("alpha", "bravo", "lcpp"):
            self.assertIn(n, r.stdout)

    def test_profile_list_json(self):
        r = _run_cli(self.tmp, "profile", "list", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        self.assertEqual(sorted(p["name"] for p in rows), ["alpha", "bravo", "lcpp"])

    def test_profile_list_filtered_by_backend(self):
        r = _run_cli(self.tmp, "profile", "list", "--backend", "llamacpp", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        self.assertEqual([p["name"] for p in rows], ["lcpp"])

    def test_profile_show_autodetects_backend(self):
        r = _run_cli(self.tmp, "profile", "show", "alpha", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        data = json.loads(r.stdout)
        self.assertEqual(data["backend"], "vllm")
        self.assertEqual(data["model_id"], "Qwen/Qwen3-0.6B")

    def test_profile_show_unknown_returns_error(self):
        r = _run_cli(self.tmp, "profile", "show", "no-such-profile")
        self.assertNotEqual(r.returncode, 0)
        # typer prints param errors to stderr by default
        combined = (r.stdout or "") + (r.stderr or "")
        self.assertIn("not found", combined)

    # --- config read paths -------------------------------------------------

    def test_config_list_includes_seeded(self):
        r = _run_cli(self.tmp, "config", "list", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        rows = json.loads(r.stdout)
        names = [row["name"] for row in rows if row["backend"] == "vllm"]
        self.assertIn("alpha", names)

    def test_config_show_yaml(self):
        r = _run_cli(self.tmp, "config", "show", "alpha")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertIn("model: Qwen/Qwen3-0.6B", r.stdout)
        self.assertIn("max-model-len: 4096", r.stdout)

    # --- profile/config write paths (round-trip) ---------------------------

    def test_profile_new_then_edit_then_delete(self):
        r = _run_cli(
            self.tmp, "profile", "new", "tmp1", "--backend", "vllm",
            "--port", "8003", "--gpu-id", "0", "--model", "test/model",
            "--set", "OMP_NUM_THREADS=4",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        data = json.loads(r.stdout)
        self.assertEqual(data["port"], 8003)
        self.assertEqual(data["env_vars"], {"OMP_NUM_THREADS": "4"})

        r = _run_cli(
            self.tmp, "profile", "edit", "tmp1", "--port", "8004",
            "--set", "VLLM_USE_FOO=1", "--unset", "OMP_NUM_THREADS",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1", "--json")
        data = json.loads(r.stdout)
        self.assertEqual(data["port"], 8004)
        self.assertEqual(data["env_vars"], {"VLLM_USE_FOO": "1"})

        r = _run_cli(self.tmp, "profile", "delete", "tmp1", "-y")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "profile", "show", "tmp1")
        self.assertNotEqual(r.returncode, 0)

    def test_config_new_set_overwrite_delete(self):
        r = _run_cli(
            self.tmp, "config", "new", "tmpcfg", "--model", "test/m",
            "--set", "max-model-len=8192", "--set", "enable-prefix-caching=",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        r = _run_cli(self.tmp, "config", "show", "tmpcfg", "--json")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        data = json.loads(r.stdout)
        self.assertEqual(data["model"], "test/m")
        self.assertEqual(data["max-model-len"], 8192)
        self.assertIs(data["enable-prefix-caching"], True)

        r = _run_cli(self.tmp, "config", "new", "tmpcfg", "--model", "x/y")
        self.assertNotEqual(r.returncode, 0)

        r = _run_cli(self.tmp, "config", "delete", "tmpcfg", "-y")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

    # --- render-env --------------------------------------------------------

    def test_render_env_writes_runtime_file(self):
        r = _run_cli(self.tmp, "render-env", "alpha")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        out = self.tmp / ".runtime" / "vllm" / "alpha.env"
        self.assertTrue(out.exists(), out)
        content = out.read_text()
        self.assertIn("VLLM_PORT=8001", content)
        self.assertIn("GPU_ID=1", content)

    def test_render_env_all(self):
        r = _run_cli(self.tmp, "render-env")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        for n in ("alpha", "bravo"):
            self.assertTrue((self.tmp / ".runtime" / "vllm" / f"{n}.env").exists())

    # --- system commands ---------------------------------------------------

    def test_gpu_runs_without_crashing(self):
        r = _run_cli(self.tmp, "gpu")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

    def test_env_check_missing_env_common(self):
        r = _run_cli(self.tmp, "env-check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MISSING", (r.stdout + r.stderr).upper())

    # --- security / hardening ------------------------------------------------

    def test_config_show_rejects_path_traversal(self):
        # `..%2Fetc%2Fpasswd` style: dots-and-slashes name should be rejected
        # by _validate_name *before* any filesystem access happens.
        r = _run_cli(self.tmp, "config", "show", "../../etc/passwd")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid", (r.stdout + r.stderr).lower())

    def test_config_delete_rejects_path_traversal(self):
        r = _run_cli(self.tmp, "config", "delete", "../alpha", "-y")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid", (r.stdout + r.stderr).lower())

    def test_profile_set_rejects_invalid_env_key(self):
        # Lowercase-and-special chars violate POSIX env var naming rules.
        # Should fail at parse time with BadParameter, not deep in profile_store.
        r = _run_cli(
            self.tmp, "profile", "new", "tmpenv", "--backend", "vllm",
            "--port", "8005", "--gpu-id", "0",
            "--set", "bad-key=1",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--set", (r.stdout + r.stderr))

    def test_profile_decline_confirm_returns_zero(self):
        # User explicitly answering "n" to confirmation is not an error.
        # Pre-create a throwaway profile.
        _run_cli(
            self.tmp, "profile", "new", "to-decline", "--backend", "vllm",
            "--port", "8006", "--gpu-id", "0", "--model", "x/y",
        )
        # Pipe "n" to stdin to decline.
        env = {**os.environ, "LLMUX_ROOT": str(self.tmp), "NO_COLOR": "1"}
        r = subprocess.run(
            [sys.executable, "-m", "tui", "profile", "delete", "to-decline"],
            input="n\n",
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # Profile should still exist.
        r2 = _run_cli(self.tmp, "profile", "show", "to-decline", "--json")
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        # Cleanup
        _run_cli(self.tmp, "profile", "delete", "to-decline", "-y")

    def test_config_show_unknown_backend_explicit(self):
        # --backend llamacpp + name only existing in vllm → clear error
        r = _run_cli(self.tmp, "config", "show", "alpha", "--backend", "llamacpp")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not found", (r.stdout + r.stderr).lower())

    def test_profile_show_llamacpp(self):
        # Symmetric to vllm — make sure llamacpp profiles also dispatch
        # correctly via auto-detect.
        r = _run_cli(self.tmp, "profile", "show", "lcpp", "--json")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["backend"], "llamacpp")
        self.assertEqual(data["port"], 8080)


if __name__ == "__main__":
    unittest.main()
