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


def _run_cli_stubbed_hf(
    tmp: Path, gguf_files: list[str], *args: str
) -> subprocess.CompletedProcess:
    """Run the CLI with `list_hf_repo_files` stubbed to return a synthetic
    GGUF file listing. Used by the llama.cpp quick-setup regression test so
    it doesn't depend on huggingface.co reachability (`NameError: run_async`
    would otherwise be masked by network flake — see T17 regression).
    """
    env = {**os.environ, "LLMUX_ROOT": str(tmp), "NO_COLOR": "1"}
    # Build the in-process stub: monkey-patch the backend symbol *before*
    # tui.cli is imported, so _quick_setup_llamacpp's local
    # `from ...backend import list_hf_repo_files` picks up the stub.
    stub_src = (
        "import sys\n"
        "import tui.backends.llamacpp.backend as _b\n"
        f"_FAKE_FILES = {[{'type': 'file', 'path': p} for p in gguf_files]!r}\n"
        "async def _stub(*_a, **_k): return _FAKE_FILES\n"
        "_b.list_hf_repo_files = _stub\n"
        "from tui.cli import main\n"
        f"sys.argv = ['llmux', *{list(args)!r}]\n"
        "main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", stub_src],
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

    # --- port / gpu-id validation (unified CLI ⇄ TUI rule) ------------------

    def test_profile_new_rejects_privileged_port(self):
        r = _run_cli(
            self.tmp, "profile", "new", "p80", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "80",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("1024", r.stderr + r.stdout)

    def test_profile_new_accepts_multi_digit_gpu_id(self):
        # `0,10` must be valid — the old `[0-9](,[0-9])*` rule capped indices
        # at one digit, making hosts with 10+ GPUs unaddressable.
        r = _run_cli(
            self.tmp, "profile", "new", "gpu10", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8099", "--gpu-id", "0,10",
        )
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        show = _run_cli(self.tmp, "profile", "show", "gpu10", "--json")
        self.assertEqual(json.loads(show.stdout)["gpu_id"], "0,10")

        _run_cli(self.tmp, "profile", "delete", "gpu10", "-y")

    def test_profile_new_rejects_malformed_gpu_id(self):
        r = _run_cli(
            self.tmp, "profile", "new", "gpubad", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8098", "--gpu-id", "0,,",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("GPU id", r.stderr + r.stdout)

    def test_profile_edit_rejects_bad_port(self):
        created = _run_cli(
            self.tmp, "profile", "new", "editport", "--backend", "llamacpp",
            "--hf-repo", "org/x", "--port", "8097",
        )
        self.assertEqual(created.returncode, 0, created.stderr or created.stdout)

        r = _run_cli(self.tmp, "profile", "edit", "editport", "--port", "80")
        self.assertNotEqual(r.returncode, 0)

        _run_cli(self.tmp, "profile", "delete", "editport", "-y")

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

    # --- llama.cpp quick-setup (T17) ------------------------------------------

    def test_quick_setup_llamacpp_writes_profile_and_config(self):
        """Regression: `_quick_setup_llamacpp` used `run_async` without
        importing it from `_runtime`, causing NameError on first call. This
        exercises the full end-to-end path with `list_hf_repo_files` stubbed
        so the test stays hermetic.
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf", "model-Q8_0.gguf"],
            "profile", "quick-setup",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--name", "lc-smoke",
            "--port", "8087",
            "--gpu-id", "0",
            "--ctx-size", "1024",
            "--n-gpu-layers", "0",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--no-flash-attn",
            "--no-jinja",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Created profile + config: lc-smoke", r.stdout)

        # Profile written through profile_store.save_profile
        r2 = _run_cli(self.tmp, "profile", "show", "lc-smoke", "--json")
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        data = json.loads(r2.stdout)
        self.assertEqual(data["backend"], "llamacpp")
        self.assertEqual(data["port"], 8087)
        self.assertEqual(data["hf_repo"], "fake/Model-GGUF")
        self.assertEqual(data["hf_file"], "model-Q4_K_M.gguf")
        self.assertEqual(data["model_file"], "model-Q4_K_M.gguf")

        # Config written through llamacpp backend save_config — verify the
        # key set + ints are typed (not str).
        cfg_path = self.tmp / "config" / "llamacpp" / "lc-smoke.yaml"
        self.assertTrue(cfg_path.exists(), cfg_path)
        cfg = yaml.safe_load(cfg_path.read_text())
        self.assertEqual(cfg.get("model-file"), "model-Q4_K_M.gguf")
        self.assertEqual(cfg.get("alias"), "lc-smoke")
        self.assertEqual(cfg.get("ctx-size"), 1024)
        self.assertEqual(cfg.get("n-gpu-layers"), 0)
        self.assertEqual(cfg.get("cache-type-k"), "f16")
        self.assertEqual(cfg.get("cache-type-v"), "f16")
        # --no-flash-attn / --no-jinja → keys must NOT be present.
        self.assertNotIn("flash-attn", cfg)
        self.assertNotIn("jinja", cfg)
        # Cleanup.
        _run_cli(self.tmp, "profile", "delete", "lc-smoke", "-y")

    def test_quick_setup_llamacpp_warns_on_positional_model(self):
        """Positional MODEL has no meaning for --backend llamacpp (GGUF profiles
        derive their identity from --hf-repo / --hf-file). Silently dropping
        the argument hides scripted typos — verify we warn on stderr but still
        proceed with the auto-derived name (no hard fail, no behavior change).
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "smalltest",  # positional MODEL — meaningless for llamacpp
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--port", "8089",
            "--gpu-id", "0",
        )
        # Should still succeed — auto-derive from repo tail.
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Warning", r.stderr)
        self.assertIn("smalltest", r.stderr)
        self.assertIn("ignored", r.stderr.lower())
        self.assertIn("--name", r.stderr)
        # Auto-derived name (lowercased, stripped, -GGUF removed).
        self.assertIn("Created profile + config: model", r.stdout)
        # Cleanup any auto-derived name that actually got created.
        r2 = _run_cli(self.tmp, "profile", "list", "--json", "--backend", "llamacpp")
        if r2.returncode == 0:
            for row in json.loads(r2.stdout):
                if row["name"].startswith("model"):
                    _run_cli(
                        self.tmp, "profile", "delete", row["name"],
                        "-y", "--with-config",
                    )

    def test_quick_setup_llamacpp_warns_when_positional_with_explicit_name(self):
        """When both positional MODEL and --name are given for --backend
        llamacpp, the positional is still ignored — but the warning should
        reference the explicit --name so the user can see which one won.
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["model-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "smalltest",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "model-Q4_K_M.gguf",
            "--name", "lc-explicit",
            "--port", "8090",
            "--gpu-id", "0",
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Warning", r.stderr)
        self.assertIn("smalltest", r.stderr)
        self.assertIn("lc-explicit", r.stderr)
        self.assertIn("Created profile + config: lc-explicit", r.stdout)
        _run_cli(self.tmp, "profile", "delete", "lc-explicit", "-y", "--with-config")

    def test_quick_setup_llamacpp_rejects_file_not_in_repo(self):
        """If the HF API returns a listing and --hf-file isn't in it, error
        out with a clear message (parity with the TUI's Fetch+select gate).
        """
        r = _run_cli_stubbed_hf(
            self.tmp,
            ["only-Q4_K_M.gguf"],
            "profile", "quick-setup",
            "--backend", "llamacpp",
            "--hf-repo", "fake/Model-GGUF",
            "--hf-file", "missing-Q8_0.gguf",
            "--name", "lc-bad",
            "--port", "8088",
            "--gpu-id", "0",
        )
        self.assertNotEqual(r.returncode, 0)
        combined = (r.stdout or "") + (r.stderr or "")
        self.assertIn("not found", combined.lower())
        self.assertIn("only-Q4_K_M.gguf", combined)


if __name__ == "__main__":
    unittest.main()
