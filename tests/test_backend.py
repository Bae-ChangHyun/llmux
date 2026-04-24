import tempfile
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tui.backends.llamacpp import backend as lbackend
from tui.backends.vllm import backend
from tui.backends.vllm.backend_inspect import (
    _pick_preferred_tag,
    get_dockerhub_nightly_date,
    get_dockerhub_release_version,
)
from tui.backends.vllm.backend_runtime import (
    _build_lora_options,
    _detect_gpu_arch,
    _ensure_common_env,
    _force_local_arch_for_deepep,
    _gpu_conflict_messages,
    _post_start_validation,
    _verify_vllm_version,
)
from tui.common import profile_store


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

    def test_load_config_missing_file_returns_defaults(self) -> None:
        config = backend.load_config("__test_missing_config_xyz__")

        self.assertEqual(config.name, "__test_missing_config_xyz__")
        self.assertEqual(config.model, "")
        self.assertEqual(config.gpu_memory_utilization, "0.9")
        self.assertEqual(config.extra_params, {})


class SaveLoadConfigRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._name = "__test_roundtrip_config__"
        self._path = backend.CONFIG_DIR / f"{self._name}.yaml"

    def tearDown(self) -> None:
        self._path.unlink(missing_ok=True)

    def test_save_then_load_preserves_fields(self) -> None:
        cfg = backend.Config(
            name=self._name,
            model="org/model",
            gpu_memory_utilization="0.75",
            extra_params={"max-model-len": 4096, "trust-remote-code": True},
        )
        backend.save_config(cfg)

        loaded = backend.load_config(self._name)
        self.assertEqual(loaded.model, "org/model")
        self.assertEqual(loaded.gpu_memory_utilization, "0.75")
        self.assertEqual(loaded.extra_params["max-model-len"], 4096)
        self.assertIs(loaded.extra_params["trust-remote-code"], True)

    def test_save_empty_string_value_becomes_true_flag(self) -> None:
        cfg = backend.Config(
            name=self._name,
            model="org/model",
            extra_params={"enforce-eager": ""},
        )
        backend.save_config(cfg)

        loaded = backend.load_config(self._name)
        self.assertIs(loaded.extra_params["enforce-eager"], True)


class ConfigParamValueTests(unittest.TestCase):
    def test_parse_blank_returns_true(self) -> None:
        self.assertIs(backend.parse_config_param_value(""), True)

    def test_parse_int_and_float(self) -> None:
        self.assertEqual(backend.parse_config_param_value("4096"), 4096)
        self.assertEqual(backend.parse_config_param_value("0.9"), 0.9)

    def test_parse_bool_and_null(self) -> None:
        self.assertIs(backend.parse_config_param_value("false"), False)
        self.assertIsNone(backend.parse_config_param_value("null"))

    def test_parse_list(self) -> None:
        self.assertEqual(backend.parse_config_param_value("[a, b, c]"), ["a", "b", "c"])

    def test_format_true_becomes_empty(self) -> None:
        self.assertEqual(backend.format_config_param_value(True), "")

    def test_format_false_preserved_as_string(self) -> None:
        self.assertEqual(backend.format_config_param_value(False), "false")

    def test_format_none_is_null(self) -> None:
        self.assertEqual(backend.format_config_param_value(None), "null")

    def test_format_list_uses_flow_style(self) -> None:
        formatted = backend.format_config_param_value([1, 2, 3])
        self.assertIn("[", formatted)
        self.assertIn("1", formatted)


class ValidateNameTests(unittest.TestCase):
    def test_accepts_plain_names(self) -> None:
        self.assertTrue(backend.validate_name("abc"))
        self.assertTrue(backend.validate_name("my-profile_01"))
        self.assertTrue(backend.validate_name("Qwen3-0-8b"))

    def test_rejects_leading_dash(self) -> None:
        self.assertFalse(backend.validate_name("-injection"))

    def test_rejects_special_chars(self) -> None:
        self.assertFalse(backend.validate_name("name with space"))
        self.assertFalse(backend.validate_name("path/traversal"))
        self.assertFalse(backend.validate_name("dot.in.name"))
        self.assertFalse(backend.validate_name(""))


class ParseEnvFileTests(unittest.TestCase):
    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_missing_file_returns_empty_dict(self) -> None:
        missing = Path(tempfile.gettempdir()) / "__definitely_missing_env_file__"
        self.assertEqual(backend._parse_env_file(missing), {})

    def test_parses_basic_key_value(self) -> None:
        path = self._write("KEY=value\nOTHER=123\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value", "OTHER": "123"})

    def test_ignores_comments_and_blank_lines(self) -> None:
        path = self._write("# comment\n\nKEY=value\n# another\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value"})

    def test_strips_inline_comments_when_unquoted(self) -> None:
        path = self._write("KEY=value # trailing\n")
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value"})

    def test_preserves_quoted_values_including_hash(self) -> None:
        path = self._write('KEY="value # keep"\n')
        self.assertEqual(backend._parse_env_file(path), {"KEY": "value # keep"})

    def test_strips_matching_quotes(self) -> None:
        path = self._write("KEY='quoted'\nOTHER=\"double\"\n")
        self.assertEqual(
            backend._parse_env_file(path),
            {"KEY": "quoted", "OTHER": "double"},
        )


class SaveLoadProfileRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._name = "__test_roundtrip_profile__"
        self._path = backend.RUNTIME_DIR / f"{self._name}.env"

    def tearDown(self) -> None:
        backend.delete_profile(self._name)

    def test_save_then_load_preserves_fields(self) -> None:
        profile = backend.Profile(
            name=self._name,
            container_name="my-container",
            port="8123",
            gpu_id="0,1",
            tensor_parallel="2",
            config_name="my-config",
            model_id="org/model",
            enable_lora="true",
            max_loras="4",
            max_lora_rank="32",
            lora_modules="alpha=/path/a,beta=/path/b",
            env_vars={"EXTRA_PIP_PACKAGES": "pkg-a pkg-b", "CUSTOM": "x"},
        )
        backend.save_profile(profile)

        loaded = backend.load_profile(self._name)
        self.assertEqual(loaded.container_name, "my-container")
        self.assertEqual(loaded.port, "8123")
        self.assertEqual(loaded.gpu_id, "0,1")
        self.assertEqual(loaded.tensor_parallel, "2")
        self.assertEqual(loaded.config_name, "my-config")
        self.assertEqual(loaded.model_id, "org/model")
        self.assertEqual(loaded.enable_lora, "true")
        self.assertEqual(loaded.max_loras, "4")
        self.assertEqual(loaded.max_lora_rank, "32")
        self.assertEqual(loaded.lora_modules, "alpha=/path/a,beta=/path/b")
        self.assertEqual(loaded.env_vars["EXTRA_PIP_PACKAGES"], "pkg-a pkg-b")
        self.assertEqual(loaded.env_vars["CUSTOM"], "x")


class ProfileStoreYamlTests(unittest.TestCase):
    def test_user_defaults_are_applied_when_profile_omits_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text(
                "version: 1\n"
                "defaults:\n"
                "  vllm:\n"
                "    port: 9100\n"
                "    gpu_id: '2'\n"
                "    tensor_parallel_size: 2\n"
                "    enable_lora: false\n"
                "profiles:\n"
                "  - name: p\n"
                "    backend: vllm\n"
                "    model_id: org/model\n"
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                stored = profile_store.load_profile("p", "vllm")

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.port, 9100)
            self.assertEqual(stored.gpu_id, "2")
            self.assertEqual(stored.tensor_parallel_size, 2)

    def test_string_false_is_parsed_as_false(self) -> None:
        stored = profile_store._to_profile(
            {
                "name": "p",
                "backend": "vllm",
                "enable_lora": "false",
            }
        )

        self.assertFalse(stored.enable_lora)

    def test_render_env_quotes_shell_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = profile_store.StoredProfile(
                name="p",
                backend="llamacpp",
                model_file="$(touch /tmp/llmux-pwned) model.gguf",
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"):
                path = profile_store.render_env(profile)

            rendered = path.read_text()
            self.assertIn("MODEL_FILE='$(touch /tmp/llmux-pwned) model.gguf'", rendered)

    def test_render_env_rejects_invalid_env_var_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                env_vars={"BAD-NAME": "value"},
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", Path(tmp) / ".runtime"):
                with self.assertRaises(ValueError):
                    profile_store.render_env(profile)

    def test_save_profile_does_not_write_yaml_when_env_render_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_yaml = root / "profiles.yaml"
            profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                env_vars={"BAD-NAME": "value"},
            )

            with patch("tui.common.profile_store.PROFILES_YAML", profiles_yaml), patch(
                "tui.common.profile_store.RUNTIME_DIR", root / ".runtime"
            ):
                with self.assertRaises(ValueError):
                    profile_store.save_profile(profile)

            self.assertEqual(
                profiles_yaml.read_text(), "version: 1\ndefaults: {}\nprofiles: []\n"
            )

    def test_vllm_env_parser_round_trips_single_quote_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = profile_store.StoredProfile(
                name="p",
                backend="vllm",
                model_id="O'Reilly model",
            )

            with patch("tui.common.profile_store.RUNTIME_DIR", root / ".runtime"):
                env_path = profile_store.render_env(profile)

            parsed = backend._parse_env_file(env_path)
            self.assertEqual(parsed["MODEL_ID"], "O'Reilly model")

    def test_invalid_backend_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            profile_store.list_profiles("bogus")  # type: ignore[arg-type]

    def test_profile_store_cli_invalid_backend_returns_usage_error(self) -> None:
        with patch("sys.argv", ["profile_store", "list", "bogus"]):
            self.assertEqual(profile_store._cli(), 2)


class DeleteProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._profile_name = "__test_del_profile__"
        self._config_name = "__test_del_shared_config__"
        self._profile_path = backend.RUNTIME_DIR / f"{self._profile_name}.env"
        self._config_path = backend.CONFIG_DIR / f"{self._config_name}.yaml"
        self._other_profile_path = backend.RUNTIME_DIR / "__test_del_other__.env"

    def tearDown(self) -> None:
        backend.delete_profile(self._profile_name)
        backend.delete_profile("__test_del_other__")
        self._config_path.unlink(missing_ok=True)

    def test_delete_removes_profile_file(self) -> None:
        backend.save_profile(
            backend.Profile(name=self._profile_name, container_name="x", port="9000")
        )
        self.assertTrue(self._profile_path.exists())

        backend.delete_profile(self._profile_name)
        self.assertFalse(self._profile_path.exists())

    def test_delete_with_delete_config_removes_orphan_config(self) -> None:
        backend.save_config(backend.Config(name=self._config_name, model="org/m"))
        backend.save_profile(
            backend.Profile(
                name=self._profile_name,
                container_name="x",
                port="9000",
                config_name=self._config_name,
            )
        )

        backend.delete_profile(self._profile_name, delete_config=True)
        self.assertFalse(self._profile_path.exists())
        self.assertFalse(self._config_path.exists())

    def test_delete_with_delete_config_keeps_shared_config(self) -> None:
        backend.save_config(backend.Config(name=self._config_name, model="org/m"))
        backend.save_profile(
            backend.Profile(
                name=self._profile_name,
                container_name="x",
                port="9000",
                config_name=self._config_name,
            )
        )
        backend.save_profile(
            backend.Profile(
                name="__test_del_other__",
                container_name="y",
                port="9001",
                config_name=self._config_name,
            )
        )

        backend.delete_profile(self._profile_name, delete_config=True)
        self.assertFalse(self._profile_path.exists())
        self.assertTrue(
            self._config_path.exists(),
            "config must remain because another profile still references it",
        )


class BuildLoraOptionsTests(unittest.TestCase):
    def test_returns_empty_when_lora_disabled(self) -> None:
        profile = backend.Profile(name="p", enable_lora="false", max_loras="4")
        self.assertEqual(_build_lora_options(profile), "")

    def test_builds_enable_only_without_optional_fields(self) -> None:
        profile = backend.Profile(name="p", enable_lora="true")
        self.assertEqual(_build_lora_options(profile), "--enable-lora")

    def test_includes_loras_and_rank(self) -> None:
        profile = backend.Profile(
            name="p",
            enable_lora="true",
            max_loras="4",
            max_lora_rank="32",
        )
        result = _build_lora_options(profile)
        self.assertIn("--enable-lora", result)
        self.assertIn("--max-loras 4", result)
        self.assertIn("--max-lora-rank 32", result)

    def test_converts_lora_modules_comma_to_space(self) -> None:
        profile = backend.Profile(
            name="p",
            enable_lora="true",
            lora_modules="alpha=/path/a,beta=/path/b",
        )
        result = _build_lora_options(profile)
        self.assertIn("--lora-modules alpha=/path/a beta=/path/b", result)


class EnsureCommonEnvTests(unittest.TestCase):
    def test_missing_common_env_returns_error(self) -> None:
        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", Path("/nonexistent/.env.common")):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any(".env.common" in m for m in messages))

    def test_missing_hf_cache_path_returns_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("HF_CACHE_PATH" in m for m in messages))

    def test_relative_hf_cache_path_rejected(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=relative/path\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("absolute" in m for m in messages))

    def test_valid_absolute_path_succeeds(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=/abs/cache\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertTrue(ok)
        self.assertEqual(messages, [])

    def test_lora_requires_lora_base_path(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
            tmp.write("HF_CACHE_PATH=/abs/cache\n")
            tmp_path = Path(tmp.name)
        self.addCleanup(tmp_path.unlink, missing_ok=True)

        profile = backend.Profile(name="p", enable_lora="true")
        with patch("tui.backends.vllm.backend_runtime.COMMON_ENV", tmp_path):
            ok, messages = _ensure_common_env(profile)
        self.assertFalse(ok)
        self.assertTrue(any("LORA_BASE_PATH" in m for m in messages))


class DetectGpuArchTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_gpu_keeps_dot_form(self) -> None:
        async def fake_run(*args, **kwargs):
            return 0, "8.9\n"

        with patch("tui.backends.vllm.backend_runtime.run_command", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "8.9")

    async def test_multi_gpu_mixed_capabilities_deduped(self) -> None:
        async def fake_run(*args, **kwargs):
            return 0, "8.9\n8.6\n8.9\n"

        with patch("tui.backends.vllm.backend_runtime.run_command", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "8.6 8.9")

    async def test_failure_returns_empty(self) -> None:
        async def fake_run(*args, **kwargs):
            return 1, ""

        with patch("tui.backends.vllm.backend_runtime.run_command", fake_run):
            result = await _detect_gpu_arch()
        self.assertEqual(result, "")


class ForceLocalArchForDeepEPTests(unittest.TestCase):
    def _write_tmp(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".Dockerfile", delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_rewrites_hardcoded_deepep_arch_line(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    mkdir -p /tmp/ep_kernels_workspace/dist && \\\n"
            "    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \\\n"
            "    /tmp/install_python_libraries.sh \\\n"
            "        --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("Patched DeepEP stage", message)
        patched = dockerfile.read_text()
        self.assertIn('export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \\', patched)
        self.assertNotIn("9.0a 10.0a", patched)

    def test_accepts_already_dynamic_arch_line(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST}\" && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("already respects local TORCH_CUDA_ARCH_LIST", message)

    def test_fails_when_deepep_export_line_missing(self) -> None:
        dockerfile = self._write_tmp(
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    mkdir -p /tmp/ep_kernels_workspace/dist && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertFalse(ok)
        self.assertIn("could not locate DeepEP arch export line", message)

    def test_ignores_copy_line_and_patches_run_step(self) -> None:
        dockerfile = self._write_tmp(
            "COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh\n"
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \\\n"
            "    /tmp/install_python_libraries.sh --workspace /tmp/ep_kernels_workspace\n"
        )

        ok, message = _force_local_arch_for_deepep(dockerfile)

        self.assertTrue(ok)
        self.assertIn("Patched DeepEP stage", message)


class PickPreferredTagTests(unittest.TestCase):
    def test_prefers_highest_stable_version(self) -> None:
        self.assertEqual(_pick_preferred_tag(["v0.6.0", "v0.8.2", "v0.7.3"]), "v0.8.2")

    def test_returns_versioned_over_latest(self) -> None:
        self.assertEqual(_pick_preferred_tag(["v0.8.2", "latest"]), "v0.8.2")

    def test_ignores_latest_alone(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["latest", "random-tag"]))

    def test_ignores_nightly_alone(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["nightly", "random-tag"]))

    def test_returns_none_when_no_versioned_tag(self) -> None:
        self.assertIsNone(_pick_preferred_tag(["zeta", "alpha", "mu"]))

    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(_pick_preferred_tag([]))


class DockerHubTagLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_release_version_scans_next_page_when_first_page_has_no_stable(self) -> None:
        page1 = {
            "results": [{"name": "nightly"}, {"name": "latest"}],
            "next": "https://hub.docker.com/page2",
        }
        page2 = {
            "results": [{"name": "v0.19.0"}, {"name": "v0.19.1"}],
            "next": None,
        }
        fetch = AsyncMock(side_effect=[page1, page2])

        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()

        self.assertEqual(version, "v0.19.1")
        self.assertEqual(fetch.await_count, 2)

    async def test_release_version_returns_unknown_when_fetch_fails(self) -> None:
        fetch = AsyncMock(return_value=None)
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "unknown")

    async def test_release_version_falls_back_to_registry_domain(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                {
                    "results": [{"name": "v0.20.0"}],
                    "next": None,
                },
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "v0.20.0")
        self.assertEqual(fetch.await_count, 2)

    async def test_release_version_falls_back_to_docker_registry_tags(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                None,
                None,
                None,
                None,
                None,
                {"token": "token"},
                {"tags": ["latest", "nightly", "v0.19.0", "v0.20.1"]},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            version = await get_dockerhub_release_version()
        self.assertEqual(version, "v0.20.1")

    async def test_nightly_date_parses_last_updated(self) -> None:
        fetch = AsyncMock(return_value={"last_updated": "2026-04-23T12:34:56.000000Z"})
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "2026-04-23")

    async def test_nightly_date_falls_back_to_registry_domain(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                {"last_updated": "2026-04-24T01:02:03.000000Z"},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "2026-04-24")
        self.assertEqual(fetch.await_count, 2)

    async def test_nightly_date_returns_available_from_docker_registry_tags(self) -> None:
        fetch = AsyncMock(
            side_effect=[
                None,
                None,
                None,
                None,
                None,
                None,
                {"token": "token"},
                {"tags": ["nightly", "v0.20.1"]},
            ]
        )
        with patch(
            "tui.backends.vllm.backend_inspect._fetch_json_url",
            fetch,
        ):
            nightly_date = await get_dockerhub_nightly_date()
        self.assertEqual(nightly_date, "available")


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
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertEqual(conflict, "profile 'other'")

    async def test_check_port_conflict_returns_none_when_profiles_stopped(self) -> None:
        """Static profile-to-profile port overlap with no running container must not conflict."""
        profile = backend.Profile(name="current", container_name="current", port="18999")
        other = backend.Profile(name="other", container_name="other", port="18999")

        async def fake_run_command(*args, **kwargs):
            return 0, ""

        with patch.dict(
            backend.check_port_conflict.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current", "other"],
                "load_profile": lambda n: {"current": profile, "other": other}[n],
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertIsNone(conflict)

    async def test_check_port_conflict_detects_external_process(self) -> None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        bound_port = sock.getsockname()[1]
        self.addCleanup(sock.close)

        profile = backend.Profile(name="current", container_name="current", port=str(bound_port))

        async def fake_run_command(*args, **kwargs):
            return 0, ""

        with patch.dict(
            backend.check_port_conflict.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["current"],
                "load_profile": lambda n: profile,
            },
        ):
            conflict = await backend.check_port_conflict(profile)

        self.assertIsNotNone(conflict)
        self.assertIn(str(bound_port), conflict)


class StreamContainerUpPortConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_port_conflict_stops_before_preflight(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_check_port_conflict(_profile):
            return "another process"

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        self.assertEqual(events[-1], ("rc", 1))
        self.assertIn("Port 8000 is already in use", events[0][1])

    async def test_no_port_conflict_reaches_common_env_preflight(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_check_port_conflict(_profile):
            return None

        globals_dict = backend.stream_container_up.__globals__
        with patch.dict(
            globals_dict,
            {
                "load_profile": lambda _: profile,
                "check_port_conflict": fake_check_port_conflict,
                "_ensure_common_env": lambda _profile: (False, ["common env missing"]),
            },
        ):
            events = [event async for event in backend.stream_container_up("p")]

        self.assertEqual(events, [("log", "common env missing"), ("rc", 1)])


class PostStartValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fails_when_container_cannot_be_inspected(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            return 1, "No such object: p"

        with patch.dict(
            _post_start_validation.__globals__,
            {"run_command": fake_run_command},
        ):
            ok, messages = await _post_start_validation(profile, timeout=0.1, poll_interval=0.05)

        self.assertFalse(ok)
        self.assertTrue(any("could not inspect container" in m for m in messages))

    async def test_fails_when_container_exits_during_startup(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "exited\tunhealthy"
            if args[:2] == ("docker", "logs"):
                return 0, "line-a\nline-b\n"
            return 0, ""

        with patch.dict(
            _post_start_validation.__globals__,
            {
                "run_command": fake_run_command,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages = await _post_start_validation(profile, timeout=0.1, poll_interval=0.05)

        self.assertFalse(ok)
        self.assertTrue(any("exited during startup" in m for m in messages))
        self.assertTrue(any("line-b" in m for m in messages))

    async def test_warns_when_models_endpoint_not_ready_before_timeout(self) -> None:
        profile = backend.Profile(name="p", container_name="p", port="8000")

        async def fake_run_command(*args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return 0, "running\tstarting"
            return 0, ""

        with patch.dict(
            _post_start_validation.__globals__,
            {
                "run_command": fake_run_command,
                "_models_endpoint_ready": AsyncMock(return_value=False),
            },
        ):
            ok, messages = await _post_start_validation(profile, timeout=0.1, poll_interval=0.05)

        self.assertTrue(ok)
        self.assertTrue(any("/v1/models is not ready yet" in m for m in messages))


class VllmRuntimeWarningTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_verification_warns_when_exec_fails(self) -> None:
        async def fake_run_command(*args, **kwargs):
            return 1, "exec failed"

        with patch.dict(
            _verify_vllm_version.__globals__,
            {"run_command": fake_run_command},
        ):
            events = [event async for event in _verify_vllm_version("p", "v0.19.1")]

        self.assertEqual(events[0][0], "log")
        self.assertIn("could not verify vLLM version", events[0][1])

    async def test_gpu_conflicts_include_llamacpp_profiles(self) -> None:
        profile = backend.Profile(name="v", container_name="v", gpu_id="0")
        other = profile_store.StoredProfile(
            name="l",
            backend="llamacpp",
            container_name="l",
            gpu_id="0",
        )

        async def fake_run_command(*args, **kwargs):
            return 0, "l\n"

        with patch.dict(
            _gpu_conflict_messages.__globals__,
            {
                "run_command": fake_run_command,
                "list_profile_names": lambda: ["v"],
                "profile_store": type(
                    "ProfileStoreStub",
                    (),
                    {"list_profiles": staticmethod(lambda backend_name: [other])},
                ),
            },
        ):
            messages = await _gpu_conflict_messages(profile)

        self.assertTrue(any("llama.cpp" in m and "GPU 0" in m for m in messages))


class LlamacppValidationTests(unittest.TestCase):
    def test_validate_name_is_compose_safe(self) -> None:
        self.assertTrue(lbackend.validate_name("qwen3_0-6b"))
        self.assertFalse(lbackend.validate_name("Qwen3"))
        self.assertFalse(lbackend.validate_name("dot.name"))


class LlamacppRenderOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp" / "render-override.py"
        spec = importlib.util.spec_from_file_location("render_override_for_tests", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_profile_model_file_is_used_when_config_omits_model_file(self) -> None:
        command = self.module.render_command({"ctx-size": 2048}, "model.gguf")
        self.assertIn("--model", command)
        self.assertIn("/models/model.gguf", command)

    def test_missing_model_file_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            self.module.render_command({"ctx-size": 2048}, "")


class QuickSetupSuffixLogicTests(unittest.TestCase):
    """Smoke-test the name collision suffix algorithm used in QuickSetupScreen."""

    @staticmethod
    def _resolve(safe_name: str, existing_profiles: set[str], existing_configs: set[str]) -> str:
        original = safe_name
        suffix = 0
        while safe_name in existing_profiles or safe_name in existing_configs:
            suffix += 1
            safe_name = f"{original}-{suffix}"
        return safe_name

    def test_no_collision_returns_original(self) -> None:
        self.assertEqual(self._resolve("llama", set(), set()), "llama")

    def test_collision_appends_suffix_1(self) -> None:
        self.assertEqual(self._resolve("llama", {"llama"}, set()), "llama-1")

    def test_collision_increments_until_free(self) -> None:
        self.assertEqual(
            self._resolve("llama", {"llama", "llama-1", "llama-2"}, {"llama-3"}),
            "llama-4",
        )

    def test_collision_across_profile_and_config(self) -> None:
        self.assertEqual(self._resolve("llama", {"llama"}, {"llama-1"}), "llama-2")


if __name__ == "__main__":
    unittest.main()
