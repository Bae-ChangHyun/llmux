import unittest

from tui.common import recipes

# A trimmed copy of the real Qwen3-32B recipe schema (network-free fixture).
_QWEN32B = {
    "meta": {"title": "Qwen3-32B", "description": "dense model"},
    "model": {
        "model_id": "Qwen/Qwen3-32B",
        "min_vllm_version": "0.8.5",
        "context_length": 40960,
        "base_args": [],
    },
    "features": {
        "tool_calling": {
            "description": "Hermes parser",
            "args": ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        },
        "reasoning": {
            "description": "Qwen3 thinking",
            "args": ["--reasoning-parser", "qwen3"],
        },
    },
    "variants": {
        "default": {"precision": "bf16", "vram_minimum_gb": 77},
        "fp8": {"model_id": "Qwen/Qwen3-32B-FP8", "precision": "fp8", "vram_minimum_gb": 39},
        "awq": {
            "model_id": "Qwen/Qwen3-32B-AWQ",
            "precision": "int4",
            "vram_minimum_gb": 20,
            "extra_args": ["--quantization", "awq"],
        },
    },
}


class RecipeUrlTests(unittest.TestCase):
    def test_url_from_model_id(self):
        self.assertEqual(
            recipes.recipe_url("Qwen/Qwen3-32B"),
            f"{recipes.RECIPES_RAW_BASE}/Qwen/Qwen3-32B.yaml",
        )


class RecipeParseTests(unittest.TestCase):
    def setUp(self):
        self.r = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")

    def test_top_level_fields(self):
        self.assertEqual(self.r.model_id, "Qwen/Qwen3-32B")
        self.assertEqual(self.r.min_vllm_version, "0.8.5")
        self.assertEqual(self.r.context_length, 40960)

    def test_variants_parsed_with_vram(self):
        by_name = {v.name: v for v in self.r.variants}
        self.assertEqual(by_name["fp8"].model_id, "Qwen/Qwen3-32B-FP8")
        self.assertEqual(by_name["awq"].vram_minimum_gb, 20)
        self.assertEqual(by_name["awq"].extra_args, ["--quantization", "awq"])

    def test_features_parsed(self):
        by_name = {f.name: f for f in self.r.features}
        self.assertIn("--tool-call-parser", by_name["tool_calling"].args)


class ArgsToConfigTests(unittest.TestCase):
    def test_value_and_bare_flag(self):
        cfg = recipes.args_to_config(
            ["--tool-call-parser", "hermes", "--enable-auto-tool-choice"]
        )
        self.assertEqual(cfg, {"tool-call-parser": "hermes", "enable-auto-tool-choice": True})

    def test_yaml_typing(self):
        self.assertEqual(recipes.args_to_config(["--max-model-len", "8192"]),
                         {"max-model-len": 8192})


class BuildConfigTests(unittest.TestCase):
    def setUp(self):
        self.r = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")
        self.variants = {v.name: v for v in self.r.variants}

    def test_default_variant_seeds_context_len(self):
        model, params = recipes.build_config(self.r, self.variants["default"], [])
        self.assertEqual(model, "Qwen/Qwen3-32B")
        self.assertEqual(params["max-model-len"], 40960)

    def test_fp8_variant_swaps_model(self):
        model, params = recipes.build_config(self.r, self.variants["fp8"], [])
        self.assertEqual(model, "Qwen/Qwen3-32B-FP8")

    def test_awq_variant_adds_quantization(self):
        model, params = recipes.build_config(self.r, self.variants["awq"], [])
        self.assertEqual(params["quantization"], "awq")

    def test_enabled_features_append_args(self):
        _, params = recipes.build_config(
            self.r, self.variants["default"], ["tool_calling", "reasoning"]
        )
        self.assertEqual(params["tool-call-parser"], "hermes")
        self.assertEqual(params["reasoning-parser"], "qwen3")
        self.assertTrue(params["enable-auto-tool-choice"])


class ReviewVariantPickTests(unittest.TestCase):
    """The review screen's GPU-fit / default-variant logic is pure given the
    GPU size, so it's exercised without a running App."""

    def setUp(self):
        from tui.backends.vllm.screens.recipe import RecipeReviewScreen

        self.r = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")
        self.Screen = RecipeReviewScreen

    def test_picks_highest_quality_that_fits(self):
        # 40 GB GPU → default(77) and awq(20) both considered; fp8(39) is the
        # first in recipe order (quality-descending) that fits.
        s = self.Screen(self.r, gpu_total_gb=40.0)
        self.assertEqual(self.r.variants[s._default_variant_index()].name, "fp8")

    def test_big_gpu_picks_default(self):
        s = self.Screen(self.r, gpu_total_gb=80.0)
        self.assertEqual(self.r.variants[s._default_variant_index()].name, "default")

    def test_no_fit_picks_smallest(self):
        # 16 GB fits nothing (smallest is awq at 20 GB) → smallest is offered.
        s = self.Screen(self.r, gpu_total_gb=16.0)
        self.assertEqual(self.r.variants[s._default_variant_index()].name, "awq")

    def test_unknown_gpu_leaves_fit_none(self):
        s = self.Screen(self.r, gpu_total_gb=None)
        self.assertIsNone(s._fits(self.r.variants[0]))


class DeriveNameTests(unittest.TestCase):
    def test_config_name_from_model_id(self):
        from tui.cli.config import _derive_config_name

        self.assertEqual(_derive_config_name("Qwen/Qwen3-32B-AWQ"), "qwen3-32b-awq")


class FromRecipeWriteTests(unittest.TestCase):
    """The write path (no --json) was crashing on a missing `param_hint`; only
    --list / --json had been exercised. Cover the actual config creation."""

    def test_from_recipe_writes_config(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from tui.backends.vllm import backend_common as vc
        from tui.backends.vllm import backend_storage as vs
        from tui.cli import config as cli_config

        recipe = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")
        with tempfile.TemporaryDirectory() as tmp:
            # variant="awq" is explicit, so no GPU auto-pick / network is touched.
            with patch.object(cli_config, "_config_dir", lambda b: Path(tmp)), \
                 patch.object(vs, "CONFIG_DIR", Path(tmp)), \
                 patch.object(vc, "CONFIG_DIR", Path(tmp)), \
                 patch("tui.common.recipes.fetch_recipe",
                       new=AsyncMock(return_value=recipe)):
                cli_config.config_from_recipe(
                    "Qwen/Qwen3-32B", recipe_from="", variant="awq",
                    feature=["reasoning"], name="q32-awq", list_only=False,
                    json_out=False, overwrite=False, merge=False,
                )
            written = (Path(tmp) / "q32-awq.yaml").read_text()
            self.assertIn("Qwen/Qwen3-32B-AWQ", written)   # awq swaps the model
            self.assertIn("quantization", written)
            self.assertIn("reasoning-parser", written)


class RecipeFromOtherModelTests(unittest.TestCase):
    """--recipe-from borrows another model's recipe; the configured model stays."""

    def _write(self, tmp, **kwargs):
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from tui.backends.vllm import backend_common as vc
        from tui.backends.vllm import backend_storage as vs
        from tui.cli import config as cli_config

        recipe = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")
        call = dict(
            recipe_from="", variant="awq", feature=[], name="borrowed",
            list_only=False, json_out=False, overwrite=False, merge=False,
        )
        call.update(kwargs)
        model_id = call.pop("model_id")
        with patch.object(cli_config, "_config_dir", lambda b: Path(tmp)), \
             patch.object(vs, "CONFIG_DIR", Path(tmp)), \
             patch.object(vc, "CONFIG_DIR", Path(tmp)), \
             patch("tui.common.recipes.fetch_recipe",
                   new=AsyncMock(return_value=recipe)):
            cli_config.config_from_recipe(model_id, **call)

    def test_keeps_the_requested_model(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, model_id="cpatonn/Qwen3-32B-AWQ", recipe_from="Qwen/Qwen3-32B"
            )
            written = (Path(tmp) / "borrowed.yaml").read_text()
            # The awq variant would normally swap in Qwen's own AWQ checkpoint.
            self.assertIn("cpatonn/Qwen3-32B-AWQ", written)
            self.assertNotIn("Qwen/Qwen3-32B-AWQ", written)
            self.assertIn("quantization", written)

    def test_merge_keeps_existing_params(self):
        import tempfile
        from pathlib import Path

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, model_id="Qwen/Qwen3-32B")
            (Path(tmp) / "borrowed.yaml").write_text(
                (Path(tmp) / "borrowed.yaml").read_text() + "swap-space: 8\n"
            )
            self._write(
                tmp, model_id="Qwen/Qwen3-32B", feature=["reasoning"], merge=True
            )
            data = yaml.safe_load((Path(tmp) / "borrowed.yaml").read_text())
            self.assertEqual(data["swap-space"], 8)          # kept
            self.assertEqual(data["reasoning-parser"], "qwen3")  # added by the recipe

    def test_merge_requires_an_existing_config(self):
        import tempfile

        import typer

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(typer.BadParameter):
                self._write(tmp, model_id="Qwen/Qwen3-32B", merge=True)

    def test_merge_and_overwrite_are_exclusive(self):
        import tempfile

        import typer

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(typer.BadParameter):
                self._write(
                    tmp, model_id="Qwen/Qwen3-32B", merge=True, overwrite=True
                )


class ReviewTargetModelTests(unittest.TestCase):
    """A recipe fetched for another model contributes flags, never its model id."""

    def test_target_model_overrides_variant_model(self):
        from tui.backends.vllm.screens.recipe import RecipeReviewScreen

        r = recipes._parse(_QWEN32B, "Qwen/Qwen3-32B")
        screen = RecipeReviewScreen(
            r, gpu_total_gb=24.0, target_model="cpatonn/Qwen3-32B-AWQ"
        )
        screen._selected_variant = lambda: r.variants[2]   # awq
        screen._enabled_features = lambda: []
        model_id, params = screen._current()
        self.assertEqual(model_id, "cpatonn/Qwen3-32B-AWQ")
        self.assertEqual(params["quantization"], "awq")


if __name__ == "__main__":
    unittest.main()
