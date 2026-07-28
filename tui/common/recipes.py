"""Fetch vLLM serving recipes from the official vllm-project/recipes repo.

Each model has a structured YAML at
``models/<hf_org>/<hf_repo>.yaml`` (rendered on recipes.vllm.ai). We fetch the
raw file — no `gh` CLI or auth needed — parse the pieces that map onto an llmux
vLLM config (base args, opt-in features, precision variants with VRAM floors),
and let the caller build a config the user reviews before it's written.

The recipe args are flat CLI fragments (``["--tool-call-parser", "hermes"]``);
`args_to_config` folds them into the ``{flag: value}`` shape our config YAML uses.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import yaml

RECIPES_RAW_BASE = "https://raw.githubusercontent.com/vllm-project/recipes/main/models"


@dataclass
class RecipeVariant:
    name: str
    precision: str = ""
    vram_minimum_gb: float | None = None
    model_id: str = ""          # overrides the base model (e.g. an FP8 checkpoint)
    extra_args: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RecipeFeature:
    name: str
    description: str = ""
    args: list[str] = field(default_factory=list)


@dataclass
class Recipe:
    model_id: str
    title: str = ""
    description: str = ""
    min_vllm_version: str = ""
    context_length: int | None = None
    base_args: list[str] = field(default_factory=list)
    variants: list[RecipeVariant] = field(default_factory=list)
    features: list[RecipeFeature] = field(default_factory=list)


def recipe_url(model_id: str) -> str:
    """`org/repo` → the raw recipe URL. Extra path segments are ignored."""
    org, _, repo = model_id.strip().strip("/").partition("/")
    return f"{RECIPES_RAW_BASE}/{org}/{repo}.yaml"


def _parse(raw: dict, model_id: str) -> Recipe:
    model = raw.get("model") or {}
    meta = raw.get("meta") or {}

    variants: list[RecipeVariant] = []
    for name, v in (raw.get("variants") or {}).items():
        v = v or {}
        vram = v.get("vram_minimum_gb")
        variants.append(
            RecipeVariant(
                name=str(name),
                precision=str(v.get("precision", "")),
                vram_minimum_gb=float(vram) if isinstance(vram, (int, float)) else None,
                model_id=str(v.get("model_id", "")),
                extra_args=[str(a) for a in (v.get("extra_args") or [])],
                description=str(v.get("description", "")),
            )
        )

    features: list[RecipeFeature] = []
    for name, f in (raw.get("features") or {}).items():
        f = f or {}
        features.append(
            RecipeFeature(
                name=str(name),
                description=str(f.get("description", "")),
                args=[str(a) for a in (f.get("args") or [])],
            )
        )

    ctx = model.get("context_length")
    return Recipe(
        model_id=str(model.get("model_id", model_id)),
        title=str(meta.get("title", "")),
        description=str(meta.get("description", "")),
        min_vllm_version=str(model.get("min_vllm_version", "")),
        context_length=int(ctx) if isinstance(ctx, int) else None,
        base_args=[str(a) for a in (model.get("base_args") or [])],
        variants=variants,
        features=features,
    )


class RecipeUnavailable(RuntimeError):
    """The recipe could not be fetched — distinct from "there is no recipe"."""


async def fetch_recipe(model_id: str, timeout: int = 15) -> Recipe | None:
    """Fetch + parse the recipe for `model_id`.

    Returns None only when upstream has no recipe (404). A network failure or a
    malformed file raises RecipeUnavailable, so callers never tell the user
    "no recipe found" for a model whose recipe they simply couldn't reach.
    """
    if not model_id or "/" not in model_id:
        return None
    url = recipe_url(model_id)
    loop = asyncio.get_running_loop()

    def _do() -> Recipe | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llmux"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = yaml.safe_load(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RecipeUnavailable(f"{url} → HTTP {exc.code}") from exc
        except Exception as exc:
            raise RecipeUnavailable(f"{url} → {exc}") from exc
        if not isinstance(raw, dict):
            raise RecipeUnavailable(f"{url} → malformed recipe file")
        return _parse(raw, model_id)

    return await loop.run_in_executor(None, _do)


def args_to_config(args: list[str]) -> dict[str, Any]:
    """Fold a flat CLI arg list into the ``{flag: value}`` config shape.

    ``--foo bar`` → ``{"foo": "bar"}``; a bare ``--flag`` (next token is another
    flag or the list ends) → ``{"flag": True}``. Values are YAML-typed so
    ``--x 8192`` stores an int. Leading ``--`` is stripped to match config keys.
    """
    out: dict[str, Any] = {}
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if i + 1 < n and not args[i + 1].startswith("--"):
            raw = args[i + 1]
            try:
                out[key] = yaml.safe_load(raw)
            except yaml.YAMLError:
                out[key] = raw
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def build_config(
    recipe: Recipe,
    variant: RecipeVariant | None,
    feature_names: list[str],
) -> tuple[str, dict[str, Any]]:
    """Return ``(model_id, config_params)`` for the chosen variant + features.

    A variant may swap the model (an FP8/AWQ checkpoint) and add quantization
    args; enabled features append their args. ``max-model-len`` is seeded from
    the recipe's context length so the user sees a concrete starting value.
    """
    model_id = (variant.model_id if variant and variant.model_id else recipe.model_id)
    args = list(recipe.base_args)
    if variant:
        args += variant.extra_args
    enabled = {f.name for f in recipe.features if f.name in feature_names}
    for f in recipe.features:
        if f.name in enabled:
            args += f.args
    params = args_to_config(args)
    if recipe.context_length and "max-model-len" not in params:
        params["max-model-len"] = recipe.context_length
    return model_id, params
