from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from textual.app import App
from textual.widgets import Input, Select, Switch

from tui.common import profile_store


@pytest.fixture
def profile_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
    runtime_dir = tmp_path / ".runtime"
    monkeypatch.setattr(profile_store, "PROFILES_YAML", profiles_yaml)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", runtime_dir)
    return tmp_path


def write_profiles(root: Path, profiles: list[dict]) -> None:
    (root / "profiles.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "defaults": {}, "profiles": profiles},
            sort_keys=False,
        )
    )


def test_rename_rejects_effective_container_collision(profile_root: Path) -> None:
    write_profiles(
        profile_root,
        [
            {"name": "old", "backend": "vllm"},
            {
                "name": "other",
                "backend": "llamacpp",
                "container_name": "target",
            },
        ],
    )

    with pytest.raises(ValueError, match="container name"):
        profile_store.rename_profile("old", "target", "vllm")

    assert profile_store.load_profile("old", "vllm") is not None
    assert profile_store.load_profile("target", "vllm") is None


@pytest.mark.parametrize(
    ("backend", "module_name"),
    [
        ("vllm", "tui.backends.vllm.screens.profile"),
        ("llamacpp", "tui.backends.llamacpp.screens.profile"),
    ],
)
@pytest.mark.asyncio
async def test_profile_form_rename_and_edit_is_atomic(
    profile_root: Path,
    backend: str,
    module_name: str,
) -> None:
    write_profiles(
        profile_root,
        [
            {"name": "old", "backend": backend},
            {
                "name": "blocker",
                "backend": "llamacpp" if backend == "vllm" else "vllm",
                "container_name": "taken",
            },
        ],
    )
    module = __import__(module_name, fromlist=["ProfileFormScreen"])
    screen = module.ProfileFormScreen(module.load_profile("old"))
    app = App()

    with patch.object(module, "list_config_names", return_value=[]), patch(
        "tui.common.docker.running_container_names",
        AsyncMock(return_value=set()),
    ):
        async with app.run_test(size=(120, 50)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#name-input", Input).value = "renamed"
            screen.query_one("#container-input", Input).value = "taken"
            screen.query_one("#save-btn").press()
            await pilot.pause()

    assert profile_store.load_profile("old", backend) is not None
    assert profile_store.load_profile("renamed", backend) is None


def test_quick_setup_concurrent_requests_do_not_mix_profile_and_config(
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tui.backends.vllm import backend_common, backend_storage
    from tui.cli.profile import _quick_setup_vllm

    config_dir = profile_root / "config" / "vllm"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(backend_common, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(backend_storage, "CONFIG_DIR", config_dir)
    barrier = threading.Barrier(2)
    real_list_profile_names = profile_store.list_profile_names

    def synchronized_list_profile_names(backend: str) -> list[str]:
        result = real_list_profile_names(backend)
        if backend == "llamacpp":
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(profile_store, "list_profile_names", synchronized_list_profile_names)

    def create(model: str) -> None:
        _quick_setup_vllm(
            model=model,
            name="shared",
            port=8000,
            gpu_id="0",
            gpu_memory_utilization="0.9",
            enable_lora=False,
            copy_config_from="",
            use_recipe=False,
            recipe_from="",
            recipe_variant="",
            recipe_features=[],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create, model) for model in ("org/one", "org/two")]
        for future in futures:
            future.result(timeout=10)

    profiles = profile_store.list_profiles("vllm")
    assert {profile.name for profile in profiles} == {"shared", "shared-1"}
    for stored in profiles:
        config = backend_storage.load_config(stored.config_name)
        assert config.model == stored.model_id


@pytest.mark.asyncio
async def test_llamacpp_quick_setup_rejects_selection_from_previous_repo(
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tui.backends.llamacpp import backend
    from tui.backends.llamacpp.screens import quick_setup as module

    config_dir = profile_root / "config" / "llamacpp"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(backend, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(module, "CONFIG_DIR", config_dir)
    app = App()
    screen = module.QuickSetupScreen()

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        screen._last_repo = "org/repo-a"
        select = screen.query_one("#gguf-select", Select)
        select.set_options([("model-a.gguf", "model-a.gguf")])
        select.value = "model-a.gguf"
        screen.query_one("#repo-input", Input).value = "org/repo-b"
        screen.query_one("#name-input", Input).value = "stale-selection"
        screen.query_one("#create-btn").press()
        await pilot.pause()

    assert profile_store.list_profiles("llamacpp") == []
    assert list(config_dir.glob("*.yaml")) == []


@pytest.mark.asyncio
async def test_llamacpp_quick_setup_defaults_are_shared_by_cli_and_tui() -> None:
    from typer.main import get_command

    from tui.backends.llamacpp import defaults
    from tui.backends.llamacpp.screens import quick_setup as tui_module
    from tui.cli import profile as cli_module

    constant_names = (
        "QUICK_SETUP_CTX_SIZE",
        "QUICK_SETUP_N_GPU_LAYERS",
        "QUICK_SETUP_CACHE_TYPE_K",
        "QUICK_SETUP_CACHE_TYPE_V",
        "QUICK_SETUP_FLASH_ATTN",
        "QUICK_SETUP_JINJA",
    )
    for name in constant_names:
        assert getattr(cli_module, name) is getattr(defaults, name)
        assert getattr(tui_module, name) is getattr(defaults, name)

    command = get_command(cli_module.app).commands["quick-setup"]
    cli_defaults = {param.name: param.default for param in command.params}
    assert cli_defaults["ctx_size"] == defaults.QUICK_SETUP_CTX_SIZE
    assert cli_defaults["n_gpu_layers"] == defaults.QUICK_SETUP_N_GPU_LAYERS
    assert cli_defaults["cache_type_k"] == defaults.QUICK_SETUP_CACHE_TYPE_K
    assert cli_defaults["cache_type_v"] == defaults.QUICK_SETUP_CACHE_TYPE_V
    assert cli_defaults["flash_attn"] is defaults.QUICK_SETUP_FLASH_ATTN
    assert cli_defaults["jinja"] is defaults.QUICK_SETUP_JINJA

    app = App()
    screen = tui_module.QuickSetupScreen()
    with patch.object(tui_module, "list_config_names", return_value=[]):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            assert screen.query_one("#ctx-input", Input).value == defaults.QUICK_SETUP_CTX_SIZE
            assert (
                screen.query_one("#ngl-input", Input).value
                == defaults.QUICK_SETUP_N_GPU_LAYERS
            )
            assert (
                screen.query_one("#ctk-input", Input).value
                == defaults.QUICK_SETUP_CACHE_TYPE_K
            )
            assert (
                screen.query_one("#ctv-input", Input).value
                == defaults.QUICK_SETUP_CACHE_TYPE_V
            )
            assert (
                screen.query_one("#flash-attn-switch", Switch).value
                is defaults.QUICK_SETUP_FLASH_ATTN
            )
            assert (
                screen.query_one("#jinja-switch", Switch).value
                is defaults.QUICK_SETUP_JINJA
            )


def test_unknown_boolean_string_is_rejected(profile_root: Path) -> None:
    write_profiles(
        profile_root,
        [{"name": "bad-bool", "backend": "vllm", "enable_lora": "treu"}],
    )

    with pytest.raises(ValueError, match="enable_lora"):
        profile_store.list_profiles("vllm")


def test_runtime_env_rewrite_forces_owner_only_mode(profile_root: Path) -> None:
    runtime_path = profile_root / ".runtime" / "vllm" / "secure.env"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text("TOKEN=old\n")
    runtime_path.chmod(0o644)

    profile_store.render_env(
        profile_store.StoredProfile(name="secure", backend="vllm")
    )

    assert runtime_path.stat().st_mode & 0o777 == 0o600


def test_config_reference_transaction_uses_effective_links_and_rolls_back(
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_profiles(
        profile_root,
        [
            {"name": "implicit", "backend": "vllm"},
            {
                "name": "explicit",
                "backend": "vllm",
                "config_name": "implicit",
            },
        ],
    )
    old_config = profile_root / "config" / "vllm" / "implicit.yaml"
    new_config = profile_root / "config" / "vllm" / "renamed.yaml"
    old_config.parent.mkdir(parents=True)
    old_config.write_text("model: org/model\n")
    original_yaml = (profile_root / "profiles.yaml").read_text()
    real_replace = os.replace
    failed = False

    def fail_profiles_once(src: str | Path, dst: str | Path) -> None:
        nonlocal failed
        if Path(dst) == profile_root / "profiles.yaml" and not failed:
            failed = True
            raise OSError("profiles replace failed")
        real_replace(src, dst)

    monkeypatch.setattr(profile_store.os, "replace", fail_profiles_once)

    with pytest.raises(OSError, match="profiles replace failed"):
        profile_store.repoint_config_references(
            "vllm",
            "implicit",
            "renamed",
            moves=((old_config, new_config),),
        )

    assert (profile_root / "profiles.yaml").read_text() == original_yaml
    assert old_config.read_text() == "model: org/model\n"
    assert not new_config.exists()


def test_profile_cascade_delete_uses_effective_config_link(profile_root: Path) -> None:
    write_profiles(
        profile_root,
        [{"name": "implicit", "backend": "llamacpp", "config_name": ""}],
    )
    config_dir = profile_root / "config" / "llamacpp"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "implicit.yaml"
    config_path.write_text("model-file: model.gguf\n")

    deleted = profile_store.delete_profile_with_config(
        "implicit", "llamacpp", config_dir
    )

    assert deleted is True
    assert profile_store.load_profile("implicit", "llamacpp") is None
    assert not config_path.exists()


def test_profile_cascade_keeps_effective_config_shared_by_another_profile(
    profile_root: Path,
) -> None:
    write_profiles(
        profile_root,
        [
            {"name": "shared", "backend": "vllm", "config_name": ""},
            {"name": "other", "backend": "vllm", "config_name": "shared"},
        ],
    )
    config_dir = profile_root / "config" / "vllm"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "shared.yaml"
    config_path.write_text("model: org/model\n")

    profile_store.delete_profile_with_config("shared", "vllm", config_dir)

    assert config_path.exists()
    assert profile_store.load_profile("other", "vllm") is not None


def test_quick_setup_transaction_rolls_back_profile_config_and_runtime(
    profile_root: Path,
) -> None:
    config_dir = profile_root / "config" / "vllm"
    config_dir.mkdir(parents=True)

    with pytest.raises(OSError, match="finalization failed"):
        with profile_store.quick_setup_transaction(
            "rollback",
            "vllm",
            config_dir,
        ) as final_name:
            (config_dir / f"{final_name}.yaml").write_text("model: org/model\n")
            profile_store.save_profile(
                profile_store.StoredProfile(
                    name=final_name,
                    backend="vllm",
                    config_name=final_name,
                )
            )
            raise OSError("finalization failed")

    assert profile_store.load_profile("rollback", "vllm") is None
    assert not (config_dir / "rollback.yaml").exists()
    assert not (profile_root / ".runtime" / "vllm" / "rollback.env").exists()


def test_config_reference_transaction_moves_source_read_under_lock(
    profile_root: Path,
) -> None:
    write_profiles(
        profile_root,
        [{"name": "implicit", "backend": "vllm"}],
    )
    config_dir = profile_root / "config" / "vllm"
    config_dir.mkdir(parents=True)
    source = config_dir / "implicit.yaml"
    destination = config_dir / "renamed.yaml"
    source.write_text("model: org/current\n")
    source.chmod(0o644)

    changed = profile_store.repoint_config_references(
        "vllm",
        "implicit",
        "renamed",
        moves=((source, destination),),
    )

    assert changed == ["implicit"]
    assert not source.exists()
    assert destination.read_text() == "model: org/current\n"
    assert destination.stat().st_mode & 0o777 == 0o644
    assert profile_store.load_profile("implicit", "vllm").config_name == "renamed"


def test_config_reference_transaction_moves_with_replacement_text(
    profile_root: Path,
) -> None:
    write_profiles(
        profile_root,
        [{"name": "implicit", "backend": "llamacpp"}],
    )
    config_dir = profile_root / "config" / "llamacpp"
    config_dir.mkdir(parents=True)
    source = config_dir / "implicit.yaml"
    destination = config_dir / "renamed.yaml"
    source.write_text("model-file: old.gguf\n")

    profile_store.repoint_config_references(
        "llamacpp",
        "implicit",
        "renamed",
        moves=((source, destination, "model-file: new.gguf\n"),),
    )

    assert not source.exists()
    assert destination.read_text() == "model-file: new.gguf\n"


def test_config_delete_accepts_path_safe_legacy_name(profile_root: Path) -> None:
    config_path = profile_root / "config" / "vllm" / "Legacy.Config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model: org/model\n")

    changed = profile_store.repoint_config_references(
        "vllm",
        "Legacy.Config",
        "",
        deletes=(config_path,),
    )

    assert changed == []
    assert not config_path.exists()


@pytest.mark.asyncio
async def test_vllm_recipe_fetch_reports_gpu_probe_failure() -> None:
    from tui.backends.vllm.screens import quick_setup as module
    from tui.common.recipes import Recipe

    app = App()
    screen = module.QuickSetupScreen()

    with patch(
        "tui.common.recipes.fetch_recipe",
        AsyncMock(return_value=Recipe(model_id="org/model")),
    ), patch(
        "tui.common.docker.get_gpu_info",
        AsyncMock(side_effect=RuntimeError("nvidia-smi query failed")),
    ):
        async with app.run_test(size=(120, 50)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen._fetch_recipe("org/model")
            await pilot.pause()
            status = screen.query_one("#recipe-status")
            assert "GPU" in str(status.content)
            assert "nvidia-smi query failed" in str(status.content)
