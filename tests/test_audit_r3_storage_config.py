from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.app import App
from textual.widgets import Input, Select
import typer

from tui.common import profile_store


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tui.backends.vllm import backend_common, backend_storage
    from tui.cli import config as cli_config

    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
    config_dir = tmp_path / "config" / "vllm"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(profile_store, "PROFILES_YAML", profiles_yaml)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path / ".runtime")
    monkeypatch.setattr(backend_common, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(backend_storage, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        cli_config,
        "_config_dir",
        lambda backend: config_dir if backend == "vllm" else tmp_path / "config" / backend,
    )
    return tmp_path


def test_profile_loader_rejects_duplicate_mapping_keys(storage_root: Path) -> None:
    path = storage_root / "profiles.yaml"
    path.write_text(
        "version: 1\n"
        "defaults: {}\n"
        "profiles:\n"
        "- name: first\n  backend: vllm\n"
        "profiles:\n"
        "- name: last\n  backend: vllm\n"
    )

    with pytest.raises(ValueError, match=r"profiles\.yaml.*duplicate.*profiles"):
        profile_store.list_profiles("vllm")


def test_vllm_loader_rejects_duplicate_mapping_keys(storage_root: Path) -> None:
    from tui.backends.vllm import backend_storage

    path = storage_root / "config" / "vllm" / "duplicate.yaml"
    path.write_text(
        "model: org/first\n"
        "model: org/last\n"
        "gpu-memory-utilization: 0.9\n"
    )

    with pytest.raises(ValueError, match=r"duplicate\.yaml.*duplicate.*model"):
        backend_storage.load_config("duplicate")


def test_cli_typed_set_rejects_malformed_yaml() -> None:
    from tui.cli import config as cli_config

    with pytest.raises(typer.BadParameter, match="structured"):
        cli_config._parse_set_kv(["structured={"])

    assert cli_config._parse_set_kv(["plain=unquoted"])["plain"] == "unquoted"
    assert cli_config._parse_set_kv(["literal='{'"])["literal"] == "{"


def test_disabled_marker_rejects_malformed_typed_value() -> None:
    from tui.common.config_markers import parse_disabled_markers

    with pytest.raises(ValueError, match="broken"):
        parse_disabled_markers("# llmux:disabled broken: {\n")


@pytest.mark.parametrize(
    "text, field",
    [
        (
            "model:\n  org: repo\ngpu-memory-utilization: 0.9\n",
            "model",
        ),
        (
            "model: org/model\ngpu-memory-utilization:\n  bad: value\n",
            "gpu-memory-utilization",
        ),
        (
            "model: org/model\ngpu-memory-utilization: true\n",
            "gpu-memory-utilization",
        ),
        (
            "model: org/model\ngpu-memory-utilization: 1.5\n",
            "gpu-memory-utilization",
        ),
    ],
)
def test_vllm_loader_rejects_invalid_core_schema(
    storage_root: Path,
    text: str,
    field: str,
) -> None:
    from tui.backends.vllm import backend_storage

    (storage_root / "config" / "vllm" / "invalid.yaml").write_text(text)

    with pytest.raises(ValueError, match=field):
        backend_storage.load_config("invalid")


def test_vllm_config_save_rejects_stale_snapshot(storage_root: Path) -> None:
    from tui.backends.vllm import backend_storage

    path = storage_root / "config" / "vllm" / "shared.yaml"
    initial = "model: org/model\ngpu-memory-utilization: 0.9\ninitial: true\n"
    external = initial + "external-only: true\n"
    path.write_text(initial)
    config = backend_storage.load_config("shared")
    path.write_text(external)
    config.model = "org/new-model"

    with pytest.raises(ValueError, match="changed since"):
        backend_storage.save_config(config)

    assert path.read_text() == external


@pytest.mark.parametrize("model", ["", "   ", {"org": "model"}])
def test_vllm_config_save_rejects_invalid_model_before_write(
    storage_root: Path,
    model: object,
) -> None:
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config

    config = Config(name="invalid-model")
    config.model = model

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        backend_storage.save_config(config)

    assert not config.path.exists()


@pytest.mark.asyncio
async def test_vllm_tui_clone_resets_source_revision(
    storage_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tui.backends.vllm import backend_storage
    from tui.backends.vllm.backend_common import Config
    from tui.backends.vllm.screens import config as screen_module

    backend_storage.save_config(Config(name="source", model="org/model"))
    monkeypatch.setattr(
        screen_module,
        "CONFIG_DIR",
        storage_root / "config" / "vllm",
    )
    app = App()
    screen = screen_module.ConfigListScreen()

    async with app.run_test(size=(120, 60)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        with patch.object(
            screen,
            "_get_selected_config",
            return_value="source",
        ), patch.object(app, "push_screen") as push_screen:
            screen.action_clone_config()
            callback = push_screen.call_args.kwargs["callback"]
            callback("copy")

    cloned = backend_storage.load_config("copy")
    assert cloned.model == "org/model"


def test_concurrent_profile_create_has_one_explicit_conflict(storage_root: Path) -> None:
    start = threading.Barrier(3)
    outcomes: list[str] = []

    def create(port: int) -> None:
        start.wait()
        try:
            profile_store.create_profile(
                profile_store.StoredProfile(
                    name="same",
                    backend="vllm",
                    port=port,
                    model_id=f"org/model-{port}",
                )
            )
        except ValueError as exc:
            outcomes.append(str(exc))
        else:
            outcomes.append("created")

    threads = [threading.Thread(target=create, args=(port,)) for port in (8100, 8101)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()

    assert outcomes.count("created") == 1
    assert len(outcomes) == 2
    assert any("already exists" in outcome for outcome in outcomes)
    assert len(profile_store.list_profiles("vllm")) == 1


def test_config_flags_rejects_credential_image_before_discovery() -> None:
    from tui.cli import config as cli_config

    with patch.object(cli_config, "run_async", side_effect=AssertionError("discovery called")):
        with pytest.raises(typer.BadParameter, match="credentials") as exc_info:
            cli_config.list_flags(
                backend="vllm",
                image="user:secret@registry.example/org/image:v1",
                profile="",
                json_out=False,
            )

    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_vllm_quick_setup_rejects_recipe_with_copy_selection(
    storage_root: Path,
) -> None:
    from tui.backends.vllm.screens import quick_setup as module

    app = App()
    screen = module.QuickSetupScreen()
    fetch = Mock()
    with patch("tui.backends.vllm.backend.list_config_names", return_value=[]), patch.object(
        screen,
        "_fetch_recipe",
        fetch,
    ):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            copy_select = screen.query_one("#copy-config-select", Select)
            copy_select.set_options([("source", "source")])
            copy_select.value = "source"
            screen.query_one("#model-input", Input).value = "org/model"
            screen.query_one("#fetch-recipe-btn").press()
            await pilot.pause()

    fetch.assert_not_called()
