from __future__ import annotations

import multiprocessing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from textual.app import App
from textual.widgets import Input, Select

from tui.common import config_store, profile_store


def _run_cli_edit(
    name: str,
    assignment: str,
    result: multiprocessing.Queue,
) -> None:
    from tui.cli import config as cli_config

    try:
        cli_config.edit_config(
            name=name,
            backend="vllm",
            set_kv=[assignment],
            unset=[],
            disable=[],
            enable=[],
            model=None,
            gpu_memory_utilization=None,
        )
    except BaseException as exc:
        result.put(repr(exc))
    else:
        result.put("")


def _run_blocked_cli_edit(
    name: str,
    assignment: str,
    loaded: multiprocessing.Event,
    release: multiprocessing.Event,
    result: multiprocessing.Queue,
) -> None:
    from tui.cli import config as cli_config

    real_load = cli_config._backend_load_config

    def blocking_load(backend: str, config_name: str) -> dict:
        data = real_load(backend, config_name)
        loaded.set()
        if not release.wait(5):
            raise RuntimeError("timed out waiting to continue config edit")
        return data

    with patch.object(cli_config, "_backend_load_config", blocking_load):
        _run_cli_edit(name, assignment, result)


def _run_config_rename(
    old: str,
    new: str,
    result: multiprocessing.Queue,
) -> None:
    try:
        config_store.rename_config("vllm", old, new)
    except BaseException as exc:
        result.put(repr(exc))
    else:
        result.put("")


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tui.backends.llamacpp import backend as llama_backend
    from tui.backends.vllm import backend as vllm_backend
    from tui.backends.vllm import backend_common, backend_storage
    from tui.cli import config as cli_config

    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
    runtime_dir = tmp_path / ".runtime"
    config_dirs = {
        "vllm": tmp_path / "config" / "vllm",
        "llamacpp": tmp_path / "config" / "llamacpp",
    }
    for path in config_dirs.values():
        path.mkdir(parents=True)

    monkeypatch.setattr(profile_store, "PROFILES_YAML", profiles_yaml)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(config_store, "config_dir", lambda backend: config_dirs[backend])
    monkeypatch.setattr(cli_config, "_config_dir", lambda backend: config_dirs[backend])
    monkeypatch.setattr(backend_common, "CONFIG_DIR", config_dirs["vllm"])
    monkeypatch.setattr(backend_storage, "CONFIG_DIR", config_dirs["vllm"])
    monkeypatch.setattr(vllm_backend, "CONFIG_DIR", config_dirs["vllm"])
    monkeypatch.setattr(llama_backend, "CONFIG_DIR", config_dirs["llamacpp"])
    return tmp_path


def _join_processes(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        process.join(10)
        assert process.exitcode == 0


def test_cli_edits_do_not_lose_a_concurrent_update(config_root: Path) -> None:
    config_path = config_root / "config" / "vllm" / "shared.yaml"
    config_path.write_text("model: org/model\ngpu-memory-utilization: 0.9\n")
    context = multiprocessing.get_context("fork")
    loaded = context.Event()
    release = context.Event()
    first_result = context.Queue()
    second_result = context.Queue()
    first = context.Process(
        target=_run_blocked_cli_edit,
        args=("shared", "first=1", loaded, release, first_result),
    )
    second = context.Process(
        target=_run_cli_edit,
        args=("shared", "second=2", second_result),
    )

    first.start()
    assert loaded.wait(5)
    second.start()
    second.join(0.5)
    release.set()
    _join_processes([first, second])

    assert first_result.get(timeout=2) == ""
    assert second_result.get(timeout=2) == ""
    saved = yaml.safe_load(config_path.read_text())
    assert saved["first"] == 1
    assert saved["second"] == 2


def test_cli_edit_cannot_resurrect_a_renamed_source(config_root: Path) -> None:
    old_path = config_root / "config" / "vllm" / "old.yaml"
    new_path = config_root / "config" / "vllm" / "new.yaml"
    old_path.write_text("model: org/model\ngpu-memory-utilization: 0.9\n")
    context = multiprocessing.get_context("fork")
    loaded = context.Event()
    release = context.Event()
    edit_result = context.Queue()
    rename_result = context.Queue()
    edit = context.Process(
        target=_run_blocked_cli_edit,
        args=("old", "edited=true", loaded, release, edit_result),
    )
    rename = context.Process(
        target=_run_config_rename,
        args=("old", "new", rename_result),
    )

    edit.start()
    assert loaded.wait(5)
    rename.start()
    rename.join(0.5)
    release.set()
    _join_processes([edit, rename])

    assert edit_result.get(timeout=2) == ""
    assert rename_result.get(timeout=2) == ""
    assert not old_path.exists()
    assert yaml.safe_load(new_path.read_text())["edited"] is True


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
@pytest.mark.asyncio
async def test_tui_edit_rejects_a_stale_config(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    if backend == "vllm":
        from tui.backends.vllm.screens import config as module

        config_path = config_root / "config" / "vllm" / "shared.yaml"
        initial = "model: org/model\ngpu-memory-utilization: 0.9\n"
        external = initial + "external-only: true\n"
        monkeypatch.setattr(module, "CONFIG_DIR", config_path.parent)
        extract_patch = patch.object(
            module, "extract_vllm_params", AsyncMock(return_value=set())
        )
    else:
        from tui.backends.llamacpp.screens import config as module

        config_path = config_root / "config" / "llamacpp" / "shared.yaml"
        initial = "ctx-size: 4096\n"
        external = initial + "external-only: true\n"
        monkeypatch.setattr(module, "CONFIG_DIR", config_path.parent)
        extract_patch = patch.object(
            module, "extract_llama_server_flags", AsyncMock(return_value=set())
        )
    config_path.write_text(initial)

    with extract_patch, patch.object(module, "list_profile_names", return_value=[]):
        app = App()
        screen = module.ConfigFormScreen("shared")
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            config_path.write_text(external)
            screen.query_one("#save-btn").press()
            await pilot.pause()

    assert config_path.read_text() == external


@pytest.mark.asyncio
async def test_vllm_quick_setup_rejects_a_deleted_copy_source(
    config_root: Path,
) -> None:
    from tui.backends.vllm.screens import quick_setup as module

    app = App()
    screen = module.QuickSetupScreen()
    with patch("tui.backends.vllm.backend.list_config_names", return_value=[]):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            copy_select = screen.query_one("#copy-config-select", Select)
            copy_select.set_options([("gone", "gone")])
            copy_select.value = "gone"
            screen.query_one("#model-input", Input).value = "org/model"
            screen.query_one("#name-input", Input).value = "created"
            screen.query_one("#create-btn").press()
            await pilot.pause()

    assert profile_store.load_profile("created", "vllm") is None
    assert not (config_root / "config" / "vllm" / "created.yaml").exists()


@pytest.mark.asyncio
async def test_llamacpp_quick_setup_rejects_a_deleted_copy_source(
    config_root: Path,
) -> None:
    from tui.backends.llamacpp.screens import quick_setup as module

    app = App()
    screen = module.QuickSetupScreen()
    with patch.object(module, "list_config_names", return_value=[]):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            copy_select = screen.query_one("#copy-config-select", Select)
            copy_select.set_options([("gone", "gone")])
            copy_select.value = "gone"
            screen.query_one("#repo-input", Input).value = "org/model-GGUF"
            screen.query_one("#name-input", Input).value = "created"
            screen._listing_failed = True
            manual_file = screen.query_one("#manual-file-input", Input)
            manual_file.disabled = False
            manual_file.value = "model.gguf"
            screen.query_one("#create-btn").press()
            await pilot.pause()

    assert profile_store.load_profile("created", "llamacpp") is None
    assert not (config_root / "config" / "llamacpp" / "created.yaml").exists()
