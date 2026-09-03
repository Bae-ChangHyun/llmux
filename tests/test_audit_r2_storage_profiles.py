from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from textual.app import App
from textual.widgets import Input, Switch, TextArea
from typer.testing import CliRunner

from tui.common import profile_store


@pytest.fixture
def profile_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text("version: 1\ndefaults: {}\nprofiles: []\n")
    monkeypatch.setattr(profile_store, "PROFILES_YAML", profiles_yaml)
    monkeypatch.setattr(profile_store, "RUNTIME_DIR", tmp_path / ".runtime")
    return tmp_path


def write_profiles(root: Path, profiles: list[dict]) -> None:
    (root / "profiles.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "defaults": {}, "profiles": profiles},
            sort_keys=False,
        )
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("# comment only\n", "empty"),
        ("version: 99\ndefaults: {}\nprofiles: []\n", "version"),
        ("version: true\ndefaults: {}\nprofiles: []\n", "version"),
        ("version: 1\ndefaults: {}\nprofiles: []\nfuture: true\n", "unknown"),
        ("version: 1\ndefaults: {}\nprofiles: []\n1: invalid\n", "unknown"),
        (
            "version: 1\ndefaults:\n  vllm:\n    porrt: 8000\nprofiles: []\n",
            "porrt",
        ),
        (
            "version: 1\ndefaults:\n  vllm:\n    port: '8000'\nprofiles: []\n",
            "defaults.vllm.port",
        ),
        (
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: bad\n  backend: vllm\n  porrt: 9999\n",
            "porrt",
        ),
        (
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: bad\n  backend: vllm\n  port: true\n",
            "port",
        ),
        (
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: bad\n  backend: vllm\n  enable_lora: 2\n",
            "enable_lora",
        ),
        (
            "version: 1\ndefaults: {}\nprofiles:\n"
            "- name: bad\n  backend: vllm\n  env_vars: []\n",
            "env_vars",
        ),
    ],
)
def test_profile_schema_rejects_unsupported_or_lossy_input(
    profile_root: Path,
    content: str,
    message: str,
) -> None:
    path = profile_root / "profiles.yaml"
    path.write_text(content)
    original = path.read_text()

    with pytest.raises(ValueError, match=message):
        profile_store.list_profiles("vllm")

    assert path.read_text() == original


@pytest.mark.parametrize(
    "key",
    [
        "HF_TOKEN",
        "HF_ENDPOINT",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_CONFIG",
        "DOCKER_DEFAULT_PLATFORM",
        "COMPOSE_FILE",
        "COMPOSE_PROJECT_NAME",
        "COMPOSE_PARALLEL_LIMIT",
    ],
)
def test_profile_env_rejects_credential_and_control_keys(
    profile_root: Path,
    key: str,
) -> None:
    with pytest.raises(ValueError, match=key):
        profile_store.save_profile(
            profile_store.StoredProfile(
                name="unsafe",
                backend="vllm",
                env_vars={key: "sentinel-secret"},
            )
        )


def test_profile_env_allows_container_runtime_keys(profile_root: Path) -> None:
    env_vars = {
        "PATH": "/opt/model/bin:/usr/bin",
        "HTTP_PROXY": "http://proxy.internal:8080",
        "SSL_CERT_FILE": "/etc/ssl/custom.pem",
        "LD_PRELOAD": "/opt/model/libcustom.so",
    }

    profile_store.save_profile(
        profile_store.StoredProfile(
            name="container-env",
            backend="vllm",
            env_vars=env_vars,
        )
    )

    stored = profile_store.load_profile("container-env", "vllm")
    assert stored is not None
    assert stored.env_vars == env_vars


def test_profile_and_runtime_files_are_owner_only_and_errors_are_redacted(
    profile_root: Path,
) -> None:
    profiles_yaml = profile_root / "profiles.yaml"
    profiles_yaml.chmod(0o644)
    profile_store.save_profile(
        profile_store.StoredProfile(
            name="secure",
            backend="vllm",
            env_vars={"APP_SECRET": "sentinel-secret"},
        )
    )

    runtime = profile_root / ".runtime" / "vllm" / "secure.env"
    assert profiles_yaml.stat().st_mode & 0o777 == 0o600
    assert runtime.stat().st_mode & 0o777 == 0o600

    with pytest.raises(ValueError) as exc_info:
        profile_store.save_profile(
            profile_store.StoredProfile(
                name="bad-secret",
                backend="vllm",
                env_vars={"APP_SECRET": "sentinel-secret'"},
            )
        )
    assert "sentinel-secret" not in str(exc_info.value)


def test_replace_files_rolls_back_after_base_exception(
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_profiles(
        profile_root,
        [{"name": "existing", "backend": "vllm", "port": 8100}],
    )
    runtime = profile_root / ".runtime" / "vllm" / "existing.env"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("OLD=1\n")
    original_yaml = (profile_root / "profiles.yaml").read_bytes()
    original_runtime = runtime.read_bytes()
    real_replace = os.replace
    replacements = 0

    def interrupt_second_replace(src: str | Path, dst: str | Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise KeyboardInterrupt("interrupt between replacements")
        real_replace(src, dst)

    monkeypatch.setattr(profile_store.os, "replace", interrupt_second_replace)

    with pytest.raises(KeyboardInterrupt, match="interrupt between replacements"):
        profile_store.save_profile(
            profile_store.StoredProfile(
                name="existing",
                backend="vllm",
                port=8200,
            )
        )

    assert (profile_root / "profiles.yaml").read_bytes() == original_yaml
    assert runtime.read_bytes() == original_runtime


def test_replace_files_rolls_back_when_interrupt_arrives_after_replace(
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_profiles(
        profile_root,
        [{"name": "existing", "backend": "vllm", "port": 8100}],
    )
    runtime = profile_root / ".runtime" / "vllm" / "existing.env"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("OLD=1\n")
    original_yaml = (profile_root / "profiles.yaml").read_bytes()
    original_runtime = runtime.read_bytes()
    real_replace = os.replace
    replacements = 0

    def interrupt_after_first_replace(src: str | Path, dst: str | Path) -> None:
        nonlocal replacements
        replacements += 1
        real_replace(src, dst)
        if replacements == 1:
            raise KeyboardInterrupt("interrupt after replacement")

    monkeypatch.setattr(profile_store.os, "replace", interrupt_after_first_replace)

    with pytest.raises(KeyboardInterrupt, match="interrupt after replacement"):
        profile_store.save_profile(
            profile_store.StoredProfile(
                name="existing",
                backend="vllm",
                port=8200,
            )
        )

    assert (profile_root / "profiles.yaml").read_bytes() == original_yaml
    assert runtime.read_bytes() == original_runtime


def test_public_storage_transaction_and_latest_profile_renderer(
    profile_root: Path,
) -> None:
    write_profiles(
        profile_root,
        [{"name": "latest", "backend": "vllm", "port": 8100}],
    )

    with profile_store.storage_transaction():
        current = profile_store.load_profile("latest", "vllm")
        assert current is not None
        current.port = 8200
        profile_store.save_profile(current)

    path = profile_store.render_env_for_profile("latest", "vllm")
    assert "VLLM_PORT=8200" in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600

    profile_store.delete_profile("latest", "vllm")
    with pytest.raises(ValueError, match="not found"):
        profile_store.render_env_for_profile("latest", "vllm")


def test_stale_profile_snapshot_is_rejected(profile_root: Path) -> None:
    write_profiles(
        profile_root,
        [{"name": "stale", "backend": "vllm", "port": 8100}],
    )
    original = profile_store.load_profile("stale", "vllm")
    assert original is not None
    concurrent = profile_store.load_profile("stale", "vllm")
    assert concurrent is not None
    concurrent.model_id = "org/new"
    profile_store.save_profile(concurrent)
    requested = profile_store.StoredProfile(
        name="stale",
        backend="vllm",
        port=8200,
    )

    with pytest.raises(ValueError, match="changed since"):
        profile_store.replace_profile(
            "stale",
            requested,
            expected=original,
        )

    stored = profile_store.load_profile("stale", "vllm")
    assert stored is not None
    assert stored.model_id == "org/new"
    assert stored.port == 8100


def test_replace_profile_keeps_implicit_container_semantics(profile_root: Path) -> None:
    write_profiles(profile_root, [{"name": "old", "backend": "vllm"}])
    original = profile_store.load_profile("old", "vllm")
    assert original is not None
    requested = profile_store.StoredProfile(
        name="new",
        backend="vllm",
        container_name="old",
    )

    profile_store.replace_profile("old", requested, expected=original)

    data = yaml.safe_load((profile_root / "profiles.yaml").read_text())
    assert "container_name" not in data["profiles"][0]
    runtime = profile_store.render_env_for_profile("new", "vllm")
    assert "CONTAINER_NAME=new" in runtime.read_text()


def test_vllm_adapter_keeps_implicit_container_after_rename_and_later_save(
    profile_root: Path,
) -> None:
    from tui.backends.vllm import backend_storage

    write_profiles(profile_root, [{"name": "old", "backend": "vllm"}])
    profile = backend_storage.load_profile("old")
    profile.name = "new"

    backend_storage.save_profile(profile)
    assert profile.container_name == "new"
    profile.port = "8200"
    backend_storage.save_profile(profile)

    data = yaml.safe_load((profile_root / "profiles.yaml").read_text())
    assert "container_name" not in data["profiles"][0]
    runtime = profile_store.render_env_for_profile("new", "vllm")
    assert "CONTAINER_NAME=new" in runtime.read_text()


def test_llamacpp_adapter_keeps_implicit_container_after_rename_and_later_save(
    profile_root: Path,
) -> None:
    from tui.backends.llamacpp import backend

    write_profiles(profile_root, [{"name": "old", "backend": "llamacpp"}])
    profile = backend.load_profile("old")
    profile.name = "new"

    backend.save_profile(profile)
    assert profile.container_name == "new"
    profile.port = 8200
    backend.save_profile(profile)

    data = yaml.safe_load((profile_root / "profiles.yaml").read_text())
    assert "container_name" not in data["profiles"][0]
    runtime = profile_store.render_env_for_profile("new", "llamacpp")
    assert "CONTAINER_NAME=new" in runtime.read_text()


def test_profile_show_redacts_environment_values(profile_root: Path) -> None:
    from tui.cli.profile import app

    profile_store.save_profile(
        profile_store.StoredProfile(
            name="shown",
            backend="vllm",
            env_vars={"APP_SECRET": "sentinel-secret"},
        )
    )

    result = CliRunner().invoke(app, ["show", "shown", "--backend", "vllm", "--json"])

    assert result.exit_code == 0
    assert "sentinel-secret" not in result.stdout
    assert '"set": true' in result.stdout


def test_cli_environment_parse_error_does_not_echo_value(profile_root: Path) -> None:
    from tui.cli.profile import app

    result = CliRunner().invoke(
        app,
        ["new", "bad-env", "--backend", "vllm", "--set", "sentinel-secret"],
    )

    assert result.exit_code != 0
    assert "sentinel-secret" not in result.output
    assert "KEY=VALUE" in result.output


def test_cli_new_and_edit_support_all_lora_fields(profile_root: Path) -> None:
    from tui.cli.profile import app

    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "new",
            "lora",
            "--backend",
            "vllm",
            "--lora",
            "--max-loras",
            "4",
            "--max-lora-rank",
            "32",
            "--lora-modules",
            "alpha=/app/lora/a,beta=/app/lora/b",
        ],
    )
    assert created.exit_code == 0, created.output
    stored = profile_store.load_profile("lora", "vllm")
    assert stored is not None
    assert stored.max_loras == 4
    assert stored.max_lora_rank == 32
    assert stored.lora_modules == "alpha=/app/lora/a,beta=/app/lora/b"

    edited = runner.invoke(
        app,
        [
            "edit",
            "lora",
            "--backend",
            "vllm",
            "--max-loras",
            "2",
            "--max-lora-rank",
            "16",
            "--lora-modules",
            "gamma=/app/lora/g",
        ],
    )
    assert edited.exit_code == 0, edited.output
    stored = profile_store.load_profile("lora", "vllm")
    assert stored is not None
    assert stored.max_loras == 2
    assert stored.max_lora_rank == 16
    assert stored.lora_modules == "gamma=/app/lora/g"

    cleared = runner.invoke(
        app,
        [
            "edit",
            "lora",
            "--backend",
            "vllm",
            "--max-loras",
            "0",
            "--max-lora-rank",
            "0",
            "--lora-modules",
            "",
        ],
    )
    assert cleared.exit_code == 0, cleared.output
    stored = profile_store.load_profile("lora", "vllm")
    assert stored is not None
    assert stored.max_loras is None
    assert stored.max_lora_rank is None
    assert stored.lora_modules == ""

    rejected = runner.invoke(
        app,
        ["new", "llama", "--backend", "llamacpp", "--max-loras", "1"],
    )
    assert rejected.exit_code != 0
    assert "--max-loras" in rejected.output


@pytest.mark.asyncio
async def test_vllm_form_preserves_explicit_tp_and_edits_lora_fields() -> None:
    from tui.backends.vllm.backend import Profile
    from tui.backends.vllm.screens import profile as module

    saved: list[Profile] = []
    profile = Profile(
        name="v",
        container_name="v",
        port="8123",
        gpu_id="0",
        tensor_parallel="1",
    )
    screen = module.ProfileFormScreen(profile)
    app = App()

    with patch.object(module, "list_config_names", return_value=[]), patch.object(
        module, "save_profile", side_effect=saved.append
    ):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#gpu-input", Input).value = "0,1"
            screen.query_one("#tp-input", Input).value = ""
            screen.query_one("#tp-input", Input).value = "1"
            await pilot.pause()
            screen.query_one("#max-loras-input", Input).value = "4"
            screen.query_one("#max-lora-rank-input", Input).value = "32"
            screen.query_one("#lora-modules-input", Input).value = "a=/app/lora/a"
            screen.query_one("#save-btn").press()
            await pilot.pause()

    assert saved[0].gpu_id == "0,1"
    assert saved[0].tensor_parallel == "1"
    assert saved[0].max_loras == "4"
    assert saved[0].max_lora_rank == "32"
    assert saved[0].lora_modules == "a=/app/lora/a"


@pytest.mark.parametrize(
    ("backend", "module_name", "delete_kw"),
    [
        ("vllm", "tui.backends.vllm.screens.profile", "delete_config"),
        ("llamacpp", "tui.backends.llamacpp.screens.profile", "delete_config_too"),
    ],
)
@pytest.mark.asyncio
async def test_tui_delete_offers_config_cascade(
    backend: str,
    module_name: str,
    delete_kw: str,
) -> None:
    module = __import__(module_name, fromlist=["ProfileDeleteScreen"])
    if backend == "vllm":
        from tui.backends.vllm.backend import Profile

        loaded = Profile(name="remove", container_name="remove", config_name="cfg")
    else:
        from tui.backends.llamacpp.backend import Profile

        loaded = Profile(name="remove", container_name="remove", config_name="cfg")
    calls: list[tuple[tuple, dict]] = []
    app = App()

    with patch.object(module, "load_profile", return_value=loaded), patch.object(
        module,
        "delete_profile",
        side_effect=lambda *args, **kwargs: calls.append((args, kwargs)),
    ), patch(
        "tui.common.docker.running_container_names",
        AsyncMock(return_value=set()),
    ):
        async with app.run_test(size=(100, 40)) as pilot:
            await app.push_screen(module.ProfileDeleteScreen("remove"))
            await pilot.pause()
            app.screen.query_one("#delete-config-switch", Switch).value = True
            app.screen.query_one("#delete-btn").press()
            await pilot.pause()

    assert calls == [(("remove",), {delete_kw: True})]


@pytest.mark.parametrize(
    ("backend", "module_name"),
    [
        ("vllm", "tui.backends.vllm.screens.profile"),
        ("llamacpp", "tui.backends.llamacpp.screens.profile"),
    ],
)
@pytest.mark.asyncio
async def test_tui_environment_editor_masks_existing_values(
    backend: str,
    module_name: str,
) -> None:
    module = __import__(module_name, fromlist=["ProfileFormScreen"])
    if backend == "vllm":
        from tui.backends.vllm.backend import Profile

        loaded = Profile(
            name="masked",
            container_name="masked",
            env_vars={"APP_SECRET": "sentinel-secret"},
        )
    else:
        from tui.backends.llamacpp.backend import Profile

        loaded = Profile(
            name="masked",
            container_name="masked",
            env_vars={"APP_SECRET": "sentinel-secret"},
        )
    app = App()

    with patch.object(module, "list_config_names", return_value=[]):
        async with app.run_test(size=(120, 60)) as pilot:
            await app.push_screen(module.ProfileFormScreen(loaded))
            await pilot.pause()
            text = app.screen.query_one("#env-vars-input", TextArea).text

    assert "APP_SECRET=<redacted>" in text
    assert "sentinel-secret" not in text
