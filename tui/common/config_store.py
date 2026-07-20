"""Cross-backend config-file operations shared by the CLI and the TUI.

Per-backend config YAML lives in `config/<backend>/<name>.yaml`. Reading and
writing a config's *contents* stays with each backend (they serialize different
shapes); only whole-file operations that must also repair `profiles.yaml`
references live here, so CLI and TUI can't drift apart on them.
"""

from __future__ import annotations

import re
from pathlib import Path

from tui.common import profile_store

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def config_dir(backend: str) -> Path:
    if backend == "vllm":
        from tui.backends.vllm.backend_common import CONFIG_DIR

        return CONFIG_DIR
    if backend == "llamacpp":
        from tui.backends.llamacpp.backend import CONFIG_DIR

        return CONFIG_DIR
    raise ValueError(f"unknown backend: {backend!r}")


def config_path(backend: str, name: str) -> Path:
    return config_dir(backend) / f"{name}.yaml"


def referencing_profiles(backend: str, config_name: str) -> list[profile_store.StoredProfile]:
    """Profiles whose config resolves to `config_name`.

    An unset `config_name` resolves to the profile's own name, and
    `profile_store` fills that fallback in on load — so comparing the loaded
    field alone already covers both the explicit and the implicit case.
    """
    return [
        p for p in profile_store.list_profiles(backend)
        if (p.config_name or p.name) == config_name
    ]


def rename_config(backend: str, old: str, new: str) -> list[str]:
    """Rename a config file and repoint every profile that referenced it.

    Returns the names of the profiles whose `config_name` was updated. Raises
    ValueError on any condition that would leave a dangling reference.
    """
    if old == new:
        raise ValueError(f"config is already named {new!r}")
    if not NAME_RE.match(new):
        raise ValueError(
            f"invalid config name {new!r}: must start with [a-z0-9], then "
            "lowercase letters, digits, dashes, or underscores only"
        )
    if "example" in (old, new):
        raise ValueError("'example' is the tracked template config and may not be renamed")

    src = config_path(backend, old)
    dst = config_path(backend, new)
    if not src.exists():
        raise ValueError(f"config not found: {src}")
    if dst.exists():
        raise ValueError(f"config already exists: {dst}")

    referencing = referencing_profiles(backend, old)
    src.rename(dst)
    updated: list[str] = []
    for p in referencing:
        p.config_name = new
        profile_store.save_profile(p)
        updated.append(p.name)
    return updated
