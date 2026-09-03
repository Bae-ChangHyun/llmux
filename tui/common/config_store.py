from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

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
    with profile_store.storage_transaction():
        return [
            p for p in profile_store.list_profiles(backend)
            if profile_store.effective_config_name(p) == config_name
        ]


def rename_config(
    backend: str,
    old: str,
    new: str,
    *,
    replacement_text: str | None = None,
    replacement: Callable[[str], str] | None = None,
    expected_text: str | None = None,
) -> list[str]:
    if old == new:
        raise ValueError(f"config is already named {new!r}")
    if not NAME_RE.match(new):
        raise ValueError(
            f"invalid config name {new!r}: must start with [a-z0-9], then "
            "lowercase letters, digits, dashes, or underscores only"
        )
    if "example" in (old, new):
        raise ValueError("'example' is the tracked template config and may not be renamed")
    if replacement_text is not None and replacement is not None:
        raise ValueError("config rename accepts only one replacement source")

    with profile_store.storage_transaction():
        src = config_path(backend, old)
        dst = config_path(backend, new)
        if not src.exists():
            raise ValueError(f"config not found: {src}")
        if dst.exists():
            raise ValueError(f"config already exists: {dst}")
        source_text = src.read_text()
        if expected_text is not None and source_text != expected_text:
            raise ValueError(
                f"config changed since it was opened: {src}; reload before saving"
            )
        if replacement is not None:
            replacement_text = replacement(source_text)
        move = (src, dst) if replacement_text is None else (src, dst, replacement_text)
        return profile_store.repoint_config_references(
            backend,
            old,
            new,
            moves=(move,),
        )


def delete_config(backend: str, name: str) -> list[str]:
    with profile_store.storage_transaction():
        path = config_path(backend, name)
        if not path.exists():
            raise ValueError(f"config not found: {path}")
        return profile_store.repoint_config_references(
            backend,
            name,
            "",
            deletes=(path,),
        )
