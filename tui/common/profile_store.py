"""Unified YAML-based profile storage.

Profiles live in a single `profiles.yaml` at the repo root. At launch time,
each profile is rendered into `.runtime/<backend>/<name>.env` for
`docker compose --env-file` consumption.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import fcntl
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml

def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        # A typo'd LLMUX_ROOT resolves fine and then reads as an empty install
        # (no profiles, no configs) instead of a bad setting.
        if not (root / "compose").is_dir():
            raise RuntimeError(
                f"LLMUX_ROOT={env_root} does not look like an llmux checkout "
                f"({root / 'compose'} is missing)."
            )
        return root
    cwd = Path.cwd()
    if (cwd / "profiles.example.yaml").exists() and (cwd / "compose").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

Backend = Literal["vllm", "llamacpp"]

DEFAULTS: dict[str, dict[str, Any]] = {
    "vllm": {
        "port": 8000,
        "gpu_id": "0",
        "tensor_parallel_size": 1,
        "enable_lora": False,
    },
    "llamacpp": {
        "port": 8080,
        "gpu_id": "0",
    },
}

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_GPU_ID_RE = re.compile(r"^[0-9]+(,[0-9]+)*$")

# Reserved env keys are emitted from StoredProfile fields (port, gpu_id, etc.).
# If env_vars overrides one of these, the rendered .env would contain duplicate
# lines whose final value silently overrides what the conflict checker, TUI,
# and CLI all believe is in effect. Refuse those overrides at render time.
_RESERVED_ENV_KEYS_BY_BACKEND: dict[str, frozenset[str]] = {
    "vllm": frozenset(
        {
            "CONTAINER_NAME",
            "VLLM_PORT",
            "GPU_ID",
            "TENSOR_PARALLEL_SIZE",
            "CONFIG_NAME",
            "MODEL_ID",
            "ENABLE_LORA",
            "MAX_LORAS",
            "MAX_LORA_RANK",
            "LORA_MODULES",
            "EXTRA_PIP_PACKAGES",
            "VLLM_IMAGE",
        }
    ),
    "llamacpp": frozenset(
        {
            "CONTAINER_NAME",
            "LLAMA_PORT",
            "GPU_ID",
            "CONFIG_NAME",
            "MODEL_FILE",
            "HF_REPO",
            "HF_FILE",
            "LLAMACPP_IMAGE",
            "LLAMACPP_DEV_TAG",
        }
    ),
}


def effective_defaults(backend: str) -> dict[str, Any]:
    """Backend defaults with the user's `defaults:` block from profiles.yaml
    applied on top.

    The profile *loader* already honors these overrides (`_backend_defaults`),
    so callers resolving a "use the backend default" sentinel must go through
    here too — reading the hardcoded DEFAULTS instead would hand back 8080 when
    the user's profiles.yaml says the llama.cpp default port is 9000.
    """
    return _backend_defaults(_load_yaml(), backend)


def reserved_env_keys(backend: str) -> frozenset[str]:
    """Reserved env keys for a backend. Public so the CLI/TUI can validate
    user-supplied --set foo=bar before persisting (better error message there
    than on the next `up`)."""
    return _RESERVED_ENV_KEYS_BY_BACKEND.get(backend, frozenset())


def parse_env_vars_text(text: str, backend: str) -> dict[str, str]:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise ValueError(f"environment line {line_number} must be KEY=VALUE")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid environment variable name on line {line_number}: {key!r}")
        if key in reserved_env_keys(backend):
            raise ValueError(
                f"environment variable {key!r} is managed by the profile fields"
            )
        reason = env_value_rejection(value)
        if reason:
            raise ValueError(f"environment value for {key!r} contains a {reason}")
        result[key] = value
    return result


def format_env_vars_text(env_vars: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(env_vars.items()))


@dataclass
class StoredProfile:
    """Superset profile record; fields not applicable to a backend stay default."""

    name: str
    backend: str
    container_name: str = ""
    port: int = 0
    gpu_id: str = "0"
    config_name: str = ""
    tensor_parallel_size: int = 1
    model_id: str = ""
    enable_lora: bool = False
    max_loras: int | None = None
    max_lora_rank: int | None = None
    lora_modules: str = ""
    extra_pip_packages: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    model_file: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    # Per-profile docker image override (e.g. "llamacpp-dev:mtp_main"). Empty
    # falls back to the default image declared in compose/<backend>/.
    image_tag: str = ""


def _load_yaml() -> dict:
    if not PROFILES_YAML.exists():
        return {"version": 1, "defaults": DEFAULTS, "profiles": []}
    parsed = yaml.safe_load(PROFILES_YAML.read_text())
    if parsed is None:
        raw: dict = {}
    elif isinstance(parsed, dict):
        raw = parsed
    else:
        raise ValueError(
            f"{PROFILES_YAML} must be a mapping, got {type(parsed).__name__} — "
            "a list or scalar here would silently read as zero profiles."
        )
    raw.setdefault("version", 1)
    raw.setdefault("defaults", DEFAULTS)
    raw.setdefault("profiles", [])
    defaults = raw["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError(f"{PROFILES_YAML}: defaults must be a mapping")
    unknown = set(defaults) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"{PROFILES_YAML}: unknown defaults backend(s): {', '.join(sorted(unknown))}"
        )
    for backend, values in defaults.items():
        if not isinstance(values, dict):
            raise ValueError(f"{PROFILES_YAML}: defaults.{backend} must be a mapping")
    return raw


def _backend_defaults(data: dict, backend: str) -> dict[str, Any]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    defaults = dict(DEFAULTS[backend])
    user_defaults = data.get("defaults", {})
    if backend in user_defaults:
        defaults.update(user_defaults[backend])
    return defaults


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def _write_yaml(data: dict) -> None:
    _atomic_write(PROFILES_YAML, _dump_yaml(data))


def _atomic_write(path: Path, content: str) -> None:
    tmp = _stage_bytes(path, content.encode())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stage_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise


def _replace_files(writes: dict[Path, str], deletes: tuple[Path, ...] = ()) -> None:
    targets = [*writes, *deletes]
    originals = {
        path: (path.read_bytes(), path.stat().st_mode & 0o777)
        if path.exists()
        else None
        for path in targets
    }
    staged: dict[Path, Path] = {}
    try:
        for path, text in writes.items():
            staged[path] = _stage_bytes(path, text.encode())
    except BaseException:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        raise
    mutated: list[Path] = []
    try:
        for path, tmp in staged.items():
            os.replace(tmp, path)
            mutated.append(path)
        for path in deletes:
            if path.exists():
                path.unlink()
                mutated.append(path)
    except OSError as exc:
        rollback_errors: list[str] = []
        for path in reversed(mutated):
            try:
                original = originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    data, mode = original
                    restored = _stage_bytes(path, data)
                    os.chmod(restored, mode)
                    os.replace(restored, path)
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"storage update failed ({exc}); rollback failed: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


@contextmanager
def _storage_lock():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = RUNTIME_DIR / ".profiles.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "rb+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_profile(entry: dict, defaults: dict[str, Any] | None = None) -> StoredProfile:
    backend = entry.get("backend")
    if backend not in ("vllm", "llamacpp"):
        raise ValueError(f"Invalid backend {backend!r} in profile {entry.get('name')}")
    defaults = defaults or DEFAULTS[backend]
    merged: dict[str, Any] = dict(defaults)
    merged.update(entry)
    name = merged.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("profile entry is missing a non-empty name")

    def _as_int(field_name: str, raw: Any) -> int:
        # Re-raise with the profile + field so list_profiles can skip just
        # this entry and say why, instead of a bare ValueError from int().
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"profile {name!r}: invalid {field_name} {raw!r} (must be an integer)"
            ) from None

    raw_env_vars = merged.get("env_vars") or {}
    if not isinstance(raw_env_vars, dict):
        raise ValueError(f"profile {name!r}: env_vars must be a mapping")

    profile = StoredProfile(
        name=name,
        backend=backend,
        container_name=merged.get("container_name", name),
        port=_as_int("port", merged.get("port", defaults["port"])),
        gpu_id=str(merged.get("gpu_id", "0")),
        config_name=merged.get("config_name", name),
        tensor_parallel_size=_as_int(
            "tensor_parallel_size", merged.get("tensor_parallel_size", 1)
        ),
        model_id=merged.get("model_id", ""),
        enable_lora=_parse_bool(merged.get("enable_lora", False)),
        max_loras=merged.get("max_loras"),
        max_lora_rank=merged.get("max_lora_rank"),
        lora_modules=merged.get("lora_modules", ""),
        extra_pip_packages=merged.get("extra_pip_packages", ""),
        env_vars=dict(raw_env_vars),
        model_file=merged.get("model_file", ""),
        hf_repo=merged.get("hf_repo", ""),
        hf_file=merged.get("hf_file", ""),
        image_tag=str(merged.get("image_tag", "") or ""),
    )
    _validate_profile(profile)
    return profile


def _validate_profile(profile: StoredProfile) -> None:
    if profile.backend not in DEFAULTS:
        raise ValueError(f"invalid backend {profile.backend!r}")
    if not isinstance(profile.name, str) or not _NAME_RE.fullmatch(profile.name):
        raise ValueError(
            f"invalid profile name {profile.name!r}: must match {_NAME_RE.pattern}"
        )
    container_name = profile.container_name or profile.name
    if not isinstance(container_name, str) or not _NAME_RE.fullmatch(container_name):
        raise ValueError(
            f"invalid container name {container_name!r}: must match {_NAME_RE.pattern}"
        )
    config_name = profile.config_name or profile.name
    if not isinstance(config_name, str) or not _NAME_RE.fullmatch(config_name):
        raise ValueError(
            f"invalid config name {config_name!r}: must match {_NAME_RE.pattern}"
        )
    effective_port = int(profile.port or DEFAULTS[profile.backend]["port"])
    if not 1024 <= effective_port <= 65535:
        raise ValueError(f"profile {profile.name!r}: port must be in 1024–65535")
    if not _GPU_ID_RE.fullmatch(profile.gpu_id):
        raise ValueError(
            f"profile {profile.name!r}: invalid GPU id {profile.gpu_id!r}"
        )
    if profile.backend == "vllm" and int(profile.tensor_parallel_size) < 1:
        raise ValueError(
            f"profile {profile.name!r}: tensor_parallel_size must be at least 1"
        )
    for field_name in ("max_loras", "max_lora_rank"):
        value = getattr(profile, field_name)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(
                f"profile {profile.name!r}: {field_name} must be a positive integer"
            )
    if not isinstance(profile.env_vars, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in profile.env_vars.items()
    ):
        raise ValueError(f"profile {profile.name!r}: env_vars must map strings to strings")
    if profile.image_tag:
        from tui.common.dev_build import image_tag_error

        error = image_tag_error(profile.image_tag)
        if error:
            raise ValueError(f"profile {profile.name!r}: {error}")


def _validated_profiles(data: dict) -> list[StoredProfile]:
    entries = data.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError(f"{PROFILES_YAML}: profiles must be a list")
    profiles: list[StoredProfile] = []
    names: dict[str, str] = {}
    containers: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{PROFILES_YAML}: profile entry {index} must be a mapping"
            )
        backend = entry.get("backend")
        if backend not in DEFAULTS:
            raise ValueError(
                f"{PROFILES_YAML}: profile entry {index} has invalid backend {backend!r}"
            )
        profile = _to_profile(entry, _backend_defaults(data, backend))
        if profile.name in names:
            raise ValueError(
                f"duplicate profile name {profile.name!r}: already used by {names[profile.name]}"
            )
        container_name = profile.container_name or profile.name
        if container_name in containers:
            raise ValueError(
                f"duplicate container name {container_name!r}: already used by "
                f"{containers[container_name]}"
            )
        names[profile.name] = f"{profile.backend}/{profile.name}"
        containers[container_name] = f"{profile.backend}/{profile.name}"
        profiles.append(profile)
    return profiles


def _profile_to_entry(
    profile: StoredProfile, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or DEFAULTS[profile.backend]
    out: dict[str, Any] = {"name": profile.name, "backend": profile.backend}
    if profile.container_name and profile.container_name != profile.name:
        out["container_name"] = profile.container_name
    if profile.port and profile.port != defaults.get("port"):
        out["port"] = profile.port
    if profile.gpu_id and profile.gpu_id != defaults.get("gpu_id", "0"):
        out["gpu_id"] = profile.gpu_id
    if profile.config_name != (defaults.get("config_name") or profile.name):
        out["config_name"] = profile.config_name

    if profile.backend == "vllm":
        if profile.tensor_parallel_size != defaults.get("tensor_parallel_size", 1):
            out["tensor_parallel_size"] = profile.tensor_parallel_size
        if profile.model_id:
            out["model_id"] = profile.model_id
        if profile.enable_lora != defaults.get("enable_lora", False):
            out["enable_lora"] = profile.enable_lora
        if profile.max_loras is not None:
            out["max_loras"] = profile.max_loras
        if profile.max_lora_rank is not None:
            out["max_lora_rank"] = profile.max_lora_rank
        if profile.lora_modules:
            out["lora_modules"] = profile.lora_modules
        if profile.extra_pip_packages:
            out["extra_pip_packages"] = profile.extra_pip_packages
    else:
        if profile.model_file:
            out["model_file"] = profile.model_file
        if profile.hf_repo:
            out["hf_repo"] = profile.hf_repo
        if profile.hf_file:
            out["hf_file"] = profile.hf_file
    if profile.env_vars:
        out["env_vars"] = dict(profile.env_vars)
    if profile.image_tag:
        out["image_tag"] = profile.image_tag
    return out


def list_profiles(backend: str) -> list[StoredProfile]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    data = _load_yaml()
    return [p for p in _validated_profiles(data) if p.backend == backend]


def find_name_owner(
    name: str, *, exclude: tuple[str, str] | None = None
) -> str | None:
    """Return the backend that already owns a profile named `name`, or None.

    Profile names must be globally unique across BOTH backends: container_name
    defaults to the profile name, so a `vllm/<name>` and a `llamacpp/<name>`
    would share one docker object — and `container_down`'s `docker rm -f <name>`
    fallback matches globally by name, so stopping one backend could tear down
    the other's running container. `exclude` is the (backend, name) of the
    profile currently being edited, so the check doesn't match it against
    itself.
    """
    for p in _validated_profiles(_load_yaml()):
        if exclude is not None and exclude == (p.backend, p.name):
            continue
        if p.name == name:
            return p.backend
    return None


def list_profile_names(backend: str) -> list[str]:
    return sorted(p.name for p in list_profiles(backend))


def load_profile(name: str, backend: str) -> StoredProfile | None:
    for p in list_profiles(backend):
        if p.name == name:
            return p
    return None


def save_profile(profile: StoredProfile) -> None:
    with _storage_lock():
        _save_profile_unlocked(profile)


def clone_profile(src: str, dst: str, backend: str) -> StoredProfile:
    with _storage_lock():
        source = next(
            (profile for profile in list_profiles(backend) if profile.name == src),
            None,
        )
        if source is None:
            raise ValueError(f"profile {src!r} not found in backend {backend!r}")
        clone = replace(
            source,
            name=dst,
            container_name=dst,
            config_name=source.config_name or src,
            env_vars=dict(source.env_vars),
        )
        _save_profile_unlocked(clone)
        return clone


def _save_profile_unlocked(profile: StoredProfile) -> None:
    """Persist a profile atomically across both profiles.yaml and the runtime .env.

    Stages both files as siblings with a `.tmp` suffix, then os.replace each
    into place once both stage writes have succeeded. If the env render or yaml
    dump raises, neither file on disk is mutated — so a permission error on
    .runtime/ no longer leaves profiles.yaml updated while .env stays stale
    (the silent-corruption pattern that fed the next `compose up` an out-of-
    sync configuration).
    """
    _validate_profile(profile)
    data = _load_yaml()
    existing_profiles = _validated_profiles(data)
    effective_container = profile.container_name or profile.name
    for existing in existing_profiles:
        if (existing.backend, existing.name) == (profile.backend, profile.name):
            continue
        if existing.name == profile.name:
            raise ValueError(
                f"profile name {profile.name!r} is already used by "
                f"{existing.backend}/{existing.name}"
            )
        if (existing.container_name or existing.name) == effective_container:
            raise ValueError(
                f"container name {effective_container!r} is already used by "
                f"{existing.backend}/{existing.name}"
            )
    profiles = data.get("profiles", [])
    entry = _profile_to_entry(profile, _backend_defaults(data, profile.backend))
    for idx, existing in enumerate(profiles):
        # A hand-edited scalar entry (e.g. `- foo`) has no .get — skip it in
        # the match loop (list_profiles already warns about it) rather than
        # crashing the write with AttributeError.
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("name") == profile.name
            and existing.get("backend") == profile.backend
        ):
            profiles[idx] = entry
            break
    else:
        profiles.append(entry)
    data["profiles"] = profiles

    env_path = runtime_env_path(profile.name, profile.backend)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = _render_env_lines(profile)
    env_text = "\n".join(env_lines)
    yaml_text = _dump_yaml(data)

    _replace_files({env_path: env_text, PROFILES_YAML: yaml_text})


def rename_profile(old: str, new: str, backend: str) -> StoredProfile:
    with _storage_lock():
        return _rename_profile_unlocked(old, new, backend)


def _rename_profile_unlocked(old: str, new: str, backend: str) -> StoredProfile:
    """Rename a profile in place, returning the renamed record.

    Callers must refuse this while the profile's container is running: the
    container is named after `container_name or name`, so renaming under a live
    container would orphan it from every lookup path. This function stays
    docker-free — the running check belongs to the CLI/TUI.

    `container_name` and `config_name` both fall back to the profile name when
    unset, and `_to_profile` fills that fallback in before callers ever see it,
    so the raw yaml entry is the only place that still distinguishes "unset"
    from "happens to equal the old name". An unset container_name follows the
    new name; an unset config_name is pinned to the old one so the profile
    keeps resolving to the config file it was already using.
    """
    if old == new:
        raise ValueError(f"profile is already named {new!r}")
    profile = load_profile(old, backend)
    if profile is None:
        raise ValueError(f"profile {old!r} not found in backend {backend!r}")
    owner = find_name_owner(new)
    if owner is not None:
        raise ValueError(
            f"profile {new!r} already exists in backend {owner!r}; profile names "
            "must be unique across both backends"
        )

    data = _load_yaml()
    profiles = data.get("profiles", [])
    for idx, existing in enumerate(profiles):
        if not isinstance(existing, dict):
            continue
        if existing.get("name") == old and existing.get("backend") == backend:
            break
    else:
        raise ValueError(f"profile {old!r} not found in profiles.yaml")

    if not existing.get("container_name"):
        profile.container_name = new
    if not existing.get("config_name"):
        profile.config_name = old
    profile.name = new
    profiles[idx] = _profile_to_entry(profile, _backend_defaults(data, backend))
    data["profiles"] = profiles

    new_env = runtime_env_path(new, backend)
    new_env.parent.mkdir(parents=True, exist_ok=True)
    env_text = "\n".join(_render_env_lines(profile))
    yaml_text = _dump_yaml(data)

    old_env = runtime_env_path(old, backend)
    deletes = (old_env,) if old_env != new_env else ()
    _replace_files(
        {new_env: env_text, PROFILES_YAML: yaml_text},
        deletes=deletes,
    )
    return profile


def delete_profile(name: str, backend: str) -> bool:
    with _storage_lock():
        return _delete_profile_unlocked(name, backend)


def _delete_profile_unlocked(name: str, backend: str) -> bool:
    data = _load_yaml()
    _validated_profiles(data)
    profiles = data.get("profiles", [])
    remaining = [
        p for p in profiles
        # A non-dict entry (hand-edited scalar) has no .get — leave it untouched
        # rather than crashing on `.get()`; only mapping entries can match.
        if not (
            isinstance(p, dict)
            and p.get("name") == name
            and p.get("backend") == backend
        )
    ]
    if len(remaining) == len(profiles):
        return False
    data["profiles"] = remaining
    rt = runtime_env_path(name, backend)
    _replace_files(
        {PROFILES_YAML: _dump_yaml(data)},
        deletes=(rt,),
    )
    return True


def runtime_env_path(name: str, backend: str) -> Path:
    if backend not in DEFAULTS:
        raise ValueError(f"invalid backend {backend!r}")
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid profile name {name!r}: must match {_NAME_RE.pattern}")
    return RUNTIME_DIR / backend / f"{name}.env"


def env_value_rejection(value: str) -> str:
    """Why this value can't be rendered into a .env, or "" if it can.

    docker compose parses the rendered file with a dotenv (godotenv-style)
    reader, not a shell. `shlex.quote` escapes a single quote as `'"'"'`,
    which that reader chokes on — the profile saves fine and then `up` dies
    with an opaque parse error. Quotes, newlines and control characters have no
    safe shlex encoding that dotenv also accepts, so refuse them up front
    rather than emitting a file that only breaks later.
    """
    if "'" in value:
        return "single quote (')"
    if '"' in value:
        return 'double quote (")'
    if "\n" in value or "\r" in value:
        return "newline"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return "control character"
    return ""


def _env_line(key: str, value: Any) -> str:
    if not _ENV_KEY_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    text = str(value)
    reason = env_value_rejection(text)
    if reason:
        raise ValueError(
            f"env value for {key!r} contains a {reason}, which docker compose's "
            f".env parser cannot read: {text!r}"
        )
    return f"{key}={shlex.quote(text)}"


def _render_env_lines(profile: StoredProfile) -> list[str]:
    lines: list[str] = [
        "# Auto-rendered from profiles.yaml — do not edit directly.",
        f"# Profile: {profile.name} ({profile.backend})",
        "",
    ]
    if profile.backend == "vllm":
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("VLLM_PORT", profile.port or DEFAULTS["vllm"]["port"]),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("TENSOR_PARALLEL_SIZE", profile.tensor_parallel_size),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
            _env_line("MODEL_ID", profile.model_id),
            _env_line("ENABLE_LORA", "true" if profile.enable_lora else "false"),
        ]
        if profile.max_loras is not None:
            lines.append(_env_line("MAX_LORAS", profile.max_loras))
        if profile.max_lora_rank is not None:
            lines.append(_env_line("MAX_LORA_RANK", profile.max_lora_rank))
        if profile.lora_modules:
            lines.append(_env_line("LORA_MODULES", profile.lora_modules))
        if profile.extra_pip_packages:
            lines.append(_env_line("EXTRA_PIP_PACKAGES", profile.extra_pip_packages))
        reserved = _RESERVED_ENV_KEYS_BY_BACKEND["vllm"]
        for k, v in profile.env_vars.items():
            if k in reserved:
                # Silently dropping would let an attacker (or just a confused
                # caller) shadow GPU_ID/VLLM_PORT through env_vars after
                # passing the conflict check on the StoredProfile fields.
                # Raise so the bad write never lands in profiles.yaml.
                raise ValueError(
                    f"env_vars cannot override reserved profile key {k!r}; "
                    f"set it on the profile itself (gpu_id/port/etc.) instead."
                )
            lines.append(_env_line(k, v))
        if profile.image_tag:
            # Per-profile docker image override; compose consumes it as
            # `image: ${VLLM_IMAGE}` with a default-fallback chain.
            lines.append(_env_line("VLLM_IMAGE", profile.image_tag))
    else:
        lines += [
            _env_line("CONTAINER_NAME", profile.container_name or profile.name),
            _env_line("LLAMA_PORT", profile.port or DEFAULTS["llamacpp"]["port"]),
            _env_line("GPU_ID", profile.gpu_id),
            _env_line("CONFIG_NAME", profile.config_name or profile.name),
        ]
        if profile.model_file:
            lines.append(_env_line("MODEL_FILE", profile.model_file))
        if profile.hf_repo:
            lines.append(_env_line("HF_REPO", profile.hf_repo))
        if profile.hf_file:
            lines.append(_env_line("HF_FILE", profile.hf_file))
        reserved = _RESERVED_ENV_KEYS_BY_BACKEND["llamacpp"]
        for k, v in profile.env_vars.items():
            if k in reserved:
                # Same rationale as the vllm branch: a reserved key set through
                # env_vars would shadow the StoredProfile field the conflict
                # checker/TUI/CLI all report, so refuse the write outright.
                raise ValueError(
                    f"env_vars cannot override reserved profile key {k!r}; "
                    f"set it on the profile itself (gpu_id/port/etc.) instead."
                )
            lines.append(_env_line(k, v))
        if profile.image_tag:
            if profile.image_tag.startswith("llamacpp-dev:"):
                # llama.cpp dev-build images are consumed in
                # compose/llamacpp/docker-compose.dev.yaml as
                # `image: llamacpp-dev:${LLAMACPP_DEV_TAG}`. The "llamacpp-dev:"
                # prefix is stripped so the compose default-fallback chain
                # stays consistent.
                lines.append(
                    _env_line(
                        "LLAMACPP_DEV_TAG", profile.image_tag.split(":", 1)[1]
                    )
                )
            else:
                # Any other reference is a full image used verbatim via
                # `image: ${LLAMACPP_IMAGE}`.
                lines.append(_env_line("LLAMACPP_IMAGE", profile.image_tag))
    lines.append("")
    return lines


def render_env(profile: StoredProfile) -> Path:
    _validate_profile(profile)
    out_path = runtime_env_path(profile.name, profile.backend)
    lines = _render_env_lines(profile)
    _atomic_write(out_path, "\n".join(lines))
    return out_path


def render_all(backend: str | None = None) -> list[Path]:
    backends = [backend] if backend else ["vllm", "llamacpp"]
    out: list[Path] = []
    for b in backends:
        for p in list_profiles(b):
            out.append(render_env(p))
    return out


def _cli() -> int:
    import sys

    argv = sys.argv[1:]
    if len(argv) == 3 and argv[0] == "render":
        backend, name = argv[1], argv[2]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        stored = load_profile(name, backend)
        if stored is None:
            print(f"Profile not found: {backend}/{name}", file=sys.stderr)
            return 1
        path = render_env(stored)
        print(path)
        return 0
    if len(argv) == 2 and argv[0] == "list":
        backend = argv[1]
        if backend not in DEFAULTS:
            print(f"Invalid backend: {backend}", file=sys.stderr)
            return 2
        for name in list_profile_names(backend):
            print(name)
        return 0
    print(
        "Usage:\n"
        "  python -m tui.common.profile_store render <backend> <name>\n"
        "  python -m tui.common.profile_store list <backend>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
