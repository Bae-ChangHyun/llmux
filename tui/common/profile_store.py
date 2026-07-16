"""Unified YAML-based profile storage.

Profiles live in a single `profiles.yaml` at the repo root. At launch time,
each profile is rendered into `.runtime/<backend>/<name>.env` for
`docker compose --env-file` consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Literal

import yaml

log = logging.getLogger(__name__)

def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLMUX_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
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
    raw = yaml.safe_load(PROFILES_YAML.read_text()) or {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("version", 1)
    raw.setdefault("defaults", DEFAULTS)
    raw.setdefault("profiles", [])
    return raw


def _backend_defaults(data: dict, backend: str) -> dict[str, Any]:
    if backend not in DEFAULTS:
        raise ValueError(f"Invalid backend: {backend!r}")
    defaults = dict(DEFAULTS[backend])
    user_defaults = data.get("defaults", {})
    if isinstance(user_defaults, dict) and isinstance(
        user_defaults.get(backend), dict
    ):
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
    """Write `content` to `path` via a sibling `.tmp` + os.replace.

    Same-filesystem rename is atomic on POSIX, so a process crash between
    create and replace can leave a stray `.tmp` but never a half-written
    target. Callers that need cross-file atomicity (e.g. save_profile updating
    both profiles.yaml and a .env) should stage every file as `.tmp` first and
    only os.replace once all stages succeed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


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
    name = merged["name"]

    def _as_int(field_name: str, raw: Any) -> int:
        # A hand-edited `port: abc` used to raise a bare ValueError deep inside
        # int(), taking down every read path. Re-raise with the profile + field
        # so list_profiles can skip just this entry and say why.
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"profile {name!r}: invalid {field_name} {raw!r} (must be an integer)"
            ) from None

    return StoredProfile(
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
        env_vars=dict(merged.get("env_vars") or {}),
        model_file=merged.get("model_file", ""),
        hf_repo=merged.get("hf_repo", ""),
        hf_file=merged.get("hf_file", ""),
        image_tag=str(merged.get("image_tag", "") or ""),
    )


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
    data = _load_yaml()
    defaults = _backend_defaults(data, backend)
    out: list[StoredProfile] = []
    seen: set[str] = set()
    for p in data.get("profiles", []):
        # A non-dict list item (e.g. a stray string from a hand edit) has no
        # .get — guard before touching it so it can't crash the whole scan.
        if not isinstance(p, dict):
            log.warning(
                "skipping non-mapping entry in profiles.yaml: %r — each profile "
                "must be a key/value mapping.", p,
            )
            continue
        if p.get("backend") != backend:
            continue
        try:
            profile = _to_profile(p, defaults)
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            # One malformed entry (missing name, non-integer port, wrong types
            # from a hand edit) must not blank out the whole list — skip it, but
            # say so loudly so the user can fix profiles.yaml.
            log.warning(
                "skipping malformed profile in profiles.yaml: %s — fix the value "
                "or remove the entry.", exc,
            )
            continue
        # A second entry with the same name+backend (merge artifact or hand
        # edit) would make the dashboard add two rows under one Textual row key
        # (DuplicateKey → the reload worker crashes). Keep the first and warn.
        if profile.name in seen:
            log.warning(
                "skipping duplicate profile %r in backend %r — keep a single "
                "entry per name; remove the extra in profiles.yaml.",
                profile.name, backend,
            )
            continue
        seen.add(profile.name)
        out.append(profile)
    return out


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
    for backend in DEFAULTS:
        for p in list_profiles(backend):
            if exclude is not None and exclude == (backend, p.name):
                continue
            if p.name == name:
                return backend
    return None


def list_profile_names(backend: str) -> list[str]:
    return sorted(p.name for p in list_profiles(backend))


def load_profile(name: str, backend: str) -> StoredProfile | None:
    for p in list_profiles(backend):
        if p.name == name:
            return p
    return None


def save_profile(profile: StoredProfile) -> None:
    """Persist a profile atomically across both profiles.yaml and the runtime .env.

    Stages both files as siblings with a `.tmp` suffix, then os.replace each
    into place once both stage writes have succeeded. If the env render or yaml
    dump raises, neither file on disk is mutated — so a permission error on
    .runtime/ no longer leaves profiles.yaml updated while .env stays stale
    (the silent-corruption pattern that fed the next `compose up` an out-of-
    sync configuration).
    """
    data = _load_yaml()
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

    env_tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    yaml_tmp = PROFILES_YAML.with_suffix(PROFILES_YAML.suffix + ".tmp")
    try:
        env_tmp.write_text(env_text)
        yaml_tmp.write_text(yaml_text)
    except OSError:
        for stale in (env_tmp, yaml_tmp):
            try:
                stale.unlink()
            except OSError:
                pass
        raise
    os.replace(env_tmp, env_path)
    os.replace(yaml_tmp, PROFILES_YAML)


def delete_profile(name: str, backend: str) -> bool:
    data = _load_yaml()
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
    _write_yaml(data)
    rt = runtime_env_path(name, backend)
    if rt.exists():
        rt.unlink()
    return True


def runtime_env_path(name: str, backend: str) -> Path:
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
            _env_line("VLLM_PORT", profile.port),
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
            _env_line("LLAMA_PORT", profile.port),
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
