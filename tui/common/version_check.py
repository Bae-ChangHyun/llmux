"""GitHub release checks and self-update."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass

from tui.common.ssl_ctx import open_url

from tui.common.profile_store import PROJECT_ROOT

log = logging.getLogger(__name__)

_CACHE_FILE = PROJECT_ROOT / ".runtime" / "version-check.json"
_FAILURE_COOLDOWN = 15 * 60  # seconds — only a failed lookup backs off
_HTTP_TIMEOUT = 3  # seconds per GitHub API call
_USER_AGENT = "llmux-version-check"

BEHIND = "behind"
CURRENT = "current"
UNKNOWN = "unknown"


@dataclass
class UpdateStatus:
    """Outcome of one update check."""

    state: str
    tag: str = ""
    url: str = ""
    local_version: str = ""
    detail: str = ""


def _git(*args: str, timeout: int = 15) -> tuple[int, str]:
    """Run git and return its code plus combined output."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _is_git_checkout() -> bool:
    rc, out = _git("rev-parse", "--is-inside-work-tree")
    return rc == 0 and out.strip() == "true"


def _repo_slug() -> str | None:
    """Return `owner/repo` parsed from the `origin` remote, or None."""
    rc, url = _git("remote", "get-url", "origin")
    if rc != 0:
        return None
    url = url.strip()
    for prefix in (
        "git@github.com:",
        "https://github.com/",
        "ssh://git@github.com/",
    ):
        if url.startswith(prefix):
            slug = url[len(prefix):]
            if slug.endswith(".git"):
                slug = slug[:-4]
            return slug or None
    return None


def _cooldown_active() -> bool:
    """True while a recent lookup failure is still backing off."""
    try:
        data = json.loads(_CACHE_FILE.read_text())
        failed_at = float(data.get("failed_at", 0))
    except (OSError, ValueError, TypeError):
        return False
    return (time.time() - failed_at) < _FAILURE_COOLDOWN


def _record_failure() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"failed_at": time.time()}))
    except OSError:
        pass


def _clear_failure() -> None:
    try:
        _CACHE_FILE.unlink()
    except OSError:
        pass


def _local_version() -> str:
    """Read the checkout version without requiring Python 3.11 tomllib."""
    try:
        text = (PROJECT_ROOT / "pyproject.toml").read_text()
    except OSError:
        return ""
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else ""


def _api_get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with open_url(req, timeout=_HTTP_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001 — offline / rate-limited / parse error
        log.debug("update check: %s failed (%s)", url, exc)
        return None


def _latest_release(slug: str) -> tuple[str, str] | None:
    """Return (tag_name, html_url) of the latest GitHub Release, or None."""
    data = _api_get(f"https://api.github.com/repos/{slug}/releases/latest")
    if data is None:
        return None
    tag = data.get("tag_name")
    if not tag:
        return None
    return str(tag), str(data.get("html_url", ""))


def _release_commit(slug: str, tag: str) -> str | None:
    """Resolve a release tag to its underlying commit SHA via the commits API."""
    data = _api_get(f"https://api.github.com/repos/{slug}/commits/{tag}")
    if data is None:
        return None
    sha = data.get("sha")
    return str(sha) if sha else None


def _cat_file_proves_missing(returncode: int, output: str) -> bool:
    if returncode == 1 and not output.strip():
        return True
    if returncode != 128:
        return False
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "not a valid object name",
            "could not get object info",
            "bad object",
        )
    )


def _is_behind(release_sha: str) -> bool | None:
    """Return whether HEAD lacks the release commit, or None if undecidable."""
    rc, output = _git("cat-file", "-e", f"{release_sha}^{{commit}}")
    if rc != 0:
        if not _cat_file_proves_missing(rc, output):
            return None
        rc_shallow, out_shallow = _git("rev-parse", "--is-shallow-repository")
        if rc_shallow != 0:
            return None
        shallow = out_shallow.strip()
        if shallow == "true":
            return None
        if shallow != "false":
            return None
        return True
    rc, _ = _git("merge-base", "--is-ancestor", release_sha, "HEAD")
    if rc == 0:
        return False
    if rc == 1:
        return True
    return None


def _local_clean_main() -> bool:
    """Return whether unattended fast-forward is allowed."""
    rc, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or branch.strip() != "main":
        return False
    rc, status = _git("status", "--porcelain", "--untracked-files=no")
    return rc == 0 and status.strip() == ""


def resolve_status(*, respect_cooldown: bool = True) -> UpdateStatus:
    """Resolve this checkout against the latest GitHub release."""
    local = _local_version()
    if not _is_git_checkout():
        return UpdateStatus(
            UNKNOWN, local_version=local,
            detail=f"{PROJECT_ROOT} is not a git checkout — update it the way you installed it",
        )
    if respect_cooldown and _cooldown_active():
        return UpdateStatus(
            UNKNOWN, local_version=local,
            detail="a recent lookup failed; backing off before retrying",
        )

    slug = _repo_slug()
    if slug is None:
        return UpdateStatus(
            UNKNOWN, local_version=local,
            detail="`origin` is not a GitHub remote",
        )

    latest = _latest_release(slug)
    if latest is None:
        _record_failure()
        return UpdateStatus(
            UNKNOWN, local_version=local,
            detail=f"could not reach the GitHub API for {slug} (offline or rate-limited)",
        )
    _clear_failure()
    tag, url = latest

    if local and tag.lstrip("v") == local:
        return UpdateStatus(CURRENT, tag=tag, url=url, local_version=local)

    release_sha = _release_commit(slug, tag)
    if release_sha is None:
        return UpdateStatus(
            UNKNOWN, tag=tag, url=url, local_version=local,
            detail=f"could not resolve {tag} to a commit",
        )

    behind = _is_behind(release_sha)
    if behind is True:
        return UpdateStatus(BEHIND, tag=tag, url=url, local_version=local)
    if behind is False:
        return UpdateStatus(CURRENT, tag=tag, url=url, local_version=local)
    return UpdateStatus(
        UNKNOWN, tag=tag, url=url, local_version=local,
        detail="could not compare the release commit with local Git history",
    )


def check_for_update() -> None:
    """Run the interactive startup update check."""
    try:
        status = resolve_status()
        if status.state != BEHIND:
            if status.detail:
                log.debug("update check: %s", status.detail)
            return
        _prompt_and_update(status.tag, status.url)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — must never break startup
        log.debug("update check failed: %s", exc)


def apply_update(tag: str) -> tuple[bool, str]:
    """`git pull --ff-only` + refresh the installed tool. (ok, message)."""
    rc, out = _git("pull", "--ff-only")
    if rc != 0:
        return False, f"git pull failed — update manually.\n{out.strip()}"

    refresh_cmd = "uv tool install --editable . --force"
    if not shutil.which("uv"):
        return False, (
            f"Checkout updated to {tag}, but `uv` was not found and the installed "
            "tool was not refreshed — "
            f"run `{refresh_cmd}` in {PROJECT_ROOT} manually."
        )
    try:
        proc = subprocess.run(
            ["uv", "tool", "install", "--editable", ".", "--force"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, (
            f"Checkout updated to {tag}, but the installed tool refresh failed: {exc}. "
            f"Run `{refresh_cmd}` in {PROJECT_ROOT} manually."
        )
    if proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()
        suffix = f" ({detail})" if detail else ""
        return False, (
            f"Checkout updated to {tag}, but the installed tool refresh failed{suffix}. "
            f"Run `{refresh_cmd}` in {PROJECT_ROOT} manually."
        )
    return True, f"Updated to {tag}. Restart llmux to use the new version."


def _prompt_and_update(tag: str, url: str) -> None:
    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()
    console.print(
        f"\n[bold cyan]⬆ llmux {tag} is available.[/bold cyan]  [dim]{url}[/dim]"
    )

    if not _local_clean_main():
        console.print(
            "[dim]  Checkout has local changes or is not on `main` — "
            "skipping auto-update; `git pull` it yourself when ready.[/dim]"
        )
        return

    try:
        if not Confirm.ask("  Update now?", default=True, console=console):
            return
    except (KeyboardInterrupt, EOFError):
        return

    ok, message = apply_update(tag)
    if not ok:
        console.print(f"[red]  {message}[/red]")
        return
    console.print(f"[bold green]  ✓ {message}[/bold green]")
    raise SystemExit(0)


def update_blocked_reason() -> str:
    """Why an automatic update is refused right now ("" when it can proceed)."""
    if _local_clean_main():
        return ""
    rc, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    current = branch.strip() if rc == 0 else "?"
    if current != "main":
        return f"checkout is on `{current}`, not `main`"
    return "checkout has uncommitted changes to tracked files"
