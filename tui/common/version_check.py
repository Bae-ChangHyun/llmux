"""Startup check for a newer llmux release.

Compares the local git checkout against the latest GitHub Release. On a clean
`main` checkout it offers an interactive `git pull` update; otherwise it leaves
the checkout alone. Every step is best-effort: a non-git install, a missing
network, or any error leaves startup completely untouched.

"Behind" is decided by commit ancestry, not version strings — the checkout is
behind only when HEAD does not yet contain the latest release's commit. That
stays correct even when local tags are stale (a `git pull`'d checkout often
has the release commit without ever fetching the tag).

A 24h cache keeps this off the hot path: at most one network round-trip per
day. The caller owns the TTY gate — this only runs for interactive sessions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request

from tui.common.ssl_ctx import get_ssl_context

from tui.common.profile_store import PROJECT_ROOT

_CACHE_FILE = PROJECT_ROOT / ".runtime" / "version-check.json"
_CACHE_TTL = 24 * 60 * 60  # seconds — one check per day
_HTTP_TIMEOUT = 4  # seconds per GitHub API call
_USER_AGENT = "llmux-version-check"

# ── git helpers ──────────────────────────────────────────────────────────────
def _git(*args: str, timeout: int = 15) -> tuple[int, str]:
    """Run `git -C PROJECT_ROOT <args>`; return (returncode, stdout+stderr).

    A missing git binary or a timeout is reported as returncode -1.
    """
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


# ── 24h cache ────────────────────────────────────────────────────────────────
def _cache_is_fresh() -> bool:
    try:
        data = json.loads(_CACHE_FILE.read_text())
        return (time.time() - float(data.get("checked_at", 0))) < _CACHE_TTL
    except (OSError, ValueError, TypeError):
        return False


def _write_cache() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"checked_at": time.time()}))
    except OSError:
        pass


def _api_get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(
            req, timeout=_HTTP_TIMEOUT, context=get_ssl_context()
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 — offline / rate-limited / parse error
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


# ── local comparison ─────────────────────────────────────────────────────────
def _is_behind(release_sha: str) -> bool | None:
    """True when HEAD does not yet contain `release_sha`; None if undecidable.

    A checkout is up to date when the release commit is an ancestor of HEAD
    (at it, or ahead of it on a later branch). If the release commit is not in
    local history at all, the checkout is genuinely behind.
    """
    rc, _ = _git("cat-file", "-e", f"{release_sha}^{{commit}}")
    if rc != 0:
        # The release commit is absent from local history. On a shallow clone
        # that is expected for any older commit and proves nothing — stay
        # undecided rather than nag. On a full clone it means the checkout
        # genuinely predates the release.
        rc_shallow, out_shallow = _git("rev-parse", "--is-shallow-repository")
        if rc_shallow == 0 and out_shallow.strip() == "true":
            return None
        return True
    rc, _ = _git("merge-base", "--is-ancestor", release_sha, "HEAD")
    if rc == 0:
        return False
    if rc == 1:
        return True
    return None  # git error → undecidable, don't nag


def _local_clean_main() -> bool:
    """True only on a `main` checkout with no modified tracked files — the one
    state where `git pull --ff-only` is safe to run unattended."""
    rc, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or branch.strip() != "main":
        return False
    rc, status = _git("status", "--porcelain", "--untracked-files=no")
    return rc == 0 and status.strip() == ""


# ── public entry point ───────────────────────────────────────────────────────
def check_for_update() -> None:
    """Best-effort: notify (and on a clean `main`, optionally apply) a newer
    llmux release. Honors the 24h cache, silent offline, never raises — except
    SystemExit, raised deliberately after a successful update so the user
    restarts on fresh code. The caller must only invoke this interactively.
    """
    try:
        _check_impl()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — a version check must never break startup
        pass


def _check_impl() -> None:
    if not _is_git_checkout() or _cache_is_fresh():
        return
    slug = _repo_slug()
    if slug is None:
        return

    latest = _latest_release(slug)
    if latest is None:
        # First API call failed — almost certainly offline or rate-limited.
        # Cache it so we don't retry the network on every command for a day.
        _write_cache()
        return
    tag, url = latest

    release_sha = _release_commit(slug, tag)
    if release_sha is None:
        # The releases API worked but resolving the tag's commit did not — a
        # transient second-call failure. Skip the cache so the next run
        # retries instead of staying silent for 24h.
        return
    _write_cache()
    if _is_behind(release_sha) is not True:  # False or None → don't nag
        return

    _prompt_and_update(tag, url)


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

    rc, out = _git("pull", "--ff-only")
    if rc != 0:
        console.print(
            "[red]  git pull failed — update manually.[/red]\n"
            f"[dim]  {out.strip()}[/dim]"
        )
        return

    # Refresh the installed tool environment, best-effort. `llmux` is installed
    # with `uv tool install --editable` (see install.sh); a bare `uv sync`
    # would only update the project's .venv, not the `uv tool` environment
    # that actually runs the command — leaving stale deps after a pull.
    refresh_cmd = "uv tool install --editable . --force"
    if shutil.which("uv"):
        refreshed = False
        try:
            proc = subprocess.run(
                ["uv", "tool", "install", "--editable", ".", "--force"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            refreshed = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            refreshed = False
        if not refreshed:
            console.print(
                f"[yellow]  ⚠ Could not refresh the install — run "
                f"`{refresh_cmd}` in {PROJECT_ROOT} manually.[/yellow]"
            )
    else:
        console.print(
            f"[yellow]  ⚠ `uv` not found — run `{refresh_cmd}` manually.[/yellow]"
        )

    console.print(
        f"[bold green]  ✓ Updated to {tag}.[/bold green] "
        "Restart llmux to use the new version."
    )
    raise SystemExit(0)
