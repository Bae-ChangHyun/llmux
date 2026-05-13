"""Docker image inventory + dev image build (vLLM)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import typer

from tui.cli._runtime import emit_json, emit_table, run_async, stream_async

app = typer.Typer(help="Docker image inventory + dev image build.", no_args_is_help=True)


@app.command("list")
def list_images(
    repo: str = typer.Option(
        "vllm/vllm-openai", "--repo", help="Image repo to list locally."
    ),
    dev: bool = typer.Option(False, "--dev", help="Also list local vllm-dev:* images."),
    remote: bool = typer.Option(
        False, "--remote",
        help="Query DockerHub for the latest stable + nightly tags (vllm/vllm-openai).",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List local (and optionally remote) docker images."""
    from tui.backends.vllm.backend_inspect import (
        get_dev_images,
        get_docker_images,
        get_dockerhub_nightly_date,
        get_dockerhub_release_version,
    )

    rows: list[dict] = []

    async def _collect():
        for img in await get_docker_images(repo=repo):
            rows.append({"source": "local", **asdict(img)})
        if dev:
            for img in await get_dev_images():
                rows.append({"source": "local-dev", **asdict(img)})
        if remote:
            release = await get_dockerhub_release_version()
            nightly = await get_dockerhub_nightly_date()
            rows.append(
                {
                    "source": "remote",
                    "repository": "vllm/vllm-openai",
                    "tag": release,
                    "size": "",
                    "created": "stable",
                }
            )
            rows.append(
                {
                    "source": "remote",
                    "repository": "vllm/vllm-openai",
                    "tag": "nightly",
                    "size": "",
                    "created": nightly,
                }
            )

    run_async(_collect())

    if json_out:
        emit_json(rows)
        return
    emit_table(rows, columns=["source", "repository", "tag", "size", "created"])


@app.command("pull")
def pull_image(
    tag: str = typer.Argument(..., help="Image tag, e.g. 'v0.20.1' or 'nightly'."),
    repo: str = typer.Option("vllm/vllm-openai", "--repo"),
) -> None:
    """`docker pull <repo>:<tag>` — surface raw output."""
    import subprocess

    full = f"{repo}:{tag}"
    rc = subprocess.run(["docker", "pull", full]).returncode
    raise typer.Exit(code=rc)


@app.command("build-dev")
def build_dev(
    branch: str = typer.Option(
        "", "--branch", "-b", help="vLLM source branch (default from .env.common)."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url", help="vLLM source repo URL (default from .env.common)."
    ),
    custom_tag: str = typer.Option(
        "", "--tag", "-t", help="Custom output tag (default = branch name)."
    ),
    official: bool = typer.Option(
        False, "--official",
        help="Build with upstream Dockerfile defaults (skips local GPU-arch detection patches).",
    ),
) -> None:
    """Build the `vllm-dev:<tag>` image from source, streaming docker output."""
    from tui.backends.vllm.backend_runtime import (
        _stream_build_dev_image,
        get_dev_build_defaults,
    )

    default_repo, default_branch = get_dev_build_defaults()
    branch = branch or default_branch
    repo_url = repo_url or default_repo

    rc = stream_async(
        _stream_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            use_official=official,
        )
    )
    raise typer.Exit(code=rc)
