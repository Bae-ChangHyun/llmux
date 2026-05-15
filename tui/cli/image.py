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
            # llama.cpp dev images live under a different prefix.
            from tui.backends.llamacpp.backend_runtime import LLAMACPP_DEV_SPEC
            from tui.common.dev_build import list_local_dev_images

            for img in await list_local_dev_images(LLAMACPP_DEV_SPEC):
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


_PULL_DEFAULTS = {
    # (default repo, default tag). vLLM has no canonical "current" tag — users
    # almost always pin a version — so we don't default a tag there. llama.cpp
    # ships a single official server image tagged `server-cuda`.
    "vllm": ("vllm/vllm-openai", ""),
    "llamacpp": ("ghcr.io/ggml-org/llama.cpp", "server-cuda"),
}


@app.command("pull")
def pull_image(
    tag: Optional[str] = typer.Argument(
        None,
        help="Image tag. vLLM requires an explicit tag (e.g. 'v0.20.1', 'nightly'). "
             "llama.cpp defaults to 'server-cuda'.",
    ),
    backend: str = typer.Option(
        "vllm", "--backend",
        help="Selects per-backend default repo/tag (vllm → vllm/vllm-openai, "
             "llamacpp → ghcr.io/ggml-org/llama.cpp:server-cuda).",
    ),
    repo: str = typer.Option(
        "", "--repo",
        help="Repo override; empty = backend default.",
    ),
) -> None:
    """`docker pull <repo>:<tag>` — backend-aware defaults, raw docker output."""
    if backend not in _PULL_DEFAULTS:
        raise typer.BadParameter(
            f"unknown backend: {backend!r} (choose vllm or llamacpp)",
            param_hint="--backend",
        )
    default_repo, default_tag = _PULL_DEFAULTS[backend]
    repo = repo or default_repo
    tag = tag or default_tag
    if not tag:
        raise typer.BadParameter(
            "vLLM has no default tag — pass a TAG (e.g. 'v0.20.1' or 'nightly').",
            param_hint="TAG",
        )

    import subprocess

    full = f"{repo}:{tag}"
    rc = subprocess.run(["docker", "pull", full]).returncode
    raise typer.Exit(code=rc)


@app.command("build-dev")
def build_dev(
    backend: str = typer.Option(
        "vllm", "--backend", help="Which backend to build for (vllm or llamacpp)."
    ),
    branch: str = typer.Option(
        "", "--branch", "-b", help="Source branch (default from .env.common)."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url", help="Source repo URL (default from .env.common)."
    ),
    custom_tag: str = typer.Option(
        "", "--tag", "-t", help="Custom output tag (default = branch name)."
    ),
    official: bool = typer.Option(
        False, "--official",
        help="vLLM only: build with upstream Dockerfile defaults (skips local GPU-arch detection patches).",
    ),
    cuda_arch: str = typer.Option(
        "", "--cuda-arch",
        help="llamacpp only: override auto-detection. Pass CMake-format like '89' (Ada) or "
             "'86;89' (mixed). Empty = auto-detect via nvidia-smi.",
    ),
    multi_arch: bool = typer.Option(
        False, "--multi-arch",
        help="llamacpp only: disable GPU auto-detection and build for all archs (portable, slow).",
    ),
) -> None:
    """Build a `<backend>-dev:<tag>` image from source, streaming docker output.

    vllm:     vllm-dev:<tag>     (target=vllm-openai, docker/Dockerfile)
    llamacpp: llamacpp-dev:<tag> (target=server, .devops/cuda.Dockerfile)
    """
    if backend not in ("vllm", "llamacpp"):
        typer.echo(f"Error: unknown --backend {backend!r}. Use vllm or llamacpp.", err=True)
        raise typer.Exit(code=2)

    if backend == "vllm":
        from tui.backends.vllm.backend_runtime import (
            _stream_build_dev_image,
            get_dev_build_defaults,
        )

        default_repo, default_branch = get_dev_build_defaults()
        branch = branch or default_branch
        repo_url = repo_url or default_repo
        if cuda_arch or multi_arch:
            typer.echo("Warning: --cuda-arch/--multi-arch are llamacpp-only and ignored for vllm", err=True)
        rc = stream_async(
            _stream_build_dev_image(
                branch,
                repo_url=repo_url,
                custom_tag=custom_tag,
                use_official=official,
            )
        )
    else:
        from tui.backends.llamacpp.backend_runtime import (
            _stream_build_dev_image,
            get_dev_build_defaults,
        )

        default_repo, default_branch = get_dev_build_defaults()
        branch = branch or default_branch
        repo_url = repo_url or default_repo
        if official:
            typer.echo("Warning: --official is vllm-only and ignored for llamacpp", err=True)
        rc = stream_async(
            _stream_build_dev_image(
                branch,
                repo_url=repo_url,
                custom_tag=custom_tag,
                cuda_arch=cuda_arch,
                use_multi_arch=multi_arch,
            )
        )
    raise typer.Exit(code=rc)
