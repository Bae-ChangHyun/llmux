# CLI Reference

`llmux` ships a single console script (`pyproject.toml` → `[project.scripts]
llmux = "tui.cli:main"`). Running it with **no arguments launches the TUI**;
every subcommand is a headless, scriptable equivalent of a TUI action.

```text
llmux                      # launch the interactive TUI (default)
llmux tui                  # launch the TUI explicitly
llmux <group> <command>    # headless control
llmux <alias>              # top-level shortcut (see below)
```

All commands accept `-h` / `--help`. The CLI is built with
[Typer](https://typer.tiangolo.com/); the option tables below are extracted
directly from the `typer.Option` / `typer.Argument` definitions in
`tui/cli/`.

!!! info "How to run it"
    Examples use `uv run llmux …` (the project's standard `uv` workflow). If
    you installed the tool globally with `uv tool install`, drop the `uv run`
    prefix and call `llmux …` directly.

## Command map

| Group | Commands | Module |
|---|---|---|
| `container` | `up` · `down` · `logs` · `ps` · `render-env` | `tui/cli/container.py` |
| `profile` | `list` · `show` · `new` · `edit` · `delete` · `quick-setup` | `tui/cli/profile.py` |
| `config` | `list` · `show` · `new` · `edit` · `delete` | `tui/cli/config.py` |
| `image` | `list` · `pull` · `build-dev` | `tui/cli/image.py` |
| `system` | `gpu` · `mem-estimate` · `env-check` | `tui/cli/system.py` |
| _(top-level)_ | `tui` · `gpu` · `env-check` · `up` · `down` · `logs` · `ps` · `render-env` | `tui/cli/__init__.py` |

## Common concepts

### Backend detection

Most commands take a **profile name** and resolve its backend (`vllm` or
`llamacpp`) automatically by scanning `profiles.yaml`:

- If the name exists in exactly one backend, that backend is used.
- If it exists in **both**, you must disambiguate with `--backend` / `-b`.
- If it exists in **neither**, the command fails with a clear usage error.

`--backend` always wins when supplied.

### JSON output

Commands that produce structured data accept `--json` to emit a JSON payload
(UTF-8, 2-space indent) instead of the default padded table or text. Commands
with `--json` support:

| Command | Table/text default | `--json` |
|---|---|---|
| `container ps` | status table | list of rows |
| `profile list` | profile table | list of rows |
| `profile show` | `key: value` text | object |
| `config list` | config table | list of rows |
| `config show` | YAML dump | object |
| `image list` | image table | list of rows |
| `system gpu` | GPU table | list of rows |
| `system env-check` | report text | findings object |

### Exit codes

Lifecycle commands propagate the underlying `docker` / `docker compose` exit
code. `Ctrl-C` while streaming (`up`, `logs`, `build-dev`) exits with `130`.
`system env-check` exits `1` when `.env.common` is missing or has issues.

---

## `container` — container lifecycle

```text
llmux container up|down|logs|ps|render-env …
```

Start, stop, inspect, and re-render profile containers. `up`, `down`, `logs`,
`ps`, and `render-env` are also exposed as [top-level aliases](#top-level-aliases).

### `container up`

Start a profile's container. Streams `docker compose` output to stdout.

```text
llmux container up PROFILE [OPTIONS]
```

| Argument | Description |
|---|---|
| `PROFILE` | Profile name (from `profiles.yaml`). Required. |

| Option | Default | Backend | Description |
|---|---|---|---|
| `--backend`, `-b` | _auto-detect_ | both | Force backend (`vllm`, `llamacpp`); auto-detect if omitted. |
| `--tag`, `-t` | `""` | both | Image tag override. vLLM: `vllm/vllm-openai:<tag>` or `vllm-dev:<tag>` with `--dev`. Empty = use highest local versioned tag. |
| `--dev` | `False` | vLLM | Use the locally-built `vllm-dev:<tag>` image. |
| `--pull` | `False` | vLLM | Force `--pull always` when bringing the container up. (Ignored for llama.cpp with a warning.) |
| `--repo-url` | `""` | vLLM | vLLM dev image: override default vLLM source repo URL. |
| `--branch` | `""` | vLLM | vLLM dev image: override default vLLM source branch. |

```bash
# Start a vLLM profile (official image, highest local versioned tag)
uv run llmux container up qwen3-0-6b

# Start it from a locally-built dev image
uv run llmux container up qwen3-0-6b --dev --tag main-20260515

# Start a llama.cpp profile, disambiguating an ambiguous name
uv run llmux up gemma-3-4b --backend llamacpp
```

!!! note "llama.cpp dev images"
    llama.cpp does not use `--dev`/`--tag` the same way vLLM does. Pin a
    llama.cpp profile to a dev image by setting `image_tag:
    llamacpp-dev:<branch>` in `profiles.yaml` (see
    [`profile edit --image-tag`](#profile-edit)); `container up` then layers
    `docker-compose.dev.yaml` automatically.

### `container down`

Stop a profile's container.

```text
llmux container down PROFILE [OPTIONS]
```

| Argument | Description |
|---|---|
| `PROFILE` | Profile name (from `profiles.yaml`). Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-detect_ | Force backend; auto-detect if omitted. |

```bash
uv run llmux container down qwen3-0-6b
```

### `container logs`

Stream container logs (`Ctrl-C` to stop following).

```text
llmux container logs PROFILE [OPTIONS]
```

| Argument | Description |
|---|---|
| `PROFILE` | Profile name (from `profiles.yaml`). Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-detect_ | Force backend; auto-detect if omitted. |
| `--tail`, `-n` | `200` | Number of recent lines to show before following. |
| `--follow / --no-follow`, `-f / -F` | `--follow` | Follow log output (default). Use `--no-follow` to print recent lines and exit. |

```bash
# Follow live logs
uv run llmux logs qwen3-0-6b

# Print the last 50 lines and exit
uv run llmux container logs qwen3-0-6b -n 50 --no-follow
```

### `container ps`

List profiles and their container status across backends.

```text
llmux container ps [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _all_ | Limit to one backend (`vllm`, `llamacpp`); show all if omitted. |
| `--json` | `False` | Emit JSON instead of a table. |
| `--running`, `-r` | `False` | Only show running containers. |

Table columns: `backend`, `profile`, `status`, `port`, `gpu`, `container`,
`model`. JSON rows additionally include a boolean `running` field.

```bash
uv run llmux ps
uv run llmux ps --running --json
uv run llmux container ps -b vllm
```

### `container render-env`

Re-render `.runtime/<backend>/<profile>.env` from `profiles.yaml`. Useful
after editing `profiles.yaml` by hand. Prints the rendered path(s).

```text
llmux container render-env [PROFILE] [OPTIONS]
```

| Argument | Description |
|---|---|
| `PROFILE` | Profile name to render. **Omit to re-render all profiles.** |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _all_ | Limit when rendering all (no `PROFILE` given). |

```bash
# Re-render a single profile
uv run llmux render-env qwen3-0-6b

# Re-render every llama.cpp profile
uv run llmux container render-env --backend llamacpp
```

---

## `profile` — profile CRUD + quick-setup

```text
llmux profile list|show|new|edit|delete|quick-setup …
```

Profiles live in `profiles.yaml` at the repo root — the single source of truth
for both backends. Fields not applicable to a backend stay at their defaults.

### `profile list`

List all profiles across (or within) backends.

```text
llmux profile list [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _all_ | Limit to one backend (`vllm`, `llamacpp`). |
| `--json` | `False` | Emit JSON instead of a table. |

Columns: `backend`, `name`, `port`, `gpu_id`, `config`, `model`.

```bash
uv run llmux profile list
uv run llmux profile list -b vllm --json
```

### `profile show`

Show a profile's full record.

```text
llmux profile show NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Profile name. Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-detect_ | Force backend; auto-detect if omitted. |
| `--json` | `False` | Emit JSON instead of YAML-ish `key: value` text. |

```bash
uv run llmux profile show qwen3-0-6b --json
```

### `profile new`

Create a new profile entry in `profiles.yaml`. The name must start with an
alphanumeric and contain only `[A-Za-z0-9_-]`; creation fails if the name
already exists in the chosen backend.

```text
llmux profile new NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Profile name (alphanumeric, dash, underscore). Required. |

| Option | Default | Backend | Description |
|---|---|---|---|
| `--backend`, `-b` | `vllm` | both | Backend (`vllm`, `llamacpp`). |
| `--port`, `-p` | `0` | both | Host port (`0` = backend default: 8000 vLLM / 8080 llama.cpp). |
| `--gpu-id`, `-g` | `"0"` | both | GPU id(s), comma-separated. (vLLM: `tensor_parallel_size` is derived from the count.) |
| `--model`, `-m` | `""` | vLLM | Hugging Face model id. |
| `--config`, `-c` | _profile name_ | both | Linked config name. |
| `--container` | _profile name_ | both | Container name. |
| `--lora / --no-lora` | `--no-lora` | vLLM | Enable LoRA. |
| `--extra-pip` | `""` | vLLM | Extra pip packages installed before serve. |
| `--set` | _(none)_ | vLLM | Repeatable: `KEY=VALUE` entries appended to `env_vars`. Key must be a valid env-var name. |
| `--model-file` | `""` | llama.cpp | GGUF filename. |
| `--hf-repo` | `""` | llama.cpp | Hugging Face repo for download. |
| `--hf-file` | `""` | llama.cpp | HF file for download. |

```bash
# vLLM profile
uv run llmux profile new qwen3-8b --backend vllm --model Qwen/Qwen3-8B \
    --gpu-id 0,1 --port 8000 --set VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

# llama.cpp profile
uv run llmux profile new gemma-3-4b --backend llamacpp --port 8080 \
    --model-file gemma-3-4b-it-q4_k_m.gguf \
    --hf-repo unsloth/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-q4_k_m.gguf
```

### `profile edit`

Edit fields of an existing profile. **Only the options you pass change**;
everything else is left untouched.

```text
llmux profile edit NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Profile name. Required. |

| Option | Backend | Description |
|---|---|---|
| `--backend`, `-b` | both | Force backend; auto-detect if omitted. |
| `--port`, `-p` | both | New host port. |
| `--gpu-id`, `-g` | both | New GPU id(s); also re-derives `tensor_parallel_size`. |
| `--model`, `-m` | vLLM | New Hugging Face model id. |
| `--config`, `-c` | both | New linked config name. |
| `--container` | both | New container name. |
| `--lora / --no-lora` | vLLM | Toggle LoRA. |
| `--extra-pip` | vLLM | Extra pip packages. |
| `--set` | vLLM | Repeatable: `KEY=VALUE` to add/override in `env_vars`. |
| `--unset` | vLLM | Repeatable: `KEY` to remove from `env_vars`. |
| `--model-file` | llama.cpp | New GGUF filename. |
| `--hf-repo` | llama.cpp | New HF download repo. |
| `--hf-file` | llama.cpp | New HF download file. |
| `--image-tag` | both | Docker image override (e.g. `llamacpp-dev:mtp_main`). Pass an empty string to clear. |

```bash
# Pin a llama.cpp profile to a locally-built dev image
uv run llmux profile edit gemma-3-4b --image-tag llamacpp-dev:mtp_main

# Add an env var, remove another
uv run llmux profile edit qwen3-8b --set VLLM_ATTENTION_BACKEND=FLASHINFER --unset OLD_VAR
```

### `profile delete`

Delete a profile (and optionally its linked config YAML).

```text
llmux profile delete NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Profile name. Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-detect_ | Force backend; auto-detect if omitted. |
| `--with-config` | `False` | Also delete the linked config YAML. |
| `--yes`, `-y` | `False` | Skip the interactive confirmation prompt. |

```bash
uv run llmux profile delete old-profile --with-config --yes
```

### `profile quick-setup`

Create a profile **and** a config from a model id in one step — mirrors the
TUI's "Quick Setup". If `--name` is omitted, a name is derived from the model
id (lowercased, non-alphanumerics replaced with `-`); collisions get a numeric
suffix.

!!! warning "vLLM only"
    `quick-setup` currently supports `--backend vllm` only. For llama.cpp use
    `profile new --backend llamacpp` directly (it needs `hf_repo` / `hf_file`).

```text
llmux profile quick-setup MODEL [OPTIONS]
```

| Argument | Description |
|---|---|
| `MODEL` | HF model id, e.g. `Qwen/Qwen3-8B`. Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | `vllm` | Backend (vLLM only is currently accepted). |
| `--name`, `-n` | _derived from model_ | Profile name. |
| `--port`, `-p` | `8000` | Host port. |
| `--gpu-id`, `-g` | `"0"` | GPU id(s), comma-separated. |
| `--gpu-mem` | `"0.9"` | vLLM `gpu-memory-utilization` (0.0–1.0). |
| `--lora / --no-lora` | `--no-lora` | Enable LoRA. |
| `--copy-from` | `""` | Copy `extra_params` from an existing config name. |

```bash
uv run llmux profile quick-setup Qwen/Qwen3-8B --gpu-id 0,1 --gpu-mem 0.92
```

---

## `config` — config (YAML) CRUD

```text
llmux config list|show|new|edit|delete …
```

Per-model engine configs live in `config/<backend>/<name>.yaml`:

- **vLLM** — `config/vllm/<name>.yaml`: `model:` + `gpu-memory-utilization:`
  plus arbitrary `vllm serve` flags.
- **llama.cpp** — `config/llamacpp/<name>.yaml`: a flat dict of `llama-server`
  flags.

Config names are validated against `^[a-zA-Z0-9][a-zA-Z0-9._-]*$` to prevent
path traversal. For `show` / `edit` / `delete`, if the same name exists in
both backends you must disambiguate with `--backend`.

### `config list`

List config files.

```text
llmux config list [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _all_ | Limit to one backend. |
| `--json` | `False` | Emit JSON instead of a table. |

Columns: `backend`, `name`, `model`, `params` (number of keys in the YAML).

### `config show`

Print a config YAML.

```text
llmux config show NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Config name. Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-resolve_ | Force backend (resolved from existing files if omitted). |
| `--json` | `False` | Emit JSON instead of a YAML dump. |

### `config new`

Create a new config YAML. For vLLM, the file is seeded with `model:` and
`gpu-memory-utilization:`; for llama.cpp it starts empty. `--set` values are
YAML-parsed, so `--set foo=2` stores an int and a bare `--set bar` stores
`true`.

```text
llmux config new NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Config name (becomes `<name>.yaml`). Required. |

| Option | Default | Backend | Description |
|---|---|---|---|
| `--backend`, `-b` | `vllm` | both | Backend. |
| `--model`, `-m` | `""` | vLLM | `model:` field. |
| `--gpu-mem` | `"0.9"` | vLLM | `gpu-memory-utilization:`. |
| `--set` | _(none)_ | both | Repeatable: `KEY=VALUE` entries (YAML-typed values). |
| `--overwrite` | `False` | both | Overwrite if the file already exists. |

```bash
# vLLM config
uv run llmux config new qwen3-8b --model Qwen/Qwen3-8B --gpu-mem 0.9 \
    --set max-model-len=8192 --set enable-prefix-caching

# llama.cpp config
uv run llmux config new gemma-3-4b --backend llamacpp \
    --set ctx-size=8192 --set n-gpu-layers=999 --set flash-attn
```

### `config edit`

Patch fields in an existing config.

```text
llmux config edit NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Config name. Required. |

| Option | Backend | Description |
|---|---|---|
| `--backend`, `-b` | both | Force backend; auto-resolved otherwise. |
| `--set` | both | Repeatable: `KEY=VALUE` (YAML-typed values). |
| `--unset` | both | Repeatable: `KEY` to remove. |
| `--model`, `-m` | vLLM | Set the `model:` field. |
| `--gpu-mem` | vLLM | Set `gpu-memory-utilization:`. |

```bash
uv run llmux config edit qwen3-8b --set max-model-len=16384 --unset enable-prefix-caching
```

### `config delete`

Delete a config YAML.

```text
llmux config delete NAME [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME` | Config name. Required. |

| Option | Default | Description |
|---|---|---|
| `--backend`, `-b` | _auto-resolve_ | Force backend; auto-resolved otherwise. |
| `--yes`, `-y` | `False` | Skip the interactive confirmation prompt. |

---

## `image` — Docker image inventory + dev build

```text
llmux image list|pull|build-dev …
```

### `image list`

List local (and optionally remote) Docker images.

```text
llmux image list [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--repo` | `vllm/vllm-openai` | Image repo to list locally. |
| `--dev` | `False` | Also list local `vllm-dev:*` **and** `llamacpp-dev:*` images. |
| `--remote` | `False` | Query DockerHub for the latest stable + nightly tags of `vllm/vllm-openai`. |
| `--json` | `False` | Emit JSON instead of a table. |

Columns: `source` (`local` / `local-dev` / `remote`), `repository`, `tag`,
`size`, `created`.

```bash
uv run llmux image list --dev
uv run llmux image list --remote --json
```

### `image pull`

`docker pull <repo>:<tag>` — surfaces raw `docker` output and exit code.

```text
llmux image pull TAG [OPTIONS]
```

| Argument | Description |
|---|---|
| `TAG` | Image tag, e.g. `v0.20.1` or `nightly`. Required. |

| Option | Default | Description |
|---|---|---|
| `--repo` | `vllm/vllm-openai` | Image repo to pull from. |

```bash
uv run llmux image pull v0.20.1
uv run llmux image pull nightly
```

### `image build-dev`

Build a `<backend>-dev:<tag>` image from source, streaming `docker build`
output. The build pipeline (git clone/update → `docker build` → image labels)
is shared; the tags applied are `<prefix>:<custom_tag or branch-YYYYMMDD>` and
`<prefix>:<branch>` (a stable alias to the latest build).

- **vLLM** → `vllm-dev:<tag>` (target `vllm-openai`, `docker/Dockerfile`)
- **llama.cpp** → `llamacpp-dev:<tag>` (target `server`, `.devops/cuda.Dockerfile`)

```text
llmux image build-dev [OPTIONS]
```

| Option | Default | Backend | Description |
|---|---|---|---|
| `--backend` | `vllm` | both | Which backend to build for (`vllm` or `llamacpp`). |
| `--branch`, `-b` | _from `.env.common`_ | both | Source branch. Falls back to `VLLM_BRANCH` / `LLAMACPP_BRANCH` (default `main` / `master`). |
| `--repo-url` | _from `.env.common`_ | both | Source repo URL. Falls back to `VLLM_REPO_URL` / `LLAMACPP_REPO_URL`. |
| `--tag`, `-t` | _branch name_ | both | Custom output tag. |
| `--official` | `False` | vLLM | Build with upstream Dockerfile defaults (skips local GPU-arch detection patches). Ignored for llama.cpp with a warning. |
| `--cuda-arch` | `""` | llama.cpp | Override auto-detection. CMake-format, e.g. `89` (Ada) or `86;89` (mixed). Empty = auto-detect via `nvidia-smi`. Ignored for vLLM with a warning. |
| `--multi-arch` | `False` | llama.cpp | Disable GPU auto-detection and build for all archs (portable, slow). Ignored for vLLM with a warning. |

```bash
# vLLM dev image from an unmerged PR branch
uv run llmux image build-dev --backend vllm --branch my-feature \
    --repo-url https://github.com/me/vllm.git

# llama.cpp dev image, pinned to a single GPU arch for a fast build
uv run llmux image build-dev --backend llamacpp --branch master --cuda-arch 89
```

See [Dev Builds](../guide/dev-build.md) for the full workflow, and
[Architecture](architecture.md#dev-build-pipeline) for how the pipeline works.

---

## `system` — system info & env validation

```text
llmux system gpu|mem-estimate|env-check …
```

### `system gpu`

Print an `nvidia-smi` summary, one row per GPU. Also available as the
top-level alias `llmux gpu`.

```text
llmux system gpu [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--json` | `False` | Emit JSON instead of a table. |

Columns: `index`, `name`, `memory_used`, `memory_total`, `utilization`,
`temperature`. Prints `(no GPUs detected — is nvidia-smi installed?)` if none
are found.

```bash
uv run llmux gpu
uv run llmux system gpu --json
```

### `system mem-estimate`

Estimate a Hugging Face model's VRAM footprint via
[`hf-mem`](https://pypi.org/project/hf-mem/) — the same engine the TUI's
memory estimator uses. Reads `HF_TOKEN` from `.env.common` (or the environment)
for gated models.

```text
llmux system mem-estimate MODEL_ID
```

| Argument | Description |
|---|---|
| `MODEL_ID` | Hugging Face model id, e.g. `Qwen/Qwen3-8B`. Required. |

```bash
uv run llmux system mem-estimate Qwen/Qwen3-8B
# → ~17.2GB (model: 15.3GB + KV: 1.9GB)
```

### `system env-check`

Validate `.env.common` and report key paths. Checks that the file exists, that
`HF_TOKEN` / `HF_CACHE_PATH` / `VLLM_VERSION` are set, and that
`HF_CACHE_PATH` is absolute. Also available as the top-level alias
`llmux env-check`.

```text
llmux system env-check [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--json` | `False` | Emit JSON findings instead of a text report. |

Exits `0` when everything checks out, `1` when `.env.common` is missing or any
issue is found. `HF_TOKEN` is masked in the text report (shown as `<set, N
chars>`).

```bash
uv run llmux env-check
uv run llmux system env-check --json
```

See [`.env.common` Reference](env-common.md) for every variable this command
validates.

---

## Top-level aliases

For convenience — and so agents can write `llmux up <profile>` directly —
the most-used commands are mirrored at the top level:

| Alias | Equivalent to |
|---|---|
| `llmux tui` | launch the TUI explicitly |
| `llmux gpu` | `llmux system gpu` |
| `llmux env-check` | `llmux system env-check` |
| `llmux up` | `llmux container up` |
| `llmux down` | `llmux container down` |
| `llmux logs` | `llmux container logs` |
| `llmux ps` | `llmux container ps` |
| `llmux render-env` | `llmux container render-env` |

These aliases share the exact arguments, options, and defaults of their
canonical commands.
