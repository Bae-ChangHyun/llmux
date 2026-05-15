# Quick Start

This guide takes you from a fresh install to a running model in about five
minutes — once with **vLLM** (Hugging Face Transformers) and once with
**llama.cpp** (GGUF). Each step shows both the **TUI** path and the headless
**CLI** path, so pick whichever fits your workflow.

!!! note "Before you start"
    Make sure you've finished [Installation](installation.md): `llmux` is on
    your `PATH`, and `llmux env-check` reports `Status: OK`. All commands below
    run from inside the repo, or anywhere if you've set `LLMUX_ROOT`.

## Path A — Serve a model with vLLM

vLLM serves Hugging Face Transformers models directly from a model id. We'll
use `Qwen/Qwen3-0.6B` because it's small and downloads fast.

### 1. Create a profile

`quick-setup` creates both a profile **and** a matching config from a single
model id — the same flow as the TUI's **Quick Setup**.

=== "TUI"

    ```bash
    llmux
    ```

    On the dashboard, press ++n++ to open **Quick Setup**, pick the **vLLM**
    backend, enter `Qwen/Qwen3-0.6B`, and confirm. llmux generates the profile
    and config, with a live memory estimate to check it fits your GPU.

=== "CLI"

    ```bash
    llmux profile quick-setup Qwen/Qwen3-0.6B --gpu-id 0
    ```

    This prints `Created profile + config: qwen3-0-6b` — the profile name is
    auto-derived from the model id. Confirm it landed:

    ```bash
    llmux profile list
    ```

### 2. Start the container

=== "TUI"

    Select the `qwen3-0-6b` row and press ++u++ (or ++enter++ → **Start**).
    A version picker appears for the vLLM image — pick **Local Latest** or an
    explicit release. llmux runs its cross-backend port/GPU conflict check,
    then streams the Compose output live.

=== "CLI"

    ```bash
    llmux up qwen3-0-6b
    ```

    `llmux up` auto-detects the backend from the profile name and streams
    `docker compose` output to your terminal. Add `--tag v0.20.1 --pull` to
    pin and pull a specific vLLM image version.

!!! note "vLLM never starts from `:latest`"
    llmux refuses `vllm/vllm-openai:latest` because its contents drift while
    the tag name doesn't. The picker resolves stable choices to a semver tag;
    `nightly` stays available as an explicit rolling option.

### 3. Check health

vLLM exposes an OpenAI-compatible server. Once the container is up, hit its
`/health` endpoint on the profile's port (default `8000`):

```bash
curl http://localhost:8000/health
```

An empty `200 OK` response means the server is ready. You can confirm status
across both backends at any time:

=== "TUI"

    The dashboard row shows live status (`starting` → `healthy`), port, GPU,
    and model. Press ++r++ to refresh.

=== "CLI"

    ```bash
    llmux ps                    # table of every profile + container status
    llmux ps --json --running   # machine-readable, running containers only
    ```

### 4. Watch the logs

=== "TUI"

    Select the profile and press ++l++ to open the log stream. Press ++f++ to
    toggle auto-follow.

=== "CLI"

    ```bash
    llmux logs qwen3-0-6b              # follow live (Ctrl-C to stop)
    llmux logs qwen3-0-6b --no-follow --tail 100   # snapshot of recent lines
    ```

### 5. Stop the container

=== "TUI"

    Select the profile and press ++d++ (or ++enter++ → **Stop**).

=== "CLI"

    ```bash
    llmux down qwen3-0-6b
    ```

## Path B — Serve a model with llama.cpp

llama.cpp serves GGUF files. A llama.cpp profile needs GGUF metadata
(`model_file`, `hf_repo`, `hf_file`) — llmux downloads the file automatically
on first start, so there's no manual `hf download`.

### 1. Create a profile

`quick-setup` currently supports vLLM only, so create the llama.cpp profile
directly with `profile new` (CLI) or **Quick Setup** (TUI).

=== "TUI"

    Press ++n++ on the dashboard, pick the **llama.cpp** backend, and provide
    the GGUF repo and file. llmux generates the profile and config.

=== "CLI"

    ```bash
    llmux profile new gemma-3-4b \
      --backend llamacpp \
      --port 8080 \
      --gpu-id 0 \
      --model-file gemma-3-4b-it-q4_k_m.gguf \
      --hf-repo unsloth/gemma-3-4b-it-GGUF \
      --hf-file gemma-3-4b-it-q4_k_m.gguf
    ```

    This prints `Created profile 'gemma-3-4b' (backend=llamacpp)`.

### 2. Start the container

=== "TUI"

    Select the `gemma-3-4b` row and press ++u++. On first start llmux
    auto-downloads the GGUF file before bringing the container up — you'll see
    the download progress in the stream.

=== "CLI"

    ```bash
    llmux up gemma-3-4b
    ```

    The GGUF is fetched into `MODEL_DIR` (from `.env.common`) on first start,
    then `docker compose` brings up the `llama.cpp` server.

!!! tip "GGUF download stuck?"
    Check `HF_TOKEN` in `.env.common` and confirm the `hf` CLI is installed
    (`uv tool install huggingface_hub`). Then retry the start.

### 3. Check health

llama.cpp's server is also OpenAI-compatible and exposes `/health` on the
profile's port (default `8080`):

```bash
curl http://localhost:8080/health
```

```bash
llmux ps --running    # confirm the llama.cpp container is up
```

### 4. Watch the logs

=== "TUI"

    Select the profile and press ++l++.

=== "CLI"

    ```bash
    llmux logs gemma-3-4b
    ```

### 5. Stop the container

=== "TUI"

    Select the profile and press ++d++.

=== "CLI"

    ```bash
    llmux down gemma-3-4b
    ```

## Next steps

You've now served a model with both backends. To go further:

- **[Profiles](../guide/profiles.md)** — the full `profiles.yaml` schema,
  `defaults` inheritance, and per-backend fields.
- **[Model Configs](../guide/configs.md)** — tune engine flags per model in
  `config/<backend>/<name>.yaml`.
- **[Container Lifecycle](../guide/container-lifecycle.md)** — image version
  policy, the conflict gate, and `render-env`.
- **[Using the TUI](../guide/tui.md)** — the dashboard, keyboard reference,
  and in-TUI editing.
- **[CLI Reference](../reference/cli.md)** — every subcommand and flag for
  scripts, agents, and CI.
