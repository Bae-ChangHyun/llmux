# Profiles

A **profile** is a named, backend-specific launch recipe. It records *which*
container to run, on *which* port and GPU, and *which* [model config](configs.md)
to apply. All profiles for both backends live in a single `profiles.yaml` at the
repo root — this file is the single source of truth that llmux reads on every
operation.

!!! info "profiles.yaml is local-only"
    `profiles.yaml` is gitignored. Copy `profiles.example.yaml` to
    `profiles.yaml` and edit it, or let the [TUI](tui.md) / CLI create entries
    for you. llmux falls back to the built-in defaults if the file is missing.

## File structure

`profiles.yaml` has three top-level keys:

```yaml
version: 1                # schema version (currently always 1)

defaults:                 # per-backend defaults; profiles inherit anything they omit
  vllm:
    port: 8000
    gpu_id: "0"
    tensor_parallel_size: 1
    enable_lora: false
  llamacpp:
    port: 8080
    gpu_id: "0"

profiles:                 # the list of profile entries
  - name: qwen3-0-6b
    backend: vllm
    # ...
  - name: gemma-3-4b
    backend: llamacpp
    # ...
```

The minimum valid entry is just `name` + `backend`. Anything you leave out is
inherited from `defaults.<backend>` (and, below that, from llmux's built-in
defaults). When llmux writes the file back, it only persists fields that differ
from the resolved defaults, so entries stay compact.

### What happens at launch

When you start a profile, llmux renders it into
`.runtime/<backend>/<name>.env` and passes that file to
`docker compose --env-file`. You never edit the `.env` file directly — it is
regenerated from `profiles.yaml` every time. See
[Container Lifecycle](container-lifecycle.md) for the full flow.

## Field reference

Every profile is stored as one `StoredProfile` record. Fields that don't apply
to a backend simply stay at their default and are ignored.

| Field | Type | Default | Backend | Description |
| --- | --- | --- | --- | --- |
| `name` | string | — (required) | both | Profile identifier. Must start with an alphanumeric and contain only `[A-Za-z0-9_-]`. Unique *within* a backend. |
| `backend` | `vllm` \| `llamacpp` | — (required) | both | Which engine this profile runs. |
| `container_name` | string | = `name` | both | Docker container name. Defaults to the profile name. |
| `port` | int | `8000` (vLLM) / `8080` (llama.cpp) | both | Host port mapped to the in-container server port. |
| `gpu_id` | string | `"0"` | both | GPU id(s). Comma-separated for multi-GPU, e.g. `"0,1"`. |
| `config_name` | string | = `name` | both | Linked [config](configs.md) file: `config/<backend>/<config_name>.yaml`. |
| `tensor_parallel_size` | int | `1` | vLLM | Tensor-parallel degree. Set automatically to the count of `gpu_id` entries by the CLI/TUI. |
| `model_id` | string | `""` | vLLM | Hugging Face model id. Used to auto-generate a config on first launch if none exists. |
| `enable_lora` | bool | `false` | vLLM | Enable LoRA serving (`--enable-lora`). |
| `max_loras` | int \| null | `null` | vLLM | `--max-loras`. Omitted from the rendered env when null. |
| `max_lora_rank` | int \| null | `null` | vLLM | `--max-lora-rank`. Omitted when null. |
| `lora_modules` | string | `""` | vLLM | Comma-separated `name=path` adapters, e.g. `a1=/app/lora/p1,a2=/app/lora/p2`. |
| `extra_pip_packages` | string | `""` | vLLM | Extra pip packages installed before `vllm serve`, e.g. `flash-attn==2.7.4.post1`. |
| `env_vars` | map | `{}` | vLLM | Arbitrary `KEY: VALUE` pairs appended to the rendered env file. Keys must be valid env-var names. |
| `model_file` | string | `""` | llama.cpp | GGUF filename. Used for the download and as a config fallback. |
| `hf_repo` | string | `""` | llama.cpp | Hugging Face repo id `llama-server` downloads from (`-hf`). |
| `hf_file` | string | `""` | llama.cpp | Specific `.gguf` file inside `hf_repo` (`-hff`). |
| `image_tag` | string | `""` | both | Per-profile Docker image override. Empty = use the default image from `compose/<backend>/`. See [below](#per-profile-image_tag). |

## Profile examples

=== "vLLM"

    ```yaml
    profiles:
      - name: qwen3-0-6b
        backend: vllm
        container_name: qwen3-0-6b
        port: 8000
        config_name: qwen3-0-6b       # → config/vllm/qwen3-0-6b.yaml
        model_id: Qwen/Qwen3-0.6B     # used by auto-config on first launch
        gpu_id: "0"
        # enable_lora: true
        # max_loras: 2
        # max_lora_rank: 16
        # lora_modules: adapter1=/app/lora/path1,adapter2=/app/lora/path2
        # extra_pip_packages: flash-attn==2.7.4.post1
        # env_vars:
        #   VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: "1"
    ```

    A vLLM profile needs *either* a `model_id` *or* a linked config with a
    valid `model:` field. If neither is present, llmux scaffolds a placeholder
    config and refuses to start until you fill it in.

=== "llama.cpp"

    ```yaml
    profiles:
      - name: gemma-3-4b
        backend: llamacpp
        container_name: gemma-3-4b
        port: 8080
        config_name: gemma-3-4b       # → config/llamacpp/gemma-3-4b.yaml
        gpu_id: "0"
        model_file: gemma-3-4b-it-q4_k_m.gguf  # download + config fallback
        hf_repo: unsloth/gemma-3-4b-it-GGUF
        hf_file: gemma-3-4b-it-q4_k_m.gguf
    ```

    A llama.cpp profile needs GGUF metadata: `hf_repo` plus `hf_file` (or
    `model_file`). `llama-server` downloads the file directly into the host HF
    cache with `-hf` / `-hff` — no host-side `huggingface-cli` needed.

## Per-profile `image_tag`

`image_tag` pins a profile to a specific Docker image instead of the default
declared in `compose/<backend>/docker-compose.yaml`. The typical use is pinning
a profile to a locally [dev-built image](dev-build.md):

```yaml
- name: gemma-3-4b-mtp
  backend: llamacpp
  config_name: gemma-3-4b
  image_tag: llamacpp-dev:mtp_main   # use the locally built dev image
  hf_repo: unsloth/gemma-3-4b-it-GGUF
  hf_file: gemma-3-4b-it-q4_k_m.gguf
```

When a llama.cpp profile's `image_tag` starts with `llamacpp-dev:`, llmux layers
`docker-compose.dev.yaml` into the compose invocation so the `image:` line is
swapped while the rest of the service definition stays intact.

!!! tip "One-off override vs. pinned"
    `image_tag` is the *persistent* pin. For a one-time launch with a different
    image, the [TUI's Start Container screen](tui.md#container-start-screen)
    and `llmux up --tag/--dev` let you override per-launch without editing
    `profiles.yaml`.

## Editing profiles from the CLI

You can manage `profiles.yaml` entirely from the headless CLI — handy for
scripts and agents. (The [TUI](tui.md) offers the same operations as forms.)

```bash
# List every profile across both backends
llmux profile list

# Show a single profile's full record
llmux profile show qwen3-0-6b

# Create a new vLLM profile
llmux profile new my-model --backend vllm --port 8001 --gpu-id 0 \
    --model Qwen/Qwen3-8B --config my-model

# Create a llama.cpp profile
llmux profile new gemma --backend llamacpp \
    --model-file gemma-3-4b-it-q4_k_m.gguf \
    --hf-repo unsloth/gemma-3-4b-it-GGUF \
    --hf-file gemma-3-4b-it-q4_k_m.gguf

# Edit fields (only the options you pass change)
llmux profile edit my-model --port 8002 --gpu-id 0,1
llmux profile edit my-model --set VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
llmux profile edit gemma --image-tag llamacpp-dev:master

# One-step profile + config from a model id (vLLM only)
llmux profile quick-setup Qwen/Qwen3-8B --gpu-id 0 --gpu-mem 0.9

# Delete (optionally with the linked config)
llmux profile delete my-model --with-config
```

!!! note "Auto-detected backend"
    Most commands auto-detect the backend from the profile name. If the same
    name exists in both backends, pass `--backend` / `-b` to disambiguate.

## See also

- [Model Configs](configs.md) — the engine-flag YAML each profile links to
- [Container Lifecycle](container-lifecycle.md) — how a profile gets started and stopped
- [Dev Builds](dev-build.md) — building the images `image_tag` can point at
