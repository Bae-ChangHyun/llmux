# Model Configs

A **config** is a YAML file holding the engine-level flags for one model. While
a [profile](profiles.md) decides *where* a container runs (port, GPU, container
name), the config decides *how the model is served* — context length, memory
budget, sampling defaults, and so on.

Configs live under `config/<backend>/<name>.yaml`:

- `config/vllm/<name>.yaml` — flags for `vllm serve`
- `config/llamacpp/<name>.yaml` — flags for `llama-server`

Each profile links to one config via its `config_name` field (defaulting to the
profile name). The key principle is **1:1 mapping**: every key in a config maps
directly to an engine command-line flag.

!!! info "example.yaml is a template"
    Each backend ships a `config/<backend>/example.yaml` template. Copy it to a
    new name — `cp config/vllm/example.yaml config/vllm/my-model.yaml` — or use
    `llmux config new`. The `example` stem is excluded from config listings.

## vLLM configs

Keys map 1:1 to `vllm serve` engine arguments. Use the long-flag name with `--`
stripped: `--gpu-memory-utilization 0.9` becomes `gpu-memory-utilization: 0.9`.
The full flag set is documented in the
[vLLM engine args reference](https://docs.vllm.ai/en/latest/serving/engine_args.html).

```yaml
# config/vllm/example.yaml
model: Qwen/Qwen3-8B               # HuggingFace repo id or local path

# ---- memory / parallelism ----
gpu-memory-utilization: 0.90       # 0.0–1.0, fraction of each GPU's VRAM vLLM may use
max-model-len: 32768               # context length (lower than model default = less memory)
# tensor-parallel-size: 2          # multi-GPU (keep in sync with the profile's GPU_ID)

# ---- sampling defaults (overridden per-request) ----
# dtype: bfloat16
# trust-remote-code: true
```

| Key | Type | Description |
| --- | --- | --- |
| `model` | string | Hugging Face repo id or local path. **Required** unless the profile sets `model_id`. |
| `gpu-memory-utilization` | float | Fraction (0.0–1.0) of each GPU's VRAM vLLM may claim. |
| `max-model-len` | int | Maximum context length. Set below the model default to save memory. |
| `tensor-parallel-size` | int | Tensor-parallel degree for multi-GPU. Keep consistent with the profile's `gpu_id` / `tensor_parallel_size`. |
| `dtype` | string | Weight dtype, e.g. `bfloat16`. |
| `trust-remote-code` | bool | Allow custom model code from the repo. |

Any other `vllm serve` flag is valid here — these are just the common ones.

!!! note "Where the config is mounted"
    The whole `config/vllm/` directory is mounted into the container at
    `/config/`, and vLLM is launched with `--config /config/<config_name>.yaml`.
    The profile's `tensor_parallel_size` and LoRA options are passed as separate
    compose-level flags, not from this file.

## llama.cpp configs

Keys map 1:1 to `llama-server` long flags: `--ctx-size 32768` becomes
`ctx-size: 32768`, `--jinja` becomes `jinja: true`, `--temp 0.6` becomes
`temp: 0.6`. At launch, `render-override.py` reads the config and renders a
`command:` block for the container. See
[Container Lifecycle](container-lifecycle.md) for that flow.

```yaml
# config/llamacpp/example.yaml
model-file: your-model.gguf        # GGUF filename (relative to models/)

ctx-size: 8192                     # context length
n-gpu-layers: 99                   # layers to offload to GPU. 99 = all

cache-type-k: bf16                 # KV cache K precision: f16 / bf16 / q8_0 / q4_0 ...
cache-type-v: bf16                 # KV cache V precision

temp: 0.7                          # sampling defaults
top-p: 0.9
top-k: 40

alias: your-model                  # name shown in /v1/models

jinja: true                        # Jinja chat template (recommended for Qwen/Llama3 etc.)

# MoE expert CPU offload (for A3B-style models):
# regex-match tensor names → offload matched tensors to CPU
# override-tensors:
#   - ".ffn_.*_exps.=CPU"

# api-key: sk-your-key             # optional auth

extra-args: []                     # verbatim args render-override.py appends as-is
```

| Key | Type | Description |
| --- | --- | --- |
| `model-file` | string | GGUF filename. Acts as a display/fallback value — the actual download source is the profile's `hf_repo` / `hf_file`. |
| `alias` | string | Model name exposed at `/v1/models`. |
| `ctx-size` | int | Context length in tokens. |
| `n-gpu-layers` | int | Number of layers offloaded to GPU. `99` = all. |
| `cache-type-k` / `cache-type-v` | string | KV cache precision: `f16`, `bf16`, `q8_0`, `q4_0`, etc. |
| `temp` / `top-p` / `top-k` / `min-p` | float / int | Default sampling parameters (overridable per request). |
| `jinja` | bool | Use the model's Jinja chat template. Recommended for modern models. |
| `flash-attn` | bool | Enable Flash Attention. |
| `parallel` | int | Number of concurrent decoding slots. |
| `cont-batching` | bool | Continuous batching. |
| `metrics` | bool | Expose the `/metrics` Prometheus endpoint. |
| `api-key` | string | Require this key on incoming requests. |
| `override-tensors` | list | MoE CPU offload. Each entry is a `<regex>=<device>` rule, e.g. `".ffn_.*_exps.=CPU"`, applied as `-ot`. Keeps experts on CPU while attention/embeddings stay on GPU. |
| `extra-args` | list | Escape hatch — items are appended to the `llama-server` command verbatim, for newer flags not expressible in the schema above. |

!!! warning "`host` / `port` / `no-webui` are forced"
    `render-override.py` always sets `--host 0.0.0.0 --port 8080 --no-webui` and
    drops any `host`, `port`, `webui`, or `no-webui` keys from your config. Map
    the host port via the profile's `port` field instead.

### MoE expert CPU offload

For Mixture-of-Experts models (e.g. Qwen3-A3B), `override-tensors` lets you keep
the large expert tensors on CPU while attention and embeddings stay on GPU:

```yaml
n-gpu-layers: 99
override-tensors:
  - ".ffn_.*_exps.=CPU"   # FFN expert tensors → CPU; everything else → GPU
```

Each rule renders to a `-ot <rule>` argument.

## Editing configs from the CLI

```bash
# List configs across both backends
llmux config list

# Print one config
llmux config show my-model

# Create a vLLM config
llmux config new my-model --backend vllm --model Qwen/Qwen3-8B --gpu-mem 0.9

# Create a llama.cpp config, setting flags inline
llmux config new gemma --backend llamacpp \
    --set model-file=gemma-3-4b-it-q4_k_m.gguf \
    --set ctx-size=8192 \
    --set n-gpu-layers=99 \
    --set jinja=true

# Patch an existing config
llmux config edit gemma --set temp=0.6 --set top-p=0.95
llmux config edit gemma --unset top-k

# Delete
llmux config delete gemma
```

!!! tip "`--set` values are YAML-typed"
    `--set` parses each value as YAML: `--set ctx-size=8192` stores an int,
    `--set jinja=true` stores a bool, and an empty value (`--set jinja=`) stores
    `true`. For list-valued keys like `override-tensors` or `extra-args`, pass a
    YAML literal — `--set 'override-tensors=[".ffn_.*_exps.=CPU"]'` — or edit the
    file directly / use the [TUI config form](tui.md#profile-and-config-forms).

## See also

- [Profiles](profiles.md) — the launch recipe that links to a config via `config_name`
- [Container Lifecycle](container-lifecycle.md) — how configs are rendered into the container at startup
- [Using the TUI](tui.md) — editing configs as interactive forms
