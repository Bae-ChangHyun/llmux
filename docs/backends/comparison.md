# Backend Comparison

llmux drives two engines from one dashboard, but they are built for different
jobs. vLLM serves **Hugging Face Transformers models** for high-throughput
inference; llama.cpp serves **GGUF models** with fine-grained control over
quantization and CPU/GPU placement. This page lays out the feature differences
and a "when to use which" guide.

## Feature matrix

| Feature | vLLM | llama.cpp |
|---|:---:|:---:|
| Model format | HF Transformers (safetensors) | GGUF |
| Container image | `vllm/vllm-openai` | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| In-container port | 8000 | 8080 |
| Multi-GPU tensor parallelism | ✓ (`gpu_id: "0,1"` + `tensor_parallel_size`) | ✗ (single-GPU device list) |
| GPU passthrough mechanism | `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES` | `deploy.resources` device-id list |
| LoRA adapters | ✓ (`enable_lora` + `docker-compose.lora.yaml` mount) | ✗ |
| KV cache precision control | ✗ | ✓ (`cache-type-k` / `cache-type-v`) |
| CPU/GPU layer offload | ✗ | ✓ (`n-gpu-layers`) |
| MoE expert CPU offload | ✗ | ✓ (`override-tensors`, regex → CPU) |
| Speculative / multi-token decoding | — | ✓ via MTP (dev build + MTP GGUF + `extra-args`) |
| Model auto-download | HF cache (engine downloads) | HF cache (`-hf`/`-hff`, engine downloads) |
| Host `hf` CLI required | No | No |
| Image source options | 5 (Local Latest / Official Release / Nightly / Dev Build / Custom Tag) | 3 (Default Image / Dev Build / Custom Tag) |
| Refuses `:latest` alias | ✓ | — (single official tag) |
| In-container version verification | ✓ (`vllm.__version__` checked after start) | ✗ |
| Dev build from source | ✓ (`vllm-dev:<tag>`, `docker/Dockerfile`) | ✓ (`llamacpp-dev:<tag>`, `.devops/cuda.Dockerfile`) |
| Extra pip packages at start | ✓ (`extra_pip_packages`) | ✗ |
| Per-profile env var overrides | ✓ (`env_vars`) | ✗ |
| Command injection method | Compose `command:` with `--config` flag | Per-profile override file from `render-override.py` |
| Post-start health probe | `/v1/models` returns a served model id | `/health` responds |
| Cross-backend conflict gate | ✓ (port + GPU overlap, both backends) | ✓ (port + GPU overlap, both backends) |
| Built-in OpenAI-compatible benchmark | ✓ | ✓ |

## Why the image-source counts differ

The asymmetry — vLLM has **five** image sources, llama.cpp has **three** — is
intentional, not an oversight.

- **vLLM** publishes many tags to Docker Hub: versioned `vX.Y.Z` releases, a
  rolling `nightly`, and the `:latest` alias. llmux refuses `:latest` outright
  (it does not describe its own contents) and instead exposes **Local Latest**
  (highest local semver tag), **Official Release** (Docker Hub's current stable
  version), and **Nightly** as distinct, well-defined choices — plus **Dev
  Build** and **Custom Tag**.
- **llama.cpp** publishes effectively a **single official `ghcr.io` tag**
  (`server-cuda`). There is no versioned-vs-nightly-vs-latest distinction to
  expose, so "Local Latest / Official Release / Nightly" collapses into one
  **Default Image** option — leaving **Default Image**, **Dev Build**, and
  **Custom Tag**.

In other words, both backends expose exactly the choices their upstream
registry actually offers.

## Why only vLLM does multi-GPU

vLLM's container uses `runtime: nvidia` with `NVIDIA_VISIBLE_DEVICES=${GPU_ID}`,
so a `gpu_id` of `"0,1"` exposes both GPUs and `tensor_parallel_size: 2` shards
the model across them. llama.cpp's container uses Compose's
`deploy.resources.reservations.devices` device-id list, which cannot reliably
expand a comma-separated value into a multi-GPU list. Tensor parallelism is a
core vLLM feature, so llmux keeps the two engines on the GPU-passthrough
mechanism each one needs rather than forcing a single format.

## When to use which

!!! tip "Use **vLLM** when..."
    - You want to run a model straight from its **Hugging Face Transformers
      repo** (safetensors weights, no separate quantization step).
    - You need **multi-GPU tensor parallelism** to fit or speed up a large
      model.
    - You want to attach **LoRA adapters** at serve time.
    - You need a specific, **verifiable engine version** (llmux checks
      `vllm.__version__` inside the container after start).
    - You want high-throughput batched serving and are happy with
      full/half-precision weights on the GPU.

!!! tip "Use **llama.cpp** when..."
    - Your model is a **GGUF** (or you want quantized inference: Q4/Q5/Q8).
    - The model is **too large for VRAM** and you need **CPU/GPU layer
      offload** (`n-gpu-layers`).
    - You are running a **MoE model** and want to push expert tensors to system
      RAM with `override-tensors` while keeping attention on the GPU.
    - You want to **squeeze context length** out of a fixed VRAM budget by
      quantizing the KV cache (`cache-type-k` / `cache-type-v`).
    - You want to experiment with **MTP / multi-token prediction** (via a dev
      build and an MTP-enabled GGUF).

Both backends speak the **OpenAI-compatible API**, render from the same
`profiles.yaml`, and share the cross-backend conflict gate and the built-in
benchmark — so you can run a vLLM and a llama.cpp profile side by side and
A/B test them with the identical request protocol.

## See also

- [vLLM backend](vllm.md) — full detail on the vLLM path
- [llama.cpp backend](llamacpp.md) — full detail on the llama.cpp path
- [Troubleshooting](../troubleshooting.md) — common problems for both backends
