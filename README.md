<div align="center">

<img src="assets/llmux.png" alt="llmux" width="240"/>

# llmux

**One TUI. Two backends. Zero config headaches.**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Latest-8A2BE2?style=flat-square)](https://github.com/ggerganov/llama.cpp)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**English** | [한국어](README_KO.md)

---

vLLM for HF Transformers. llama.cpp for GGUF.
<br/>
**Two different toolchains, two different configs, two different terminals.**
<br/><br/>
llmux unifies both under a single Docker Compose dashboard.
<br/>
**Pick a profile, press Enter &mdash; whichever engine it belongs to just runs.**

<br/>

<img src="assets/demo.gif" alt="llmux Demo" width="700"/>

</div>

<br/>

## Get Started in 30 Seconds

```bash
git clone https://github.com/Bae-ChangHyun/llmux.git && cd llmux

# Set HuggingFace token (shared by both backends)
cat > .env.common << 'EOF'
HF_TOKEN=your_token_here
HF_CACHE_PATH=/home/your-username/.cache/huggingface
EOF

# Launch
uv sync
uv run llmux
```

> Press `n` to create a profile for either **vLLM** or **llama.cpp** &mdash; llmux figures out the rest.

<br/>

## Why llmux?

| | Manual | llmux |
|:---|:---|:---|
| **Switch engines** | Different CLI, different compose, different TUI per engine | One dashboard for both vLLM and llama.cpp |
| **Port / GPU conflicts** | Find out when the container crashes | Pre-start conflict gate across both backends |
| **Benchmark** | `curl` the endpoint, eyeball the timing | Built-in `/v1/chat/completions` bench for either engine |
| **Memory sizing** | Guess and hope it fits | Per-model HF memory estimator (`m` key) |
| **GGUF setup** | `hf download` &rarr; edit compose &rarr; mount | Auto-download on start, paths resolved by profile |

<br/>

## Core Features

**Unified TUI** &mdash; Single Textual dashboard lists every vLLM and llama.cpp profile side by side

**Cross-backend conflict gate** &mdash; Detects port + GPU clashes between engines before spin-up

**OpenAI-compatible bench** &mdash; One-shot `/v1/chat/completions` benchmark for vLLM *and* llama.cpp

**Quick Setup** &mdash; Type a HuggingFace model name &rarr; profile + config auto-generated (either backend)

**GGUF auto-download** &mdash; llama.cpp profiles pull the file on first start, no manual `hf download`

**Memory Estimator** &mdash; [hf-mem](https://github.com/alvarobartt/hf-mem) integration with per-GPU progress

**Real-time monitors** &mdash; GPU usage bar, container status, logs follow, all from keyboard

<br/>

---

<details>
<summary><b>TUI Keyboard Shortcuts</b></summary>

<br/>

| Key | Action |
|:---|:---|
| `Enter` | Profile action menu (start / stop / logs / bench / edit / delete) |
| `n` | New profile (backend picker: vLLM or llama.cpp) |
| `w` | Quick Setup (backend picker) |
| `m` | Memory estimator |
| `s` | System info |
| `c` | Configs |
| `u` / `d` / `l` | Start / Stop / Logs (selected profile) |
| `?` | Full shortcut help |

</details>

<details>
<summary><b>Profile & Config Structure</b></summary>

<br/>

```
profiles/vllm/*.env         # vLLM container settings
profiles/llamacpp/*.env     # llama.cpp container settings
config/vllm/*.yaml          # vLLM serving config (per profile)
config/llamacpp/*.yaml      # llama-server flags (per profile)
```

```yaml
# config/vllm/my-model.yaml
model: Qwen/Qwen3-30B-A3B
gpu-memory-utilization: 0.9
max-model-len: 32768
```

```yaml
# config/llamacpp/my-model.yaml
# Flags map 1:1 to llama-server CLI args
n-gpu-layers: 99
ctx-size: 16384
parallel: 4
```

</details>

<details>
<summary><b>Cross-backend conflict gate</b></summary>

<br/>

Start a llama.cpp profile on port `8080` while a vLLM profile is already bound to it?
llmux catches that **before** launching docker, showing exactly which profile owns the port or GPU.

The same check runs across engines &mdash; starting `vllm/qwen` won't silently collide with `llamacpp/qwen-q4` if they share `GPU_ID=0`.

</details>

<details>
<summary><b>Troubleshooting</b></summary>

<br/>

| Problem | Solution |
|:---|:---|
| Container won't start | Open the profile in TUI and inspect logs |
| GPU OOM (vLLM) | Lower `gpu-memory-utilization` or raise `TENSOR_PARALLEL_SIZE` |
| GPU OOM (llama.cpp) | Lower `n-gpu-layers` in config YAML |
| Port conflict | TUI will warn pre-start; change `VLLM_PORT` or `LLAMACPP_PORT` in profile |
| GGUF download stuck | Check `HF_TOKEN` in `.env.common`; retry from TUI &mdash; `switch.sh` auto-calls `pull-model.sh` on start |
| Copy logs in TUI | Shift+drag to select, Ctrl+C to copy ([details](https://textual.textualize.io/FAQ/#how-can-i-select-and-copy-text-in-a-textual-app)) |

</details>

---

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.10+ (for TUI)
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU(s)

---

## Roadmap

- [ ] **Profile clone across backends** &mdash; Duplicate a vLLM profile as a llama.cpp GGUF version for quick A/B testing
- [ ] **Batch operations** &mdash; Start/stop multiple profiles across both backends at once
- [ ] **Export/Import bundles** &mdash; Share full profile + config sets between machines
- [ ] **Quantization recommender** &mdash; Suggest `Q4_K_M` vs `Q8_0` based on target GPU + memory estimator
- [ ] **Web UI** &mdash; Optional browser-based dashboard for remote access
- [ ] **Model-specific presets** &mdash; Curated configs per family (Llama, Qwen, DeepSeek) for both engines

---

## Credits

llmux absorbs and unifies two prior projects by the same author:

- [`vllm-compose`](https://github.com/Bae-ChangHyun/vllm-compose) &mdash; vLLM profiles
- [`llamacpp-compose`](https://github.com/Bae-ChangHyun/llamacpp-compose) &mdash; llama.cpp profiles

---

<div align="center">

**MIT License**

</div>
