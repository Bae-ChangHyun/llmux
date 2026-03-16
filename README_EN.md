<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose" width="240"/>

# vLLM Compose

**Multi-model vLLM serving, simplified.**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**English** | [한국어](README.md)

---

**Spin up models, swap GPUs, juggle ports, remember configs...**
<br/>
**Stop repeating yourself. Manage it all from one screen.**

</div>

<br/>

## Get Started in 30 Seconds

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git && cd vllm-compose

# Common settings (HuggingFace token, cache path)
cat > .env.common << 'EOF'
VLLM_VERSION=latest
HF_TOKEN=your_token_here
HF_CACHE_PATH=~/.cache/huggingface
EOF

# Launch TUI
pip install textual && ./run.sh
```

> Quick Setup auto-generates profile + config from just a model name.

<br/>

## Why vLLM Compose?

| | Manual | vLLM Compose |
|:---|:---|:---|
| **Switch models** | Repeat docker commands | One press in TUI |
| **Manage configs** | Long CLI args by hand | YAML files + Tab autocomplete |
| **GPU allocation** | Check nvidia-smi manually | Live GPU monitoring |
| **Multi-model** | Edit compose files directly | Independent per-profile management |
| **Versioning** | Track image tags manually | Latest/Official/Nightly selector |

<br/>

## Core Features

**TUI** &mdash; Modern terminal UI for managing everything in one place

**Profiles** &mdash; Per-model settings as `.env` files, instant switching

**Config** &mdash; vLLM params as YAML, Tab autocomplete for 51 known parameters

**Source Build** &mdash; Auto GPU detection Fast Build (10-30 min), fork support

**LoRA** &mdash; Multi-adapter loading with automatic path mapping

<br/>

---

<details>
<summary><b>TUI Keyboard Shortcuts</b></summary>

<br/>

| Key | Action |
|:---|:---|
| `Enter` | Profile action menu (start/stop/logs/edit/delete) |
| `w` | Quick Setup |
| `n` | New profile |
| `F1` `F2` `F3` | Dashboard / Configs / System |
| `?` | Full shortcut help |

</details>

<details>
<summary><b>CLI Usage</b></summary>

<br/>

```bash
./run.sh list                    # List profiles
./run.sh {profile} up            # Start
./run.sh {profile} down          # Stop
./run.sh {profile} logs          # Logs
./run.sh build                   # Source build
./run.sh build --repo <url>      # Fork build
```

</details>

<details>
<summary><b>Profile & Config Structure</b></summary>

<br/>

```yaml
# config/my-model.yaml — vLLM serving config
model: Qwen/Qwen3-30B
gpu-memory-utilization: 0.9
max-model-len: 32768
trust-remote-code: true
```

```bash
# profiles/my-model.env — Container settings
CONTAINER_NAME=my-model
VLLM_PORT=8000
CONFIG_NAME=my-model
GPU_ID=0
TENSOR_PARALLEL_SIZE=1
ENABLE_LORA=false
```

</details>

<details>
<summary><b>Source Build</b></summary>

<br/>

```bash
# Fast Build — current GPU only, 10-30 min
./run.sh build                              # main
./run.sh build v0.15.0                      # specific version
./run.sh build main --repo <fork-url>       # fork

# Official Build — all GPUs, 3-6 hours
./run.sh build --official

# Run with dev image
./run.sh mymodel up --dev
```

</details>

<details>
<summary><b>LoRA Adapters</b></summary>

<br/>

```bash
# .env.common
LORA_BASE_PATH=/home/user/lora-adapters

# profiles/mymodel.env
ENABLE_LORA=true
MAX_LORAS=2
LORA_MODULES=ko=/app/lora/ko_adapter,en=/app/lora/en_adapter
```

```python
response = client.chat.completions.create(
    model="ko",  # LoRA adapter name
    messages=[...]
)
```

</details>

<details>
<summary><b>Troubleshooting</b></summary>

<br/>

| Problem | Solution |
|:---|:---|
| Container won't start | Check logs: `./run.sh {profile} logs` |
| GPU OOM | Set `gpu-memory-utilization: 0.7` or `TENSOR_PARALLEL_SIZE=2` |
| Port conflict | Change `VLLM_PORT`, verify with `./run.sh ps` |
| Add vLLM args | Write any CLI arg as YAML in `config/*.yaml` |

</details>

---

<div align="center">

**MIT License** · Made for AI Developers

</div>
