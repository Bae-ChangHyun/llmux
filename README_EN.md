<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose Logo" width="280"/>

# vLLM Compose

**Docker Compose based vLLM Multi-Model Serving Manager**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

English | [한국어](README.md)

</div>

---

## 💡 Why?

When serving models with vLLM on a server...

- Start this model to test, stop it, start another model...
- Which GPU was it? What port? What settings?

**Too annoying.** So I built this.

**Easy to start with TUI menu-based interface - no complex setup needed!**

---

## ✨ Key Features

🖥️ **Interactive TUI Mode** - Menu-driven interface like a GUI

🔄 **Smart Version Management** - Docker Hub integration, auto-fetch Official/Nightly versions

🚀 **Profile-Based Management** - Independent model configs with `.env` files

⚡ **Auto GPU Mapping** - Automatic port allocation per GPU, conflict prevention

🔨 **Source Build** - Auto GPU detection for fast builds (10-30 min)

🔗 **LoRA Support** - Multi-adapter loading with automatic path mapping

---

## 🚀 Quick Start

### Prerequisites

```bash
docker --version        # Check Docker
docker compose version  # Check Docker Compose
nvidia-smi             # Check NVIDIA GPU
```

### Clone & Common Settings

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git
cd vllm-compose

# Create .env.common
cat > .env.common << EOF
VLLM_VERSION=latest
HF_TOKEN=your_huggingface_token
HF_CACHE_PATH=~/.cache/huggingface
EOF
```

---

### Method 1: Using TUI (Recommended)

**All tasks made easy with menu-based interface**

```bash
./run.sh
```

1. Select **Quick Setup**
2. Enter model name (e.g., `Qwen/Qwen3-VL-30B`)
3. Enter GPU ID, port
4. Profile + config auto-generated
5. Start container directly from menu!

> TUI menu provides profile creation/edit/delete, container management, version selection, etc.

---

### Method 2: Using CLI

**Direct control with terminal commands**

#### 1) Create Profile

```bash
# config/mymodel.yaml
cat > config/mymodel.yaml << EOF
model: huggingface/model-name
gpu-memory-utilization: 0.9
EOF

# profiles/mymodel.env
cat > profiles/mymodel.env << EOF
CONTAINER_NAME=mymodel
VLLM_PORT=8000
CONFIG_NAME=mymodel
GPU_ID=0
TENSOR_PARALLEL_SIZE=1
ENABLE_LORA=false
EOF
```

#### 2) Run

```bash
./run.sh mymodel up
```

---

## 🎨 Project Structure

```
vllm-compose/
├── profiles/              # Model profiles (.env)
│   ├── model1.env
│   └── model2.env
├── config/                # vLLM configs (YAML)
│   ├── model1.yaml
│   └── model2.yaml
├── docker-compose.yaml
├── .env.common            # Common settings
└── run.sh                 # Management script
```

---

## 📚 Detailed Guides

<details>
<summary><strong>🖥️ TUI Mode Details</strong></summary>

### Launch TUI

```bash
./run.sh        # or
./run.sh tui
```

> **Requirements**: `whiptail` or `dialog`
> ```bash
> sudo apt-get install whiptail
> ```

### Main Menu

```
┌──────────────── vLLM Compose ─────────────────┐
│  Q. Quick Setup         Auto-create Profile    │
│  1. Container Mgmt      Start/Stop/Logs        │
│  2. Profile Mgmt        Create/Edit/Delete     │
│  3. Config Mgmt         Create/Edit            │
│  4. Build Mgmt          Build/Images           │
│  5. System Info         GPU/Version/Status     │
│  X. Exit                                       │
└───────────────────────────────────────────────┘
```

### Version Selection

When starting a container, select version:

```
1. Current running: vllm/vllm-openai:nightly (...)
2. Official Latest: v0.15.0
3. Nightly: 2026-01-29
4. Dev build (local source builds)
5. Custom tag
```

- **Official/Nightly**: Auto-pull latest + cleanup unused images

</details>

<details>
<summary><strong>⌨️ Full CLI Commands</strong></summary>

### Profile Management

| Command | Description |
|:---|:---|
| `./run.sh list` | List profiles and status |
| `./run.sh {profile} up` | Start container |
| `./run.sh {profile} up --dev` | Start with dev build |
| `./run.sh {profile} down` | Stop container |
| `./run.sh {profile} logs` | View logs (real-time) |
| `./run.sh {profile} status` | Check status |

### Version & Images

| Command | Description |
|:---|:---|
| `./run.sh version` | Show version info |
| `./run.sh images` | List dev build images |

### Build

| Command | Description |
|:---|:---|
| `./run.sh build` | Fast build main branch |
| `./run.sh build [branch]` | Build specific branch |
| `./run.sh build [branch] --repo <url>` | Build from custom repo/fork |
| `./run.sh build --official` | Official build (all GPUs) |
| `./run.sh build [branch] --tag TAG` | Build with custom tag |

### System

| Command | Description |
|:---|:---|
| `./run.sh ps` | Running containers |
| `./run.sh gpu` | GPU status |

</details>

<details>
<summary><strong>🔧 Add New Model Profile</strong></summary>

### Method 1: Quick Setup (TUI)

```bash
./run.sh
# → Select "Quick Setup"
# → Enter model name, GPU, port
# → Auto-generated
```

### Method 2: Manual Creation

#### 1) Create Config

```yaml
# config/mymodel.yaml
model: huggingface/model-name
gpu-memory-utilization: 0.9
max-model-len: 32768
```

#### 2) Create Profile

```bash
# profiles/mymodel.env
CONTAINER_NAME=mymodel
VLLM_PORT=8003
CONFIG_NAME=mymodel

GPU_ID=0
TENSOR_PARALLEL_SIZE=1

ENABLE_LORA=false
```

#### 3) Run

```bash
./run.sh mymodel up
```

### Key Settings

| Setting | Description | Example |
|:---|:---|:---|
| `CONTAINER_NAME` | Container name | `mymodel` |
| `VLLM_PORT` | API serving port | `8000` |
| `CONFIG_NAME` | Config filename (no ext) | `mymodel` |
| `GPU_ID` | GPU number | `0` or `0,1` |
| `TENSOR_PARALLEL_SIZE` | TP size | `1`, `2`, `4` |

</details>

<details>
<summary><strong>🔨 vLLM Source Build</strong></summary>

### Fast Build (Recommended)

Detects current GPU only for fast build (10-30 min)

```bash
./run.sh build              # main branch
./run.sh build v0.15.0      # specific version
./run.sh build my-branch    # specific branch
```

Output:
```
Detected GPU: NVIDIA RTX 4080 (sm_8.9)
Building for your GPU only - MUCH faster!
```

### Custom Repository (Fork)

Build from someone's fork or your own repo

```bash
# Custom repo
./run.sh build main --repo https://github.com/username/vllm.git

# Custom repo + specific branch
./run.sh build custom-feature --repo https://github.com/username/vllm.git
```

### Official Build

All GPU architectures (3-6 hours)

```bash
./run.sh build --official
./run.sh build v0.15.0 --official
```

### Check Build Information

View detailed info of built images (repo, branch, commit, date)

```bash
./run.sh images
```

Output:
```
Tag: vllm-dev:main-20260130
  Size: 42GB | Created: 2026-01-30 15:30
  Repository: vllm-project/vllm
  Branch: main | Commit: abc1234
  Built: 2026-01-30 | Type: fast
```

### Run with Dev Build

```bash
./run.sh mymodel up --dev
./run.sh mymodel up --dev --tag main-20260130
```

</details>

<details>
<summary><strong>🔗 LoRA Adapter</strong></summary>

### 1. Set Base Path

```bash
# .env.common
LORA_BASE_PATH=/home/user/lora-adapters
```

### 2. Configure Profile

```bash
# profiles/mymodel.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=adapter1=/app/lora/my_adapter_v1
```

### 3. Path Mapping

```
Local: /home/user/lora-adapters/my_adapter_v1
  ↓ Auto mount
Container: /app/lora/my_adapter_v1
```

⚠️ LORA_MODULES must start with `/app/lora/`

### 4. Multiple Adapters

```bash
LORA_MODULES=ko=/app/lora/ko_adapter,en=/app/lora/en_adapter
```

### 5. API Usage

```python
# Use specific adapter
response = client.chat.completions.create(
    model="ko",  # Name from LORA_MODULES
    messages=[...]
)
```

### Supported Models

| Model | LoRA Support |
|:---|:---:|
| Qwen2-VL, Qwen3-VL | ✅ |
| LLaVA | ✅ |
| DeepSeek-OCR | ✅ (v0.13.0+) |

</details>

<details>
<summary><strong>🐛 Troubleshooting & FAQ</strong></summary>

### Where do I add vLLM arguments?

All vLLM arguments can be added to `config/*.yaml` files.

```yaml
# config/mymodel.yaml
model: huggingface/model-name
gpu-memory-utilization: 0.9
max-model-len: 32768

# Additional vLLM arguments
trust-remote-code: true
dtype: bfloat16
max-num-seqs: 256
enable-chunked-prefill: true
```

> Any vLLM CLI argument can be written in YAML format
> Reference: [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server)

### Container Won't Start

```bash
./run.sh {profile} logs     # Check logs
./run.sh gpu                # GPU status
./run.sh {profile} status   # Container status
```

### GPU OOM

```yaml
# config/mymodel.yaml
gpu-memory-utilization: 0.7  # Lower
```

Or use TP:
```bash
# profiles/mymodel.env
TENSOR_PARALLEL_SIZE=2
```

### Port Conflict

```bash
# profiles/mymodel.env
VLLM_PORT=8001  # Change
```

Check:
```bash
sudo lsof -i :8000
./run.sh ps
```

### LoRA Path Error

1. Check `LORA_BASE_PATH` in `.env.common`
2. Verify `LORA_MODULES` starts with `/app/lora/`
3. Confirm files exist locally

```bash
ls -la /home/user/lora-adapters/
docker exec -it {container} ls -la /app/lora/
```

</details>

---

## 💻 Tech Stack

- Docker, Docker Compose
- [vLLM](https://github.com/vllm-project/vllm)
- NVIDIA CUDA
- Bash Shell Script

---

## 📄 License

MIT License

---

<div align="center">

**vLLM Compose**
Made with ❤️ for AI Developers

</div>
