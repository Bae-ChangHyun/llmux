<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose Logo" width="280"/>

# vLLM Compose

**Docker Compose based vLLM Multi-Model Serving Manager**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-v0.13.0-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)

English | [한국어](README.md)

[Quick Start](#-quick-start) • [Usage](#-usage) • [Add Profile](#-add-new-model-profile) • [LoRA](#-lora-adapter)

</div>

---

## 💡 Why?

When serving models with vLLM on a multi-GPU server...

- Start this model to test
- Stop it to switch to another model
- Start yet another model
- Wait, which GPU was it? What port?

**Too annoying.** So I built this.

Create a profile file once, then just `./run.sh vlm up`. Done.

---

## 🎨 Structure

```
vllm-compose/
├── profiles/              # Model profiles (.env)
│   ├── vlm.env
│   ├── llm.env
│   └── clova.env
├── config/                # vLLM configs (YAML)
│   ├── qwen3-vl-30b-a3b-fp8.yaml
│   └── gpt-oss-120b.yaml
├── docker-compose.yaml
├── .env.common            # Common settings (HF Token, etc.)
└── run.sh                 # Management script
```

---

## 🚀 Quick Start

### Prerequisites

```bash
docker --version
docker compose version
nvidia-smi
```

### 1. Clone

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git
cd vllm-compose
```

### 2. Common Settings

Create `.env.common`:

```bash
HF_TOKEN=your_huggingface_token
HF_CACHE_PATH=~/.cache/huggingface
LORA_BASE_PATH=/path/to/lora/adapters  # If using LoRA
```

### 3. Check Profiles

```bash
./run.sh list
```

### 4. Start Serving

```bash
./run.sh vlm up
```

---

## 📖 Usage

### run.sh Commands

| Command | Description |
|:---:|:---|
| `./run.sh list` | List profiles |
| `./run.sh {profile} up` | Start container |
| `./run.sh {profile} down` | Stop container |
| `./run.sh {profile} logs` | View logs |
| `./run.sh {profile} status` | Check status |
| `./run.sh ps` | Running containers |
| `./run.sh gpu` | GPU status |

### Direct Docker Compose

```bash
# Start
docker compose --env-file .env.common --env-file profiles/vlm.env -p vlm up -d

# Stop
docker compose -p vlm down

# Logs
docker logs -f vlm
```

---

## 🔧 Add New Model Profile

### 1. Model Config (config/)

```yaml
# config/my-model.yaml
model: huggingface/model-name
max-model-len: 32768
gpu-memory-utilization: 0.8
```

### 2. Profile (profiles/)

```bash
# profiles/mymodel.env
CONTAINER_NAME=mymodel
VLLM_PORT=8003
CONFIG_NAME=my-model

GPU_ID=0
TENSOR_PARALLEL_SIZE=1

# LoRA (optional)
ENABLE_LORA=false
```

### 3. Run

```bash
./run.sh mymodel up
```

---

## 🔗 LoRA Adapter

<details>
<summary><strong>LoRA Configuration</strong></summary>

### Profile Settings

```bash
# profiles/vlm.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=ocr-li=/app/lora/qwen3vl_li/lora_model
```

### Parameters

| Parameter | Description |
|----------|------|
| `ENABLE_LORA` | Enable LoRA (true/false) |
| `MAX_LORAS` | Max concurrent LoRAs |
| `MAX_LORA_RANK` | Max rank |
| `LORA_MODULES` | name=path format, comma separated |

### Path Mapping

```
Host: $LORA_BASE_PATH/qwen3vl_li/lora_model
Container: /app/lora/qwen3vl_li/lora_model
```

### VLM LoRA Limitation

- ✅ Language Model layers: Supported
- ❌ Vision Encoder layers: Not supported

</details>

---

## 🐛 Troubleshooting

<details>
<summary><strong>Container won't start</strong></summary>

```bash
./run.sh {profile} logs
./run.sh gpu
```

</details>

<details>
<summary><strong>GPU OOM</strong></summary>

Lower `gpu-memory-utilization` in `config/{model}.yaml`

</details>

<details>
<summary><strong>Port conflict</strong></summary>

Change `VLLM_PORT` in profile

</details>

---

<div align="center">

**vLLM Compose** - When switching models gets annoying

</div>
