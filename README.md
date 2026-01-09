<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose Logo" width="280"/>

# vLLM Compose

**Docker Compose 기반 vLLM 멀티 모델 서빙 관리 도구**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-v0.13.0-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)

[English](README_EN.md) | 한국어

[Quick Start](#-quick-start) • [Usage](#-usage) • [Add Profile](#-add-new-model-profile) • [LoRA](#-lora-adapter)

</div>

---

## 💡 Why?

여러 GPU가 있는 서버에서 vLLM으로 모델 서빙을 하다 보면...

- 이 모델 테스트하려고 올렸다가
- 저 모델로 바꾸려고 내렸다가
- 다시 다른 모델 올렸다가
- GPU 번호 뭐였지? 포트 뭐였지?

**너무 귀찮아서** 만들었습니다.

프로필 파일 하나 만들어두면 `./run.sh vlm up` 한 줄로 끝.

---

## 🎨 Structure

```
vllm-compose/
├── profiles/              # 모델별 프로필 (.env)
│   ├── vlm.env
│   ├── llm.env
│   └── clova.env
├── config/                # vLLM 설정 (YAML)
│   ├── qwen3-vl-30b-a3b-fp8.yaml
│   └── gpt-oss-120b.yaml
├── docker-compose.yaml
├── .env.common            # 공통 설정 (HF Token 등)
└── run.sh                 # 관리 스크립트
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

`.env.common` 파일 생성:

```bash
VLLM_VERSION=v0.13.0                   # vLLM 이미지 버전
HF_TOKEN=your_huggingface_token
HF_CACHE_PATH=~/.cache/huggingface
LORA_BASE_PATH=/path/to/lora/adapters  # LoRA 사용시
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
| `./run.sh list` | 프로필 목록 |
| `./run.sh {profile} up` | 컨테이너 시작 |
| `./run.sh {profile} up --dev` | 소스 빌드로 컨테이너 시작 |
| `./run.sh {profile} down` | 컨테이너 중지 |
| `./run.sh {profile} logs` | 로그 보기 |
| `./run.sh {profile} status` | 상태 확인 |
| `./run.sh build [branch]` | vLLM 소스 빌드 |
| `./run.sh ps` | 실행 중인 컨테이너 |
| `./run.sh gpu` | GPU 상태 |

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

## 🔨 Development Build

<details>
<summary><strong>Build vLLM from Source</strong></summary>

공식 릴리즈에 포함되지 않은 최신 기능이나 버그 수정이 필요할 때 사용합니다.

vLLM 공식 저장소를 clone하여 [공식 Dockerfile](https://github.com/vllm-project/vllm/tree/main/docker)로 빌드합니다.

### Build from main branch

```bash
./run.sh build
```

### Build from specific branch/tag

```bash
# 특정 브랜치
./run.sh build fix-lora-bug

# 특정 버전 태그
./run.sh build v0.8.0
```

### Run with dev build

```bash
./run.sh vlm up --dev
```

### How it works

1. `./run.sh build` 실행 시:
   - `.vllm-src/`에 vLLM 저장소 clone (이미 있으면 업데이트)
   - 공식 `docker/Dockerfile`로 빌드 (`--target vllm-openai`)
   - `vllm-dev:{branch}` 이미지 생성

2. `./run.sh {profile} up --dev` 실행 시:
   - 빌드된 `vllm-dev:{branch}` 이미지로 컨테이너 실행

### Configuration (.env.common)

```bash
# 빌드할 브랜치 지정 (optional)
VLLM_BRANCH=main
```

> **Note:** 첫 빌드는 30분 이상 소요될 수 있습니다.

</details>

---

## 🔗 LoRA Adapter

<details>
<summary><strong>LoRA Configuration</strong></summary>

### Step 1: Set Base Path (.env.common)

```bash
# .env.common
LORA_BASE_PATH=/home/user/models/lora-adapters
```

### Step 2: Configure Profile (profiles/*.env)

```bash
# profiles/vlm.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=adapter_name=/app/lora/adapter_folder
```

### Path Mapping (중요!)

로컬 경로가 컨테이너 내부 `/app/lora`로 마운트됩니다:

```
📁 Local (LORA_BASE_PATH)              📁 Container
/home/user/models/lora-adapters/   →   /app/lora/
├── my_adapter_v1/                     ├── my_adapter_v1/
├── my_adapter_v2/                     ├── my_adapter_v2/
└── project_adapter/                   └── project_adapter/
```

**예시:**
- 로컬: `/home/user/models/lora-adapters/my_adapter_v1`
- 컨테이너: `/app/lora/my_adapter_v1`
- LORA_MODULES 설정: `my_lora=/app/lora/my_adapter_v1`

⚠️ **주의:** LORA_MODULES의 경로는 반드시 `/app/lora/`로 시작해야 합니다!

### Multiple Adapters (여러 어댑터 사용)

쉼표(`,`)로 구분하여 여러 어댑터 등록:

```bash
# 단일 어댑터
LORA_MODULES=adapter1=/app/lora/my_adapter_v1

# 복수 어댑터
LORA_MODULES=adapter1=/app/lora/my_adapter_v1,adapter2=/app/lora/my_adapter_v2

# 실제 예시
LORA_MODULES=ocr_ko=/app/lora/qwen3vl_ocr_korean,ocr_en=/app/lora/qwen3vl_ocr_english
```

### API 호출 시 어댑터 선택

```python
# 기본 모델 사용
response = client.chat.completions.create(
    model="base-model-name",
    messages=[...]
)

# 특정 LoRA 어댑터 사용
response = client.chat.completions.create(
    model="adapter1",  # LORA_MODULES에서 정의한 이름
    messages=[...]
)
```

### Parameters

| Parameter | Description |
|----------|------|
| `ENABLE_LORA` | LoRA 활성화 (true/false) |
| `MAX_LORAS` | 동시 로드 가능한 최대 LoRA 수 |
| `MAX_LORA_RANK` | 최대 rank 값 |
| `LORA_MODULES` | name=/app/lora/path 형식, 쉼표로 구분 |

### VLM LoRA Limitation

- ✅ Language Model layers: Supported
- ❌ Vision Encoder layers: Not supported

### Model-Specific LoRA Support

일부 모델은 vLLM에서 LoRA를 지원하지 않을 수 있습니다. `SupportsLoRA` 인터페이스 구현 여부에 따라 결정됩니다.

| Model | LoRA Support | Note |
|-------|:------------:|------|
| Qwen2-VL, Qwen3-VL | ✅ | |
| LLaVA | ✅ | |
| DeepSeek-OCR | ✅ | v0.13.0 이후 버전 필요 ([PR #31569](https://github.com/vllm-project/vllm/pull/31569)) |

> **Note:** DeepSeek-OCR LoRA 지원은 2026년 1월 2일 main 브랜치에 머지되었습니다.
> v0.13.0에서는 지원되지 않으며, `latest` 또는 다음 릴리즈 버전이 필요합니다.
> ```bash
> # .env.common에서 버전 변경
> VLLM_VERSION=latest
> ```

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

**vLLM Compose** 

</div>
