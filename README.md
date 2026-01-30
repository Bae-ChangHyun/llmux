<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose Logo" width="280"/>

# vLLM Compose

**Docker Compose 기반 vLLM 멀티 모델 서빙 관리 도구**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Shell Script](https://img.shields.io/badge/Shell-Bash-4EAA25?style=flat-square&logo=gnu-bash)](run.sh)

[English](README_EN.md) | 한국어

[Quick Start](#-quick-start) • [Usage](#-usage) • [Add Profile](#-add-new-model-profile) • [LoRA](#-lora-adapter)

</div>

---

## 📖 소개

**vLLM Compose**는 Docker Compose 기반 vLLM 멀티 모델 서빙 관리 도구입니다.

서버에서 다양한 LLM/VLM 모델을 프로필 기반으로 손쉽게 관리하고, GPU 별 포트 매핑을 자동화하며, vLLM 소스 빌드까지 지원합니다.

### 💡 왜 만들었나요?

서버에서 vLLM으로 다양한 모델 서빙을 하다 보면...

- **문제:** 이 모델 테스트하려고 올렸다가, 저 모델로 바꾸려고 내렸다가, 다시 다른 모델 올렸다가... GPU 번호 뭐였지? 포트 뭐였지? 모델별 설정 뭐였지? 
- **해결:** 각 모델별 프로필 파일 하나 만들어두면 `./run.sh vlm up` 한 줄로 끝.

**너무 귀찮아서** 만들었습니다.

### 🎯 대상 사용자

하루에도 몇번씩 모델을 바꿔서 올렸다 내렸다가 하고, 방금 나온 따끈따끈한 모델들을 올려보고 싶은 사람

---

## ✨ 핵심 기능

### 🖥️ 인터랙티브 TUI 모드
* **메뉴 기반 인터페이스**: `./run.sh` 실행만으로 모든 기능을 GUI처럼 사용
* **직관적인 조작**: 프로필 선택, 컨테이너 관리, 빌드까지 메뉴에서 선택
* **CLI 호환**: TUI 없이 기존 명령어 방식도 그대로 지원

### 🔄 스마트 버전 관리
* **실시간 버전 조회**: Docker Hub API 연동으로 최신 릴리즈/Nightly 버전 자동 조회
* **인터랙티브 버전 선택**: TUI에서 Current/Official/Nightly/Dev/Custom 선택
* **자동 이미지 정리**: 새 버전 Pull 시 미사용 이전 이미지 자동 삭제
* **버전 추적**: 실행 중인 컨테이너의 실제 이미지 버전 및 생성일 표시

### 🚀 프로필 기반 멀티 모델 관리
* **간단한 프로필 시스템**: `.env` 파일로 모델별 설정을 독립적으로 관리
* **원클릭 실행**: `./run.sh {profile} up` 한 줄로 모델 서빙 시작
* **빠른 전환**: 다른 모델로 전환 시 프로필만 바꾸면 끝

### ⚡ GPU 별 포트 매핑 자동화
* **자동 포트 관리**: 각 GPU에 할당된 모델마다 독립적인 포트 자동 매핑
* **충돌 방지**: 프로필별 포트 설정으로 여러 모델 동시 실행 가능
* **명확한 관리**: `./run.sh gpu` 명령으로 GPU 상태 한눈에 확인

### 🔨 vLLM 소스 빌드 자동화
* **Fast Build**: 현재 PC의 GPU만 감지하여 빠르게 빌드 (10-30분)
* **공식 Dockerfile 사용**: vLLM 공식 빌드 프로세스 그대로 활용
* **버전 관리**: 특정 브랜치/태그 지정 빌드 지원

### 🔗 LoRA 어댑터 지원
* **멀티 어댑터**: 여러 LoRA 어댑터 동시 로드 및 API로 선택 사용
* **경로 자동 매핑**: 로컬 경로를 컨테이너 내부로 자동 마운트
* **다양한 모델 지원**: Qwen2-VL, Qwen3-VL, LLaVA, DeepSeek-OCR 등
---

## 🎨 프로젝트 구조

```
vllm-compose/
├── profiles/              # 모델별 프로필 (.env 파일)
│   ├── model1.env         
│   └── model2.env         
├── config/                # vLLM 설정 (YAML 파일)
│   ├── model1.yaml    
│   └── model2.yaml             
├── docker-compose.yaml    # Docker Compose 설정
├── .env.common            # 공통 환경 변수 (HF Token 등)
└── run.sh                 # 관리 스크립트 (프로필 실행, 빌드 등)
```

---

## 🚀 빠른 시작

### 사전 준비

```bash
docker --version        # Docker 설치 확인
docker compose version  # Docker Compose 설치 확인
nvidia-smi             # NVIDIA GPU 드라이버 확인
```

### 1. 저장소 클론

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git
cd vllm-compose
```

### 2. 공통 설정 파일 생성

`.env.common` 파일을 생성하고 아래 내용을 입력합니다.

```bash
VLLM_VERSION=latest                    # vLLM 이미지 버전 (latest/nightly/v0.15.0 등)
HF_TOKEN=your_huggingface_token        # Hugging Face 토큰
HF_CACHE_PATH=~/.cache/huggingface     # 모델 캐시 경로
LORA_BASE_PATH=/path/to/lora/adapters  # LoRA 어댑터 경로 (선택)
```

> **Tip**: `./run.sh version` 명령으로 사용 가능한 버전 확인 가능

### 3. 버전 정보 확인 (선택)

```bash
./run.sh version
```

출력 예시:
```
=== vLLM Version Information ===

Current Setting: vllm/vllm-openai:latest

Running Containers:
  - vlm: vllm/vllm-openai:v0.15.0 (ID: abc123, Created: 2026-01-20)

Available Versions:
  - Official Latest: v0.15.0
  - Nightly: Updated on 2026-01-29
  - Dev Builds: 1 local build(s)
```

### 4. 사용 가능한 프로필 확인

```bash
./run.sh list
```

출력 예시:
```
Available profiles:
  - vlm
  - llm
  - clova
```

### 5. 모델 서빙 시작

```bash
./run.sh vlm up
```

서빙이 시작되면 `http://localhost:8000` (프로필에서 설정한 포트)로 API 요청을 보낼 수 있습니다.

---

## 📖 사용 방법

vLLM Compose는 **TUI (인터랙티브) 모드**와 **CLI (명령어) 모드** 두 가지 방식을 지원합니다.

<details open>
<summary><strong>🖥️ TUI 모드 (인터랙티브)</strong></summary>

터미널에서 메뉴 기반 인터페이스로 모든 기능을 사용할 수 있습니다.

```bash
# TUI 모드 실행
./run.sh tui

# 또는 인자 없이 실행
./run.sh
```

> **요구사항**: `whiptail` 또는 `dialog` 설치 필요
> ```bash
> # Ubuntu/Debian
> sudo apt-get install whiptail
> ```

#### TUI 메인 메뉴

```
┌──────────────── vLLM Compose ─────────────────┐
│                                               │
│  1. 🚀 Start Container    컨테이너 시작        │
│  2. 🛑 Stop Container     컨테이너 중지        │
│  3. 📋 View Logs          로그 보기           │
│  4. 📊 Container Status   컨테이너 상태        │
│  5. 🎮 GPU Status         GPU 상태 확인        │
│  6. 🔨 Build vLLM         vLLM 소스 빌드       │
│  7. 🗂️  Manage Images      이미지 관리         │
│  8. ❌ Exit               종료                │
│                                               │
└───────────────────────────────────────────────┘
```

#### 주요 기능

| 메뉴 | 설명 |
|:---:|:---|
| Start Container | 프로필 선택 → 버전 선택 (Current/Official/Nightly/Dev/Custom) → 컨테이너 시작 |
| Stop Container | 실행 중인 컨테이너 선택 → 중지 및 삭제 |
| View Logs | 컨테이너 선택 → 실시간 로그 표시 |
| GPU Status | nvidia-smi 출력을 팝업으로 표시 |
| Build vLLM | 브랜치 입력 → Fast/Official 빌드 선택 |
| Manage Images | vllm-dev 이미지 목록 조회 및 삭제 |
| Version Info | 현재/최신 버전 정보 및 실행 중인 컨테이너 버전 표시 |

</details>

<details>
<summary><strong>⌨️ CLI 모드 (명령어)</strong></summary>

터미널에서 직접 명령어를 입력하여 사용합니다.

#### 명령어 목록

| 명령어 | 설명 |
|:---:|:---|
| `./run.sh list` | 사용 가능한 프로필 목록 및 상태 표시 |
| `./run.sh version` | vLLM 버전 정보 (Current/Latest/Nightly/Dev) 조회 |
| `./run.sh {profile} up` | 지정된 프로필로 컨테이너 시작 |
| `./run.sh {profile} up --dev` | 소스 빌드한 vLLM 이미지로 컨테이너 시작 |
| `./run.sh {profile} down` | 지정된 프로필의 컨테이너 중지 및 삭제 |
| `./run.sh {profile} logs` | 컨테이너 로그 실시간 보기 |
| `./run.sh {profile} status` | 컨테이너 상태 확인 |
| `./run.sh build [branch]` | vLLM 소스 빌드 (기본: main 브랜치) |
| `./run.sh images` | 빌드된 vllm-dev 이미지 목록 |
| `./run.sh ps` | 현재 실행 중인 모든 컨테이너 목록 |
| `./run.sh gpu` | GPU 상태 및 사용량 확인 |

#### 사용 예시

```bash
# 프로필 목록 확인
./run.sh list

# 버전 정보 확인 (현재/최신/Nightly)
./run.sh version

# VLM 모델 시작
./run.sh vlm up

# 로그 확인
./run.sh vlm logs

# GPU 상태 확인
./run.sh gpu

# 컨테이너 중지
./run.sh vlm down
```

#### 직접 Docker Compose 명령 사용하기

run.sh 스크립트 없이 Docker Compose를 직접 사용할 수도 있습니다.

```bash
# 컨테이너 시작
docker compose --env-file .env.common --env-file profiles/vlm.env -p vlm up -d

# 컨테이너 중지
docker compose -p vlm down

# 로그 보기
docker logs -f vlm
```

</details>

---

## 🔧 새로운 모델 프로필 추가하기

새로운 모델을 추가하려면 설정 파일과 프로필 파일을 생성하면 됩니다.

### 1. 모델 설정 파일 생성 (config/)

`config/` 디렉토리에 YAML 파일을 생성합니다.

```yaml
# config/my-model.yaml
model: huggingface/model-name      # Hugging Face 모델 경로
max-model-len: 32768               # 최대 컨텍스트 길이
gpu-memory-utilization: 0.8        # GPU 메모리 사용률 (0.0-1.0)
```

### 2. 프로필 파일 생성 (profiles/)

`profiles/` 디렉토리에 `.env` 파일을 생성합니다.

```bash
# profiles/mymodel.env
CONTAINER_NAME=mymodel             # 컨테이너 이름
VLLM_PORT=8003                     # API 서빙 포트
CONFIG_NAME=my-model               # config/ 디렉토리의 YAML 파일명 (확장자 제외)

GPU_ID=0                           # 사용할 GPU 번호
TENSOR_PARALLEL_SIZE=1             # Tensor Parallelism 크기 (멀티 GPU 사용시)

# LoRA 어댑터 (선택 사항)
ENABLE_LORA=false
```

### 3. 모델 실행

```bash
./run.sh mymodel up
```

### 프로필 설정 항목 설명

| 설정 항목 | 설명 | 예시 |
|:---|:---|:---|
| `CONTAINER_NAME` | Docker 컨테이너 이름 | `mymodel` |
| `VLLM_PORT` | vLLM API 서빙 포트 | `8000`, `8001` |
| `CONFIG_NAME` | `config/` 디렉토리의 YAML 파일명 (확장자 제외) | `my-model` |
| `GPU_ID` | 사용할 GPU 번호 (단일 GPU) | `0`, `1` |
| `TENSOR_PARALLEL_SIZE` | Tensor Parallelism 크기 (멀티 GPU) | `1`, `2`, `4` |
| `ENABLE_LORA` | LoRA 어댑터 활성화 여부 | `true`, `false` |

---

## 🔨 개발 빌드

<details>
<summary><strong>vLLM 소스에서 직접 빌드하기</strong></summary>

공식 릴리즈에 포함되지 않은 최신 기능이나 버그 수정이 필요할 때 vLLM을 소스에서 직접 빌드할 수 있습니다.

### Fast Build (권장)

[공식 Dockerfile](https://github.com/vllm-project/vllm/tree/main/docker)을 사용하되, **자동으로 현재 PC의 GPU를 감지**하여 해당 아키텍처만 빌드합니다.

```bash
# main 브랜치 빌드
./run.sh build

# 특정 브랜치/태그 빌드
./run.sh build v0.8.0
./run.sh build fix-some-bug
```

출력 예시:
```
Detected GPU: NVIDIA GeForce RTX 4080 SUPER (sm_8.9)
Building for your GPU only - MUCH faster!
```

> vLLM 공식 Dockerfile을 그대로 사용하므로, vLLM 업데이트 시 자동 반영됩니다.

### Official Build (모든 GPU 지원)

공식 Dockerfile로 모든 GPU 아키텍처용 빌드. 다른 서버에 배포할 이미지가 필요할 때 사용합니다.

```bash
./run.sh build --official
./run.sh build v0.8.0 --official
```

> **경고**: 공식 빌드는 **수 시간**이 소요될 수 있습니다.

### 개발 빌드로 컨테이너 실행

```bash
./run.sh vlm up --dev
```

### 빌드 시간 비교

| 빌드 방식 | 소요 시간 | 용도 |
|----------|----------|------|
| Fast (기본) | 10-30분 | 로컬 테스트, 개발 |
| Official | 3-6시간 | 배포용 이미지 |

### 빌드 설정 (.env.common)

```bash
# 빌드할 브랜치 지정 (선택 사항)
VLLM_BRANCH=main
```

</details>

---

## 🔗 LoRA 어댑터

<details>
<summary><strong>LoRA 어댑터 설정 및 사용</strong></summary>

LoRA (Low-Rank Adaptation) 어댑터를 사용하면 기본 모델을 fine-tuning한 여러 버전을 동시에 서빙할 수 있습니다.

### 1단계: 베이스 경로 설정 (.env.common)

```bash
# .env.common
LORA_BASE_PATH=/home/user/models/lora-adapters
```

### 2단계: 프로필 설정 (profiles/*.env)

```bash
# profiles/vlm.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=adapter_name=/app/lora/adapter_folder
```

### 경로 매핑 (중요!)

로컬 경로가 컨테이너 내부 `/app/lora`로 자동 마운트됩니다.

```
📁 로컬 (LORA_BASE_PATH)                  📁 컨테이너
/home/user/models/lora-adapters/   →   /app/lora/
├── my_adapter_v1/                     ├── my_adapter_v1/
├── my_adapter_v2/                     ├── my_adapter_v2/
└── project_adapter/                   └── project_adapter/
```

**예시:**
- 로컬: `/home/user/models/lora-adapters/my_adapter_v1`
- 컨테이너: `/app/lora/my_adapter_v1`
- LORA_MODULES 설정: `my_lora=/app/lora/my_adapter_v1`

**주의**: LORA_MODULES의 경로는 반드시 `/app/lora/`로 시작해야 합니다!

### 여러 어댑터 사용하기

쉼표(`,`)로 구분하여 여러 어댑터를 등록할 수 있습니다.

```bash
# 단일 어댑터
LORA_MODULES=adapter1=/app/lora/my_adapter_v1

# 복수 어댑터
LORA_MODULES=adapter1=/app/lora/my_adapter_v1,adapter2=/app/lora/my_adapter_v2

# 실제 사용 예시
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

### 설정 파라미터

| 파라미터 | 설명 |
|----------|------|
| `ENABLE_LORA` | LoRA 활성화 여부 (`true`/`false`) |
| `MAX_LORAS` | 동시에 로드 가능한 최대 LoRA 어댑터 수 |
| `MAX_LORA_RANK` | 최대 rank 값 |
| `LORA_MODULES` | `name=/app/lora/path` 형식, 쉼표로 구분 |

### VLM LoRA 제한 사항

- ✅ **Language Model layers**: 지원됨
- ❌ **Vision Encoder layers**: 지원되지 않음

### 모델별 LoRA 지원 현황

일부 모델은 vLLM에서 LoRA를 지원하지 않을 수 있습니다. `SupportsLoRA` 인터페이스 구현 여부에 따라 결정됩니다.

| 모델 | LoRA 지원 | 비고 |
|-------|:------------:|------|
| Qwen2-VL, Qwen3-VL | ✅ | |
| LLaVA | ✅ | |
| DeepSeek-OCR | ✅ | v0.13.0 이후 버전 필요 ([PR #31569](https://github.com/vllm-project/vllm/pull/31569)) |

> **참고**: DeepSeek-OCR LoRA 지원은 2026년 1월 2일 main 브랜치에 머지되었습니다.
> v0.13.0에서는 지원되지 않으며, `latest` 또는 다음 릴리즈 버전이 필요합니다.
> ```bash
> # .env.common에서 버전 변경
> VLLM_VERSION=latest
> ```

</details>

---

## 🐛 문제 해결

<details>
<summary><strong>컨테이너가 시작되지 않아요</strong></summary>

로그를 확인하여 문제를 파악할 수 있습니다.

```bash
# 컨테이너 로그 확인
./run.sh {profile} logs

# GPU 상태 확인
./run.sh gpu

# 컨테이너 상태 확인
./run.sh {profile} status
```

일반적인 원인:
- GPU 메모리 부족
- Hugging Face 토큰 오류
- 모델 다운로드 실패
- 설정 파일 오류

</details>

<details>
<summary><strong>GPU 메모리 부족 (OOM) 오류</strong></summary>

`config/{model}.yaml` 파일에서 `gpu-memory-utilization` 값을 낮춰보세요.

```yaml
# config/my-model.yaml
gpu-memory-utilization: 0.7  # 기본값 0.8에서 0.7로 변경
```

또는 더 작은 모델을 사용하거나, Tensor Parallelism으로 여러 GPU에 분산할 수 있습니다.

```bash
# profiles/mymodel.env
TENSOR_PARALLEL_SIZE=2  # 2개의 GPU 사용
```

</details>

<details>
<summary><strong>포트 충돌 오류</strong></summary>

프로필 파일에서 `VLLM_PORT`를 다른 포트로 변경하세요.

```bash
# profiles/mymodel.env
VLLM_PORT=8001  # 8000에서 8001로 변경
```

사용 중인 포트 확인:

```bash
# Linux/Mac
sudo lsof -i :8000

# 실행 중인 컨테이너 확인
./run.sh ps
```

</details>

<details>
<summary><strong>LoRA 어댑터를 찾을 수 없어요</strong></summary>

경로 설정을 확인하세요.

1. `.env.common`에서 `LORA_BASE_PATH`가 올바른지 확인
2. 프로필의 `LORA_MODULES` 경로가 `/app/lora/`로 시작하는지 확인
3. 로컬 경로에 실제 어댑터 파일이 있는지 확인

```bash
# 로컬 경로 확인
ls -la /home/user/models/lora-adapters/

# 컨테이너 내부 경로 확인 (컨테이너 실행 중)
docker exec -it {container_name} ls -la /app/lora/
```

</details>

<details>
<summary><strong>모델 다운로드가 느려요</strong></summary>

Hugging Face 미러를 사용하거나, 캐시 경로를 SSD에 설정하세요.

```bash
# .env.common
HF_CACHE_PATH=/path/to/fast/ssd/cache
```

또는 모델을 미리 다운로드해둘 수 있습니다.

```python
from huggingface_hub import snapshot_download
snapshot_download("model-name", cache_dir="/path/to/cache")
```

</details>

---

## 💻 기술 스택

- **컨테이너화**: Docker, Docker Compose
- **모델 서빙**: [vLLM](https://github.com/vllm-project/vllm)
- **GPU 가속**: NVIDIA CUDA
- **자동화**: Bash Shell Script

---

## 📄 라이선스

MIT License로 배포됩니다.

---

<div align="center">

**vLLM Compose**<br/>
Made with ❤️ for AI Developers and Researchers

</div>
