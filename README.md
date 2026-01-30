<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose Logo" width="280"/>

# vLLM Compose

**Docker Compose 기반 vLLM 멀티 모델 서빙 관리 도구**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[English](README_EN.md) | 한국어

</div>

---

## 💡 Why?

서버에서 vLLM으로 모델 서빙을 하다 보면...

- 이 모델 테스트하려고 올렸다가, 저 모델로 바꾸려고 내렸다가...
- GPU 번호 뭐였지? 포트 뭐였지? 설정 뭐였지?

**너무 귀찮아서** 만들었습니다.

**TUI 메뉴 기반 인터페이스로 복잡한 설정 없이 쉽게 시작하세요!**

---

## ✨ 핵심 기능

🖥️ **TUI 인터랙티브 모드** - 메뉴 기반으로 모든 기능을 GUI처럼 사용

🔄 **스마트 버전 관리** - Docker Hub 연동, Official/Nightly 자동 조회 및 정리

🚀 **프로필 기반 관리** - `.env` 파일로 모델별 설정을 독립 관리

⚡ **GPU 자동 매핑** - GPU별 포트 자동 할당 및 충돌 방지

🔨 **소스 빌드** - GPU 자동 감지로 10-30분 빌드 (Fast Build)

🔗 **LoRA 지원** - 멀티 어댑터 동시 로드 및 자동 경로 매핑

---

## 🚀 Quick Start

### 사전 준비

```bash
docker --version        # Docker 설치 확인
docker compose version  # Docker Compose 설치 확인
nvidia-smi             # NVIDIA GPU 확인
```

### Clone & 공통 설정

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git
cd vllm-compose

# .env.common 파일 생성
cat > .env.common << EOF
VLLM_VERSION=latest
HF_TOKEN=your_huggingface_token
HF_CACHE_PATH=~/.cache/huggingface
EOF
```

---

### 방법 1: TUI 사용 (추천)

**메뉴 기반으로 모든 작업을 쉽게 수행**

```bash
./run.sh
```

1. **Quick Setup** 선택
2. 모델명 입력 (예: `Qwen/Qwen3-VL-30B`)
3. GPU ID, 포트 입력
4. 자동으로 프로필+설정 생성
5. 컨테이너 시작 메뉴에서 바로 실행!

> TUI 메뉴에서 프로필 생성/수정/삭제, 컨테이너 관리, 버전 선택 등 모든 작업 가능

---

### 방법 2: CLI 사용

**터미널 명령어로 직접 제어**

#### 1) 프로필 생성

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

#### 2) 실행

```bash
./run.sh mymodel up
```

---

## 🎨 프로젝트 구조

```
vllm-compose/
├── profiles/              # 모델별 프로필 (.env)
│   ├── model1.env
│   └── model2.env
├── config/                # vLLM 설정 (YAML)
│   ├── model1.yaml
│   └── model2.yaml
├── docker-compose.yaml
├── .env.common            # 공통 설정
└── run.sh                 # 관리 스크립트
```

---

## 📚 상세 가이드

<details>
<summary><strong>🖥️ TUI 모드 상세</strong></summary>

### TUI 실행

```bash
./run.sh        # 또는
./run.sh tui
```

> **요구사항**: `whiptail` 또는 `dialog` 필요
> ```bash
> sudo apt-get install whiptail
> ```

### 메인 메뉴

```
┌──────────────── vLLM Compose ─────────────────┐
│  Q. Quick Setup         프로필+Config 자동생성  │
│  1. Container Mgmt      시작/중지/로그/상태     │
│  2. Profile Mgmt        프로필 생성/수정/삭제   │
│  3. Config Mgmt         설정 생성/수정          │
│  4. Build Mgmt          소스 빌드/이미지 관리   │
│  5. System Info         GPU/버전/컨테이너 정보  │
│  X. Exit                종료                   │
└───────────────────────────────────────────────┘
```

### 버전 선택 메뉴

컨테이너 시작 시 버전을 선택할 수 있습니다:

```
1. Current running: vllm/vllm-openai:nightly (...)
2. Official Latest: v0.15.0
3. Nightly: 2026-01-29
4. Dev build (local source builds)
5. Custom tag
```

- **Official/Nightly 선택 시**: 최신 버전 자동 pull + 미사용 이전 이미지 자동 정리

</details>

<details>
<summary><strong>⌨️ CLI 명령어 전체 목록</strong></summary>

### 프로필 관리

| 명령어 | 설명 |
|:---|:---|
| `./run.sh list` | 프로필 목록 및 상태 |
| `./run.sh {profile} up` | 컨테이너 시작 |
| `./run.sh {profile} up --dev` | Dev 빌드로 시작 |
| `./run.sh {profile} down` | 컨테이너 중지 |
| `./run.sh {profile} logs` | 로그 보기 (실시간) |
| `./run.sh {profile} status` | 상태 확인 |

### 버전 & 이미지

| 명령어 | 설명 |
|:---|:---|
| `./run.sh version` | 버전 정보 조회 |
| `./run.sh images` | Dev 빌드 이미지 목록 |

### 빌드

| 명령어 | 설명 |
|:---|:---|
| `./run.sh build` | main 브랜치 Fast Build |
| `./run.sh build [branch]` | 특정 브랜치 빌드 |
| `./run.sh build --official` | Official Build (모든 GPU) |
| `./run.sh build [branch] --tag TAG` | 커스텀 태그로 빌드 |

### 시스템

| 명령어 | 설명 |
|:---|:---|
| `./run.sh ps` | 실행 중인 컨테이너 |
| `./run.sh gpu` | GPU 상태 |

</details>

<details>
<summary><strong>🔧 새 모델 프로필 추가</strong></summary>

### 방법 1: Quick Setup (TUI)

```bash
./run.sh
# → "Quick Setup" 선택
# → 모델명, GPU, 포트 입력
# → 자동 생성
```

### 방법 2: 수동 생성

#### 1) Config 파일 생성

```yaml
# config/mymodel.yaml
model: huggingface/model-name
gpu-memory-utilization: 0.9
max-model-len: 32768
```

#### 2) Profile 파일 생성

```bash
# profiles/mymodel.env
CONTAINER_NAME=mymodel
VLLM_PORT=8003
CONFIG_NAME=mymodel

GPU_ID=0
TENSOR_PARALLEL_SIZE=1

ENABLE_LORA=false
```

#### 3) 실행

```bash
./run.sh mymodel up
```

### 주요 설정

| 설정 | 설명 | 예시 |
|:---|:---|:---|
| `CONTAINER_NAME` | 컨테이너 이름 | `mymodel` |
| `VLLM_PORT` | API 서빙 포트 | `8000` |
| `CONFIG_NAME` | Config 파일명 (확장자 제외) | `mymodel` |
| `GPU_ID` | GPU 번호 | `0` or `0,1` |
| `TENSOR_PARALLEL_SIZE` | TP 크기 | `1`, `2`, `4` |

</details>

<details>
<summary><strong>🔨 vLLM 소스 빌드</strong></summary>

### Fast Build (권장)

현재 PC의 GPU만 감지하여 빠르게 빌드 (10-30분)

```bash
./run.sh build              # main 브랜치
./run.sh build v0.15.0      # 특정 버전
./run.sh build my-branch    # 특정 브랜치
```

출력:
```
Detected GPU: NVIDIA RTX 4080 (sm_8.9)
Building for your GPU only - MUCH faster!
```

### Official Build

모든 GPU 아키텍처 지원 (3-6시간)

```bash
./run.sh build --official
./run.sh build v0.15.0 --official
```

### Dev 빌드로 실행

```bash
./run.sh mymodel up --dev
./run.sh mymodel up --dev --tag main-20260130
```

</details>

<details>
<summary><strong>🔗 LoRA 어댑터</strong></summary>

### 1. 베이스 경로 설정

```bash
# .env.common
LORA_BASE_PATH=/home/user/lora-adapters
```

### 2. 프로필 설정

```bash
# profiles/mymodel.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=adapter1=/app/lora/my_adapter_v1
```

### 3. 경로 매핑

```
로컬: /home/user/lora-adapters/my_adapter_v1
  ↓ 자동 마운트
컨테이너: /app/lora/my_adapter_v1
```

⚠️ LORA_MODULES는 반드시 `/app/lora/`로 시작

### 4. 여러 어댑터

```bash
LORA_MODULES=ko=/app/lora/ko_adapter,en=/app/lora/en_adapter
```

### 5. API 호출

```python
# 특정 어댑터 사용
response = client.chat.completions.create(
    model="ko",  # LORA_MODULES의 이름
    messages=[...]
)
```

### 지원 모델

| 모델 | LoRA 지원 |
|:---|:---:|
| Qwen2-VL, Qwen3-VL | ✅ |
| LLaVA | ✅ |
| DeepSeek-OCR | ✅ (v0.13.0+) |

</details>

<details>
<summary><strong>🐛 문제 해결</strong></summary>

### 컨테이너가 시작되지 않음

```bash
./run.sh {profile} logs     # 로그 확인
./run.sh gpu                # GPU 상태
./run.sh {profile} status   # 컨테이너 상태
```

### GPU 메모리 부족 (OOM)

```yaml
# config/mymodel.yaml
gpu-memory-utilization: 0.7  # 낮추기
```

또는 TP 사용:
```bash
# profiles/mymodel.env
TENSOR_PARALLEL_SIZE=2
```

### 포트 충돌

```bash
# profiles/mymodel.env
VLLM_PORT=8001  # 변경
```

확인:
```bash
sudo lsof -i :8000
./run.sh ps
```

### LoRA 경로 오류

1. `.env.common`의 `LORA_BASE_PATH` 확인
2. 프로필의 `LORA_MODULES`가 `/app/lora/`로 시작하는지 확인
3. 로컬 경로에 실제 파일 존재 확인

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
