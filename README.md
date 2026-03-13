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

**Textual 기반 모던 TUI로 복잡한 설정 없이 쉽게 시작하세요!**

---

## ✨ 핵심 기능

🖥️ **Textual TUI** - Python 기반 모던 TUI로 모든 기능을 GUI처럼 사용

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

# Textual TUI 사용 시 (선택, 권장)
pip install textual
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
├── tui/                   # Textual TUI (Python)
│   ├── app.py             # 메인 앱 (진입점)
│   ├── app.tcss           # 글로벌 스타일
│   ├── backend.py         # Docker/프로필/설정 I/O
│   └── screens/           # 화면 모듈
│       ├── dashboard.py   # 메인 대시보드
│       ├── container.py   # 시작/중지/로그
│       ├── profile.py     # 프로필 CRUD
│       ├── config.py      # 설정 CRUD
│       ├── system.py      # GPU/이미지/컨테이너 정보
│       └── quick_setup.py # 빠른 설정
├── profiles/              # 모델별 프로필 (.env)
├── config/                # vLLM 설정 (YAML)
├── lib/                   # Bash 유틸리티 (fallback TUI)
├── scripts/
│   └── entrypoint-wrapper.sh
├── docker-compose.yaml
├── .env.common            # 공통 설정
├── pyproject.toml         # Python 의존성
└── run.sh                 # 관리 스크립트
```

---

## 📚 상세 가이드

<details>
<summary><strong>🖥️ TUI 모드 상세</strong></summary>

### 설치 및 실행

```bash
# Textual 설치
pip install textual

# TUI 실행
./run.sh
```

> Textual이 없으면 자동으로 기존 whiptail/dialog TUI로 fallback됩니다.

### 키보드 단축키

#### 전역

| 키 | 기능 |
|:---|:---|
| `F1` | Dashboard |
| `F2` | Config 관리 |
| `F3` | System 정보 |
| `q` | 종료 |
| `?` | 도움말 |

#### Dashboard

| 키 | 기능 |
|:---|:---|
| `u` | 선택한 프로필 시작 |
| `d` | 선택한 프로필 중지 |
| `l` | 실시간 로그 보기 |
| `n` | 새 프로필 생성 |
| `e` | 프로필 편집 |
| `x` | 프로필 삭제 |
| `w` | Quick Setup (모델→프로필+설정 자동생성) |
| `c` | Config 목록 |
| `s` | System 정보 (GPU/이미지/컨테이너) |
| `r` | 새로고침 |

### 주요 화면

- **Dashboard** - 프로필 목록 + 실행 상태 (5초마다 자동갱신)
- **Container Up** - 버전 선택 (Local/Official/Nightly/Dev/Custom) 후 시작
- **Log Viewer** - 실시간 컨테이너 로그 스트리밍
- **System Info** - GPU 상태 (3초 갱신), Docker 이미지, 컨테이너 탭 뷰

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
| `./run.sh build [branch] --repo <url>` | 커스텀 repo/fork에서 빌드 |
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

### 커스텀 Repository (Fork)

다른 사람의 fork나 자신의 repo에서 빌드

```bash
# 커스텀 repo
./run.sh build main --repo https://github.com/username/vllm.git

# 커스텀 repo + 특정 브랜치
./run.sh build custom-feature --repo https://github.com/username/vllm.git
```

### Official Build

모든 GPU 아키텍처 지원 (3-6시간)

```bash
./run.sh build --official
./run.sh build v0.15.0 --official
```

### 빌드 정보 조회

빌드한 이미지의 상세 정보 확인 (repo, branch, commit, 날짜)

```bash
./run.sh images
```

출력:
```
Tag: vllm-dev:main-20260130
  Size: 42GB | Created: 2026-01-30 15:30
  Repository: vllm-project/vllm
  Branch: main | Commit: abc1234
  Built: 2026-01-30 | Type: fast
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
<summary><strong>🐛 문제 해결 & FAQ</strong></summary>

### vLLM 추가 인자는 어디에 설정하나요?

모든 vLLM 인자는 `config/*.yaml` 파일에 추가하면 됩니다.

```yaml
# config/mymodel.yaml
model: huggingface/model-name
gpu-memory-utilization: 0.9
max-model-len: 32768

# 추가 vLLM 인자들
trust-remote-code: true
dtype: bfloat16
max-num-seqs: 256
enable-chunked-prefill: true
```

> vLLM의 모든 CLI 인자를 YAML 형식으로 작성 가능
> 참고: [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server)

TUI에서 **Config Mgmt → Edit Config**에서 커스텀 파라미터를 추가/수정/삭제할 수 있습니다.

### 추가 pip 패키지 설치

컨테이너 시작 시 추가 Python 패키지를 설치하려면 `.env.common`에 설정:

```bash
# .env.common
EXTRA_PIP_PACKAGES=transformers==4.50.0 flash-attn
```

> `docker-compose.extra-packages.yaml`을 통해 entrypoint에서 자동 설치됩니다.

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
- [Textual](https://github.com/Textualize/textual) (Python TUI)
- Bash Shell Script (CLI + fallback TUI)

---

## 📄 License

MIT License

---

<div align="center">

**vLLM Compose**
Made with ❤️ for AI Developers

</div>
