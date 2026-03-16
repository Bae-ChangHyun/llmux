<div align="center">

<img src="assets/vllm-compose.png" alt="vLLM Compose" width="240"/>

# vLLM Compose

**여러 LLM을 올렸다 내렸다, 터미널 하나로 관리하세요.**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-Latest-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[English](README.md) | **한국어**

---

Qwen 테스트하다가 Llama로 바꾸고, DeepSeek 잠깐 띄웠다가 다시 내리고...
<br/>
**모델마다 설정 파일 따로, 컨테이너 따로, 전부 기억하고 있어야 하죠.**
<br/><br/>
vLLM Compose는 모델별 설정을 프로필로 저장하고,
<br/>
**TUI에서 선택만 하면 바로 올리고 내릴 수 있습니다.**

</div>

<br/>

## 30초 안에 시작하기

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git && cd vllm-compose

# HuggingFace 토큰 설정
cat > .env.common << 'EOF'
HF_TOKEN=your_token_here
HF_CACHE_PATH=~/.cache/huggingface
EOF

# 실행 (의존성 자동 설치)
uv run vllm-compose    # 또는 ./run.sh
```

> Quick Setup에서 모델명만 입력하면 프로필 + 설정이 자동 생성됩니다.

<br/>

## 왜 vLLM Compose인가?

| | 직접 관리 | vLLM Compose |
|:---|:---|:---|
| **모델 전환** | docker 명령어 반복, 설정 다시 입력 | 프로필 선택 후 Enter |
| **설정 관리** | 긴 CLI 인자를 매번 기억 | 모델별 YAML + Tab 자동완성 |
| **멀티 모델** | compose 파일 수동 편집 | 프로필별 독립 관리, 동시 실행 |
| **상태 확인** | docker ps, nvidia-smi 반복 | 대시보드에서 실시간 확인 |
| **버전 선택** | 이미지 태그 직접 관리 | Latest / Official / Nightly 선택 |

<br/>

## 핵심 기능

**TUI** &mdash; 모델 시작/중지/로그/설정을 한 화면에서 관리

**프로필** &mdash; 모델별 설정을 독립 저장, 언제든 한 번에 전환

**Config** &mdash; vLLM 파라미터를 YAML로 관리, 51개 파라미터 Tab 자동완성

**소스 빌드** &mdash; GPU 자동 감지 Fast Build (10-30분), Fork 빌드 지원

**LoRA** &mdash; 멀티 어댑터 동시 로드, 경로 자동 매핑

<br/>

---

<details>
<summary><b>TUI 키보드 단축키</b></summary>

<br/>

| 키 | 기능 |
|:---|:---|
| `Enter` | 프로필 액션 메뉴 (시작/중지/로그/편집/삭제) |
| `w` | Quick Setup |
| `n` | 새 프로필 |
| `F1` `F2` `F3` | Dashboard / Configs / System |
| `?` | 전체 단축키 도움말 |

</details>

<details>
<summary><b>CLI 사용법</b></summary>

<br/>

```bash
./run.sh list                    # 프로필 목록
./run.sh {profile} up            # 시작
./run.sh {profile} down          # 중지
./run.sh {profile} logs          # 로그
./run.sh build                   # 소스 빌드
./run.sh build --repo <url>      # Fork 빌드
```

</details>

<details>
<summary><b>프로필 & Config 구조</b></summary>

<br/>

```yaml
# config/my-model.yaml — vLLM 서빙 설정
model: Qwen/Qwen3-30B
gpu-memory-utilization: 0.9
max-model-len: 32768
trust-remote-code: true
```

```bash
# profiles/my-model.env — 컨테이너 설정
CONTAINER_NAME=my-model
VLLM_PORT=8000
CONFIG_NAME=my-model
GPU_ID=0
TENSOR_PARALLEL_SIZE=1
ENABLE_LORA=false
```

</details>

<details>
<summary><b>소스 빌드</b></summary>

<br/>

```bash
# Fast Build — 현재 GPU만 대상, 10-30분
./run.sh build                              # main
./run.sh build v0.15.0                      # 특정 버전
./run.sh build main --repo <fork-url>       # Fork

# Official Build — 모든 GPU, 3-6시간
./run.sh build --official

# Dev 이미지로 실행
./run.sh mymodel up --dev
```

</details>

<details>
<summary><b>LoRA 어댑터</b></summary>

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
    model="ko",  # LoRA 어댑터명
    messages=[...]
)
```

</details>

<details>
<summary><b>문제 해결</b></summary>

<br/>

| 문제 | 해결 |
|:---|:---|
| 컨테이너 미시작 | `./run.sh {profile} logs` 로 로그 확인 |
| GPU OOM | `gpu-memory-utilization: 0.7` 또는 `TENSOR_PARALLEL_SIZE=2` |
| 포트 충돌 | `VLLM_PORT` 변경 후 `./run.sh ps` 확인 |
| vLLM 인자 추가 | `config/*.yaml`에 아무 CLI 인자나 YAML로 작성 |

</details>

---

<div align="center">

**MIT License** · Made for AI Developers

</div>
