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

<br/>

<img src="assets/demo.gif" alt="vLLM Compose Demo" width="700"/>

</div>

<br/>

## 30초 안에 시작하기

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git && cd vllm-compose

# HuggingFace 토큰 설정
cat > .env.common << 'EOF'
HF_TOKEN=your_token_here
HF_CACHE_PATH=/home/your-username/.cache/huggingface  # 절대 경로 필수
EOF

# 실행
uv sync
uv run vllm-compose
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
| **버전 선택** | 이미지 태그 직접 관리 | Latest / Official / Nightly 선택, 버전 태그로 pull |

<br/>

## 핵심 기능

**TUI** &mdash; Textual 전용 인터페이스로 모델 시작/중지/로그/설정을 관리

**프로필** &mdash; 모델별 설정을 독립 저장, 언제든 한 번에 전환

**Config** &mdash; vLLM 파라미터를 YAML로 관리, 로컬 vLLM 이미지에서 자동 추출한 파라미터 Tab 자동완성

**GPU 모니터** &mdash; 대시보드에서 실시간 GPU 사용량 바, 5초 자동 갱신

**메모리 추정** &mdash; [hf-mem](https://github.com/alvarobartt/hf-mem) 연동으로 배포 전 GPU 메모리 사전 확인, GPU별 progress bar 표시

**소스 빌드** &mdash; TUI에서 GPU 자동 감지 Fast Build (10-30분)

**LoRA** &mdash; 활성화된 경우에만 경로를 마운트하는 멀티 어댑터 로드

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
| `m` | 메모리 추정기 (검색바 포커스) |
| `s` | 시스템 정보 |
| `c` | Config 관리 |
| `u` / `d` / `l` | 시작 / 중지 / 로그 (선택된 프로필) |
| `?` | 전체 단축키 도움말 |

</details>

<details>
<summary><b>프로필 & Config 구조</b></summary>

<br/>

```yaml
# config/my-model.yaml — vLLM 서빙 설정
model: Qwen/Qwen3-30B
gpu-memory-utilization: 0.9
max-model-len: 32768
enable-auto-tool-choice: true   # boolean 플래그: 빈 값 → 자동 true 변환
```

```bash
# profiles/my-model.env — 컨테이너 설정
CONTAINER_NAME=my-model
VLLM_PORT=8000
CONFIG_NAME=my-model
GPU_ID=0
TENSOR_PARALLEL_SIZE=1
ENABLE_LORA=false
MODEL_ID=Qwen/Qwen3-30B   # 선택 사항, 기본 config 자동 생성 시 사용
```

</details>

<details>
<summary><b>소스 빌드</b></summary>

<br/>

```bash
# TUI 실행
uv run vllm-compose
```

이후 프로필 화면에서 `Dev Build`를 선택하고 시작하면 됩니다.

vLLM Compose가 자동으로:
- vLLM 소스 트리를 clone/update 하고
- 현재 GPU 기준 `vllm-dev:<branch>` 이미지를 빌드한 뒤
- 선택한 프로필을 dev 이미지로 실행합니다

기본 소스 브랜치는 `.env.common`의 `VLLM_BRANCH`로 지정할 수 있습니다.

Dev Build 화면에서 repository URL과 branch를 직접 바꿔 custom fork로 빌드할 수도 있습니다.

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

> LoRA 마운트는 `ENABLE_LORA=true`일 때만 추가됩니다.

</details>

<details>
<summary><b>문제 해결</b></summary>

<br/>

| 문제 | 해결 |
|:---|:---|
| 컨테이너 미시작 | TUI에서 해당 프로필의 로그를 열어 확인 |
| API를 로컬에서만 열고 싶음 | 기본 바인드는 `127.0.0.1:${VLLM_PORT}` 이며, 원격 공개가 필요하면 프록시 뒤에 두는 것을 권장 |
| GPU OOM | `gpu-memory-utilization: 0.7` 또는 `TENSOR_PARALLEL_SIZE=2` |
| 포트 충돌 | `VLLM_PORT`를 변경한 뒤 다시 시작 |
| Distilled 모델 토크나이저 에러 | config YAML에 `tokenizer: 원본Org/원본Model` 추가 |
| 추가 Python 패키지가 필요함 | 프로필에 `EXTRA_PIP_PACKAGES`를 설정하고 버전을 신중하게 pinning |
| vLLM 인자 추가 | `config/*.yaml`에 아무 CLI 인자나 YAML로 작성 |
| TUI 로그 복사 | Shift+드래그로 선택, Ctrl+C로 복사 ([상세](https://textual.textualize.io/FAQ/#how-can-i-select-and-copy-text-in-a-textual-app)) |

</details>

---

## 요구사항

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.10+ (TUI용)
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU

---

## 로드맵

- [ ] **모델별 추천 Config** — 모델 제조사/커뮤니티 권장 vLLM 설정 (Llama, Qwen, DeepSeek, Gemma 등) `max-model-len`, `quantization`, `rope-scaling` 등 사전 설정 제공
- [ ] **Config 프리셋/템플릿** — Quick Setup 시 모델 크기와 GPU 용량에 맞는 설정 자동 추천
- [ ] **.env.common 설정 위자드** — 첫 실행 시 HF 토큰, 캐시 경로 대화형 설정
- [ ] **API 연결 테스트** — `/v1/models` 호출로 모델 서빙 상태 즉시 확인
- [ ] **프로필 복제** — 한 클릭으로 프로필 복사하여 A/B 테스트
- [ ] **일괄 작업** — 여러 컨테이너 동시 시작/중지
- [ ] **내보내기/가져오기** — 프로필 + Config 번들을 서버 간 공유
- [ ] **웹 UI** — 팀 환경을 위한 브라우저 기반 대시보드

---

<div align="center">

**MIT License**

</div>
