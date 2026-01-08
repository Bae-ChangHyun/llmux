<div align="center">

<img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" alt="vLLM Logo" width="300"/>

# vLLM Compose

**Docker Compose 기반 vLLM 멀티 모델 서빙 관리 도구**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-v0.13.0-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)

[빠른 시작](#-빠른-시작) • [사용법](#-사용법) • [프로필 추가](#-새-모델-프로필-추가하기) • [LoRA 설정](#-lora-adapter-설정)

</div>

---

## 💡 왜 만들었나?

여러 GPU가 있는 서버에서 vLLM으로 모델 서빙을 하다 보면...

- 이 모델 테스트하려고 올렸다가
- 저 모델로 바꾸려고 내렸다가
- 다시 다른 모델 올렸다가
- GPU 번호 뭐였지? 포트 뭐였지?

**너무 귀찮아서** 만들었습니다.

프로필 파일 하나 만들어두면 `./run.sh vlm up` 한 줄로 끝.

---

## 🎨 구조

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

## 🚀 빠른 시작

### 사전 요구사항

```bash
docker --version
docker compose version
nvidia-smi
```

### 1. 저장소 클론

```bash
git clone https://github.com/Bae-ChangHyun/vllm-compose.git
cd vllm-compose
```

### 2. 공통 설정

`.env.common` 파일 생성:

```bash
HF_TOKEN=your_huggingface_token
HF_CACHE_PATH=~/.cache/huggingface
LORA_BASE_PATH=/path/to/lora/adapters  # LoRA 사용시
```

### 3. 프로필 확인

```bash
./run.sh list
```

### 4. 모델 서빙

```bash
./run.sh vlm up
```

---

## 📖 사용법

### run.sh 명령어

| 명령어 | 설명 |
|:---:|:---|
| `./run.sh list` | 프로필 목록 |
| `./run.sh {profile} up` | 컨테이너 시작 |
| `./run.sh {profile} down` | 컨테이너 중지 |
| `./run.sh {profile} logs` | 로그 보기 |
| `./run.sh {profile} status` | 상태 확인 |
| `./run.sh ps` | 실행 중인 컨테이너 |
| `./run.sh gpu` | GPU 상태 |

### Docker Compose 직접 사용

```bash
# 시작
docker compose --env-file .env.common --env-file profiles/vlm.env -p vlm up -d

# 중지
docker compose -p vlm down

# 로그
docker logs -f vlm
```

---

## 🔧 새 모델 프로필 추가하기

### 1. 모델 설정 (config/)

```yaml
# config/my-model.yaml
model: huggingface/model-name
max-model-len: 32768
gpu-memory-utilization: 0.8
```

### 2. 프로필 생성 (profiles/)

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

### 3. 실행

```bash
./run.sh mymodel up
```

---

## 🔗 LoRA Adapter 설정

<details>
<summary><strong>LoRA 설정 방법</strong></summary>

### 프로필에 LoRA 설정

```bash
# profiles/vlm.env
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=ocr-li=/app/lora/qwen3vl_li/lora_model
```

### 파라미터

| 파라미터 | 설명 |
|----------|------|
| `ENABLE_LORA` | LoRA 활성화 (true/false) |
| `MAX_LORAS` | 동시 LoRA 수 |
| `MAX_LORA_RANK` | 최대 rank |
| `LORA_MODULES` | name=path 형식, 쉼표 구분 |

### 경로 매핑

```
호스트: $LORA_BASE_PATH/qwen3vl_li/lora_model
컨테이너: /app/lora/qwen3vl_li/lora_model
```

### VLM LoRA 제한

- ✅ Language Model 레이어: 지원
- ❌ Vision Encoder 레이어: 미지원

</details>

---

## 🐛 문제 해결

<details>
<summary><strong>컨테이너 시작 안됨</strong></summary>

```bash
./run.sh {profile} logs
./run.sh gpu
```

</details>

<details>
<summary><strong>GPU 메모리 부족</strong></summary>

`config/{model}.yaml`에서 `gpu-memory-utilization` 값 낮추기

</details>

<details>
<summary><strong>포트 충돌</strong></summary>

프로필의 `VLLM_PORT` 변경

</details>

---

<div align="center">

**vLLM Compose** - 모델 바꾸기 귀찮을 때

<img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="Docker" width="80"/>

</div>
