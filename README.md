# vLLM Container Manager

vLLM을 docker로 서빙할때 여러 모델을 손쉽게 바꿔가며 서빙하기 위한 도구입니다.

## Usage

### Method 1: run.sh 스크립트 사용

```bash
# 프로필 목록 보기
./run.sh list

# 컨테이너 시작
./run.sh {profile} up

# 컨테이너 중지
./run.sh {profile} down

# 로그 확인
./run.sh {profile} logs

# 컨테이너 상태 확인
./run.sh {profile} status

# 실행 중인 모든 vLLM 컨테이너 보기
./run.sh ps

# GPU 사용량 확인
./run.sh gpu
```

### 방법 2: docker compose 직접 사용

스크립트 없이 docker compose 명령어로 직접 실행할 수 있습니다. <br>
{profile}에 원하는 프로필명을 넣으세요

```bash
# 시작 (--env-file 두 번 사용)
docker compose --env-file .env.common --env-file profiles/{profile}.env -p {profile} up -d

# 중지
docker compose -p {profile} down

# 로그 확인
docker logs -f vlm
```

**LoRA 사용 시:**
```bash
# LORA_OPTIONS 환경변수 설정 후 실행
export LORA_OPTIONS="--enable-lora --max-loras 2 --max-lora-rank 16 --lora-modules ocr-li=/app/lora/qwen3vl_li/lora_model"
docker compose --env-file .env.common --env-file profiles/vlm.env -p vlm up -d
```

## 새 모델 추가하기

1. `config/` 디렉토리에 모델 설정 yaml 추가
2. `profiles/` 디렉토리에 프로필 env 파일 생성
   - 네이밍: `{NAME}.env` (파일명이 곧 프로필명)


---

<details>
<summary>LoRA Adapter 상세 설정</summary>

### LoRA 설정 방법

프로필 파일에서 LoRA 설정을 활성화합니다:

```bash
# profiles/vlm.env

# LoRA Configuration
ENABLE_LORA=true
MAX_LORAS=2
MAX_LORA_RANK=16
LORA_MODULES=ocr-li=/app/lora/qwen3vl_li/lora_model,ocr-jk=/app/lora/qwen3vl_jk/lora_model
```

### LoRA 설정 파라미터

| 파라미터 | 설명 |
|----------|------|
| `ENABLE_LORA` | LoRA 지원 활성화 (true/false) |
| `MAX_LORAS` | 동시 처리할 최대 LoRA 수 |
| `MAX_LORA_RANK` | 허용할 최대 LoRA rank |
| `LORA_MODULES` | LoRA 모듈 목록 (name=path 형식, 쉼표 구분) |

### LoRA 경로 규칙

- `.env.common`의 `LORA_BASE_PATH`가 컨테이너의 `/app/lora`에 마운트됩니다
- `LORA_MODULES`에서는 `/app/lora/` 하위 경로를 사용합니다
- 예: 호스트의 `/root/working/ocr_finetune/models/finetuned/qwen3vl_li/lora_model`
  → 컨테이너에서 `/app/lora/qwen3vl_li/lora_model`

### VLM LoRA 제한사항

vLLM의 VLM LoRA 지원에는 제한이 있습니다:
- Language Model 레이어 LoRA: **지원됨**
- Vision Encoder 레이어 LoRA: **미지원**

Vision 레이어도 파인튜닝한 경우, `merge_and_unload()`로 병합한 모델을 사용하세요.

</details>
