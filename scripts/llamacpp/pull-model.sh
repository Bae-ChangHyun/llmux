#!/usr/bin/env bash
# pull-model.sh — HuggingFace 에서 GGUF 다운로드.
#
# 사용법:
#   ./scripts/llamacpp/pull-model.sh <profile-name>

source "$(dirname "$0")/_common.sh"

PROFILE=${1:?"사용법: pull-model.sh <profile-name>"}
# 프로필 이름은 파일명 stem 으로 사용되므로 path traversal 차단.
if [[ ! "$PROFILE" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$PROFILE" == *".."* ]]; then
  die "잘못된 프로필 이름: '$PROFILE' (허용: A-Z a-z 0-9 . _ -)"
fi
require_env_common
require_profile "$PROFILE" > /dev/null

# shellcheck disable=SC1091
set -a; source "$ROOT/.env.common"; source "$PROFILES_DIR/${PROFILE}.env"; set +a

: "${MODEL_DIR:?MODEL_DIR 미설정 (.env.common 확인)}"
: "${HF_REPO:?HF_REPO 미설정 (profiles/llamacpp/${PROFILE}.env 확인)}"
: "${HF_FILE:?HF_FILE 미설정 (profiles/llamacpp/${PROFILE}.env 확인)}"

mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$HF_FILE"

if [[ -f "$TARGET" ]]; then
  ok "이미 존재: $TARGET ($(du -h "$TARGET" | cut -f1))"
  exit 0
fi

command -v hf >/dev/null 2>&1 || die "hf CLI 필요. 'pip install -U huggingface_hub' 또는 'uv tool install huggingface_hub'."

info "다운로드: $HF_REPO / $HF_FILE → $MODEL_DIR"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

hf download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"
ok "완료: $TARGET"
