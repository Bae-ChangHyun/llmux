#!/usr/bin/env bash
# switch.sh — 프로필 기동: GGUF 확인 → override 렌더 → compose up.
# (과거 single-active 시절 이름. llmux 에서는 기존 컨테이너 건드리지 않음.)
#
# 사용법:
#   ./scripts/llamacpp/switch.sh <profile-name>

source "$(dirname "$0")/_common.sh"

PROFILE=${1:?"사용법: switch.sh <profile-name>"}
require_env_common
require_profile "$PROFILE" > /dev/null

# shellcheck disable=SC1091
set -a; source "$ROOT/.env.common"; source "$PROFILES_DIR/${PROFILE}.env"; set +a

# 모델 파일 존재 확인 (없으면 다운로드)
if [[ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]]; then
  info "모델 파일 없음 → 다운로드 시도"
  "$SCRIPT_DIR/pull-model.sh" "$PROFILE"
fi

# override 렌더
info "command 렌더링"
python3 "$SCRIPT_DIR/render-override.py" "$PROFILE"

# 기동
info "'${PROFILE}' 프로필로 기동"
run_compose "$PROFILE" up -d

echo "$PROFILE" > "$CURRENT_PROFILE_FILE"

ok "프로필 '${PROFILE}' 활성화됨"
echo "  Endpoint: http://localhost:${LLAMA_PORT}/v1"
echo "  Health:   curl http://localhost:${LLAMA_PORT}/health"
echo "  Logs:     ./scripts/llamacpp/logs.sh"
