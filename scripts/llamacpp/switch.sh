#!/usr/bin/env bash
# switch.sh — 프로필 전환: 현재 컨테이너 중지 → 렌더 → 새 프로필로 기동.
#
# 사용법:
#   ./scripts/llamacpp/switch.sh <profile-name>

source "$(dirname "$0")/_common.sh"

PROFILE=${1:?"사용법: switch.sh <profile-name>"}
require_env_common
require_profile "$PROFILE" > /dev/null

# 이전 프로필 있으면 down
if [[ -f "$CURRENT_PROFILE_FILE" ]]; then
  PREV=$(cat "$CURRENT_PROFILE_FILE")
  if [[ -f "$PROFILES_DIR/${PREV}.env" ]]; then
    info "이전 프로필 '${PREV}' 중지"
    $(compose_cmd "$PREV") down 2>/dev/null || true
  fi
fi

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
$(compose_cmd "$PROFILE") up -d

echo "$PROFILE" > "$CURRENT_PROFILE_FILE"

ok "프로필 '${PROFILE}' 활성화됨"
echo "  Endpoint: http://localhost:${LLAMA_PORT}/v1"
echo "  Health:   curl http://localhost:${LLAMA_PORT}/health"
echo "  Logs:     ./scripts/llamacpp/logs.sh"
