#!/usr/bin/env bash
# stop.sh — 지정 프로필(또는 .current-profile.llamacpp) 의 컨테이너 중지.
# llmux 는 여러 llama.cpp 컨테이너 동시 실행을 허용하므로 항상 프로필 이름을 받는 편이 안전하다.
#
# 사용법:
#   ./scripts/llamacpp/stop.sh <profile-name>   # 권장 (명시적)
#   ./scripts/llamacpp/stop.sh                  # 인자 없으면 .current-profile.llamacpp 폴백

source "$(dirname "$0")/_common.sh"

if [[ $# -ge 1 ]]; then
  PROFILE=$1
else
  [[ -f "$CURRENT_PROFILE_FILE" ]] || die "프로필 인자 필요 (또는 이전에 switch 한 이력 필요)"
  PROFILE=$(cat "$CURRENT_PROFILE_FILE")
fi

ENV_FILE="$(render_profile "$PROFILE")"
# shellcheck disable=SC1091
set -a; source "$ROOT/.env.common"; source "$ENV_FILE"; set +a

info "'${PROFILE}' 중지"
if [[ -f "$(_override_path "$PROFILE")" ]]; then
  run_compose "$PROFILE" down
else
  info "override 파일이 없어 docker container 이름으로 직접 중지합니다"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
  else
    info "컨테이너 없음: $CONTAINER_NAME"
  fi
fi

ok "중지 완료"
