#!/usr/bin/env bash
# start.sh — 프로필 기동 (switch 의 얇은 별칭).
# 인자 없으면 .current-profile.llamacpp 에 기록된 마지막 프로필 사용.

source "$(dirname "$0")/_common.sh"

if [[ $# -ge 1 ]]; then
  PROFILE=$1
else
  [[ -f "$CURRENT_PROFILE_FILE" ]] || die "프로필 인자 필요 (또는 이전에 switch 한 이력 필요)"
  PROFILE=$(cat "$CURRENT_PROFILE_FILE")
fi

exec "$SCRIPT_DIR/switch.sh" "$PROFILE"
