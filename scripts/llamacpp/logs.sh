#!/usr/bin/env bash
# logs.sh — 지정 프로필(또는 .current-profile.llamacpp)의 컨테이너 로그 follow.
#
# 사용법:
#   ./scripts/llamacpp/logs.sh <profile-name>   # 권장
#   ./scripts/llamacpp/logs.sh                  # 인자 없으면 .current-profile.llamacpp 폴백

source "$(dirname "$0")/_common.sh"

if [[ $# -ge 1 ]]; then
  PROFILE=$1
else
  [[ -f "$CURRENT_PROFILE_FILE" ]] || die "프로필 인자 필요"
  PROFILE=$(cat "$CURRENT_PROFILE_FILE")
fi
require_profile "$PROFILE" > /dev/null

exec_compose "$PROFILE" logs -f
