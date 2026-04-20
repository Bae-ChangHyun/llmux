#!/usr/bin/env bash
# stop.sh — 현재 활성 프로필 컨테이너 중지.

source "$(dirname "$0")/_common.sh"

[[ -f "$CURRENT_PROFILE_FILE" ]] || die "활성 프로필 없음 (.current-profile.llamacpp 부재)"

PROFILE=$(cat "$CURRENT_PROFILE_FILE")
require_profile "$PROFILE" > /dev/null

info "'${PROFILE}' 중지"
$(compose_cmd "$PROFILE") down

ok "중지 완료"
