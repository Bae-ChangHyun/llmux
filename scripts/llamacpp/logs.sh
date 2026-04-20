#!/usr/bin/env bash
# logs.sh — 현재 활성 프로필 컨테이너 로그 follow.

source "$(dirname "$0")/_common.sh"

[[ -f "$CURRENT_PROFILE_FILE" ]] || die "활성 프로필 없음"
PROFILE=$(cat "$CURRENT_PROFILE_FILE")
require_profile "$PROFILE" > /dev/null

exec $(compose_cmd "$PROFILE") logs -f
