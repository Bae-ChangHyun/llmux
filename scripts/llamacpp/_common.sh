# llamacpp 전용 공용 함수. 다른 스크립트에서 source.
# 새 구조 (llmux 통합 후) 기준 경로:
#   scripts/llamacpp/_common.sh  ← 여기
#   profiles/llamacpp/*.env
#   config/llamacpp/*.yaml
#   compose/llamacpp/docker-compose*.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILES_DIR="$ROOT/profiles/llamacpp"
CONFIG_DIR="$ROOT/config/llamacpp"
COMPOSE_DIR="$ROOT/compose/llamacpp"
CURRENT_PROFILE_FILE="$ROOT/.current-profile.llamacpp"

die() { echo "✗ $*" >&2; exit 1; }
info() { echo "▸ $*" >&2; }
ok()  { echo "✓ $*" >&2; }

require_env_common() {
  [[ -f "$ROOT/.env.common" ]] || die ".env.common 없음. 'cp .env.common.example .env.common' 후 값 수정."
}

require_profile() {
  local profile=${1:?프로필 이름 필요}
  local path="$PROFILES_DIR/${profile}.env"
  [[ -f "$path" ]] || die "프로필 없음: profiles/llamacpp/${profile}.env"
  echo "$path"
}

run_compose() {
  # 안전판: 인자 분리 유지. profile 이름 + compose 서브명령/옵션을 넘긴다.
  local profile=${1:?프로필 이름 필요}
  shift
  docker compose \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$COMPOSE_DIR/docker-compose.override.yaml" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$PROFILES_DIR/${profile}.env" \
    "$@"
}

exec_compose() {
  local profile=${1:?프로필 이름 필요}
  shift
  exec docker compose \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$COMPOSE_DIR/docker-compose.override.yaml" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$PROFILES_DIR/${profile}.env" \
    "$@"
}
