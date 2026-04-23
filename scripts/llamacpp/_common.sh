# llamacpp 전용 공용 함수. 다른 스크립트에서 source.
# 프로필 정의는 repo root 의 profiles.yaml 에 단일화되어 있고,
# compose 에는 `.runtime/llamacpp/<name>.env` 를 렌더해서 넘긴다.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_DIR="$ROOT/.runtime/llamacpp"
CONFIG_DIR="$ROOT/config/llamacpp"
COMPOSE_DIR="$ROOT/compose/llamacpp"
PROFILES_YAML="$ROOT/profiles.yaml"
CURRENT_PROFILE_FILE="$ROOT/.current-profile.llamacpp"

die() { echo "✗ $*" >&2; exit 1; }
info() { echo "▸ $*" >&2; }
ok()  { echo "✓ $*" >&2; }

require_env_common() {
  [[ -f "$ROOT/.env.common" ]] || die ".env.common 없음. 'cp .env.common.example .env.common' 후 값 수정."
}

render_profile() {
  # YAML 의 프로필을 런타임 .env 로 렌더. 경로를 stdout 으로 출력.
  local profile=${1:?프로필 이름 필요}
  local py
  py="$(command -v python3 || true)"
  [[ -n "$py" ]] || die "python3 를 찾을 수 없음"
  ( cd "$ROOT" && "$py" -m tui.common.profile_store render llamacpp "$profile" ) \
    || die "프로필 렌더 실패: llamacpp/$profile (profiles.yaml 확인)"
}

require_profile() {
  # 레거시 호환: 호출자가 .env 경로를 받아서 source 하는 패턴.
  local profile=${1:?프로필 이름 필요}
  render_profile "$profile"
}

run_compose() {
  # 안전판: 인자 분리 유지. profile 이름 + compose 서브명령/옵션을 넘긴다.
  local profile=${1:?프로필 이름 필요}
  shift
  local env_file
  env_file="$(render_profile "$profile")"
  docker compose \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$COMPOSE_DIR/docker-compose.override.yaml" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$env_file" \
    "$@"
}

exec_compose() {
  local profile=${1:?프로필 이름 필요}
  shift
  local env_file
  env_file="$(render_profile "$profile")"
  exec docker compose \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$COMPOSE_DIR/docker-compose.override.yaml" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$env_file" \
    "$@"
}
