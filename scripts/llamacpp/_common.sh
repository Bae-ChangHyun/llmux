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

_override_path() {
  local profile=${1:?프로필 이름 필요}
  echo "$ROOT/.runtime/llamacpp/override-${profile}.yaml"
}

validate_container_start() {
  local container=${1:?컨테이너 이름 필요}
  local port=${2:?포트 필요}
  local deadline=$((SECONDS + 45))
  local status health

  while (( SECONDS < deadline )); do
    if ! state=$(docker inspect "$container" --format '{{.State.Status}}	{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>&1); then
      die "컨테이너 상태 확인 실패: $state"
    fi
    status="${state%%	*}"
    health="${state#*	}"

    if [[ "$status" == "exited" || "$status" == "dead" || "$status" == "restarting" || "$health" == "unhealthy" ]]; then
      docker logs --tail 80 "$container" >&2 || true
      die "컨테이너가 시작 중 실패했습니다 (status=${status}, health=${health})"
    fi
    if [[ "$status" != "running" ]]; then
      die "컨테이너가 실행 상태가 아닙니다 (status=${status})"
    fi
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  info "컨테이너는 실행 중이지만 /health 가 아직 준비되지 않았습니다. 로그를 확인하세요."
  return 0
}

run_compose() {
  # Per-profile project (-p) + per-profile override file so multiple
  # llamacpp profiles can run concurrently without overwriting each other's
  # command or fighting for the shared `llama-server` service slot.
  local profile=${1:?프로필 이름 필요}
  shift
  local env_file override
  env_file="$(render_profile "$profile")"
  override="$(_override_path "$profile")"
  [[ -f "$override" ]] || die "override 파일 없음: $override (render-override.py 를 먼저 실행하세요)"
  docker compose \
    -p "$profile" \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$override" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$env_file" \
    "$@"
}

exec_compose() {
  local profile=${1:?프로필 이름 필요}
  shift
  local env_file override
  env_file="$(render_profile "$profile")"
  override="$(_override_path "$profile")"
  [[ -f "$override" ]] || die "override 파일 없음: $override (render-override.py 를 먼저 실행하세요)"
  exec docker compose \
    -p "$profile" \
    -f "$COMPOSE_DIR/docker-compose.yaml" \
    -f "$override" \
    --project-directory "$ROOT" \
    --env-file "$ROOT/.env.common" \
    --env-file "$env_file" \
    "$@"
}
