#!/usr/bin/env bash
# benchmark.sh — 표준 프롬프트로 /v1/chat/completions 호출, tok/s 계산.
#
# 사용법:
#   ./scripts/llamacpp/benchmark.sh              # 활성 프로필 기준
#   ./scripts/llamacpp/benchmark.sh <profile>    # 특정 프로필 포트 사용

source "$(dirname "$0")/_common.sh"

PROFILE=${1:-}
if [[ -z "$PROFILE" ]]; then
  [[ -f "$CURRENT_PROFILE_FILE" ]] || die "프로필 인자 필요"
  PROFILE=$(cat "$CURRENT_PROFILE_FILE")
fi
require_profile "$PROFILE" > /dev/null

# shellcheck disable=SC1091
set -a; source "$ROOT/.env.common"; source "$PROFILES_DIR/${PROFILE}.env"; set +a

PORT=${LLAMA_PORT:-8080}

info "벤치마크 시작: http://localhost:${PORT}"

for _ in $(seq 1 60); do
  curl -sf "http://localhost:${PORT}/health" > /dev/null && break
  sleep 0.5
done

PORT="$PORT" MODEL="${CONFIG_NAME:-default}" PROFILE="$PROFILE" python3 - <<'PY'
import json, os, time, urllib.request

PORT = os.environ["PORT"]
MODEL = os.environ["MODEL"]
PROFILE = os.environ["PROFILE"]

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Explain the theory of relativity in about 150 words."}],
    "max_tokens": 200,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()

req = urllib.request.Request(
    f"http://localhost:{PORT}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    raw = r.read().decode()
elapsed = time.time() - t0

d = json.loads(raw, strict=False)
u = d.get("usage", {})
pt = u.get("prompt_tokens", 0)
ct = u.get("completion_tokens", 0)
tps = ct / elapsed if elapsed > 0 else 0

bar = "─" * 40
print(bar)
print(f" Profile:        {PROFILE}")
print(f" Model:          {MODEL}")
print(f" Prompt tokens:  {pt}")
print(f" Gen tokens:     {ct}")
print(f" Elapsed:        {elapsed:.2f}s")
print(f" Throughput:     {tps:.1f} tok/s")
print(bar)
PY
