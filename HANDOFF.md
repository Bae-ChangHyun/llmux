# HANDOFF — llm-compose

> 다음 세션 시작 시 이 문서를 먼저 읽어주세요. 프로젝트 목적/배경/설계 방향이 모두 여기에 있습니다.

---

## 프로젝트 한 줄 정의

**`llm-compose`** — 실행 시 "vLLM 으로 띄울까, llama.cpp 로 띄울까?" 를 물어본 뒤, 각각의 기존 TUI (`vllm-compose`, `llamacpp-compose`) 로 라우팅하는 **런처 (dispatcher)**.

## 왜 필요한가 (배경)

사용자는 현재 두 개의 독립된 TUI 도구를 운용 중:

| 프로젝트 | 목적 | 위치 | GitHub |
|---|---|---|---|
| **vllm-compose** | vLLM (HF Transformers 기반) 서빙 프로필 관리 | `/home/bch/Project/docker/vllm-compose` | `Bae-ChangHyun/vllm-compose` (public) |
| **llamacpp-compose** | llama.cpp (GGUF) 서빙 프로필 관리 | `/home/bch/Project/docker/llamacpp-compose` | https://github.com/Bae-ChangHyun/llamacpp-compose (public, 2026-04-17 공개) |

두 도구는 **별도 폴더 / 별도 entry command** 를 가지고 있어서, 모델을 띄우려고 할 때마다 "오늘 vLLM 용 폴더로 갈지, llama.cpp 용으로 갈지" 를 머리로 기억해야 함. 이게 귀찮다.

→ **하나의 진입점** 을 두고, TUI 첫 화면에서 백엔드를 고르면 나머지는 각 도구의 기존 UI 그대로 쓰자.

## 명시적 non-goals (중요)

다음은 **하지 않는다:**

1. **모듈 통합 X** — 두 프로젝트의 코드를 하나로 머지하지 않는다. 각자 독립 repo / 독립 진화.
2. **공통 추상화 X** — "Backend" 인터페이스를 뽑아서 추상화하지 않는다. YAGNI.
3. **기능 중복 구현 X** — 프로필 CRUD, Docker Compose 래퍼 등은 이미 각 프로젝트에 있다. 재구현 금지.
4. **상태 보관 X** — 런처는 stateless 하게. 사용자 선택 기억 기능 (마지막 백엔드 자동 선택 등) 은 **v1 범위 밖**.

사용자 직접 인용:
> "굳이 모든 모듈을 통합할필요없어. 내가 나중에 구현하고자하는건 그냥 ui만 하나인거고 거기서 맨처음에 vllm인자 llamacpp인지 선택하면 그 이후에는 지금이랑 똑같이 각각의 기능대로 사용하면돼"

## 설계 방향 (제안)

**~50 줄짜리 얇은 dispatcher.** 의존성 최소.

### 옵션 A — 순수 프롬프트 (가장 단순)
```python
#!/usr/bin/env python3
# llm-compose
import os, sys
BACKENDS = {
    "1": ("vLLM",     "/home/bch/Project/docker/vllm-compose",     ["uv", "run", "python", "-m", "tui"]),
    "2": ("llama.cpp","/home/bch/Project/docker/llamacpp-compose", ["uv", "run", "python", "-m", "tui"]),
}
print("1) vLLM\n2) llama.cpp")
choice = input("> ").strip()
name, cwd, cmd = BACKENDS[choice]
os.chdir(cwd)
os.execvp(cmd[0], cmd)   # 현재 프로세스 교체 → TUI가 완전히 올라감
```

### 옵션 B — Textual 시작 화면
Textual `ModalScreen` 으로 두 버튼만 있는 화면. 선택 후 `app.exit()` → 부모 shell script 가 해당 백엔드 `uv run` 실행.

### 옵션 C — 심볼릭 진입점
`/usr/local/bin/llm-compose` 에 bash 스크립트 배치:
```bash
#!/usr/bin/env bash
select choice in "vLLM" "llama.cpp" "quit"; do
  case $choice in
    vLLM)      cd ~/Project/docker/vllm-compose     && exec uv run python -m tui ;;
    llama.cpp) cd ~/Project/docker/llamacpp-compose && exec uv run python -m tui ;;
    quit)      exit 0 ;;
  esac
done
```

**추천**: 옵션 A 또는 C. Textual 띄우기 위해 런처 자체가 Textual 의존성을 갖는 건 과함.

## 엔트리 동작 검증용 체크리스트

구현 후 확인해야 할 것:

- [ ] `llm-compose` 명령으로 선택 화면이 뜬다
- [ ] vLLM 선택 → 기존 vllm-compose TUI 가 아무 변화 없이 뜬다
- [ ] llama.cpp 선택 → 기존 llamacpp-compose TUI 가 아무 변화 없이 뜬다
- [ ] TUI 종료 후 원래 shell 로 깨끗하게 복귀한다 (좀비 프로세스 없음)
- [ ] 두 백엔드 TUI 내부의 Docker Compose 동작에 **영향 없음** (컨테이너 이름 / 네트워크 / 볼륨 충돌 없음)

## 새 세션 시작 시 할 일

1. 이 문서 (`HANDOFF.md`) 와 `CLAUDE.local.md` 읽기
2. 두 참조 프로젝트 상태 확인
   ```bash
   ls /home/bch/Project/docker/vllm-compose/tui/
   ls /home/bch/Project/docker/llamacpp-compose/tui/
   ```
3. 각 프로젝트의 **실제 entry command 확인** — 위 설계의 `uv run python -m tui` 는 추정값임. 실제 방식이 `./run.sh` 든 `python main.py` 든 둘 다 같은 패턴인지 먼저 확인할 것.
4. 사용자에게 옵션 A/B/C 중 선택 확인 후 Plan → Confirm → Execute.

## 참고 자료

- `llamacpp-compose` 의 TUI 구조: `tui/app.py`, `tui/screens/` 하위 (dashboard/config/profile/quick_setup/logs)
- `vllm-compose` 의 TUI 구조: 비슷한 레이아웃으로 기대됨 (같은 저자)
- 두 프로젝트 모두 `uv` 기반 Python 환경 사용
