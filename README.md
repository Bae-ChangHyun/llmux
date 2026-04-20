# llm-compose

Unified Textual TUI that launches and manages both **vLLM** and **llama.cpp** model servers
as Docker Compose profiles from a single dashboard.

Absorbs the functionality of two upstream projects and runs them side-by-side:

- [`vllm-compose`](https://github.com/Bae-ChangHyun/vllm-compose) — vLLM (HF Transformers) profiles
- [`llamacpp-compose`](https://github.com/Bae-ChangHyun/llamacpp-compose) — llama.cpp (GGUF) profiles

## Status

In active development. Phase 1 scaffolding.

## Quick start

```bash
uv sync
uv run llm-compose
```

## Layout

```
profiles/{vllm,llamacpp}/*.env       # backend별 프로필
config/{vllm,llamacpp}/*.yaml        # backend별 서버 config
compose/{vllm,llamacpp}/docker-compose*.yaml
scripts/{vllm,llamacpp}/             # backend별 쉘 유틸
models/                              # GGUF 저장소 (llama.cpp 볼륨)
tui/                                 # Textual TUI
  backends/{vllm,llamacpp}/          # backend별 runtime/storage 로직
  screens/                           # 통합 UI
```

## License

MIT
