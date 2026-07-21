# llmux — notes for AI coding agents

llmux is a terminal multiplexer for LLM servers. One Textual TUI and one
headless CLI drive both **vLLM** (HF Transformers models) and **llama.cpp**
(GGUF) via Docker Compose. Profiles live in `profiles.yaml` and are rendered
to per-profile `.env` files at runtime.

## Prefer the CLI

For any scripted or agent-driven change, **use the CLI, not the TUI.** The CLI
mirrors every TUI capability and is the only interface usable headlessly.

Start from `llmux --help`. Top-level shortcuts and sub-apps:

- Container lifecycle: `llmux up <profile>`, `down`, `logs`, `ps`, `bench`, `render-env`
- Profiles: `llmux profile {list,show,new,edit,rename,clone,delete,quick-setup}`
- Configs: `llmux config {list,show,new,edit,clone,rename,delete}`
- Images: `llmux image {list,pull,remove,build-dev}`
- System: `llmux system {gpu,mem-estimate,disk,env-check}` (`gpu` / `env-check` also available top-level)

Every `list` / `show` / `check` command takes `--json` for machine-readable
output, and `llmux --version` prints the version — the CLI is meant to be
driven from scripts, agents, and CI. Running `llmux` with no arguments launches
the TUI (intended for humans).

## Repo layout (high-level)

- `tui/cli/` — Typer CLI (the headless entrypoint).
- `tui/screens/`, `tui/app.py` — Textual TUI.
- `tui/backends/{vllm,llamacpp}/` — backend runtimes; keep parity between them.
- `compose/{vllm,llamacpp}/` — Docker Compose stacks.
- `config/{vllm,llamacpp}/` — per-profile YAML overrides (gitignored; only `example.yaml` is tracked).
- `profiles.yaml`, `.env.common` — user-local, gitignored; copy from `profiles.example.yaml` / `.env.common.example` before first run.

## Conventions

- **Python env: `uv`** — `uv sync`, `uv run llmux …`, `uv run pytest`.
- **Commits: gitmoji + Conventional Commits** — `✨ feat(scope): subject`,
  `🐛 fix(...)`, `♻️ refactor(...)`, `📝 docs(...)`.
- **Branching: `git worktree`** — feature work happens on `feat/<thing>`
  branches in a worktree, never directly on `main`.
- **Parity rules:**
  - **TUI ⇄ CLI parity** — any feature added to one interface belongs in the other.
  - **vLLM ⇄ llama.cpp parity** — features should land on both backends unless the asymmetry is deliberate and documented.

## Per-developer notes

If you maintain personal scratch notes for your own AI agent (workflow
preferences, session handoff, in-flight TODOs), put them in `*.local.md`
files — `AGENTS.local.md`, `CLAUDE.local.md`, `HANDOFF.md` are all gitignored
by convention. Claude Code auto-loads `CLAUDE.local.md`; other AGENTS.md-aware
tools follow the same `.local.md` override pattern.
