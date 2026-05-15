# Using the TUI

The TUI is llmux's default interface — a single [Textual](https://textual.textualize.io/)
dashboard that manages vLLM **and** llama.cpp profiles side by side. Everything
the [headless CLI](../reference/cli.md) does is available here as keyboard-driven
screens and forms.

## Launching

```bash
llmux          # no arguments → launches the TUI
llmux tui      # explicit
```

Running `llmux` with no subcommand drops you straight into the unified
dashboard.

## The dashboard

The dashboard is one `DataTable` showing every profile from both backends.
Running containers float to the top so the active workload is always visible
without scrolling; within the running / stopped groups, rows stay sorted by
`(backend, name)`.

```
┌ llmux ─ vLLM + llama.cpp ─────────────────────────────────────────┐
│  vLLM 1/2  ·  llama.cpp 0/1  ·  Enter = actions                   │
├───────────────────────────────────────────────────────────────────┤
│ Backend    Profile      Status      Port   Model        Detail    │
│ vLLM       qwen3-8b     ● running   8000   Qwen3-8B     TP=1      │
│ llama.cpp  gemma-3-4b   ○ stopped   8080   gemma-3-4b   —         │
│ vLLM       qwen3-0-6b   ○ stopped   8001   Qwen3-0.6B   —         │
└───────────────────────────────────────────────────────────────────┘
 GPU0 ████████░░░░ 64%  12.1/24GB
 🔍  Estimate HF model memory (press m then type, Enter to run)
```

A status bar shows running / total counts per backend. A GPU bar below the
table auto-refreshes every few seconds, and the table itself re-scans every 5
seconds.

## Keyboard shortcuts

Move the cursor with <kbd>↑</kbd> / <kbd>↓</kbd>, then act on the selected row.

### Dashboard

| Key | Action | Notes |
| --- | --- | --- |
| <kbd>Enter</kbd> | Open the action menu for the selected profile | Context-aware (running vs. stopped) |
| <kbd>n</kbd> | New profile | Opens the backend picker, then Quick Setup |
| <kbd>m</kbd> | Memory estimator | Focuses the HF model search input |
| <kbd>s</kbd> | System info | For the selected row's backend (or asks) |
| <kbd>r</kbd> | Refresh | Re-scan all profiles and container status |
| <kbd>q</kbd> | Quit | |
| <kbd>u</kbd> | Start container | Power-user shortcut (hidden from footer) |
| <kbd>d</kbd> | Stop container | Power-user shortcut |
| <kbd>l</kbd> | View logs | Running containers only |
| <kbd>e</kbd> | Edit profile | Power-user shortcut |
| <kbd>c</kbd> | Edit config | Power-user shortcut |
| <kbd>x</kbd> | Delete profile | Stopped containers only |
| <kbd>?</kbd> | Help | Quick key reference popup |

The <kbd>u</kbd> / <kbd>d</kbd> / <kbd>l</kbd> / <kbd>e</kbd> / <kbd>c</kbd> /
<kbd>x</kbd> keys are deliberately hidden from the footer — they do exactly the
same thing as picking the matching item from the <kbd>Enter</kbd> action menu.

### App-level

| Key | Action |
| --- | --- |
| <kbd>q</kbd> | Quit |
| <kbd>F1</kbd> | Return to the dashboard |
| <kbd>?</kbd> | Help |

### Log / startup viewers

| Key | Action |
| --- | --- |
| <kbd>f</kbd> | Toggle auto-follow on/off |
| <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>PgUp</kbd> / <kbd>PgDn</kbd> | Scroll |
| <kbd>q</kbd> / <kbd>Esc</kbd> | Back / close |

## The action menu

Pressing <kbd>Enter</kbd> on a profile opens a context menu. The options depend
on whether the container is running:

| State | Available actions |
| --- | --- |
| **Running** | Stop Container · View Logs · Benchmark · Edit Profile · Edit Config |
| **Stopped** | Start Container · Edit Profile · Edit Config · Delete Profile |

**Benchmark** fires a single `/v1/chat/completions` request at the running
server and reports tokens/second. **Delete** is blocked while a container is
running — stop it first.

## Quick Setup

<kbd>n</kbd> opens a backend picker, then a Quick Setup form that creates a
matching **profile + config** in one step. For vLLM the form takes a Hugging
Face model id, GPU id, port, GPU-memory-utilization, an Enable-LoRA toggle, and
an optional "copy params from" existing config. The profile name is derived
from the model id automatically.

## Memory estimator

Press <kbd>m</kbd> (or click the search box) and type a Hugging Face model id,
then <kbd>Enter</kbd>. llmux estimates the model's memory footprint and renders
it against your detected GPUs:

- A per-GPU bar shows the estimated fit ratio — green / yellow / red, or a red
  `OVER` marker when the estimate exceeds available VRAM.
- With multiple GPUs, it also shows the per-GPU figure assuming tensor
  parallelism (`TP=N`).

This runs entirely on the dashboard — no profile or container needed.

## Container start screen

Picking **Start Container** opens a full-screen launcher with a **Version**
radio set. The options differ per backend:

=== "vLLM"

    Five startup modes:

    | Mode | What it does |
    | --- | --- |
    | **Local Latest** | Highest local `vllm/vllm-openai` versioned tag |
    | **Official Release** | Pulls Docker Hub's latest *stable* version (explicit semver, never `:latest`) |
    | **Nightly** | Pulls `vllm/vllm-openai:nightly` |
    | **Dev Build** | Uses / builds `vllm-dev:<branch>` from source — reveals Repo URL + Branch inputs |
    | **Custom Tag** | An arbitrary image tag you type in |

=== "llama.cpp"

    Three startup modes — llama.cpp has a single official `ghcr.io` tag, so
    "Local / Official / Nightly" collapses into one:

    | Mode | What it does |
    | --- | --- |
    | **Default Image** | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
    | **Dev Build** | Uses / builds `llamacpp-dev:<branch>` from source — reveals Repo URL + Branch inputs |
    | **Custom Tag** | An arbitrary `<repo>:<tag>` you type in |

    If the profile is already pinned to a `llamacpp-dev:` image via its
    `image_tag`, the screen shows that pin — selecting Default Image or Custom
    Tag overrides it for that launch only.

Selecting **Dev Build** reveals **Repo URL** and **Branch** inputs, pre-filled
from `.env.common`. Pressing **Start** streams the startup log live, then
switches to following the container's logs. See [Dev Builds](dev-build.md) for
the build mechanics.

!!! note "Conflict gate before start"
    The dashboard runs a cross-backend port + GPU conflict check before opening
    the start screen. If something clashes, you get a confirmation dialog with
    the details and a **Start anyway** button.

## Profile and config forms

**Edit Profile** and **Edit Config** open modal forms:

- The **profile form** edits container name, port, GPU id, the linked config,
  and `image_tag`. The profile name itself is locked once created.
- The **config form** is a dynamic list of flag rows — add a row, type a flag
  name (with autocompletion against known `llama-server` / `vllm serve` flags),
  and a value. Leaving a boolean flag's value blank stores `true`. A help line
  shows a one-line description of the focused flag.

Both forms map straight onto [`profiles.yaml`](profiles.md) and the
[config YAML](configs.md) — there is no separate TUI state.

## See also

- [Profiles](profiles.md) / [Model Configs](configs.md) — what the forms edit
- [Container Lifecycle](container-lifecycle.md) — what Start / Stop / Logs do
- [Dev Builds](dev-build.md) — the Dev Build start mode
- [CLI Reference](../reference/cli.md) — the same operations, headless
