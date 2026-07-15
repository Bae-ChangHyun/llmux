# demo/llmux.tape — LLMUX intro walkthrough (single take)
#   1. fine-tune a config: toggle a flag off + add one with autocomplete
#   2. create a new profile from an official vLLM recipe, reviewed vs your GPU
# Run: cd <worktree> && vhs demo/llmux.tape
# Pre: internet (fetches vllm-project/recipes), a local NVIDIA GPU (fit check),
#      the clean demo profiles.yaml / config/ from the recording setup.

Output demo/llmux.gif

Set Shell "bash"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 14
Set Width 1280
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 42ms
Set Framerate 20
Set LoopOffset 3%

Env PS1 "$ "
Env LLMUX_LANG "en"

Type "source .venv/bin/activate && llmux"
Enter
Sleep 4s

# ── 1. Fine-tune a model config — no engine flags to memorize ──
Down
Sleep 600ms
Type "c"
Sleep 1600ms
# toggle a flag OFF without deleting it (parked as an inert marker)
Tab@70ms 3
Sleep 500ms
Space
Sleep 1500ms
# add a new flag — inline autocomplete completes it from the engine's arg list
Tab@70ms 4
Sleep 300ms
Enter
Sleep 700ms
Tab@55ms 11
Sleep 400ms
Type "max-num"
Sleep 1100ms
Right
Sleep 900ms
Tab
Type "8192"
Sleep 900ms
Tab@70ms 3
Sleep 400ms
Enter
Sleep 1500ms
Escape
Sleep 1600ms

# ── 2. New profile from an official vLLM recipe, checked against your GPU ──
Type "n"
Sleep 1200ms
Type "1"
Sleep 1800ms
Tab
Sleep 300ms
Type "Qwen/Qwen3-4B"
Sleep 800ms
Tab
Sleep 400ms
Enter
Sleep 3800ms
# review window: precision variants vs your GPU's VRAM (✓/✗), live preview
Tab
Sleep 800ms
Down
Sleep 1400ms
Up
Sleep 1000ms
Tab
Sleep 300ms
Tab
Sleep 300ms
Space
Sleep 1600ms
Tab
Sleep 400ms
Enter
Sleep 3200ms

Type "q"
Sleep 1500ms
