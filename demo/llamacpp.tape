# demo/llamacpp.tape — Quick Setup peek → edit gemma4 config in TUI → start → logs
# Pre: gemma4 profile+config exist.

Output demo/llamacpp.gif

Set Shell "bash"
Set FontSize 14
Set Width 1280
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 45ms
Set Framerate 24
Set LoopOffset 3%

Env PS1 "$ "

Type "source .venv/bin/activate && llmux"
Enter
Sleep 3500ms

# 1. Show Quick Setup entry point — press `n` → backend picker → llama.cpp
Type "n"
Sleep 1200ms
Type "2"
Sleep 2200ms

# Type HF repo so the demo shows Quick Setup is ready to fetch+build
Tab
Sleep 300ms
Type "unsloth/gemma-4-26B-A4B-it-GGUF"
Sleep 800ms

# Cancel — for the demo we'll use the existing gemma4 profile instead
Escape
Sleep 1200ms

# 2. gemma4 is the llama.cpp row (top). Open its config editor.
Type "c"
Sleep 2500ms

# Navigate into the ctx-size VALUE input (6 Tabs from the focused name field:
#   Add → model-file.key → model-file.value → model-file.x → ctx-size.key → ctx-size.value)
Tab 6
Sleep 500ms
Ctrl+U
Sleep 250ms
Type "16384"
Sleep 1600ms

# Leave the editor without persisting (keep gemma4 config as originally configured).
Escape
Sleep 800ms

# 3. Back on dashboard — start gemma4
Type "u"
Sleep 7s

# 4. View logs
Type "l"
Sleep 5s
Escape
Sleep 600ms

# Quit
Type "q"
Sleep 1500ms
