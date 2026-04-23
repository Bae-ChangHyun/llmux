# demo/vllm.tape — Quick Setup peek → edit qwen3-0-6b config → start → logs
# Pre: qwen3-0-6b profile+config exist.

Output demo/vllm.gif

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

# 1. Show Quick Setup entry point — `n` → backend picker → vLLM
Type "n"
Sleep 1200ms
Type "1"
Sleep 2200ms

# Focus the HF model Input, type a tiny model name
Tab
Sleep 300ms
Type "Qwen/Qwen3-0.6B"
Sleep 800ms

# For the demo we'll use the already-configured profile instead of submitting.
Escape
Sleep 1200ms

# 2. Navigate to qwen3-0-6b in the dashboard list.
# Row order: gemma4 (llama.cpp) first, then 16 vllm sorted — qwen3-0-6b is at index 11.
Down@100ms 11
Sleep 500ms

# 3. Open its config editor and actually edit GPU memory utilization live.
Type "c"
Sleep 2500ms

# Navigate to GPU Memory Utilization input (Config Name is disabled, so Tab hits
# Model first, then GPU Mem). Edit it live from 0.25 → 0.35.
Tab 2
Sleep 400ms
Ctrl+U
Sleep 250ms
Type "0.35"
Sleep 1400ms

# Leave without persisting
Escape
Sleep 800ms

# 4. Start the container
Type "u"
Sleep 2s
# In the version picker, Tab to "Start" button and confirm
Tab
Sleep 400ms
Enter
Sleep 10s

# 5. View logs
Type "l"
Sleep 6s
Escape
Sleep 600ms

# Quit
Type "q"
Sleep 1500ms
