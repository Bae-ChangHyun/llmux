# demo/gpu.tape — HF model memory estimator + system GPU view

Output demo/gpu.gif

Set Shell "bash"
Set FontSize 14
Set Width 1280
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 50ms
Set Framerate 24
Set LoopOffset 3%

Env PS1 "$ "

Type "source .venv/bin/activate && llmux"
Enter
Sleep 3500ms

# `m` focuses the HF model memory estimator input on the dashboard
Type "m"
Sleep 600ms

# First estimate — small non-gated model: fit check with GPU bar
Type "Qwen/Qwen3-0.6B"
Sleep 400ms
Enter
Sleep 4s

# Clear with Ctrl+U, try a much larger model to show the OVER indicator
Ctrl+U
Sleep 200ms
Type "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
Sleep 400ms
Enter
Sleep 4s

# Move focus out of the Input so dashboard shortcuts fire again
Tab
Sleep 400ms

# `s` → System screen (GPU table, docker images, disk)
Type "s"
Sleep 4s
Escape
Sleep 800ms

Type "q"
Sleep 1500ms
