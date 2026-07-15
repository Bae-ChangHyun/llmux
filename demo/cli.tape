# demo/cli.tape — headless CLI: recipe import, parameter on/off, status
# Run: cd <worktree> && vhs demo/cli.tape

Output demo/cli.gif

Set Shell "bash"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 14
Set Width 1280
Set Height 760
Set Theme "Dracula"
Set TypingSpeed 42ms
Set Framerate 24
Set LoopOffset 4%

Env PS1 "$ "
Env LLMUX_LANG "en"

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 500ms
Show

# 1. Peek at a model's official vLLM recipe (variants + features)
Type "llmux config from-recipe Qwen/Qwen3-32B --list"
Enter
Sleep 3500ms

# 2. Import a variant into a config (auto-picks flags for that precision)
Type "llmux config from-recipe Qwen/Qwen3-4B --variant fp8 --feature reasoning --name qwen3-4b"
Enter
Sleep 2800ms

# 3. Toggle a parameter OFF without deleting it — parked as an inert marker
Type "llmux config edit qwen3-4b --disable reasoning-parser"
Enter
Sleep 1600ms
Type "llmux config show qwen3-4b"
Enter
Sleep 3000ms

# 4. Profile status across both backends
Type "llmux ps"
Enter
Sleep 3500ms
