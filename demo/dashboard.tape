# demo/dashboard.tape — 대시보드 스틸컷: 두 엔진 동시 실행 + 모델별 tok/s
# Pre: vLLM(8002) + llama.cpp(8091) 이 healthy 하고 부하가 걸려 있어야 함.
# Run: cd <repo> && vhs demo/dashboard.tape

Output demo/.dashboard-capture.gif

Set Shell "bash"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 14
Set Width 1280
Set Height 460
Set Theme "Dracula"
Set WindowBar Colorful
Set Padding 20

Env PS1 "$ "
Env LLMUX_LANG "en"

Type "llmux"
Enter
Sleep 12s
Screenshot assets/dashboard-tokps.png
Sleep 3s
