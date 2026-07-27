# demo/monitor-multi.tape — 모니터: 두 엔진이 동시에 떠 있을 때의 요약 뷰 (스틸)
# Pre: vLLM(8002) + llama.cpp(8091) healthy + 부하.
# Run: cd <repo> && vhs demo/monitor-multi.tape

Output demo/.monitor-multi-capture.gif

Set Shell "bash"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 14
Set Width 1300
Set Height 310
Set Theme "Dracula"
Set WindowBar Colorful
Set Padding 20
Set Framerate 12

Env PS1 "$ "
Env LLMUX_LANG "en"

Type "llmux top"
Enter
Sleep 16s
Screenshot assets/monitor-multi.png
Sleep 4s
