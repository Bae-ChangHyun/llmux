# demo/monitor.tape — btop 스타일 모니터 상세 뷰 (스틸)
# 그래프는 btop 처럼 오른쪽부터 채워진다. `+` 로 폴링을 0.25s 까지 올린 뒤
# 폭만큼 히스토리가 쌓이길 기다렸다가 찍는다.
# Pre: vLLM(8002) healthy + 동시 요청 부하(KV 가 차 있어야 KV 그래프가 보인다).
# Run: cd <repo> && vhs demo/monitor.tape

Output demo/.monitor-capture.gif

Set Shell "bash"
Set FontFamily "DejaVu Sans Mono"
Set FontSize 14
Set Width 1000
Set Height 570
Set Theme "Dracula"
Set WindowBar Colorful
Set Padding 20
Set Framerate 2

Env PS1 "$ "
Env LLMUX_LANG "en"

Type "llmux top qwen3-0-6b"
Enter
Sleep 3s
Type "+"
Sleep 300ms
Type "+"
Sleep 300ms
Type "+"
Sleep 120s
Screenshot assets/monitor.png
Sleep 3s
