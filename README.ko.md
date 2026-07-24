<div align="center">

<img src="assets/llmux-hero.png" alt="llmux — vLLM과 llama.cpp를 위한 터미널 도구" width="440"/>

# llmux

**vLLM과 llama.cpp 서버를 터미널 하나에서 올리고 관리합니다 — TUI로도, CLI로도.**

[![CI](https://github.com/Bae-ChangHyun/llmux/actions/workflows/ci.yml/badge.svg)](https://github.com/Bae-ChangHyun/llmux/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-7c4dff?style=flat-square)](https://Bae-ChangHyun.github.io/llmux/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-semver-green?style=flat-square)](https://github.com/vllm-project/vllm)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-server--cuda-8A2BE2?style=flat-square)](https://github.com/ggml-org/llama.cpp)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[English](README.md) | **한국어**

vLLM은 HF Transformers, llama.cpp는 GGUF를 다룹니다.
<br/>
**엔진도, config도, 터미널도 따로 노는 두 세계입니다.**
<br/><br/>
llmux는 이 둘을 Docker Compose 위의 Textual 대시보드 하나로 묶습니다.
<br/>
**프로필을 고르고 Enter만 누르면, 그 프로필의 엔진이 알아서 뜹니다.**

<br/>

<sub>Linux + NVIDIA GPU + Docker 필요. macOS, AMD/ROCm, CPU 전용은 아직 미지원.</sub>

<br/>

**[📖 전체 문서 →](https://Bae-ChangHyun.github.io/llmux/)**

</div>

<br/>

## 데모

<div align="center">

<b>한 GPU에서 두 엔진, 모델별 실시간 tok/s</b> &mdash; vLLM 모델과 llama.cpp 모델을 나란히 띄우고, 각각의 처리 속도를 한 대시보드에서 봅니다<br/>
<img src="assets/dashboard-tokps.png" alt="llmux 대시보드 — vLLM과 llama.cpp 모델이 동시에 돌며 각자 tok/s 표시" width="820"/>

<br/><br/>

<b>vLLM과 llama.cpp를 한 대시보드에서</b> &mdash; config 플래그를 켜고 끄고, 자동완성으로 추가하고, vLLM 공식 recipe에서 모델을 가져와 GPU VRAM에 맞는지 확인<br/>
<img src="demo/llmux.gif" alt="llmux 워크스루 — config 편집, 자동완성, GPU 인지 recipe import" width="800"/>

<br/><br/>

<table>
<tr>
<td align="center" width="50%">
  <b>Headless CLI</b> &mdash; recipe import, 파라미터 on/off, 상태 조회<br/>
  <img src="demo/cli.gif" alt="CLI 데모" width="100%"/>
</td>
<td align="center" width="50%">
  <b>GPU 메모리 추정</b> &mdash; 모델별 GPU fit 바<br/>
  <img src="demo/gpu.gif" alt="GPU 메모리 추정 데모" width="100%"/>
</td>
</tr>
</table>

</div>

<br/>

## 기능

- **두 엔진을 한 대시보드에서** &mdash; vLLM과 llama.cpp 프로필을 하나의 Textual TUI에 나란히 놓고, 실행 중인 모델마다 **tok/s**를 실시간으로 표시합니다.
- **프로필은 한 번 만들어두고, 올렸다 내렸다** &mdash; 실험하는 모델을 하나의 `profiles.yaml`에 프로필로 등록해두면, 매번 config를 새로 짤 필요 없이 골라서 실행하고 내립니다. 하나 멈추고 다른 걸 띄우는 것도 그만큼 빠릅니다.
- **엔진 버전을 프로필별로 고정** &mdash; 공식 릴리스, `nightly`, 소스에서 직접 빌드한 이미지를 프로필마다 `image_tag`로 고정합니다. 한 프로필은 vLLM `v0.21.0`, 다른 프로필은 내가 패치한 빌드를 쓸 수 있습니다. 모호한 `:latest`는 거부하고, 안정 버전은 구체적인 버전으로 해석하며, 컨테이너 안에서 실제로 돌고 있는 버전까지 검증합니다.
- **VRAM을 아는 vLLM recipe import** &mdash; 모델의 공식 [vllm-project/recipes](https://github.com/vllm-project/recipes) 설정을 가져와, 정밀도 변형(bf16 / fp8 / awq)이 실제 내 GPU VRAM에 맞는지 확인한 뒤 씁니다. 80GB 카드에서 검증된 recipe가 16GB에서 조용히 넘치는 일을 막습니다.
- **메모리 추정** &mdash; HF 모델을 넣으면 다운로드나 실행 전에 GPU별 fit 바를 보여줍니다 ([`hf-mem`](https://github.com/alvarobartt/hf-mem)).
- **엔진 플래그를 1:1로** &mdash; `config/<backend>/<name>.yaml`이 엔진 플래그에 그대로 대응합니다. 샘플링, context 길이, KV 캐시 정밀도, MoE CPU offload 등. 플래그는 **지우지 않고 켜고 끌 수 있고**, 손으로 쓴 주석도 편집 후 그대로 남습니다.
- **실제 이미지에서 뽑은 플래그 자동완성** &mdash; config 편집기가 지금 쓰는 `vllm serve` / `llama-server` 이미지의 실제 플래그 목록에서 이름을 자동완성합니다. 한 번 추출해 버전별로 캐시하므로, 제안이 실제로 띄우는 엔진 빌드와 정확히 맞습니다.
- **실시간 처리량 + 벤치마크** &mdash; 각 컨테이너의 `/metrics`에서 생성 tok/s를 실시간으로 읽고(`llmux stats`), warmup + 중앙값 벤치마크(`llmux bench`)로 같은 하드웨어에서 quant A와 B를 비교합니다.
- **btop 스타일 라이브 모니터** &mdash; 실행 중인 모델에서 `v`를 누르면 전체 화면 상세 뷰가 뜹니다. 처리량 braille 그래프, KV 캐시 추이, 캐시 적중·요청·GPU별 util·mem·temp·power·PCIe를 heat bar로, 그리고 TTFT·E2E percentile(p50/p95/p99)과 prefill/decode 구간을 담은 지연 패널을 보여줍니다. `p` 일시정지, `r` 피크 초기화, `+/-` 주기, `l` 언어. 두 엔진 모두 지원하며, llama.cpp가 노출하지 않는 지표는 `—`로 표시합니다.
- **일반 터미널 모니터** &mdash; `v` 모니터를 Textual 없이: `llmux top <프로필>`(또는 실행 중인 행에서 `t`)로 같은 그래프·캐시 적중·지연 percentile·GPU/PCIe 패널을 자동 갱신되는 터미널 페이지로 봅니다. 저대역 SSH나 단순 터미널에 좋고, `q`로 나갑니다.
- **멀티 GPU** &mdash; vLLM 프로필을 `tensor_parallel_size`로 GPU에 분할합니다. GPU 목록에서 자동으로 도출하거나 직접 지정합니다.
- **LoRA 어댑터** &mdash; 호스트 디렉터리의 LoRA 모듈을 마운트해 vLLM 베이스 모델에 얹어 서빙합니다.
- **GGUF 자동 다운로드** &mdash; llama.cpp가 첫 실행 때 `-hf`/`-hff`로 GGUF를 HF 캐시에 바로 받습니다. 별도의 `hf download`나 `models/` 연결이 없습니다.
- **소스에서 dev 빌드** &mdash; 임의의 fork/branch에서 `vllm-dev:` / `llamacpp-dev:` 이미지를 빌드하고(GPU arch 자동 감지) `image_tag`로 프로필에 고정합니다.
- **전부 스크립트로도** &mdash; TUI의 모든 동작은 `--json`을 지원하는 headless `llmux` 서브커맨드로도 됩니다. 스크립트, 에이전트, CI에 그대로 붙습니다.
- **이름 즉시 변경** &mdash; 프로필이나 config를 다시 빌드하지 않고 이름만 바꿉니다. 그 config를 참조하던 프로필들은 자동으로 다시 연결됩니다.
- **한/영 UI** &mdash; TUI 전체가 `LLMUX_LANG=ko|en`으로 한국어와 영어를 오갑니다.

<br/>

## 빠른 시작

한 줄로 설치합니다. llmux를 clone하고, 의존성을(그리고 `uv`가 없으면 `uv`까지) 설치한 뒤 `llmux` 명령을 PATH에 올립니다:

```bash
curl -fsSL https://raw.githubusercontent.com/Bae-ChangHyun/llmux/main/install.sh | sh
```

그다음 그냥 실행하면 됩니다. 첫 실행 때 짧은 설정 마법사(HF 캐시 경로, 모델 디렉터리, 선택 사항인 토큰)를 안내하고 `.env.common`을 대신 써줍니다:

```bash
llmux
```

설치된 폴더는 그대로 git 저장소로 남아 있어서, `~/.llmux`에서 `git pull`만 하면 재설치 없이 업데이트됩니다. llmux는 시작할 때 하루 한 번 GitHub에서 새 릴리스를 확인하고 받을지 물어봅니다. 다른 위치에 설치하려면 스크립트에 `LLMUX_DIR`을 넘깁니다: `curl -fsSL ... | LLMUX_DIR=/path sh`.

<details>
<summary>수동 설치</summary>

```bash
git clone https://github.com/Bae-ChangHyun/llmux.git && cd llmux
uv tool install --editable .   # editable — 코드 수정이 바로 반영됨
uv tool update-shell           # 최초 1회 — ~/.local/bin 을 PATH에 추가
```

</details>

> 전역 설치가 싫다면 저장소 안에서 `uv run llmux`로 실행할 수 있습니다. 다른 위치에서 쓰려면 `LLMUX_ROOT=/path/to/llmux`를 지정하세요.

> **언어:** TUI는 한/영 이중 언어입니다. 기본은 시스템 로케일을 따르고, `LLMUX_LANG=en` 또는 `LLMUX_LANG=ko`로 강제할 수 있습니다.

전체 과정은 [설치 가이드](https://Bae-ChangHyun.github.io/llmux/getting-started/installation.html)와 [빠른 시작](https://Bae-ChangHyun.github.io/llmux/getting-started/quickstart.html) 문서를 참고하세요. (문서 사이트는 영어입니다.)

### Headless CLI

TUI의 모든 기능은 비대화형 서브커맨드로도 있습니다. 인자 없이 `llmux`를 실행하면 TUI가 뜨고, 서브커맨드를 붙이면 TUI를 건너뜁니다:

```bash
llmux up <profile>                 # 컨테이너 시작
llmux logs <profile>               # 로그 follow
llmux ps --json --running          # 기계가 읽는 상태, 양쪽 백엔드
llmux stats --once --json          # 실행 중 컨테이너의 실시간 tok/s
llmux bench <profile> --runs 3     # warmup + 중앙값 tok/s 벤치마크
llmux profile quick-setup Qwen/Qwen3-8B --gpu-id 0,1
llmux config edit <name> --disable trust-remote-code   # 플래그를 끄되 남겨둠
llmux config from-recipe Qwen/Qwen3-32B --variant fp8   # vLLM 공식 recipe
llmux profile rename old-name new-name                  # 컨테이너가 멈춰 있어야 함
llmux image build-dev --backend llamacpp --branch master
```

`--json`은 모든 list/show/check 명령이 지원합니다. 전체 명령과 플래그는 [CLI 레퍼런스](https://Bae-ChangHyun.github.io/llmux/reference/cli.html)에 있습니다.

<br/>

## 문서

전체 문서는 **[Bae-ChangHyun.github.io/llmux](https://Bae-ChangHyun.github.io/llmux/)** 에 있습니다. (영어)

| 섹션 | 내용 |
|:---|:---|
| [Getting Started](https://Bae-ChangHyun.github.io/llmux/getting-started/installation.html) | 설치, 첫 모델 실행 워크스루 (TUI + CLI) |
| [Guide](https://Bae-ChangHyun.github.io/llmux/guide/profiles.html) | 프로필, 모델 config, 컨테이너 라이프사이클, TUI 단축키, dev 빌드 |
| [Backends](https://Bae-ChangHyun.github.io/llmux/backends/comparison.html) | vLLM과 llama.cpp 심화 + 기능 비교표 |
| [Reference](https://Bae-ChangHyun.github.io/llmux/reference/cli.html) | 모든 CLI 명령/플래그, `.env.common` 변수, 내부 아키텍처 |
| [Troubleshooting](https://Bae-ChangHyun.github.io/llmux/troubleshooting.html) | 흔한 시작/다운로드/GPU 문제 — 증상 → 원인 → 해결 |

<br/>

## 왜 llmux인가?

|  | 수동, 두 툴체인 | llmux |
|:---|:---|:---|
| **엔진 전환** | 엔진마다 CLI, compose, TUI가 따로 | 둘 다 하나의 Textual 대시보드 |
| **프로필 형식** | 프로필마다 `.env`, 두 저장소에 흩어짐 | 하나의 `profiles.yaml`, 실행 때 `.env`로 렌더 |
| **포트 / GPU 충돌** | 컨테이너가 죽고 나서야 알게 됨 | 시작 전에 양쪽 백엔드에 걸쳐 확인 |
| **이미지 버전** | `docker pull latest` 하고 기도 | `:latest` 거부, semver로 해석 + 컨테이너 내부 버전 검증 |
| **dev 빌드** | clone, 빌드, compose 연결을 손으로 | `llmux image build-dev`로 임의 브랜치에서, 프로필별 고정 |
| **메모리 사이징** | 맞겠지 하고 추측 | [`hf-mem`](https://github.com/alvarobartt/hf-mem)로 GPU별 fit 바 |
| **GGUF 설정** | `hf download` → compose 편집 → 마운트 | llama.cpp가 첫 실행 때 받아 호스트에 캐시 |

**llmux가 아닌 것.** 추론 요청을 프록시하거나 라우팅하지 않습니다. 프로필마다 자기 포트에서 서빙하고, llmux는 그것들을 시작·중지·빌드·벤치마크하는 계층입니다. 하나의 엔드포인트 뒤에서 요청 단위로 모델을 바꾸고 싶다면 그건 [llama-swap](https://github.com/mostlygeek/llama-swap)의 역할입니다. 채팅 UI도 아닙니다. [Open WebUI](https://github.com/open-webui/open-webui)나 OpenAI 호환 클라이언트를 llmux가 띄운 서버에 붙이면 됩니다.

<br/>

## 요구 사항

- NVIDIA GPU가 있는 Linux + 최신 드라이버
- [Docker Engine](https://docs.docker.com/engine/install/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU 패스스루용)
- Python 3.10+, 그리고 TUI 환경용 [uv](https://docs.astral.sh/uv/)

첫 실행 전에 GPU 패스스루가 되는지 확인하세요:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

> macOS, AMD/ROCm, CPU 전용은 아직 지원하지 않습니다. llmux는 CUDA 이미지를 NVIDIA GPU 패스스루로 구동하며, 이는 Linux + NVIDIA GPU를 필요로 합니다.

<br/>

## 로드맵

- [x] **Recipe 기반 config 추천** &mdash; v2.4.0에서 `llmux config from-recipe`로 출시
- [ ] **백엔드 간 프로필 clone** &mdash; 현재 `llmux profile clone`은 같은 백엔드 안에서 복제. vLLM 프로필을 llama.cpp GGUF 등가물로 복제하는 건 아직 미구현
- [ ] **일괄 작업** &mdash; 여러 프로필을 양쪽 백엔드에서 한 번에 시작/중지
- [ ] **번들 내보내기/가져오기** &mdash; 프로필 + config 세트를 기기 간에 공유
- [ ] **AMD / ROCm** &mdash; 대체로 base-image + 디바이스 마운트 교체 수준으로, macOS보다 현실적
- [ ] **macOS (native Metal)** &mdash; 검토 중이고 llama.cpp 전용: Docker가 애플 GPU를 Linux 컨테이너에 넘기지 못해, llama.cpp를 컨테이너가 아닌 네이티브 프로세스로 돌려야 함 (vLLM은 CUDA 전용 유지)
- [ ] **Web UI** &mdash; 원격 접근용 선택적 브라우저 대시보드

<br/>

## 기반

llmux는 컨트롤 계층입니다. 어려운 부분은 업스트림에 있습니다. 공식 이미지를 수정 없이 그대로 구동하며, 다음 위에 서 있습니다:

- [vLLM](https://github.com/vllm-project/vllm)과 [llama.cpp](https://github.com/ggml-org/llama.cpp) &mdash; 구동하는 추론 엔진
- [vllm-project/recipes](https://github.com/vllm-project/recipes) &mdash; import하는 vLLM 설정의 출처
- [hf-mem](https://github.com/alvarobartt/hf-mem) &mdash; 모델 메모리 추정
- [Textual](https://github.com/Textualize/textual)과 [Typer](https://github.com/fastapi/typer) &mdash; TUI와 CLI

<br/>

<div align="center">

**MIT License**

</div>
