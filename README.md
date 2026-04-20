# miniforge — native-Windows multi-engine LLM stack

Run MiniMax-M2.7, Kimi K2.5, Qwen3, and other MoE models on a 28 GB mini-PC with an iGPU — no Docker, no WSL, no compose. Four inference engines behind one OpenAI-compatible endpoint, a hot-swappable launcher, a benchmark harness, and a chat UI.

Built for the GMKtech M7 (AMD Ryzen 7 PRO 6850H, Radeon 680M iGPU, 28 GB DDR5, Windows 11). Portable to any comparable x86 box.

## What's in the box

```
Browser :8080 ─▶ Open WebUI ─▶ LiteLLM :4000 ─▶ engine_manager :9292 ─┬─▶ llama.cpp Vulkan .exe :8001
                                                                       ├─▶ ik_llama.cpp .exe  :8002
                                                                       ├─▶ miniforge-server  :8003
                                                                       └─▶ airllm-server     :8004
```

- **engine_manager** is the compose replacement: ~350 LOC of Python that owns one child subprocess at a time, swaps engines on `model` field, TTL-unloads idle models, and presents a stable OpenAI API to LiteLLM.
- **miniforge** is the original CPU library (M7Config, MoE-aware memory sizing, TurboQuant KV aliasing, Unsloth GGUF auto-discovery), wrapped in a FastAPI OpenAI shim.
- **llama.cpp Vulkan** handles iGPU-accelerated attention on the Radeon 680M (`-ngl 20 --cpu-moe`).
- **ik_llama.cpp** is the stretch-goal engine with custom IQ quants (IQ2_KS), `--fused-moe`, `-rtr` expert tensor repacking, and better MLA.
- **AirLLM** is the reference / educational engine for layer-streaming inference.
- **guidellm-style benchmark harness** (`bench/run.ps1` + `bench/runner.py`) hits every engine on the same prompts and emits a markdown SUMMARY.md.

## Honest TPS expectations on the M7 for MiniMax-M2.7-161B-REAP Q2_K (56 GB)

| Engine | Expected TPS | Why |
|---|---|---|
| llama.cpp Vulkan `--cpu-moe -ngl 20` | 1.5 – 2.0 | iGPU attention offload; experts CPU-bound on DDR5 |
| ik_llama.cpp IQ2_KS `-rtr --fused-moe` | **2.0 – 2.8** | +30% over mainline on sub-2-bit MoE; 2.5 TPS lives here |
| miniforge (llama-cpp-python CPU) | 1.0 – 1.5 | No iGPU, lowest overhead |
| AirLLM (layer streaming) | ~0.3 | Educational reference |

The benchmark harness settles the actual ranking on your silicon. The fast-track control, **Qwen3-30B-A3B UD-Q4_K_M (18 GB)**, fits fully in page cache and clocks 12–18 TPS on Vulkan — use it when you need real work done.

### Upgrade path (one-line edits documented below)

1. **+ 64 GB SODIMM → 92 GB total RAM**: REAP Q4_K_M (92 GB) fully resident; expect **3–6 TPS**.
2. **External RTX 4060 Ti 16 GB via OCuLink / USB4**: switch to `install-llama-cuda.ps1`; expect **4–8 TPS** on REAP Q3_K_M.

## Quickstart (3 commands)

Prereqs:
- Python ≥ 3.11 on PATH
- [uv](https://docs.astral.sh/uv/) (installed by `setup.ps1` if missing)
- AMD Adrenalin ≥ 25.Q1 (Vulkan 1.3 runtime)

Everything is driven by one script: `.\scripts\start.ps1 -Action <verb>`.

```powershell
# 1. Bootstrap — venv, pip deps, llama.cpp Vulkan binary. (default action)
.\scripts\start.ps1

# 2. (optional) Pull the 56 GB REAP Q2_K + 18 GB Qwen3-30B control.
#    Skip if you're happy to test with the existing MiniMax-M2.7 UD-IQ2_XXS
#    you already have in C:\Users\midwe\miniforge\gguf\UD-IQ2_XXS\.
.\scripts\start.ps1 -Action pull

# 3. Launch the stack (engine_manager + LiteLLM + Open WebUI).
.\scripts\start.ps1 -Action up

# Or bootstrap and launch in one go:
.\scripts\start.ps1 -Action all
```

Open <http://localhost:8080>, pick one of:

- `minimax-m27-full-vulkan` (228 B UD-IQ2_XXS, your existing download)
- `minimax-reap-vulkan` (161 B Q2_K, 0xSero REAP)
- `minimax-reap-ikllama` (161 B IQ2_KS — only after `install-ik-llama.ps1 -WithIkLlama -ConvertIK`)
- `qwen3-30b-vulkan` (fast control)

Benchmark them against each other:

```powershell
.\scripts\start.ps1 -Action bench -BenchAliases minimax-m27-full-vulkan,qwen3-30b-vulkan -BenchMaxPrompts 4
# -> bench/results/SUMMARY.md
```

Live status dashboard:

```powershell
.\scripts\start.ps1 -Action status -Watch 5
```

Stop everything:

```powershell
.\scripts\start.ps1 -Action down
```

Everything else (`install-vulkan`, `install-ik-llama`, `obs-up`, ...) lives behind `-Action`. Run `Get-Help .\scripts\start.ps1 -Full` for the whole menu.

## Model inventory (automatic)

| Alias | Model | Quant | Size | Location |
|---|---|---|---|---|
| `minimax-m27-full-*` | MiniMax-M2.7 228 B | UD-IQ2_XXS | ~61 GB | `C:\Users\midwe\miniforge\gguf\UD-IQ2_XXS\` (already yours) |
| `minimax-reap-*` | MiniMax-M2.7-161B-REAP | Q2_K | ~56 GB | `$env:MODEL_DIR_REAP\MiniMax-M2.7-161B-REAP-GGUF\` |
| `qwen3-30b-*` | Qwen3-30B-A3B | UD-Q4_K_M | ~18 GB | `$env:MODEL_DIR\Qwen3-30B-A3B-Instruct-GGUF\` |

See [`config/engines.yaml`](config/engines.yaml) to add more models or tweak engine flags.

## Observability (opt-in)

```powershell
.\scripts\start.ps1 -Action obs-up
# Prometheus http://127.0.0.1:9090
# Grafana    http://127.0.0.1:3000  (admin/admin, miniforge dashboard preloaded)
```

Every engine exposes `/metrics` on its port. The engine_manager aggregates a few custom Prometheus metrics (`miniforge_tps`, `miniforge_ttft_seconds`, `miniforge_engine_info`, etc.).

## Repo layout

```
llm-optimizer-compose/
├── pyproject.toml              unified project, [server,ui,bench,airllm,dev] extras
├── src/miniforge/              original miniforge library + server/ + launcher/
│   ├── core/ …                 engine, memory, backends (llama_cpp, transformers)
│   ├── models/ …               registry, minimax, gguf_convert
│   ├── server/                 FastAPI OpenAI shims (miniforge + airllm)
│   └── launcher/               engine_manager.py, engines.py
├── engines/                    downloaded/built binaries per engine
├── config/                     engines.yaml, litellm.yaml, openwebui.env, prometheus.yml, grafana/
├── configs/                    M7Config YAMLs (m7-optimized.yaml)
├── bench/                      runner.py, compare.py, prompts.jsonl, results/
├── benchmarks/                 miniforge's existing suite (context / memory / perf / quality)
├── scripts/                    start.ps1 (single entry point for every action)
├── examples/                   basic_chat.py, streaming_chat.py, tool_agent.py, vision_chat.py
├── docs/diagrams/              architecture markdown diagrams
├── paper/                      LaTeX + paper.md
└── tests/                      pytest suite
```

## Troubleshooting

- **`llama-server.exe --list-devices` doesn't show the Radeon 680M.**
  Update AMD Adrenalin to ≥ 25.Q1. In BIOS, set UMA Frame Buffer Size to 4 GB (some vendors default to 512 MB which kneecaps iGPU VRAM).

- **Vulkan says `ErrorOutOfDeviceMemory` mid-generation.**
  Lower `-ngl 20` to `-ngl 12` (or 0) in `config/engines.yaml` for the `llama_vulkan` engine. 680M shares RAM with the system; big KV caches + batched prefill can starve it.

- **Nothing on :8080 / :4000, or `Invoke-RestMethod ... 4000` can't connect (Windows).**
  LiteLLM and Open WebUI both print Unicode banners at startup; under cp1252 that raises `UnicodeEncodeError` in `logs/litellm.log.err` (`banner.py` / `click.echo`) or `logs/openwebui.log.err`. `scripts/start.ps1 -Action up` sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` for the whole launch; `config/openwebui.env` also sets them for WebUI. Run `.\scripts\start.ps1 -Action down` then `-Action up` again. First WebUI startup can sit on "Fetching … files" for the local embedding model; wait until the log shows Uvicorn listening.

- **Open WebUI doesn't see the models.**
  Open WebUI normally **persists** admin connection settings in its SQLite DB (`data/openwebui`), which can override `config/openwebui.env` after the first run. This repo sets `ENABLE_PERSISTENT_CONFIG=False` there so LiteLLM URL/key from env always apply; restart with `.\scripts\start.ps1 -Action down` then `-Action up`. The proxy must accept the same key Open WebUI sends: `start.ps1` sets `OPENAI_API_KEY` from `LITELLM_MASTER_KEY` when LiteLLM is enabled. Verify the proxy: `Invoke-RestMethod http://127.0.0.1:4000/v1/models -Headers @{ 'Authorization' = 'Bearer sk-local-dev-key' }` (or your `LITELLM_MASTER_KEY`). Check LiteLLM logs: `Get-Content -Wait .\logs\litellm.log`. If you run `-NoLiteLLM`, point Open WebUI at `http://127.0.0.1:9292/v1` with API key `sk-local` instead.

- **REAP download failed partway through the 56 GB.**
  `hf download` resumes on re-run; just call `.\scripts\start.ps1 -Action pull` again.

- **Engine startup takes forever.**
  `engine_manager` waits up to 600 s for MoE engines to cold-start (first-time mmap of a 56–92 GB file is painful). Check `logs/engine_manager.log` for the actual subprocess stderr. On a slow NVMe or with Defender real-time scanning, bump `ready_timeout_s` in `config/engines.yaml`.

- **`The term 'uv' is not recognized`.**
  The setup action will install it with `python -m pip install --user uv`. Close and reopen PowerShell after install so PATH updates take effect.

- **ik_llama build fails: missing Visual C++ toolset.**
  Install "Build Tools for Visual Studio 2022" with the "Desktop development with C++" workload: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>. Or skip `-WithIkLlama` / `-Action install-ik-llama` — the rest of the stack runs without it.

- **`paper/paper.md` claims MiniMax-M2.7 is 2.7 B params and fits in 4 GB.**
  That's a naming collision: the codename `M2.7` does not mean "2.7 billion". MiniMax-M2.7 is a ~228 B MoE (see [the HF card](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)). The paper is out of date; the `models/registry.py` + `models/minimax.py` code correctly treats it as 228 B and warns at >100 B.

## License

MIT — see [LICENSE](LICENSE).
