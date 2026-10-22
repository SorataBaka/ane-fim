# ane-fim

Local, offline code completion for VS Code (Copilot-style gray text), running
**Qwen2.5-Coder-1.5B** (or 0.5B) on the **Apple Neural Engine** via [ANEMLL](https://github.com/Anemll/Anemll) and Core ML.

No cloud, no usage limits, low power. The server speaks llama.cpp's `/infill` API,
so the existing [llama.vscode](https://github.com/ggml-org/llama.vscode) extension works unchanged.

> **Status: working prototype.** Python server, tested on one Mac (M3, 16 GB, macOS 27).
> Setup is still manual. A self-contained Swift binary is planned.

## Results (M3, real llama.vscode traffic replayed)

| | Typical latency | p95 | Power under load | Energy / completion |
|---|---|---|---|---|
| llama-server on GPU, 0.5B (baseline) | 24 ms | 449 ms | 8.1 W | 0.70 J |
| **ane-fim on ANE, 0.5B** | 64 ms | 532 ms | **4.4 W** | **0.52 J** |
| ane-fim on ANE, 1.5B (default) | ~140 ms | ~540 ms | not measured yet | |

Completing the rest of a real line of code (150 cases from unseen Python/Swift): **1.5B 57% exact match vs 0.5B 43%**,
so the 1.5B is the default and the 0.5B is the faster, lower-power option.

The GPU stays idle. Most of the speed comes from reusing the KV cache between keystrokes:
typically only ~3 new tokens are processed per request. Details and caveats are in
[CLAUDE.md](CLAUDE.md) (it doubles as the project log).

## How it works

- **Model:** Qwen2.5-Coder-1.5B base (2 chunks) or 0.5B, converted with ANEMLL (context 2048, 6-bit LUT; 0.5B also as FP16).
- **KV-cache reuse:** the cache is kept between requests and rewound to the shared token prefix.
  Only the changed tokens are prefilled, as one padded 64-token batch (~28 ms).
- **Stable prompt start:** long files are trimmed on line boundaries at a remembered "anchor" line,
  so the cache survives while you type.
- **Stop rules:** mid-line → finish the line; empty line → a block that ends at a blank line or dedent;
  plus token/time caps and a repetition check.

## Setup (manual, for now)

Requirements: Apple Silicon Mac, macOS 15+, Xcode command line tools, [uv](https://github.com/astral-sh/uv).

```bash
# 1. ANEMLL (tested at tag 0.3.5_beta)
git clone https://github.com/Anemll/Anemll ~/src/Anemll
cd ~/src/Anemll && git checkout 0.3.5_beta
./create_uv_env.sh && source env-anemll/bin/activate && ./install_dependencies.sh

# 2. Convert the model (from this repo)
SIZE=1.5B CHUNK=2 CONTEXT=2048 scripts/convert.sh lut6   # default model, ~47 min
CONTEXT=2048 scripts/convert.sh lut6                     # optional 0.5B, ~11 min

# 3. Run the server
make start            # 1.5B on http://127.0.0.1:8012; make start SIZE=0.5b for the small one
make status | logs | stop | restart
```

Then install the **llama.vscode** extension and set `llama-vscode.endpoint` to `http://127.0.0.1:8012`.

## Repo layout

```
Makefile     start/stop/status/logs, bench, test, baseline
server-py/   server.py (HTTP, stop rules), engine.py (ANE + KV reuse), prompt.py (FIM prompt builder)
scripts/     convert.sh, measure_power.sh, power_compare.sh
eval/        bench_latency.py, compare_hf.py, check_cache.py, line_completion.py, kv_rewind_test.py,
             proxy_logger.py, snippets/
```

## Credits

- [ANEMLL](https://github.com/Anemll/Anemll) (MIT): conversion pipeline and Core ML runtime.
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) (Apache 2.0): the model.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama.vscode](https://github.com/ggml-org/llama.vscode): `/infill` API and editor client.
