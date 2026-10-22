# ane-fim

Local, offline code completion for VS Code (Copilot-style gray text), running
**Qwen2.5-Coder-0.5B** on the **Apple Neural Engine** via [ANEMLL](https://github.com/Anemll/Anemll) and Core ML.

No cloud, no usage limits, low power. The server speaks llama.cpp's `/infill` API,
so the existing [llama.vscode](https://github.com/ggml-org/llama.vscode) extension works unchanged.

> **Status: working prototype.** Python server, tested on one Mac (M3, 16 GB, macOS 27).
> Setup is still manual. A self-contained Swift binary is planned.

## Results (M3, real llama.vscode traffic replayed)

| | Typical latency | p95 | Power under load | Energy / completion |
|---|---|---|---|---|
| llama-server on GPU (baseline) | 24 ms | 449 ms | 8.1 W | 0.70 J |
| **ane-fim on ANE** | 64 ms | 532 ms | **4.4 W** | **0.52 J** |

The GPU stays idle. Most of the speed comes from reusing the KV cache between keystrokes:
typically only ~3 new tokens are processed per request. Details and caveats are in
[CLAUDE.md](CLAUDE.md) (it doubles as the project log).

## How it works

- **Model:** Qwen2.5-Coder-0.5B base, converted with ANEMLL (context 2048, 6-bit LUT; FP16 fallback).
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

# 2. Convert the model (~11 min for LUT6, ~3 min for FP16)
CONTEXT=2048 scripts/convert.sh lut6

# 3. Run the server (from the ANEMLL env)
python server-py/server.py            # --model fp16 for the FP16 fallback
```

Then install the **llama.vscode** extension and set `llama-vscode.endpoint` to `http://127.0.0.1:8012`.

## Repo layout

```
server-py/   server.py (HTTP, stop rules), engine.py (ANE + KV reuse), prompt.py (FIM prompt builder)
scripts/     convert.sh, measure_power.sh, power_compare.sh
eval/        bench_latency.py, compare_hf.py, check_cache.py, kv_rewind_test.py, proxy_logger.py, snippets/
```

## Credits

- [ANEMLL](https://github.com/Anemll/Anemll) (MIT): conversion pipeline and Core ML runtime.
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) (Apache 2.0): the model.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama.vscode](https://github.com/ggml-org/llama.vscode): `/infill` API and editor client.
