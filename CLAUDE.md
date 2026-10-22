# CLAUDE.md — ANE Autocomplete (working name: `ane-fim`)

Local, offline, unlimited inline code completion (Copilot-style gray text) for VS Code, running **Qwen2.5-Coder** on the **Apple Neural Engine (ANE)** via **ANEMLL** + Core ML.

Primary goal: **low power + no cloud + no usage limits**, not raw speed.
Secondary goal: latency good enough that suggestions feel instant (< ~300 ms p50).

---

## 1. Context for Claude

- Owner: Chris (Mac, Apple Silicon). Explain things in plain words; keep summaries short.
- This is a personal side project. Prefer simple, working solutions over clever ones.
- Do **not** add cloud services, telemetry, or Copilot/GitHub dependencies.
- Things marked **[VERIFY]** are assumptions from research that have not been tested yet. Check them before building on top of them, and update this file with what you find.
- When a phase is done, tick its checkboxes here and add a short "Findings" note under it.

---

## 2. Architecture

```
┌──────────────┐  HTTP (llama.cpp /infill API)   ┌──────────────────────────────┐
│ VS Code      │ ──────────────────────────────▶ │ ane-fim server (localhost)   │
│ llama.vscode │ ◀────────────────────────────── │  - FIM prompt builder        │
└──────────────┘        { "content": "..." }     │  - tokenizer (Qwen2.5)       │
                                                  │  - KV-cache prefix reuse     │
 (optional) Continue ── /v1/completions ───────▶ │  - stop / time-limit logic   │
                                                  └──────────────┬───────────────┘
                                                                 │ AnemllCore (Swift)
                                                                 ▼
                                                  ┌──────────────────────────────┐
                                                  │ Core ML model on ANE         │
                                                  │ Qwen2.5-Coder-0.5B (base)    │
                                                  │ ctx 1024, FP16/LUT6, argmax  │
                                                  └──────────────────────────────┘
```

Key decision: **copy llama.cpp's `/infill` API** so the existing `llama.vscode` extension works with zero changes. We do not write our own VS Code extension.

---

## 3. Tech stack

| Part | Choice | Why |
|---|---|---|
| Model | `Qwen/Qwen2.5-Coder-0.5B` (**base**, not instruct) | Trained with FIM; small enough for ANE |
| Conversion | ANEMLL `convert_model.sh` (v0.3.5+) | Only maintained HF → ANE LLM pipeline |
| Runtime | ANEMLL Swift package (`anemll-swift-cli`, AnemllCore) | Batch prefill + KV cache already done |
| Server (prototype) | Python + FastAPI, adapted from `alexgusevski/anemll-server` | Fast to iterate |
| Server (final) | Swift 6 + Hummingbird (or Swift NIO) | No Python overhead, one binary |
| Editor client | `llama.vscode` extension (primary), Continue (optional) | Already speaks `/infill` |
| Baseline | `llama-server` (llama.cpp) on Metal GPU | The number to beat |

Requirements (from ANEMLL docs): macOS 15+, Apple Silicon, Xcode CLT, Python 3.9 (3.9–3.11), coremltools ≥ 9.0, 16 GB RAM listed as minimum.

---

## 4. Repo layout (target)

```
ane-fim/
├── CLAUDE.md
├── README.md
├── models/                  # converted models (gitignored)
├── scripts/
│   ├── convert.sh           # wraps ANEMLL convert_model.sh with our flags
│   ├── baseline.sh          # starts llama-server baseline
│   └── measure_power.sh     # powermetrics wrapper
├── eval/
│   ├── snippets/            # FIM test cases (prefix/suffix/expected)
│   ├── compare_hf.py        # ANE vs HF transformers output check
│   └── bench_latency.py     # replays edit sessions against /infill
├── server-py/               # Phase 3 prototype
│   └── server.py
└── server-swift/            # Phase 3b final server
    ├── Package.swift
    └── Sources/AneFim/
        ├── main.swift
        ├── InfillRoute.swift
        ├── PromptBuilder.swift
        ├── KVCacheManager.swift
        └── StopPolicy.swift
```

`models/`, `.venv/`, `env-anemll/`, `*.mlpackage`, `*.mlmodelc` must be in `.gitignore`.

---

## 5. Key facts & formats

### 5.1 Qwen2.5-Coder FIM prompt

```
<|fim_prefix|>{code before cursor}<|fim_suffix|>{code after cursor}<|fim_middle|>
```

Repo-level context (use later, Phase 6):

```
<|repo_name|>{repo}
<|file_sep|>{path_a}
{content_a}
<|file_sep|>{path_b}
{content_b}
<|file_sep|>{current_path}
<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>
```

Special token IDs (from Qwen repo): `<|fim_pad|>` = 151662, `<|repo_name|>` = 151663, `<|file_sep|>` = 151664. Look up `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endoftext|>` from `tokenizer.json` — **do not hardcode guesses**.

Rules:
- **No chat template.** Raw prompt only.
- No BOS token is added (verified: `bos_token` is None; `add_special_tokens=True` adds nothing). The tokenizer *does* ship a chat template — never call `apply_chat_template`. Verified IDs: `<|fim_prefix|>`=151659, `<|fim_middle|>`=151660, `<|fim_suffix|>`=151661, `<|endoftext|>`=151643.

### 5.2 `/infill` API to implement (llama.cpp-compatible subset)

Request (`POST /infill`, JSON):

| Field | Type | Use |
|---|---|---|
| `input_prefix` | string | code before cursor |
| `input_suffix` | string | code after cursor |
| `input_extra` | array of `{filename, text}` | extra context chunks (Phase 6; accept and ignore at first) |
| `prompt` | string | text typed on the current line after prefix (llama.vscode sends this) **[VERIFY]** |
| `n_predict` | int | max tokens (default 64) |
| `t_max_predict_ms` | int | stop after this many ms once a newline has been generated |
| `temperature`, `top_k`, `top_p`, `samplers` | — | accept, ignore (we are greedy) |
| `cache_prompt` | bool | accept; we always reuse cache |
| `stream` | bool | support `false` first; streaming optional |

Response:

```json
{
  "content": "generated middle text",
  "stop": true,
  "stop_type": "eos | limit | word",
  "tokens_cached": 812,
  "timings": { "prompt_n": 7, "prompt_ms": 18.2, "predicted_n": 21, "predicted_ms": 140.5 }
}
```

Also: `GET /health` → `{"status":"ok"}` (503 while loading).

Observed in real llama.vscode v0.0.70 traffic (2026-10-22, `eval/llama_vscode_requests/`):
- Extra request fields: `id_slot`, `n_cmpl` (= 3: asks for 3 alternative completions), `t_max_prompt_ms` (500), `samplers` = `["top_k","top_p","infill"]`. Defaults: `n_predict` 128, `t_max_predict_ms` 500, `stream` false, `cache_prompt` true.
- With `n_cmpl > 1` the response is a **JSON list** of completion objects (`index`, `content`, `tokens_predicted`, `tokens_evaluated`, `stop`, ...), not a single object.
- `prompt` = text already typed on the current line (e.g. `"p"` → completion `"rint(..."`).
- `input_extra` chunks are sent and push prompts past 1024 tokens (seen: 1,113 tokens evaluated with a 44-char prefix). Server must trim; or lower `n_prefix`/`n_suffix`/ring settings in the extension.
- `input_extra` is **not RAG**. It's llama.vscode's "ring buffer": ~32-line snippets (`{text, filename, time}`) taken from recently opened/edited files, up to `ring_n_chunks` (16). The extension's RAG (`rag_enabled`, BM25 + optional embeddings endpoint) is only used by its chat/agent features, never by completions (`completion.js` has no RAG references).
- "FIM_PREP" requests: `n_predict: 0`, empty prefix/suffix, only `input_extra`. They just warm the server's cache with the ring chunks. Our server can answer them right away with empty content until Phase 6.
- Degenerate repetition (`000000...`) can run to full `n_predict` since `t_max_predict_ms` only applies after a newline → stop policy should also catch repeats.

Original note: Capture real llama.vscode traffic against `llama-server` in Phase 0 (e.g. run with `-v` / `--log-prompts-dir`, or a tiny logging proxy) and save sample requests to `eval/llama_vscode_requests/`. Build the server against these real payloads, not this table.

### 5.3 ANEMLL conversion flags we care about

| Flag | Value | Note |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-0.5B` | HF ID auto-downloads |
| `--context` | `2048` (chosen; `1024` build kept for comparison) | Fixed at convert time; 512–2048 recommended. 2048 = −15% decode, +50 MB RAM vs 1024 (see Phase 1 findings) |
| `--batch` | `64` | prefill batch size |
| `--chunk` | `1` / monolithic | 0.5B fits in one |
| LUT | none (FP16) first, then LUT6 | LUT4 quality is poor on ANE (no block quant). FP16 = `--lut1 "" --lut2 "" --lut3 ""` (`--no-lut` is in `--help` but not implemented in 0.3.5_beta; default is LUT4 FFN!) |
| `--argmax` | on | argmax inside LM head → tiny ANE→host transfer; we decode greedily anyway |

---

## 6. Phases

Each phase has: tasks, acceptance criteria, findings. Do them in order. Do not start Phase 4 before Phase 2 passes.

### Phase 0 — Baseline (target: half a day)

Goal: know what "good" looks like, and capture real client traffic.

- [x] `brew install llama.cpp`; run `llama-server` with Qwen2.5-Coder-0.5B base (GGUF Q8) on port 8012, `--cache-reuse 256`. _(runs on 8013; `eval/proxy_logger.py` listens on 8012 and records)_
- [x] Install `llama.vscode`, point it at `http://127.0.0.1:8012/infill`, confirm completions work.
- [x] Log 50+ real `/infill` requests from normal editing → `eval/llama_vscode_requests/*.json`.
- [x] `eval/bench_latency.py`: replay those requests, record time-to-response p50/p95, tokens_cached.
- [x] `scripts/measure_power.sh`: `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500` during a replay; record average W. _(replay version: `scripts/power_compare.sh`)_

Acceptance: a table in this file with baseline latency p50/p95 and average power (CPU/GPU/ANE).

Findings (2026-10-22, llama.cpp 0.7.0 via brew, `ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF`, llama.vscode 0.0.70):
- Captured 282 requests in ~9 min of editing (280 completions + 2 warm-up), all HTTP 200.
- Live round-trip through the proxy (not a replay yet): p50 49 ms, p95 653 ms, max 1,152 ms.
- Prompt size (`tokens_evaluated`): p50 1,201, p95 1,502, max 1,513 → **98% of requests exceed our 1024 ctx** with default extension settings. Trimming (and/or dropping `input_extra`) is mandatory.
- Tokens generated: p50 **2**, p95 54. Most completions are very short, which is good news for the slower ANE decode (~19 ms/token).
- GPU decode speed on a test prompt: ~127 tok/s (vs ANE ~52).
- Replay (`eval/bench_latency.py`) and GPU power: done, see the Phase 5 table.
- Log review at 460 requests:
  - llama-server's KV reuse: `cache_n` median 1,202, **new prompt tokens (`prompt_n`) median 4, p95 54**. Reuse is why its p50 is ~50 ms. On ANE, a 4-token diff = one 64-batch prefill (~27 ms).
  - `n_cmpl=3` → 1,380 completions, **766 empty**. Greedy makes alternatives identical → generate once. **[VERIFY]** that llama.vscode accepts a 1-element list.
  - 112/1,380 (8%) stopped at the 128-token limit; 3 were `000…` repeats. At ANE decode speed that's ~3 s each → server must cap tokens/time itself (§7).
  - `input_extra` snippets came from `.json`, `.md`, `.ipynb` (not the imported code) → mostly noise for completion.
  - llama-server warning "need to evaluate at least 1 token … n_past was set to N-1": an identical prompt was resent. KV manager must back up one token on a full match to get logits.

### Phase 1 — Convert the model (target: 1–2 days)

- [x] Clone upstream `https://github.com/Anemll/Anemll`, pin a release tag, note it here.
- [x] `./create_uv_env.sh && source env-anemll/bin/activate && ./install_dependencies.sh`
- [ ] Sanity-run ANEMLL's own Qwen 2.5 test (`python tests/test_qwen2.5_model.py`). _(skipped — our own conversion + run worked, which covers it)_
- [x] Run `anemll/utils/fp16_preflight.sh --model Qwen/Qwen2.5-Coder-0.5B` (check FP16 overflow risk).
- [x] `scripts/convert.sh`: convert with flags from §5.3 (FP16 first) into `models/qwen25coder-0.5b-ctx1024-fp16/`.
- [x] Use ANEMLL's ANE Profiler to confirm ops are placed on the **ANE**, not CPU/GPU.
- [x] Get a raw (no chat template) generation out of it. If `tests/chat.py` always applies a template, add a `--raw` path or write `eval/raw_generate.py` that calls the model directly.

Acceptance:
- Model loads, runs on ANE (profiler shows ANE placement for the main graph).
- A raw FIM prompt for `def fib(n):\n    <cursor>\n` produces sensible code.

Findings (2026-10-22, M3 / 16 GB / macOS 27.0.1; done before Phase 0 as a go/no-go check):
- ANEMLL tag `0.3.5_beta` (commit f4ad26d), cloned at `~/src/Anemll`, Python 3.9 env via uv. coremltools 9.0, torch 2.5.0.
- FP16 preflight: 100% compatible, no overflow, no clamp needed. (`fp16_preflight.sh` crashes on macOS bash 3.2 with an empty-array bug; run `python anemll/utils/fp16_compatibility_check.py --model ... --sweep` directly.)
- Conversion: 2 min 43 s. Output = 3 parts: embeddings 260 MB, FFN+prefill 685 MB, LM head 260 MB (~1.2 GB compiled).
- Profiler: FFN/prefill graph **99.5% ANE** (8 tiny GPU ops: casts/gather for position). Embeddings = GPU (just a gather). LM head 67% ANE (the per-chunk argmax ops fall back to GPU).
- Raw FIM for `def fib(n):` → correct recursive body, stopped on EOS by itself. `chat.py --no-template --prompt "..."` is a working raw path.
- Speed via Python `chat.py`: prefill 774-token prompt ≈ 340–360 ms (~2,200 tok/s); decode ≈ 51.5 tok/s (~19 ms/token).
- Implication: a 20-token completion costs ~390 ms of decode alone → decode, not prefill, is the bottleneck vs. the 300 ms p50 target. With KV reuse, prefill should drop to ~1 batch. Swift runtime (no Python per-token overhead) and short `n_predict` are the levers.
- Power (`scripts/measure_power.sh`, 20 s idle vs 20 s continuous generation, FP16, Python `chat.py`):

  | State | CPU mW | GPU mW | ANE mW | Total mW |
  |---|---|---|---|---|
  | Idle | 512 | 1 | 0 | 514 |
  | Generating | 577 | 2 | 2692 | 3272 |

- ctx 1024 vs 2048 (both FP16, `CONTEXT=2048 scripts/convert.sh fp16`, same 1.2 GB on disk, 64-token generations, Python `chat.py`):

  | Build | Prompt | Prefill | Decode | Peak RSS |
  |---|---|---|---|---|
  | ctx1024 | 774 tok | ~350 ms | 51.5 tok/s (19.4 ms/tok) | ~490 MB |
  | ctx2048 | 774 tok | ~465 ms | 43.8 tok/s (22.8 ms/tok) | ~540 MB |
  | ctx2048 | 1,367 tok | ~1,020 ms | 43.7 tok/s | ~540 MB |

  → 2048 costs ~15% decode speed and ~33% prefill on the same prompt; RAM is a non-issue (+50 MB; RSS may not count all Core ML/ANE-mapped memory). For the p50 2-token completion that's +7 ms. Full prefill of a ~1.4K prompt is ~1 s, so Phase 4 KV reuse is required either way. **Decision: use ctx2048** (fits every captured request without trimming).

  → ~2.8 W extra while generating (ctx1024), almost all on the ANE; GPU stays idle (the profiler's "GPU fallback" ops cost nothing measurable). ≈ 53 mJ/token at 52 tok/s → ~1 J per 20-token completion. Compare against llama-server GPU in Phase 0.

### Phase 2 — Correctness check (target: 1 day)

- [x] Write ~20 FIM cases in `eval/snippets/` (Python, TS, Swift, Go; mid-line, new line, inside function, inside call args). _(one file: `eval/snippets/cases.txt`, `=== name` headers, `<CURSOR>` marks the gap)_
- [x] `eval/compare_hf.py`: run each case through HF `transformers` (FP16/FP32, greedy) and through the ANE model (greedy). Compare outputs token by token.
- [x] If outputs diverge early: check tokenization (BOS, special tokens), RoPE, FP16 overflow, and ANEMLL version notes. _(not needed)_
- [x] Repeat with LUT6 build; record quality loss.

Acceptance: ≥ 18/20 cases identical for the first 16 tokens on FP16. Document LUT6 result.

Findings (2026-10-22, ctx2048 builds, HF reference = FP32 on CPU, greedy):
- **FP16: 19/20 identical** → pass. The one miss (`py_call_args`) is a near-tie: HF's top-2 logit margin was 0.115 at the first differing token (`obj, f, indent=4` vs `obj, f`). Both outputs are valid.
- The model often ends with `<|file_sep|>` or `<|fim_pad|>` instead of `<|endoftext|>` → those must be stop tokens (already in §7).
- Weak answers (e.g. JS-style `toUpperCase()` in Swift) are identical in HF → 0.5B model limits, not conversion error.
- **LUT6 (all parts, ctx2048): 16/20** → below the 18/20 bar. All 4 misses are near-ties (HF top-2 margin 0.20–0.51) and three are equally valid (`s.lower() == s.lower()[::-1]`, `x - other.x` without `self.`, indent difference). One is worse: `ts_switch_case` doesn't stop after `return …;` and runs on into a new function (the stop policy's newline/time cap would cut it).
- LUT6 vs FP16 speed (ctx2048, 774-token prompt, `chat.py`): **decode 72.6 vs 44.0 tok/s (13.8 vs 22.7 ms/token)**, prefill ~333 vs ~465 ms. Size 463 MB vs 1.2 GB. RSS about the same (~540 MB). Conversion 11 min (LUT) vs 3 min.
- Not tried yet: FFN-only LUT6 (`--lut2 6`, embeddings/LM head FP16) as a middle ground.

### Phase 3a — Python prototype server (target: 1–2 days)

- [ ] `server-py/server.py` (FastAPI, uvicorn), single model, single worker, one request at a time (serialize with a lock; cancel stale requests).
- [ ] `PromptBuilder`: trim to fit context — budget ≈ `context - n_predict - margin`; keep ~75% prefix (closest to cursor) and ~25% suffix; cut on line boundaries.
- [ ] Greedy decode loop with stop policy (§7).
- [ ] Implement `/infill` + `/health` exactly matching captured llama.vscode payloads.
- [ ] Point llama.vscode at it; confirm suggestions appear (slow is OK at this stage — no KV reuse yet).

Acceptance: llama.vscode shows ANE-generated suggestions end to end.

Status (2026-10-22): `server-py/` = `server.py` (stdlib `ThreadingHTTPServer`, no FastAPI), `engine.py` (KV reuse from day one: padded-batch rewind), `prompt.py` (line-boundary trimming with a stable "anchor" start line so the cache survives; suffix capped at 128 tokens). Default model LUT6 ctx2048, `--model fp16` fallback. Returns a **single object** even when `n_cmpl` > 1 (the extension's prefetch reads `.content` only). Warm-up (`n_predict: 0`) answered empty. Newer request cancels the one in flight. Not yet tried in VS Code.
- Replay of all 460 captured requests (`eval/bench_latency.py`): ANE p50 526 / p95 539 ms vs llama-server p50 30 / p95 549 ms. Prefill is fine (median 3 new tokens). **The ANE server generates ~36 tokens every time and hits the 500 ms cap**, where llama-server generates p50 2.
- Cause: llama.vscode asks for `samplers: [top_k, top_p, infill]`. llama.cpp's infill sampler ends the completion when the end-of-text probability is even modest (3·p_eog·n_candidates > p_text). Plain greedy only stops when EOS is top-1. HF check on 35 requests: greedy median 48 tokens, greedy + infill EOG rule 22, llama-server actual 2 (it also has input_extra context and samples at temperature 0.8).
- The EOG rule needs probabilities, but our `--argmax` build only returns per-chunk maxima → needs a build **without `--argmax`** (full logits). Converting: `CONTEXT=2048 ARGMAX=0 scripts/convert.sh lut6` → `models/qwen25coder-0.5b-ctx2048-lut6-logits/`.
- Added line rules (`make_stop_rule` in `server.py`): cursor line has code → finish that line only; cursor on an empty line → allow a block, stop at a blank line or a dedent below the cursor's indent. Replay: **p50 62 ms** (was 526), p95 532 ms, generated p50 2 / p95 33. Measured while a conversion was using the CPU. The remaining p95 is empty-line cases in near-empty files (e.g. a long `// 1. Write a function…` comment).
- Live use in VS Code (190 requests): model time p50 52 / p95 547 ms; stops: 154 line rule, 10 eos, 25 limit, 1 cancel. "Working pretty well."
- Tried the infill EOG rule (`pick_token` in `engine.py`, logits build `lut6-logits`, 12 min to convert). Replay: p50 56 ms, p95 530 (no p95 gain). Same build with the rule on vs off: 410/460 identical, **47 cut shorter** (mostly worse: lost closing `"`, `)`, `,`, sentences cut mid-way, 7 became empty), 3 different. Cost ~16 vs ~14 ms/token for full logits. **Decision: off by default** (`--eog-rule` opt-in); default model stays the argmax LUT6 build. llama.cpp gets away with it because it samples (temp 0.8) and returns 3 alternatives.
- Open: p95 ≈ the 500 ms hard cap, from long single-line generations on empty lines. Options: lower `--hard-ms`, or a per-line token cap in block mode. In live use, a newer keystroke cancels the request anyway.

### Phase 4 — KV-cache prefix reuse (target: 2–4 days) — **the critical piece**

Problem: every keystroke resends ~1K tokens; without reuse, every request pays full prefill.

Design (`KVCacheManager`):
1. Keep the model's KV state alive between requests, plus `cached_tokens: [Int]` (what the cache currently holds, in order).
2. On a new request, tokenize the full prompt → `new_tokens`.
3. `n_match` = length of the longest common prefix of `cached_tokens` and `new_tokens`.
4. Set current position = `n_match` (rewind). Prefill only `new_tokens[n_match:]`, in batches.
5. After generation, `cached_tokens = prompt + generated` (generated tokens are in the cache too).
6. If `new_tokens` exceeds context → full reset and re-prefill.
7. If `n_match == len(new_tokens)` (identical prompt resent, which is common), rewind to `n_match - 1` and re-run that last token. Otherwise there are no logits to sample from (llama.cpp does the same).

Why rewind works: causal mask hides positions ≥ current position, so stale entries are ignored and then overwritten. **Verified 2026-10-22** with `eval/kv_rewind_test.py` (spike, done before Phase 2). ANEMLL passes `position_ids`, `current_pos` and a row slice of one global `[ctx×ctx]` causal mask to every call, and the converted model writes KV at `current_pos` (`prefill_dynamic_slice: true`). The model has only `infer`/`prefill`, no rotate variants (Qwen, no sliding window).
- Replayed 40 consecutive real requests (ctx2048, `--with-extra`, prompts 1,192–1,506 tokens): **39/39 rewound outputs identical** to fresh full prefill (greedy, 16 tokens). A 30-request run with short prompts also gave 29/29.
- Rewound prefill: **median 28 ms, p95 33 ms** (median 3 new tokens) vs full prefill median ~1,090 ms.
- Trick: prefill the diff as one 64-token `prefill` call starting at `n_match`, padded with `<|fim_pad|>`. Padding KV lands past the prompt end, is masked, and is overwritten later. ANEMLL's own `run_prefill` always starts at 0 and feeds the leftover (<64) tokens one by one through `infer` (~18 ms each), which is why its full-prefill times jump around (560–1,560 ms). The padded batch is faster for cold prompts too (1,256 tokens: 560 ms vs 1,326 ms).
- `generate_next_token(pos=len)` re-runs the last prompt token itself, so only `ids[:-1]` needs prefilling; this also handles the identical-prompt case (step 7).
- After generation the cache holds `prompt + generated[:-1]` (the last generated token is never fed back).

Watch-outs:
- FIM prompt order is `prefix → suffix → middle`. Typing changes the **end of the prefix**, which sits *before* the suffix in the prompt → the suffix gets re-prefilled every keystroke. Options to evaluate:
  - (a) Accept it — suffix budget is small (~25%).
  - (b) Try SPM order (`<|fim_suffix|>{suf}<|fim_prefix|>{pre}<|fim_middle|>`; llama.cpp has `--spm-infill`) so the stable suffix comes first. **[VERIFY]** Qwen2.5-Coder quality in SPM mode — it may be trained PSM only.
  - (c) Shrink suffix to a few lines.
- Prefill runs in fixed batches (64): a 3-token diff still costs one batch call. Measured: ~28 ms per padded 64-batch, ~18 ms per single `infer`, so use `infer` only for 1-token diffs.
- Single-slot cache is fine (one editor, one user).

Tests:
- [x] Cached vs uncached generation must be **identical** for the 20 snippets (greedy).
- [x] Replay the Phase 0 request log: report `tokens_cached` ratio and p50/p95 latency.

Acceptance: identical outputs; median prefill tokens per request < 100 on the replay log. **Passed.**

Findings (2026-10-22, done together with Phase 3a; implemented in `server-py/engine.py` + `prompt.py`):
- `eval/check_cache.py` (server engine, LUT6): simulates typing the last 6 chars of each snippet with the cache reused every step → **20/20 cached == uncached**.
- `eval/kv_rewind_test.py` (spike): 39/39 on real long prompts, 29/29 short (see design notes above).
- Replay of 460 real requests via the server: new prompt tokens p50 3 / p95 51; latency p50 ~55–62 ms (see Phase 3a for p95).
- Suffix watch-out: chose (c), suffix capped at 128 tokens (`--suffix-tokens`). Live log shows ~116 suffix tokens re-prefilled per keystroke ≈ 45–50 ms. (b) SPM order is untested.
- Prompt start stability: `PromptBuilder` keeps an "anchor" start line so trimming long files doesn't invalidate the cache every keystroke.

### Phase 3b — Port server to Swift (target: 2–3 days)

- [ ] Swift package depending on local `anemll-swift-cli` (AnemllCore) as a local package.
- [ ] Hummingbird (or NIO) HTTP server; same routes and JSON as the Python prototype.
- [ ] Port `PromptBuilder`, `KVCacheManager`, `StopPolicy`.
- [ ] Use a serial queue for all model calls (ANEMLL notes ANE race issues; they use serial prediction queues).
- [ ] Cancel in-flight generation when a newer request arrives.

Acceptance: same outputs as Python server on the eval set; lower latency than Python.

### Phase 5 — Measure & compare (target: 1 day)

- [ ] Replay Phase 0 log against: llama-server (GPU), ane-fim Python, ane-fim Swift.
- [ ] Record p50/p95 latency, avg power (CPU/GPU/ANE W), memory, and a quick quality check (accept-rate feel over a day of real use).

| Setup | p50 ms | p95 ms | Avg power W | RAM | Notes |
|---|---|---|---|---|---|
| llama-server GPU (baseline) | 24 | 449 | 8.09 (CPU 1.23, GPU 6.86) | 627 MB RSS | Q8 GGUF; does 3 completions/request + input_extra context |
| ane-fim Python | 64 | 532 | 4.37 (CPU 0.64, ANE 3.73) | 58 MB RSS* | LUT6 ctx2048, line rules, 1 completion/request |
| ane-fim Swift | | | | | |

Measured 2026-10-22 with `scripts/power_compare.sh` (30 s continuous replay of the 460 captured requests per server; idle with both loaded = 0.13 W). *Process RSS doesn't count Core ML/ANE model memory, so the true footprint is higher (chat.py peaked at ~540 MB).

| Setup | req/s | Energy per completion | Extra over idle |
|---|---|---|---|
| llama-server GPU | 11.4 | 0.708 J | 0.696 J |
| ane-fim Python | 8.3 | **0.525 J** | **0.509 J** |

→ Under load the ANE server draws **~46% less power** (4.4 vs 8.1 W) and uses **~27% less energy per completion**. The GPU stays at 0 W with the ANE server. Caveats: as-deployed comparison (llama-server does more work per request: 3 completions + extra context); continuous replay is harsher than real bursty typing. Python overhead is the CPU's 0.64 W; a Swift port could trim some of it.

Success targets: p50 < 300 ms on 1K context; clearly lower total power than baseline.

### Phase 6 — Polish (optional)

- [ ] `launchd` user agent to auto-start the server at login; warm up the model on start.
- [ ] Unload after N minutes idle; reload on first request.
- [ ] Use `input_extra` → repo-level prompt with `<|repo_name|>` / `<|file_sep|>` (budget carefully, ctx is small).
- [ ] Import-aware context. llama.vscode's snippets are not import-aware: it only adds chunks from the current file outside the prefix/suffix window, from files on save (around the cursor, or a random 32 lines), and from copied text. So a class imported from a far-away file usually isn't in context. Idea: the server parses import lines in the prefix, finds those files on disk (local server; `input_extra` filenames give absolute paths to anchor the project root), and adds a compact **outline** (class + method signatures, no bodies, ~50–150 tokens each) as `<|file_sep|>path` entries. Catch: the current file's path is not in the request, so relative imports (`from ..x import Y`) need guessing; absolute/package imports are easy. Workaround until then: copy the class definition (`Cmd+C`) to push it into the extension's snippets.
- [ ] Add `/v1/completions` (`prompt` + `suffix`) for Continue.
- [ ] Try 1.5B model if latency allows.
- [ ] README with setup steps.

---

### Distribution (started 2026-10-23)

- [x] Public GitHub repo: https://github.com/SorataBaka/ane-fim. Captured editor traffic, logs, raw powermetrics dumps and `experiments/` are gitignored (personal data). No LICENSE file yet (owner to choose).
- [x] HF packages prepared locally (not uploaded) in `models/hf/`: `Qwen2.5-Coder-0.5B-ANE-ctx2048-LUT6` (474 MB) and `-FP16` (1.2 GB). Built from `prepare_hf.sh`, then flattened (no `ios/` subfolder), ANEMLL chat scripts removed, local-path comment block stripped from `meta.yaml`, Qwen's Apache-2.0 `LICENSE` added, custom model card (ANEMLL's template wrongly says MIT). Both verified to load and run from their folders.
- [ ] Upload: `hf upload tianharjuno/Qwen2.5-Coder-0.5B-ANE-ctx2048-LUT6 models/hf/Qwen2.5-Coder-0.5B-ANE-ctx2048-LUT6 .` (same for FP16).
- [ ] Swift port (3b) → single binary → Homebrew tap (`brew services` covers auto-start). Test on other chips/macOS versions.

## 7. Stop policy (generation)

Stop when any is true:
- token is `<|endoftext|>`, `<|fim_pad|>`, `<|file_sep|>`, `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|repo_name|>`
- `n_predict` reached (default 64; cap server-side at ~32–48 even if the client asks 128: on ANE, 128 tokens ≈ 3 s)
- `t_max_predict_ms` elapsed **and** at least one newline generated
- hard time cap (~300–500 ms) even with no newline, plus a repeat check (same token/short pattern N times in a row) to kill `000…` runs
- `n_cmpl > 1`: generate once (greedy → alternatives are identical)
- generated text starts repeating the beginning of the suffix (trim the overlap)

Post-process: strip trailing whitespace-only lines; never return special tokens in `content`.

---

## 8. Commands (fill in exact ones as they're confirmed)

```bash
# ANEMLL env
cd ~/src/Anemll && source env-anemll/bin/activate

# Convert (confirmed) — wraps convert_model.sh, output in ./models/
scripts/convert.sh fp16      # or: scripts/convert.sh lut6
# Single-file model instead of 3 parts: anemll/utils/convert_monolith.sh (not tried yet)

# Raw FIM generation (confirmed)
python tests/chat.py --meta <model_dir>/meta.yaml --no-template --max-tokens 48 \
  --prompt $'<|fim_prefix|>def fib(n):\n    <|fim_suffix|>\n<|fim_middle|>'

# ANE placement check (confirmed)
python anemll/utils/ane_profiler.py --model <model_dir>/qwen25_FFN_PF_chunk_01of01.mlpackage --analyze

# Baseline
llama-server -hf <qwen2.5-coder-0.5b GGUF repo> --port 8012 --cache-reuse 256

# Power
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500
```

---

## 9. Known risks

1. **ANE may not be faster than the GPU** for a 0.5B model. The win is power/thermals. That's fine — don't chase speed past the targets.
2. **KV rewind may not work** cleanly with ANEMLL's state handling → fallback: re-prefill but shrink context to 512.
3. **ANEMLL is beta**, partly built on undocumented behavior; pin the version, expect breakage on macOS updates.
4. **Fixed context**: changing context length means re-converting.
5. **Tokenizer mismatch** (BOS/template) silently ruins quality — Phase 2 exists to catch this.
6. Memory: README states 16 GB minimum; check actual RSS on this Mac.

---

## 10. References

- ANEMLL (upstream): https://github.com/Anemll/Anemll — docs: `docs/convert.md`, `docs/convert_model.md`, `docs/swift_cli.md`, `anemll/utils/ANE_PROFILER.md`
- Pre-converted ANEMLL models: https://huggingface.co/anemll
- anemll-server (chat-only reference server): https://github.com/alexgusevski/anemll-server
- llama.cpp server API (`/infill`, `cache_prompt`, `--cache-reuse`, `--spm-infill`): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.vscode: https://github.com/ggml-org/llama.vscode
- Qwen2.5-Coder (FIM + repo-level formats): https://github.com/QwenLM/Qwen2.5-Coder
- Apple ANE transformer guidance: https://github.com/apple/ml-ane-transformers
