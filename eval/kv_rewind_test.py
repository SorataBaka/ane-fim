"""Feasibility check for Phase 4: can ANEMLL's KV cache be rewound and reused?

Replays consecutive captured llama.vscode requests. For each request:
  reference: fresh state, full prefill (ANEMLL's own run_prefill), greedy generate
  rewind:    keep the state from the previous request, rewind to the common token
             prefix, prefill only the new tokens, greedy generate
and checks that both produce the same tokens. Also times the partial prefill.

Run inside the ANEMLL env:
    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/eval/kv_rewind_test.py --meta <model_dir>/meta.yaml [--n 30] [--mode padded|single]
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ANEMLL_DIR = Path(os.environ.get("ANEMLL_DIR", Path.home() / "src/Anemll"))
sys.path.insert(0, str(ANEMLL_DIR / "tests"))
import chat  # noqa: E402  (ANEMLL's tests/chat.py)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FIM_PAD, FILE_SEP = 151659, 151661, 151660, 151662, 151664
STOP_IDS = {151643, 151659, 151660, 151661, 151662, 151663, 151664}
MAX_NEW = 16


def load(meta_path):
    sys.argv = ["chat.py", "--meta", meta_path, "--eval", "--nw"]
    args = chat.parse_args()
    model_dir = Path(args.d).resolve()
    tokenizer = chat.initialize_tokenizer(str(model_dir), True)
    args.embed = str(model_dir / args.embed)
    args.ffn = str(model_dir / args.ffn)
    args.lmhead = str(model_dir / args.lmhead)
    embed, ffns, lmhead, metadata = chat.load_models(args, {})
    metadata["split_lm_head"] = getattr(args, "split_lm_head", 8)
    metadata["argmax_in_model"] = getattr(args, "argmax_in_model", False)
    metadata["lm_head_chunk_sizes"] = chat._parse_lm_head_chunk_sizes(getattr(args, "lm_head_chunk_sizes", None))
    metadata["attention_size"] = metadata["context_length"]
    metadata["sliding_window"] = None
    return tokenizer, embed, ffns, lmhead, metadata


def build_prompt(tokenizer, req, budget, with_extra=False):
    """FIM prompt from a captured request, trimmed to budget tokens.

    with_extra: put input_extra chunks first in repo-level format (<|file_sep|>name\\ntext),
    like llama-server does, dropping the oldest chunks if over budget.
    """
    enc = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    prefix = enc(req["input_prefix"] + req.get("prompt", ""))
    suffix = enc(req["input_suffix"])
    suffix = suffix[: budget // 4]
    prefix = prefix[-(budget - len(suffix) - 3):]
    fim = [FIM_PREFIX] + prefix + [FIM_SUFFIX] + suffix + [FIM_MIDDLE]
    if not with_extra:
        return fim
    chunks = [[FILE_SEP] + enc(f"{c.get('filename', '')}\n{c.get('text', '')}") for c in req.get("input_extra", [])]
    while chunks and sum(map(len, chunks)) + len(fim) > budget:
        chunks.pop(0)
    return [t for c in chunks for t in c] + fim


def prefill_range(embed, ffns, ids, start, end, ctx, causal_mask, state, batch, mode):
    """Write KV for ids[start:end] at positions start..end-1, leaving earlier cache untouched.

    padded: 64-token prefill calls from `start`; a short last batch is padded with
            <|fim_pad|> (its KV lands past the prompt end: masked, overwritten later).
    single: one infer call per token.
    """
    pos = start
    while pos < end:
        n = min(batch, end - pos)
        if mode == "padded" and pos + batch <= ctx and n > 1:
            chunk = ids[pos:pos + n] + [FIM_PAD] * (batch - n)
            fn, size = "prefill", batch
        else:
            chunk, fn, size, n = ids[pos:pos + 1], "infer", 1, 1
        hidden = embed.predict({"input_ids": np.array([chunk], dtype=np.int32)})["hidden_states"]
        for ffn in ffns:
            out = ffn[fn].predict({
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.arange(pos, pos + size, dtype=np.int32),
                "causal_mask": causal_mask[:, :, pos:pos + size, :].numpy().astype(np.float16),
                "current_pos": np.array([pos], dtype=np.int32),
            }, state)
            hidden = out["output_hidden_states"]
        pos += n


def generate(embed, ffns, lmhead, metadata, ids, state, causal_mask, ctx):
    ids = torch.tensor([ids], dtype=torch.int32)
    pos, out = ids.size(1), []
    while len(out) < MAX_NEW and pos < ctx - 1:
        tok = chat.generate_next_token(embed, ffns, lmhead, ids, pos, ctx, metadata, state, causal_mask)
        out.append(tok)
        if tok in STOP_IDS:
            break
        ids = torch.cat([ids, torch.tensor([[tok]], dtype=torch.int32)], dim=1)
        pos += 1
    return out


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=30, help="number of consecutive requests to replay")
    ap.add_argument("--start", type=int, default=0, help="index of first request in the capture log")
    ap.add_argument("--mode", choices=["padded", "single"], default="padded")
    ap.add_argument("--with-extra", action="store_true", help="include input_extra chunks (long prompts)")
    a = ap.parse_args()

    tokenizer, embed, ffns, lmhead, metadata = load(a.meta)
    ctx, batch = metadata["context_length"], metadata.get("batch_size", 64)
    causal_mask = chat.initialize_causal_mask(ctx, True)
    budget = ctx - MAX_NEW - 64

    files = sorted(glob.glob(str(PROJECT_DIR / "eval/llama_vscode_requests/*.json")))
    reqs = [json.load(open(f))["request"] for f in files]
    reqs = [r for r in reqs if r.get("n_predict") != 0][a.start:a.start + a.n]

    live_state = chat.create_unified_state(ffns, ctx, True, metadata=metadata)
    cached = []  # tokens whose KV is currently valid in live_state (prompt + generated)
    rows = []
    for i, req in enumerate(reqs):
        ids = build_prompt(tokenizer, req, budget, a.with_extra)

        # Reference: fresh state, ANEMLL's own full prefill.
        ref_state = chat.create_unified_state(ffns, ctx, True, metadata=metadata)
        t0 = time.perf_counter()
        chat.run_prefill(embed, ffns, torch.tensor([ids], dtype=torch.int32), len(ids), ctx, batch,
                         ref_state, causal_mask, None, use_update_mask=False)
        full_ms = (time.perf_counter() - t0) * 1000
        ref_out = generate(embed, ffns, lmhead, metadata, ids, ref_state, causal_mask, ctx)

        # Rewind: reuse live_state; generate_next_token re-runs the last prompt token itself,
        # so we only need KV for ids[:-1] (also covers the identical-prompt case).
        n_match = min(common_prefix(cached, ids), len(ids) - 1)
        t0 = time.perf_counter()
        prefill_range(embed, ffns, ids, n_match, len(ids) - 1, ctx, causal_mask, live_state, batch, a.mode)
        part_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        rew_out = generate(embed, ffns, lmhead, metadata, ids, live_state, causal_mask, ctx)
        gen_ms = (time.perf_counter() - t0) * 1000
        # KV now holds ids + all generated tokens except the last (it was never fed back in).
        cached = ids + rew_out[:-1]

        same = ref_out == rew_out
        rows.append((len(ids), len(ids) - 1 - n_match, full_ms, part_ms, gen_ms, len(rew_out), same))
        print(f"{i:3d} prompt {len(ids):5d}  new {len(ids) - 1 - n_match:5d}  full {full_ms:7.1f} ms  "
              f"partial {part_ms:7.1f} ms  gen {gen_ms:6.1f} ms ({len(rew_out)} tok)  "
              f"{'MATCH' if same else 'DIFF'}", flush=True)
        if not same:
            print("      ref:", repr(tokenizer.decode(ref_out)))
            print("      rew:", repr(tokenizer.decode(rew_out)))

    med = lambda xs: float(np.median(xs))
    print(f"\nmode={a.mode}  ctx={ctx}  requests={len(rows)}")
    print(f"identical outputs: {sum(r[6] for r in rows[1:])}/{len(rows) - 1} (excluding the cold first request)")
    print(f"new tokens/request: median {med([r[1] for r in rows[1:]]):.0f}")
    print(f"full prefill: median {med([r[2] for r in rows]):.0f} ms | partial prefill: median "
          f"{med([r[3] for r in rows[1:]]):.0f} ms, p95 {np.percentile([r[3] for r in rows[1:]], 95):.0f} ms")
    print(f"generation ({MAX_NEW} max tok): median {med([r[4] for r in rows]):.0f} ms")


if __name__ == "__main__":
    main()
