"""Phase 2: compare ANE (Core ML) output against Hugging Face transformers, greedy.

Runs each case in eval/snippets/cases.txt through HF (FP32, CPU) and the ANE model,
compares the first 16 generated tokens. On a mismatch it prints HF's top-2 logit
margin at the first differing token, so near-ties are easy to spot.

Run inside the ANEMLL env:
    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/eval/compare_hf.py --meta <model_dir>/meta.yaml
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_rewind_test as ane  # noqa: E402  (shares ANEMLL loading/prefill/generate code)

HF_MODEL = "Qwen/Qwen2.5-Coder-0.5B"
CASES = Path(__file__).resolve().parent / "snippets/cases.txt"
N_TOKENS = 16


def load_cases():
    text = CASES.read_text()
    cases = []
    for block in re.split(r"^=== ", text, flags=re.M)[1:]:
        name, body = block.split("\n", 1)
        prefix, suffix = body.split("<CURSOR>")
        cases.append((name.strip(), prefix, suffix))
    return cases


def hf_generate(model, ids):
    """Greedy decode; returns (tokens, per-step top-2 margins)."""
    out, margins = [], []
    past, inp = None, torch.tensor([ids])
    with torch.no_grad():
        for _ in range(N_TOKENS):
            res = model(input_ids=inp, past_key_values=past, use_cache=True)
            past, logits = res.past_key_values, res.logits[0, -1].float()
            top2 = torch.topk(logits, 2)
            tok = int(top2.indices[0])
            out.append(tok)
            margins.append(float(top2.values[0] - top2.values[1]))
            if tok in ane.STOP_IDS:
                break
            inp = torch.tensor([[tok]])
    return out, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--hf-model", default=HF_MODEL, help="HF reference, e.g. Qwen/Qwen2.5-Coder-1.5B")
    a = ap.parse_args()

    tokenizer, embed, ffns, lmhead, metadata = ane.load(a.meta)
    ctx, batch = metadata["context_length"], metadata.get("batch_size", 64)
    causal_mask = ane.chat.initialize_causal_mask(ctx, True)
    hf = AutoModelForCausalLM.from_pretrained(a.hf_model, dtype=torch.float32).eval()
    ane.MAX_NEW = N_TOKENS

    passed = 0
    cases = load_cases()
    for name, prefix, suffix in cases:
        ids = ane.build_prompt(tokenizer, {"input_prefix": prefix, "input_suffix": suffix}, ctx - 128)

        state = ane.chat.create_unified_state(ffns, ctx, True, metadata=metadata)
        ane.prefill_range(embed, ffns, ids, 0, len(ids) - 1, ctx, causal_mask, state, batch, "padded")
        ane_out = ane.generate(embed, ffns, lmhead, metadata, ids, state, causal_mask, ctx)
        hf_out, margins = hf_generate(hf, ids)

        # Any stop token ends the completion the same way, so <|fim_pad|> == <|endoftext|> here.
        norm = lambda toks: [-1 if t in ane.STOP_IDS else t for t in toks]
        same = norm(ane_out) == norm(hf_out)
        passed += same
        print(f"{'PASS' if same else 'FAIL'}  {name:28s} {repr(tokenizer.decode(hf_out))[:70]}")
        if not same:
            k = next((i for i, (x, y) in enumerate(zip(ane_out, hf_out)) if x != y), min(len(ane_out), len(hf_out)))
            print(f"      first diff at token {k}; HF top-2 margin there: "
                  f"{margins[k] if k < len(margins) else float('nan'):.3f}")
            print(f"      ane: {repr(tokenizer.decode(ane_out))[:90]}")

    print(f"\n{passed}/{len(cases)} identical for the first {N_TOKENS} tokens  (target: >= 18/20)")


if __name__ == "__main__":
    main()
