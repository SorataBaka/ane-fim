"""ANE model wrapper: one KV cache kept between requests, rewound to the common prefix.

Uses ANEMLL's tests/chat.py for model loading and single-token decode.
See CLAUDE.md Phase 4 for why rewinding is safe.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ANEMLL_DIR = Path(os.environ.get("ANEMLL_DIR", Path.home() / "src/Anemll"))
sys.path.insert(0, str(ANEMLL_DIR / "tests"))
import chat  # noqa: E402  (ANEMLL's tests/chat.py)

FIM_PAD = 151662
# <|endoftext|>, <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>, <|fim_pad|>, <|repo_name|>, <|file_sep|>
STOP_IDS = {151643, 151659, 151660, 151661, 151662, 151663, 151664}


STOP_ARRAY = np.array(sorted(STOP_IDS))


def pick_token(logits, eog_rule, top_k=40, top_p=0.99):
    """Greedy pick from full logits. With eog_rule, mimic llama.cpp's infill sampler:
    end the completion if end-of-text tokens hold a modest share of the top_k/top_p
    candidates (3 * p_eog * n_candidates > p_text). Returns a STOP id to end."""
    if not eog_rule:
        return int(np.argmax(logits))
    idx = np.argpartition(-logits, top_k)[:top_k]
    idx = idx[np.argsort(-logits[idx])]
    p = np.exp(logits[idx] - logits[idx[0]])
    p /= p.sum()
    keep = int(np.searchsorted(np.cumsum(p), top_p)) + 1
    idx, p = idx[:keep], p[:keep]
    is_eog = np.isin(idx, STOP_ARRAY)
    p_eog, p_txt = p[is_eog].sum(), p[~is_eog].sum()
    if 3 * p_eog * len(p) > p_txt:
        return int(idx[is_eog][0])
    return int(idx[~is_eog][0])


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def repeated_tail(tokens, reps=6, max_period=4):
    """Length of a degenerate repeating tail (e.g. 000000...), or 0 if none."""
    for p in range(1, max_period + 1):
        if len(tokens) >= p * reps and tokens[-p * reps:] == tokens[-p:] * reps:
            n = p * reps
            while len(tokens) >= n + p and tokens[-n - p:-n] == tokens[-p:]:
                n += p
            return n
    return 0


class Engine:
    def __init__(self, meta_path):
        sys.argv = ["chat.py", "--meta", str(meta_path), "--eval", "--nw"]
        args = chat.parse_args()
        model_dir = Path(args.d).resolve()
        self.tokenizer = chat.initialize_tokenizer(str(model_dir), True)
        args.embed = str(model_dir / args.embed)
        args.ffn = str(model_dir / args.ffn)
        args.lmhead = str(model_dir / args.lmhead)
        self.embed, self.ffns, self.lmhead, md = chat.load_models(args, {})
        md["split_lm_head"] = getattr(args, "split_lm_head", 8)
        md["argmax_in_model"] = getattr(args, "argmax_in_model", False)
        md["lm_head_chunk_sizes"] = chat._parse_lm_head_chunk_sizes(getattr(args, "lm_head_chunk_sizes", None))
        md["attention_size"] = md["context_length"]
        md["sliding_window"] = None
        self.metadata = md
        self.has_logits = not md["argmax_in_model"]  # needed for the infill EOG rule
        self.ctx = md["context_length"]
        self.batch = md.get("batch_size", 64)
        self.causal_mask = chat.initialize_causal_mask(self.ctx, True)
        self.state = chat.create_unified_state(self.ffns, self.ctx, True, metadata=md)
        self.cached = []  # tokens whose KV is valid in self.state, in order

    def _chunk_state(self, i):
        """KV state for FFN chunk i: usually one state shared by all chunks, a list for older conversions."""
        return self.state[i] if isinstance(self.state, list) else self.state

    def _prefill(self, ids, start, end):
        """Write KV for ids[start:end]. Diffs go in one padded 64-batch (~28 ms); 1 token uses infer."""
        pos = start
        while pos < end:
            n = min(self.batch, end - pos)
            if n > 1 and pos + self.batch <= self.ctx:
                chunk = ids[pos:pos + n] + [FIM_PAD] * (self.batch - n)
                fn, size = "prefill", self.batch
            else:
                chunk, fn, size, n = ids[pos:pos + 1], "infer", 1, 1
            hidden = self.embed.predict({"input_ids": np.array([chunk], dtype=np.int32)})["hidden_states"]
            for i, ffn in enumerate(self.ffns):
                hidden = ffn[fn].predict({
                    "hidden_states": hidden.astype(np.float16),
                    "position_ids": np.arange(pos, pos + size, dtype=np.int32),
                    "causal_mask": self.causal_mask[:, :, pos:pos + size, :].numpy().astype(np.float16),
                    "current_pos": np.array([pos], dtype=np.int32),
                }, self._chunk_state(i))["output_hidden_states"]
            pos += n

    def _next_token(self, cur, eog_rule):
        """Feed cur[-1] at position len(cur)-1 (writes its KV) and pick the next token."""
        if not self.has_logits:  # argmax build: ANEMLL's own path, plain greedy
            return chat.generate_next_token(self.embed, self.ffns, self.lmhead, torch.tensor([cur], dtype=torch.int32),
                                            len(cur), self.ctx, self.metadata, self.state, self.causal_mask)
        pos = len(cur) - 1
        hidden = self.embed.predict({"input_ids": np.array([[cur[-1]]], dtype=np.int32)})["hidden_states"]
        for i, ffn in enumerate(self.ffns):
            hidden = ffn["infer"].predict({
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": self.causal_mask[:, :, pos:pos + 1, :].numpy().astype(np.float16),
                "current_pos": np.array([pos], dtype=np.int32),
            }, self._chunk_state(i))["output_hidden_states"]
        out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        n = self.metadata["split_lm_head"]
        logits = np.concatenate([out[f"logits{i}"].reshape(-1) for i in range(1, n + 1)]).astype(np.float32)
        return pick_token(logits, eog_rule)

    def complete(self, ids, max_tokens, t_max_predict_ms=None, hard_ms=None, cancelled=lambda: False,
                 stop_text=None, eog_rule=True):
        """Greedy FIM completion with KV reuse. Returns (tokens, info dict).

        stop_text(decoded_text) -> True ends generation early (e.g. line/block rules).
        eog_rule: llama.cpp infill-style early end-of-text (logits builds only).
        """
        assert 1 < len(ids) < self.ctx - 1
        n_match = min(common_prefix(self.cached, ids), len(ids) - 1)
        self.cached = ids[:n_match]  # stays correct if anything below fails midway

        t0 = time.perf_counter()
        self._prefill(ids, n_match, len(ids) - 1)
        self.cached = ids[:-1]
        prompt_ms = (time.perf_counter() - t0) * 1000

        cur, out, stop_type = list(ids), [], "limit"
        t0 = time.perf_counter()
        while len(cur) < self.ctx - 1:
            if cancelled():
                stop_type = "cancel"
                break
            tok = self._next_token(cur, eog_rule)
            self.cached = list(cur)  # that call wrote KV for cur[-1]
            if tok in STOP_IDS:
                stop_type = "eos"
                break
            out.append(tok)
            cur.append(tok)

            text = self.tokenizer.decode(out, skip_special_tokens=True)
            if stop_text and stop_text(text):
                stop_type = "word"
                break
            elapsed = (time.perf_counter() - t0) * 1000
            if len(out) >= max_tokens or (hard_ms and elapsed > hard_ms):
                break
            if t_max_predict_ms and elapsed > t_max_predict_ms and "\n" in text:
                break
            rep = repeated_tail(out)
            if rep:
                del out[-rep:]
                stop_type = "word"
                break

        return out, {
            "tokens_cached": n_match,
            "prompt_n": len(ids) - 1 - n_match,
            "prompt_ms": prompt_ms,
            "predicted_n": len(out),
            "predicted_ms": (time.perf_counter() - t0) * 1000,
            "stop_type": stop_type,
        }
