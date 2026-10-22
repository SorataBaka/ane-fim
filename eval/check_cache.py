"""Phase 4 test: the server engine's cached path must match a cold (uncached) run.

For each snippet in eval/snippets/cases.txt, simulates typing: the prefix grows one
character at a time over its last few characters, so every step reuses the cache
from the previous one (stale KV beyond the rewind point included). The final
output is compared with a cold run on a fresh cache.

    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/eval/check_cache.py [--model lut6|fp16]
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "server-py"))
sys.path.insert(0, str(PROJECT_DIR / "eval"))
from engine import Engine  # noqa: E402
from prompt import PromptBuilder  # noqa: E402
from compare_hf import load_cases  # noqa: E402

TYPED_CHARS = 6
MAX_TOKENS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lut6")
    a = ap.parse_args()

    eng = Engine(PROJECT_DIR / f"models/qwen25coder-0.5b-ctx2048-{a.model}/meta.yaml")
    pb = PromptBuilder(eng.tokenizer, eng.ctx)
    passed = 0
    cases = load_cases()
    for name, prefix, suffix in cases:
        # Warm path: type the last few characters one by one, reusing the cache each step.
        for k in range(TYPED_CHARS, -1, -1):
            ids = pb.build(prefix[:len(prefix) - k], suffix, MAX_TOKENS)
            warm, info = eng.complete(ids, MAX_TOKENS)
        new_tokens = info["prompt_n"]

        # Cold path: fresh cache, same final prompt.
        eng.cached = []
        cold, _ = eng.complete(ids, MAX_TOKENS)

        same = warm == cold
        passed += same
        print(f"{'PASS' if same else 'FAIL'}  {name:28s} last step prefilled {new_tokens:3d} new tokens  "
              f"{eng.tokenizer.decode(warm)[:40]!r}")
        if not same:
            print(f"      cold: {eng.tokenizer.decode(cold)[:60]!r}")

    print(f"\n{passed}/{len(cases)} cached == uncached")


if __name__ == "__main__":
    main()
