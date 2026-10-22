"""Objective model comparison: complete the rest of a real line of code.

Samples random cut points in real source files (default: the ANEMLL repo, written after
Qwen2.5-Coder's training data), blanks out the rest of that line, and asks each model to
fill it in. Scores exact match and edit similarity against the real text.

    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/eval/line_completion.py [--sizes 0.5b,1.5b] [--n 150] [--corpus ~/src/Anemll]
"""

import argparse
import difflib
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "server-py"))
from engine import Engine  # noqa: E402
from prompt import PromptBuilder  # noqa: E402

EXTS = {".py": "Python", ".swift": "Swift"}
MAX_TOKENS = 48
MAX_FILE_CHARS = 200_000


def sample_cases(corpus, n, seed):
    files = [p for p in corpus.rglob("*") if p.suffix in EXTS and p.is_file()
             and not any(part.startswith(".") or part in ("env-anemll", "build", "node_modules") for part in p.parts)]
    files = [p for p in files if 2_000 < p.stat().st_size < MAX_FILE_CHARS]
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        path = rng.choice(files)
        text = path.read_text(errors="ignore")
        lines = text.split("\n")
        candidates = [i for i, l in enumerate(lines) if len(l.strip()) >= 12 and not l.strip().startswith(("#", "//"))]
        if not candidates:
            continue
        i = rng.choice(candidates)
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        # Cut somewhere in the line: at the indent (whole line) or mid-line.
        cut = indent if rng.random() < 0.3 else rng.randint(indent + 1, len(line) - 4)
        prefix = "\n".join(lines[:i]) + "\n" + line[:cut]
        suffix = "\n" + "\n".join(lines[i + 1:])
        cases.append((f"{path.relative_to(corpus)}:{i + 1}", EXTS[path.suffix], prefix, suffix, line[cut:]))
    return cases


def run(engine, cases):
    pb = PromptBuilder(engine.tokenizer, engine.ctx)
    results = []
    for _, _, prefix, suffix, _ in cases:
        pb.anchor, engine.cached = None, []  # every case is a different file: start cold
        ids = pb.build(prefix, suffix, MAX_TOKENS)
        t0 = time.perf_counter()
        out, _ = engine.complete(ids, MAX_TOKENS, stop_text=lambda t: "\n" in t)
        ms = (time.perf_counter() - t0) * 1000
        results.append((engine.tokenizer.decode(out, skip_special_tokens=True).split("\n")[0], ms))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.5b,1.5b")
    ap.add_argument("--model", default="lut6")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=str(Path.home() / "src/Anemll"))
    ap.add_argument("--show", type=int, default=12, help="print this many cases where the models differ")
    a = ap.parse_args()

    cases = sample_cases(Path(a.corpus).expanduser(), a.n, a.seed)
    sizes = a.sizes.split(",")
    outputs = {}
    for size in sizes:
        eng = Engine(PROJECT_DIR / f"models/qwen25coder-{size}-ctx2048-{a.model}/meta.yaml")
        outputs[size] = run(eng, cases)
        del eng

    print(f"{len(cases)} cases from {a.corpus}  (exact = matches the real rest of line, ignoring trailing spaces)\n")
    print(f"{'model':8s} {'lang':7s} {'exact':>7s} {'similarity':>11s} {'empty':>6s} {'median ms':>10s}")
    for size in sizes:
        for lang in ["all"] + sorted(set(EXTS.values())):
            idx = [k for k, c in enumerate(cases) if lang == "all" or c[1] == lang]
            exact = sum(outputs[size][k][0].rstrip() == cases[k][4].rstrip() for k in idx)
            sim = sum(difflib.SequenceMatcher(None, outputs[size][k][0].strip(), cases[k][4].strip()).ratio() for k in idx)
            empty = sum(not outputs[size][k][0].strip() for k in idx)
            ms = sorted(outputs[size][k][1] for k in idx)[len(idx) // 2]
            print(f"{size:8s} {lang:7s} {exact / len(idx):6.0%} {sim / len(idx):11.2f} {empty:6d} {ms:10.0f}")

    if len(sizes) == 2:
        s0, s1 = sizes
        wins = {s0: 0, s1: 0}
        shown = 0
        print(f"\nCases where exactly one model got it right:")
        for k, (name, lang, prefix, suffix, truth) in enumerate(cases):
            ok0 = outputs[s0][k][0].rstrip() == truth.rstrip()
            ok1 = outputs[s1][k][0].rstrip() == truth.rstrip()
            if ok0 != ok1:
                wins[s0 if ok0 else s1] += 1
                if shown < a.show:
                    shown += 1
                    print(f"  {name}  ...{prefix.split(chr(10))[-1][-40:]!r}")
                    print(f"      real : {truth.strip()[:70]!r}")
                    print(f"      {s0:5s}: {outputs[s0][k][0].strip()[:70]!r}")
                    print(f"      {s1:5s}: {outputs[s1][k][0].strip()[:70]!r}")
        print(f"  only {s0} right: {wins[s0]}   only {s1} right: {wins[s1]}")


if __name__ == "__main__":
    main()
