"""Objective comparison: complete the rest of a real line of code.

Samples random cut points in real source files (default: the ANEMLL repo, written after
Qwen2.5-Coder's training data), blanks out the rest of that line, and asks the model to
fill it in. Scores exact match and edit similarity against the real text.

Variants:
  none  plain FIM prompt (what the server does today)
  fmt   Qwen's repo-level header with no other files (does the format alone change anything?)
  ctx   + cross-file outlines from server-py/repo_index.py in Qwen's repo-level format
        (same total token budget: context tokens come out of the prefix; plain FIM if nothing found)

--xfile-only samples only lines whose answer uses a name defined in another file of the repo
(the case cross-file context is meant for, like CrossCodeEval). Without it, such cases are
still reported separately as the "xfile" row.

    cd ~/src/Anemll && source env-anemll/bin/activate
    python <project>/eval/line_completion.py [--sizes 1.5b] [--variants none,ctx] [--n 150]
"""

import argparse
import difflib
import random
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "server-py"))
from engine import Engine  # noqa: E402
from prompt import PromptBuilder  # noqa: E402
from repo_index import IDENT_RE, STOP_IDENTS, RepoIndex  # noqa: E402

EXTS = {".py": "Python", ".swift": "Swift"}
MAX_TOKENS = 48
MAX_FILE_CHARS = 200_000


def xfile_names(index, rel, text):
    """Names in text that are defined in another file of the same language, and not in this file."""
    lang = "py" if rel.endswith(".py") else "swift"
    own = {s.name for s in index.symbols.get(rel, [])} | {m for s in index.symbols.get(rel, []) for m, _ in s.members}
    out = set()
    for ident in IDENT_RE.findall(text):
        if ident in STOP_IDENTS or ident in own or len(ident) < 3:
            continue
        if any(s.file != rel and s.lang == lang for s, _ in index.by_name.get(ident, ())):
            out.add(ident)
    return out


def sample_cases(index, n, seed, xfile_only=False):
    files = [index.root / rel for rel in sorted(index.symbols)]
    files = [p for p in files if 2_000 < p.stat().st_size < MAX_FILE_CHARS]
    rng = random.Random(seed)
    cases, tries = [], 0
    while len(cases) < n and tries < n * 200:
        tries += 1
        path = rng.choice(files)
        lines = path.read_text(errors="ignore").split("\n")
        candidates = [i for i, l in enumerate(lines) if len(l.strip()) >= 12 and not l.strip().startswith(("#", "//"))]
        if not candidates:
            continue
        i = rng.choice(candidates)
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        # Cut somewhere in the line: at the indent (whole line) or mid-line.
        cut = indent if rng.random() < 0.3 else rng.randint(indent + 1, len(line) - 4)
        rel = str(path.relative_to(index.root))
        xnames = xfile_names(index, rel, line[cut:])
        if xfile_only and not xnames:
            continue
        cases.append(dict(name=f"{rel}:{i + 1}", rel=rel, lang=EXTS[path.suffix], line=line, xfile=bool(xnames),
                          prefix="\n".join(lines[:i]) + "\n" + line[:cut], suffix="\n" + "\n".join(lines[i + 1:]),
                          truth=line[cut:]))
    return cases


def oracle_source(index, c, budget, count):
    """Like oracle_context, but real source code (bodies included) starting at each definition,
    cut to fit: closer to the full-file {content} Qwen saw in repo-level training."""
    picks = oracle_context(index, c, 10**9, count, return_symbols=True)
    if not picks:
        return []
    per_symbol = budget // len(picks) - 8
    by_file = {}
    for s, member in picks:
        lines = (index.root / s.file).read_text(errors="ignore").split("\n")
        start, chunk, used = s.line - 1, [], 0
        if member:  # a method/field of a class: class header line, then that member's real source
            pat = re.compile(rf"\b(def|func|var|let|case)\s+{re.escape(member)}\b|^\s*{re.escape(member)}\s*[:=]")
            hit = next((k for k in range(s.line, len(lines)) if pat.search(lines[k])), None)
            if hit is not None:
                chunk, used = [lines[s.line - 1], ""], count(lines[s.line - 1]) + 1
                start = hit
        while start > 0 and lines[start - 1].lstrip().startswith(("@", "///", "#")):  # keep decorators/doc comments
            start -= 1
        for line in lines[start:]:
            n = count(line + "\n")
            if used + n > per_symbol:
                break
            chunk.append(line)
            used += n
        by_file.setdefault(s.file, []).append("\n".join(chunk).rstrip())
    return [(f, "\n\n".join(t)) for f, t in by_file.items()]


def oracle_context(index, c, budget, count, return_symbols=False):
    """Outlines of the symbols whose names appear in the answer (cheating: an upper bound for retrieval)."""
    lang = "py" if c["rel"].endswith(".py") else "swift"
    imported = index.imports.get(c["rel"], set())
    cur_dir = str(Path(c["rel"]).parent)
    picked, used = {}, 0
    for name in sorted(xfile_names(index, c["rel"], c["truth"])):
        entries = [(s, m) for s, m in index.by_name.get(name, ()) if s.file != c["rel"] and s.lang == lang]
        # most plausible definition: imported file, then same directory, then first found
        entries.sort(key=lambda e: (e[0].file not in imported, str(Path(e[0].file).parent) != cur_dir))
        s, member = entries[0]
        text = s.render({member} if member else set())
        cost = count(text) + count(s.file) + 3
        if id(s) not in picked and used + cost <= budget:
            picked[id(s)] = (s, text, member)
            used += cost
    if return_symbols:  # at most 3 definitions for the source variant
        return [(s, member) for s, _, member in list(picked.values())[:3]]
    picked = {k: (s, text) for k, (s, text, _) in picked.items()}
    by_file = {}
    for s, text in picked.values():
        by_file.setdefault(s.file, []).append(text)
    return [(f, "\n\n".join(t)) for f, t in by_file.items()]


def run(engine, cases, variant, index, ctx_tokens, min_score):
    pb = PromptBuilder(engine.tokenizer, engine.ctx)
    count = lambda t: len(engine.tokenizer(t, add_special_tokens=False).input_ids)
    results = []
    for c in cases:
        context, used, leak, rms = None, 0, False, 0.0
        if variant == "ctx":
            t0 = time.perf_counter()
            files, used = index.context_for(c["prefix"], c["suffix"], c["rel"], ctx_tokens, count, min_score)
            rms = (time.perf_counter() - t0) * 1000
            context = (index.root.name, files, c["rel"]) if files else None
            leak = any(c["line"].strip() in text for _, text in files)
        elif variant == "fmt":
            context = (index.root.name, [], c["rel"])
        elif variant in ("oracle", "oracle_src"):  # upper bound: definitions of the cross-file names in the *answer*
            files = (oracle_context if variant == "oracle" else oracle_source)(index, c, ctx_tokens, count)
            used = sum(count(t) for _, t in files)
            context = (index.root.name, files, c["rel"]) if files else None
            leak = any(c["line"].strip() in text for _, text in files)
        pb.anchor, engine.cached = None, []  # every case is a different file: start cold
        ids = pb.build(c["prefix"], c["suffix"], MAX_TOKENS, context)
        t0 = time.perf_counter()
        out, _ = engine.complete(ids, MAX_TOKENS, stop_text=lambda t: "\n" in t)
        ms = (time.perf_counter() - t0) * 1000
        results.append(dict(text=engine.tokenizer.decode(out, skip_special_tokens=True).split("\n")[0],
                            ms=ms, ctx_tokens=used, leak=leak, retrieval_ms=rms, prompt_len=len(ids)))
    return results


def exact(r, c):
    return r["text"].rstrip() == c["truth"].rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1.5b")
    ap.add_argument("--variants", default="none,ctx")
    ap.add_argument("--model", default="lut6")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=str(Path.home() / "src/Anemll"))
    ap.add_argument("--ctx-tokens", type=int, default=400, help="token budget for cross-file context")
    ap.add_argument("--min-score", type=float, default=0.5, help="retrieval score threshold")
    ap.add_argument("--xfile-only", action="store_true", help="only lines that use names from other files")
    ap.add_argument("--show", type=int, default=10, help="print this many cases where the runs differ")
    a = ap.parse_args()

    index = RepoIndex(a.corpus).refresh()
    print(f"index: {len(index.symbols)} files, {sum(len(v) for v in index.symbols.values())} symbols, "
          f"built in {index.build_ms:.0f} ms")
    cases = sample_cases(index, a.n, a.seed, a.xfile_only)
    runs = {}
    for size in a.sizes.split(","):
        eng = Engine(PROJECT_DIR / f"models/qwen25coder-{size}-ctx2048-{a.model}/meta.yaml")
        for variant in a.variants.split(","):
            runs[f"{size}/{variant}"] = run(eng, cases, variant, index, a.ctx_tokens, a.min_score)
        del eng

    leaked = {k for res in runs.values() for k, r in enumerate(res) if r["leak"]}
    clean = [k for k in range(len(cases)) if k not in leaked]
    print(f"\n{sum(c['xfile'] for c in cases)} of {len(cases)} cases use a name defined in another file ('xfile' row)")
    print(f"{len(cases)} cases from {a.corpus}; {len(leaked)} where the context contained the answer line "
          f"(copy-pasted code), so 'exact (clean)' uses the other {len(clean)}\n")
    print(f"{'run':12s} {'lang':7s} {'exact':>6s} {'exact (clean)':>14s} {'similarity':>11s} {'empty':>6s} "
          f"{'ctx tok':>8s} {'prompt':>7s}")
    for key, res in runs.items():
        for lang in ["all"] + sorted(set(EXTS.values())) + ["xfile"]:
            idx = [k for k, c in enumerate(cases) if lang == "all" or c["lang"] == lang or (lang == "xfile" and c["xfile"])]
            if not idx:
                continue
            cidx = [k for k in idx if k in clean]
            ex = sum(exact(res[k], cases[k]) for k in idx) / len(idx)
            exc = sum(exact(res[k], cases[k]) for k in cidx) / max(1, len(cidx))
            sim = sum(difflib.SequenceMatcher(None, res[k]["text"].strip(), cases[k]["truth"].strip()).ratio()
                      for k in idx) / len(idx)
            empty = sum(not res[k]["text"].strip() for k in idx)
            ctx_tok = sorted(res[k]["ctx_tokens"] for k in idx)[len(idx) // 2]
            plen = sorted(res[k]["prompt_len"] for k in idx)[len(idx) // 2]
            print(f"{key:12s} {lang:7s} {ex:6.0%} {exc:14.0%} {sim:11.2f} {empty:6d} {ctx_tok:8d} {plen:7d}")

    keys = list(runs)
    if any(r["retrieval_ms"] for res in runs.values() for r in res):
        rms = sorted(r["retrieval_ms"] for res in runs.values() for r in res if r["retrieval_ms"])
        print(f"\nretrieval time: median {rms[len(rms) // 2]:.1f} ms, max {rms[-1]:.1f} ms")
    for k0, k1 in [(keys[0], k) for k in keys[1:]]:
        wins = {k0: 0, k1: 0}
        shown = 0
        print(f"\nClean cases where exactly one run got it right:")
        for k in clean:
            c = cases[k]
            ok0, ok1 = exact(runs[k0][k], c), exact(runs[k1][k], c)
            if ok0 != ok1:
                wins[k0 if ok0 else k1] += 1
                if shown < a.show:
                    shown += 1
                    print(f"  {c['name']}  ...{c['prefix'].split(chr(10))[-1][-40:]!r}")
                    print(f"      real      : {c['truth'].strip()[:70]!r}")
                    print(f"      {k0:10s}: {runs[k0][k]['text'].strip()[:70]!r}")
                    print(f"      {k1:10s}: {runs[k1][k]['text'].strip()[:70]!r}")
        print(f"  only {k0} right: {wins[k0]}   only {k1} right: {wins[k1]}")


if __name__ == "__main__":
    main()
