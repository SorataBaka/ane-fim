"""FIM prompt builder that fits the fixed ANE context and keeps the prompt start stable.

The KV cache only helps if the start of the prompt stays the same between keystrokes.
When the prefix is too long we cut on a line boundary and remember that cut line
(the "anchor"), reusing it on later requests until the prefix no longer fits.
A fresh cut leaves ~30% headroom so typing doesn't force a new cut right away.
"""

FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 151659, 151660, 151661
REPO_NAME, FILE_SEP = 151663, 151664
ANCHOR_LINES = 3
HEADROOM = 0.7


class PromptBuilder:
    def __init__(self, tokenizer, ctx, suffix_tokens=128, margin=8):
        self.tok = tokenizer
        self.ctx = ctx
        self.suffix_tokens = suffix_tokens
        self.margin = margin
        self.anchor = None

    def _enc(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def _line_counts(self, lines):
        return [len(x) for x in self.tok(lines, add_special_tokens=False).input_ids] if lines else []

    def _suffix(self, suffix):
        """Whole lines from the start of the suffix, up to suffix_tokens."""
        lines = suffix.splitlines(keepends=True)
        counts, used, keep = self._line_counts(lines), 0, 0
        for c in counts:
            if used + c > self.suffix_tokens:
                break
            used, keep = used + c, keep + 1
        ids = self._enc("".join(lines[:keep]))
        return ids[:self.suffix_tokens]

    def _prefix_start(self, lines, counts, budget):
        if self.anchor:
            for i in range(len(lines)):
                if tuple(lines[i:i + ANCHOR_LINES]) == self.anchor and sum(counts[i:]) <= budget:
                    return i
        if sum(counts) <= budget:
            return 0
        target, used, start = int(budget * HEADROOM), 0, len(lines)
        while start > 0 and used + counts[start - 1] <= target:
            start -= 1
            used += counts[start]
        return start

    def _context_ids(self, context):
        """Qwen repo-level header: <|repo_name|>repo, <|file_sep|>path + outline per file, then the current file."""
        repo, files, current = context
        ids = [REPO_NAME] + self._enc(f"{repo}\n")
        for path, text in files:
            ids += [FILE_SEP] + self._enc(f"{path}\n{text.rstrip()}\n")
        return ids + [FILE_SEP] + self._enc(f"{current}\n")

    def build(self, prefix, suffix, n_predict, context=None):
        """Token ids for <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>.

        context: optional (repo_name, [(path, outline_text)], current_path) placed before the
        FIM part in Qwen's repo-level format. Its tokens come out of the prefix budget.
        """
        ctx_ids = self._context_ids(context) if context else []
        suf_ids = self._suffix(suffix)
        budget = self.ctx - n_predict - self.margin - 3 - len(suf_ids) - len(ctx_ids)

        lines = prefix.splitlines(keepends=True)
        counts = self._line_counts(lines)
        start = self._prefix_start(lines, counts, budget)
        self.anchor = tuple(lines[start:start + ANCHOR_LINES])

        pre_ids = self._enc("".join(lines[start:]))
        if len(pre_ids) > budget:  # per-line counts are approximate; rare
            pre_ids = pre_ids[-budget:]
        return ctx_ids + [FIM_PREFIX] + pre_ids + [FIM_SUFFIX] + suf_ids + [FIM_MIDDLE]
