"""Lightweight repo symbol index: cross-file context for code completion.

Indexes definitions (classes, functions, methods, constants, Swift types) as compact
outlines (signatures + first docstring line, no bodies), plus Python import links between
files. For a cursor position it picks the outlines whose names appear near the cursor.

Pure Python, no extra dependencies: `ast` for .py, a brace-tracking line scanner for .swift.
Python 3.9 compatible (runs in the ANEMLL env).
"""

import ast
import builtins
import keyword
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SKIP_DIRS = {".git", "node_modules", "build", "dist", "__pycache__", ".venv", "venv", "env-anemll",
             ".build", "DerivedData", "models"}
SWIFT_KEYWORDS = {
    "func", "let", "var", "if", "else", "guard", "return", "self", "Self", "nil", "true", "false", "import",
    "struct", "class", "enum", "case", "switch", "for", "in", "while", "try", "await", "async", "throws",
    "static", "private", "public", "internal", "init", "some", "any", "where", "extension", "protocol",
    "default", "break", "continue", "do", "catch", "defer", "inout", "weak", "lazy", "override", "final",
    "mutating", "get", "set", "didSet", "willSet", "is", "as", "super", "typealias", "String", "Int",
    "Double", "Float", "Bool", "Array", "Dictionary", "Void",
}
STOP_IDENTS = set(keyword.kwlist) | set(dir(builtins)) | SWIFT_KEYWORDS | {"self", "cls", "__init__", "main"}
KEEP_DECORATORS = {"property", "staticmethod", "classmethod", "dataclass", "cached_property", "abstractmethod"}
MAX_MEMBERS = 12
MAX_LINE = 160
MAX_FILE_BYTES = 400_000


@dataclass
class Symbol:
    name: str
    kind: str                     # class | function | const | type
    lang: str                     # py | swift
    file: str                     # path relative to the repo root
    line: int
    header: List[str]             # signature line(s), no body
    doc: str = ""
    members: List[Tuple[str, str]] = field(default_factory=list)  # (name, one-line signature, pre-indented)

    def render(self, hits: Set[str] = frozenset()) -> str:
        members = self.members
        if len(members) > MAX_MEMBERS:  # keep the members we were asked about, then the first ones
            keep = {i for i, (n, _) in enumerate(members) if n in hits}
            for i in range(len(members)):
                if len(keep) >= MAX_MEMBERS:
                    break
                keep.add(i)
            members = [m for i, m in enumerate(members) if i in keep]
        truncated = len(members) < len(self.members)

        if self.lang == "py":
            lines = list(self.header)
            body_indent = re.match(r"\s*", self.header[-1]).group(0) + "    "
            if self.kind == "const":
                return "\n".join(lines)
            if self.doc:
                lines.append(f'{body_indent}"""{self.doc}"""')
            lines += [text for _, text in members]
            if truncated:
                lines.append(f"{body_indent}...")
            if len(lines) == len(self.header):  # nothing in the body: stub style
                lines[-1] += " ..."
            return "\n".join(lines)

        # swift
        lines = [f"/// {self.doc}"] if self.doc else []
        if self.kind == "function":
            return "\n".join(lines + self.header)
        lines.append(self.header[0] + " {")
        lines += [text for _, text in members]
        if truncated:
            lines.append("    // ...")
        lines.append("}")
        return "\n".join(lines)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LINE else text[:MAX_LINE - 3] + "..."


# ---------------------------------------------------------------- Python

def _py_header(lines: List[str], node) -> List[str]:
    start, body = node.lineno - 1, node.body[0]
    if body.lineno - 1 > start:
        hdr = lines[start:body.lineno - 1]
    else:  # body on the header line: "def f(): return 1"
        hdr = [lines[start][:body.col_offset]]
    hdr = [h.rstrip() for h in hdr if h.strip() and not h.strip().startswith("#")]
    decos = []
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in KEEP_DECORATORS:
            decos.append(lines[d.lineno - 1].rstrip())
    if len(hdr) > 4:  # long multi-line signature: squash into one line
        indent = re.match(r"\s*", hdr[0]).group(0)
        hdr = [indent + _clip(" ".join(h.strip() for h in hdr))]
    return decos + hdr


def _py_doc(node) -> str:
    doc = ast.get_docstring(node)
    return _clip(doc.strip().splitlines()[0]) if doc and doc.strip() else ""


def index_python(rel: str, text: str) -> Tuple[List[Symbol], List[Tuple[str, int, List[str]]]]:
    tree = ast.parse(text)
    lines = text.split("\n")
    symbols, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol(node.name, "function", "py", rel, node.lineno, _py_header(lines, node), _py_doc(node)))
        elif isinstance(node, ast.ClassDef):
            sym = Symbol(node.name, "class", "py", rel, node.lineno, _py_header(lines, node), _py_doc(node))
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name.startswith("_") and m.name not in ("__init__", "__call__"):
                        continue
                    hdr = _py_header(lines, m)
                    indent = re.match(r"\s*", hdr[-1]).group(0)
                    sig = indent + _clip(" ".join(h.strip() for h in hdr if not h.strip().startswith("@"))) + " ..."
                    sym.members.append((m.name, sig))
                elif isinstance(m, (ast.Assign, ast.AnnAssign)):
                    targets = m.targets if isinstance(m, ast.Assign) else [m.target]
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    if names:
                        raw = lines[m.lineno - 1]
                        indent = re.match(r"\s*", raw).group(0)
                        sym.members.append((names[0], indent + _clip(raw)))
            symbols.append(sym)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.isupper():
            symbols.append(Symbol(node.targets[0].id, "const", "py", rel, node.lineno, [_clip(lines[node.lineno - 1])]))
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", node.level, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports.append((a.name, 0, []))
    return symbols, imports


# ---------------------------------------------------------------- Swift

_MODS = r"(?:(?:public|private|internal|fileprivate|open|final|static|class|override|mutating|nonmutating|" \
        r"nonisolated|convenience|required|lazy|weak|unowned|dynamic|optional|indirect)\s+)*"
_ATTRS = r"(?:@[\w.]+(?:\([^)]*\))?\s+)*"
SWIFT_TYPE = re.compile(rf"^\s*{_ATTRS}{_MODS}(class|struct|enum|protocol|extension|actor)\s+([A-Za-z_][\w.]*)")
SWIFT_MEMBER = re.compile(rf"^\s*{_ATTRS}{_MODS}(func|init|var|let|case|typealias|subscript)\b\s*([A-Za-z_]\w*)?")
SWIFT_STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def index_swift(rel: str, text: str) -> Tuple[List[Symbol], list]:
    lines = text.split("\n")
    symbols: List[Symbol] = []
    stack: List[Tuple[Symbol, int]] = []  # (type symbol, depth at which its body opened)
    depth, doc, i, in_multiline = 0, "", 0, False
    while i < len(lines):
        raw = lines[i]
        if raw.count('"""') % 2 == 1:  # entering/leaving a multi-line string literal
            in_multiline = not in_multiline
            i += 1
            continue
        if in_multiline:
            i += 1
            continue
        code = SWIFT_STRING.sub('""', raw).split("//")[0]
        stripped = raw.strip()
        if stripped.startswith("///"):
            doc = doc or _clip(stripped[3:].strip())
            i += 1
            continue
        if not stripped or stripped.startswith("//"):
            i += 1
            continue

        mt = SWIFT_TYPE.match(code)
        mm = SWIFT_MEMBER.match(code)
        last = i  # last line consumed by this declaration
        if mt and mt.group(2) not in ("func", "var", "let"):
            sym = Symbol(mt.group(2).split(".")[-1], "type", "swift", rel, i + 1, [_clip(code.split("{")[0])], doc)
            symbols.append(sym)
            if "{" in code:
                stack.append((sym, depth))
        elif mm:
            sig_lines, j = [code.split("{")[0]], i
            while sig_lines[-1].count("(") > sig_lines[-1].count(")") and j + 1 < len(lines) and j - i < 5:
                j += 1
                sig_lines[-1] += " " + SWIFT_STRING.sub('""', lines[j]).split("//")[0].split("{")[0].strip()
            sig = _clip(sig_lines[0])
            name = "init" if mm.group(1) == "init" else (mm.group(2) or "")
            private = re.search(r"\b(private|fileprivate)\b", code) is not None
            if stack and depth == stack[-1][1] + 1:
                if name and not private:
                    stack[-1][0].members.append((name, "    " + sig.strip()))
            elif depth == 0 and mm.group(1) == "func" and name:
                symbols.append(Symbol(name, "function", "swift", rel, i + 1, [sig.strip()], doc))
            if j > i:  # account braces on the extra signature lines, then skip them
                for k in range(i + 1, j + 1):
                    c = SWIFT_STRING.sub('""', lines[k]).split("//")[0]
                    depth += c.count("{") - c.count("}")
                last = j
        doc = ""
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][1]:
            stack.pop()
        i = last + 1
    return symbols, []


# ---------------------------------------------------------------- index + retrieval

class RepoIndex:
    def __init__(self, root, exts=(".py", ".swift")):
        self.root = Path(root).expanduser().resolve()
        self.exts = set(exts)
        self.symbols: Dict[str, List[Symbol]] = {}            # file -> symbols
        self.raw_imports: Dict[str, list] = {}
        self.mtimes: Dict[str, float] = {}
        self.by_name: Dict[str, List[Tuple[Symbol, Optional[str]]]] = defaultdict(list)
        self.imports: Dict[str, Set[str]] = {}                # file -> repo files it imports
        self.build_ms = 0.0

    def _files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]  # prune walk
            for name in filenames:
                if os.path.splitext(name)[1] in self.exts:
                    yield Path(dirpath) / name

    def refresh(self):
        """(Re)index new or changed files; drop deleted ones. Cheap when nothing changed."""
        t0 = time.perf_counter()
        seen = set()
        for p in self._files():
            rel = str(p.relative_to(self.root))
            seen.add(rel)
            st = p.stat()
            if st.st_size > MAX_FILE_BYTES or self.mtimes.get(rel) == st.st_mtime:
                continue
            self.mtimes[rel] = st.st_mtime
            try:
                text = p.read_text(errors="ignore")
                syms, imps = (index_python if p.suffix == ".py" else index_swift)(rel, text)
            except (SyntaxError, ValueError, RecursionError):
                syms, imps = [], []
            self.symbols[rel], self.raw_imports[rel] = syms, imps
        for rel in set(self.symbols) - seen:
            del self.symbols[rel], self.raw_imports[rel], self.mtimes[rel]
        self._link()
        self.build_ms = (time.perf_counter() - t0) * 1000
        return self

    def _link(self):
        self.by_name = defaultdict(list)
        for syms in self.symbols.values():
            for s in syms:
                self.by_name[s.name].append((s, None))
                for m, _ in s.members:
                    self.by_name[m].append((s, m))
        dotted = {}
        for rel in self.symbols:
            if rel.endswith(".py"):
                mod = rel[:-3].replace("/", ".")
                dotted[mod[:-9] if mod.endswith(".__init__") else mod] = rel
        self.imports = {}
        for rel, imps in self.raw_imports.items():
            pkg = rel[:-3].replace("/", ".").rsplit(".", 1)[0] if "/" in rel else ""
            out = set()
            for module, level, names in imps:
                if level:
                    base = pkg.split(".") if pkg else []
                    base = base[:len(base) - (level - 1)] if level > 1 else base
                    module = ".".join([*base, module] if module else base)
                for cand in [module] + [f"{module}.{n}" for n in names]:
                    for mod, path in dotted.items():
                        if mod == cand or mod.endswith("." + cand):
                            out.add(path)
            self.imports[rel] = out

    def context_for(self, prefix: str, suffix: str, current_file: Optional[str], budget: int,
                    count_tokens: Callable[[str], int], min_score: float = 0.5) -> Tuple[List[Tuple[str, str]], int]:
        """Pick outlines for names used near the cursor. Returns ([(path, text)], tokens used)."""
        lines = prefix.split("\n")
        weights: Dict[str, float] = defaultdict(float)
        for text, w in ((lines[-1], 4.0), ("\n".join(lines[-11:-1]), 2.0), ("\n".join(lines[-60:-11]), 1.0),
                        ("\n".join(suffix.split("\n")[:6]), 1.0)):
            for ident in IDENT_RE.findall(text):
                if ident not in STOP_IDENTS and len(ident) > 1:
                    weights[ident] += w

        imported = self.imports.get(current_file, set()) if current_file else set()
        cur_dir = str(Path(current_file).parent) if current_file else None
        lang = {".py": "py", ".swift": "swift"}.get(Path(current_file).suffix) if current_file else None
        scores: Dict[int, float] = defaultdict(float)
        hits: Dict[int, Set[str]] = defaultdict(set)
        by_id: Dict[int, Symbol] = {}
        for ident, w in weights.items():
            entries = [(s, m) for s, m in self.by_name.get(ident, ())
                       if s.file != current_file and (lang is None or s.lang == lang)]
            for s, member in entries:
                bonus = (2.0 if s.file in imported else 1.0) * (1.3 if cur_dir and str(Path(s.file).parent) == cur_dir else 1.0)
                scores[id(s)] += w * bonus / len(entries)
                by_id[id(s)] = s
                if member:
                    hits[id(s)].add(member)

        chosen, used, files_used = [], 0, set()
        for sid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            if score < min_score:
                break
            s = by_id[sid]
            text = s.render(hits[sid])
            cost = count_tokens(text) + 1 + (0 if s.file in files_used else count_tokens(s.file) + 2)
            if used + cost > budget:
                continue
            used += cost
            files_used.add(s.file)
            chosen.append((score, s, text))

        # Group by file; most relevant file last (closest to the code being completed).
        by_file: Dict[str, List[Tuple[float, Symbol, str]]] = defaultdict(list)
        for item in chosen:
            by_file[item[1].file].append(item)
        ordered = sorted(by_file.items(), key=lambda kv: max(x[0] for x in kv[1]))
        return [(path, "\n\n".join(t for _, _, t in sorted(items, key=lambda x: x[1].line)))
                for path, items in ordered], used
