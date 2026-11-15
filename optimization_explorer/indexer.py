"""Safe source indexing: Python AST, C/C++/MLIR text heuristics, ZIP without extraction."""
import ast
import hashlib
import io
import json
import os
import re
import tokenize
import zipfile
from pathlib import Path

EXTS = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh", ".td", ".mlir", ".hlo", ".inc"}
MAX_SOURCE_BYTES = 2_000_000
SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist"}
HINTS = re.compile(r"fusion|fuse|fused|epilogue|rewrite|pattern|lowering|kernel|dispatch|decompos|post.?op|matmul|attention|batch.?norm|conv", re.I)
C_DEFINITION = re.compile(r"^\s*(?:[A-Za-z_]\w*\s+)+[*&\s]*([A-Za-z_]\w*)\s*\([^();]*\)\s*\{")
C_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
HLO_ENTRY = re.compile(r"^\s*ENTRY\s+%?([\w.-]+)\s*\{\s*$")
MLIR_FUNCTION = re.compile(r"^\s*(?:func\.func|llvm\.func)\s+(?:(?:public|private|nested)\s+)?@([\w.$-]+)\(")

def _extract_tasks(data: bytes, repository: str, relpath: str, revision: str, max_hits: int) -> list[dict]:
    if len(data) > MAX_SOURCE_BYTES or b"\x00" in data[:4096]:
        return []
    source_sha256 = hashlib.sha256(data).hexdigest()
    suffix = Path(relpath).suffix.lower()
    if suffix == ".py":
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
            src = data.decode(encoding)
        except (SyntaxError, UnicodeError, LookupError):
            return []
    else:
        src = data.decode("utf-8", errors="replace")
    lines = src.splitlines()
    functions = []
    if suffix == ".py":
        try:
            tree = ast.parse(src)
            functions = [(n.lineno, getattr(n, "end_lineno", n.lineno), n.name) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        except (SyntaxError, ValueError):
            pass
    tasks = []
    deferred = []
    seen_scopes = set()
    hlo_entry = None
    for i, line in enumerate(lines):
        if suffix == ".hlo":
            if entry := HLO_ENTRY.match(line):
                hlo_entry = entry[1]
            elif line.strip() == "}":
                hlo_entry = None
        if not HINTS.search(line):
            continue
        if len(tasks) >= max_hits:
            break
        n = i + 1
        enclosing = [f for f in functions if f[0] <= n <= f[1]]
        scope = min(enclosing, key=lambda x: x[1] - x[0]) if enclosing else None
        func = scope[2] if scope else "UNKNOWN"
        if hlo_entry:
            func = hlo_entry
        if suffix == ".mlir" and (definition := MLIR_FUNCTION.match(line)):
            func = definition[1]
        if suffix in C_EXTS and (definition := C_DEFINITION.match(line)):
            func = definition[1]
        lo, hi = max(1, n - 8), min(len(lines), n + 12)
        excerpt = "\n".join(f"{j}: {lines[j-1][:350]}" for j in range(lo, hi+1))
        identity = json.dumps([repository, revision, relpath, source_sha256, n])
        tid = hashlib.sha256(identity.encode()).hexdigest()[:20]
        if scope is not None and scope in seen_scopes and len(deferred) >= max_hits:
            continue
        target = deferred if scope is not None and scope in seen_scopes else tasks
        if scope is not None:
            seen_scopes.add(scope)
        target.append({"task_id": tid, "repository": repository, "revision": revision,
                      "source_sha256": source_sha256,
                      "source_file": relpath, "source_function": func, "source_lines": f"{lo}-{hi}",
                      "context": excerpt, "line": n})
    return tasks + deferred[:max_hits - len(tasks)]

def _source_paths(root: Path):
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRECTORIES)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() in EXTS and not path.is_symlink() and path.is_file():
                yield path


def index_directory(path: str, repository: str, revision: str = "UNKNOWN", max_files: int = 1000, max_hits: int = 3) -> list[dict]:
    return list(iter_directory_tasks(path, repository, revision, max_files, max_hits))


def iter_directory_tasks(path: str, repository: str, revision: str = "UNKNOWN", max_files: int = 1000, max_hits: int = 3):
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = 0
    for p in _source_paths(root):
        if files >= max_files:
            break
        files += 1
        try:
            with p.open("rb") as stream:
                data = stream.read(MAX_SOURCE_BYTES + 1)
            for task in _extract_tasks(data, repository, p.relative_to(root).as_posix(), revision, max_hits):
                yield dict(task, local_source_path=str(p))
        except (OSError, PermissionError):
            continue

def index_zip(path: str, repository: str, revision: str = "UNKNOWN", max_files: int = 1000, max_hits: int = 3) -> list[dict]:
    return list(iter_zip_tasks(path, repository, revision, max_files, max_hits))


def iter_zip_tasks(path: str, repository: str, revision: str = "UNKNOWN", max_files: int = 1000, max_hits: int = 3):
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as z:
        members = [info for info in z.infolist() if not info.is_dir()]
        roots = {info.filename.partition("/")[0] for info in members}
        strip_root = len(roots) == 1 and all("/" in info.filename for info in members)
        files = 0
        for info in sorted(members, key=lambda x: x.filename):
            p = Path(info.filename)
            if info.is_dir() or p.suffix.lower() not in EXTS or info.file_size > MAX_SOURCE_BYTES:
                continue
            if files >= max_files:
                break
            files += 1
            try:
                data = z.read(info)
                rel = info.filename.partition("/")[2] if strip_root else info.filename
                for task in _extract_tasks(data, repository, rel, revision, max_hits):
                    yield dict(task, local_source_path=str(path), zip_member=info.filename)
            except (OSError, RuntimeError, zipfile.BadZipFile):
                continue
