#!/usr/bin/env python3
"""Evidence-first layer/operation and fusion-combination atlas.

This is a *discovery* engine, not a proof of cross-library lowering or runtime fusion.
No arbitrary source fragments are queued as optimization cases.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

EXTENSIONS = {'.py', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.td', '.mlir'}
IGNORED = {'.git', '.venv', '__pycache__', 'third_party', 'third-party', 'test', 'tests', 'benchmarks', 'build', 'dist'}
REPO_ROLES = {'keras': 'framework', 'tensorflow': 'framework',
              'pytorch': 'framework', 'cudnn_frontend': 'backend',
              'cutlass': 'backend', 'decompiled': 'binary_evidence'}
EXCLUDED_REPOSITORIES = {'jax'}
REPOSITORY_ROOTS = {
    'keras': ('keras/src',),
    'tensorflow': (
        'tensorflow/python/ops', 'tensorflow/core/kernels',
        'tensorflow/core/framework', 'tensorflow/core/common_runtime',
        'tensorflow/cc/ops', 'tensorflow/cc/framework',
        'tensorflow/compiler/mlir', 'tensorflow/compiler/tf2xla',
        'tensorflow/compiler/xla/service/gpu',
    ),
    'pytorch': (
        'torch/nn', 'torch/_inductor', 'torch/_dynamo', 'torch/csrc',
        'aten/src/ATen/native', 'aten/src/ATen/cuda', 'c10/cuda',
    ),
    'cudnn_frontend': ('include', 'src', 'samples'),
    'cutlass': ('include', 'examples', 'tools/library/include', 'tools/library/src'),
    'decompiled': ('sm75',),
}
DEFAULT_REPOSITORIES = tuple(REPOSITORY_ROOTS)
OP_WORDS = re.compile(r'(?:conv|dense|linear|matmul|gemm|dot|attention|softmax|norm|pool|relu|gelu|silu|swish|sigmoid|tanh|activation|embedding|lstm|gru|rnn|reshape|transpose|reduce|concat|split|add|multiply|bias|dropout|gather|scatter|pad|slice|quant|dequant|fuse|fusion|epilogue|batch|einsum|fft|sort|topk|argmax|cast|broadcast|where|select)', re.I)
FUSION_WORDS = re.compile(r'\b(?:fused?|fusion|epilogue|pattern|rewrite|combine)\b', re.I)
BRANCH_WORDS = re.compile(r'\b(?:if|elif|else|switch|case|constexpr|dispatch|heuristic|capability|architecture|sm_75)\b', re.I)
DEFINITION = re.compile(r'\b(?:class|struct|def|void|auto|bool|static|constexpr|template|func\.func)\s+([A-Za-z_]\w*)')
CPP_SYMBOL = re.compile(r'\b(?:class|struct|enum|void|auto|bool)\s+([A-Za-z_]\w*)')
BINARY_EXT = {'.asm', '.sass', '.ptx', '.cuobjdump', '.nvdisasm', '.txt'}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]


def normalized_symbol(name):
    """Conservative canonicalization; distinct operations are never silently merged."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def inspect_python(source, repository, relpath, sha):
    """AST definitions, parameters and directly nested calls; no inferred dispatch edges."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return [], [], []
    lines = source.splitlines()
    symbols, compositions, predicates = [], [], []

    def call_name(call):
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    def relevant(name):
        return bool(OP_WORDS.search(name))

    def add_symbol(node, name, kind):
        if not relevant(name):
            return
        line = node.lineno
        sid = digest(f'{repository}:{relpath}:{sha}:{line}:{name}:{kind}')
        parameters = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
        elif isinstance(node, ast.ClassDef):
            init = next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == '__init__'), None)
            if init:
                parameters = [a.arg for a in init.args.posonlyargs + init.args.args + init.args.kwonlyargs]
        symbols.append(dict(id=sid, repository=repository, file=relpath, line=line, symbol=name,
                            normalized=normalized_symbol(name), kind=kind, parameters=parameters,
                            sha256=sha, snippet=lines[line-1][:500], status='source_definition'))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            add_symbol(node, node.name, 'class')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_symbol(node, node.name, 'function')
        elif isinstance(node, ast.Call):
            outer = call_name(node)
            if not outer:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for inner_node in ast.walk(arg):
                    if isinstance(inner_node, ast.Call):
                        inner = call_name(inner_node)
                        if inner and relevant(inner) and relevant(outer):
                            cid = digest(f'{repository}:{relpath}:{sha}:{node.lineno}:{inner}:{outer}')
                            compositions.append(dict(id=cid, repository=repository, file=relpath,
                                line=node.lineno, input_op=inner, output_op=outer,
                                evidence='syntactic_nested_call', status='observed_syntax_not_runtime_fusion', sha256=sha))
        elif isinstance(node, (ast.If, ast.IfExp)):
            line = node.lineno
            excerpt = ast.get_source_segment(source, node.test) or ''
            if OP_WORDS.search(excerpt) or re.search(r'\b(?:dtype|shape|stride|padding|layout|device|compute_capability|groups|dilation|rank|axis|sm)\b', excerpt, re.I):
                predicates.append(dict(id=digest(f'{repository}:{relpath}:{sha}:{line}:predicate'),
                    repository=repository, file=relpath, line=line, expression=excerpt[:1200],
                    status='source_predicate_unlinked', sha256=sha))
    return symbols, compositions, predicates


def inspect_cpp(source, repository, relpath, sha):
    symbols, compositions, predicates = [], [], []
    for line_number, line in enumerate(source.splitlines(), 1):
        if len(line) > 5000:
            continue
        match = CPP_SYMBOL.search(line)
        if match and OP_WORDS.search(match.group(1)):
            name = match.group(1)
            symbols.append(dict(id=digest(f'{repository}:{relpath}:{sha}:{line_number}:{name}:cpp'),
                repository=repository, file=relpath, line=line_number, symbol=name,
                normalized=normalized_symbol(name), kind='cpp_declaration', parameters=[],
                sha256=sha, snippet=line.strip()[:500], status='source_declaration_unverified'))
        if FUSION_WORDS.search(line) and OP_WORDS.search(line):
            compositions.append(dict(id=digest(f'{repository}:{relpath}:{sha}:{line_number}:fusion'),
                repository=repository, file=relpath, line=line_number, input_op='', output_op='',
                evidence=line.strip()[:1200], status='fusion_site_candidate', sha256=sha))
        if BRANCH_WORDS.search(line) and OP_WORDS.search(line):
            predicates.append(dict(id=digest(f'{repository}:{relpath}:{sha}:{line_number}:branch'),
                repository=repository, file=relpath, line=line_number, expression=line.strip()[:1200],
                status='source_predicate_unlinked', sha256=sha))
    return symbols, compositions, predicates


def discover(sources, max_files=0, max_bytes=2_000_000, repositories=None):
    """0 means no file-count limit; deterministic and streaming over repository paths."""
    if max_files < 0 or max_bytes < 1:
        raise ValueError('max-files must be nonnegative and max-bytes positive')
    root = Path(sources)
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = set(repositories or DEFAULT_REPOSITORIES) - EXCLUDED_REPOSITORIES
    output = dict(symbols=[], combinations=[], predicates=[], binaries=[], repositories=[], scanned_files=0, skipped_files=0)
    for repo_dir in sorted(root.iterdir()):
        if (not repo_dir.is_dir() or repo_dir.is_symlink()
                or repo_dir.name.startswith('.') or repo_dir.name in IGNORED
                or repo_dir.name not in selected):
            continue
        repo = repo_dir.name
        output['repositories'].append(dict(name=repo, role=REPO_ROLES.get(repo, 'unknown')))
        requested_roots = REPOSITORY_ROOTS.get(repo, ())
        scan_roots = [repo_dir / part for part in requested_roots
                      if (repo_dir / part).is_dir()]
        if not scan_roots:
            continue
        paths = sorted(path for scan_root in scan_roots for path in scan_root.rglob('*'))
        for path in paths:
            if max_files and output['scanned_files'] >= max_files:
                return output
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(repo_dir)
            if (any(part in IGNORED or part.startswith('.') for part in rel.parts)
                    or path.stem.endswith('_test') or path.name.startswith('test_')):
                continue
            try:
                size = path.stat().st_size
                if size > max_bytes:
                    output['skipped_files'] += 1
                    continue
                if path.suffix.lower() in BINARY_EXT and ('sm75' in str(rel).lower() or 'sm_75' in str(rel).lower()):
                    raw = path.read_bytes()
                    output['binaries'].append(dict(id=digest(f'{repo}:{rel}:{hashlib.sha256(raw).hexdigest()}'),
                        repository=repo, file=str(rel), sha256=hashlib.sha256(raw).hexdigest(),
                        status='unlinked_binary_text', architecture='sm_75'))
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                raw = path.read_bytes()
                source = raw.decode('utf-8')
            except (OSError, UnicodeError):
                output['skipped_files'] += 1
                continue
            output['scanned_files'] += 1
            sha = hashlib.sha256(raw).hexdigest()
            fn = inspect_python if path.suffix.lower() == '.py' else inspect_cpp
            sym, comp, pred = fn(source, repo, str(rel), sha)
            output['symbols'].extend(sym)
            output['combinations'].extend(comp)
            output['predicates'].extend(pred)
    return output


def initialize(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS atlas_symbols (
      id TEXT PRIMARY KEY, repository TEXT NOT NULL, source_file TEXT NOT NULL,
      source_line INTEGER NOT NULL, normalized TEXT NOT NULL, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS atlas_symbols_normalized ON atlas_symbols(normalized);
    CREATE TABLE IF NOT EXISTS atlas_combinations (
      id TEXT PRIMARY KEY, repository TEXT NOT NULL, source_file TEXT NOT NULL,
      source_line INTEGER NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS atlas_predicates (
      id TEXT PRIMARY KEY, repository TEXT NOT NULL, source_file TEXT NOT NULL,
      source_line INTEGER NOT NULL, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS atlas_binary_text (
      id TEXT PRIMARY KEY, repository TEXT NOT NULL, source_file TEXT NOT NULL, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS atlas_coverage (
      symbol_id TEXT PRIMARY KEY, discovery TEXT NOT NULL, lowering TEXT NOT NULL,
      fusion TEXT NOT NULL, kernel TEXT NOT NULL, sm75 TEXT NOT NULL);
    ''')


def persist(db_path, result):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        with db:
            initialize(db)
            repositories = sorted(
                {item['name'] for item in result['repositories']} | EXCLUDED_REPOSITORIES
            )
            if repositories:
                marks=','.join('?' for _ in repositories)
                db.execute(
                    f'DELETE FROM atlas_coverage WHERE symbol_id IN '
                    f'(SELECT id FROM atlas_symbols WHERE repository IN ({marks}))',
                    repositories,
                )
                for table in ('atlas_symbols', 'atlas_combinations', 'atlas_predicates', 'atlas_binary_text'):
                    db.execute(f'DELETE FROM {table} WHERE repository IN ({marks})',repositories)
            for table, key in [('atlas_symbols', 'symbols'), ('atlas_combinations', 'combinations'), ('atlas_predicates', 'predicates')]:
                seen_ids = set()
                for obj in result[key]:
                    if obj['id'] in seen_ids:
                        continue
                    seen_ids.add(obj['id'])
                    db.execute(f'INSERT INTO {table} VALUES (?,?,?,?,?,?)' if table == 'atlas_symbols' else
                               f'INSERT INTO {table} VALUES (?,?,?,?,?,?)' if table == 'atlas_combinations' else
                               f'INSERT INTO {table} VALUES (?,?,?,?,?)',
                               ((obj['id'], obj['repository'], obj['file'], obj['line'], obj['normalized'], json.dumps(obj, ensure_ascii=False))
                                if table == 'atlas_symbols' else
                                (obj['id'], obj['repository'], obj['file'], obj['line'], obj['status'], json.dumps(obj, ensure_ascii=False))
                                if table == 'atlas_combinations' else
                                (obj['id'], obj['repository'], obj['file'], obj['line'], json.dumps(obj, ensure_ascii=False))))
            for obj in result['binaries']:
                db.execute('INSERT INTO atlas_binary_text VALUES (?,?,?,?)',
                           (obj['id'], obj['repository'], obj['file'], json.dumps(obj, ensure_ascii=False)))
            for obj in result['symbols']:
                db.execute('INSERT INTO atlas_coverage VALUES (?,?,?,?,?,?)',
                           (obj['id'], 'source_definition', 'unknown', 'unknown', 'unknown', 'unknown'))
    finally:
        db.close()


def summary(result):
    by_repo = defaultdict(lambda: dict(symbols=0, combinations=0, predicates=0, binaries=0))
    for kind in ('symbols', 'combinations', 'predicates', 'binaries'):
        for item in result[kind]:
            by_repo[item['repository']][kind] += 1
    return dict(scanned_files=result['scanned_files'], skipped_files=result['skipped_files'],
                repositories=result['repositories'], counts={k: len(result[k]) for k in ('symbols', 'combinations', 'predicates', 'binaries')},
                by_repository=dict(sorted(by_repo.items())),
                verified_cross_repository_lowering_edges=0, verified_runtime_fusions=0,
                caveat='Candidate inventory only; no cross-repository or binary linkage is inferred')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', default='sources')
    p.add_argument('--db', default='results/optimization_cases.sqlite')
    p.add_argument('--out', default='results/layer-atlas-summary.json')
    p.add_argument('--max-files', type=int, default=0, help='0=unlimited within selected source roots')
    p.add_argument('--max-bytes', type=int, default=2_000_000)
    p.add_argument('--repositories', nargs='+', choices=sorted(REPOSITORY_ROOTS),
                   help='Repositories to scan; defaults to framework and CUDA backend source roots')
    args = p.parse_args(argv)
    try:
        result = discover(args.sources, args.max_files, args.max_bytes, args.repositories)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    persist(args.db, result)
    report = summary(result)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
