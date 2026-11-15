#!/usr/bin/env python3
"""Layer-rooted, evidence-constrained, resumable DFS source tracer.

Static Python branches are forked on demand; CUDA execution is NEVER inferred from
source inspection. Agent-selected cross-file steps require a source-backed target.
"""
from __future__ import annotations
import argparse
import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from pathlib import Path

from optimization_explorer.schema import COLUMNS, normalize_case

FRAMEWORKS = ('keras', 'tensorflow', 'pytorch')
SKIP = {'.git', '.venv', '__pycache__', 'tests', 'test', 'examples', 'third_party', 'build'}
SCHEMA = '''
CREATE TABLE IF NOT EXISTS trace_layers (
 id TEXT PRIMARY KEY, framework TEXT NOT NULL, name TEXT NOT NULL,
 source_file TEXT NOT NULL, line INTEGER NOT NULL, sha256 TEXT NOT NULL,
 kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS trace_tasks (
 id TEXT PRIMARY KEY, layer_id TEXT NOT NULL REFERENCES trace_layers(id),
 parent_id TEXT, file TEXT NOT NULL, line INTEGER NOT NULL,
 decisions TEXT NOT NULL, visited TEXT NOT NULL DEFAULT '[]', depth INTEGER NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending', error TEXT, result TEXT,
 attempts INTEGER NOT NULL DEFAULT 0, available_at INTEGER NOT NULL DEFAULT 0,
 retry_priority INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS trace_tasks_state ON trace_tasks(state,layer_id,depth);
CREATE TABLE IF NOT EXISTS trace_scheduler (
 layer_id TEXT PRIMARY KEY, last_selected INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS trace_model_assignments (
 layer_id TEXT PRIMARY KEY, model_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trace_leases (task_id TEXT PRIMARY KEY, token TEXT NOT NULL, expires INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS trace_task_errors (
 task_id TEXT NOT NULL, attempt INTEGER NOT NULL, stage TEXT NOT NULL,
 error TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(task_id,attempt));
CREATE TABLE IF NOT EXISTS trace_steps (
 task_id TEXT NOT NULL, seq INTEGER NOT NULL, file TEXT NOT NULL,
 line INTEGER NOT NULL, sha256 TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(task_id,seq));
CREATE TABLE IF NOT EXISTS trace_results (
 task_id TEXT PRIMARY KEY, layer_id TEXT NOT NULL, conditions TEXT NOT NULL,
 outcome TEXT NOT NULL, evidence TEXT NOT NULL, runtime_verified INTEGER NOT NULL DEFAULT 0,
 sm75_verified INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS trace_case_evidence (
 case_id TEXT NOT NULL, task_id TEXT NOT NULL, layer_id TEXT NOT NULL,
 source_sha256 TEXT NOT NULL, branch_line INTEGER NOT NULL, evidence TEXT NOT NULL,
 PRIMARY KEY(case_id,task_id));
CREATE TABLE IF NOT EXISTS trace_case_aliases (
 alias_case_id TEXT PRIMARY KEY, canonical_case_id TEXT NOT NULL,
 case_payload TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trace_case_semantics (
 semantic_key TEXT PRIMARY KEY, canonical_case_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS optimization_case_exclusions (
 case_id TEXT PRIMARY KEY, case_payload TEXT NOT NULL, evidence_payload TEXT NOT NULL,
 aliases_payload TEXT NOT NULL, reason TEXT NOT NULL,
 excluded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
'''


def stable(*items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def setup(db):
    db.executescript(SCHEMA)
    columns = ','.join(f'\"{name}\" TEXT NOT NULL' for name in COLUMNS)
    db.execute(f'CREATE TABLE IF NOT EXISTS optimization_cases ({columns}, PRIMARY KEY(case_id))')
    task_columns = {row[1] for row in db.execute('PRAGMA table_info(trace_tasks)')}
    for name, definition in (('attempts', 'INTEGER NOT NULL DEFAULT 0'),
                             ('available_at', 'INTEGER NOT NULL DEFAULT 0'),
                             ('retry_priority', 'INTEGER NOT NULL DEFAULT 0')):
        if name not in task_columns:
            db.execute(f'ALTER TABLE trace_tasks ADD COLUMN {name} {definition}')


def safe_source(root, repo, relative):
    base = (root / repo).resolve()
    path = (base / relative).resolve()
    if (not path.is_relative_to(base) or not path.is_file() or path.is_symlink()
            or not eligible_source(relative)):
        raise ValueError('source outside eligible framework code or missing')
    return path


FUNCTIONAL_ROOTS = {
    'pytorch': ('torch/nn/functional.py',),
    'tensorflow': ('tensorflow/python/ops/nn_ops.py',
                   'tensorflow/python/ops/nn_impl.py'),
    'keras': ('keras/src/ops/nn.py',),
}
FUNCTIONAL_NAMES = {
    'pytorch': {'linear','conv1d','conv2d','conv3d','conv_transpose1d',
                'conv_transpose2d','conv_transpose3d','relu','gelu',
                'layer_norm','batch_norm','group_norm','softmax',
                'scaled_dot_product_attention','max_pool2d','avg_pool2d',
                'dropout','embedding','multi_head_attention_forward'},
    'tensorflow': {'conv1d','conv2d','conv3d','bias_add','relu','gelu',
                   'softmax','batch_normalization','fused_batch_norm',
                   'depthwise_conv2d','max_pool','avg_pool','dropout',
                   'layer_norm','scaled_dot_product_attention'},
    'keras': {'conv','depthwise_conv','matmul','relu','gelu','softmax',
              'batch_normalization','layer_normalization',
              'dot_product_attention'},
}


def _bases(node, aliases):
    result = []
    for base in node.bases:
        try:
            name = ast.unparse(base)
        except (ValueError, TypeError):
            continue
        head, sep, tail = name.partition('.')
        result.append(aliases.get(head, head) + (sep + tail if sep else ''))
    return result


def _imports(tree):
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                if item.name != '*':
                    aliases[item.asname or item.name] = (node.module or '') + '.' + item.name
    return aliases


def _direct_layer(node, repo, rel, aliases):
    bases = _bases(node, aliases)
    path = '/' + rel.lower().replace('\\', '/')
    if repo == 'pytorch':
        return any(b.split('.')[-1] == 'Module' for b in bases) and '/torch/nn/' in path
    if repo in ('keras', 'tensorflow'):
        return any(b.split('.')[-1] == 'Layer' for b in bases)
    return False


def _is_public_class(node):
    return not node.name.startswith('_') and node.name not in {'Layer', 'Module'}


def _exported_layer(node, repo):
    """Recognize public layer classes whose intermediate base is imported."""
    if repo == 'keras':
        prefixes = ('keras.layers.',)
    elif repo == 'tensorflow':
        prefixes = ('keras.layers.', 'tf.keras.layers.', 'layers.',
                    'compat.v1.layers.', 'tf.compat.v1.layers.')
    else:
        return False
    return any(
        isinstance(value, ast.Constant) and isinstance(value.value, str)
        and value.value.startswith(prefixes)
        for decorator in node.decorator_list for value in ast.walk(decorator)
    )


def _functional_entry(node, repo, rel):
    if rel.replace('\\', '/') not in FUNCTIONAL_ROOTS.get(repo, ()):
        return False
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONAL_NAMES[repo]


def eligible_source(file):
    path = Path(file)
    return (not path.stem.endswith('_test') and not path.stem.startswith('test_')
            and path.name not in {'jax_layer.py', 'flax_layer.py'}
            and not any(part in {'tests', 'test', 'jax', 'flax'}
                        or part.endswith('_tests') or part.startswith('test_')
                        for part in path.parts))


def scan_layers(sources, frameworks=FRAMEWORKS):
    """Yield reproducible source-backed layer and public operation entrypoints.

    Only top-level functions/classes are roots; methods are explored *within*
    their parent task. Indirect inheritance is resolved within the same file.
    """
    for repo in frameworks:
        base = sources / repo
        if not base.is_dir():
            continue
        for directory, dirs, files in os.walk(base):
            dirs[:] = sorted(
                d for d in dirs
                if d not in SKIP and not d.startswith('.')
                and not d.endswith('_tests') and not d.startswith('test_')
            )
            for filename in sorted(files):
                if not filename.endswith('.py'):
                    continue
                path = Path(directory) / filename
                if path.is_symlink() or path.stat().st_size > 2_000_000:
                    continue
                try:
                    raw = path.read_bytes()
                    tree = ast.parse(raw.decode('utf-8'), filename=str(path))
                except (OSError, SyntaxError, UnicodeError, ValueError):
                    continue
                rel = path.relative_to(base).as_posix()
                if not eligible_source(rel):
                    continue
                sha = hashlib.sha256(raw).hexdigest()
                aliases = _imports(tree)
                classes = {n.name:n for n in tree.body if isinstance(n, ast.ClassDef)}
                direct = {name for name,node in classes.items()
                          if _direct_layer(node,repo,rel,aliases)
                          or (_is_public_class(node) and _exported_layer(node,repo))}
                layer_names = set(direct)
                changed = True
                while changed:
                    changed = False
                    for name,node in classes.items():
                        if name not in layer_names and any(b.split('.')[-1] in layer_names
                                for b in _bases(node,aliases)):
                            layer_names.add(name)
                            changed = True
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name in layer_names and _is_public_class(node):
                        kind = 'layer_class' if repo in ('keras','tensorflow') else 'module_class'
                    elif _functional_entry(node,repo,rel):
                        kind = 'functional_operation'
                    else:
                        continue
                    yield (stable(repo,rel,node.lineno,node.name,sha),repo,node.name,
                           rel,node.lineno,sha,kind)

def ensure_layer_inventory(db, sources, discover=False):
    setup(db)
    existing = db.execute('SELECT COUNT(*) FROM trace_layers').fetchone()[0]
    if existing and not discover:
        return 0
    return enqueue_layers(db, sources)


def enqueue_layers(db, sources):
    setup(db)
    inserted = 0
    for row in scan_layers(sources):
        identity = row[1:7]
        existing = db.execute(
            '''SELECT id FROM trace_layers
               WHERE framework=? AND name=? AND source_file=? AND line=? AND sha256=? AND kind=?
                 AND state NOT GLOB 'duplicate_of:*'
               ORDER BY CASE WHEN id=? THEN 0 ELSE 1 END,id LIMIT 1''',
            (*identity,row[0]),
        ).fetchone()
        if existing:
            layer_id = existing[0]
        else:
            cur = db.execute(
                'INSERT OR IGNORE INTO trace_layers(id,framework,name,source_file,line,sha256,kind) VALUES(?,?,?,?,?,?,?)',
                row,
            )
            inserted += cur.rowcount
            layer_id = row[0]
        db.execute('''INSERT OR IGNORE INTO trace_tasks(id,layer_id,parent_id,file,line,decisions,depth)
                      VALUES(?,?,NULL,?,?,'{}',0)''',
                   (stable('root',layer_id),layer_id,row[3],row[4]))
    return inserted


def deduplicate_layer_inventory(db_path):
    path = Path(db_path)
    with sqlite3.connect(path,timeout=30) as db:
        setup(db)
        groups = db.execute(
            '''SELECT framework,name,source_file,line,sha256,kind,group_concat(id)
               FROM trace_layers WHERE state NOT GLOB 'duplicate_of:*'
               GROUP BY framework,name,source_file,line,sha256,kind HAVING COUNT(*)>1'''
        ).fetchall()
        if not groups:
            return {'aliases':0,'backup':None}
        aliases = []
        for framework,name,source_file,line,sha256,kind,ids in groups:
            ids = set(ids.split(','))
            expected = stable(framework,source_file,line,name,sha256)
            canonical = expected if expected in ids else min(ids)
            aliases.extend((alias,canonical) for alias in sorted(ids-{canonical}))
        backup = path.with_name(path.name + '.before-layer-dedup.bak')
        if backup.exists():
            backup = path.with_name(path.name + '.before-layer-dedup.' + uuid.uuid4().hex[:8] + '.bak')
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            db.execute('BEGIN IMMEDIATE')
            active = db.execute(
                f'''SELECT COUNT(*) FROM trace_leases
                    WHERE expires>unixepoch() AND task_id IN (
                      SELECT id FROM trace_tasks WHERE layer_id IN ({','.join('?' for _ in aliases)})
                    )''',
                [alias for alias,_ in aliases],
            ).fetchone()[0]
            if active:
                raise RuntimeError(f'{active} duplicate-root tasks still have live leases')
            for alias,canonical in aliases:
                db.execute('UPDATE trace_layers SET state=? WHERE id=?',
                           (f'duplicate_of:{canonical}',alias))
                db.execute(
                    "UPDATE trace_tasks SET state='suppressed',error=? WHERE layer_id=? AND state='pending'",
                    (f'duplicate root suppressed; canonical layer id is {canonical}',alias),
                )
            db.execute(
                '''CREATE UNIQUE INDEX IF NOT EXISTS trace_layers_natural_unique
                   ON trace_layers(framework,name,source_file,line,sha256,kind)
                   WHERE state NOT GLOB 'duplicate_of:*' '''
            )
        return {'aliases':len(aliases),'backup':str(backup)}


def _case_semantic_key(values):
    semantic = tuple(value for name,value in zip(COLUMNS,values) if name != 'case_id')
    return hashlib.sha256(json.dumps(semantic,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def deduplicate_trace_cases(db_path):
    path=Path(db_path)
    with sqlite3.connect(path,timeout=30) as db:
        setup(db)
        names=','.join(f'"{name}"' for name in COLUMNS)
        rows=db.execute(f'SELECT {names} FROM optimization_cases').fetchall()
        groups={}
        for row in rows:
            groups.setdefault(_case_semantic_key(row),[]).append(row)
        duplicates=sum(len(group)-1 for group in groups.values())
        mapped=db.execute('SELECT COUNT(*) FROM trace_case_semantics').fetchone()[0]
        if not duplicates and mapped==len(rows):
            return {'aliases':0,'backup':None}
        backup=path.with_name(path.name+'.before-case-dedup.bak')
        if backup.exists():
            backup=path.with_name(path.name+'.before-case-dedup.'+uuid.uuid4().hex[:8]+'.bak')
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute(f'SELECT {names} FROM optimization_cases').fetchall()
            groups={}
            for row in rows:
                groups.setdefault(_case_semantic_key(row),[]).append(row)
            for semantic_key,group in groups.items():
                cases=[dict(zip(COLUMNS,row)) for row in group]
                mapped=db.execute(
                    'SELECT canonical_case_id FROM trace_case_semantics WHERE semantic_key=?',
                    (semantic_key,),
                ).fetchone()
                if mapped:
                    canonical=mapped[0]
                    if not any(case['case_id']==canonical for case in cases):
                        raise ValueError('case semantic index points to a missing canonical case')
                else:
                    active=[]
                    for case in cases:
                        states=db.execute(
                            '''SELECT l.state FROM trace_case_evidence e
                               JOIN trace_layers l ON l.id=e.layer_id WHERE e.case_id=?''',
                            (case['case_id'],),
                        ).fetchall()
                        if any(not state.startswith('duplicate_of:') for (state,) in states):
                            active.append(case)
                    canonical_case=min(active or cases,key=lambda item:item['case_id'])
                    canonical=canonical_case['case_id']
                    db.execute('INSERT INTO trace_case_semantics VALUES(?,?)',(semantic_key,canonical))
                for case in cases:
                    alias=case['case_id']
                    if alias==canonical:
                        continue
                    payload=json.dumps(case,ensure_ascii=False,sort_keys=True)
                    db.execute(
                        '''INSERT OR IGNORE INTO trace_case_aliases
                           (alias_case_id,canonical_case_id,case_payload,reason) VALUES(?,?,?,?)''',
                        (alias,canonical,payload,'identical optimization case fields except case_id'),
                    )
                    saved=db.execute(
                        'SELECT canonical_case_id,case_payload FROM trace_case_aliases WHERE alias_case_id=?',
                        (alias,),
                    ).fetchone()
                    if saved!=(canonical,payload):
                        raise ValueError('conflicting optimization case alias '+alias)
                    evidence_rows=db.execute(
                        '''SELECT task_id,layer_id,source_sha256,branch_line,evidence
                           FROM trace_case_evidence WHERE case_id=?''',(alias,),
                    ).fetchall()
                    for task_id,layer_id,source_sha,branch_line,evidence in evidence_rows:
                        db.execute(
                            'INSERT OR IGNORE INTO trace_case_evidence VALUES(?,?,?,?,?,?)',
                            (canonical,task_id,layer_id,source_sha,branch_line,evidence),
                        )
                        preserved=db.execute(
                            '''SELECT layer_id,source_sha256,branch_line,evidence
                               FROM trace_case_evidence WHERE case_id=? AND task_id=?''',
                            (canonical,task_id),
                        ).fetchone()
                        if preserved!=(layer_id,source_sha,branch_line,evidence):
                            raise ValueError('conflicting canonical case provenance '+alias)
                    db.execute('DELETE FROM optimization_cases WHERE case_id=?',(alias,))
        return {'aliases':duplicates,'backup':str(backup)}


OPTIMIZATION_FLAGS = {
    'bias', 'has_bias', 'use_bias', 'tf32', 'allow_tf32', 'amp', 'autocast',
    'deterministic', 'benchmark', 'autotune', 'jit', 'xla', 'fusion', 'fused',
    'cudnn', 'cublas', 'cublaslt', 'cutlass', 'triton', 'flash', 'algorithm',
    'isgloballydisabledaddmmcudalt', 'isinputcompliesaddmmcudalt',
    'disable_addmm_cuda_lt', 'use_bias_ptr_lt', 'distinct_c_and_d_candidate',
    'channels_last', 'contiguous', 'is_cuda', 'is_contiguous', 'requires_grad',
    'enable_tf32', 'enable_flash', 'is_xla', 'is_mkldnn', 'is_contiguous_or_false', 'is_bias_likely_fusable', 'parseLinearFlatten3d', 'flatten3d',
}
OPTIMIZATION_HYPERPARAMETERS = {
    'stride', 'strides', 'dtype', 'layout', 'alignment', 'padding',
    'input_dim', 'weight_dim', 'batch_size', 'dim', 'numel', 'scalar_type',
    'dilation', 'groups', 'num_groups', 'heads', 'num_heads', 'head_dim',
    'kernel_size', 'channels', 'in_channels', 'out_channels', 'input_size',
    'output_size', 'width', 'height', 'dropout', 'rate', 'recurrent_dropout',
}


def _identifier_names(node):
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            names.add(item.id.lower())
        elif isinstance(item, ast.Attribute):
            names.add(item.attr.lower())
    return names


def _optimization_dependency(test):
    names = _identifier_names(test)
    if names & OPTIMIZATION_FLAGS or names & {'dropout', 'rate', 'recurrent_dropout'} or any(
            name.startswith(('use_', 'enable_', 'disable_', 'allow_', 'force_', 'prefer_'))
            for name in names):
        return True
    if names & {'shape', 'rank', 'dim'} and any(
            isinstance(item, ast.Compare) for item in ast.walk(test)):
        return True
    for item in ast.walk(test):
        if not isinstance(item, ast.Compare):
            continue
        operands = [item.left, *item.comparators]
        operand_names = set().union(*(_identifier_names(value) for value in operands))
        if not operand_names & OPTIMIZATION_HYPERPARAMETERS:
            continue
        if any(isinstance(value, ast.Constant) for value in operands):
            return True
        if any(isinstance(value, ast.Attribute) and value.attr.lower() in {
                'float16', 'bfloat16', 'float32', 'float64', 'channels_last',
                'contiguous', 'cuda', 'cpu'} for value in operands):
            return True
    return False


def _has_optimization_dispatch_names(names):
    markers = (
        'vectorized_map', 'fast_path', 'fastpath', 'nested_tensor', 'flash',
        'fused', 'addmm', 'bmm', 'gemm', 'matmul', 'convolution', 'cudnn',
        'triton', 'cutlass', 'algorithm', 'kernel', 'constant_op', 'optimized',
    )
    return any(marker in name for name in names for marker in markers)


def _has_optimization_dispatch(statements):
    names = _identifier_names(ast.Module(body=statements, type_ignores=[]))
    return _has_optimization_dispatch_names(names)


def _optimization_if(node, fallback=None):
    if not isinstance(node, ast.If) or not _optimization_dependency(node.test):
        return False
    condition_names = _identifier_names(node.test)
    shape_names = {'shape', 'rank', 'dim'}
    explicit_controls = OPTIMIZATION_FLAGS | (OPTIMIZATION_HYPERPARAMETERS - shape_names)
    explicit_controls |= {'dropout', 'rate', 'recurrent_dropout'}
    prefixed_controls = any(
        name.startswith(('use_', 'enable_', 'disable_', 'allow_', 'force_', 'prefer_'))
        for name in condition_names
    )
    shape_branch = node.body + node.orelse
    if isinstance(fallback, (ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
        shape_branch.append(fallback)
    if condition_names & shape_names and not condition_names & explicit_controls \
            and not prefixed_controls and not _has_optimization_dispatch(shape_branch):
        return False
    true_body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
    false_body = ast.dump(ast.Module(body=node.orelse, type_ignores=[]), include_attributes=False)
    if true_body == false_body:
        return False
    meaningful = (ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)
    def changes_path(statements):
        for statement in statements:
            if isinstance(statement, meaningful):
                return True
            if isinstance(statement, ast.If) and (
                    changes_path(statement.body) or changes_path(statement.orelse)):
                return True
        return False
    return changes_path(node.body) or changes_path(node.orelse)




def _optimization_nodes(scope):
    followers = {}
    for parent in ast.walk(scope):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for current, following in zip(value, value[1:]):
                if isinstance(current, ast.stmt) and isinstance(following, ast.stmt):
                    followers[current] = following
    return [node for node in ast.walk(scope) if isinstance(node, ast.If)
            and _optimization_if(node, followers.get(node))]

def relevant_branch(expression):
    """Require a recognized flag or a compared hyperparameter."""
    try:
        test = ast.parse(expression, mode='eval').body
    except (SyntaxError, ValueError):
        return False
    return _optimization_dependency(test)


NATIVE_SUFFIXES = {'.c', '.cc', '.cpp', '.cxx', '.cu', '.cuh', '.h', '.hh', '.hpp'}

def _cpp_code(line):
    return re.sub(r'//.*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '', line)

def _cpp_function_bounds(lines, line):
    signature = re.compile(r'^\s*(?:(?:static|inline|constexpr|TORCH_API)\s+)*(?:[\w:<>]+(?:\s*[*&])?\s+)+([~\w:]+)\s*\(')
    matches=[]
    for start in range(line):
        clean=_cpp_code(lines[start])
        match=signature.match(clean)
        if not match:
            continue
        opening=None
        for j in range(start,min(len(lines),start+12)):
            text=_cpp_code(lines[j])
            if '{' in text:
                opening=(j,text.index('{')); break
            if ';' in text: break
        if opening is None: continue
        depth=0; end=None
        for j in range(opening[0],len(lines)):
            text=_cpp_code(lines[j])
            for k,char in enumerate(text):
                if j==opening[0] and k<opening[1]: continue
                if char=='{': depth+=1
                elif char=='}':
                    depth-=1
                    if depth==0:
                        end=j; break
            if end is not None: break
        if end is not None and start+1 <= line <= end+1:
            matches.append((start+1,end+1,match.group(1).split('::')[-1]))
    return min(matches,key=lambda item:item[1]-item[0]) if matches else None

def native_branch_nodes(raw, line):
    lines=raw.splitlines(); bounds=_cpp_function_bounds(lines,line)
    if not bounds: return []
    scope_start,scope_end,function=bounds; found=[]
    for i in range(scope_start-1,scope_end):
        clean=_cpp_code(lines[i]); m=re.search(r'\bif\s*\(',clean)
        if not m: continue
        cond_start=m.end(); depth=1; j=i; k=cond_start
        while j<len(lines) and depth:
            text=_cpp_code(lines[j])
            while k<len(text):
                if text[k]=='(': depth+=1
                elif text[k]==')':
                    depth-=1
                    if depth==0: break
                k+=1
            if depth:
                j+=1;k=0
        if depth: continue
        condition=' '.join([clean[cond_start:k]]+[ _cpp_code(x) for x in lines[i+1:j] ]).strip()
        opener=None
        for q in range(j,min(scope_end,j+3)):
            text=_cpp_code(lines[q])
            if '{' in text: opener=(q,text.index('{')); break
            if ';' in text: break
        if opener is None: continue
        d=0; close=None
        for q in range(opener[0],scope_end):
            text=_cpp_code(lines[q])
            for z,ch in enumerate(text):
                if q==opener[0] and z<opener[1]: continue
                if ch=='{': d+=1
                elif ch=='}':
                    d-=1
                    if d==0: close=(q,z); break
            if close: break
        if not close: continue
        names=set(re.findall(r'[A-Za-z_]\w*',condition.lower()))
        if not names & (OPTIMIZATION_FLAGS|OPTIMIZATION_HYPERPARAMETERS) and not any(
                name.startswith(('use_','enable_','disable_','allow_','force_','prefer_')) for name in names):
            continue
        true_start=opener[0]+2; true_end=close[0]+1
        body='\n'.join(lines[true_start-1:true_end])
        if not re.search(r'\b(return|co_return)\b|\bat::|\boutput\s*(?:=|\.)|\b(?:input|weight|bias)\s*(?:=|\.)',body):
            continue
        tail=_cpp_code(lines[close[0]])[close[1]+1:]
        nextline=close[0]+1
        if not re.search(r'\belse\b',tail) and nextline<scope_end and re.match(r'^\s*else\b',_cpp_code(lines[nextline])):
            tail=_cpp_code(lines[nextline])
        em=re.search(r'\belse\s*\{',tail)
        if em:
            el_open=(close[0] if 'else' in _cpp_code(lines[close[0]]) else nextline,
                     ( _cpp_code(lines[close[0]]).find('{',close[1]+1) if 'else' in _cpp_code(lines[close[0]]) else _cpp_code(lines[nextline]).find('{')))
            d=0; el_close=None
            for q in range(el_open[0],scope_end):
                text=_cpp_code(lines[q])
                for z,ch in enumerate(text):
                    if q==el_open[0] and z<el_open[1]: continue
                    if ch=='{': d+=1
                    elif ch=='}':
                        d-=1
                        if d==0: el_close=(q,z); break
                if el_close: break
            false_start=el_open[0]+2; false_end=el_close[0]+1 if el_close else false_start
        else:
            false_start=false_end=None
        generic_shape_names = {'shape', 'rank', 'dim'}
        explicit_controls = OPTIMIZATION_FLAGS | (OPTIMIZATION_HYPERPARAMETERS - generic_shape_names)
        prefixed_controls = any(
            name.startswith(('use_', 'enable_', 'disable_', 'allow_', 'force_', 'prefer_'))
            for name in names
        )
        if names & generic_shape_names and not names & explicit_controls \
                and not prefixed_controls:
            both_paths = body + ('\n' + '\n'.join(lines[false_start-1:false_end])
                                 if false_start is not None else '')
            path_names = set(re.findall(r'[A-Za-z_]\w*', both_paths.lower()))
            if not _has_optimization_dispatch_names(path_names):
                continue
        found.append({'line':i+1,'expression':condition,'true_start':true_start,
                      'true_end':true_end,'false_start':false_start,'false_end':false_end,
                      'scope_start':scope_start,'scope_end':scope_end,'function':function,
                      'branch_end':close[0]+1})
    return found

def branch_candidates(raw, line):
    """Return conditions controlling distinct source paths, not keyword matches."""
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return []
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
              and n.lineno <= line <= getattr(n, 'end_lineno', n.lineno)]
    scope = min(scopes, key=lambda n: getattr(n, 'end_lineno', n.lineno)-n.lineno) if scopes else tree
    return sorted((n.lineno, ast.get_source_segment(raw, n.test) or ast.unparse(n.test))
                  for n in _optimization_nodes(scope))


def _branch_node(raw, line):
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return None
    return next((node for node in _optimization_nodes(tree) if node.lineno == line), None)


def _branch_span(statements, fallback):
    if not statements:
        return fallback, fallback
    return min(node.lineno for node in statements), max(
        getattr(node, 'end_lineno', node.lineno) for node in statements
    )


def _branch_text(lines, statements, label, fallback):
    if not statements:
        return f'{label}: implicit fallthrough after line {fallback}'
    first, last = _branch_span(statements, fallback)
    excerpt = '\n'.join(lines[first - 1:last])[:1200]
    return f'{label} source path, lines {first}-{last}: {excerpt}'


def _source_function(raw, line, fallback):
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return fallback
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
              and node.lineno <= line <= getattr(node, 'end_lineno', node.lineno)]
    return min(scopes, key=lambda node: getattr(node, 'end_lineno', node.lineno)-node.lineno).name if scopes else fallback


def _trace_case(db, task, ctx, branch):
    line = int(branch[0].rsplit(':', 1)[1])
    source = '\n'.join(ctx['lines'])
    node = _branch_node(source, line)
    if node is None:
        return None
    _, layer_id, repo, file = task
    layer = db.execute('SELECT name FROM trace_layers WHERE id=?', (layer_id,)).fetchone()
    layer_name = layer[0] if layer else 'UNKNOWN'
    expression = ast.get_source_segment(source, node.test) or ast.unparse(node.test)
    first, last = line, getattr(node, 'end_lineno', line)
    true_path = _branch_text(ctx['lines'], node.body, 'predicate true', last)
    false_path = _branch_text(ctx['lines'], node.orelse, 'predicate false', last)
    case_id = 'TRACE_' + stable('optimization-branch', layer_id, file, ctx['sha256'], line, expression)[:20]
    return normalize_case({
        'case_id': case_id, 'framework': repo, 'framework_version': 'UNKNOWN',
        'library': repo, 'library_version': 'UNKNOWN', 'backend': 'UNKNOWN',
        'hardware': 'UNKNOWN', 'optimization_stage': 'source_dispatch_candidate',
        'optimization_name': f'source branch: {expression}'[:500],
        'layer_combination': layer_name, 'operator_pattern': 'UNKNOWN',
        'topology_constraints': 'UNKNOWN',
        'layer_hyperparameters': {'controlling_predicate': expression},
        'tensor_constraints': 'UNKNOWN', 'compiler_flags': 'UNKNOWN',
        'applicability_predicate': expression, 'optimization_outcome': true_path,
        'fusion_type': 'none', 'fused_operations': [],
        'remaining_operations': 'UNKNOWN', 'expected_kernel_effect': 'possible_change',
        'alternative_outcome': false_path, 'source_repository': repo,
        'source_file': file, 'source_function': _source_function(source, line, layer_name),
        'source_lines': f'{first}-{last}', 'evidence_status': 'INFERRED',
        'validation_status': 'NEEDS_REVIEW',
    })


def prune_ineligible_optimization_cases(db, sources):
    """Archive trace cases whose current source no longer has an eligible optimization branch."""
    setup(db)
    lock_started = not db.in_transaction
    if lock_started:
        db.execute('BEGIN IMMEDIATE')
    active_leases = db.execute(
        'SELECT COUNT(*) FROM trace_leases WHERE expires>unixepoch()'
    ).fetchone()[0]
    if active_leases:
        raise RuntimeError(f'{active_leases} trace tasks still have live leases')
    tables = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if 'case_repro_scripts' in tables:
        running_repros = db.execute(
            "SELECT COUNT(*) FROM case_repro_scripts WHERE status='running'"
        ).fetchone()[0]
        if running_repros:
            raise RuntimeError(f'{running_repros} repro generation tasks are still running')
    evidence_columns = ('case_id', 'task_id', 'layer_id', 'source_sha256', 'branch_line', 'evidence')
    aliases_columns = ('alias_case_id', 'canonical_case_id', 'case_payload', 'reason')
    cases = db.execute(
        'SELECT ' + ','.join(f'"{name}"' for name in COLUMNS) + ' FROM optimization_cases'
    ).fetchall()
    excluded = []
    source_cache = {}
    for values in cases:
        case = dict(zip(COLUMNS, values))
        evidence = [dict(zip(evidence_columns, row)) for row in db.execute(
            'SELECT case_id,task_id,layer_id,source_sha256,branch_line,evidence '
            'FROM trace_case_evidence WHERE case_id=?', (case['case_id'],),
        )]
        if not evidence:
            continue
        eligible = False
        for item in evidence:
            layer = db.execute(
                'SELECT framework,source_file FROM trace_layers WHERE id=?',
                (item['layer_id'],),
            ).fetchone()
            if not layer:
                continue
            framework, relative = layer
            key = (framework, relative)
            if key not in source_cache:
                try:
                    path = safe_source(Path(sources), framework, relative)
                    raw = path.read_bytes()
                    source_cache[key] = (hashlib.sha256(raw).hexdigest(), raw.decode('utf-8'))
                except (OSError, UnicodeError, ValueError):
                    source_cache[key] = (None, None)
            digest, raw = source_cache[key]
            if digest != item['source_sha256'] or raw is None:
                continue
            if any(line == item['branch_line'] for line, _ in branch_candidates(raw, item['branch_line'])):
                eligible = True
                break
        if not eligible:
            aliases = [dict(zip(aliases_columns, row)) for row in db.execute(
                'SELECT alias_case_id,canonical_case_id,case_payload,reason '
                'FROM trace_case_aliases WHERE alias_case_id=? OR canonical_case_id=?',
                (case['case_id'], case['case_id']),
            )]
            excluded.append((case, evidence, aliases))
    if not excluded:
        if lock_started:
            db.commit()
        return {'cases_excluded': 0, 'backup': None}
    if not lock_started and db.in_transaction:
        raise RuntimeError('commit database changes before pruning optimization cases')
    path = Path(db.execute('PRAGMA database_list').fetchone()[2])
    backup = path.with_name(path.name + '.before-optimization-case-prune.bak')
    if backup.exists():
        backup = path.with_name(
            path.name + '.before-optimization-case-prune.' + uuid.uuid4().hex[:8] + '.bak'
        )
    with sqlite3.connect(path, timeout=30) as source_db, sqlite3.connect(backup) as out:
        source_db.backup(out)
    reason = 'no current source-backed optimization branch matches stored evidence'
    with db:
        for case, evidence, aliases in excluded:
            case_id = case['case_id']
            db.execute(
                'INSERT OR REPLACE INTO optimization_case_exclusions '
                '(case_id,case_payload,evidence_payload,aliases_payload,reason) VALUES(?,?,?,?,?)',
                (case_id, json.dumps(case, ensure_ascii=False, sort_keys=True),
                 json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                 json.dumps(aliases, ensure_ascii=False, sort_keys=True), reason),
            )
            db.execute('DELETE FROM optimization_cases WHERE case_id=?', (case_id,))
            db.execute('DELETE FROM trace_case_evidence WHERE case_id=?', (case_id,))
            db.execute('DELETE FROM trace_case_semantics WHERE canonical_case_id=?', (case_id,))
            db.execute(
                'DELETE FROM trace_case_aliases WHERE alias_case_id=? OR canonical_case_id=?',
                (case_id, case_id),
            )
    return {'cases_excluded': len(excluded), 'backup': str(backup)}


def seed_branch_cases(db, sources):
    """Persist every optimization-dependent source branch as a reviewable candidate."""
    setup(db)
    exclusion = prune_ineligible_optimization_cases(db, sources)
    candidates = []
    for layer_id, repo, file, root_line, source_sha in db.execute(
            'SELECT id,framework,source_file,line,sha256 FROM trace_layers '
            "WHERE framework IN ('keras','tensorflow','pytorch') "
            "AND state NOT GLOB 'duplicate_of:*' "
            'ORDER BY framework,source_file,line'):
        ctx = context(sources, repo, file, root_line, source_sha)
        root_task = db.execute(
            'SELECT id FROM trace_tasks WHERE layer_id=? AND parent_id IS NULL '
            'ORDER BY depth,created_at,id LIMIT 1', (layer_id,),
        ).fetchone()
        task_id = root_task[0] if root_task else stable('root', layer_id)
        for branch_line, expression in ctx['branches']:
            branch = (f'{file}:{branch_line}', expression)
            case = (_trace_case(db, (task_id,layer_id,repo,file), ctx, branch)
                    if not ctx['native'] else None)
            if case is not None:
                candidates.append((case,task_id,layer_id,source_sha,branch_line))

    names = ','.join(f'"{name}"' for name in COLUMNS)
    by_id = {}
    for case,task_id,layer_id,source_sha,branch_line in candidates:
        values = tuple(case[name] for name in COLUMNS)
        prior = by_id.setdefault(case['case_id'], values)
        if prior != values:
            raise ValueError('conflicting seeded optimization case ' + case['case_id'])
        existing = db.execute(
            f'SELECT {names} FROM optimization_cases WHERE case_id=?',
            (case['case_id'],),
        ).fetchone()
        if existing is not None and existing != values:
            raise ValueError('conflicting seeded optimization case ' + case['case_id'])

    changed = 0
    for case,task_id,layer_id,source_sha,branch_line in candidates:
        evidence = db.execute(
            'SELECT 1 FROM trace_case_evidence WHERE case_id=? AND task_id=?',
            (case['case_id'],task_id),
        ).fetchone()
        exists = db.execute(
            'SELECT 1 FROM optimization_cases WHERE case_id=?', (case['case_id'],),
        ).fetchone()
        changed += int(not exists or not evidence)
    backup = None
    if changed:
        path = Path(db.execute('PRAGMA database_list').fetchone()[2])
        backup_path = path.with_name(path.name + '.before-branch-case-seed.bak')
        if backup_path.exists():
            backup_path = path.with_name(
                path.name + '.before-branch-case-seed.' + uuid.uuid4().hex[:8] + '.bak')
        with sqlite3.connect(backup_path) as out:
            db.backup(out)
        backup = str(backup_path)

    inserted_cases = inserted_evidence = 0
    with db:
        for case,task_id,layer_id,source_sha,branch_line in candidates:
            has_evidence = db.execute(
                'SELECT 1 FROM trace_case_evidence WHERE case_id=? AND task_id=?',
                (case['case_id'],task_id),
            ).fetchone()
            inserted_cases += _persist_trace_case(
                db,case,task_id,layer_id,source_sha,branch_line)
            inserted_evidence += int(has_evidence is None)
    return {'candidates':len(by_id),'cases_inserted':inserted_cases,
            'evidence_rows_added':inserted_evidence,'backup':backup,
            'cases_excluded':exclusion['cases_excluded'],
            'exclusion_backup':exclusion['backup']}


def _persist_trace_case(db, case, task_id, layer_id, source_sha, branch_line):
    names = ','.join(f'"{name}"' for name in COLUMNS)
    marks = ','.join('?' for _ in COLUMNS)
    values = tuple(case[name] for name in COLUMNS)
    semantic_key = _case_semantic_key(values)
    if not db.in_transaction:
        db.execute('BEGIN IMMEDIATE')
    db.execute('INSERT OR IGNORE INTO trace_case_semantics VALUES(?,?)',
               (semantic_key,case['case_id']))
    canonical = db.execute(
        'SELECT canonical_case_id FROM trace_case_semantics WHERE semantic_key=?',
        (semantic_key,),
    ).fetchone()[0]
    canonical_row = db.execute(
        f'SELECT {names} FROM optimization_cases WHERE case_id=?',(canonical,)
    ).fetchone()
    if canonical_row is None:
        if canonical != case['case_id']:
            raise ValueError('case semantic index points to a missing canonical case')
        inserted = db.execute(
            f'INSERT OR IGNORE INTO optimization_cases ({names}) VALUES ({marks})', values
        ).rowcount
        if not inserted:
            existing = db.execute(
                f'SELECT {names} FROM optimization_cases WHERE case_id=?',(case['case_id'],)
            ).fetchone()
            if existing != values:
                raise ValueError("conflicting DFS optimization case " + case["case_id"])
    else:
        semantic = tuple(v for name,v in zip(COLUMNS,canonical_row) if name != 'case_id')
        proposed = tuple(v for name,v in zip(COLUMNS,values) if name != 'case_id')
        if semantic != proposed:
            raise ValueError('case semantic index conflicts with case payload')
        inserted = 0
        if canonical != case['case_id']:
            payload=json.dumps(case,ensure_ascii=False,sort_keys=True)
            db.execute(
                '''INSERT OR IGNORE INTO trace_case_aliases
                   (alias_case_id,canonical_case_id,case_payload,reason) VALUES(?,?,?,?)''',
                (case['case_id'],canonical,payload,'identical optimization case fields except case_id'),
            )
            saved=db.execute(
                'SELECT canonical_case_id,case_payload FROM trace_case_aliases WHERE alias_case_id=?',
                (case['case_id'],),
            ).fetchone()
            if saved!=(canonical,payload):
                raise ValueError('conflicting optimization case alias '+case['case_id'])
    evidence = {
        'predicate': case['applicability_predicate'],
        'source_file': case['source_file'],
        'source_lines': case['source_lines'],
        'true_path': case['optimization_outcome'],
        'false_path': case['alternative_outcome'],
        'evidence_status': case['evidence_status'],
    }
    evidence_json = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    db.execute(
        'INSERT OR IGNORE INTO trace_case_evidence VALUES(?,?,?,?,?,?)',
        (canonical, task_id, layer_id, source_sha, branch_line, evidence_json),
    )
    saved = db.execute(
        'SELECT layer_id,source_sha256,branch_line,evidence FROM trace_case_evidence WHERE case_id=? AND task_id=?',
        (canonical, task_id),
    ).fetchone()
    if saved != (layer_id,source_sha,branch_line,evidence_json):
        raise ValueError("conflicting DFS optimization case evidence " + canonical)
    return inserted

def context(sources, repo, file, line, expected_sha=None):
    path = safe_source(sources, repo, file)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and sha != expected_sha:
        raise ValueError('source drift: file differs from initial layer inventory')
    src = raw.decode('utf-8')
    lines = src.splitlines()
    if not 1 <= line <= len(lines):
        raise ValueError('line out of bounds')
    lo, hi = max(1,line-8), min(len(lines),line+60)
    excerpt=[]
    visible_lines={}
    for number in range(lo,hi+1):
        encoded=lines[number-1].encode('utf-8')
        visible=encoded[:300].decode('utf-8','ignore')
        visible_lines[number]=visible
        suffix=' …[line truncated]' if len(encoded)>300 else ''
        excerpt.append(f"{number}: {visible}{suffix}")
    while len('\n'.join(excerpt).encode('utf-8')) > 5000:
        removed=excerpt.pop()
        visible_lines.pop(int(removed.split(':',1)[0]),None)

    native = path.suffix in NATIVE_SUFFIXES
    native_nodes = native_branch_nodes(src,line) if native else []
    branches = ([(node['line'],node['expression']) for node in native_nodes]
                if native else branch_candidates(src,line))
    return {'file':file,'line':line,'sha256':sha,
            'source':'\n'.join(excerpt),
            'lines':lines,'visible_lines':visible_lines,'branches':branches,'native':native,
            'native_branches':native_nodes}


def choose_branch(ctx, decisions):
    for line, expression in ctx['branches']:
        key = f"{ctx['file']}:{line}"
        if key not in decisions and line == ctx['line']:
            return key, expression
    return None


def next_native_branch(ctx, decisions):
    for node in ctx.get('native_branches',[]):
        key=f"{ctx['file']}:{node['line']}"
        if key not in decisions:
            return node['line'],node['expression'],node['scope_start'],node['scope_end']
    return None

def native_branch_at(ctx, line):
    return next((node for node in ctx.get('native_branches',[]) if node['line']==line),None)

def _native_source_path(lines, first, last, label):
    if first is None:
        return f"{label}: implicit fallthrough"
    excerpt='\\n'.join(f"{i}: {lines[i-1]}" for i in range(first,last+1))
    return f"{label} source path, lines {first}-{last}: {excerpt[:1200]}"

def _trace_native_case(db, task, ctx, branch):
    line=int(branch[0].rsplit(':',1)[1]); node=native_branch_at(ctx,line)
    if node is None: return None
    _,layer_id,repo,file=task
    layer=db.execute('SELECT name FROM trace_layers WHERE id=?',(layer_id,)).fetchone()
    layer_name=layer[0] if layer else 'UNKNOWN'
    true_path=_native_source_path(ctx['lines'],node['true_start'],node['true_end'],'predicate true')
    false_path=(_native_source_path(ctx['lines'],node['false_start'],node['false_end'],'predicate false')
                if node['false_start'] is not None else f"predicate false: implicit fallthrough after line {node['branch_end']}")
    case_id='TRACE_'+stable('optimization-branch',layer_id,file,ctx['sha256'],line,node['expression'])[:20]
    return normalize_case({
        'case_id':case_id,'framework':repo,'framework_version':'UNKNOWN',
        'library':repo,'library_version':'UNKNOWN','backend':'UNKNOWN','hardware':'UNKNOWN',
        'optimization_stage':'source_dispatch_candidate','optimization_name':f"source branch: {node['expression']}"[:500],
        'layer_combination':layer_name,'operator_pattern':'UNKNOWN','topology_constraints':'UNKNOWN',
        'layer_hyperparameters':{'controlling_predicate':node['expression']},
        'tensor_constraints':'UNKNOWN','compiler_flags':'UNKNOWN','applicability_predicate':node['expression'],
        'optimization_outcome':true_path,'fusion_type':'none','fused_operations':[],
        'remaining_operations':'UNKNOWN','expected_kernel_effect':'possible_change',
        'alternative_outcome':false_path,'source_repository':repo,'source_file':file,
        'source_function':node['function'],'source_lines':f"{line}-{node['branch_end']}",
        'evidence_status':'INFERRED','validation_status':'NEEDS_REVIEW'})

def next_unvisited_branch(ctx, decisions):
    if ctx.get('native'):
        return next_native_branch(ctx, decisions)
    """Return the next branch in the current Python function, if the trace skipped it."""
    source = '\n'.join(ctx['lines'])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    class_scopes = [node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                    and node.lineno <= ctx['line'] <= getattr(node, 'end_lineno', node.lineno)]
    if class_scopes:
        scope = min(class_scopes, key=lambda node: getattr(node, 'end_lineno', node.lineno)-node.lineno)
    else:
        function_scopes = [node for node in ast.walk(tree)
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.lineno <= ctx['line'] <= getattr(node, 'end_lineno', node.lineno)]
        if not function_scopes:
            return None
        scope = min(function_scopes, key=lambda node: getattr(node, 'end_lineno', node.lineno)-node.lineno)
    candidates = sorted(
        (node.lineno, ast.get_source_segment(source, node.test) or ast.unparse(node.test))
        for node in _optimization_nodes(scope)
    )
    for line, expression in candidates:
        key = f"{ctx['file']}:{line}"
        if key not in decisions:
            return line, expression, scope.lineno, getattr(scope, 'end_lineno', scope.lineno)
    return None


def next_source_call(ctx):
    """Return the next same-function source call as an exact, local candidate."""
    source = "\n".join(ctx["lines"])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.lineno <= ctx["line"] <= getattr(node, "end_lineno", node.lineno)]
    if not scopes:
        return None
    scope = min(scopes, key=lambda node: getattr(node, "end_lineno", node.lineno)-node.lineno)

    class Calls(ast.NodeVisitor):
        def __init__(self):
            self.lines = []

        def visit_Call(self, node):
            if node.lineno > ctx["line"]:
                self.lines.append(node.lineno)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node is scope:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            return

    calls = Calls()
    for statement in scope.body:
        calls.visit(statement)
    if not calls.lines:
        return None
    line = min(calls.lines)
    return {"file": ctx["file"], "line": line,
            "quote": ctx["lines"][line - 1],
            "sha256": ctx["sha256"],
            "navigation_basis": "next_source_call_candidate"}


def same_file_call_target(ctx):
    """Resolve a call on the current line to one unique local function body."""
    try:
        tree = ast.parse("\n".join(ctx["lines"]))
    except (SyntaxError, ValueError):
        return None, None
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.lineno <= ctx["line"] <= getattr(node, "end_lineno", node.lineno)]
    if not scopes:
        return None, None
    scope = min(scopes, key=lambda node: getattr(node, "end_lineno", node.lineno)-node.lineno)
    names = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = [node for node in ast.walk(scope)
             if isinstance(node, ast.Call) and node.lineno == ctx["line"]
             and isinstance(node.func, ast.Name) and node.func.id in names]
    targets = {node.func.id for node in calls}
    if len(targets) != 1:
        return None, None
    symbol = targets.pop()
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == symbol and node.lineno > ctx["line"]]
    if len(functions) != 1:
        return None, None
    line = functions[0].lineno
    quote = ctx["lines"][line - 1]
    claim_line = ctx["line"]
    claim_quote = ctx["lines"][claim_line - 1].encode("utf-8")[:300].decode("utf-8", "ignore")
    return ({"file": ctx["file"], "line": line, "quote": quote,
             "sha256": ctx["sha256"], "symbol": symbol,
             "navigation_basis": "same_file_call_candidate"},
            {"line": claim_line, "quote": claim_quote,
             "claim": f"This source call invokes the local function {symbol}."})


def imported_base_class_target(sources, repo, ctx):
    """Resolve the first imported base through module aliases and re-exports."""
    source = "\n".join(ctx["lines"])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None, None
    containing = [node for node in ast.walk(tree)
                  if isinstance(node, ast.ClassDef)
                  and node.lineno <= ctx["line"] <= getattr(node, "end_lineno", node.lineno)]
    if not containing:
        return None, None
    cls = min(containing, key=lambda node: getattr(node, "end_lineno", node.lineno)-node.lineno)
    if not cls.bases:
        return None, None

    def module_name(module, level):
        if not level:
            return module or ""
        package = Path(ctx["file"]).parent.parts
        parts = list(package[:max(0, len(package)-level+1)])
        if module:
            parts.extend(module.split("."))
        return ".".join(parts)

    module_aliases = {}
    imported_symbols = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = module_name(node.module, node.level)
            for alias in node.names:
                local = alias.asname or alias.name
                imported_symbols[local] = (module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                module_aliases[local] = alias.name if alias.asname else alias.name.split(".", 1)[0]

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            return node.id, list(reversed(parts))
        return None, []

    root, attrs = dotted(cls.bases[0])
    if root in imported_symbols:
        module, symbol = imported_symbols[root]
        if attrs:
            module = ".".join(part for part in (module, symbol) if part)
            symbol = ".".join(attrs)
    elif root in module_aliases:
        module, symbol = module_aliases[root], ".".join(attrs)
    else:
        return None, None

    visited = set()
    for _ in range(8):
        key = (module, symbol)
        if key in visited or not module or not symbol:
            return None, None
        visited.add(key)
        relative = "/".join(module.split("."))
        target_path = None
        for target_file in (relative + ".py", relative + "/__init__.py"):
            try:
                target_path = safe_source(sources, repo, target_file)
                break
            except ValueError:
                pass
        if target_path is None:
            return None, None
        target_lines = target_path.read_text(encoding="utf-8").splitlines()
        try:
            target_tree = ast.parse("\n".join(target_lines))
        except (SyntaxError, ValueError):
            return None, None
        target_class = next((node for node in target_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == symbol), None)
        if target_class is not None:
            claim_quote = ctx["lines"][cls.lineno - 1]
            claim = {"line": cls.lineno, "quote": claim_quote,
                     "claim": f"{cls.name} inherits from {symbol}"}
            line = target_class.lineno
            return ({"file": target_file, "line": line,
                     "quote": target_lines[line - 1], "symbol": symbol,
                     "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                     "navigation_basis": "imported_base_class_candidate"}, claim)
        reexport = next((node for node in target_tree.body
                         if isinstance(node, ast.ImportFrom)
                         and any((alias.asname or alias.name) == symbol for alias in node.names)), None)
        if reexport is None:
            return None, None
        alias = next(alias for alias in reexport.names
                     if (alias.asname or alias.name) == symbol)
        module = module_name(reexport.module, reexport.level)
        symbol = alias.name
    return None, None

def validate_trace_response(response, ctx, repo, sources, visited):
    claims,nxt,terminal = validate_step(response,ctx,repo,sources)
    if terminal == 'unresolved' and not claims and nxt is None:
        candidate,source_claim = same_file_call_target(ctx)
        if candidate is None:
            candidate = next_source_call(ctx)
        if candidate is None:
            candidate,source_claim = imported_base_class_target(sources,repo,ctx)
        if candidate is None:
            method = next_class_method(ctx,visited)
            if method:
                method_line = method[0]
                candidate = {'file':ctx['file'],'line':method_line,
                             'quote':ctx['lines'][method_line - 1]}
        if candidate is not None:
            hints = []
            if source_claim:
                hint_line = source_claim['line']
                visible = ctx['lines'][hint_line - 1].encode('utf-8')[:300].decode('utf-8','ignore')
                hints.append({'line':hint_line,'exact_visible_quote':visible})
            elif candidate['file'] == ctx['file']:
                hint_line = candidate['line']
                visible = ctx['lines'][hint_line - 1].encode('utf-8')[:300].decode('utf-8','ignore')
                hints.append({'line':hint_line,'exact_visible_quote':visible})
            raise TraceValidationError(
                'unresolved response has no source claims despite a locally verifiable continuation',
                hints,
            )
    return claims,nxt,terminal


def next_class_method(ctx, visited):
    source = "\n".join(ctx["lines"])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    containing = [node for node in ast.walk(tree)
                  if isinstance(node, ast.ClassDef)
                  and node.lineno <= ctx["line"] <= getattr(node, "end_lineno", node.lineno)]
    if not containing:
        return None
    cls = min(containing, key=lambda node: getattr(node, "end_lineno", node.lineno)-node.lineno)
    methods = [node for node in cls.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    current = next((node for node in methods
                    if node.lineno <= ctx["line"] <= getattr(node, "end_lineno", node.lineno)), None)
    for method in methods:
        if current and method.lineno <= current.lineno:
            continue
        marker = f"class_method:{method.lineno}"
        if marker not in visited:
            return method.lineno, marker, cls.lineno, getattr(cls, "end_lineno", cls.lineno)

    pending = [ast.unparse(base).split(".")[-1] for base in cls.bases]
    seen = {cls.name}
    while pending:
        name = pending.pop(0)
        if name in seen or name not in classes:
            continue
        seen.add(name)
        parent = classes[name]
        parent_methods = [node for node in parent.body
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for method in parent_methods:
            marker = f"class_method:{method.lineno}"
            if marker not in visited:
                return (method.lineno, marker, parent.lineno,
                        getattr(parent, "end_lineno", parent.lineno))
        pending.extend(ast.unparse(base).split(".")[-1] for base in parent.bases)
    return None


def linear_binding_candidate(sources, repo, file, ctx):
    if repo != 'pytorch':
        return None
    if file == 'torch/nn/modules/linear.py':
        if 'F.linear(' not in ctx['source']:
            return None
        target='torch/nn/functional.py'
        try:
            lines=safe_source(sources,repo,target).read_text(encoding='utf-8').splitlines()
        except (OSError,ValueError):
            return None
        matches=[i for i,line in enumerate(lines) if line.strip()=='torch._C._nn.linear,']
    elif file == 'torch/nn/functional.py':
        if 'torch._C._nn.linear' not in ctx['source']:
            return None
        target='aten/src/ATen/native/native_functions.yaml'
        try:
            lines=safe_source(sources,repo,target).read_text(encoding='utf-8').splitlines()
        except (OSError,ValueError):
            return None
        matches=[i for i,line in enumerate(lines)
                 if line.strip().startswith('- func: linear(Tensor input, Tensor weight, Tensor? bias=None)')]
    elif file == 'aten/src/ATen/native/native_functions.yaml':
        lines=ctx['lines']
        starts=[i for i,line in enumerate(lines)
                if line.strip().startswith('- func: linear(Tensor input, Tensor weight, Tensor? bias=None)')]
        matches=[]
        for i in starts:
            end=next((j for j in range(i+1,len(lines)) if lines[j].lstrip().startswith('- func:')),len(lines))
            stanza='\n'.join(lines[i:end])
            if i+1 <= ctx['line'] <= end and 'CompositeImplicitAutograd: linear' in stanza:
                target='aten/src/ATen/native/Linear.cpp'
                cpp=safe_source(sources,repo,target).read_text(encoding='utf-8').splitlines()
                matches=[j for j,line in enumerate(cpp)
                         if re.match(r'^\s*Tensor linear\(const Tensor& input, const Tensor& weight,',line)]
                lines=cpp
                break
    else:
        return None
    if len(matches)!=1:
        return None
    i=matches[0]
    return {'file':target,'line':i+1,
            'source':'\n'.join(f'{j+1}: {lines[j]}' for j in range(max(0,i-2),min(len(lines),i+1)))}

def linear_addmm_candidate(sources, repo, file, ctx):
    """Resolve Linear's addmm call through the ATen schema to its CUDA entry."""
    if repo != 'pytorch':
        return None
    if file == 'aten/src/ATen/native/Linear.cpp':
        if not re.search(r'\bat::addmm\s*\(', ctx['source']):
            return None
        target = 'aten/src/ATen/native/native_functions.yaml'
        try:
            lines = safe_source(sources, repo, target).read_text(encoding='utf-8').splitlines()
        except (OSError, ValueError):
            return None
        matches = [i for i, text in enumerate(lines)
                   if text.strip().startswith('- func: addmm.out(')]
    elif file == 'aten/src/ATen/native/native_functions.yaml':
        target = 'aten/src/ATen/native/cuda/Blas.cpp'
        lines = ctx['lines']
        matches = []
        for i, text in enumerate(lines):
            if not text.strip().startswith('- func: addmm.out('):
                continue
            end = next((j for j in range(i + 1, len(lines))
                        if lines[j].lstrip().startswith('- func:')), len(lines))
            if i + 1 <= ctx['line'] <= end and 'CUDA: addmm_out_cuda' in '\n'.join(lines[i:end]):
                try:
                    cuda = safe_source(sources, repo, target).read_text(encoding='utf-8').splitlines()
                except (OSError, ValueError):
                    return None
                matches = [j for j, row in enumerate(cuda)
                           if re.match(r'^\s*TORCH_IMPL_FUNC\(addmm_out_cuda\)', row)]
                lines = cuda
                break
    else:
        return None
    if len(matches) != 1:
        return None
    i = matches[0]
    return {'file': target, 'line': i + 1,
            'source': '\n'.join(f'{j+1}: {lines[j]}'
                                 for j in range(max(0, i-2), min(len(lines), i+3)))}


def verified_linear_addmm_target(sources, repo, file, ctx, claims):
    if not any(re.search(r'\baddmm\b', claim['quote']) for claim in claims):
        return None
    candidate = linear_addmm_candidate(sources, repo, file, ctx)
    if not candidate:
        return None
    target = context(sources, repo, candidate['file'], candidate['line'])
    quote = target['lines'][candidate['line'] - 1]
    if not re.search(r'\baddmm\b', quote):
        return None
    return {'file': candidate['file'], 'line': candidate['line'],
            'sha256': target['sha256'], 'symbol': 'addmm', 'quote': quote,
            'navigation_basis': 'verified_linear_addmm_binding'}


def verified_linear_binding_target(sources, repo, file, ctx, claims):
    if not any(re.search(r'\blinear\b',claim['quote']) for claim in claims):
        return None
    candidate = linear_binding_candidate(sources, repo, file, ctx)
    if not candidate:
        return None
    target = context(sources, repo, candidate['file'], candidate['line'])
    quote = target['lines'][candidate['line'] - 1]
    if not re.search(r'\blinear\b', quote):
        return None
    return {'file':candidate['file'],'line':candidate['line'],
            'sha256':target['sha256'],'symbol':'linear','quote':quote,
            'navigation_basis':'verified_linear_binding'}


def sass_address(line):
    match = re.match(r'\s*/\*([0-9a-f]+)\*/', line)
    return int(match[1], 16) if match else None


def sass_control_instruction(line):
    instruction = line.split('*/', 1)[-1].split(';', 1)[0].upper()
    return bool(re.search(r'\b(?:BRA|BRX|EXIT|RET|JMP|JMX|CALL|JCAL)\b', instruction))


def sass_predicate(line):
    instruction = line.split('*/', 1)[-1].split(';', 1)[0]
    match = re.match(r'\s*@(!?P\d+)\s+', instruction)
    return match[1] if match else None


def sass_sequential_successor(lines, line):
    current = sass_address(lines[line - 1])
    if current is None or sass_control_instruction(lines[line - 1]):
        return None
    for number in range(line + 1, len(lines) + 1):
        address = sass_address(lines[number - 1])
        if address is not None and sass_control_instruction(lines[number - 1]):
            if address <= current:
                return None
            return {'line': number, 'pc': address}
    return None


def load_sass_branch_data(results, evidence, sass_sha):
    if not isinstance(evidence, dict):
        return None
    artifact_name = evidence.get('artifact')
    observation = evidence.get('branch_observation')
    if not isinstance(artifact_name, str) or not isinstance(observation, dict):
        return None
    artifact = Path(artifact_name).resolve()
    if not artifact.is_relative_to(results) or not artifact.is_file():
        raise ValueError('branch-edge artifact outside results or missing')
    artifact_bytes = artifact.read_bytes()
    edge_data = json.loads(artifact_bytes)
    if edge_data.get('sass_sha256') != sass_sha:
        raise ValueError('branch-edge artifact SASS hash mismatch')
    counters = Path(observation.get('path', '')).resolve()
    if not counters.is_relative_to(results) or not counters.is_file():
        raise ValueError('branch counters outside results or missing')
    counter_bytes = counters.read_bytes()
    counter_sha = hashlib.sha256(counter_bytes).hexdigest()
    if counter_sha != observation.get('sha256'):
        raise ValueError('branch counters hash mismatch')
    counter_rows = json.loads(counter_bytes)
    if not isinstance(edge_data.get('edges'), list) or not isinstance(counter_rows, list):
        raise ValueError('invalid branch-edge or counter artifact')
    return {
        'artifact': artifact_name,
        'artifact_sha256': hashlib.sha256(artifact_bytes).hexdigest(),
        'edges': edge_data['edges'],
        'counter_path': observation['path'],
        'counter_sha256': counter_sha,
        'counters': counter_rows,
    }


def sass_branch_at(data, pc, line, predicate):
    if not data:
        return None
    matches = [edge for edge in data['edges']
               if edge.get('pc') == pc and edge.get('line') == line]
    if len(matches) != 1:
        return None
    edge = matches[0]
    if predicate is not None and edge.get('predicate') != predicate:
        return None
    if not sass_optimization_dependent(edge):
        return None
    counter_matches = [row for row in data['counters']
                       if row.get('pc') == pc and row.get('predicate') == edge.get('predicate')
                       and row.get('line') == line]
    if len(counter_matches) != 1:
        raise ValueError('branch counters do not uniquely match SASS edge')
    counter = counter_matches[0]
    if (counter.get('taken') != edge.get('taken')
            or counter.get('fallthrough') != edge.get('fallthrough')):
        raise ValueError('branch counters disagree with SASS edge')
    if (type(counter.get('predicate_on')) is not int
            or type(counter.get('predicate_off')) is not int):
        raise ValueError('branch counters lack predicate counts')
    return edge, counter


def sass_optimization_dependent(edge):
    """Require measured branch variation across optimization conditions.

    Static control flow and aggregate counters alone do not establish that a
    branch affects optimization. The edge artifact must link observations for
    at least two values of an optimization flag or hyperparameter and show
    opposite branch outcomes.
    """
    proof = edge.get('optimization_dependency')
    if not isinstance(proof, dict):
        return False
    if proof.get('dimension') not in ('optimization_flag', 'hyperparameter'):
        return False
    if not isinstance(proof.get('name'), str) or not proof['name'].strip():
        return False
    observations = proof.get('observations')
    if not isinstance(observations, list) or len(observations) < 2:
        return False
    values = set()
    outcomes = set()
    for item in observations:
        if not isinstance(item, dict) or 'value' not in item:
            return False
        if type(item.get('taken_count')) is not int or type(item.get('fallthrough_count')) is not int:
            return False
        if item['taken_count'] < 0 or item['fallthrough_count'] < 0:
            return False
        values.add(json.dumps(item['value'], sort_keys=True))
        if item['taken_count'] and not item['fallthrough_count']:
            outcomes.add('taken')
        elif item['fallthrough_count'] and not item['taken_count']:
            outcomes.add('fallthrough')
    return len(values) >= 2 and len(outcomes) == 2


def branch_observation_record(data, edge, counter, side):
    count = counter['predicate_on' if side == 'taken' else 'predicate_off']
    return {
        'path': data['counter_path'], 'sha256': data['counter_sha256'],
        'pc': edge['pc'], 'predicate': edge['predicate'], 'side': side,
        'thread_count': count, 'observed': count > 0,
        'scope': 'aggregate_kernel_run',
    }


class TraceValidationError(ValueError):
    def __init__(self, message, correction_hints=None):
        super().__init__(message)
        self.correction_hints = correction_hints or []


MAX_TRACE_REPAIR_ATTEMPTS = 3


def validate_step(response, ctx, repo, sources):
    if not isinstance(response,dict):
        raise ValueError('agent must return JSON object')
    claims = response.get('claims',[])
    if not isinstance(claims,list) or len(claims)>30:
        raise ValueError('invalid claims')
    accepted=[]
    for claim in claims:
        if not isinstance(claim,dict):
            raise ValueError('invalid claim')
        line, quote = claim.get('line'),claim.get('quote')
        valid_line = type(line) is int and 1 <= line <= len(ctx['lines'])
        source_line = ctx['lines'][line-1] if valid_line else '<line out of range>'
        visible_line = ctx.get('visible_lines',{}).get(line, '') if valid_line else ''
        if (not valid_line or not isinstance(quote,str) or not quote
                or quote not in source_line or quote not in visible_line):
            hint = ({'line': line, 'exact_visible_quote': visible_line}
                    if valid_line and visible_line else None)
            raise TraceValidationError(
                f'claim lacks exact source evidence: line={line!r}, '
                f'quote={str(quote)[:120]!r}, source={source_line[:180]!r}',
                [hint] if hint else [],
            )
        accepted.append({'line':line,'quote':quote,'claim':str(claim.get('claim',''))[:1000],
                         'evidence_level':'source_quote_only'})
    nxt=response.get('next')
    if nxt is not None:
        if not isinstance(nxt,dict) or not isinstance(nxt.get('file'),str) or type(nxt.get('line')) is not int:
            raise ValueError('invalid next location')
        if ctx.get('sass_file'):
            expected = ctx.get('sass_expected_next')
            if (nxt['file'] != ctx['sass_file'] or not expected
                    or nxt['line'] != expected['line']
                    or not 1 <= nxt['line'] <= len(ctx['lines'])
                    or sass_address(ctx['lines'][nxt['line'] - 1]) != expected['pc']):
                return accepted,None,'unresolved'
            quote=nxt.get('quote')
            if (not accepted or not isinstance(quote,str) or not quote.strip()
                    or quote not in ctx.get('visible_lines',{}).get(nxt['line'],'')
                    or quote not in ctx['lines'][nxt['line']-1]):
                return accepted,None,'unresolved'
            return accepted,{'file':nxt['file'],'line':nxt['line'],'sha256':ctx['sha256'],'quote':quote},response.get('terminal')
        else:
            target=context(sources,repo,nxt['file'],nxt['line'])
        quote=nxt.get('quote')
        symbol=nxt.get('symbol')
        if (not accepted or not isinstance(quote,str) or not quote
                or quote not in target.get('visible_lines',{}).get(nxt['line'],'')
                or quote not in target['lines'][nxt['line']-1]):
            return accepted,None,'unresolved'
        if nxt['file'] != ctx['file']:
            if (not isinstance(symbol,str) or not symbol.isidentifier()
                    or not re.search(r'\b'+re.escape(symbol)+r'\b',quote)
                    or not any(re.search(r'\b'+re.escape(symbol)+r'\b',c['quote']) for c in accepted)):
                return accepted,None,'unresolved'
        elif symbol is not None and (
                not isinstance(symbol,str) or not symbol.isidentifier()
                or not re.search(r'\b'+re.escape(symbol)+r'\b',quote)):
            return accepted,None,'unresolved'
        nxt={'file':nxt['file'],'line':nxt['line'],'sha256':target['sha256'],
             'symbol':symbol,'quote':quote}
    if nxt is None and ctx.get('sass_file') and ctx.get('sass_expected_next'):
        expected=ctx['sass_expected_next']
        quote=ctx['lines'][expected['line']-1]
        return accepted,{
            'file':ctx['sass_file'],'line':expected['line'],'sha256':ctx['sha256'],
            'quote':quote,'navigation_basis':expected['basis'],
        },'binary_candidate'
    terminal=response.get('terminal')
    if terminal == 'source_end' and (not accepted or nxt is not None):
        raise TraceValidationError('source_end requires validated source claims and no next location')
    if nxt is None and terminal not in ('source_end','unresolved','cuda_launch_candidate','binary_candidate'):
        raise ValueError('must specify next or an explicit terminal status')
    return accepted,nxt,terminal


def fork(db, task, branch_key, expression, follow_value=True):
    tid,lid,_,file,line,decisions_json,visited_json,depth=task
    decisions=json.loads(decisions_json)
    alternate=dict(decisions)
    alternate[branch_key]=not follow_value
    child=stable('fork',lid,file,line,alternate)
    db.execute('''INSERT OR IGNORE INTO trace_tasks(id,layer_id,parent_id,file,line,decisions,visited,depth)
                  VALUES(?,?,?,?,?,?,?,?)''', (child,lid,tid,file,line,json.dumps(alternate,sort_keys=True),visited_json,depth))
    decisions[branch_key]=follow_value
    db.execute('UPDATE trace_tasks SET decisions=? WHERE id=?',(json.dumps(decisions,sort_keys=True),tid))
    return child,decisions


async def process_one(db_path,sources,task,model_call,max_depth=40,lease=None):
    (tid,lid,repo,file,line,decisions_json,visited_json,depth,root_sha,
     prior_error,attempts)=task
    stage = 'source'
    try:
        branch=None
        decisions=json.loads(decisions_json)
        visited=json.loads(visited_json)
        prior_evidence={}
        sass_data=None
        selected_observation=None
        with sqlite3.connect(db_path,timeout=30) as model_db:
            assignment=model_db.execute(
                'SELECT model_id FROM trace_model_assignments WHERE layer_id=?',(lid,)
            ).fetchone()
        pinned_model=assignment[0] if assignment else None
        if Path(file).suffix == '.sass':
            artifact = Path(file).resolve()
            results = Path(os.getenv('OE_DATA_ROOT', 'results')).resolve()
            if not artifact.is_relative_to(results):
                raise ValueError('SASS artifact outside results')
            ctx=context(results.parent,results.name,artifact.relative_to(results).as_posix(),
                        line,decisions['sass_sha256'])
            ctx['branches']=[]
            ctx['sass_file']=file
            with sqlite3.connect(db_path,timeout=30) as db:
                prior=db.execute('SELECT evidence FROM trace_results WHERE task_id=?',(tid,)).fetchone()
            if prior:
                saved=json.loads(prior[0])
                if isinstance(saved,dict):
                    prior_evidence=saved
            sass_data=load_sass_branch_data(results,prior_evidence,ctx['sha256'])
            current_pc=sass_address(ctx['lines'][line-1])
            expected_pc=None
            expected_basis=None
            if current_pc == decisions.get('pc'):
                side=decisions['side']
                expected_pc=decisions.get('target')
                expected_basis='recorded_sass_branch_edge'
                branch_match=sass_branch_at(
                    sass_data,current_pc,line,decisions.get('predicate'))
                if sass_data and not branch_match:
                    raise ValueError('selected branch missing from edge artifact')
                if branch_match:
                    edge,counter=branch_match
                    if edge.get(side) != expected_pc:
                        raise ValueError('selected branch target differs from edge artifact')
                    selected_observation=branch_observation_record(sass_data,edge,counter,side)
                    saved_observation=prior_evidence.get('branch_observation')
                    if saved_observation and (
                            saved_observation.get('thread_count') != selected_observation['thread_count']
                            or saved_observation.get('observed') != selected_observation['observed']):
                        raise ValueError('branch observation differs from counters')
                    ctx['sass_branch_edge']=edge
            elif sass_control_instruction(ctx['lines'][line-1]):
                predicate=sass_predicate(ctx['lines'][line-1])
                branch_match=sass_branch_at(sass_data,current_pc,line,predicate)
                if branch_match:
                    edge,counter=branch_match
                    branch_key=f'sass_{current_pc}'
                    if branch_key not in decisions:
                        follow_taken=counter['predicate_on'] > counter['predicate_off']
                        branch=(branch_key,f"SASS {edge['predicate']} branch at PC {current_pc}",follow_taken)
                        decisions[branch_key]=follow_taken
                    side='taken' if decisions[branch_key] else 'fallthrough'
                    expected_pc=edge.get(side)
                    expected_basis='recorded_sass_branch_edge'
                    selected_observation=branch_observation_record(sass_data,edge,counter,side)
                    ctx['sass_branch_edge']=edge
            else:
                successor=sass_sequential_successor(ctx['lines'],line)
                expected_pc=successor['pc'] if successor else None
                if successor:
                    expected_basis='straight_line_to_next_control_transfer'
            if expected_pc is not None:
                target_line=next((number for number,instruction in enumerate(ctx['lines'],1)
                                  if sass_address(instruction)==expected_pc),None)
                if target_line is not None:
                    ctx['sass_expected_next']={'line':target_line,'pc':expected_pc,'basis':expected_basis}
        else:
            ctx=context(sources,repo,file,line,root_sha if depth==0 else None)
        marker=f'{file}:{line}:{json.dumps(decisions,sort_keys=True)}'
        if marker in visited or depth>=max_depth:
            result={'terminal':'unresolved','reason':'cycle_or_depth_limit'}
            state='unresolved'
        else:
            if branch is None:
                branch=choose_branch(ctx,decisions)
            branch_line = int(branch[0].rsplit(':', 1)[1]) if branch and (Path(file).suffix == '.py' or ctx.get('native')) else None
            if branch:
                follow_value=branch[2] if len(branch)>2 else True
                decisions[branch[0]]=follow_value
                if Path(file).suffix == '.py' or ctx.get('native'):
                    with sqlite3.connect(db_path,timeout=30) as case_db:
                        with case_db:
                            if lease and not case_db.execute(
                                    'SELECT 1 FROM trace_leases WHERE task_id=? AND token=? AND expires>unixepoch()',
                                    (tid,lease)).fetchone():
                                return {'task':tid,'state':'lease_lost'}
                            case = (_trace_case(case_db, (tid,lid,repo,file), ctx, branch) if not ctx.get('native')
                                    else _trace_native_case(case_db, (tid,lid,repo,file), ctx, branch))
                            if case is not None:
                                _persist_trace_case(case_db, case, tid, lid, ctx['sha256'], branch_line)
            branch_node = (_branch_node('\n'.join(ctx['lines']), branch_line) if branch_line and not ctx.get('native')
                           else native_branch_at(ctx,branch_line) if branch_line else None)
            payload_branch = None
            if branch_node and not ctx.get('native'):
                last = getattr(branch_node, 'end_lineno', branch_node.lineno)
                payload_branch = {
                    'line': branch_line, 'predicate': branch[1],
                    'true_lines': list(_branch_span(branch_node.body, last)),
                    'false_lines': list(_branch_span(branch_node.orelse, last)) if branch_node.orelse else None,
                }
            elif branch_node and ctx.get('native'):
                payload_branch={'line':branch_line,'predicate':branch_node['expression'],
                    'true_lines':[branch_node['true_start'],branch_node['true_end']],
                    'false_lines':[branch_node['false_start'],branch_node['false_end']] if branch_node['false_start'] else None}
            payload={'framework':repo,'layer_id':lid,'file':file,'line':line,
                     'source':ctx['source'],'branch_assumptions':decisions,
                     'previous_failure':prior_error,'attempt':attempts + 1,
                     'optimization_branch_candidate':payload_branch,
                     'instruction':('Trace the actual next source call/lowering target. '
                                    'Return JSON {"claims":[{"line":N,"quote":"exact source substring","claim":"..."}],'
                                    '"next":{"file":"relative/path","line":N,"symbol":"shared identifier for cross-file links","quote":"exact target substring"} OR null,'
                                    '"terminal":"source_end|unresolved|cuda_launch_candidate|binary_candidate"}. '
                                    'Do not claim executed CUDA or sm75 proof from source alone. '
                                    'Same-file navigation needs only an exact target quote. Cross-file navigation needs a shared quoted symbol. '
                                    'If next cannot be grounded, use unresolved.')}
            if pinned_model:
                payload['_kilo_model']=pinned_model
            if Path(file).suffix == '.sass':
                expected = ctx.get('sass_expected_next')
                if expected:
                    instruction = ctx['lines'][expected['line'] - 1]
                    payload['target_candidate'] = {
                        'file': file, 'line': expected['line'], 'source': instruction,
                    }
                payload['instruction'] += (
                    ' For SASS, inspect the straight-line instructions leading to '
                    'target_candidate or the supplied selected branch edge. Claims '
                    'should address the current block and predicate setup. If a '
                    'verified candidate is supplied, next should quote that exact '
                    'line. The shared identifier rule is for Python only.'
                )
                if ctx.get('sass_branch_edge'):
                    payload['branch_edge']=ctx['sass_branch_edge']
                if selected_observation:
                    payload['branch_observation']=selected_observation
            candidate=(linear_binding_candidate(sources,repo,file,ctx)
                       or linear_addmm_candidate(sources,repo,file,ctx))
            if candidate:
                payload['target_candidate']=candidate
            stage = 'planner'
            plan=await model_call('validator',{'task':payload},'planner')
            focus=plan.get('focus') if isinstance(plan,dict) else None
            if not isinstance(focus,str) or not focus.strip():
                raise ValueError('invalid planner focus')
            payload['focus']=focus[:2000]
            stage = 'worker'
            repair_model = None
            repair_attempts = 0
            supervisor_model = None
            supervisor_feedback_attempts = 0
            claim_supervisor_model = None
            claim_feedback_attempts = 0
            try:
                response=await model_call('worker',payload,'agent_trace_worker')
            except Exception as worker_error:
                stage = 'repair'
                repair_task = {key:value for key,value in payload.items() if key != '_kilo_model'}
                validation_error = worker_error
                response = None
                for repair_attempts in range(1, MAX_TRACE_REPAIR_ATTEMPTS + 1):
                    try:
                        response = await model_call('validator', {
                            'task': repair_task,
                            'worker_response': None,
                            'worker_error': f'{type(validation_error).__name__}: {validation_error}',
                            'validation_error': f'OpenKilo worker request failed: {type(validation_error).__name__}: {validation_error}',
                            'correction_hints': [],
                            'repair_attempt': repair_attempts,
                            'repair_attempt_limit': MAX_TRACE_REPAIR_ATTEMPTS,
                        }, 'agent_trace_repair')
                        claims,nxt,terminal=validate_trace_response(response,ctx,repo,sources,visited)
                        break
                    except (ValueError, TypeError) as repair_error:
                        validation_error = repair_error
                else:
                    raise validation_error
                repair_model = os.getenv('ORCHESTRATOR_MODEL', 'openai/conductor')
            routed_model=None
            if isinstance(response,dict):
                from optimization_explorer.llm import normalize_kilo_model_id
                routed_model=normalize_kilo_model_id(response.pop('_kilo_model',None))
            worker_model=routed_model or pinned_model
            if routed_model:
                with sqlite3.connect(db_path,timeout=30) as model_db:
                    with model_db:
                        model_db.execute(
                            'INSERT OR REPLACE INTO trace_model_assignments VALUES(?,?)',
                            (lid,routed_model),
                        )
                pinned_model=routed_model
            if response is not None and repair_attempts == 0:
                try:
                    claims,nxt,terminal=validate_trace_response(response,ctx,repo,sources,visited)
                except (ValueError, TypeError) as worker_error:
                    stage = 'supervision'
                    repair_task = {key:value for key,value in payload.items() if key != '_kilo_model'}
                    validation_error = worker_error
                    for supervisor_feedback_attempts in range(1, MAX_TRACE_REPAIR_ATTEMPTS + 1):
                        feedback = await model_call('validator', {
                            'task': repair_task,
                            'worker_response': response,
                            'validation_error': f'{type(validation_error).__name__}: {validation_error}',
                            'correction_hints': getattr(validation_error, 'correction_hints', []),
                            'repair_attempt': supervisor_feedback_attempts,
                            'repair_attempt_limit': MAX_TRACE_REPAIR_ATTEMPTS,
                        }, 'agent_trace_supervisor_feedback')
                        supervisor_model = os.getenv('ORCHESTRATOR_MODEL', 'openai/conductor')
                        retry_payload = dict(payload)
                        retry_payload.update({
                            'previous_worker_response': response,
                            'previous_failure': f'{type(validation_error).__name__}: {validation_error}',
                            'correction_hints': getattr(validation_error, 'correction_hints', []),
                            'supervisor_feedback': feedback['correction'][:2000],
                            'supervisor_feedback_attempt': supervisor_feedback_attempts,
                        })
                        stage = 'worker'
                        response = await model_call('worker', retry_payload, 'agent_trace_worker')
                        retry_model = None
                        if isinstance(response, dict):
                            from optimization_explorer.llm import normalize_kilo_model_id
                            retry_model = normalize_kilo_model_id(response.pop('_kilo_model', None))
                        if retry_model:
                            worker_model = retry_model
                            with sqlite3.connect(db_path,timeout=30) as model_db:
                                with model_db:
                                    model_db.execute(
                                        'INSERT OR REPLACE INTO trace_model_assignments VALUES(?,?)',
                                        (lid,retry_model),
                                    )
                            pinned_model = retry_model
                        try:
                            claims,nxt,terminal=validate_trace_response(response,ctx,repo,sources,visited)
                            break
                        except (ValueError, TypeError) as retry_error:
                            validation_error = retry_error
                            stage = 'supervision'
                    else:
                        stage = 'repair'
                        for repair_attempts in range(1, MAX_TRACE_REPAIR_ATTEMPTS + 1):
                            try:
                                response = await model_call('validator', {
                                    'task': repair_task,
                                    'worker_response': response,
                                    'validation_error': f'{type(validation_error).__name__}: {validation_error}',
                                    'correction_hints': getattr(validation_error, 'correction_hints', []),
                                    'repair_attempt': repair_attempts,
                                    'repair_attempt_limit': MAX_TRACE_REPAIR_ATTEMPTS,
                                }, 'agent_trace_repair')
                                claims,nxt,terminal=validate_trace_response(response,ctx,repo,sources,visited)
                                break
                            except (ValueError, TypeError) as repair_error:
                                validation_error = repair_error
                        else:
                            raise validation_error
                        repair_model = os.getenv('ORCHESTRATOR_MODEL', 'openai/conductor')
            if nxt is None:
                binding = (verified_linear_binding_target(sources,repo,file,ctx,claims)
                           or verified_linear_addmm_target(sources,repo,file,ctx,claims))
                if binding:
                    nxt = binding
                    terminal = None
            barrier = next_unvisited_branch(ctx,decisions) if Path(file).suffix == '.py' or ctx.get('native') else None
            if barrier:
                branch_line, _, scope_start, scope_end = barrier
                stays_before_branch = (
                    nxt is not None and nxt['file'] == file
                    and scope_start <= nxt['line'] <= branch_line
                    and nxt['line'] <= scope_end
                )
                if not stays_before_branch:
                    quote = ctx['lines'][branch_line - 1]
                    nxt = {'file':file, 'line':branch_line, 'sha256':ctx['sha256'],
                           'quote':quote, 'navigation_basis':'optimization_branch_barrier'}
                    terminal = None
            if Path(file).suffix == '.py' and nxt is None and terminal == 'unresolved':
                method_target = next_class_method(ctx, visited)
                if method_target:
                    method_line, method_marker, _, _ = method_target
                    nxt = {'file':file,'line':method_line,'sha256':ctx['sha256'],
                           'quote':ctx['lines'][method_line-1],
                           'navigation_basis':'next_class_method'}
                    terminal = None
                    if method_marker not in visited:
                        visited.append(method_marker)
            review_model = None
            navigation_review_attempted = False
            navigation_feedback_attempts = 0
            navigation_validation_error = None
            if nxt is None and terminal == 'unresolved' and claims:
                navigation_review_attempted = True
                stage = 'navigation_review'
                navigation_candidate, navigation_claim = same_file_call_target(ctx)
                if navigation_candidate is None:
                    navigation_candidate = next_source_call(ctx)
                if navigation_candidate is None:
                    navigation_candidate, navigation_claim = imported_base_class_target(sources, repo, ctx)
                if navigation_claim:
                    quote = navigation_claim['quote']
                    visible = ctx['lines'][navigation_claim['line'] - 1].encode('utf-8')[:300].decode('utf-8', 'ignore')
                    if quote not in visible:
                        navigation_candidate, navigation_claim = None, None
                    else:
                        ctx['visible_lines'][navigation_claim['line']] = visible
                if navigation_claim:
                    generated_claims, _, _ = validate_step(
                        {'claims': [navigation_claim], 'next': None, 'terminal': 'unresolved'},
                        ctx, repo, sources,
                    )
                    claims.extend(generated_claims)
                navigation_payload = {
                    'task': {key:value for key,value in payload.items() if key != '_kilo_model'},
                    'navigation_candidate': navigation_candidate,
                    'worker_response': response,
                    'validated_claims': claims,
                    'instruction': (
                        'Review only whether the unresolved source step has a concrete next source location. '
                        'Return JSON {"claims":[],"next":{"file":"relative/path","line":N,"symbol":"shared identifier for cross-file links","quote":"exact visible target substring"},"terminal":"unresolved"}. '
                        'For a same-file transition, choose the next executable call or source-level operation inside the current function when it is visible in task.source; quote that exact line. Do not jump to the callee body. For a cross-file transition, symbol must occur in a validated current claim and the exact target quote. Use next:null only when no source continuation is visible. '
                        'The proposal will be checked against source and optimization branch barriers.'
                    ),
                }
                if pinned_model:
                    navigation_payload['_kilo_model'] = pinned_model
                navigation_feedback_attempts = 0
                navigation_validation_error = None
                reviewed_next = None
                for review_attempt in range(2):
                    navigation = await model_call(
                        'validator', navigation_payload, 'agent_trace_navigation_review'
                    )
                    if isinstance(navigation, dict):
                        from optimization_explorer.llm import normalize_kilo_model_id
                        review_model = normalize_kilo_model_id(navigation.pop('_kilo_model', None))
                    try:
                        if not isinstance(navigation, dict):
                            raise ValueError('invalid navigation review response')
                        _, reviewed_next, _ = validate_step(
                            {'claims': claims, 'next': navigation.get('next'), 'terminal': 'unresolved'},
                            ctx, repo, sources,
                        )
                        navigation_validation_error = None
                        break
                    except (ValueError, TypeError) as review_error:
                        navigation_validation_error = (
                            f'{type(review_error).__name__}: {review_error}'
                        )
                        if review_attempt == 0:
                            navigation_feedback_attempts = 1
                            navigation_payload.update({
                                'previous_review_response': navigation,
                                'validation_error': navigation_validation_error,
                                'correction_instruction': (
                                    'Your previous next location failed deterministic source validation. '
                                    'Return a corrected in-root, source-backed next location, or next:null. '
                                    'Do not repeat the rejected path or invent a target.'
                                ),
                            })
                            continue
                        reviewed_next = None
                if reviewed_next is not None:
                    nxt, terminal = reviewed_next, None
                elif navigation_candidate is not None:
                    _, candidate_next, _ = validate_step(
                        {'claims': claims, 'next': navigation_candidate, 'terminal': 'unresolved'},
                        ctx, repo, sources,
                    )
                    if candidate_next is not None:
                        nxt, terminal = candidate_next, None
                        nxt['navigation_basis'] = navigation_candidate['navigation_basis']
                stage = 'validator'
            validator_payload={'source': '\n'.join(f"{i}: {ctx['lines'][i-1]}" for i in sorted({c['line'] for c in claims})), 'claims':claims}
            if pinned_model:
                validator_payload['_kilo_model']=pinned_model
            stage = 'claim_validator'
            review=await model_call('validator',validator_payload,'agent_trace_validator') if claims else {'approved_indices':[]}
            validator_model=None
            claim_review_history=[]
            for review_round in range(MAX_TRACE_REPAIR_ATTEMPTS + 1):
                if not isinstance(review,dict):
                    raise ValueError('invalid validator response')
                from optimization_explorer.llm import normalize_kilo_model_id
                routed_validator=normalize_kilo_model_id(review.pop('_kilo_model',None))
                if routed_validator:
                    validator_model=routed_validator
                    with sqlite3.connect(db_path,timeout=30) as model_db:
                        with model_db:
                            model_db.execute(
                                'INSERT OR REPLACE INTO trace_model_assignments VALUES(?,?)',
                                (lid,routed_validator),
                            )
                    pinned_model=routed_validator
                approved=review.get('approved_indices')
                if (not isinstance(approved,list)
                        or any(type(i) is not int or not 0<=i<len(claims) for i in approved)):
                    raise ValueError('invalid validator response')
                rejected=review.get('rejected_indices')
                if rejected is None:
                    rejected=[i for i in range(len(claims)) if i not in set(approved)]
                if (not isinstance(rejected,list)
                        or any(type(i) is not int or not 0<=i<len(claims) for i in rejected)
                        or set(approved) & set(rejected)
                        or set(approved) | set(rejected) != set(range(len(claims)))):
                    raise ValueError('invalid validator response rejected_indices')
                reasons=review.get('rejection_reasons',[])
                if (not isinstance(reasons,list)
                        or any(not isinstance(item,dict)
                               or type(item.get('index')) is not int
                               or item['index'] not in rejected
                               or not isinstance(item.get('reason'),str)
                               or not item['reason'].strip() for item in reasons)):
                    raise ValueError('invalid validator response rejection_reasons')
                reason_by_index={item['index']:item['reason'] for item in reasons}
                rejected_claims=[
                    {'index':i,'claim':claims[i],
                     'reason':reason_by_index.get(i,'Independent validator did not approve this claim.')}
                    for i in rejected
                ]
                claim_review_history.append({
                    'approved_indices':sorted(set(approved)),
                    'rejected_claims':rejected_claims,
                })
                accepted=[claims[i] for i in sorted(set(approved))]
                if not rejected or review_round == MAX_TRACE_REPAIR_ATTEMPTS:
                    break
                stage='claim_supervision'
                feedback=await model_call('validator',{
                    'task':{key:value for key,value in payload.items() if key != '_kilo_model'},
                    'worker_response':response,
                    'accepted_claims':accepted,
                    'rejected_claims':rejected_claims,
                    'independent_validator_verdict':{
                        'approved_indices':sorted(set(approved)),
                        'rejected_indices':sorted(set(rejected)),
                    },
                    'repair_attempt':review_round + 1,
                    'repair_attempt_limit':MAX_TRACE_REPAIR_ATTEMPTS,
                },'agent_trace_claim_feedback')
                claim_supervisor_model=os.getenv('ORCHESTRATOR_MODEL','openai/conductor')
                claim_feedback_attempts=review_round + 1
                retry_payload=dict(payload)
                retry_payload.update({
                    'previous_worker_response':response,
                    'accepted_claims_to_preserve':accepted,
                    'rejected_claims':rejected_claims,
                    'supervisor_feedback':feedback['correction'][:2000],
                    'supervisor_feedback_attempt':claim_feedback_attempts,
                })
                stage='worker'
                response=await model_call('worker',retry_payload,'agent_trace_worker')
                if isinstance(response,dict):
                    routed_worker=normalize_kilo_model_id(response.pop('_kilo_model',None))
                    if routed_worker:
                        worker_model=routed_worker
                        pinned_model=routed_worker
                        with sqlite3.connect(db_path,timeout=30) as model_db:
                            with model_db:
                                model_db.execute(
                                    'INSERT OR REPLACE INTO trace_model_assignments VALUES(?,?)',
                                    (lid,routed_worker),
                                )
                claims,nxt,terminal=validate_trace_response(response,ctx,repo,sources,visited)
                if barrier:
                    branch_line, _, scope_start, scope_end=barrier
                    stays_before_branch=(
                        nxt is not None and nxt['file']==file
                        and scope_start<=nxt['line']<=branch_line
                        and nxt['line']<=scope_end
                    )
                    if not stays_before_branch:
                        nxt={'file':file,'line':branch_line,'sha256':ctx['sha256'],
                             'quote':ctx['lines'][branch_line-1],
                             'navigation_basis':'optimization_branch_barrier'}
                        terminal=None
                validator_payload={
                    'source':'\n'.join(f"{i}: {ctx['lines'][i-1]}" for i in sorted({c['line'] for c in claims})),
                    'claims':claims,
                }
                if pinned_model:
                    validator_payload['_kilo_model']=pinned_model
                stage='claim_validator'
                review=await model_call('validator',validator_payload,'agent_trace_validator') if claims else {'approved_indices':[]}
            stage='claim_validator'
            result={'claims':accepted,'terminal':terminal,'next':nxt,'branch_assumptions':decisions,
                    'worker_model':worker_model,
                    'validator_model':validator_model,
                    'navigation_review_attempted':navigation_review_attempted,
                    'navigation_review_model':review_model,
                    'navigation_feedback_attempts':navigation_feedback_attempts,
                    'navigation_validation_error':navigation_validation_error,
                    'repair_model':repair_model,
                    'repair_attempts':repair_attempts,
                    'supervisor_model':supervisor_model,
                    'supervisor_feedback_attempts':supervisor_feedback_attempts,
                    'claim_supervisor_model':claim_supervisor_model,
                    'claim_feedback_attempts':claim_feedback_attempts,
                    'claim_review_history':claim_review_history,
                    'runtime_verified':False,'sm75_verified':False}
            if selected_observation:
                result['branch_observation']=selected_observation
            state='pending' if nxt else ('completed' if terminal=='source_end' else 'unresolved')
            visited.append(marker)
        stage = 'persistence'
        with sqlite3.connect(db_path,timeout=30) as db:
            with db:
                if lease and not db.execute(
                    'DELETE FROM trace_leases WHERE task_id=? AND token=? AND expires>unixepoch()',
                    (tid,lease),
                ).rowcount:
                    return {'task':tid,'state':'lease_lost'}
                if branch:
                    follow_value=branch[2] if len(branch)>2 else True
                    child,_=fork(db,task[:8],
                                 branch[0],branch[1],follow_value)
                    if Path(file).suffix == '.sass' and prior_evidence:
                        child_row=db.execute('SELECT decisions FROM trace_tasks WHERE id=?',(child,)).fetchone()
                        db.execute('''INSERT OR IGNORE INTO trace_results
                                      (task_id,layer_id,conditions,outcome,evidence)
                                      VALUES(?,?,?,'pending',?)''',
                                   (child,lid,child_row[0],json.dumps(prior_evidence,ensure_ascii=False)))
                db.execute('INSERT OR REPLACE INTO trace_steps VALUES(?,?,?,?,?,?)',
                           (tid,depth,file,line,ctx['sha256'],json.dumps(result,ensure_ascii=False)))
                if state=='pending':
                    db.execute('''UPDATE trace_tasks SET file=?,line=?,decisions=?,visited=?,depth=depth+1,state='pending',
                               error=NULL,attempts=0,available_at=0,retry_priority=0 WHERE id=?''',
                               (nxt['file'],nxt['line'],json.dumps(decisions,sort_keys=True),json.dumps(visited),tid))
                else:
                    db.execute('UPDATE trace_tasks SET state=?,result=?,error=NULL,retry_priority=0 WHERE id=?',
                               (state,json.dumps(result,ensure_ascii=False),tid))
                    evidence=result
                    if Path(file).suffix == '.sass':
                        prior=db.execute('SELECT evidence FROM trace_results WHERE task_id=?',(tid,)).fetchone()
                        evidence=json.loads(prior[0]) if prior else {}
                        if not isinstance(evidence,dict):
                            evidence={}
                        observations=[]
                        for (payload,) in db.execute('SELECT payload FROM trace_steps WHERE task_id=? ORDER BY seq',(tid,)):
                            step=json.loads(payload)
                            if step.get('branch_observation'):
                                observations.append(step['branch_observation'])
                        if observations:
                            evidence['branch_observations']=observations
                        evidence['analysis']=result
                    db.execute('''INSERT OR REPLACE INTO trace_results(task_id,layer_id,conditions,outcome,evidence)
                                  VALUES(?,?,?,?,?)''',
                               (tid,lid,json.dumps(decisions,sort_keys=True),state,json.dumps(evidence,ensure_ascii=False)))
        return {'task':tid,'state':state,'terminal':result.get('terminal')}
    except Exception as exc:
        failed_model=getattr(exc,'kilo_model_id',None)
        if failed_model:
            from optimization_explorer.llm import normalize_kilo_model_id
            failed_model=normalize_kilo_model_id(failed_model)
            if failed_model:
                with sqlite3.connect(db_path,timeout=30) as model_db:
                    with model_db:
                        model_db.execute(
                            'INSERT OR REPLACE INTO trace_model_assignments VALUES(?,?)',
                            (lid,failed_model),
                        )
        error = f'{stage}: {type(exc).__name__}: {exc}'[:500]
        from optimization_explorer.llm import OpenKiloQuotaWaitTimeout
        quota_wait_expired = isinstance(exc, OpenKiloQuotaWaitTimeout)
        unrecoverable = attempts > 0 and stage in {'worker', 'repair'} and isinstance(exc, ValueError)
        with sqlite3.connect(db_path,timeout=30) as db:
            with db:
                if lease and not db.execute(
                    'DELETE FROM trace_leases WHERE task_id=? AND token=? AND expires>unixepoch()',
                    (tid,lease),
                ).rowcount:
                    return {'task':tid,'state':'lease_lost'}
                if quota_wait_expired:
                    delay = int(os.getenv('LLM_QUOTA_WAIT_SECONDS', '1800'))
                    db.execute('INSERT OR REPLACE INTO trace_task_errors VALUES(?,?,?,?,CURRENT_TIMESTAMP)',
                               (tid, attempts, stage, error))
                    db.execute("UPDATE trace_tasks SET state='pending',error=?,available_at=unixepoch()+? WHERE id=?",
                               (error, delay, tid))
                    return {'task':tid,'state':'pending','error':error[:300],
                            'retry_in_seconds':delay,'attempt':attempts,'quota_wait_expired':True}
                if unrecoverable:
                    db.execute('INSERT OR REPLACE INTO trace_task_errors VALUES(?,?,?,?,CURRENT_TIMESTAMP)', (tid, attempts + 1, stage, error))
                    evidence = {'terminal':'unresolved','reason':error}
                    db.execute('UPDATE trace_tasks SET state=?,error=?,result=?,retry_priority=0 WHERE id=?', ('unresolved',error,json.dumps(evidence,ensure_ascii=False),tid))
                    db.execute("INSERT OR REPLACE INTO trace_results(task_id,layer_id,conditions,outcome,evidence) VALUES(?,?,?,'unresolved',?)", (tid,lid,decisions_json,json.dumps(evidence,ensure_ascii=False)))
                    return {'task':tid,'state':'unresolved','error':error[:300],'retry_in_seconds':0,'attempt':attempts + 1}
                attempts = attempts + 1
                final = attempts >= getattr(process_one, 'max_attempts', 5)
                delay = min(60, 2 ** min(attempts, 6))
                db.execute('INSERT OR REPLACE INTO trace_task_errors VALUES(?,?,?,?,CURRENT_TIMESTAMP)',
                           (tid, attempts, stage, error))
                db.execute('UPDATE trace_tasks SET state=?,error=?,attempts=?,available_at=unixepoch()+?,retry_priority=CASE WHEN ? THEN retry_priority ELSE 0 END WHERE id=?',
                           ('failed' if final else 'pending', error, attempts,
                            0 if final else delay, int(not final), tid))
        return {'task':tid,'state':'failed' if final else 'pending','error':error[:300],
                'retry_in_seconds':0 if final else delay,'attempt':attempts}


def renew_lease(db_path,task_id,token):
    with sqlite3.connect(db_path,timeout=30) as db:
        return bool(db.execute(
            'UPDATE trace_leases SET expires=unixepoch()+600 '
            'WHERE task_id=? AND token=? AND expires>unixepoch()',
            (task_id,token),
        ).rowcount)


async def keep_lease(db_path,task_id,token):
    while True:
        await asyncio.sleep(60)
        if not await asyncio.to_thread(renew_lease,db_path,task_id,token):
            return


RETRYABLE_UNRESOLVED_VALIDATION_ERRORS = (
    'worker: ValueError: claim lacks exact source evidence:%',
    'worker: ValueError: invalid next location%',
    'worker: ValueError: line out of bounds%',
    'worker: ValueError: trace validator response has invalid approved_indices%',
    'repair: ValueError: trace repair response claims must be a list%',
    'repair: ValueError: claim lacks exact source evidence:%',
    'repair: ValueError: invalid next location%',
    'repair: TraceValidationError: claim lacks exact source evidence:%',
)


def retry_unresolved_validation_errors(db_path):
    'Back up and requeue unresolved tasks stopped by deterministic response validation.'
    path = Path(db_path)
    with sqlite3.connect(path, timeout=30) as db:
        setup(db)
        where = ' OR '.join('error LIKE ?' for _ in RETRYABLE_UNRESOLVED_VALIDATION_ERRORS)
        params = RETRYABLE_UNRESOLVED_VALIDATION_ERRORS
        rows = db.execute(
            f"SELECT id FROM trace_tasks WHERE state='unresolved' AND ({where})",
            params,
        ).fetchall()
        if not rows:
            return {'requeued': 0, 'backup': None}
        ids = [row[0] for row in rows]
        marks = ','.join('?' for _ in ids)
        live = db.execute(
            f"SELECT COUNT(*) FROM trace_leases WHERE expires>unixepoch() AND task_id IN ({marks})",
            ids,
        ).fetchone()[0]
        if live:
            raise RuntimeError(f'{live} unresolved validation tasks still have live leases')
        backup = path.with_name(
            f'{path.name}.before-validation-retry.{uuid.uuid4().hex[:8]}.bak'
        )
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            changed = db.execute(
                f"UPDATE trace_tasks SET state='pending',available_at=0,error=NULL,result=NULL,retry_priority=1 "
                f"WHERE state='unresolved' AND id IN ({marks})",
                ids,
            ).rowcount
    return {'requeued': changed, 'backup': str(backup)}


def retry_unresolved_navigation_candidates(db_path, sources):
    'Back up and requeue unresolved tasks with a locally verifiable source continuation.'
    path = Path(db_path)
    sources = Path(sources)
    candidates = []
    with sqlite3.connect(path, timeout=30) as db:
        setup(db)
        rows = db.execute('''SELECT t.id,l.framework,l.sha256,t.file,t.line,t.visited,
                                   t.decisions,t.depth,s.payload
                            FROM trace_tasks t JOIN trace_layers l ON l.id=t.layer_id
                            JOIN trace_steps s ON s.task_id=t.id
                            WHERE t.state='unresolved'
                            AND s.seq=(SELECT MAX(s2.seq) FROM trace_steps s2 WHERE s2.task_id=t.id)
                            ORDER BY t.id''').fetchall()
        for tid,repo,root_sha,file,line,visited_json,decisions_json,depth,payload_json in rows:
            try:
                ctx = context(sources,repo,file,line,root_sha if depth == 0 else None)
                visited = json.loads(visited_json)
                decisions = json.loads(decisions_json)
                payload = json.loads(payload_json)
                target, generated_claim = same_file_call_target(ctx)
                if target is None:
                    target = next_source_call(ctx)
                if target is None:
                    target, generated_claim = imported_base_class_target(sources,repo,ctx)
                    if generated_claim:
                        claim_line = generated_claim['line']
                        ctx['visible_lines'][claim_line] = (
                            ctx['lines'][claim_line - 1].encode('utf-8')[:300]
                            .decode('utf-8','ignore')
                        )
                if target is None:
                    method = next_class_method(ctx,visited)
                    if method:
                        method_line = method[0]
                        target = {'file':file,'line':method_line,
                                  'quote':ctx['lines'][method_line - 1],
                                  'sha256':ctx['sha256'],
                                  'navigation_basis':'next_class_method'}
                if target is None:
                    continue
                claims = payload.get('claims',[])
                if generated_claim:
                    claims = [*claims,generated_claim]
                elif not claims and target['file'] == file:
                    target_line = target['line']
                    visible = ctx['lines'][target_line - 1].encode('utf-8')[:300].decode('utf-8','ignore')
                    target['quote'] = visible
                    ctx['visible_lines'][target_line] = visible
                    claims = [{'line':target_line,'quote':visible,
                               'claim':'The source contains this source-level continuation.'}]
                _,validated_target,_ = validate_step(
                    {'claims':claims,'next':target,'terminal':'unresolved'},
                    ctx,repo,sources,
                )
                if validated_target is None:
                    continue
                marker = f'{file}:{line}:{json.dumps(decisions,sort_keys=True)}'
                visited = [item for item in visited if item != marker]
                candidates.append((tid,depth + 1,json.dumps(visited)))
            except (OSError,TypeError,ValueError,json.JSONDecodeError):
                continue
        if not candidates:
            return {'requeued':0,'backup':None}
        ids = [row[0] for row in candidates]
        marks = ','.join('?' for _ in ids)
        live = db.execute(
            f"SELECT COUNT(*) FROM trace_leases WHERE expires>unixepoch() AND task_id IN ({marks})",
            ids,
        ).fetchone()[0]
        if live:
            raise RuntimeError(f'{live} unresolved navigation tasks still have live leases')
        backup = path.with_name(f'{path.name}.before-navigation-retry.{uuid.uuid4().hex[:8]}.bak')
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            for tid,depth,visited in candidates:
                db.execute('''UPDATE trace_tasks SET state='pending',depth=?,visited=?,
                              available_at=0,error=NULL,result=NULL,retry_priority=1
                              WHERE id=? AND state='unresolved' ''',(depth,visited,tid))
    return {'requeued':len(candidates),'backup':str(backup)}


def recover_expired_tasks(db_path):
    with sqlite3.connect(db_path,timeout=30) as db:
        with db:
            recovered=db.execute(
                "UPDATE trace_tasks SET state='pending' "
                "WHERE state='running' AND id IN "
                "(SELECT task_id FROM trace_leases WHERE expires<=unixepoch())"
            ).rowcount
            db.execute("DELETE FROM trace_leases WHERE expires<=unixepoch()")
    return recovered


async def run(db_path,sources,limit,concurrency,model_call,max_attempts=5):
    """DFS per layer with durable bounded retries for failed agent steps."""
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    process_one.max_attempts = max_attempts
    with sqlite3.connect(db_path,timeout=30) as db:
        setup(db)
    with sqlite3.connect(db_path,timeout=30) as db:
        with db:
            db.execute('INSERT OR IGNORE INTO trace_scheduler(layer_id,last_selected) '
                       'SELECT id,0 FROM trace_layers')
    results=[]
    claimed=0
    claim_lock=asyncio.Lock()
    recover_expired_tasks(db_path)
    async def work():
        nonlocal claimed
        while True:
            async with claim_lock:
                if claimed>=limit:
                    return
                claimed+=1
                with sqlite3.connect(db_path,timeout=30) as db:
                    with db:
                        db.create_function('eligible_source',1,eligible_source)
                        db.execute('BEGIN IMMEDIATE')
                        row=db.execute('''SELECT t.id,t.layer_id,l.framework,t.file,t.line,t.decisions,t.visited,t.depth,l.sha256,t.error,t.attempts
                                          FROM trace_tasks t JOIN trace_layers l ON l.id=t.layer_id
                                          LEFT JOIN trace_scheduler s ON s.layer_id=t.layer_id
                                          WHERE t.state='pending' AND t.available_at<=unixepoch()
                                            AND t.attempts<? AND eligible_source(l.source_file)
                                            AND l.framework IN ('keras','tensorflow','pytorch')
                                            AND l.state NOT GLOB 'duplicate_of:*'
                                          ORDER BY t.retry_priority DESC,
                                                   CASE WHEN t.attempts>0 THEN 0 ELSE 1 END,
                                                   t.attempts DESC,COALESCE(s.last_selected,0),
                                                   t.depth DESC,t.created_at,t.id LIMIT 1''',(max_attempts,)).fetchone()
                        if row:
                            db.execute("UPDATE trace_tasks SET state='running' WHERE id=?",(row[0],))
                            db.execute('''INSERT INTO trace_scheduler(layer_id,last_selected) VALUES(?,1)
                                          ON CONFLICT(layer_id) DO UPDATE SET last_selected=last_selected+1''',(row[1],))
                            lease=uuid.uuid4().hex
                            db.execute("INSERT INTO trace_leases VALUES(?,?,unixepoch()+600)",(row[0],lease))

            if row is None:
                return
            heartbeat=asyncio.create_task(keep_lease(db_path,row[0],lease))
            try:
                result=await process_one(db_path,sources,row,model_call,lease=lease)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat,return_exceptions=True)
            results.append(result)
    await asyncio.gather(*(work() for _ in range(concurrency)))
    return results[:limit]


def reset(db_path, backup=True):
    """Delete ONLY obsolete agent queues/findings and trace-derived research.

    Preserve atlas, SM75 imported evidence, original optimization cases and other
    user data. Full DB deletion would destroy expensive existing research.
    """
    path=Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as db:
        if backup:
            dest=path.with_name(path.name+'.before-trace-reset.bak')
            with sqlite3.connect(dest) as out:
                db.backup(out)
        with db:
            for table in ('layer_agent_jobs','layer_agent_findings','trace_tasks','trace_steps',
                          'trace_results','trace_scheduler','trace_task_errors',
                          'trace_model_assignments','trace_layers'):
                db.execute(f'DROP TABLE IF EXISTS {table}')
            setup(db)


def invalid_branch_decisions(sources, framework, decisions):
    invalid = []
    for key in decisions:
        if key.startswith('sass_'):
            invalid.append(key)
            continue
        try:
            relative, raw_line = key.rsplit(':', 1)
            line = int(raw_line)
            path = safe_source(Path(sources), framework, relative)
            candidates = branch_candidates(path.read_text(encoding='utf-8'), line)
        except (OSError, UnicodeError, ValueError):
            invalid.append(key)
            continue
        if not any(candidate_line == line for candidate_line, _ in candidates):
            invalid.append(key)
    return invalid


def prune_ineligible_branch_tasks(db_path, sources):
    """Back up SQLite and stop pending tasks forked on non-optimization conditions."""
    path = Path(db_path)
    with sqlite3.connect(path, timeout=30) as db:
        setup(db)
        lock_started = not db.in_transaction
        if lock_started:
            db.execute('BEGIN IMMEDIATE')
        live = db.execute(
            'SELECT COUNT(*) FROM trace_leases WHERE expires>unixepoch()'
        ).fetchone()[0]
        if live:
            raise RuntimeError(f'{live} trace tasks still have live leases')
        invalid = []
        for task_id, framework, state, decisions_json in db.execute(
                '''SELECT t.id,l.framework,t.state,t.decisions
                   FROM trace_tasks t JOIN trace_layers l ON l.id=t.layer_id
                   WHERE t.state='pending' AND t.decisions!='{}' '''):
            stale = invalid_branch_decisions(sources, framework, json.loads(decisions_json))
            if stale:
                invalid.append((task_id, stale))
        if not invalid:
            if lock_started:
                db.commit()
            return {'marked_unresolved': 0, 'backup': None}
        backup = path.with_name(path.name + '.before-ineligible-branch-prune.bak')
        if backup.exists():
            backup = path.with_name(
                path.name + '.before-ineligible-branch-prune.' + uuid.uuid4().hex[:8] + '.bak')
        with sqlite3.connect(path, timeout=30) as source_db, sqlite3.connect(backup) as out:
            source_db.backup(out)
        with db:
            for task_id, stale in invalid:
                db.execute(
                    "UPDATE trace_tasks SET state='unresolved',error=? WHERE id=? AND state='pending'",
                    ('suppressed ineligible optimization branch: ' + ', '.join(stale), task_id),
                )
    return {'marked_unresolved': len(invalid), 'backup': str(backup)}


def prune_invalid_roots(db_path, sources):
    """Back up the database, then remove roots excluded by current source rules."""
    path=Path(db_path)
    invalid=[]
    with sqlite3.connect(path,timeout=30) as db:
        setup(db)
        for layer_id,framework,source_file in db.execute(
                'SELECT id,framework,source_file FROM trace_layers'):
            if framework not in FRAMEWORKS or not eligible_source(source_file):
                invalid.append(layer_id)
                continue
            try:
                safe_source(Path(sources),framework,source_file)
            except (OSError,ValueError):
                invalid.append(layer_id)
        if not invalid:
            return {'removed_roots':0,'backup':None}
        placeholders=','.join('?' for _ in invalid)
        live=db.execute(
            f"SELECT COUNT(*) FROM trace_leases x JOIN trace_tasks t ON t.id=x.task_id WHERE t.layer_id IN ({placeholders}) AND x.expires>unixepoch()",
            invalid,
        ).fetchone()[0]
        if live:
            raise RuntimeError(f'{live} invalid-root tasks still have live leases')
        backup=path.with_name(path.name+'.before-invalid-root-prune.bak')
        if backup.exists():
            backup=path.with_name(
                path.name+'.before-invalid-root-prune.'+uuid.uuid4().hex[:8]+'.bak')
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            task_ids=[row[0] for row in db.execute(
                f'SELECT id FROM trace_tasks WHERE layer_id IN ({placeholders})',invalid)]
            if task_ids:
                task_marks=','.join('?' for _ in task_ids)
                db.execute(f'DELETE FROM trace_leases WHERE task_id IN ({task_marks})',task_ids)
                db.execute(f'DELETE FROM trace_results WHERE task_id IN ({task_marks})',task_ids)
                db.execute(f'DELETE FROM trace_steps WHERE task_id IN ({task_marks})',task_ids)
                db.execute(f'DELETE FROM trace_tasks WHERE layer_id IN ({placeholders})',invalid)
            db.execute(f'DELETE FROM trace_model_assignments WHERE layer_id IN ({placeholders})',invalid)
            db.execute(f'DELETE FROM trace_layers WHERE id IN ({placeholders})',invalid)
    return {'removed_roots':len(invalid),'backup':str(backup)}


def rebuild_layer_queue(db_path, sources):
    """Back up DB and replace only unstarted trace inventory/tasks."""
    path = Path(db_path)
    with sqlite3.connect(path, timeout=30) as db:
        setup(db)
        active = db.execute("SELECT COUNT(*) FROM trace_tasks WHERE state!='pending'").fetchone()[0]
        if active:
            raise RuntimeError(f'{active} trace tasks already started; refusing to discard research')
        backup = path.with_name(path.name + '.before-0007-rebuild.bak')
        with sqlite3.connect(backup) as out:
            db.backup(out)
        with db:
            for table in ('trace_results','trace_steps','trace_task_errors','trace_scheduler',
                          'trace_model_assignments','trace_tasks','trace_layers'):
                db.execute(f'DELETE FROM {table}')
            count = enqueue_layers(db, sources)
    return {'new_layers':count, 'backup':str(backup)}

def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logging.getLogger('optimization_explorer.llm').setLevel(logging.INFO)
    p=argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.getenv('OE_DATA_ROOT', 'results'))
    p.add_argument('--db',default=str(data_root / 'multilayer-live.sqlite'))
    p.add_argument('--sources',default='sources')
    p.add_argument('--rebuild-layer-queue',action='store_true',help='Backup and replace pending trace roots only')
    p.add_argument('--prune-invalid-roots',action='store_true',help='Back up DB and remove roots excluded by current source rules')
    p.add_argument('--prune-ineligible-branches',action='store_true',help='Back up DB and mark pending tasks forked on non-optimization conditions unresolved')
    p.add_argument('--reset-old-queue',action='store_true',help='Backup DB, remove obsolete symbol queue and previous trace tasks')
    p.add_argument('--discover-layers',action='store_true',help='Scan framework source trees for newly added public layers')
    p.add_argument('--seed-branch-cases',action='store_true',help='Back up the DB and persist source-only optimization branch candidates')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--watch',action='store_true',help='keep running and recover pending trace work')
    p.add_argument('--poll-seconds',type=int,default=10)
    p.add_argument('--max-attempts',type=int,default=5)
    p.add_argument('--retry-failed',action='store_true',help='reset failed trace steps for another bounded retry cycle')
    p.add_argument('--retry-unresolved-validation',action='store_true',help='back up and retry unresolved tasks with deterministic response-validation errors')
    p.add_argument('--retry-unresolved-navigation',action='store_true',help='back up and retry unresolved tasks with locally verified source continuations')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args(argv)
    if args.limit<1 or not 1<=args.concurrency<=4 or args.poll_seconds<1 or args.max_attempts<1:
        p.error('limit, poll interval, and max attempts must be positive; concurrency must be 1..4')
    db_path=Path(args.db)
    configured_root = os.getenv("OE_DATA_ROOT")
    if configured_root and not db_path.resolve().is_relative_to(Path(configured_root).resolve()):
        p.error("--db must be under OE_DATA_ROOT")
    if db_path.is_file():
        recover_expired_tasks(db_path)
    if args.reset_old_queue:
        reset(db_path)
    if args.retry_failed and db_path.is_file():
        with sqlite3.connect(db_path,timeout=30) as db:
            setup(db)
            with db:
                db.execute("UPDATE trace_tasks SET state='pending',attempts=0,available_at=0 WHERE state='failed'")
    if args.retry_unresolved_validation:
        print(json.dumps(retry_unresolved_validation_errors(db_path),indent=2))
        if not args.execute:
            return
    if args.retry_unresolved_navigation:
        print(json.dumps(retry_unresolved_navigation_candidates(db_path,Path(args.sources)),indent=2))
        if not args.execute:
            return
    if args.prune_invalid_roots:
        print(json.dumps(prune_invalid_roots(db_path,Path(args.sources)),indent=2))
    if args.prune_ineligible_branches:
        print(json.dumps(prune_ineligible_branch_tasks(db_path,Path(args.sources)),indent=2))
    if args.rebuild_layer_queue:
        print(json.dumps(rebuild_layer_queue(db_path,Path(args.sources)),indent=2))
        return
    deduplication=deduplicate_layer_inventory(db_path)
    case_deduplication=deduplicate_trace_cases(db_path)
    with sqlite3.connect(db_path,timeout=30) as db:
        with db:
            new=ensure_layer_inventory(db,Path(args.sources),args.discover_layers)
        if args.seed_branch_cases:
            print(json.dumps({'branch_case_seed':seed_branch_cases(db,Path(args.sources))},indent=2))
        if deduplication['aliases']:
            print(json.dumps({'layer_deduplication':deduplication},indent=2))
        if case_deduplication['aliases']:
            print(json.dumps({'case_deduplication':case_deduplication},indent=2))
        stats=dict(db.execute('SELECT state,COUNT(*) FROM trace_tasks GROUP BY state'))
        counts=dict(db.execute('SELECT framework,COUNT(*) FROM trace_layers GROUP BY framework'))
    if not args.execute:
        print(json.dumps({'new_layers':new,'layers_by_framework':counts,'tasks':stats,
                          'warning':'Static layer discovery; uncovered functional APIs are not layers'},indent=2));return
    for name, value in {
        'ORCHESTRATOR_BASE_URL': 'http://127.0.0.1:18327/v1',
        'ORCHESTRATOR_MODEL': 'openai/conductor',
        'ORCHESTRATOR_API_KEY': 'local-server',
        'OPENKILO_BASE_URL': 'https://ai-gateway.kilo.ai/api/v1',
        'OPENKILO_MODEL': 'openai/kilo-auto/free',
    }.items():
        os.environ.setdefault(name, value)
    from optimization_explorer.llm import call_model
    async def run_until_stopped():
        while True:
            out=await run(db_path,Path(args.sources),args.limit,args.concurrency,
                          call_model,max_attempts=args.max_attempts)
            if out:
                print(json.dumps({'processed':out},ensure_ascii=False,indent=2),flush=True)
            if not args.watch:
                return
            if not out:
                await asyncio.sleep(args.poll_seconds)
    try:
        asyncio.run(run_until_stopped())
    except KeyboardInterrupt:
        return


if __name__=='__main__':
    main()
