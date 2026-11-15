#!/usr/bin/env python3
"""Import existing SM75 analysis artifacts without inventing provenance edges."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

BINARY_DIRECTORIES = {
    'capture_cubins': 'sources/server-kernel-evidence/artifacts/pilot-linear-20261114/capture/cubins',
}

ARTIFACTS = {
    'kernel_index': 'sources/decompiled/sm75/manifests/sm75-kernels.jsonl',
    'selective_scan': 'selective-scan-sm75.jsonl',
    'execution_graph': 'sources/decompiled/sm75/manifests/sm75-execution-graph-v2.json',
    'launch_sites': 'sources/server-kernel-evidence/sm75-source-map/binary-launch-sites-v5.json',
    'launch_handles': 'sources/server-kernel-evidence/sm75-source-map/binary-launch-handles.json',
    'bulk_summary': 'sources/decompiled/sm75/manifests/sm75-bulk-index-summary.tsv',
    'binary_inventory': 'binary-inventory.tsv',
    'fatbin_inventory': 'cuda-fatbin-inventory.tsv',
    'architectures': 'cuda-architectures.tsv',
    'sm75_libraries': 'sources/decompiled/sm75/manifests/sm75-libraries.txt',
    'nvidia_libraries': 'nvidia-libraries.txt',
    'project_binaries': 'project-binaries.txt',
    'demangled_libtorch': 'sources/server-kernel-evidence/sm75-demangled/_C_stable_libtorch.abi3.so.functions.txt.demangled.txt',
    'demangled_moe_libtorch': 'sources/server-kernel-evidence/sm75-demangled/_moe_C_stable_libtorch.abi3.so.functions.txt.demangled.txt',
    'demangled_cublaslt': 'sources/server-kernel-evidence/sm75-demangled/libcublasLt.so.13.functions.txt.demangled.txt',
    'demangled_ggml_cuda': 'sources/server-kernel-evidence/sm75-demangled/libggml-cuda.so.0.27.0.functions.txt.demangled.txt',
    'linear_manifest': 'sources/server-kernel-evidence/artifacts/pilot-linear-20261114-retry/experiment.json',
    'linear_trace': 'sources/server-kernel-evidence/artifacts/pilot-linear-20261114-retry/execution_graph.json',
    'linear_source_provenance': 'app/results/linear-sm75.8hW8WW/source-provenance.json',
    'linear_cupti_launches': 'app/results/linear-cubin.H6MM4S/same-run.b2gV0M/launch.log',
    'linear_sass': 'results/linear-cubin.H6MM4S/same-run.b2gV0M/linear.sass',
    'linear_branch_edges': 'app/results/linear-cubin.H6MM4S/same-run.b2gV0M/device-branch-edges.json',
    'linear_branch_review': 'app/results/linear-cubin.H6MM4S/same-run.b2gV0M/device-branch-review.json',
    'linear_branch_counters': 'app/results/linear-ncu.ffe8PD/branch-counters.json',
    'linear_cubin': 'results/linear-cubin.H6MM4S/same-run.b2gV0M/module-28.cubin',
    'linear_binary_hashes': 'app/results/linear-cubin.H6MM4S/binary-evidence.sha256',
    'linear_cublas_symbols': 'app/results/linear-sm75.8hW8WW/libcublas.so.12.9.2.10.symbols',
    'linear_cublaslt_symbols': 'app/results/linear-sm75.8hW8WW/libcublasLt.so.12.9.2.10.symbols',
    'so_ggml_cuda': 'sources/server-kernel-evidence/external/build/llama.cpp/build/bin/libggml-cuda.so.0.27.0',
    'so_cublaslt': 'sources/server-kernel-evidence/external/venv-vllm/lib/python3.10/site-packages/nvidia/cu13/lib/libcublasLt.so.13',
    'so_libtorch': 'sources/server-kernel-evidence/external/venv-vllm/lib/python3.10/site-packages/vllm/_C_stable_libtorch.abi3.so',
    'so_moe_libtorch': 'sources/server-kernel-evidence/external/venv-vllm/lib/python3.10/site-packages/vllm/_moe_C_stable_libtorch.abi3.so',
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_records(path: Path, kind: str, sha256: str | None = None):
    if path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{number}: {exc}') from exc
                yield str(number), obj
    elif path.suffix == '.json':
        with path.open('r', encoding='utf-8') as handle:
            obj = json.load(handle)
        yield 'document', obj
    elif path.suffix == '.cubin' or re.search(r'\.so(?:\.\d+)*$', path.name):
        yield 'binary', {
            'file': path.name, 'size_bytes': path.stat().st_size,
            'sha256': sha256,
        }
    else:
        with path.open('r', encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield str(number), line.rstrip('\r\n').split('\t')


def import_evidence(root: Path, db_path: Path, strict: bool = False):
    root, db_path = Path(root), Path(db_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {'imported': {}, 'missing': [], 'status': {}}
    with sqlite3.connect(db_path) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('''CREATE TABLE IF NOT EXISTS sm75_artifacts (
            kind TEXT PRIMARY KEY, relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL)''')
        db.execute('''CREATE TABLE IF NOT EXISTS sm75_evidence (
            kind TEXT NOT NULL, record_key TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
            PRIMARY KEY(kind, record_key),
            FOREIGN KEY(kind) REFERENCES sm75_artifacts(kind))''')
        for kind, relative in [*ARTIFACTS.items(), *BINARY_DIRECTORIES.items()]:
            path = root / relative
            directory_records = None
            if kind in BINARY_DIRECTORIES:
                if not path.is_dir():
                    stats['missing'].append(relative)
                    if strict:
                        raise FileNotFoundError(path)
                    continue
                binary_files = sorted(path.glob('*.cubin'))
                if not binary_files:
                    stats['missing'].append(relative)
                    if strict:
                        raise FileNotFoundError(f'No cubin files under {path}')
                    continue
                directory_records = []
                file_hash = hashlib.sha256()
                size_bytes = 0
                for binary_path in binary_files:
                    binary_hash = hashlib.sha256()
                    with binary_path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            binary_hash.update(chunk)
                    sha_binary = binary_hash.hexdigest()
                    key = binary_path.relative_to(path).as_posix()
                    record_path = (Path(relative) / key).as_posix()
                    record = {'file': record_path, 'size_bytes': binary_path.stat().st_size,
                              'sha256': sha_binary}
                    file_hash.update(record_path.encode('utf-8') + b'\0' + bytes.fromhex(sha_binary))
                    size_bytes += record['size_bytes']
                    directory_records.append((record_path, record))
                sha = file_hash.hexdigest()
            else:
                if not path.is_file():
                    stats['missing'].append(relative)
                    if strict:
                        raise FileNotFoundError(path)
                    continue
                file_hash = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        file_hash.update(chunk)
                sha = file_hash.hexdigest()
                size_bytes = path.stat().st_size
            db.execute('DELETE FROM sm75_evidence WHERE kind=?', (kind,))
            db.execute('''INSERT INTO sm75_artifacts(kind, relative_path, sha256, size_bytes)
                          VALUES(?,?,?,?) ON CONFLICT(kind) DO UPDATE SET
                          relative_path=excluded.relative_path, sha256=excluded.sha256,
                          size_bytes=excluded.size_bytes''',
                       (kind, relative, sha, size_bytes))
            count = 0
            records = directory_records if directory_records is not None else read_records(path, kind, sha)
            for key, obj in records:
                db.execute('INSERT INTO sm75_evidence VALUES(?,?,?,?)',
                           (kind, key, sha, json.dumps(obj, ensure_ascii=False, separators=(',', ':'))))
                count += 1
                if kind == 'bulk_summary' and isinstance(obj, list) and len(obj) >= 2:
                    status = obj[1].strip().lower()
                    if status in ('ok', 'empty', 'timeout', 'error', 'failed'):
                        stats['status'][obj[0]] = status
            stats['imported'][kind] = count
            db.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Existing parent directory containing sm75-kernels.jsonl')
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--strict', action='store_true', help='Fail if any known artifact is absent')
    args = parser.parse_args()
    result = import_evidence(args.root, args.db, args.strict)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print('Verified cross-repository lowering edges: NOT INFERRED')
    print('Verified runtime fusions: NOT INFERRED')


if __name__ == '__main__':
    main()
