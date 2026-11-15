#!/usr/bin/env python3
"""Validate existing selective-scan static evidence; never infer runtime fusion.

Derived tables are separate from atlas, imported evidence and optimization cases.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(db, kind):
    row = db.execute('SELECT relative_path, sha256 FROM sm75_artifacts WHERE kind=?', (kind,)).fetchone()
    if row is None:
        raise ValueError('Missing imported artifact: ' + kind)
    records = db.execute('SELECT record_key, payload_json FROM sm75_evidence WHERE kind=?', (kind,)).fetchall()
    if not records:
        raise ValueError('Empty imported artifact: ' + kind)
    return row, [(key, json.loads(payload)) for key, payload in records]


def trace(db_path: Path, root: Path | None = None):
    """Cross-check exact identifiers; record graph hypotheses without upgrading them."""
    with sqlite3.connect(db_path) as db:
        db.execute('PRAGMA foreign_keys=ON')
        inputs = {}
        for kind in ('execution_graph', 'kernel_index', 'launch_sites'):
            meta, records = _load(db, kind)
            if root is not None:
                path = root / meta[0]
                if not path.is_file() or _sha(path) != meta[1]:
                    raise ValueError('Imported artifact differs from disk: ' + kind)
            inputs[kind] = (meta, records)
        graph = inputs['execution_graph'][1][0][1]
        if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list):
            raise ValueError('Malformed execution graph')
        if graph.get('architecture') != 'sm_75':
            raise ValueError('Graph is not sm_75')
        kernels = {}
        for _, obj in inputs['kernel_index'][1]:
            if obj.get('architecture') == 'sm_75':
                key = (obj.get('library_basename'), obj.get('mangled_symbol'))
                kernels.setdefault(key, []).append(obj)
        sites = inputs['launch_sites'][1][0][1]
        if not isinstance(sites, list):
            raise ValueError('Malformed launch sites')
        site_keys = Counter((s.get('dispatcher_mangled'), s.get('call_address'), s.get('kernel_mangled'))
                            for s in sites if isinstance(s, dict) and s.get('status') == 'binary_symbol_matched')
        nodes = {}
        for node in graph['nodes']:
            if not isinstance(node, dict) or not isinstance(node.get('id'), str) or node['id'] in nodes:
                raise ValueError('Invalid or duplicate graph node id')
            nodes[node['id']] = node
        edges = []
        for number, edge in enumerate(graph['edges']):
            if not isinstance(edge, dict) or edge.get('source') not in nodes or edge.get('target') not in nodes:
                raise ValueError('Graph edge has missing endpoint')
            status = edge.get('status', 'unknown')
            classification = 'graph_assertion_unverified'
            kernel_id = None
            if status == 'hypothesis':
                classification = 'hypothesis'
            elif status == 'source_supported':
                classification = 'source_asserted_not_locally_rechecked'
            elif status == 'source_and_binary_supported':
                classification = 'source_and_binary_asserted_not_locally_rechecked'
            elif status == 'binary_supported':
                classification = 'binary_asserted_not_correlated'
                target = nodes[edge['target']]
                source = nodes[edge['source']]
                if target.get('kind') == 'cuda_kernel_specialization' and source.get('kind') == 'cpp_cuda_dispatch':
                    evidences = [e for e in edge.get('evidence', []) if isinstance(e, dict) and e.get('kind') == 'static_cuda_launch']
                    for evidence in evidences:
                        symbol = evidence.get('kernel_mangled')
                        library = next((e.get('library') for e in target.get('evidence', [])
                                        if isinstance(e, dict) and e.get('library')), None)
                        matches = kernels.get((library, symbol), [])
                        matched_site = any(site_keys[(s.get('dispatcher_mangled'), s.get('call_address'), symbol)]
                                           for s in sites if isinstance(s, dict) and
                                           s.get('dispatcher_address') == evidence.get('dispatcher_address') and
                                           s.get('call_address') == evidence.get('call_address') and
                                           s.get('kernel_mangled') == symbol and
                                           s.get('status') == 'binary_symbol_matched')
                        if (symbol == target.get('name') and len(matches) == 1 and matched_site and
                                matches[0].get('id') and evidence.get('architecture') == 'sm_75'):
                            classification = 'static_binary_correlated'
                            kernel_id = matches[0]['id']
                            break
            edges.append((number, edge, classification, kernel_id))
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS sm75_trace_nodes (
                node_id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sm75_trace_edges (
                edge_index INTEGER PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                relation TEXT NOT NULL, original_status TEXT NOT NULL,
                classification TEXT NOT NULL, kernel_id TEXT, runtime_verified INTEGER NOT NULL,
                payload_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sm75_trace_metadata (
                key TEXT PRIMARY KEY, value_json TEXT NOT NULL)""")
            db.execute('DELETE FROM sm75_trace_edges')
            db.execute('DELETE FROM sm75_trace_nodes')
            db.execute('DELETE FROM sm75_trace_metadata')
            for node in nodes.values():
                db.execute('INSERT INTO sm75_trace_nodes VALUES (?,?,?,?)',
                           (node['id'], node.get('kind', ''), node.get('name', ''), json.dumps(node, ensure_ascii=False)))
            for index, edge, classification, kernel_id in edges:
                db.execute('INSERT INTO sm75_trace_edges VALUES (?,?,?,?,?,?,?,?,?)',
                           (index, edge['source'], edge['target'], edge.get('relation', ''),
                            edge.get('status', ''), classification, kernel_id, 0,
                            json.dumps(edge, ensure_ascii=False)))
            meta = {'graph_target': graph.get('target'), 'architecture': graph['architecture'],
                    'source_artifact_sha256': {k: v[0][1] for k, v in inputs.items()},
                    'scope': 'existing graph only; no inferred cross-framework lowering or runtime fusion'}
            for key, value in meta.items():
                db.execute('INSERT INTO sm75_trace_metadata VALUES (?,?)', (key, json.dumps(value)))
        return {'nodes': len(nodes), 'edges': len(edges),
                'classifications': dict(sorted(Counter(e[2] for e in edges).items())),
                'graph_target': graph.get('target'), 'runtime_verified_edges': 0,
                'verified_cross_framework_lowering': 0, 'verified_runtime_fusions': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--root', type=Path, help='Optional: verify source file hashes against imported artifacts')
    args = p.parse_args()
    print(json.dumps(trace(args.db, args.root), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
