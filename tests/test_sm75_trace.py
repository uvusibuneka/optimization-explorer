import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from tools.sm75_evidence import ARTIFACTS, import_evidence
from tools.sm75_trace import trace


class TraceTests(unittest.TestCase):
    def test_correlates_only_exact_binary_match(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ARTIFACTS['launch_sites']).parent.mkdir(parents=True, exist_ok=True)
            kernel = {'id': 'kernel-1', 'architecture': 'sm_75', 'library_basename': 'lib.so', 'mangled_symbol': '_Zkernel'}
            (root / ARTIFACTS['kernel_index']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['kernel_index']).write_text(json.dumps(kernel) + '\n')
            graph = {'architecture': 'sm_75', 'target': 'fixture', 'nodes': [
                {'id': 'd', 'kind': 'cpp_cuda_dispatch', 'name': 'dispatch'},
                {'id': 'k', 'kind': 'cuda_kernel_specialization', 'name': '_Zkernel', 'evidence': [{'library': 'lib.so'}]}],
                'edges': [{'source': 'd', 'target': 'k', 'status': 'binary_supported', 'relation': 'may_launch',
                           'evidence': [{'kind': 'static_cuda_launch', 'kernel_mangled': '_Zkernel',
                                         'dispatcher_address': '0x1', 'call_address': '0x2', 'architecture': 'sm_75'}]},
                          {'source': 'd', 'target': 'k', 'status': 'hypothesis', 'relation': 'may_launch'}]}
            (root / ARTIFACTS['execution_graph']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['execution_graph']).write_text(json.dumps(graph))
            sites = [{'status': 'binary_symbol_matched', 'dispatcher_mangled': '_Zdispatch',
                      'dispatcher_address': '0x1', 'call_address': '0x2', 'kernel_mangled': '_Zkernel'}]
            (root / ARTIFACTS['launch_sites']).write_text(json.dumps(sites))
            db = root / 'test.sqlite'
            import_evidence(root, db)
            result = trace(db, root)
            self.assertEqual(result['classifications']['static_binary_correlated'], 1)
            self.assertEqual(result['classifications']['hypothesis'], 1)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute('SELECT kernel_id FROM sm75_trace_edges WHERE edge_index=0').fetchone()[0], 'kernel-1')
                self.assertEqual(conn.execute('SELECT runtime_verified FROM sm75_trace_edges WHERE edge_index=0').fetchone()[0], 0)
            sites[0]['kernel_mangled'] = '_Zother'
            (root / ARTIFACTS['launch_sites']).write_text(json.dumps(sites))
            with self.assertRaisesRegex(ValueError, 'differs from disk'):
                trace(db, root)
            import_evidence(root, db)
            self.assertEqual(trace(db, root)['classifications']['binary_asserted_not_correlated'], 1)

    def test_rejects_dangling_edge_without_writing_trace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ARTIFACTS['launch_sites']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['kernel_index']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['execution_graph']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['kernel_index']).write_text('{"id":"x"}\n')
            (root / ARTIFACTS['launch_sites']).write_text('[]')
            (root / ARTIFACTS['execution_graph']).write_text(json.dumps({
                'architecture': 'sm_75', 'nodes': [{'id': 'a'}],
                'edges': [{'source': 'a', 'target': 'missing'}]}))
            db = root / 'test.sqlite'
            import_evidence(root, db)
            with self.assertRaisesRegex(ValueError, 'missing endpoint'):
                trace(db, root)
            with sqlite3.connect(db) as conn:
                self.assertFalse(conn.execute("SELECT name FROM sqlite_master WHERE name='sm75_trace_edges'").fetchone())


if __name__ == '__main__':
    unittest.main()
