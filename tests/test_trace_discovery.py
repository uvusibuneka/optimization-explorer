import sqlite3
import tempfile
import unittest
from pathlib import Path
from tools.trace_orchestrator import scan_layers, rebuild_layer_queue, branch_candidates, eligible_source


class DiscoveryTests(unittest.TestCase):
    def test_three_framework_roots_and_no_unrelated_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                'keras': ('keras/src/layers/dense.py', 'class Dense(Layer):\n    pass\n'),
                'pytorch': ('torch/nn/modules/linear.py',
                            'from torch.nn import Module as Base\nclass Linear(Base):\n    pass\nclass Fancy(Linear):\n    pass\n'),
                'jax': ('jax/_src/nn/functions.py',
                        'def relu(x):\n    return x\ndef internal_helper(x):\n    return x\n'),
                'tensorflow': ('tensorflow/python/ops/nn_ops.py',
                               'def conv2d(x):\n    return x\ndef _private(x):\n    return x\n'),
            }
            for repo, (rel, content) in cases.items():
                p = root / repo / rel
                p.parent.mkdir(parents=True)
                p.write_text(content)
            rows = list(scan_layers(root))
            self.assertEqual({r[1] for r in rows}, set(cases) - {'jax'})
            self.assertEqual({r[2] for r in rows}, {'Dense','Linear','Fancy','conv2d'})
            raw = 'def call(x):\n    if x is None:\n        return None\n    if x.dtype == "float16":\n        return x\n'
            self.assertEqual(branch_candidates(raw, 1), [(4, 'x.dtype == "float16"')])

    def test_kernel_test_directories_are_not_source_roots(self):
        self.assertFalse(eligible_source('tensorflow/python/kernel_tests/ops.py'))
        self.assertFalse(eligible_source('pytorch/torch/nn/test_helpers/ops.py'))

    def test_branch_filter_requires_optimization_dependency_and_distinct_paths(self):
        raw = '''def forward(x, use_bias):
    if x is None:
        return x
    if x.dtype == torch.float16:
        return fast_linear(x)
    if use_bias:
        return x + bias
    return linear(x)
'''
        branches = branch_candidates(raw, 1)
        self.assertEqual([line for line, _ in branches], [4, 6])

    def test_bias_configuration_branch_is_an_optimization_branch(self):
        raw = '''class Linear(Module):
    def __init__(self, bias):
        if bias:
            self.bias = Parameter()
        else:
            self.register_parameter("bias", None)
'''
        self.assertEqual(branch_candidates(raw, 1), [(3, 'bias')])

    def test_input_validation_guards_do_not_become_optimization_branches(self):
        raw = '''def conv(x, groups):
    if groups <= 0:
        raise ValueError("groups must be positive")
    if x.shape[-1] >= 32:
        return optimized(x)
    return fallback(x)
'''
        self.assertEqual([line for line, _ in branch_candidates(raw, 1)], [4])

    def test_rebuild_preserves_evidence_and_refuses_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'sources' / 'pytorch' / 'torch' / 'nn' / 'modules' / 'linear.py'
            p.parent.mkdir(parents=True)
            p.write_text('class Linear(Module):\n    pass\n')
            db = root / 'cases.sqlite'
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE sm75_kernel_index(id TEXT)')
                conn.execute("INSERT INTO sm75_kernel_index VALUES('keep')")
            result = rebuild_layer_queue(db, root / 'sources')
            self.assertEqual(result['new_layers'], 1)
            self.assertTrue(Path(result['backup']).is_file())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM sm75_kernel_index').fetchone()[0], 1)
                conn.execute("UPDATE trace_tasks SET state='running'")
            with self.assertRaisesRegex(RuntimeError, 'already started'):
                rebuild_layer_queue(db, root / 'sources')


if __name__ == '__main__':
    unittest.main()
