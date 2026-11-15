import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.layer_atlas import discover, persist, summary


class LayerAtlasTests(unittest.TestCase):
    def test_discovery_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'sources' / 'keras' / 'keras' / 'src'
            root.mkdir(parents=True)
            (root / 'layers.py').write_text('''\
class CustomDepthwiseConv:
    def __init__(self, kernel_size, strides, padding):
        self.kernel_size = kernel_size

def fused_matmul(x):
    if x.shape[-1] % 8 == 0:
        return relu(matmul(x, x))
    return matmul(x, x)
''')
            result = discover(Path(tmp) / 'sources')
            names = {item['symbol'] for item in result['symbols']}
            self.assertIn('CustomDepthwiseConv', names)
            self.assertIn('fused_matmul', names)
            self.assertTrue(any(c['input_op'] == 'matmul' and c['output_op'] == 'relu' for c in result['combinations']))
            self.assertTrue(result['predicates'])
            dbpath = Path(tmp) / 'atlas.sqlite'
            persist(dbpath, result)
            with sqlite3.connect(dbpath) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM atlas_symbols').fetchone()[0], len(result['symbols']))
                self.assertEqual(db.execute('SELECT count(*) FROM atlas_coverage WHERE sm75="unknown"').fetchone()[0], len(result['symbols']))
                self.assertFalse(db.execute("SELECT name FROM sqlite_master WHERE name='tasks'").fetchall())
            self.assertEqual(summary(result)['verified_runtime_fusions'], 0)

    def test_indexes_decompiled_so_symbols_and_dispatch_assembly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'sources'/'decompiled'/'sm75'
            functions=root/'functions'
            dispatch=root/'host_dispatch'
            functions.mkdir(parents=True)
            dispatch.mkdir(parents=True)
            (functions/'libcublas.so.functions.txt').write_text('symbol list\n')
            (dispatch/'dispatcher.asm').write_text('host dispatch dump\n')
            result=discover(Path(tmp)/'sources')
            self.assertEqual({item['file'] for item in result['binaries']},{
                'sm75/functions/libcublas.so.functions.txt',
                'sm75/host_dispatch/dispatcher.asm',
            })
            self.assertTrue(all(item['status']=='unlinked_binary_text' for item in result['binaries']))
            self.assertEqual(result['repositories'][0]['role'],'binary_evidence')

    def test_scans_cuda_sources_and_skips_unrelated_repository_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/'sources'
            native=sources/'cutlass'/'include'/'cutlass'
            native.mkdir(parents=True)
            (native/'fused_gemm.cu').write_text('__global__ void fused_gemm() {}\n')
            unrelated=sources/'cutlass'/'tools'/'unrelated.cpp'
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text('void unrelated_matmul() {}\n')
            result=discover(sources)
            self.assertEqual([symbol['symbol'] for symbol in result['symbols']],['fused_gemm'])
            self.assertEqual(result['scanned_files'],1)

    def test_excludes_jax_repository_and_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/'sources'
            (sources/'jax').mkdir(parents=True)
            (sources/'jax'/'ops.py').write_text('def matmul(x):\n    return x\n')
            pytorch=sources/'pytorch'/'torch'/'nn'
            pytorch.mkdir(parents=True)
            (pytorch/'linear_test.py').write_text('class TestLinear: pass\n')
            (pytorch/'linear.py').write_text('class Linear: pass\n')
            result=discover(sources)
            self.assertEqual({item['repository'] for item in result['symbols']},{'pytorch'})
            self.assertEqual({item['symbol'] for item in result['symbols']},{'Linear'})

    def test_partial_persist_preserves_other_repositories(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/'sources'
            keras=sources/'keras'/'keras'/'src'
            keras.mkdir(parents=True)
            (keras/'layers.py').write_text('class ConvLayer: pass\n')
            first=discover(sources)
            db=Path(tmp)/'atlas.sqlite'
            persist(db,first)

            decompiled=sources/'decompiled'/'sm75'/'functions'
            decompiled.mkdir(parents=True)
            (decompiled/'libcudnn.so.functions.txt').write_text('symbols\n')
            partial=discover(sources,repositories=['decompiled'])
            persist(db,partial)
            with sqlite3.connect(db) as conn:
                repos={row[0] for row in conn.execute('SELECT DISTINCT repository FROM atlas_symbols')}
                self.assertEqual(repos,{'keras'})
                binary_repos={row[0] for row in conn.execute('SELECT DISTINCT repository FROM atlas_binary_text')}
                self.assertEqual(binary_repos,{'decompiled'})

    def test_duplicate_nested_calls_and_repeat_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'sources' / 'keras' / 'keras' / 'src'
            root.mkdir(parents=True)
            (root / 'ops.py').write_text(
                'def fused_matmul(x):\n'
                '    return relu(matmul(x, x), matmul(x, x))\n'
            )
            result = discover(Path(tmp) / 'sources')
            self.assertGreater(len(result['combinations']), 1)
            dbpath = Path(tmp) / 'atlas.sqlite'
            persist(dbpath, result)
            persist(dbpath, result)
            with sqlite3.connect(dbpath) as db:
                rows = db.execute('SELECT id FROM atlas_combinations').fetchall()
                self.assertEqual(len(rows), len({row[0] for row in rows}))
                self.assertEqual(len(rows), 1)


        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jax').mkdir()
            (root / 'jax' / 'ops.py').write_text('def matmul(x):\n    return x\n')
            cutlass = root / 'cutlass' / 'include'
            cutlass.mkdir(parents=True)
            (cutlass / 'ops.py').write_text('def matmul(x):\n    return x\n')
            result = discover(root)
            self.assertEqual(len(result['symbols']), 1)
            self.assertEqual({repo['name'] for repo in result['repositories']}, {'cutlass'})
            self.assertEqual(len(result['combinations']), 0)
            self.assertEqual(summary(result)['verified_cross_repository_lowering_edges'], 0)


if __name__ == '__main__':
    unittest.main()
