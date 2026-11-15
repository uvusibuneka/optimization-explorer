import json
import tempfile
import unittest
from pathlib import Path
from optimization_explorer.binary_inspector import inspect_binary, binary_tasks
from optimization_explorer.workspace import ingest, inventory
from optimization_explorer.storage import Store
from optimization_explorer.schema import COLUMNS

class WorkspaceTests(unittest.TestCase):
    def test_binary_inspection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'cudnn'/'libcudnn.so'; p.parent.mkdir()
            p.write_bytes(b'\x7fELF'+b'\x00'*16+b'cudnn_fused_attention_kernel\x00'+b'gemm_epilogue_relu\x00')
            r=inspect_binary(p,root)
            self.assertEqual(r['format'],'ELF')
            self.assertEqual(len(r['sha256']),64)
            self.assertTrue(r['strings'])
            tasks=binary_tasks([r]); self.assertEqual(len(tasks),1)
            self.assertEqual(tasks[0]['evidence_kind'],'binary')
    def test_workspace_ingest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); sources=root/'sources'; sources.mkdir()
            (sources/'repositories.json').write_text(json.dumps({'repositories':[{'id':'toy','url':'https://example.org'}]}))
            toy=sources/'toy';toy.mkdir();(toy/'a.py').write_text('def fuse(x):\n    return x # fusion kernel\n')
            binaries=root/'binaries'; binaries.mkdir(); (binaries/'cudnn').mkdir()
            (binaries/'cudnn'/'libcudnn.so').write_bytes(b'\x7fELF'+b'cudnn_fused_attention_kernel')
            store=Store(str(root/'result.db'))
            report=ingest(sources,store,max_files=10,max_hits=3)
            self.assertGreater(report['stats']['tasks'],0)
            self.assertEqual(report['stats']['binary_evidence'],0)
            self.assertTrue(inventory(sources)['repositories'][0]['present'])
            self.assertEqual(len(COLUMNS),28)
            store.conn.close()
