import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from tools.sm75_evidence import ARTIFACTS, BINARY_DIRECTORIES, import_evidence


class EvidenceTests(unittest.TestCase):
    def test_preserves_graph_and_status_without_fabricated_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ARTIFACTS['launch_sites']).parent.mkdir(parents=True)
            graph = {'nodes': [{'id': 'a'}], 'edges': [{'from': 'a', 'to': 'b', 'proof': None}]}
            (root / ARTIFACTS['execution_graph']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['execution_graph']).write_text(json.dumps(graph))
            (root / ARTIFACTS['kernel_index']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['kernel_index']).write_text('{"kernel":"x"}\n{"kernel":"y"}\n')
            (root / ARTIFACTS['bulk_summary']).parent.mkdir(parents=True, exist_ok=True)
            (root / ARTIFACTS['bulk_summary']).write_text('library\tstatus\tfunctions\nlibtorch_cuda.so\ttimeout\t0\n')
            db_path = root / 'cases.sqlite'
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE optimization_cases (id INTEGER PRIMARY KEY)')
                db.execute('INSERT INTO optimization_cases VALUES (1)')
            for _ in range(2):
                result = import_evidence(root, db_path)
                self.assertEqual(result['imported']['kernel_index'], 2)
                self.assertEqual(result['status']['libtorch_cuda.so'], 'timeout')
            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM optimization_cases').fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT count(*) FROM sm75_evidence WHERE kind='kernel_index'").fetchone()[0], 2)
                payload = db.execute("SELECT payload_json FROM sm75_evidence WHERE kind='execution_graph'").fetchone()[0]
                self.assertEqual(json.loads(payload), graph)
                self.assertFalse(db.execute("SELECT name FROM sqlite_master WHERE name='sm75_verified_edges'").fetchall())

    def test_imports_demangled_and_run_specific_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            demangled=root/ARTIFACTS['demangled_cublaslt']
            demangled.parent.mkdir(parents=True)
            demangled.write_text('void gemm_kernel()\n')
            cubin=root/ARTIFACTS['linear_cubin']
            cubin.parent.mkdir(parents=True)
            cubin.write_bytes(b'cubin-bytes')
            db_path=root/'cases.sqlite'
            result=import_evidence(root,db_path)
            self.assertEqual(result['imported']['demangled_cublaslt'],1)
            self.assertEqual(result['imported']['linear_cubin'],1)
            with sqlite3.connect(db_path) as db:
                rows=dict(db.execute(
                    "SELECT kind,payload_json FROM sm75_evidence WHERE kind IN (?,?)",
                    ('demangled_cublaslt','linear_cubin'),
                ))
                self.assertEqual(json.loads(rows['demangled_cublaslt']),['void gemm_kernel()'])
                binary=json.loads(rows['linear_cubin'])
                self.assertEqual(binary['size_bytes'],len(b'cubin-bytes'))
                self.assertEqual(binary['sha256'],__import__('hashlib').sha256(b'cubin-bytes').hexdigest())

    def test_imports_shared_object_and_cubin_hashes_without_binary_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_object = root / ARTIFACTS["so_libtorch"]
            shared_object.parent.mkdir(parents=True)
            shared_object.write_bytes(b"shared-object")
            cubin = root / ARTIFACTS["linear_cubin"]
            cubin.parent.mkdir(parents=True)
            cubin.write_bytes(b"cubin")
            db_path = root / "cases.sqlite"
            result = import_evidence(root, db_path)
            self.assertEqual(result["imported"]["so_libtorch"], 1)
            self.assertEqual(result["imported"]["linear_cubin"], 1)
            with sqlite3.connect(db_path) as db:
                records = dict(db.execute(
                    "SELECT kind,payload_json FROM sm75_evidence WHERE kind IN (?,?)",
                    ("so_libtorch", "linear_cubin"),
                ))
            so = json.loads(records["so_libtorch"])
            cubin_record = json.loads(records["linear_cubin"])
            self.assertEqual(so["size_bytes"], len(b"shared-object"))
            self.assertEqual(so["sha256"], __import__("hashlib").sha256(b"shared-object").hexdigest())
            self.assertEqual(cubin_record["sha256"], __import__("hashlib").sha256(b"cubin").hexdigest())

    def test_imports_every_cubin_from_binary_directory_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cubins = root / BINARY_DIRECTORIES["capture_cubins"]
            cubins.mkdir(parents=True)
            (cubins / "image_1.cubin").write_bytes(b"first cubin")
            (cubins / "image_2.cubin").write_bytes(b"second cubin")
            db_path = root / "cases.sqlite"
            result = import_evidence(root, db_path)
            self.assertEqual(result["imported"]["capture_cubins"], 2)
            with sqlite3.connect(db_path) as db:
                artifact = db.execute(
                    "SELECT relative_path,size_bytes,sha256 FROM sm75_artifacts WHERE kind=?",
                    ("capture_cubins",),
                ).fetchone()
                records = db.execute(
                    "SELECT record_key,artifact_sha256,payload_json FROM sm75_evidence "
                    "WHERE kind=? ORDER BY record_key",
                    ("capture_cubins",),
                ).fetchall()
            self.assertEqual(artifact[0], BINARY_DIRECTORIES["capture_cubins"])
            self.assertEqual(artifact[1], len(b"first cubin") + len(b"second cubin"))
            self.assertEqual([row[0].rsplit("/", 1)[-1] for row in records],
                             ["image_1.cubin", "image_2.cubin"])
            for _, artifact_hash, payload in records:
                self.assertEqual(artifact_hash, artifact[2])
                self.assertIn("sha256", json.loads(payload))

    def test_bad_jsonl_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ARTIFACTS['kernel_index']
            path.parent.mkdir(parents=True)
            db_path = root / 'cases.sqlite'
            path.write_text('{"a":1}\n')
            import_evidence(root, db_path)
            path.write_text('{"a":2}\nBAD\n')
            with self.assertRaises(ValueError):
                import_evidence(root, db_path)
            with sqlite3.connect(db_path) as db:
                payload = db.execute("SELECT payload_json FROM sm75_evidence WHERE kind='kernel_index'").fetchone()[0]
                self.assertEqual(json.loads(payload), {'a': 1})


if __name__ == '__main__':
    unittest.main()
