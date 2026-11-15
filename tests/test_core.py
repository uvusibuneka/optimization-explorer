import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from optimization_explorer.schema import COLUMNS, normalize_case, validate_shape
from optimization_explorer.indexer import MAX_SOURCE_BYTES, index_directory, index_zip
from optimization_explorer.llm import demo_worker
from optimization_explorer.storage import Store
from optimization_explorer.validation import check_case

ROOT = Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(len(COLUMNS), 28)
        c = normalize_case({"fusion_type": "epilogue", "expected_kernel_effect": "possible_change", "evidence_status": "INFERRED", "validation_status": "PENDING"})
        self.assertEqual(validate_shape(c), [])

    def test_schema_rejects_unhashable_enum_values(self):
        case = normalize_case({"fusion_type": [], "expected_kernel_effect": {}})
        errors = validate_shape(case)
        self.assertIn("fusion_type: invalid value []", errors)
        self.assertIn("expected_kernel_effect: invalid value {}", errors)

    def test_synthetic_index_and_demo(self):
        tasks = index_directory(str(ROOT / "examples"), "synthetic-demo", max_files=5)
        self.assertTrue(tasks)
        matches = [t for t in tasks if t["source_function"] == "example_fuse_conv_bn"]
        self.assertTrue(matches)
        case = demo_worker(matches[0])[0]
        self.assertEqual(check_case(case, matches[0]), [])
        self.assertEqual(case["evidence_status"], "INFERRED")

    def test_index_task_identity_tracks_source_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "example.py"
            source.write_text("def fuse():\n    return 1\n", encoding="utf-8")
            first, = index_directory(tmp, "fixture")
            self.assertEqual(index_directory(tmp, "fixture"), [first])
            case, = demo_worker(first)

            source.write_text("def fuse():\n    return 2\n", encoding="utf-8")
            self.assertIn("source content changed since indexing", check_case(case, first))
            updated, = index_directory(tmp, "fixture")
            self.assertEqual(first["source_lines"], updated["source_lines"])
            self.assertNotEqual(first["source_sha256"], updated["source_sha256"])
            self.assertNotEqual(first["task_id"], updated["task_id"])
            source.unlink()
            self.assertIn("indexed source file cannot be read", check_case(case, first))

    def test_index_directory_prunes_service_directories_before_applying_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("z.py", "a.py", "nested/b.py", "build/ignored.py", ".git/ignored.py"):
                source = Path(tmp) / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("def fuse():\n    return 1\n", encoding="utf-8")
            tasks = index_directory(tmp, "fixture")
            self.assertEqual([task["source_file"] for task in tasks], ["a.py", "z.py", "nested/b.py"])
            self.assertEqual(index_directory(tmp, "fixture", max_files=2), tasks[:2])

    def test_malformed_source_lines_report_validation_errors(self):
        task = index_directory(str(ROOT / "examples"), "synthetic-demo", max_files=5)[0]
        case = demo_worker(task)[0]
        for invalid in (1, None, "9" * 5000):
            with self.subTest(invalid_type=type(invalid)):
                self.assertIn("source_lines must be N or N-M", check_case(dict(case, source_lines=invalid), task))

    def test_index_directory_skips_oversized_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.cpp"
            prefix = b"void fuse() {}\n"
            source.write_bytes(prefix + b" " * (MAX_SOURCE_BYTES - len(prefix)))
            task, = index_directory(tmp, "fixture")
            self.assertEqual(task["source_function"], "fuse")
            with source.open("ab") as stream:
                stream.write(b" ")
            self.assertEqual(index_directory(tmp, "fixture"), [])

    def test_cuda_definition_is_distinguished_from_call_and_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.cu"
            source.write_text(
                "__global__ void fuse() {}\n"
                "    fuse();\n"
                "// void fuse_other() {}\n",
                encoding="utf-8",
            )
            tasks = index_directory(tmp, "fixture")
            self.assertEqual(
                [(task["line"], task["source_function"]) for task in tasks],
                [(1, "fuse"), (2, "UNKNOWN"), (3, "UNKNOWN")],
            )

    def test_zip_preserves_distinct_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "sources.zip"
            expected = ["a/f.py", "b/f.py"]
            for prefix in ("", "repo/"):
                with self.subTest(prefix=prefix):
                    with ZipFile(archive_path, "w") as archive:
                        for name in expected:
                            archive.writestr(prefix + name, "def fuse(): pass\n")
                    tasks = index_zip(archive_path, "fixture")
                    self.assertEqual([task["source_file"] for task in tasks], expected)
                    self.assertEqual(len({task["task_id"] for task in tasks}), 2)
                    case, = demo_worker(tasks[0])
                    initial_errors = check_case(case, tasks[0])
                    with ZipFile(archive_path, "w") as archive:
                        archive.writestr(prefix + expected[0], "def fuse(): return 1\n")
                    self.assertNotIn("source content changed since indexing", initial_errors)
                    self.assertEqual(check_case(case, tasks[0]),
                                     ["source content changed since indexing", *initial_errors])

    def test_queue_resumes_without_requeuing_completed_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.db"
            tasks = [{"task_id": "finished"}, {"task_id": "pending"}]
            store = Store(str(path))
            try:
                store.add_tasks(tasks)
                case = dict.fromkeys(COLUMNS, "UNKNOWN")
                case["case_id"] = "saved"
                store.save_case(case, "finished")
            finally:
                store.conn.close()

            store = Store(str(path))
            try:
                store.add_tasks([
                    dict(tasks[0], local_source_path="source.py"),
                    tasks[1],
                ])
                self.assertEqual(store.pending(10), [tasks[1]])
                self.assertEqual(store.conn.execute(
                    "SELECT status, json_extract(payload, '$.local_source_path') "
                    "FROM tasks WHERE task_id='finished'"
                ).fetchone(), ("DONE", "source.py"))
                self.assertEqual(store.stats()["optimization_cases"], 1)
            finally:
                store.conn.close()

    def test_index_batches_survive_interruption_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "index.db")
            tasks = [{"task_id": str(i)} for i in range(300)]

            def interrupted():
                yield from tasks[:270]
                raise KeyboardInterrupt

            store = Store(path)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    store.add_indexed_tasks(interrupted())
            finally:
                store.conn.close()

            store = Store(path)
            try:
                self.assertEqual(store.stats()["tasks"], 256)
                self.assertEqual(store.add_indexed_tasks(iter(tasks)), 300)
                self.assertEqual(store.stats()["tasks"], 300)
            finally:
                store.conn.close()

    def test_index_text_hlo_preserves_source_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.hlo"
            source.write_text(
                "HloModule fixture\nENTRY main {\n  ROOT result = f32[] fusion()\n}\n",
                encoding="utf-8",
            )
            task, = index_directory(tmp, "fixture")
            self.assertEqual(task["source_file"], "fixture.hlo")
            self.assertEqual(task["source_function"], "main")
            self.assertEqual(task["line"], 3)
            self.assertIn("3:   ROOT result = f32[] fusion()", task["context"])

    def test_index_mlir_function_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.mlir"
            source.write_text("func.func private @fused_kernel()\n", encoding="utf-8")
            task, = index_directory(tmp, "fixture")
            self.assertEqual(task["source_function"], "fused_kernel")
            self.assertEqual(task["line"], 1)

    def test_python_index_respects_source_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "valid.py").write_bytes(b'# coding: latin-1\ndef fuse(): return "caf\xe9"\n')
            (root / "invalid.py").write_bytes(b'# coding: invalid-codec\ndef fuse(): pass\n')
            task, = index_directory(tmp, "fixture")
            self.assertEqual(task["source_file"], "valid.py")
            self.assertIn("caf\u00e9", task["context"])

    def test_malformed_indexed_range_reports_validation_error(self):
        task = index_directory(str(ROOT / "examples"), "synthetic-demo")[0]
        case, = demo_worker(task)
        for bounds in (None, "UNKNOWN", "0-10", "9-2"):
            with self.subTest(bounds=bounds):
                self.assertIn(
                    "indexed source line range is invalid",
                    check_case(case, dict(task, source_lines=bounds)),
                )

    def test_csv_export_rejects_database_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.db"
            store = Store(str(path))
            try:
                task = {"task_id": "pending"}
                store.add_tasks([task])
                for output in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                    with self.assertRaisesRegex(ValueError, "output must differ"):
                        store.export(output)
                self.assertEqual(store.pending(1), [task])
                self.assertEqual(store.conn.execute("PRAGMA integrity_check").fetchone(), ("ok",))
            finally:
                store.conn.close()

    def test_storage_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp)/"test.db"))
            task = index_directory(str(ROOT / "examples"), "synthetic-demo", max_files=5)[0]
            store.add_tasks([task])
            case = demo_worker(task)[0]
            store.save_case(case, task["task_id"])
            out = Path(tmp)/"out.csv"
            self.assertEqual(store.export(str(out)), 1)
            with out.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), COLUMNS)
            store.conn.close()

    def test_storage_rolls_back_partial_task_and_allows_retry(self):
        store = Store(":memory:")
        self.addCleanup(store.conn.close)
        task = {"task_id": "task"}
        store.add_tasks([task])
        case = dict.fromkeys(COLUMNS, "UNKNOWN")
        case["case_id"] = "first"
        with self.assertRaises(KeyError):
            store.save_cases([case, {}], task["task_id"])
        self.assertEqual(store.pending(1), [task])
        self.assertEqual(store.stats()["optimization_cases"], 0)
        self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM case_tasks").fetchone()[0], 0)

        store.save_cases([case, dict(case, case_id="second")], task["task_id"])
        self.assertEqual(store.pending(1), [])
        self.assertEqual(store.stats()["optimization_cases"], 2)
        self.assertEqual(
            store.conn.execute("SELECT case_id, task_id FROM case_tasks ORDER BY case_id").fetchall(),
            [("first", "task"), ("second", "task")],
        )

if __name__ == "__main__":
    unittest.main()
