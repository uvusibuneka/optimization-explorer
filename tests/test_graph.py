import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from optimization_explorer.graph import case_splitter, persist, run_graph, validator
from optimization_explorer.indexer import index_directory
from optimization_explorer.llm import call_model, demo_worker
from optimization_explorer.schema import COLUMNS
from optimization_explorer.storage import Store


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_openkilo_default_timeout_exceeds_gateway_deadline(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"cases": []}'))], usage=None)
        completion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": SimpleNamespace(acompletion=completion)}):
            with patch.dict("os.environ", {
                "OPENKILO_MODEL": "openai/worker",
                "OPENKILO_BASE_URL": "http://127.0.0.1:18765/v1",
            }, clear=True):
                await call_model("worker", {}, "worker")
        self.assertEqual(completion.await_args.kwargs["timeout"], 300)

    async def test_validator_cannot_approve_fabricated_file(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        task = index_directory(str(examples), "synthetic-demo")[0]
        case, = demo_worker(task)
        invented = dict(case, source_file="invented.py", validation_status="VALIDATED")
        with patch("optimization_explorer.graph.call_model",
                   return_value={"cases": [invented], "issues": []}):
            result = await validator({"task": task, "cases": [case], "demo": False})
        self.assertFalse(result["done"])
        self.assertIn("case[0]: source_file mismatch", result["errors"])

    async def test_model_request_omits_local_source_metadata(self):
        task = {"source_file": "src/f.py", "context": "def fuse(): pass",
                "local_source_path": "/private/source.zip", "zip_member": "repo/src/f.py"}
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"cases": []}'))])
        completion = AsyncMock(return_value=response)
        with patch.dict("sys.modules", {"litellm": SimpleNamespace(acompletion=completion)}):
            with patch.dict("os.environ", {"OPENKILO_MODEL": "fixture", "OPENKILO_API_KEY": "fixture",
                                           "OPENKILO_BASE_URL": "https://example.invalid",
                                           "OPENKILO_TIMEOUT_SECONDS": "12.5",
                                           "OPENKILO_MAX_RETRIES": "0",
                                           "OPENKILO_MAX_OUTPUT_TOKENS": "512"}):
                self.assertEqual(await call_model("worker", {"task": task}, "worker"), {"cases": []})
                completion.return_value = SimpleNamespace(choices=[])
                with self.assertRaisesRegex(ValueError, "contains no choices"):
                    await call_model("worker", {"task": task}, "worker")
        messages = completion.await_args.kwargs["messages"]
        self.assertEqual(completion.await_args.kwargs["timeout"], 12.5)
        self.assertEqual(completion.await_args.kwargs["max_retries"], 0)
        self.assertEqual(completion.await_args.kwargs["max_tokens"], 512)
        self.assertEqual(json.loads(messages[-1]["content"]), {
            "task": {"source_file": "src/f.py", "context": "def fuse(): pass"},
        })
        self.assertIn("local_source_path", task)

    def test_persist_records_cross_task_case_conflict(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        first = index_directory(str(examples), "synthetic-demo")[0]
        second = dict(first, task_id="second")
        case, = demo_worker(first)
        conflict = dict(case, optimization_outcome="different outcome")
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cases.db")
            store = Store(db)
            try:
                store.add_tasks([first, second])
                result = persist({"db_path": db, "reviewed": [
                    {"task": first, "cases": [case], "errors": []},
                    {"task": second, "cases": [conflict], "errors": []},
                ]})
                self.assertEqual(result["summary"], {"cases_saved": 1, "tasks_unresolved": 1})
                self.assertEqual(
                    store.conn.execute("SELECT optimization_outcome FROM optimization_cases").fetchone(),
                    (case["optimization_outcome"],),
                )
                self.assertEqual(
                    store.conn.execute("SELECT task_id FROM unresolved_cases").fetchall(), [("second",)]
                )
            finally:
                store.conn.close()

    async def test_validator_rejects_duplicate_case_ids(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        task = next(t for t in index_directory(str(examples), "synthetic-demo")
                    if t["source_function"] == "example_fuse_conv_bn")
        case, = demo_worker(task)
        with patch("optimization_explorer.graph.call_model",
                   return_value={"cases": [case, dict(case)], "issues": []}):
            result = await validator({"task": task, "cases": [case], "demo": False})
        self.assertFalse(result["done"])
        self.assertIn("review contains duplicate case IDs", result["errors"])

    async def test_repair_replaces_invalid_source_reference(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        task = next(t for t in index_directory(str(examples), "synthetic-demo")
                    if t["source_function"] == "example_fuse_conv_bn")
        good, = demo_worker(task)
        bad = dict(good, source_file="missing.py")
        stages = []

        async def model(role, payload, stage):
            stages.append(stage)
            if stage == "worker":
                return {"cases": [bad]}
            if stage == "repair":
                self.assertTrue(any("source_file mismatch" in error for error in payload["errors"]))
                return {"cases": [good]}
            return {"cases": payload["cases"], "issues": []}

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cases.db")
            store = Store(db)
            try:
                store.add_tasks([task])
                with patch("optimization_explorer.graph.call_model", side_effect=model):
                    summary = await run_graph(db, max_repair=1)
                self.assertEqual(summary, {"cases_saved": 1, "tasks_unresolved": 0})
                self.assertEqual(stages, ["worker", "validator", "repair", "validator"])
                self.assertEqual(
                    store.conn.execute("SELECT source_file FROM optimization_cases").fetchone(),
                    (task["source_file"],),
                )
            finally:
                store.conn.close()

    async def test_worker_failure_does_not_lose_successful_task(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        task = next(t for t in index_directory(str(examples), "synthetic-demo")
                    if t["source_function"] == "example_fuse_conv_bn")
        failed_task = dict(task, task_id="failed")

        def worker(current_task):
            if current_task["task_id"] == "failed":
                raise RuntimeError("fixture worker failure")
            return demo_worker(current_task)

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cases.db")
            store = Store(db)
            try:
                store.add_tasks([task, failed_task])
                with patch("optimization_explorer.graph.demo_worker", side_effect=worker):
                    summary = await run_graph(db, demo=True, concurrency=2)
                self.assertGreater(summary["cases_saved"], 0)
                self.assertEqual(summary["tasks_unresolved"], 1)
                self.assertEqual(store.pending(2), [])
                row, = store.conn.execute("SELECT task_id, errors FROM unresolved_cases").fetchall()
                self.assertEqual(row[0], "failed")
                self.assertIn("fixture worker failure", row[1])
            finally:
                store.conn.close()

    def test_splitter_reports_conflicts_and_collapses_exact_duplicates(self):
        first = {"case_id": "shared", "applicability_predicate": "enabled"}
        alternative = {"case_id": "alternative", "applicability_predicate": "disabled"}
        output = case_splitter({"worker_outputs": [{
            "task": {"task_id": "fixture"},
            "cases": [
                first, dict(first), alternative,
                dict(first, applicability_predicate="disabled"),
                {"case_id": ["invalid"]}, None,
            ],
            "errors": [],
        }]})
        bundle, = output["review_inputs"]
        self.assertEqual([case["case_id"] for case in bundle["cases"]], ["shared", "alternative"])
        self.assertEqual(bundle["errors"], [
            "distinct worker cases share a case_id",
            "case_id must be a string",
            "worker case must be an object",
        ])

    async def test_demo_pipeline_persists_indexed_case_and_exports_csv(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        tasks = index_directory(str(examples), "synthetic-demo", max_files=5)
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "cases.db")
            store = Store(db)
            try:
                store.add_tasks(tasks)
                store.add_tasks([
                    {"task_id": f"000-{kind}", "evidence_kind": kind}
                    for kind in ("binary", "decompiled", "ghidra")
                ])
                summary = await run_graph(db, demo=True, limit=len(tasks))
                self.assertEqual(summary["cases_saved"], 2)
                self.assertEqual(summary["tasks_unresolved"], 0)
                self.assertEqual(store.pending(1, source_only=True), [])
                self.assertEqual(len(store.pending(10)), 3)
                output = Path(tmp) / "cases.csv"
                self.assertEqual(store.export(output), summary["cases_saved"])
                with output.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, COLUMNS)
                    self.assertEqual(
                        {row["source_function"] for row in reader},
                        {"example_fuse_conv_bn", "example_epilogue"},
                    )
            finally:
                store.conn.close()
