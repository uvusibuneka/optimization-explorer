import sqlite3
import tempfile
import unittest
from pathlib import Path
from tools.agent_supervisor import (endpoints, memory_snapshot, queues, recover_stopped_worker,
                                    roles_to_start)


class AgentSupervisorTests(unittest.TestCase):
    def test_worker_health_probe_uses_current_kilo_gateway(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
        from unittest.mock import patch
        with patch("tools.agent_supervisor.urllib.request.urlopen", return_value=Response()) as openurl:
            result=endpoints()
        self.assertTrue(result["worker"])
        self.assertIn("https://ai-gateway.kilo.ai/api/v1/models",
                      [call.args[0] for call in openurl.call_args_list])

    def test_memory_threshold_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "meminfo"
            f.write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
            self.assertEqual(memory_snapshot(f)["used_percent"], 75.0)

    def test_stopped_workers_return_leased_tasks_and_cases_to_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "q.sqlite"
            with sqlite3.connect(db_path) as db:
                db.execute("create table trace_tasks(id text,state text,error text)")
                db.execute("insert into trace_tasks values('t1','running',null)")
                db.execute("create table trace_leases(task_id text)")
                db.execute("insert into trace_leases values('t1')")
                db.execute("create table case_eval_state(case_id text,status text,claim text,claimed_at integer,last_error text,updated_at text)")
                db.execute("insert into case_eval_state values('c1','running','token',1,null,'now')")
            self.assertEqual(recover_stopped_worker(db_path, "trace"), 1)
            self.assertEqual(recover_stopped_worker(db_path, "fixeval"), 1)
            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute("select state from trace_tasks").fetchone()[0], "pending")
                self.assertEqual(db.execute("select count(*) from trace_leases").fetchone()[0], 0)
                self.assertEqual(db.execute("select status,claim from case_eval_state").fetchone(), ("pending", None))

    def test_stopped_repro_generator_releases_its_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "q.sqlite"
            with sqlite3.connect(db_path) as db:
                db.execute("""CREATE TABLE case_repro_scripts (
                    case_id TEXT, status TEXT, claim TEXT, claimed_at INTEGER,
                    error TEXT, updated_at TEXT)""")
                db.execute("INSERT INTO case_repro_scripts VALUES('c1','running','token',1,null,'now')")
            self.assertEqual(recover_stopped_worker(db_path, "reprogen"), 1)
            with sqlite3.connect(db_path) as db:
                self.assertEqual(db.execute(
                    "SELECT status,claim,error FROM case_repro_scripts").fetchone(),
                    ("failed", None, "supervisor recovered interrupted generation"))

    def test_trace_and_repro_workers_can_run_concurrently(self):
        roles = {"trace": [], "reprogen": []}
        self.assertEqual(roles_to_start(roles, {}), ["trace", "reprogen"])
        self.assertEqual(roles_to_start(roles, {"trace": object()}), ["reprogen"])
        self.assertEqual(roles_to_start(roles, {"reprogen": object()}), ["trace"])

    def test_branch_seed_bootstraps_before_repro_worker(self):
        roles = {"trace": [], "reprogen": []}
        self.assertEqual(roles_to_start(roles, {}, seed_pending=True), ["trace"])
        self.assertEqual(roles_to_start(roles, {"trace": object()}, seed_pending=True), [])


    def test_queue_snapshot_reads_both_collaborating_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "q.sqlite"
            with sqlite3.connect(db_path) as db:
                db.execute("create table trace_tasks(state text)")
                db.execute("insert into trace_tasks values('pending')")
                db.execute("create table case_eval_state(status text)")
                db.execute("insert into case_eval_state values('passed')")
                db.execute("create table trace_task_errors(created_at text)")
            result = queues(db_path)
        self.assertEqual(result["trace"], {"pending": 1})
        self.assertEqual(result["eval"], {"passed": 1})
        self.assertEqual(result["reprogen"], {})
        self.assertEqual(result["reprogen_pending"], 0)
        self.assertEqual(result["errors_10m"], 0)

    def test_repro_queue_counts_only_bounded_retryable_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "q.sqlite"
            with sqlite3.connect(db_path) as db:
                db.execute("CREATE TABLE optimization_cases(case_id TEXT)")
                db.executemany("INSERT INTO optimization_cases VALUES(?)",
                               [("retryable",), ("exhausted",), ("fresh",)])
                db.execute("""CREATE TABLE case_repro_scripts(
                    case_id TEXT,status TEXT,claimed_at INTEGER,attempts INTEGER)""")
                db.executemany("INSERT INTO case_repro_scripts VALUES(?,?,NULL,?)",
                               [("retryable", "failed", 2), ("exhausted", "failed", 3)])
            result = queues(db_path)
        self.assertEqual(result["reprogen_pending"], 2)


if __name__ == "__main__":
    unittest.main()
