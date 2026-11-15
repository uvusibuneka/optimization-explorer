import asyncio
import os
import sqlite3
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tools.fixevalloop import (execute_script, generate_repro_case, interpreter_for_framework,
                              read_repro_cases, run_case, validate_script,
                              watch_repro_generation)


class FixEvalLoopTests(unittest.TestCase):
    def test_generate_only_worker_polls_independent_repro_queue(self):
        class StopPolling(Exception):
            pass

        args = type("Args", (), {"db": Path("cases.sqlite"), "sources": Path("sources"),
                                  "out": Path("generated"), "limit": 1, "retry": False,
                                  "watch": True, "poll_seconds": 10})()
        case = {"case_id": "CASE_repro"}
        generated = [{"case_id": "CASE_repro", "status": "generated_ast_valid"}]

        async def stop_after_poll(_):
            raise StopPolling

        with patch("tools.fixevalloop.read_repro_cases", return_value=[case]) as read_cases, \
             patch("tools.fixevalloop.generate_repros", new=AsyncMock(return_value=generated)) as generate, \
             patch("tools.fixevalloop.asyncio.sleep", side_effect=stop_after_poll):
            with self.assertRaises(StopPolling):
                asyncio.run(watch_repro_generation(args))
        read_cases.assert_called_once_with(args.db, 1, False)
        generate.assert_awaited_once_with(args.db, args.sources, args.out, [case], False)

    def test_generate_only_selects_missing_and_expired_repro_claims(self):
        from optimization_explorer.schema import COLUMNS
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cases.sqlite"
            with sqlite3.connect(db) as conn:
                columns = ",".join(f'"{name}" TEXT' for name in COLUMNS)
                conn.execute(f"CREATE TABLE optimization_cases ({columns})")
                for case_id in ("CASE_A", "CASE_B", "CASE_C", "CASE_D"):
                    conn.execute(
                        "INSERT INTO optimization_cases(case_id) VALUES(?)", (case_id,))
                conn.execute("""CREATE TABLE case_repro_scripts (
                    case_id TEXT PRIMARY KEY, status TEXT, claim TEXT, claimed_at INTEGER,
                    script_path TEXT, script_sha256 TEXT, error TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
                from tools.fixevalloop import setup_repro_generation
                setup_repro_generation(conn)
                conn.execute("INSERT INTO case_repro_scripts(case_id,status) VALUES('CASE_B','generated_ast_valid')")
                conn.execute("INSERT INTO case_repro_scripts(case_id,status,attempts) VALUES('CASE_C','failed',2)")
                conn.execute("INSERT INTO case_repro_scripts(case_id,status,claimed_at) VALUES('CASE_D','running',1)")
            cases = read_repro_cases(db, 2)
            self.assertEqual([case["case_id"] for case in cases], ["CASE_C", "CASE_D"])
            retried = read_repro_cases(db, 2, retry=True)
            self.assertEqual([case["case_id"] for case in retried], ["CASE_C", "CASE_D"])
            from tools.fixevalloop import claim_repro_generation
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE case_repro_scripts SET attempts=3 WHERE case_id='CASE_C'")
            self.assertNotIn("CASE_C", [case["case_id"] for case in read_repro_cases(db, 10)])
            self.assertIn("CASE_C", [case["case_id"] for case in read_repro_cases(db, 10, retry=True)])
            self.assertFalse(claim_repro_generation(db, "CASE_C", "bounded", False))
            self.assertTrue(claim_repro_generation(db, "CASE_C", "manual", True))
            with sqlite3.connect(db) as conn:
                self.assertNotIn("case_eval_state", {
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")})

    def test_repro_prompt_allows_tensorflow_and_keras_modules(self):
        from tools.fixevalloop import prompt_payload
        payload = prompt_payload({"framework": "tensorflow"}, {}, None, None)
        self.assertIn("tensorflow", payload["requirement"])
        self.assertIn("keras", payload["requirement"])
        from optimization_explorer.llm import load_prompt
        self.assertIn("RuntimeError; do not import sys", load_prompt("agent_case_repro"))
        self.assertIn("do not import sys", load_prompt("agent_case_fix"))
    def test_framework_selects_installed_interpreter(self):
        self.assertEqual(interpreter_for_framework("pytorch"), sys.executable)
        self.assertEqual(interpreter_for_framework("tensorflow"), "/opt/conda/envs/base-tf/bin/python")
        self.assertEqual(interpreter_for_framework("keras"), "/opt/conda/envs/base-tf/bin/python")

    def test_script_import_and_builtin_policy(self):
        self.assertEqual(validate_script("import torch\nif __name__ == '__main__': print(torch.__version__)"),
                         "import torch\nif __name__ == '__main__': print(torch.__version__)")
        self.assertEqual(validate_script("import tensorflow\nimport keras\n"), "import tensorflow\nimport keras\n")
        self.assertEqual(validate_script("import torch\nx = getattr(torch, '__version__')"), "import torch\nx = getattr(torch, '__version__')")
        for script in ("import os\n", "open('x')\n", "x.__class__\n", "exit(1)\n", "quit()\n",
                       "import torch\nx = getattr(torch, 'hub')"):
            with self.assertRaises(ValueError):
                validate_script(script)

    def test_repro_report_observation_fields_must_not_be_hardcoded(self):
        prefix = "import torch\nobserved = {}\n"
        for report, expected in (
            ("{'case_id':'C','reached':True,'framework_version':torch.__version__,'device':'NVIDIA RTX','input_shapes':list(x.shape),'observed_result':observed}", "device"),
            ("{'case_id':'C','reached':True,'framework_version':'2.11.0','device':str(x.device),'input_shapes':list(x.shape),'observed_result':observed}", "framework_version"),
            ("{'case_id':'C','reached':True,'framework_version':torch.__version__,'device':str(x.device),'input_shapes':[32,128],'observed_result':observed}", "input_shapes"),
        ):
            with self.subTest(field=expected), self.assertRaisesRegex(ValueError, expected):
                validate_script(prefix + "x = torch.ones(32, 128)\nresult = " + report)
        valid = (prefix + "x = torch.ones(32, 128)\n"
                 "result = {'case_id':'C','reached':True,'framework_version':torch.__version__,"
                 "'device':str(x.device),'input_shapes':list(x.shape),'observed_result':observed}")
        self.assertEqual(validate_script(valid), valid)

    def test_execution_requires_success_record_and_captures_failure(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                relative_root = Path(os.path.relpath(root, Path.cwd()))
                good = await execute_script(
                    "import json\nprint(json.dumps({'reached': True}))\n",
                    relative_root, 5
                )
                bad = await execute_script("raise RuntimeError('broken')\n", root, 5)
                self.assertIsNone(good["error"])
                self.assertIn("broken", bad["stderr"])
                self.assertIn("status", bad["error"])
        asyncio.run(run())

    def test_generate_only_reuses_hash_verified_passing_eval_script(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old_script = root / "old" / "repro.py"
                old_script.parent.mkdir()
                script = "import json\nprint(json.dumps({'reached': True}))\n"
                old_script.write_text(script, encoding="utf-8")
                digest = __import__("hashlib").sha256(script.encode()).hexdigest()
                db = root / "cases.sqlite"
                with sqlite3.connect(db) as conn:
                    from tools.fixevalloop import setup
                    setup(conn)
                    conn.execute(
                        "INSERT INTO case_eval_state(case_id,status,script_path,script_sha256) VALUES(?,?,?,?)",
                        ("CASE_reuse", "passed", str(old_script), digest))
                    conn.execute(
                        "INSERT INTO case_eval_runs(case_id,iteration,script_path,script_sha256,exit_code,status,stdout,stderr,error) VALUES(?,?,?,?,?,?,?,?,?)",
                        ("CASE_reuse", 0, str(old_script), digest, 0, "passed", "{}", "", None))
                case = {"case_id": "CASE_reuse"}

                with patch("tools.fixevalloop.model_script", side_effect=AssertionError("must reuse")):
                    result = await generate_repro_case(
                        db, root / "sources", root / "generated", case)

                copied = root / "generated" / "CASE_reuse" / "repro.py"
                self.assertEqual(result["status"], "reused_eval_passed")
                self.assertTrue(copied.is_file())
                self.assertEqual(copied.read_text(encoding="utf-8"), script)
                with sqlite3.connect(db) as conn:
                    self.assertEqual(conn.execute(
                        "SELECT status FROM case_repro_scripts WHERE case_id='CASE_reuse'"
                    ).fetchone()[0], "reused_eval_passed")
        asyncio.run(run())

    def test_generate_only_saves_ast_valid_script_without_running_or_touching_eval_tables(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "sources" / "pytorch" / "x.py"
                source.parent.mkdir(parents=True)
                source.write_text("class Example: pass\n")
                db = root / "cases.sqlite"
                case = {
                    "case_id": "CASE_generate", "source_repository": "pytorch",
                    "source_file": "x.py", "source_lines": "1-1",
                    "applicability_predicate": "enabled",
                }

                async def fake_model(payload, prompt):
                    self.assertEqual(prompt, "agent_case_repro")
                    self.assertEqual(payload["source"]["path"], "x.py")
                    return "import json\nprint(json.dumps({'case_id': 'CASE_generate'}))\n"

                with patch("tools.fixevalloop.model_script", side_effect=fake_model), \
                     patch("tools.fixevalloop.execute_script", side_effect=AssertionError("must not execute")):
                    result = await generate_repro_case(
                        db, root / "sources", root / "generated", case)

                script_path = root / "generated" / "CASE_generate" / "repro.py"
                self.assertEqual(result["status"], "generated_ast_valid")
                self.assertEqual(Path(result["script_path"]), script_path)
                self.assertTrue(script_path.is_file())
                with sqlite3.connect(db) as conn:
                    tables = {row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")}
                    self.assertEqual(tables, {"case_repro_scripts"})
                    row = conn.execute(
                        "SELECT status,script_sha256 FROM case_repro_scripts WHERE case_id='CASE_generate'"
                    ).fetchone()
                    self.assertEqual(row[0], "generated_ast_valid")
                    self.assertEqual(len(row[1]), 64)
        asyncio.run(run())

    def test_failed_generation_retries_with_previous_error_and_script(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "sources" / "pytorch" / "x.py"
                source.parent.mkdir(parents=True)
                source.write_text("class Example: pass\n")
                db = root / "cases.sqlite"
                case = {
                    "case_id": "CASE_retry", "source_repository": "pytorch",
                    "source_file": "x.py", "source_lines": "1-1",
                    "applicability_predicate": "enabled",
                }
                from tools.fixevalloop import setup_repro_generation
                with sqlite3.connect(db) as conn:
                    setup_repro_generation(conn)
                    conn.execute("""INSERT INTO case_repro_scripts
                        (case_id,status,error,previous_script,attempts)
                        VALUES('CASE_retry','failed','SyntaxError: invalid syntax','broken python',1)""")
                observed = {}

                async def corrected(payload, prompt):
                    observed.update(payload)
                    return "import json\nprint(json.dumps({'reached': False}))\n"

                with patch("tools.fixevalloop.model_script", side_effect=corrected):
                    result = await generate_repro_case(
                        db, root / "sources", root / "generated",
                        {**case, "_repro_error": "SyntaxError: invalid syntax",
                         "_previous_script": "broken python"})
                self.assertEqual(result["status"], "generated_ast_valid")
                self.assertEqual(observed["last_execution"], "SyntaxError: invalid syntax")
                self.assertEqual(observed["previous_script"], "broken python")
                with sqlite3.connect(db) as conn:
                    self.assertEqual(conn.execute(
                        "SELECT status,attempts,error,previous_script FROM case_repro_scripts WHERE case_id='CASE_retry'"
                    ).fetchone(), ("generated_ast_valid", 2, None, None))
        asyncio.run(run())

    def test_nemotron_feedback_repairs_ast_rejected_repro_without_execution(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "sources" / "pytorch" / "x.py"
                source.parent.mkdir(parents=True)
                source.write_text("class Example: pass\n")
                case = {"case_id": "CASE_supervised", "source_repository": "pytorch",
                        "source_file": "x.py", "source_lines": "1-1",
                        "applicability_predicate": "enabled"}
                scripts = iter(("import sys\nsys.exit(1)\n", "import os\n", "open('x', 'w')\n", "raise RuntimeError('blocked')\n"))
                prompts = []

                async def fake_model(payload, prompt):
                    prompts.append((prompt, payload))
                    return next(scripts)

                with patch("tools.fixevalloop.model_script", side_effect=fake_model), \
                     patch("optimization_explorer.llm.call_model", new=AsyncMock(side_effect=[
                         {"correction": "Do not import sys; raise RuntimeError directly."},
                         {"correction": "Do not import os; raise RuntimeError directly."},
                         {"correction": "Do not use open; raise RuntimeError directly."},
                     ])) as supervisor, \
                     patch("tools.fixevalloop.execute_script",
                           side_effect=AssertionError("generation must not execute scripts")):
                    result = await generate_repro_case(
                        root / "cases.sqlite", root / "sources", root / "generated", case)

                self.assertEqual(result["status"], "generated_ast_valid")
                self.assertEqual([prompt for prompt, _ in prompts],
                                 ["agent_case_repro", "agent_case_fix", "agent_case_fix", "agent_case_fix"])
                self.assertEqual(supervisor.await_count, 3)
                feedback = [call.args[1] for call in supervisor.await_args_list]
                self.assertIn("import sys", feedback[0]["worker_script"])
                self.assertIn("outside allowed", feedback[0]["validation_error"])
                self.assertIn("import os", feedback[1]["worker_script"])
                self.assertIn("Do not import sys", prompts[1][1]["last_execution"]["supervisor_correction"])
                self.assertIn("Do not import os", prompts[2][1]["last_execution"]["supervisor_correction"])
                self.assertIn("Do not use open", prompts[3][1]["last_execution"]["supervisor_correction"])
        asyncio.run(run())

    def test_fix_worker_gets_failure_and_success_is_persisted(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "sources" / "pytorch" / "x.py"
                source.parent.mkdir(parents=True)
                source.write_text("class Example: pass\n")
                db = root / "cases.sqlite"
                case = {
                    "case_id": "CASE_test", "source_repository": "pytorch",
                    "source_file": "x.py", "source_lines": "1-1",
                    "applicability_predicate": "enabled",
                }
                responses = iter((
                    "raise RuntimeError('first attempt')\n",
                    "import json\nvalue = 1\n"
                    "runtime_version = str(type(json).__name__)\n"
                    "runtime_device = str(type(None))\n"
                    "input_shape = list((32, 128))\n"
                    "observed = json.loads(json.dumps({'value': value}))\n"
                    "print(json.dumps({'case_id':'CASE_test','reached':True,"
                    "'predicate':'enabled','framework_version':runtime_version,"
                    "'device':runtime_device,'input_shapes':input_shape,"
                    "'observed_result':observed}))\n",
                ))
                prompts = []

                async def fake_model(payload, prompt):
                    prompts.append((prompt, payload.get("last_execution")))
                    return next(responses)

                with patch("tools.fixevalloop.model_script", side_effect=fake_model):
                    result = await run_case(
                        db, root / "sources", root / "repros", case,
                        asyncio.Semaphore(1), 1, 5, False,
                    )
                self.assertEqual(result["status"], "passed")
                self.assertEqual([row[0] for row in prompts],
                                 ["agent_case_repro", "agent_case_fix"])
                self.assertIn("first attempt", prompts[1][1]["stderr"])
                with sqlite3.connect(db) as conn:
                    self.assertEqual(conn.execute(
                        "SELECT status FROM case_eval_state WHERE case_id='CASE_test'"
                    ).fetchone()[0], "passed")
                    self.assertEqual(conn.execute(
                        "SELECT status FROM case_eval_runs ORDER BY iteration"
                    ).fetchall(), [("failed",), ("passed",)])
                self.assertTrue((root / "repros" / "CASE_test" / "repro.py").is_file())
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()


class ExecuteScriptCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_script_cancellation_terminates_child(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from tools.fixevalloop import execute_script
        with tempfile.TemporaryDirectory() as tmp:
            task = asyncio.create_task(execute_script(
                "import time\ntime.sleep(60)\n", Path(tmp), 120))
            await asyncio.sleep(0.2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
