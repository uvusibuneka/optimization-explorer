#!/usr/bin/env python3
"""Generate, run, and repair one executable Python repro per source case."""
from __future__ import annotations
import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from optimization_explorer.schema import COLUMNS

MAX_SCRIPT_BYTES = 64000
MAX_OUTPUT_BYTES = 32000


def setup(db):
    db.execute("""CREATE TABLE IF NOT EXISTS case_eval_state (
        case_id TEXT PRIMARY KEY, status TEXT NOT NULL, claim TEXT, claimed_at INTEGER,
        iteration INTEGER NOT NULL DEFAULT 0, script_path TEXT, script_sha256 TEXT,
        last_error TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    db.execute("""CREATE TABLE IF NOT EXISTS case_eval_runs (
        case_id TEXT NOT NULL, iteration INTEGER NOT NULL, script_path TEXT NOT NULL,
        script_sha256 TEXT NOT NULL, exit_code INTEGER, status TEXT NOT NULL,
        stdout TEXT NOT NULL, stderr TEXT NOT NULL, error TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(case_id, iteration))""")


def read_cases(db_path, limit, initialize_eval=True):
    with sqlite3.connect(db_path, timeout=30) as db:
        if initialize_eval:
            setup(db)
        rows = db.execute("SELECT " + ",".join(f'"{c}"' for c in COLUMNS)
                          + " FROM optimization_cases ORDER BY case_id").fetchall()
    return [dict(zip(COLUMNS, row)) for row in rows[:limit]]


def read_repro_cases(db_path, limit, retry=False):
    """Return the next cases that do not yet have an accepted repro artifact."""
    with sqlite3.connect(db_path, timeout=30) as db:
        setup_repro_generation(db)
        rows = db.execute(
            "SELECT " + ",".join(f'o."{c}"' for c in COLUMNS) + ",r.error,r.previous_script "
            "FROM optimization_cases o LEFT JOIN case_repro_scripts r "
            "ON r.case_id=o.case_id WHERE r.case_id IS NULL "
            "OR (r.status='failed' AND (r.attempts<3 OR ?)) "
            "OR (r.status='running' AND COALESCE(r.claimed_at,0)<unixepoch()-3600) "
            "ORDER BY CASE WHEN r.status='failed' THEN 0 "
            "WHEN r.status='running' THEN 1 ELSE 2 END,o.case_id LIMIT ?",
            (int(retry), limit)
        ).fetchall()
    cases = []
    for row in rows:
        case = dict(zip(COLUMNS, row[:len(COLUMNS)]))
        case["_repro_error"], case["_previous_script"] = row[-2:]
        cases.append(case)
    return cases


def setup_repro_generation(db):
    db.execute("""CREATE TABLE IF NOT EXISTS case_repro_scripts (
        case_id TEXT PRIMARY KEY, status TEXT NOT NULL, claim TEXT, claimed_at INTEGER,
        script_path TEXT, script_sha256 TEXT, error TEXT, previous_script TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    columns = {row[1] for row in db.execute("PRAGMA table_info(case_repro_scripts)")}
    if "previous_script" not in columns:
        db.execute("ALTER TABLE case_repro_scripts ADD COLUMN previous_script TEXT")
    if "attempts" not in columns:
        db.execute("ALTER TABLE case_repro_scripts ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")


def claim_repro_generation(db_path, case_id, token, retry):
    with sqlite3.connect(db_path, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        setup_repro_generation(db)
        row = db.execute(
            "SELECT status,claimed_at,attempts FROM case_repro_scripts WHERE case_id=?", (case_id,)
        ).fetchone()
        if row:
            status, claimed_at, attempts = row
            expired = status == "running" and (claimed_at or 0) < int(time.time()) - 3600
            if status in {"generated_ast_valid", "reused_eval_passed"} or (status == "running" and not expired):
                db.rollback()
                return False
            if status == "failed" and attempts >= 3 and not retry:
                db.rollback()
                return False
        db.execute("""INSERT INTO case_repro_scripts
            (case_id,status,claim,claimed_at,attempts) VALUES(?, 'running', ?, unixepoch(), 1)
            ON CONFLICT(case_id) DO UPDATE SET status='running',claim=excluded.claim,
              claimed_at=excluded.claimed_at,attempts=case_repro_scripts.attempts+1,
              updated_at=CURRENT_TIMESTAMP""",
            (case_id, token))
        db.commit()
        return True


def reuse_passing_eval_script(db_path, case_id, out_root, claim):
    try:
        with sqlite3.connect(db_path, timeout=30) as db:
            row = db.execute("""SELECT s.script_path,s.script_sha256
                FROM case_eval_state s
                JOIN case_eval_runs r ON r.case_id=s.case_id
                  AND r.script_sha256=s.script_sha256
                WHERE s.case_id=? AND s.status='passed'
                  AND r.status='passed' AND r.exit_code=0
                ORDER BY r.iteration DESC LIMIT 1""", (case_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or not row[0] or not row[1]:
        return None
    source_path = Path(row[0])
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if source_path.is_symlink() or not source_path.is_file():
        return None
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row[1]:
        return None
    try:
        script = validate_script(raw.decode("utf-8"))
    except (UnicodeError, ValueError, SyntaxError):
        return None
    target = case_dir(out_root, case_id) / "repro.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".py.tmp")
    temporary.write_text(script, encoding="utf-8")
    temporary.replace(target)
    with sqlite3.connect(db_path, timeout=30) as db:
        db.execute("""UPDATE case_repro_scripts SET status='reused_eval_passed',
            claim=NULL,claimed_at=NULL,script_path=?,script_sha256=?,error=NULL,previous_script=NULL,
            updated_at=CURRENT_TIMESTAMP WHERE case_id=? AND claim=?""",
            (str(target), row[1], case_id, claim))
    return target


async def generate_repro_case(db_path, sources, out_root, case, retry=False):
    configured_root = os.getenv("OE_DATA_ROOT")
    if configured_root:
        root = Path(configured_root).resolve()
        if not db_path.resolve().is_relative_to(root) or not out_root.resolve().is_relative_to(root):
            raise ValueError(f"database and artifact paths must be under OE_DATA_ROOT ({root})")
    case_id, token = case["case_id"], uuid.uuid4().hex
    if not claim_repro_generation(db_path, case_id, token, retry):
        return {"case_id": case_id, "status": "skipped"}
    reused = reuse_passing_eval_script(db_path, case_id, out_root, token)
    if reused:
        return {"case_id": case_id, "status": "reused_eval_passed", "script_path": str(reused), "error": None}
    script_path = case_dir(out_root, case_id) / "repro.py"
    raw_script = None
    try:
        source = source_context(sources, case)
        prompt_case = {key:value for key,value in case.items() if not key.startswith("_")}
        from optimization_explorer.llm import call_model
        previous = case.get("_previous_script")
        failure = case.get("_repro_error")
        raw_script = await model_script(prompt_payload(prompt_case, source, previous, failure),
                                        "agent_case_repro")
        for _ in range(3):
            try:
                script = validate_script(raw_script)
                break
            except (ValueError, SyntaxError) as exc:
                correction = await call_model("supervisor", {
                    "case": prompt_case, "source": source, "worker_script": raw_script,
                    "validation_error": str(exc),
                }, "agent_case_supervisor")
                failure = {"error": str(exc), "supervisor_correction": correction["correction"]}
                raw_script = await model_script(
                    prompt_payload(prompt_case, source, raw_script, failure), "agent_case_fix")
        else:
            script = validate_script(raw_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = script_path.with_suffix(".py.tmp")
        temporary.write_text(script, encoding="utf-8")
        temporary.replace(script_path)
        digest = hashlib.sha256(script.encode()).hexdigest()
        status, error = "generated_ast_valid", None
    except asyncio.CancelledError:
        with sqlite3.connect(db_path, timeout=30) as db:
            db.execute("""UPDATE case_repro_scripts SET status='failed',claim=NULL,
                claimed_at=NULL,error=?,updated_at=CURRENT_TIMESTAMP
                WHERE case_id=? AND claim=?""", ("generation interrupted", case_id, token))
        raise
    except Exception as exc:
        status, error, digest = "failed", f"{type(exc).__name__}: {exc}"[:2000], None
        script_path = None
    with sqlite3.connect(db_path, timeout=30) as db:
        db.execute("""UPDATE case_repro_scripts SET status=?,claim=NULL,claimed_at=NULL,
            script_path=?,script_sha256=?,error=?,previous_script=?,updated_at=CURRENT_TIMESTAMP
            WHERE case_id=? AND claim=?""",
            (status, str(script_path) if script_path else None, digest, error,
             None if status == "generated_ast_valid" else
             raw_script[:MAX_SCRIPT_BYTES] if isinstance(raw_script, str) else case.get("_previous_script"),
             case_id, token))
    return {"case_id": case_id, "status": status, "script_path": str(script_path) if script_path else None,
            "error": error}


async def generate_repros(db_path, sources, out_root, cases, retry=False):
    results = []
    for case in cases:
        results.append(await generate_repro_case(db_path, sources, out_root, case, retry))
    return results


def source_context(sources, case):
    framework, relative = case["source_repository"], case["source_file"]
    if framework not in {"pytorch", "keras", "tensorflow"}:
        raise ValueError(f"unsupported source repository: {framework}")
    base = (sources / framework).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.is_symlink():
        raise ValueError(f"case source is missing or outside source tree: {relative}")
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    try:
        first, last = (int(v) for v in case["source_lines"].split("-", 1))
    except (ValueError, AttributeError):
        first, last = 1, min(len(lines), 80)
    first, last = max(1, first - 12), min(len(lines), max(first, last) + 24)
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "excerpt": "\n".join(f"{i}: {lines[i-1]}" for i in range(first, last + 1))}


def validate_script(script):
    if not isinstance(script, str) or not script.strip():
        raise ValueError("worker returned an empty Python script")
    if len(script.encode()) > MAX_SCRIPT_BYTES:
        raise ValueError("generated script exceeds 64 KiB")
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        fields = {
            key.value: value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if not {"case_id", "reached"}.issubset(fields):
            continue
        required = {"framework_version", "device", "input_shapes", "observed_result"}
        missing = required - set(fields)
        if missing:
            raise ValueError("repro JSON report is missing runtime fields: " + ", ".join(sorted(missing)))
        for name in ("framework_version", "device", "observed_result"):
            if isinstance(fields[name], ast.Constant):
                raise ValueError(f"repro report field {name} must be read from runtime results")
        shapes = fields["input_shapes"]
        if isinstance(shapes, (ast.List, ast.Tuple)) and all(
                isinstance(item, ast.Constant) for item in shapes.elts):
            raise ValueError("repro report input_shapes must be read from the input tensors")
        break
    allowed = {"torch", "tensorflow", "keras", "numpy", "math", "json", "time", "statistics"}
    blocked_attributes = {
        "hub", "cpp_extension", "load", "save", "load_library",
        "download_url_to_file", "load_state_dict_from_url", "system", "popen", "run",
    }
    blocked_builtins = {
        "eval", "exec", "compile", "__import__", "open",
        "setattr", "delattr", "vars", "globals", "locals", "breakpoint", "input",
        "exit", "quit",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([x.name.split(".", 1)[0] for x in node.names]
                     if isinstance(node, ast.Import) else
                     [node.module.split(".", 1)[0]] if node.module else [])
            if any(name not in allowed for name in names):
                raise ValueError("script imports outside allowed Python/Torch modules")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in blocked_builtins:
            raise ValueError(f"script uses prohibited builtin {node.func.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
                raise ValueError("getattr requires a literal safe attribute name")
            attr = node.args[1].value
            if attr.startswith("__") and attr not in {"__version__"} or attr in blocked_attributes:
                raise ValueError(f"script uses prohibited dynamic API {attr}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in {"__name__", "__version__", "__init__"}:
                raise ValueError(f"script uses prohibited dunder attribute {node.attr}")
            if node.attr in blocked_attributes:
                raise ValueError(f"script uses prohibited API {node.attr}")
    return script


def case_dir(root, case_id):
    return root / re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)[:100]


def claim_case(db_path, case_id, token, retry):
    with sqlite3.connect(db_path, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        setup(db)
        row = db.execute("SELECT status,claimed_at FROM case_eval_state WHERE case_id=?",
                         (case_id,)).fetchone()
        if row:
            status, claimed_at = row
            expired = status == "running" and (claimed_at or 0) < int(time.time()) - 3600
            if status == "passed" or (status == "running" and not expired):
                db.rollback()
                return False
            if status == "failed" and not retry:
                db.rollback()
                return False
        db.execute("""INSERT INTO case_eval_state
            (case_id,status,claim,claimed_at,iteration,last_error)
            VALUES(?, 'running', ?, unixepoch(), 0, NULL)
            ON CONFLICT(case_id) DO UPDATE SET status='running',claim=excluded.claim,
              claimed_at=excluded.claimed_at,iteration=0,last_error=NULL,
              updated_at=CURRENT_TIMESTAMP""", (case_id, token))
        db.commit()
        return True


def prompt_payload(case, source, previous, failure):
    return {"case": case, "source": source, "previous_script": previous,
            "last_execution": failure,
            "requirement": (
                'Return JSON {"script":"complete Python source"}. Create a standalone '
                "repro importing the real framework API, constructing inputs that satisfy "
                "the recorded predicate, and invoking the real layer/operator to exercise "
                "that source path. Print one final JSON line with case_id, reached=true "
                "only after invocation, predicate, framework_version, device, input_shapes, "
                "and observed result. Never claim a path was reached by evaluating only its "
                "predicate. Import only approved modules: torch, tensorflow, keras, numpy, "
                "math, json, time, statistics. No files, network, "
                "subprocess, shell commands, or secrets.")}


async def model_script(payload, prompt):
    from optimization_explorer.llm import call_model
    result = await call_model("worker", payload, prompt)
    return result.get("script") if isinstance(result, dict) else None


def interpreter_for_framework(framework):
    if framework == "tensorflow":
        default = "/opt/conda/envs/base-tf/bin/python"
        configured = os.environ.get("OE_TENSORFLOW_PYTHON", default)
    elif framework == "keras":
        default = "/opt/conda/envs/base-tf/bin/python"
        configured = os.environ.get("OE_KERAS_PYTHON", default)
    else:
        configured = sys.executable
    return configured


async def execute_script(script, directory, timeout, case_id=None, predicate=None,
                         python_executable=None):
    directory = directory.resolve()
    path = directory / "repro.py"
    path.write_text(script, encoding="utf-8")
    child_env = os.environ.copy()
    remote_host = child_env.get("OE_EVAL_SSH_HOST")
    if remote_host:
        remote_python = (python_executable if python_executable and python_executable.startswith("/opt/conda/") else "/home/jovyan/data-storage/llm/optimization-explorer/app/.venv/bin/python")
        command = ["ssh", "-o", "BatchMode=yes", remote_host,
                   f"cd /home/jovyan/data-storage/llm/optimization-explorer/app && {json.dumps(remote_python)} -I -"]
        child_env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    else:
        command = [python_executable or sys.executable, "-I", str(path)]
    proc = await asyncio.create_subprocess_exec(
        *command, cwd=directory, env=child_env,
        stdin=asyncio.subprocess.PIPE if remote_host else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(script.encode() if remote_host else None), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        return {"exit_code": None, "stdout": stdout.decode("utf-8", "replace")[-MAX_OUTPUT_BYTES:],
                "stderr": stderr.decode("utf-8", "replace")[-MAX_OUTPUT_BYTES:],
                "error": f"repro exceeded {timeout}s timeout"}
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        raise
    out, err = stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")
    if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
        error = "output limit exceeded"
    elif proc.returncode:
        error = f"repro exited with status {proc.returncode}"
    else:
        try:
            record = json.loads([x for x in out.splitlines() if x.strip()][-1])
            if not isinstance(record, dict) or record.get("reached") is not True:
                error = "repro did not confirm the real operation was invoked"
            elif case_id is not None and record.get("case_id") != case_id:
                error = "repro JSON case_id does not match the queued case"
            elif predicate is not None and record.get("predicate") != predicate:
                error = "repro JSON predicate does not match the queued case"
            elif case_id is not None and not all(
                    key in record for key in ("framework_version", "device", "input_shapes")):
                error = "repro JSON lacks framework/device/input shape evidence"
            elif case_id is not None and not any(
                    key in record for key in ("observed", "observed_result")):
                error = "repro JSON lacks an observed operation result"
            else:
                error = None
        except (IndexError, json.JSONDecodeError):
            error = "repro did not print a final JSON record"
    return {"exit_code": proc.returncode, "stdout": out[-MAX_OUTPUT_BYTES:],
            "stderr": err[-MAX_OUTPUT_BYTES:], "error": error}


async def run_case(db_path, sources, out_root, case, semaphore, max_fixes, timeout, retry):
    configured_root = os.getenv("OE_DATA_ROOT")
    if configured_root:
        root = Path(configured_root).resolve()
        if not db_path.resolve().is_relative_to(root) or not out_root.resolve().is_relative_to(root):
            raise ValueError(f"database and artifact paths must be under OE_DATA_ROOT ({root})")
    case_id, token = case["case_id"], uuid.uuid4().hex
    if not claim_case(db_path, case_id, token, retry):
        return {"case_id": case_id, "status": "skipped"}
    async with semaphore:
        directory = case_dir(out_root, case_id)
        directory.mkdir(parents=True, exist_ok=True)
        iteration_dir = directory / "iterations"
        iteration_dir.mkdir(exist_ok=True)
        previous, failure, final_status, iteration = None, None, "failed", -1
        try:
            source = source_context(sources, case)
            for iteration in range(max_fixes + 1):
                prompt = "agent_case_repro" if previous is None else "agent_case_fix"
                script = await model_script(prompt_payload(case, source, previous, failure), prompt)
                script_path = iteration_dir / f"{iteration:02d}.py"
                script_path.write_text(script if isinstance(script, str) else "", encoding="utf-8")
                try:
                    script = validate_script(script)
                    framework = case.get("source_repository")
                    result = await execute_script(
                        script, directory, timeout, case_id,
                        case.get("applicability_predicate"),
                        interpreter_for_framework(framework))
                except (ValueError, SyntaxError) as exc:
                    result = {"exit_code": None, "stdout": "", "stderr": "",
                              "error": f"script validation failed: {exc}"}
                digest = hashlib.sha256(script.encode()).hexdigest()
                with sqlite3.connect(db_path, timeout=30) as db:
                    db.execute("""INSERT OR REPLACE INTO case_eval_runs
                        (case_id,iteration,script_path,script_sha256,exit_code,status,stdout,stderr,error)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                        (case_id, iteration, str(script_path), digest, result["exit_code"],
                         "passed" if result["error"] is None else "failed",
                         result["stdout"], result["stderr"], result["error"]))
                    db.execute("""UPDATE case_eval_state SET iteration=?,script_path=?,
                        script_sha256=?,last_error=?,updated_at=CURRENT_TIMESTAMP
                        WHERE case_id=? AND claim=?""",
                        (iteration, str(script_path), digest, result["error"], case_id, token))
                if result["error"] is None:
                    final_status = "passed"
                    (directory / "repro.py").write_text(script, encoding="utf-8")
                    break
                previous, failure = script, result
        except asyncio.CancelledError:
            with sqlite3.connect(db_path, timeout=30) as db:
                db.execute("UPDATE case_eval_state SET status='pending',claim=NULL,claimed_at=NULL,last_error=?,updated_at=CURRENT_TIMESTAMP WHERE case_id=? AND claim=?",
                    ("worker stopped; case returned to queue", case_id, token))
            raise
        except Exception as exc:
            failure = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        with sqlite3.connect(db_path, timeout=30) as db:
            db.execute("""UPDATE case_eval_state SET status=?,claim=NULL,claimed_at=NULL,
                last_error=?,updated_at=CURRENT_TIMESTAMP WHERE case_id=? AND claim=?""",
                (final_status, failure["error"] if failure else None, case_id, token))
        return {"case_id": case_id, "status": final_status, "iterations": iteration + 1,
                "error": failure["error"] if failure else None}


async def run_batch(db_path, sources, out_root, cases, concurrency, max_fixes, timeout, retry):
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(run_case(db_path, sources, out_root, case, semaphore,
                                           max_fixes, timeout, retry) for case in cases))


async def watch(args):
    seen = set()
    while True:
        cases = read_cases(args.db, args.limit)
        fresh = [case for case in cases if case["case_id"] not in seen]
        if fresh:
            results = await run_batch(args.db, args.sources, args.out, fresh,
                                      args.concurrency, args.max_fixes, args.timeout, args.retry)
            seen.update(item["case_id"] for item in results)
            print(json.dumps({"processed": results}, ensure_ascii=False), flush=True)
        if not args.watch:
            return
        await asyncio.sleep(args.poll_seconds)


async def watch_repro_generation(args):
    while True:
        cases = read_repro_cases(args.db, args.limit, args.retry)
        if cases:
            results = await generate_repros(args.db, args.sources, args.out, cases, args.retry)
            print(json.dumps({"generated": results}, ensure_ascii=False), flush=True)
        if not args.watch:
            return
        await asyncio.sleep(args.poll_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.getenv("OE_DATA_ROOT", "results"))
    p.add_argument("--db", type=Path, default=data_root / "multilayer-live.sqlite")
    p.add_argument("--sources", type=Path, default=Path(os.getenv("OE_SOURCES_ROOT", "sources")))
    p.add_argument("--out", type=Path)
    p.add_argument("--generate-only", action="store_true",
                   help="generate and AST-validate repros without executing scripts or updating eval tables")
    p.add_argument("--limit", type=int, default=1_000_000)
    p.add_argument("--concurrency", type=int, default=1, help="OpenKilo repro workers (1..4)")
    p.add_argument("--max-fixes", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--poll-seconds", type=int, default=10)
    p.add_argument("--retry", action="store_true")
    p.add_argument("--watch", action="store_true",
                   help="poll for new work (new DFS cases in eval mode, missing repros in generate-only mode)")
    args = p.parse_args()
    if args.limit < 1 or not 1 <= args.concurrency <= 4 or args.max_fixes < 0 \
            or args.timeout < 1 or args.poll_seconds < 1:
        p.error("invalid limit, concurrency, fix count, timeout, or poll interval")
    if args.generate_only and args.concurrency != 1:
        p.error("--generate-only uses exactly one OpenKilo worker")
    if args.out is None:
        args.out = data_root / ("generated-repros" if args.generate_only else "case-repros")
    configured_root = os.getenv("OE_DATA_ROOT")
    if configured_root:
        root = Path(configured_root).resolve()
        if not args.db.resolve().is_relative_to(root) or not args.out.resolve().is_relative_to(root):
            p.error("database and artifact paths must be under OE_DATA_ROOT")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "OPENKILO_BASE_URL": "https://ai-gateway.kilo.ai/api/v1",
        "OPENKILO_MODEL": "openai/kilo-auto/free",
    }.items():
        os.environ.setdefault(name, value)
    try:
        if args.generate_only:
            asyncio.run(watch_repro_generation(args))
        else:
            asyncio.run(watch(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
