#!/usr/bin/env python3
"""Run and monitor the cooperating DFS and fix/eval workers."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request

_STOP = False


def memory_snapshot(path=Path("/proc/meminfo")):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.split()[0]) * 1024
    total, available = values["MemTotal"], values["MemAvailable"]
    return {"total_bytes": total, "available_bytes": available,
            "used_percent": round(100 * (total - available) / total, 2)}


def endpoints():
    urls = {
        "planner": "http://127.0.0.1:18327/v1/models",
        "worker": "https://ai-gateway.kilo.ai/api/v1/models",
    }
    result = {}
    for name, url in urls.items():
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                result[name] = response.status == 200
        except Exception:
            result[name] = False
    return result


def queues(path):
    result = {"trace": {}, "eval": {}, "reprogen": {},
              "reprogen_pending": 0, "errors_10m": 0}
    with sqlite3.connect(path, timeout=3) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "trace_tasks" in tables:
            result["trace"] = dict(db.execute(
                "SELECT state,COUNT(*) FROM trace_tasks GROUP BY state"))
        if "case_eval_state" in tables:
            result["eval"] = dict(db.execute(
                "SELECT status,COUNT(*) FROM case_eval_state GROUP BY status"))
        if "case_repro_scripts" in tables:
            result["reprogen"] = dict(db.execute(
                "SELECT status,COUNT(*) FROM case_repro_scripts GROUP BY status"))
            if "optimization_cases" in tables:
                repro_columns = {row[1] for row in db.execute(
                    "PRAGMA table_info(case_repro_scripts)")}
                failed_clause = (" OR (r.status='failed' AND r.attempts<3)"
                                 if "attempts" in repro_columns else
                                 " OR r.status='failed'")
                result["reprogen_pending"] = db.execute(
                    "SELECT COUNT(*) FROM optimization_cases o "
                    "LEFT JOIN case_repro_scripts r ON r.case_id=o.case_id "
                    "WHERE r.case_id IS NULL OR (r.status='running' "
                    "AND COALESCE(r.claimed_at,0)<unixepoch()-3600)" + failed_clause
                ).fetchone()[0]
        if "trace_task_errors" in tables:
            result["errors_10m"] = db.execute(
                "SELECT COUNT(*) FROM trace_task_errors "
                "WHERE created_at >= datetime('now','-10 minutes')").fetchone()[0]
    return result


def roles_to_start(roles, children, seed_pending=False):
    available = [role for role in roles if role not in children]
    if seed_pending:
        return ["trace"] if "trace" in available else []
    return available


def recover_stopped_worker(path, role):
    recovered = 0
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("BEGIN IMMEDIATE")
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if role == "trace" and {"trace_tasks", "trace_leases"} <= tables:
            recovered = db.execute(
                "SELECT COUNT(*) FROM trace_leases WHERE task_id IN "
                "(SELECT id FROM trace_tasks WHERE state='running')").fetchone()[0]
            db.execute(
                "UPDATE trace_tasks SET state='pending',error='supervisor recovered stopped worker' "
                "WHERE state='running' AND id IN (SELECT task_id FROM trace_leases)")
            db.execute("DELETE FROM trace_leases")
        elif role == "fixeval" and "case_eval_state" in tables:
            recovered = db.execute(
                "SELECT COUNT(*) FROM case_eval_state WHERE status='running'").fetchone()[0]
            db.execute(
                "UPDATE case_eval_state SET status='pending',claim=NULL,claimed_at=NULL,"
                "last_error='supervisor recovered stopped worker',updated_at=CURRENT_TIMESTAMP "
                "WHERE status='running'")
        elif role == "reprogen" and "case_repro_scripts" in tables:
            recovered = db.execute(
                "SELECT COUNT(*) FROM case_repro_scripts WHERE status='running'").fetchone()[0]
            db.execute(
                "UPDATE case_repro_scripts SET status='failed',claim=NULL,claimed_at=NULL,"
                "error='supervisor recovered interrupted generation',updated_at=CURRENT_TIMESTAMP "
                "WHERE status='running'")
        db.commit()
    return recovered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.getenv("OE_DATA_ROOT", "results"))
    p.add_argument("--db", type=Path, default=data_root / "multilayer-live.sqlite")
    p.add_argument("--sources", type=Path, default=Path(os.getenv("OE_SOURCES_ROOT", "sources")))
    p.add_argument("--out", type=Path, default=data_root / "case-repros")
    p.add_argument("--log-dir", type=Path, default=data_root / "agent-supervisor")
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--duration-seconds", type=int, default=0)
    p.add_argument("--trace-concurrency", type=int, default=2)
    p.add_argument("--eval-concurrency", type=int, default=3)
    p.add_argument("--trace-limit", type=int, default=8)
    p.add_argument("--discover-layers", action="store_true")
    p.add_argument("--seed-branch-cases", action="store_true",
                   help="back up the database and seed source-only branch candidates")
    p.add_argument("--trace-only", action="store_true",
                   help="run DFS tracing without starting the fix/eval worker")
    p.add_argument("--generate-repros", action="store_true",
                   help="run a dedicated generate-only repro worker alongside DFS; evaluation stays disabled")
    p.add_argument("--repro-limit", type=int, default=1,
                   help="maximum repro cases per generator polling batch")
    p.add_argument("--max-used-percent", type=float, default=90)
    p.add_argument("--min-available-gib", type=float, default=16)
    a = p.parse_args()
    if a.interval < 1 or a.duration_seconds < 0 or a.trace_limit < 1 or a.repro_limit < 1:
        p.error("invalid interval, duration, or trace limit")
    if a.generate_repros and not a.trace_only:
        p.error("--generate-repros requires --trace-only so evaluation remains disabled")
    if not 1 <= a.trace_concurrency <= 4 or not 1 <= a.eval_concurrency <= 4:
        p.error("concurrency must be between 1 and 4")
    a.out.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key, value in {
        "ORCHESTRATOR_BASE_URL": "http://127.0.0.1:18327/v1",
        "ORCHESTRATOR_MODEL": "openai/conductor",
        "ORCHESTRATOR_API_KEY": "local-server",
        "OPENKILO_BASE_URL": "https://ai-gateway.kilo.ai/api/v1",
        "OPENKILO_MODEL": "openai/kilo-auto/free",
        "OE_EVAL_SSH_HOST": "jlab-wsl",
    }.items():
        env.setdefault(key, value)
    trace_command = [sys.executable, "tools/trace_orchestrator.py", "--db", str(a.db),
                     "--sources", str(a.sources), "--execute",
                     "--poll-seconds", str(a.interval), "--concurrency", str(a.trace_concurrency),
                     "--max-attempts", "4", "--limit", str(a.trace_limit)]
    if not a.generate_repros:
        trace_command.extend(("--watch", "--retry-failed"))
    roles = {
        "trace": trace_command,
        "fixeval": [sys.executable, "tools/fixevalloop.py", "--db", str(a.db),
                    "--sources", str(a.sources), "--out", str(a.out), "--watch", "--retry",
                    "--poll-seconds", str(a.interval), "--concurrency", str(a.eval_concurrency),
                    "--max-fixes", "3", "--timeout", "180"],
    }
    if a.trace_only:
        roles.pop("fixeval")
    if a.generate_repros:
        roles["reprogen"] = [sys.executable, "tools/fixevalloop.py", "--db", str(a.db),
                              "--sources", str(a.sources), "--out",
                              str(data_root / "generated-repros"), "--generate-only",
                              "--watch", "--poll-seconds", str(a.interval),
                              "--limit", str(a.repro_limit)]
    children, retries, next_start = {}, {k: 2 for k in roles}, {k: 0 for k in roles}
    discover_pending, seed_pending = a.discover_layers, a.seed_branch_cases
    discover_inflight = False
    seed_inflight = False
    def stop(_sig, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock_handle = open(a.log_dir / "supervisor.lock", "a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another supervisor already owns this log directory")
    events = a.log_dir / "events.jsonl"
    handles = {}
    started = time.monotonic()
    try:
        while not _STOP:
            mem = memory_snapshot()
            pressure = (mem["used_percent"] >= a.max_used_percent or
                        mem["available_bytes"] < a.min_available_gib * 1024**3)
            health = endpoints()
            try:
                queue, db_error = queues(a.db), None
            except Exception as exc:
                queue, db_error = {}, f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            event = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "memory": mem, "memory_pressure": pressure, "endpoints": health,
                     "queue": queue, "database_error": db_error, "workers": {},
                     "worker_exits": [], "started": []}
            for role, child in list(children.items()):
                code = child.poll()
                if code is None:
                    event["workers"][role] = {"pid": child.pid, "exit_code": None}
                else:
                    event["workers"][role] = {"pid": child.pid, "exit_code": code}
                    event["worker_exits"].append({"role": role, "pid": child.pid, "exit_code": code})
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    children.pop(role)
                    handle = handles.pop(role, None)
                    if handle:
                        handle.close()
                    if role == "trace" and seed_inflight:
                        seed_inflight = False
                        if code == 0:
                            seed_pending = False
                    if role == "trace" and discover_inflight:
                        discover_inflight = False
                        if code != 0:
                            discover_pending = True
                    next_start[role] = now + retries[role]
                    event.setdefault("recovered_claims", {})[role] = recover_stopped_worker(a.db, role)
                    retries[role] = min(retries[role] * 2, 120)
            ready = health["planner"] and health["worker"]
            if not pressure and ready and not db_error:
                for role in roles_to_start(roles, children, seed_pending):
                    if now >= next_start[role]:
                        handle = open(a.log_dir / f"{role}.log", "ab", buffering=0)
                        handles[role] = handle
                        command = list(roles[role])
                        if role == "trace":
                            if discover_pending:
                                command.append("--discover-layers")
                                discover_inflight = True
                            if seed_pending:
                                command.append("--seed-branch-cases")
                                seed_inflight = True
                        child = subprocess.Popen(command,
                            cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        children[role] = child
                        if role == "trace":
                            discover_pending = False
                        retries[role] = 2
                        event["workers"][role] = {"pid": child.pid, "exit_code": None}
                        event["started"].append(role)
            event["waiting_for"] = ("memory" if pressure else
                "endpoints" if not ready else
                "database" if db_error else None)
            with events.open("a") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if a.duration_seconds and now - started >= a.duration_seconds:
                break
            time.sleep(a.interval)
    finally:
        for child in children.values():
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 20
        for role, child in children.items():
            try:
                child.wait(timeout=max(0, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            recover_stopped_worker(a.db, role)
        for handle in handles.values():
            handle.close()
        lock_handle.close()


if __name__ == "__main__":
    main()
