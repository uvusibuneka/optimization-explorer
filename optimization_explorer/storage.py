"""Single-writer SQLite store; final table always has exactly 28 columns."""
import csv
import json
import sqlite3
from itertools import islice
from pathlib import Path
from .schema import COLUMNS

class CaseConflict(ValueError):
    pass


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        cols = ",".join(f'"{k}" TEXT NOT NULL' for k in COLUMNS)
        self.conn.execute(f'CREATE TABLE IF NOT EXISTS optimization_cases ({cols}, PRIMARY KEY(case_id))')
        self.conn.execute('CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT "PENDING")')
        self.conn.execute("""CREATE INDEX IF NOT EXISTS pending_tasks_by_id
            ON tasks(task_id) WHERE status='PENDING'""")
        self.conn.execute('CREATE TABLE IF NOT EXISTS unresolved_cases (task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, errors TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS case_tasks (case_id TEXT NOT NULL, task_id TEXT NOT NULL, PRIMARY KEY(case_id, task_id))')
        self.conn.execute('CREATE TABLE IF NOT EXISTS coverage_gaps (gap_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, reason TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS binary_evidence (path TEXT PRIMARY KEY, sha256 TEXT, format TEXT, payload TEXT NOT NULL)')
        self.conn.commit()

    def add_tasks(self, tasks):
        count = 0

        def rows():
            nonlocal count
            for task in tasks:
                count += 1
                yield task["task_id"], json.dumps(task, ensure_ascii=False)

        with self.conn:
            self.conn.executemany(
                """INSERT INTO tasks(task_id,payload,status) VALUES(?, ?, 'PENDING')
                ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload
                WHERE json_extract(tasks.payload, '$.local_source_path') IS NULL
                  AND json_extract(excluded.payload, '$.local_source_path') IS NOT NULL""",
                rows(),
            )
        return count

    def pending(self, limit: int, source_only: bool = False):
        query = "SELECT payload FROM tasks WHERE status='PENDING'"
        if source_only:
            query += " AND COALESCE(json_extract(payload, '$.evidence_kind'), 'source') NOT IN ('binary', 'decompiled', 'ghidra')"
        rows = self.conn.execute(query + " ORDER BY task_id LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def source_fragments(self, repository):
        rows = self.conn.execute(
            """SELECT payload FROM tasks
               WHERE json_extract(payload, '$.repository') = ?
                 AND COALESCE(json_extract(payload, '$.evidence_kind'), 'source') = 'source'
                 AND json_type(payload, '$.context') = 'text'""",
            (repository,),
        )
        for (payload,) in rows:
            yield json.loads(payload)

    def add_indexed_tasks(self, tasks):
        iterator = iter(tasks)
        count = 0
        while batch := list(islice(iterator, 256)):
            count += self.add_tasks(batch)
        return count

    def save_case(self, case, task_id):
        self.save_cases([case], task_id)

    def save_cases(self, cases, task_id):
        count = 0
        names = ",".join('"'+k+'"' for k in COLUMNS)
        placeholders = ",".join("?" for _ in COLUMNS)
        with self.conn:
            for case in cases:
                values = tuple(case[k] for k in COLUMNS)
                inserted = self.conn.execute(
                    f'INSERT INTO optimization_cases ({names}) VALUES ({placeholders}) ON CONFLICT(case_id) DO NOTHING',
                    values,
                )
                if not inserted.rowcount:
                    existing = self.conn.execute(
                        f'SELECT {names} FROM optimization_cases WHERE case_id=?', (case["case_id"],)
                    ).fetchone()
                    if existing != values:
                        raise CaseConflict(f"conflicting case_id: {case['case_id']}")
                count += inserted.rowcount
                self.conn.execute(
                    'INSERT OR IGNORE INTO case_tasks(case_id, task_id) SELECT ?, task_id FROM tasks WHERE task_id=?',
                    (case["case_id"], task_id),
                )
            self.conn.execute('UPDATE tasks SET status="DONE" WHERE task_id=?', (task_id,))
        return count

    def unresolved(self, task_id, payload, errors):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO unresolved_cases VALUES (?, ?, ?)', (task_id, json.dumps(payload, ensure_ascii=False), json.dumps(errors, ensure_ascii=False)))
            self.conn.execute('UPDATE tasks SET status="UNRESOLVED" WHERE task_id=?', (task_id,))
            self.conn.execute('INSERT INTO coverage_gaps(task_id,reason) VALUES (?, ?)', (task_id, "; ".join(errors)[:2000]))

    def add_binary_evidence(self, records):
        with self.conn:
            self.conn.executemany('INSERT OR REPLACE INTO binary_evidence(path,sha256,format,payload) VALUES (?,?,?,?)',
                ((r.get('path','UNKNOWN'),r.get('sha256','UNKNOWN'),r.get('format','UNKNOWN'),json.dumps(r,ensure_ascii=False)) for r in records))

    def export(self, filename):
        if self.path != ":memory:":
            output = Path(filename)
            database = Path(self.path).resolve()
            protected = [database, *(Path(str(database) + suffix) for suffix in ("-wal", "-shm", "-journal"))]
            for path in protected:
                if output.resolve() == path or (
                    output.exists() and path.exists() and output.samefile(path)
                ):
                    raise ValueError("CSV output must differ from the SQLite database and its journal files")
        rows = self.conn.execute('SELECT '+','.join('"'+k+'"' for k in COLUMNS)+' FROM optimization_cases ORDER BY case_id')
        count = 0
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for row in rows:
                w.writerow(row)
                count += 1
        return count

    def stats(self):
        return {t: self.conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('tasks', 'optimization_cases', 'unresolved_cases', 'coverage_gaps', 'binary_evidence')}
