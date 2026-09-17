"""Small persistent job queue. SQLite connections are never shared across threads."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL, inputs TEXT NOT NULL,
                results TEXT NOT NULL DEFAULT '[]', error TEXT NOT NULL DEFAULT ''
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, job_id, inputs):
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id, status, inputs) VALUES (?, 'queued', ?)",
                       (job_id, json.dumps(inputs)))

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created, rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='processing' WHERE id=?", (row["id"],))
                job = self.decode(row)
                job["status"] = "processing"
                return job
        return None

    def update(self, job_id, status, results, error=""):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?, results=?, error=? WHERE id=?",
                       (status, json.dumps(results), error, job_id))

    def recover(self):
        # One application process owns the worker; interrupted jobs remain reviewable.
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='interrupted', error=? WHERE status='processing'",
                       ("The server stopped during ingestion. Upload the source again to retry.",))

    @staticmethod
    def decode(row):
        result = dict(row)
        for key in ("inputs", "results"):
            result[key] = json.loads(result[key])
        return result

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self.decode(row) if row else None

    def recent(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC, rowid DESC LIMIT 20").fetchall()
        return [self.decode(row) for row in rows]


def new_id():
    return uuid4().hex
