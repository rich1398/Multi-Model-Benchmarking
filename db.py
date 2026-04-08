"""SQLite storage for Occursus-Claude benchmark results."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = "results/occursus_claude.db"


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at REAL NOT NULL,
            finished_at REAL,
            status TEXT NOT NULL DEFAULT 'running',
            config_json TEXT,
            total_tasks INTEGER DEFAULT 0,
            completed_tasks INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS task_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_prompt TEXT,
            task_rubric TEXT,
            task_category TEXT,
            task_difficulty TEXT,
            pipeline_id TEXT NOT NULL,
            ok INTEGER NOT NULL DEFAULT 1,
            final_text TEXT,
            error TEXT,
            score INTEGER DEFAULT 0,
            judge_reasoning TEXT,
            judge_ok INTEGER NOT NULL DEFAULT 1,
            judge_error TEXT,
            judge_backend TEXT,
            judge_model TEXT,
            wall_ms REAL DEFAULT 0,
            llm_calls INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            steps_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_results_run ON task_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_results_pipeline ON task_results(pipeline_id);
    """)
    _ensure_column(conn, "task_results", "task_rubric", "TEXT")
    _ensure_column(conn, "task_results", "task_category", "TEXT")
    _ensure_column(conn, "task_results", "task_difficulty", "TEXT")
    _ensure_column(conn, "task_results", "judge_ok", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "task_results", "judge_error", "TEXT")
    _ensure_column(conn, "task_results", "judge_backend", "TEXT")
    _ensure_column(conn, "task_results", "judge_model", "TEXT")
    conn.commit()
    conn.close()


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


_init_db()


def _summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("ok"))
    failures = total - successes
    judge_failures = sum(1 for row in rows if not row.get("judge_ok", True))
    total_score = sum(int(row.get("score") or 0) for row in rows)
    success_scores = [int(row.get("score") or 0) for row in rows if row.get("ok")]
    average_all = round(total_score / total, 2) if total else 0.0
    average_success_only = (
        round(sum(success_scores) / len(success_scores), 2)
        if success_scores else None
    )
    return {
        "total_results": total,
        "success_count": successes,
        "failure_count": failures,
        "judge_failure_count": judge_failures,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "average_score_all_tasks": average_all,
        "average_score_success_only": average_success_only,
        "has_failures": failures > 0,
        "valid_for_thesis": (
            total > 0 and failures == 0 and judge_failures == 0 and successes >= 2
        ),
    }


def _looks_like_embedding_model(model: str) -> bool:
    name = (model or "").strip().lower()
    markers = (
        "embed",
        "embedding",
        "nomic-embed",
        "bge-",
        "e5-",
        "snowflake-arctic-embed",
    )
    return any(marker in name for marker in markers)


def _normalized_status(
    raw_status: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    status = (raw_status or "unknown").strip()
    model = str(config.get("model", "")).strip()
    failure_count = int(summary.get("failure_count") or 0)
    judge_failure_count = int(summary.get("judge_failure_count") or 0)
    success_count = int(summary.get("success_count") or 0)
    reason = ""

    if _looks_like_embedding_model(model) and failure_count > 0 and success_count == 0:
        return "invalid_config", (
            f'Configured generation model "{model}" is not suitable for chat benchmarks.'
        )

    if results:
        errors = [str(row.get("error", "")).lower() for row in results if not row.get("ok")]
        if errors and all("does not support chat" in err for err in errors):
            return "invalid_config", "All cells failed because the selected model does not support chat."

    if status == "completed" and failure_count > 0:
        status = "completed_with_failures"
    elif status == "completed" and judge_failure_count > 0:
        status = "completed_with_judge_failures"

    if status == "completed_with_failures":
        reason = "Some benchmark cells failed."
    elif status == "completed_with_judge_failures":
        reason = "Generation completed, but some judge evaluations failed."

    return status, reason


async def save_run(run_id: str, run_data: dict[str, Any]) -> None:
    def _save():
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM task_results WHERE run_id = ?", (run_id,))
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, started_at, finished_at, status, config_json,
                    total_tasks, completed_tasks)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_data.get("started_at", time.time()),
                    run_data.get("finished_at"),
                    run_data.get("status", "completed"),
                    json.dumps(run_data.get("config", {})),
                    run_data.get("total", 0),
                    run_data.get("completed", 0),
                ),
            )

            for r in run_data.get("results", []):
                conn.execute(
                    """INSERT INTO task_results
                       (run_id, task_id, task_prompt, task_rubric, task_category,
                        task_difficulty, pipeline_id, ok, final_text, error, score,
                        judge_reasoning, judge_ok, judge_error, judge_backend,
                        judge_model, wall_ms, llm_calls, total_tokens, steps_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        r.get("task_id", ""),
                        r.get("task_prompt", ""),
                        r.get("task_rubric", ""),
                        r.get("task_category", ""),
                        r.get("task_difficulty", ""),
                        r.get("pipeline_id", ""),
                        1 if r.get("ok", True) else 0,
                        r.get("final_text", ""),
                        r.get("error", ""),
                        r.get("score", 0),
                        r.get("judge_reasoning", ""),
                        1 if r.get("judge_ok", True) else 0,
                        r.get("judge_error", ""),
                        r.get("judge_backend", ""),
                        r.get("judge_model", ""),
                        r.get("wall_ms", 0),
                        r.get("llm_calls", 0),
                        r.get("total_tokens", r.get("tokens", 0)),
                        json.dumps(r.get("steps", [])),
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    await asyncio.to_thread(_save)


async def list_runs() -> list[dict[str, Any]]:
    def _list():
        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    r.*,
                    COUNT(tr.id) AS total_results,
                    SUM(CASE WHEN tr.ok = 1 THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN tr.ok = 0 THEN 1 ELSE 0 END) AS failure_count,
                    SUM(CASE WHEN tr.judge_ok = 0 THEN 1 ELSE 0 END) AS judge_failure_count,
                    AVG(CAST(tr.score AS REAL)) AS average_score_all_tasks,
                    AVG(CASE WHEN tr.ok = 1 THEN CAST(tr.score AS REAL) END) AS average_score_success_only
                FROM runs r
                LEFT JOIN task_results tr ON tr.run_id = r.run_id
                GROUP BY r.run_id
                ORDER BY r.started_at DESC
                LIMIT 50
                """
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                entry = dict(row)
                total_results = int(entry.get("total_results") or 0)
                success_count = int(entry.get("success_count") or 0)
                failure_count = int(entry.get("failure_count") or 0)
                judge_failure_count = int(entry.get("judge_failure_count") or 0)
                entry["success_rate"] = (
                    round(success_count / total_results, 4) if total_results else 0.0
                )
                entry["judge_failure_count"] = judge_failure_count
                entry["average_score_all_tasks"] = round(
                    float(entry.get("average_score_all_tasks") or 0.0), 2
                )
                avg_success = entry.get("average_score_success_only")
                entry["average_score_success_only"] = (
                    round(float(avg_success), 2)
                    if avg_success is not None else None
                )
                entry["valid_for_thesis"] = (
                    total_results > 0
                    and failure_count == 0
                    and judge_failure_count == 0
                    and success_count >= 2
                )
                config = json.loads(entry.get("config_json") or "{}")
                status, reason = _normalized_status(entry.get("status", ""), config, entry)
                entry["status"] = status
                entry["status_reason"] = reason
                out.append(entry)
            return out
        finally:
            conn.close()

    return await asyncio.to_thread(_list)


async def get_run_results(run_id: str) -> dict[str, Any]:
    def _get():
        conn = _get_conn()
        try:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

            if not run_row:
                return None

            result_rows = conn.execute(
                "SELECT * FROM task_results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()

            results = []
            for r in result_rows:
                entry = dict(r)
                entry["ok"] = bool(entry["ok"])
                entry["judge_ok"] = bool(entry.get("judge_ok", 1))
                entry["steps"] = json.loads(entry.get("steps_json", "[]"))
                entry.pop("steps_json", None)
                results.append(entry)

            summary = _summarize_results(results)
            config = json.loads(run_row["config_json"] or "{}")
            status, reason = _normalized_status(run_row["status"], config, summary, results)

            return {
                "run_id": run_id,
                "status": status,
                "status_reason": reason,
                "started_at": run_row["started_at"],
                "finished_at": run_row["finished_at"],
                "config": config,
                "total": run_row["total_tasks"],
                "completed": run_row["completed_tasks"],
                "summary": summary,
                "results": results,
            }
        finally:
            conn.close()

    return await asyncio.to_thread(_get)


async def delete_run(run_id: str) -> bool:
    def _delete():
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM task_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    return await asyncio.to_thread(_delete)


async def list_failed_results(run_id: str) -> list[dict[str, Any]]:
    def _list_failed():
        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM task_results
                WHERE run_id = ? AND ok = 0
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
            out = []
            for row in rows:
                entry = dict(row)
                entry["ok"] = bool(entry["ok"])
                entry["judge_ok"] = bool(entry.get("judge_ok", 1))
                entry["steps"] = json.loads(entry.get("steps_json") or "[]")
                entry.pop("steps_json", None)
                out.append(entry)
            return out
        finally:
            conn.close()

    return await asyncio.to_thread(_list_failed)
