"""Schema of the experiment store, upgraded in place (Phase 7, Deliverable 7).

Each entry in `MIGRATIONS` upgrades the database by one version, tracked by
`PRAGMA user_version`. Never edit a released migration; append a new one.
"""

from __future__ import annotations

import sqlite3

BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_name TEXT,
    problem_type TEXT,
    target_column TEXT,
    meta_features_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    weakest_block_history_json TEXT NOT NULL,
    final_decision TEXT,
    final_composite_score REAL
);
"""

MIGRATIONS: list[list[str]] = [
    [
        "ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'finalized'",
        "ALTER TABLE runs ADD COLUMN error TEXT",
        "ALTER TABLE runs ADD COLUMN iteration_logs_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE runs ADD COLUMN final_metrics_json TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE runs ADD COLUMN model_path TEXT",
        "ALTER TABLE runs ADD COLUMN dataset_fingerprint TEXT",
        "ALTER TABLE runs ADD COLUMN code_version TEXT",
        "ALTER TABLE runs ADD COLUMN lib_versions_json TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE runs ADD COLUMN final_report_json TEXT",
    ],
    [  # Phase 4 + 6: MLflow parent run, and components the registry could not use.
        "ALTER TABLE runs ADD COLUMN mlflow_run_id TEXT",
        """
        CREATE TABLE IF NOT EXISTS unsupported_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            requested TEXT NOT NULL,
            category TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            alternative_used TEXT,
            semantic_change INTEGER NOT NULL,
            context_json TEXT NOT NULL
        )
        """,
    ],
    [  # Phase 7: evidence gating, and every measured attempt (not just the final model).
        # Runs stored before leakage handling existed have no report and a churn-like
        # name; the audit found those churn models leaked CustomerID, so they are
        # kept in the table but never returned as evidence.
        "ALTER TABLE runs ADD COLUMN evidence_quality TEXT NOT NULL DEFAULT 'ok'",
        """
        UPDATE runs SET evidence_quality = 'suspect_leakage'
        WHERE status = 'finalized' AND final_report_json IS NULL AND lower(dataset_name) LIKE '%churn%'
        """,
        """
        CREATE TABLE IF NOT EXISTS attempts (
            run_id TEXT NOT NULL,
            signature TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            source TEXT NOT NULL,
            model_family TEXT NOT NULL,
            primary_metric TEXT NOT NULL,
            cv_mean REAL NOT NULL,
            cv_std REAL NOT NULL,
            composite_score REAL NOT NULL,
            feasible INTEGER NOT NULL,
            latency_ms_per_1k REAL,
            params_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (run_id, signature)
        )
        """,
    ],
    [  # Web UI: the live activity log, so a finished run can be reopened with its timeline.
        "ALTER TABLE runs ADD COLUMN activity_log_json TEXT NOT NULL DEFAULT '[]'",
    ],
]
SCHEMA_VERSION = len(MIGRATIONS)


def migrate(conn: sqlite3.Connection) -> None:
    """Apply every migration newer than the database's recorded version."""
    conn.execute(BASE_SCHEMA)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for target_version in range(version + 1, SCHEMA_VERSION + 1):
        for statement in MIGRATIONS[target_version - 1]:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {target_version}")
    conn.commit()
