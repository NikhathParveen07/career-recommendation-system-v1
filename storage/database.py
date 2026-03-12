"""
CounselAI Pipeline — Database Manager
Handles all SQLite storage for the pipeline
"""

import sqlite3
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_db_connection(db_path: str = "storage/counselai.db"):
    """Get database connection with row factory"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database(db_path: str = "storage/counselai.db"):
    """
    Create all tables if they do not exist.
    Safe to run multiple times.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # ── Table 1: Raw Content ──────────────────────────────────────────
    # Everything we fetch from sources before processing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            source_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            force TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            url TEXT,
            published_date TEXT,
            fetched_at TEXT NOT NULL,
            processed INTEGER DEFAULT 0,
            processing_status TEXT DEFAULT 'pending'
        )
    """)

    # Index for fast lookup of unprocessed content
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_raw_content_processed
        ON raw_content(processed, force)
    """)

    # ── Table 2: Force Signals ────────────────────────────────────────
    # Validated signals extracted from raw content
    # Pure force data — no career references
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS force_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT UNIQUE NOT NULL,
            force TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            signal_summary TEXT NOT NULL,
            key_facts TEXT NOT NULL,
            geographic_scope TEXT,
            timeline TEXT,
            magnitude TEXT,
            direction TEXT,
            evidence_quote TEXT,
            full_evidence_text TEXT,
            confidence REAL NOT NULL,
            credibility TEXT NOT NULL,
            raw_content_id INTEGER,
            extracted_at TEXT NOT NULL,
            is_superseded INTEGER DEFAULT 0,
            superseded_by TEXT,
            FOREIGN KEY (raw_content_id) REFERENCES raw_content(id)
        )
    """)

    # Index for fast signal retrieval by force
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_force_signals_force
        ON force_signals(force, is_superseded)
    """)

    # Index for confidence filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_force_signals_confidence
        ON force_signals(confidence, force)
    """)

    # ── Table 3: Pipeline Log ─────────────────────────────────────────
    # Complete audit trail of everything pipeline does
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            source_id TEXT,
            force TEXT,
            message TEXT NOT NULL,
            details TEXT,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # ── Table 4: Source Status ────────────────────────────────────────
    # Tracks health of each source
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_status (
            source_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            force TEXT NOT NULL,
            last_polled TEXT,
            last_success TEXT,
            last_error TEXT,
            error_count INTEGER DEFAULT 0,
            total_fetched INTEGER DEFAULT 0,
            total_signals INTEGER DEFAULT 0,
            is_healthy INTEGER DEFAULT 1
        )
    """)

    # ── Table 5: Duplicate Tracker ────────────────────────────────────
    # Content hashes we have already seen
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_content (
            content_hash TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            first_seen TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised successfully")
    print("✅ Database initialised")


def content_already_seen(content_hash: str,
                         db_path: str = "storage/counselai.db") -> bool:
    """Check if we have already processed this content"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM seen_content WHERE content_hash = ?",
        (content_hash,)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def mark_content_seen(content_hash: str, source_id: str,
                      db_path: str = "storage/counselai.db"):
    """Mark content as seen so we never process it twice"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO seen_content
        (content_hash, source_id, first_seen)
        VALUES (?, ?, ?)
    """, (content_hash, source_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def store_raw_content(source_id: str, source_name: str, force: str,
                      title: str, content: str, url: str = None,
                      published_date: str = None,
                      db_path: str = "storage/counselai.db") -> int | None:
    """
    Store raw content fetched from a source.
    Returns record ID if new, None if already seen.
    """
    # Create unique hash from content
    content_hash = hashlib.sha256(
        f"{source_id}:{content[:500]}".encode()
    ).hexdigest()

    # Check if already seen
    if content_already_seen(content_hash, db_path):
        return None

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO raw_content
            (content_hash, source_id, source_name, force, title,
             content, url, published_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_hash, source_id, source_name, force,
            title, content, url, published_date,
            datetime.utcnow().isoformat()
        ))
        record_id = cursor.lastrowid
        conn.commit()

        # Mark as seen
        mark_content_seen(content_hash, source_id, db_path)

        return record_id

    except sqlite3.IntegrityError:
        # Duplicate — already in database
        return None
    finally:
        conn.close()


def get_unprocessed_content(force: str = None, limit: int = 50,
                             db_path: str = "storage/counselai.db"):
    """Get raw content waiting to be processed"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    if force:
        cursor.execute("""
            SELECT * FROM raw_content
            WHERE processed = 0
            AND processing_status = 'pending'
            AND force = ?
            ORDER BY fetched_at ASC
            LIMIT ?
        """, (force, limit))
    else:
        cursor.execute("""
            SELECT * FROM raw_content
            WHERE processed = 0
            AND processing_status = 'pending'
            ORDER BY fetched_at ASC
            LIMIT ?
        """, (limit,))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def store_force_signal(signal_data: dict,
                       db_path: str = "storage/counselai.db") -> str:
    """Store a validated force signal"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Generate signal ID
    signal_id = f"{signal_data['force']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{signal_data['source_id']}"

    cursor.execute("""
        INSERT INTO force_signals
        (signal_id, force, source_id, source_name, signal_summary,
         key_facts, geographic_scope, timeline, magnitude, direction,
         evidence_quote, full_evidence_text, confidence, credibility,
         raw_content_id, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_id,
        signal_data.get('force'),
        signal_data.get('source_id'),
        signal_data.get('source_name'),
        signal_data.get('signal_summary'),
        json.dumps(signal_data.get('key_facts', [])),
        signal_data.get('geographic_scope'),
        signal_data.get('timeline'),
        signal_data.get('magnitude'),
        signal_data.get('direction'),
        signal_data.get('evidence_quote'),
        signal_data.get('full_evidence_text'),
        signal_data.get('confidence', 0.5),
        signal_data.get('credibility', 'medium'),
        signal_data.get('raw_content_id'),
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

    logger.info(f"Stored signal: {signal_id}")
    return signal_id


def update_source_status(source_id: str, source_name: str, force: str,
                         success: bool, error_msg: str = None,
                         new_items: int = 0,
                         db_path: str = "storage/counselai.db"):
    """Update health status of a source"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO source_status
        (source_id, source_name, force, last_polled,
         last_success, last_error, error_count,
         total_fetched, is_healthy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            last_polled = excluded.last_polled,
            last_success = CASE WHEN ? THEN excluded.last_polled
                           ELSE last_success END,
            last_error = CASE WHEN NOT ? THEN ?
                         ELSE last_error END,
            error_count = CASE WHEN NOT ? THEN error_count + 1
                          ELSE 0 END,
            total_fetched = total_fetched + ?,
            is_healthy = ?
    """, (
        source_id, source_name, force, now,
        now if success else None,
        error_msg if not success else None,
        0 if success else 1,
        new_items,
        1 if success else 0,
        success, success, error_msg, success, new_items, success
    ))

    conn.commit()
    conn.close()


def log_pipeline_event(event_type: str, message: str,
                       source_id: str = None, force: str = None,
                       details: dict = None, status: str = "info",
                       db_path: str = "storage/counselai.db"):
    """Log a pipeline event for audit trail"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO pipeline_log
        (event_type, source_id, force, message, details, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        event_type, source_id, force, message,
        json.dumps(details) if details else None,
        status,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()


def get_pipeline_stats(db_path: str = "storage/counselai.db") -> dict:
    """Get overall pipeline statistics"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    stats = {}

    # Total raw content
    cursor.execute("SELECT COUNT(*) as total FROM raw_content")
    stats['total_raw_content'] = cursor.fetchone()['total']

    # Pending processing
    cursor.execute(
        "SELECT COUNT(*) as total FROM raw_content WHERE processed = 0"
    )
    stats['pending_processing'] = cursor.fetchone()['total']

    # Total signals
    cursor.execute(
        "SELECT COUNT(*) as total FROM force_signals WHERE is_superseded = 0"
    )
    stats['total_active_signals'] = cursor.fetchone()['total']

    # Signals by force
    cursor.execute("""
        SELECT force, COUNT(*) as count
        FROM force_signals
        WHERE is_superseded = 0
        GROUP BY force
        ORDER BY count DESC
    """)
    stats['signals_by_force'] = {
        row['force']: row['count']
        for row in cursor.fetchall()
    }

    # Healthy sources
    cursor.execute(
        "SELECT COUNT(*) as total FROM source_status WHERE is_healthy = 1"
    )
    stats['healthy_sources'] = cursor.fetchone()['total']

    # Total sources
    cursor.execute("SELECT COUNT(*) as total FROM source_status")
    stats['total_sources_tracked'] = cursor.fetchone()['total']

    conn.close()
    return stats


def mark_content_processed(raw_content_id: int, status: str = "success",
                            db_path: str = "storage/counselai.db"):
    """Mark raw content as processed"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE raw_content
        SET processed = 1, processing_status = ?
        WHERE id = ?
    """, (status, raw_content_id))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialise_database()
    stats = get_pipeline_stats()
    print(f"Pipeline stats: {json.dumps(stats, indent=2)}")
