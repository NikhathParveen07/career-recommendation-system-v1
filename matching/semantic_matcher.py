import json
import sqlite3
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import log_pipeline_event

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
CAREERS_DIR = Path("careers")
DB_PATH     = os.getenv("DB_PATH", "storage/counselai.db")
MODEL_NAME  = "all-MiniLM-L6-v2"  # Small, fast, free, runs on CPU
TOP_K       = 20  # Top signals per career


# ── Database Helpers ──────────────────────────────────────────────────

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables():
    """Create tables for storing matcher results"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Career trajectory scores
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS career_trajectories (
            career_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            stream TEXT NOT NULL,
            trajectory TEXT NOT NULL,
            confidence REAL NOT NULL,
            timeframe TEXT,
            signal_count INTEGER DEFAULT 0,
            top_forces TEXT,
            evidence_summary TEXT,
            last_updated TEXT NOT NULL
        )
    """)

    # Career-signal connections
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS career_signal_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            career_id TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            force TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_connections_career
        ON career_signal_connections(career_id)
    """)

    conn.commit()
    conn.close()


# ── Data Loaders ──────────────────────────────────────────────────────

def load_all_careers():
    """Load all 120 career JSON files"""
    careers = []

    for json_file in sorted(CAREERS_DIR.rglob("*.json")):
        if json_file.name == "career_index.json":
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                career = json.load(f)
            careers.append(career)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(careers)} careers")
    return careers


def load_all_signals():
    """Load all active force signals from database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT signal_id, force, source_name,
               signal_summary, key_facts,
               magnitude, direction, confidence
        FROM force_signals
        WHERE is_superseded = 0
        AND confidence >= 0.6
        ORDER BY extracted_at DESC
    """)

    signals = []
    for row in cursor.fetchall():
        signal = dict(row)
        try:
            signal["key_facts"] = json.loads(
                signal.get("key_facts", "[]")
            )
        except:
            signal["key_facts"] = []
        signals.append(signal)

    conn.close()
    logger.info(f"Loaded {len(signals)} force signals")
    return signals


# ── Text Builders ─────────────────────────────────────────────────────

def build_career_text(career):
    """
    Build rich text representation of a career
    for embedding — captures what the career is about
    """
    parts = []

    parts.append(f"Career: {career.get('title', '')}")
    parts.append(f"Stream: {career.get('stream', '')}")

    desc = career.get("description", "") or career.get("india_description", "")
    if desc:
        parts.append(f"Description: {desc}")

    market = career.get("india_market", {})
    if market:
        parts.append(
            f"Market: {market.get('demand_trend', '')} demand. "
            f"Cities: {', '.join(market.get('top_hiring_cities', []))}"
        )

    progression = career.get("career_progression", {})
    if progression:
        parts.append(
            f"Progression: {progression.get('year_0_2', '')}"
        )

    return " | ".join(parts)


def build_signal_text(signal):
    """
    Build text representation of a force signal for embedding
    """
    parts = []

    parts.append(f"Force: {signal.get('force', '')}")
    parts.append(f"Signal: {signal.get('signal_summary', '')}")

    facts = signal.get("key_facts", [])
    if facts:
        parts.append(f"Facts: {'. '.join(facts[:3])}")

    direction  = signal.get("direction", "")
    magnitude  = signal.get("magnitude", "")
    if direction and magnitude:
        parts.append(f"Direction: {direction}. Magnitude: {magnitude}")

    return " | ".join(parts)


# ── Core Matcher ──────────────────────────────────────────────────────

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot     = np.dot(vec1, vec2)
    norm1   = np.linalg.norm(vec1)
    norm2   = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def match_signals_to_careers(careers, signals, model):
    """
    Core matching function.
    Returns dict of career_id -> list of matched signals with scores
    """
    if not signals:
        logger.warning("No signals to match against")
        return {}

    logger.info("Building career embeddings...")
    career_texts = [build_career_text(c) for c in careers]
    career_embeddings = model.encode(
        career_texts,
        batch_size=32,
        show_progress_bar=False
    )

    logger.info("Building signal embeddings...")
    signal_texts = [build_signal_text(s) for s in signals]
    signal_embeddings = model.encode(
        signal_texts,
        batch_size=32,
        show_progress_bar=False
    )

    logger.info("Calculating similarities...")
    results = {}

    for i, career in enumerate(careers):
        career_id  = career.get("career_id")
        career_vec = career_embeddings[i]

        # Calculate similarity with every signal
        scored_signals = []
        for j, signal in enumerate(signals):
            score = cosine_similarity(career_vec, signal_embeddings[j])
            scored_signals.append({
                "signal_id"       : signal["signal_id"],
                "force"           : signal["force"],
                "signal_summary"  : signal["signal_summary"],
                "direction"       : signal.get("direction", "neutral"),
                "magnitude"       : signal.get("magnitude", "minor"),
                "confidence"      : signal.get("confidence", 0.5),
                "similarity_score": score
            })

        # Keep top K most similar signals
        scored_signals.sort(
            key=lambda x: x["similarity_score"], reverse=True
        )
        results[career_id] = scored_signals[:TOP_K]

    return results


# ── Trajectory Scorer ─────────────────────────────────────────────────

def score_trajectory(matched_signals):
    """
    Calculate trajectory score from matched signals.
    Pure mathematical — no LLM needed.

    Logic:
    - Each signal has direction: positive/negative/neutral/mixed
    - Each signal has magnitude: minor/moderate/significant/major
    - Each signal has confidence and similarity score

    Weighted sum gives trajectory
    """

    if not matched_signals:
        return {
            "trajectory" : "insufficient_data",
            "confidence" : 0.0,
            "timeframe"  : "unknown",
            "score"      : 0.0
        }

    # Direction weights
    direction_weights = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral" : 0.0,
        "mixed"   : 0.2
    }

    # Magnitude weights
    magnitude_weights = {
        "major"      : 1.0,
        "significant": 0.75,
        "moderate"   : 0.5,
        "minor"      : 0.25
    }

    weighted_score = 0.0
    total_weight   = 0.0

    for signal in matched_signals:
        direction   = signal.get("direction", "neutral")
        magnitude   = signal.get("magnitude", "minor")
        confidence  = signal.get("confidence", 0.5)
        similarity  = signal.get("similarity_score", 0.0)

        # Only consider signals above similarity threshold
        if similarity < 0.3:
            continue

        dir_w = direction_weights.get(direction, 0.0)
        mag_w = magnitude_weights.get(magnitude, 0.25)

        # Weight by similarity, confidence, and magnitude
        weight = similarity * confidence * mag_w
        weighted_score += dir_w * weight
        total_weight   += weight

    if total_weight == 0:
        return {
            "trajectory": "insufficient_data",
            "confidence": 0.0,
            "timeframe" : "unknown",
            "score"     : 0.0
        }

    final_score = weighted_score / total_weight
    confidence  = min(total_weight / 5.0, 1.0)

    # Convert score to label
    if final_score > 0.4:
        trajectory = "strongly_growing"
    elif final_score > 0.15:
        trajectory = "growing"
    elif final_score > -0.15:
        trajectory = "stable"
    elif final_score > -0.4:
        trajectory = "declining"
    else:
        trajectory = "strongly_declining"

    # Timeframe based on signal types
    timeframe = "medium_term_3_5_years"

    return {
        "trajectory": trajectory,
        "confidence": round(confidence, 3),
        "timeframe" : timeframe,
        "score"     : round(final_score, 3)
    }


# ── Database Writer ───────────────────────────────────────────────────

def save_connections(career, matched_signals):
    """Save career-signal connections to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    career_id = career.get("career_id")

    # Delete old connections for this career
    cursor.execute(
        "DELETE FROM career_signal_connections WHERE career_id = ?",
        (career_id,)
    )

    # Insert new connections
    for signal in matched_signals:
        if signal["similarity_score"] < 0.3:
            continue
        cursor.execute("""
            INSERT INTO career_signal_connections
            (career_id, signal_id, similarity_score, force, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            career_id,
            signal["signal_id"],
            signal["similarity_score"],
            signal["force"],
            now
        ))

    conn.commit()
    conn.close()


def save_trajectory(career, trajectory_data, matched_signals):
    """Save career trajectory score to database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    career_id = career.get("career_id")

    # Count signals above threshold
    valid_signals = [
        s for s in matched_signals
        if s["similarity_score"] >= 0.3
    ]

    # Get top forces
    force_counts = {}
    for s in valid_signals:
        force = s["force"]
        force_counts[force] = force_counts.get(force, 0) + 1
    top_forces = sorted(
        force_counts, key=force_counts.get, reverse=True
    )[:3]

    # Build evidence summary
    evidence = [
        s["signal_summary"]
        for s in valid_signals[:3]
    ]
    evidence_summary = " | ".join(evidence)

    cursor.execute("""
        INSERT OR REPLACE INTO career_trajectories
        (career_id, title, stream, trajectory, confidence,
         timeframe, signal_count, top_forces,
         evidence_summary, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        career_id,
        career.get("title"),
        career.get("stream"),
        trajectory_data["trajectory"],
        trajectory_data["confidence"],
        trajectory_data["timeframe"],
        len(valid_signals),
        json.dumps(top_forces),
        evidence_summary[:500],
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()


# ── Main Runner ───────────────────────────────────────────────────────

def run_matching_cycle():
    """Run complete matching cycle"""
    logger.info("🔗 Starting semantic matching cycle...")
    start = datetime.utcnow()

    # Ensure tables exist
    ensure_tables()

    # Load data
    careers = load_all_careers()
    signals = load_all_signals()

    if not careers:
        logger.error("No careers found. Check careers/ folder.")
        return

    if not signals:
        logger.warning("No signals yet. Skipping matching.")
        return

    logger.info(f"Matching {len(careers)} careers against {len(signals)} signals")

    # Load model — downloads once, cached after
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded")

    # Run matching
    all_matches = match_signals_to_careers(careers, signals, model)

    # Score trajectories and save
    growing  = 0
    stable   = 0
    declining = 0

    for career in careers:
        career_id      = career.get("career_id")
        matched        = all_matches.get(career_id, [])
        trajectory     = score_trajectory(matched)

        save_connections(career, matched)
        save_trajectory(career, trajectory, matched)

        t = trajectory["trajectory"]
        if "growing" in t:
            growing += 1
        elif "declining" in t:
            declining += 1
        else:
            stable += 1

        logger.info(
            f"  {career_id} {career.get('title')}: "
            f"{t} (confidence: {trajectory['confidence']})"
        )

    duration = (datetime.utcnow() - start).total_seconds()

    logger.info("=" * 60)
    logger.info(f"MATCHING COMPLETE in {duration:.1f}s")
    logger.info(f"Growing:   {growing}")
    logger.info(f"Stable:    {stable}")
    logger.info(f"Declining: {declining}")
    logger.info("=" * 60)

    log_pipeline_event(
        event_type="MATCHING_CYCLE_COMPLETE",
        message=f"Matched {len(careers)} careers to {len(signals)} signals",
        details={
            "careers": len(careers),
            "signals": len(signals),
            "growing": growing,
            "stable": stable,
            "declining": declining,
            "duration_seconds": duration
        },
        status="success",
        db_path=DB_PATH
    )


if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    run_matching_cycle()
