import json
import sqlite3
import logging
import os
import sys
import math
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import log_pipeline_event

logger = logging.getLogger(__name__)

CAREERS_DIR = Path("careers")
DB_PATH     = os.getenv("DB_PATH", "storage/counselai.db")
TOP_K       = 20


# ── Database ──────────────────────────────────────────────────────────

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
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
    careers = []
    for json_file in sorted(CAREERS_DIR.rglob("*.json")):
        if json_file.name == "career_index.json":
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                careers.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    logger.info(f"Loaded {len(careers)} careers")
    return careers


def load_all_signals():
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
            signal["key_facts"] = json.loads(signal.get("key_facts", "[]"))
        except:
            signal["key_facts"] = []
        signals.append(signal)
    conn.close()
    logger.info(f"Loaded {len(signals)} force signals")
    return signals


# ── Text Builders ─────────────────────────────────────────────────────

def build_career_text(career):
    parts = []
    parts.append(career.get("title", ""))
    parts.append(career.get("stream", ""))
    desc = career.get("description", "") or career.get("india_description", "")
    if desc:
        parts.append(desc)
    market = career.get("india_market", {})
    if market:
        parts.append(market.get("demand_trend", ""))
        parts.extend(market.get("top_hiring_cities", []))
    prog = career.get("career_progression", {})
    if prog:
        parts.append(prog.get("year_0_2", ""))
    return " ".join(parts)


def build_signal_text(signal):
    parts = []
    parts.append(signal.get("force", "").replace("_", " "))
    parts.append(signal.get("signal_summary", ""))
    for fact in signal.get("key_facts", [])[:3]:
        parts.append(fact)
    parts.append(signal.get("direction", ""))
    parts.append(signal.get("magnitude", ""))
    return " ".join(parts)


# ── Pure Python TF-IDF ────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "it", "its", "as", "from"
}


def tokenize(text):
    """Simple tokenizer"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def compute_tfidf(texts):
    """
    Compute TF-IDF matrix for a list of texts.
    Returns: (matrix as list of dicts, vocabulary)
    """
    # Tokenize all texts
    tokenized = [tokenize(t) for t in texts]

    # Build vocabulary
    vocab = set()
    for tokens in tokenized:
        vocab.update(tokens)
    vocab = list(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    n_docs = len(texts)

    # Compute document frequency
    df = Counter()
    for tokens in tokenized:
        for word in set(tokens):
            df[word] += 1

    # Compute IDF
    idf = {}
    for word in vocab:
        idf[word] = math.log((n_docs + 1) / (df[word] + 1)) + 1

    # Compute TF-IDF vectors as numpy arrays
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = np.zeros(len(vocab))
        for word, count in tf.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                vec[idx] = (count / total) * idf[word]
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)

    return vectors


def cosine_sim(vec1, vec2):
    """Cosine similarity between two normalized vectors"""
    return float(np.dot(vec1, vec2))


# ── Core Matcher ──────────────────────────────────────────────────────

def match_signals_to_careers(careers, signals):
    """Match force signals to careers using TF-IDF"""

    if not signals:
        logger.warning("No signals to match")
        return {}

    career_texts = [build_career_text(c) for c in careers]
    signal_texts = [build_signal_text(s) for s in signals]

    logger.info("Computing TF-IDF vectors...")
    all_texts = career_texts + signal_texts
    all_vectors = compute_tfidf(all_texts)

    n_careers = len(careers)
    career_vecs = all_vectors[:n_careers]
    signal_vecs = all_vectors[n_careers:]

    logger.info("Calculating similarities...")
    results = {}

    for i, career in enumerate(careers):
        career_id = career.get("career_id")
        scored = []

        for j, signal in enumerate(signals):
            score = cosine_sim(career_vecs[i], signal_vecs[j])
            scored.append({
                "signal_id"       : signal["signal_id"],
                "force"           : signal["force"],
                "signal_summary"  : signal["signal_summary"],
                "direction"       : signal.get("direction", "neutral"),
                "magnitude"       : signal.get("magnitude", "minor"),
                "confidence"      : signal.get("confidence", 0.5),
                "similarity_score": score
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        results[career_id] = scored[:TOP_K]

    return results


# ── Trajectory Scorer ─────────────────────────────────────────────────

def score_trajectory(matched_signals):
    if not matched_signals:
        return {
            "trajectory": "insufficient_data",
            "confidence": 0.0,
            "timeframe" : "unknown",
            "score"     : 0.0
        }

    direction_weights = {
        "positive": 1.0, "negative": -1.0,
        "neutral" : 0.0, "mixed"   : 0.2
    }
    magnitude_weights = {
        "major": 1.0, "significant": 0.75,
        "moderate": 0.5, "minor": 0.25
    }

    weighted_score = 0.0
    total_weight   = 0.0

    for signal in matched_signals:
        similarity = signal.get("similarity_score", 0.0)
        if similarity < 0.1:
            continue
        dir_w  = direction_weights.get(signal.get("direction", "neutral"), 0.0)
        mag_w  = magnitude_weights.get(signal.get("magnitude", "minor"), 0.25)
        conf   = signal.get("confidence", 0.5)
        weight = similarity * conf * mag_w
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

    return {
        "trajectory": trajectory,
        "confidence": round(confidence, 3),
        "timeframe" : "medium_term_3_5_years",
        "score"     : round(final_score, 3)
    }


# ── Database Writers ──────────────────────────────────────────────────

def save_connections(career, matched_signals):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    career_id = career.get("career_id")
    cursor.execute(
        "DELETE FROM career_signal_connections WHERE career_id = ?",
        (career_id,)
    )
    for signal in matched_signals:
        if signal["similarity_score"] < 0.1:
            continue
        cursor.execute("""
            INSERT INTO career_signal_connections
            (career_id, signal_id, similarity_score, force, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            career_id, signal["signal_id"],
            signal["similarity_score"],
            signal["force"], now
        ))
    conn.commit()
    conn.close()


def save_trajectory(career, trajectory_data, matched_signals):
    conn = get_db_connection()
    cursor = conn.cursor()
    career_id = career.get("career_id")

    valid = [s for s in matched_signals if s["similarity_score"] >= 0.1]

    force_counts = {}
    for s in valid:
        force_counts[s["force"]] = force_counts.get(s["force"], 0) + 1
    top_forces = sorted(
        force_counts, key=force_counts.get, reverse=True
    )[:3]

    evidence = [s["signal_summary"] for s in valid[:3]]
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
        len(valid),
        json.dumps(top_forces),
        evidence_summary[:500],
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


# ── Main Runner ───────────────────────────────────────────────────────

def run_matching_cycle():
    logger.info("🔗 Starting semantic matching cycle...")
    start = datetime.utcnow()

    ensure_tables()
    careers = load_all_careers()
    signals = load_all_signals()

    if not careers:
        logger.error("No careers found")
        return

    if not signals:
        logger.warning("No signals yet — skipping matching")
        return

    logger.info(
        f"Matching {len(careers)} careers "
        f"against {len(signals)} signals"
    )

    all_matches = match_signals_to_careers(careers, signals)

    growing = stable = declining = 0

    for career in careers:
        career_id  = career.get("career_id")
        matched    = all_matches.get(career_id, [])
        trajectory = score_trajectory(matched)

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
            f"{t} ({trajectory['confidence']})"
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
        message=(
            f"Matched {len(careers)} careers "
            f"to {len(signals)} signals"
        ),
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
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    run_matching_cycle()
