"""
CounselAI Pipeline — Main Scheduler
Orchestrates all monitoring tasks
Runs continuously — polls sources every 6 hours
"""

import schedule
import time
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storage.database import (
    initialise_database,
    get_pipeline_stats,
    log_pipeline_event
)
from monitoring.rss_monitor import poll_all_rss_sources
from monitoring.news_monitor import poll_all_news_sources
from monitoring.worldbank_monitor import poll_all_world_bank_sources
from extraction.extractor import run_extraction_cycle
from matching.semantic_matcher import run_matching_cycle

# ── Configuration ─────────────────────────────────────────────────────
DB_PATH = os.getenv('DB_PATH', 'storage/counselai.db')
LOG_PATH = os.getenv('LOG_PATH', 'logs/pipeline.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
POLL_INTERVAL_HOURS = int(os.getenv('POLL_INTERVAL_HOURS', 6))
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# ── Logging Setup ─────────────────────────────────────────────────────
Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('counselai.scheduler')


def load_sources_config() -> dict:
    """Load sources configuration from JSON"""
    config_path = Path('config/sources.json')
    if not config_path.exists():
        raise FileNotFoundError(
            "config/sources.json not found. "
            "Make sure you are running from the project root."
        )
    with open(config_path) as f:
        return json.load(f)


def run_monitoring_cycle():
    """
    One complete monitoring cycle.
    Fetches from all source types.
    Called every POLL_INTERVAL_HOURS hours.
    """
    cycle_start = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"MONITORING CYCLE STARTED: {cycle_start.isoformat()}")
    logger.info("=" * 60)

    sources_config = load_sources_config()
    total_new = 0

    # ── Step 1: RSS Feeds ─────────────────────────────────────────────
    logger.info("📡 Stage 1: Polling RSS feeds...")
    try:
        rss_results = poll_all_rss_sources(sources_config, DB_PATH)
        rss_new = rss_results.get('total_new_items', 0)
        total_new += rss_new
        logger.info(f"RSS complete: {rss_new} new items")

        # Log by force
        for force, data in rss_results.get('by_force', {}).items():
            if data['new_items'] > 0:
                logger.info(
                    f"  {force}: {data['new_items']} new items "
                    f"from {data['sources_polled']} sources"
                )

    except Exception as e:
        logger.error(f"RSS polling failed: {e}")
        log_pipeline_event(
            "RSS_CYCLE_ERROR", f"RSS cycle failed: {e}",
            status="error", db_path=DB_PATH
        )

    # ── Step 2: News API ──────────────────────────────────────────────
    if NEWS_API_KEY:
        logger.info("📰 Stage 2: Polling News API...")
        try:
            news_results = poll_all_news_sources(
                sources_config, NEWS_API_KEY, DB_PATH
            )
            news_new = news_results.get('total_new_items', 0)
            total_new += news_new
            logger.info(
                f"News API complete: {news_new} new articles "
                f"({news_results.get('requests_made', 0)} requests)"
            )
        except Exception as e:
            logger.error(f"News API polling failed: {e}")
    else:
        logger.info(
            "⏭️  Stage 2: News API skipped (no API key set)"
        )

    # ── Step 3: World Bank API ────────────────────────────────────────
    logger.info("🌍 Stage 3: Polling World Bank API...")
    try:
        wb_new = poll_all_world_bank_sources(sources_config, DB_PATH)
        total_new += wb_new
        logger.info(f"World Bank complete: {wb_new} new records")
    except Exception as e:
        logger.error(f"World Bank polling failed: {e}")
    # ── Step 4: LLM Extraction ────────────────────────────────────────
    logger.info("🔍 Stage 4: Running LLM extraction...")
    try:
        extraction_results = run_extraction_cycle(DB_PATH)
        signals = extraction_results.get("signals_extracted", 0)
        logger.info(f"Extraction complete: {signals} new signals extracted")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
         # ── Step 5: Semantic Matching ─────────────────────────────────────
    logger.info("🔗 Stage 5: Running semantic matching...")
    try:
        run_matching_cycle()
    except Exception as e:
        logger.error(f"Matching failed: {e}")
    # ── Cycle Summary ─────────────────────────────────────────────────
    cycle_end = datetime.utcnow()
    duration = (cycle_end - cycle_start).total_seconds()

    stats = get_pipeline_stats(DB_PATH)

    logger.info("=" * 60)
    logger.info(f"CYCLE COMPLETE in {duration:.1f}s")
    logger.info(f"New items this cycle: {total_new}")
    logger.info(f"Total raw content: {stats['total_raw_content']}")
    logger.info(f"Pending extraction: {stats['pending_processing']}")
    logger.info(f"Active signals: {stats['total_active_signals']}")
    logger.info("=" * 60)

    log_pipeline_event(
        event_type="MONITORING_CYCLE_COMPLETE",
        message=f"Cycle complete. {total_new} new items in {duration:.1f}s",
        details={
            "total_new_items": total_new,
            "duration_seconds": duration,
            "stats": stats
        },
        status="success",
        db_path=DB_PATH
    )

    return total_new


def print_dashboard():
    """Print current pipeline status to console"""
    stats = get_pipeline_stats(DB_PATH)

    print("\n" + "=" * 50)
    print("  COUNSELAI PIPELINE DASHBOARD")
    print("=" * 50)
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Total raw content collected: {stats['total_raw_content']}")
    print(f"  Pending extraction:          {stats['pending_processing']}")
    print(f"  Active force signals:        {stats['total_active_signals']}")
    print(f"  Healthy sources:             {stats['healthy_sources']}")
    print()
    print("  Signals by force:")
    for force, count in stats.get('signals_by_force', {}).items():
        bar = "█" * min(count, 20)
        print(f"    {force:<30} {bar} {count}")
    print("=" * 50 + "\n")


def start_pipeline(run_once: bool = False):
    """
    Start the continuous monitoring pipeline.

    Args:
        run_once: If True, run one cycle and exit.
                  Used for testing.
                  If False, run continuously on schedule.
    """
    logger.info("🚀 CounselAI Knowledge Pipeline Starting...")
    logger.info(f"Poll interval: every {POLL_INTERVAL_HOURS} hours")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"News API: {'configured' if NEWS_API_KEY else 'not configured'}")

    # Initialise database
    initialise_database(DB_PATH)
    logger.info("Database ready")

    if run_once:
        # Single run for testing
        logger.info("Running single monitoring cycle (test mode)...")
        run_monitoring_cycle()
        print_dashboard()
        return

    # Schedule recurring runs
    schedule.every(POLL_INTERVAL_HOURS).hours.do(run_monitoring_cycle)
    schedule.every(1).hours.do(print_dashboard)

    # Run immediately on start
    logger.info("Running initial monitoring cycle...")
    run_monitoring_cycle()
    print_dashboard()

    # Then run on schedule
    logger.info(
        f"Pipeline running. Next cycle in {POLL_INTERVAL_HOURS} hours."
    )
    logger.info("Press Ctrl+C to stop.\n")

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(300)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='CounselAI Monitoring Pipeline'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one cycle and exit (for testing)'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show current stats and exit'
    )

    args = parser.parse_args()

    if args.dashboard:
        initialise_database(DB_PATH)
        print_dashboard()
    else:
        start_pipeline(run_once=args.once)
