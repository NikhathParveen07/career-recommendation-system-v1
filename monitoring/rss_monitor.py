"""
CounselAI Pipeline — RSS Feed Monitor
Polls RSS feeds and stores new content for processing
"""

import feedparser
import logging
import time
from datetime import datetime
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import (
    store_raw_content,
    update_source_status,
    log_pipeline_event
)

logger = logging.getLogger(__name__)


def fetch_rss_feed(url: str, timeout: int = 30) -> Optional[dict]:
    """
    Fetch and parse an RSS feed.
    Returns parsed feed or None if failed.
    """
    try:
        feed = feedparser.parse(url, request_headers={
            'User-Agent': 'CounselAI Research Pipeline/1.0'
        })

        # Check if feed parsed correctly
        if feed.bozo and not feed.entries:
            logger.warning(f"Feed may be malformed: {url}")
            return None

        return feed

    except Exception as e:
        logger.error(f"Failed to fetch RSS feed {url}: {e}")
        return None


def clean_content(entry: dict) -> str:
    """
    Extract clean text content from an RSS entry.
    Removes HTML tags, excessive whitespace.
    """
    import re

    # Try different content fields in order of preference
    content = ""

    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].get('value', '')
    elif hasattr(entry, 'summary') and entry.summary:
        content = entry.summary
    elif hasattr(entry, 'description') and entry.description:
        content = entry.description
    elif hasattr(entry, 'title') and entry.title:
        content = entry.title

    # Remove HTML tags
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content).strip()

    # Remove special characters that cause encoding issues
    content = content.encode('ascii', 'ignore').decode('ascii')

    return content


def get_entry_title(entry: dict) -> str:
    """Extract title from RSS entry"""
    if hasattr(entry, 'title'):
        return entry.title[:500]
    return "No title"


def get_entry_url(entry: dict) -> Optional[str]:
    """Extract URL from RSS entry"""
    if hasattr(entry, 'link'):
        return entry.link
    return None


def get_entry_date(entry: dict) -> Optional[str]:
    """Extract published date from RSS entry"""
    if hasattr(entry, 'published'):
        return entry.published
    elif hasattr(entry, 'updated'):
        return entry.updated
    return datetime.utcnow().isoformat()


def is_india_relevant(title: str, content: str,
                      source_India_specific: bool) -> bool:
    """
    Check if content is relevant to India.
    For India-specific sources always True.
    For global sources check for India mentions.
    """
    if source_India_specific:
        return True

    # For global sources check for India relevance
    combined = f"{title} {content}".lower()
    india_keywords = [
        'india', 'indian', 'delhi', 'mumbai', 'bangalore',
        'hyderabad', 'chennai', 'kolkata', 'pune', 'ahmedabad',
        'modi', 'rupee', 'inr', 'crore', 'lakh', 'niti aayog',
        'ministry', 'parliament', 'lok sabha', 'rajya sabha'
    ]

    return any(keyword in combined for keyword in india_keywords)


def poll_rss_source(source_config: dict,
                    db_path: str = "storage/counselai.db") -> int:
    """
    Poll a single RSS source and store new content.
    Returns number of new items stored.
    """
    source_id = source_config['source_id']
    source_name = source_config['name']
    force = source_config.get('force', 'UNKNOWN')
    url = source_config['url']
    India_specific = source_config.get('India_specific', True)

    logger.info(f"Polling RSS: {source_name} ({force})")

    # Fetch feed
    feed = fetch_rss_feed(url)

    if feed is None:
        logger.error(f"Failed to fetch: {source_name}")
        update_source_status(
            source_id, source_name, force,
            success=False,
            error_msg="Failed to fetch feed",
            db_path=db_path
        )
        log_pipeline_event(
            event_type="RSS_FETCH_FAILED",
            message=f"Failed to fetch {source_name}",
            source_id=source_id,
            force=force,
            status="error",
            db_path=db_path
        )
        return 0

    new_items = 0

    for entry in feed.entries:
        title = get_entry_title(entry)
        content = clean_content(entry)
        url_entry = get_entry_url(entry)
        pub_date = get_entry_date(entry)

        # Skip empty content
        if not content or len(content) < 50:
            continue

        # Skip non-India content for global sources
        if not is_india_relevant(title, content, India_specific):
            continue

        # Combine title and content for storage
        full_content = f"TITLE: {title}\n\nCONTENT: {content}"

        # Store in database — returns None if already seen
        record_id = store_raw_content(
            source_id=source_id,
            source_name=source_name,
            force=force,
            title=title,
            content=full_content,
            url=url_entry,
            published_date=pub_date,
            db_path=db_path
        )

        if record_id is not None:
            new_items += 1
            logger.debug(f"New content stored: {title[:60]}...")

        # Small delay to be respectful to servers
        time.sleep(0.1)

    # Update source health status
    update_source_status(
        source_id=source_id,
        source_name=source_name,
        force=force,
        success=True,
        new_items=new_items,
        db_path=db_path
    )

    if new_items > 0:
        logger.info(f"✅ {source_name}: {new_items} new items")
        log_pipeline_event(
            event_type="RSS_FETCH_SUCCESS",
            message=f"Fetched {new_items} new items from {source_name}",
            source_id=source_id,
            force=force,
            details={"new_items": new_items},
            status="success",
            db_path=db_path
        )
    else:
        logger.info(f"⏭️  {source_name}: No new items")

    return new_items


def poll_all_rss_sources(sources_config: dict,
                         db_path: str = "storage/counselai.db") -> dict:
    """
    Poll all RSS sources across all forces.
    Returns summary of what was fetched.
    """
    results = {
        'total_new_items': 0,
        'by_force': {},
        'by_source': {},
        'errors': []
    }

    for force_key, force_data in sources_config['forces'].items():
        force_results = {'new_items': 0, 'sources_polled': 0}

        for source in force_data['sources']:
            # Only process RSS sources
            if source['type'] != 'rss':
                continue

            # Skip inactive sources
            if not source.get('active', True):
                continue

            # Add force key to source config for storage
            source_with_force = {**source, 'force': force_key}

            try:
                new_items = poll_rss_source(source_with_force, db_path)
                force_results['new_items'] += new_items
                force_results['sources_polled'] += 1
                results['by_source'][source['source_id']] = new_items
                results['total_new_items'] += new_items

                # Respectful delay between sources
                time.sleep(1)

            except Exception as e:
                error_msg = f"Error polling {source['name']}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        results['by_force'][force_key] = force_results

    return results


if __name__ == "__main__":
    # Test single source
    import json

    logging.basicConfig(level=logging.INFO)

    test_source = {
        "source_id": "F6_RSS_004",
        "name": "Economic Times Energy",
        "type": "rss",
        "url": "https://economictimes.indiatimes.com/rss/energy.cms",
        "credibility": "high",
        "India_specific": True,
        "force": "F6_CLIMATE"
    }

    from storage.database import initialise_database
    initialise_database()

    print("Testing RSS monitor...")
    new_items = poll_rss_source(test_source)
    print(f"New items found: {new_items}")
