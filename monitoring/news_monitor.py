"""
CounselAI Pipeline — News API Monitor
Fetches relevant news articles from NewsAPI.org
Free tier: 100 requests/day — we use wisely
"""

import requests
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import (
    store_raw_content,
    update_source_status,
    log_pipeline_event
)

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2/everything"


def fetch_news_articles(query: str, api_key: str,
                        language: str = "en",
                        days_back: int = 1,
                        page_size: int = 10) -> Optional[list]:
    """
    Fetch news articles from NewsAPI.
    Looks back days_back days to catch recent news.
    """
    if not api_key:
        logger.warning("No NEWS_API_KEY set — skipping news API")
        return None

    # Calculate date range
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime(
        '%Y-%m-%d'
    )

    params = {
        'q': query,
        'language': language,
        'from': from_date,
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': api_key
    }

    try:
        response = requests.get(
            NEWS_API_BASE,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'ok':
            return data.get('articles', [])
        else:
            logger.error(f"NewsAPI error: {data.get('message')}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI request failed: {e}")
        return None


def process_news_article(article: dict) -> tuple:
    """
    Extract clean content from a news article dict.
    Returns (title, content, url, published_date)
    """
    import re

    title = article.get('title', '') or ''
    description = article.get('description', '') or ''
    content = article.get('content', '') or ''
    url = article.get('url', '')
    published_at = article.get('publishedAt', '')

    # Build full content
    full_content = f"TITLE: {title}\n\n"
    if description:
        full_content += f"DESCRIPTION: {description}\n\n"
    if content:
        # NewsAPI truncates at 200 chars — take what we have
        full_content += f"CONTENT: {content}"

    # Clean
    full_content = re.sub(r'\s+', ' ', full_content).strip()
    full_content = full_content.encode('ascii', 'ignore').decode('ascii')

    return title, full_content, url, published_at


def poll_news_api_source(source_config: dict, api_key: str,
                         db_path: str = "storage/counselai.db") -> int:
    """
    Poll a single News API source configuration.
    Returns number of new items stored.
    """
    source_id = source_config['source_id']
    source_name = source_config['name']
    force = source_config.get('force', 'UNKNOWN')
    query = source_config['query']
    language = source_config.get('language', 'en')

    logger.info(f"Polling NewsAPI: {source_name} ({force})")

    articles = fetch_news_articles(
        query=query,
        api_key=api_key,
        language=language,
        days_back=1,
        page_size=10
    )

    if articles is None:
        update_source_status(
            source_id, source_name, force,
            success=False,
            error_msg="NewsAPI fetch failed",
            db_path=db_path
        )
        return 0

    new_items = 0

    for article in articles:
        title, content, url, pub_date = process_news_article(article)

        # Skip very short content
        if len(content) < 100:
            continue

        # Skip articles without India relevance
        combined = f"{title} {content}".lower()
        india_keywords = ['india', 'indian', 'delhi', 'mumbai',
                          'bangalore', 'hyderabad', 'modi', 'rupee']
        if not any(kw in combined for kw in india_keywords):
            continue

        # Store content
        record_id = store_raw_content(
            source_id=source_id,
            source_name=source_name,
            force=force,
            title=title,
            content=content,
            url=url,
            published_date=pub_date,
            db_path=db_path
        )

        if record_id is not None:
            new_items += 1
            logger.debug(f"New article: {title[:60]}...")

        time.sleep(0.2)

    update_source_status(
        source_id=source_id,
        source_name=source_name,
        force=force,
        success=True,
        new_items=new_items,
        db_path=db_path
    )

    logger.info(f"NewsAPI {source_name}: {new_items} new articles")
    return new_items


def poll_all_news_sources(sources_config: dict, api_key: str,
                          db_path: str = "storage/counselai.db") -> dict:
    """
    Poll all News API sources across all forces.
    Careful with rate limits — 100 requests/day free tier.
    """
    results = {'total_new_items': 0, 'requests_made': 0}

    # Count total news sources first
    total_news_sources = sum(
        1
        for force_data in sources_config['forces'].values()
        for source in force_data['sources']
        if source['type'] == 'news_api' and source.get('active', True)
    )

    logger.info(f"Polling {total_news_sources} News API sources")

    for force_key, force_data in sources_config['forces'].items():
        for source in force_data['sources']:
            if source['type'] != 'news_api':
                continue
            if not source.get('active', True):
                continue

            source_with_force = {**source, 'force': force_key}

            try:
                new_items = poll_news_api_source(
                    source_with_force, api_key, db_path
                )
                results['total_new_items'] += new_items
                results['requests_made'] += 1

                # Important — respect rate limits
                # Free tier: 100 requests/day
                time.sleep(2)

            except Exception as e:
                logger.error(f"News API error for {source['name']}: {e}")

    return results
