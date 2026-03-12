"""
CounselAI Pipeline — World Bank API Monitor
Fetches India economic and demographic indicators
Free API — no key needed
"""

import requests
import logging
import json
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import (
    store_raw_content,
    update_source_status,
    log_pipeline_event
)

logger = logging.getLogger(__name__)

WORLD_BANK_BASE = "https://api.worldbank.org/v2"

# Human readable indicator names
INDICATOR_NAMES = {
    "SP.POP.TOTL": "Total Population India",
    "SP.POP.GROW": "Population Growth Rate India",
    "SP.URB.TOTL.IN.ZS": "Urban Population Percentage India",
    "SP.POP.65UP.TO.ZS": "Population 65+ Years India",
    "SP.POP.0014.TO.ZS": "Population 0-14 Years India",
    "NY.GNP.PCAP.CD": "GNI Per Capita India",
    "BX.GSR.NFSV.CD": "India Services Exports USD",
    "BM.GSR.NFSV.CD": "India Services Imports USD",
    "BX.TRF.PWKR.DT.GD.ZS": "Remittances Percentage GDP India",
    "SE.TER.ENRR": "Tertiary Education Enrollment India",
    "SL.UEM.ADVN.ZS": "Advanced Education Unemployment India",
    "SE.TER.GRAD.FE.SI.ZS": "Female Graduates Tertiary India"
}


def fetch_world_bank_indicator(indicator: str,
                                country: str = "IND",
                                years: int = 5) -> dict | None:
    """
    Fetch a single World Bank indicator for India.
    Gets last N years of data.
    """
    url = f"{WORLD_BANK_BASE}/country/{country}/indicator/{indicator}"
    params = {
        'format': 'json',
        'mrv': years,
        'per_page': years
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if len(data) < 2 or not data[1]:
            logger.warning(f"No data for indicator: {indicator}")
            return None

        return data[1]

    except Exception as e:
        logger.error(f"World Bank API error for {indicator}: {e}")
        return None


def format_indicator_as_content(indicator: str,
                                  data: list) -> str:
    """
    Format World Bank indicator data as readable content
    for LLM extraction pipeline.
    """
    name = INDICATOR_NAMES.get(indicator, indicator)

    # Filter out null values
    valid_data = [
        d for d in data
        if d.get('value') is not None
    ]

    if not valid_data:
        return None

    # Sort by year
    valid_data.sort(key=lambda x: x.get('date', '0'), reverse=True)

    content = f"WORLD BANK INDICATOR: {name}\n"
    content += f"Country: India\n"
    content += f"Indicator Code: {indicator}\n\n"
    content += "RECENT DATA:\n"

    for record in valid_data[:5]:
        year = record.get('date', 'Unknown')
        value = record.get('value', 'N/A')
        if isinstance(value, float):
            value = round(value, 2)
        content += f"  {year}: {value}\n"

    # Add trend analysis
    if len(valid_data) >= 2:
        latest = valid_data[0].get('value')
        previous = valid_data[1].get('value')
        if latest and previous and previous != 0:
            change = ((latest - previous) / abs(previous)) * 100
            direction = "increased" if change > 0 else "decreased"
            content += f"\nTREND: Value {direction} by {abs(change):.1f}% "
            content += f"from {valid_data[1].get('date')} to "
            content += f"{valid_data[0].get('date')}\n"

    return content


def poll_world_bank_source(source_config: dict,
                            db_path: str = "storage/counselai.db") -> int:
    """
    Poll all indicators for a World Bank source configuration.
    Returns number of new records stored.
    """
    source_id = source_config['source_id']
    source_name = source_config['name']
    force = source_config.get('force', 'UNKNOWN')
    indicators = source_config.get('indicators', [])
    country = source_config.get('country_code', 'IND')

    logger.info(f"Polling World Bank: {source_name}")
    new_items = 0

    for indicator in indicators:
        data = fetch_world_bank_indicator(indicator, country)

        if not data:
            continue

        content = format_indicator_as_content(indicator, data)
        if not content:
            continue

        indicator_name = INDICATOR_NAMES.get(indicator, indicator)

        record_id = store_raw_content(
            source_id=f"{source_id}_{indicator}",
            source_name=f"World Bank - {indicator_name}",
            force=force,
            title=f"World Bank Indicator: {indicator_name}",
            content=content,
            url=f"https://data.worldbank.org/indicator/{indicator}?locations=IN",
            published_date=datetime.utcnow().isoformat(),
            db_path=db_path
        )

        if record_id is not None:
            new_items += 1

        time.sleep(0.5)

    update_source_status(
        source_id=source_id,
        source_name=source_name,
        force=force,
        success=True,
        new_items=new_items,
        db_path=db_path
    )

    logger.info(f"World Bank {source_name}: {new_items} new records")
    return new_items


def poll_all_world_bank_sources(sources_config: dict,
                                 db_path: str = "storage/counselai.db") -> int:
    """Poll all World Bank sources across all forces"""
    total_new = 0

    for force_key, force_data in sources_config['forces'].items():
        for source in force_data['sources']:
            if source['type'] != 'world_bank_api':
                continue
            if not source.get('active', True):
                continue

            source_with_force = {**source, 'force': force_key}

            try:
                new_items = poll_world_bank_source(
                    source_with_force, db_path
                )
                total_new += new_items
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"World Bank error for {source['name']}: {e}"
                )

    return total_new
