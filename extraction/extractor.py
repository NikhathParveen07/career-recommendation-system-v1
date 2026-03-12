import json
import logging
import os
import sys
import time
from datetime import datetime

from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import (
    get_unprocessed_content,
    store_force_signal,
    mark_content_processed,
    log_pipeline_event
)

logger = logging.getLogger(__name__)

FORCE_DESCRIPTIONS = {
    "F1_TECHNOLOGY": "Technology changes happening in India including emerging technologies, automation, digital infrastructure, and technology investment.",
    "F2_GOVERNMENT_POLICY": "Indian government policy signals including budget allocations, new schemes, PLI incentives, PMKVY programmes, ministry announcements, regulatory changes.",
    "F3_DEMOGRAPHICS": "India population and demographic changes including urbanisation, age structure, middle class expansion, migration patterns.",
    "F4_GLOBAL_MARKET": "Global market signals affecting India including trade flows, foreign investment, Indian exports, global demand, remittances, bilateral agreements.",
    "F5_EDUCATION_SUPPLY": "India education output including graduate numbers by discipline, enrollment trends, skill gaps, employability trends.",
    "F6_CLIMATE": "Climate and sustainability signals for India including renewable energy targets, climate commitments, green economy investment, energy transition.",
    "F7_SOCIAL_SHIFTS": "Indian society behaviour changes including consumer preferences, lifestyle shifts, social awareness, wellbeing trends, cultural attitude changes."
}

EXTRACTION_PROMPT = """
You are a research analyst extracting factual signals about India.

CONTENT TO ANALYSE:
{content}

FORCE TO EXTRACT: {force_name}
{force_description}

STRICT RULES:
1. Extract ONLY what is factually stated in the content
2. Do NOT mention any careers, jobs, or occupations
3. Do NOT mention any skills or qualifications
4. Do NOT say what this means for employment
5. Do NOT infer anything not in the content
6. If content is not relevant to this force, set is_relevant to false

Return ONLY a JSON object with exactly these fields:
{{
    "is_relevant": true or false,
    "signal_summary": "One sentence. No careers mentioned.",
    "key_facts": ["fact 1", "fact 2", "fact 3"],
    "geographic_scope": "national / state-specific / city-specific / global",
    "timeline": "immediate / short-term 1-2 years / medium-term 3-5 years / long-term 5+ years",
    "magnitude": "minor / moderate / significant / major",
    "direction": "positive / negative / neutral / mixed",
    "evidence_quote": "Most relevant quote from content under 50 words",
    "confidence": 0.0 to 1.0
}}

Return ONLY the JSON. No explanation.
"""


def extract_signal_from_content(content, force, client):
    force_desc = FORCE_DESCRIPTIONS.get(force, "")
    prompt = EXTRACTION_PROMPT.format(
        content=content[:3000],
        force_name=force,
        force_description=force_desc
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise research analyst. You extract factual signals. You always return valid JSON only. You never mention careers, jobs, or skills."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"LLM extraction error: {e}")
        return None


def validate_signal_purity(signal):
    career_indicators = [
        "engineer", "doctor", "nurse", "teacher", "analyst",
        "developer", "manager", "officer", "technician",
        "scientist", "architect", "lawyer", "accountant",
        "programmer", "designer", "consultant", "specialist",
        "profession", "occupation", "career", "job title",
        "hiring", "recruitment", "vacancy"
    ]
    text = " ".join([
        signal.get("signal_summary", ""),
        " ".join(signal.get("key_facts", [])),
        signal.get("evidence_quote", "")
    ]).lower()
    for indicator in career_indicators:
        if indicator in text:
            return False, f"Contains: {indicator}"
    return True, "clean"


def process_raw_content_batch(batch_size=20, force_filter=None,
                               db_path="storage/counselai.db",
                               min_confidence=0.6):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)

    raw_items = get_unprocessed_content(
        force=force_filter, limit=batch_size, db_path=db_path
    )
    if not raw_items:
        return {"processed": 0, "signals_extracted": 0}

    logger.info(f"Processing {len(raw_items)} raw items...")
    results = {
        "processed": 0, "signals_extracted": 0,
        "not_relevant": 0, "low_confidence": 0,
        "career_contaminated": 0, "errors": 0
    }

    for item in raw_items:
        try:
            signal = extract_signal_from_content(
                item["content"], item["force"], client
            )
            if signal is None:
                mark_content_processed(item["id"], "error", db_path)
                results["errors"] += 1
                continue

            if not signal.get("is_relevant", False):
                mark_content_processed(item["id"], "not_relevant", db_path)
                results["not_relevant"] += 1
                results["processed"] += 1
                continue

            confidence = signal.get("confidence", 0)
            if confidence < min_confidence:
                mark_content_processed(item["id"], "low_confidence", db_path)
                results["low_confidence"] += 1
                results["processed"] += 1
                continue

            is_pure, reason = validate_signal_purity(signal)
            if not is_pure:
                mark_content_processed(item["id"], "contaminated", db_path)
                results["career_contaminated"] += 1
                results["processed"] += 1
                continue

            signal_data = {
                "force": item["force"],
                "source_id": item["source_id"],
                "source_name": item["source_name"],
                "signal_summary": signal.get("signal_summary"),
                "key_facts": signal.get("key_facts", []),
                "geographic_scope": signal.get("geographic_scope"),
                "timeline": signal.get("timeline"),
                "magnitude": signal.get("magnitude"),
                "direction": signal.get("direction"),
                "evidence_quote": signal.get("evidence_quote"),
                "full_evidence_text": item["content"][:1000],
                "confidence": confidence,
                "credibility": "medium",
                "raw_content_id": item["id"]
            }
            store_force_signal(signal_data, db_path)
            mark_content_processed(item["id"], "success", db_path)

            results["signals_extracted"] += 1
            results["processed"] += 1

            logger.info(
                f"✅ Signal [{item['force']}]: "
                f"{signal.get('signal_summary','')[:60]}..."
            )
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing item {item['id']}: {e}")
            mark_content_processed(item["id"], "error", db_path)
            results["errors"] += 1

    return results


def run_extraction_cycle(db_path="storage/counselai.db"):
    logger.info("🔍 Starting LLM extraction cycle...")
    start_time = datetime.utcnow()
    min_confidence = float(os.getenv("MIN_CONFIDENCE", 0.6))

    total = {
        "processed": 0, "signals_extracted": 0,
        "not_relevant": 0, "low_confidence": 0,
        "career_contaminated": 0, "errors": 0
    }

    while True:
        batch = process_raw_content_batch(
            batch_size=20, db_path=db_path,
            min_confidence=min_confidence
        )
        for key in total:
            total[key] += batch.get(key, 0)
        if batch["processed"] == 0:
            break
        time.sleep(2)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"EXTRACTION COMPLETE in {duration:.1f}s")
    logger.info(f"Signals extracted: {total['signals_extracted']}")
    logger.info(f"Not relevant: {total['not_relevant']}")
    logger.info(f"Low confidence: {total['low_confidence']}")
    logger.info(f"Contaminated: {total['career_contaminated']}")
    logger.info(f"Errors: {total['errors']}")

    return total
