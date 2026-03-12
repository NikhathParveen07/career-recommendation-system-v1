import json
import os
import time
from pathlib import Path
from groq import Groq

# ── Setup ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OUTPUT_DIR   = Path("career_pipeline/careers")

# ── Full Career List ──────────────────────────────────────────────────
CAREERS = {
    "Science": [
        "Software Engineer", "Data Scientist", "AI ML Engineer",
        "Cybersecurity Analyst", "Cloud Architect", "Robotics Engineer",
        "Aerospace Engineer", "Civil Engineer", "Mechanical Engineer",
        "Electrical Engineer", "Electronics Engineer", "Chemical Engineer",
        "Biomedical Engineer", "Environmental Engineer",
        "Renewable Energy Engineer", "Doctor MBBS", "Dentist BDS",
        "Pharmacist", "Nurse", "Physiotherapist", "Research Scientist",
        "Biotechnologist", "Forensic Scientist", "Geologist",
        "Agricultural Scientist", "Space Technology Engineer",
        "Electric Vehicle Engineer", "Drone Technology Engineer",
        "Food Scientist", "Veterinarian"
    ],
    "Commerce": [
        "Chartered Accountant", "Cost Accountant CMA",
        "Company Secretary", "Investment Banker", "Financial Analyst",
        "Actuary", "Stock Broker", "Insurance Professional",
        "Tax Consultant", "Auditor", "Business Analyst",
        "Marketing Manager", "Human Resources Manager",
        "Supply Chain Manager", "Operations Manager", "Retail Manager",
        "Logistics Manager", "Import Export Specialist", "Economist",
        "Economic Policy Analyst", "Fintech Professional",
        "Ecommerce Manager", "Digital Marketing Manager",
        "Product Manager", "Entrepreneur Startup Founder",
        "Real Estate Professional", "Tourism and Travel Manager",
        "Healthcare Administrator", "Education Administrator",
        "Bank Officer"
    ],
    "Arts": [
        "Lawyer Advocate", "Civil Services IAS IPS",
        "State Government Services PSC", "Political Scientist",
        "Policy Analyst", "Journalist", "Content Writer",
        "PR Specialist", "Advertising Professional",
        "Film Media Professional", "Podcast Content Creator",
        "Radio Jockey Anchor", "Graphic Designer", "UI UX Designer",
        "Fashion Designer", "Animator", "Game Designer",
        "Interior Designer", "AR VR Developer",
        "Psychologist Counsellor", "Social Worker",
        "NGO Professional", "Archaeologist", "Linguist Translator",
        "Museum Curator", "Teacher Professor", "Sports Professional",
        "Chef Culinary Arts", "Event Manager",
        "Hotel Management Professional"
    ],
    "Vocational": [
        "Electrician", "Plumber", "Construction Supervisor",
        "HVAC Technician", "Interior Decorator",
        "CNC Machine Operator", "Welder Fabricator",
        "Industrial Mechanic", "Quality Control Inspector",
        "Fitter and Turner", "Automobile Mechanic",
        "Commercial Vehicle Driver", "Aviation Technician",
        "Merchant Navy Rating", "Railway Technician",
        "Hardware Technician", "Network Technician",
        "Mobile Repair Technician", "Solar Panel Technician",
        "CCTV Security Systems Technician",
        "Beauty Professional Cosmetologist", "Yoga Instructor",
        "Fitness Trainer", "Spa Therapist", "Cook Kitchen Professional",
        "Bakery Professional", "Barista Cafe Professional",
        "Agricultural Technician", "Horticulture Professional",
        "Dairy Technician"
    ]
}

# ── Prompt ────────────────────────────────────────────────────────────
PROMPT = """
You are an expert Indian career counsellor with 20 years experience.
You have deep knowledge of Indian education system, job market,
entrance exams, colleges, and social context.

Generate complete and accurate data for this career in India:
Career: {title}
Stream: {stream}

Be realistic and honest. Not too optimistic. Think about what
actually happens in India for this career — not what should happen.

Return ONLY a valid JSON object with exactly these fields.
No explanation before or after. Just JSON.

{{
  "career_id": "{career_id}",
  "title": "{title}",
  "stream": "{stream}",

  "description": "3 sentences. What does this person actually do every day in India. Be specific.",

  "riasec_primary": "one of: Realistic Investigative Artistic Social Enterprising Conventional",
  "riasec_secondary": "one of: Realistic Investigative Artistic Social Enterprising Conventional",
  "riasec_scores": {{
    "Realistic": 0,
    "Investigative": 0,
    "Artistic": 0,
    "Social": 0,
    "Enterprising": 0,
    "Conventional": 0
  }},
  "work_values": ["value1", "value2", "value3"],

  "eligible_streams": ["list which Class 12 streams can enter this career"],

  "entrance_exams": ["exact Indian exam names like JEE Main, NEET, CLAT, CAT etc. Empty list if none."],
  "degree_name": "exact degree name used in India",
  "degree_duration_years": 4,
  "course_cost_lakhs": {{
    "government_college": 3,
    "private_college": 15
  }},

  "top_govt_colleges": ["5 real Indian government college names"],
  "top_private_colleges": ["5 real Indian private college names"],

  "salary_lpa": {{
    "entry_0_2_years": 4,
    "mid_3_7_years": 10,
    "senior_8_15_years": 22,
    "top_earner": 45
  }},

  "india_market": {{
    "top_hiring_cities": ["city1", "city2", "city3", "city4"],
    "govt_vs_private": "20% govt 80% private",
    "demand_trend": "growing",
    "approximate_annual_openings_india": 50000
  }},

  "career_progression": {{
    "year_0_2": "what the first 2 years actually look like in India",
    "year_3_5": "what years 3 to 5 look like",
    "year_6_10": "what years 6 to 10 look like",
    "senior_paths": ["path1", "path2", "path3"],
    "entrepreneurship_exit": "how someone starts a business from this career in India",
    "global_pathway": "realistic opportunities abroad for Indians in this career"
  }},

  "india_social_context": {{
    "family_acceptance": "very_high or high or medium or low",
    "regional_availability": "national or metro_only or limited",
    "first_year_reality": "honest truth about what starting out actually feels like in India",
    "common_misconceptions": ["misconception1", "misconception2"]
  }},

  "force_connections": {{}}
}}
"""

# ── Builder ───────────────────────────────────────────────────────────

def make_career_id(stream, index):
    prefix = stream[:3].upper()
    return f"{prefix}_{str(index+1).zfill(3)}"


def build_single_career(career_id, title, stream, client):
    """Build one complete career using LLM"""

    prompt = PROMPT.format(
        career_id = career_id,
        title     = title,
        stream    = stream
    )

    try:
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Indian career counsellor. "
                        "You know everything about Indian education, "
                        "entrance exams, colleges, salaries, and job market. "
                        "You always return valid JSON only. "
                        "You are honest and realistic about Indian career realities."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature = 0.2,
            max_tokens  = 2000
        )

        raw = response.choices[0].message.content.strip()

        # Clean JSON if wrapped in code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return data

    except json.JSONDecodeError as e:
        print(f"    JSON error: {e}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def save_career(career_data):
    """Save career JSON to file"""
    stream    = career_data["stream"].lower()
    stream_dir = OUTPUT_DIR / stream
    stream_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{career_data['career_id']}.json"
    filepath = stream_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(career_data, f, indent=2, ensure_ascii=False)


def build_all():
    """Build all 120 careers"""

    client = Groq(api_key=GROQ_API_KEY)

    total   = 0
    success = 0
    failed  = []

    print("=" * 60)
    print("CounselAI Career Builder — 120 Careers")
    print("=" * 60)

    for stream, titles in CAREERS.items():
        print(f"\nStream: {stream}")
        print("-" * 40)

        for index, title in enumerate(titles):
            career_id = make_career_id(stream, index)

            # Skip if already built
            stream_dir = OUTPUT_DIR / stream.lower()
            filepath   = stream_dir / f"{career_id}.json"
            if filepath.exists():
                print(f"  SKIP {career_id}: {title} (already built)")
                success += 1
                total   += 1
                continue

            total += 1
            print(f"  Building {career_id}: {title}...")

            data = build_single_career(
                career_id, title, stream, client
            )

            if data:
                save_career(data)
                success += 1
                print(f"  DONE {career_id}: {title}")
            else:
                failed.append(f"{career_id}: {title}")
                print(f"  FAILED {career_id}: {title}")

            # Respect rate limits
            time.sleep(1.5)

    # Build index
    build_index()

    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE")
    print(f"Success : {success} / {total}")
    print(f"Failed  : {len(failed)}")
    if failed:
        print("Failed careers:")
        for f in failed:
            print(f"  {f}")
    print("=" * 60)


def build_index():
    """Build master index of all careers"""
    index = {"total": 0, "careers": []}

    for json_file in sorted(OUTPUT_DIR.rglob("*.json")):
        if json_file.name == "career_index.json":
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                career = json.load(f)
            index["careers"].append({
                "career_id"        : career.get("career_id"),
                "title"            : career.get("title"),
                "stream"           : career.get("stream"),
                "riasec_primary"   : career.get("riasec_primary"),
                "riasec_secondary" : career.get("riasec_secondary"),
                "demand_trend"     : career.get(
                    "india_market", {}
                ).get("demand_trend"),
                "family_acceptance": career.get(
                    "india_social_context", {}
                ).get("family_acceptance"),
                "eligible_streams" : career.get("eligible_streams", [])
            })
        except Exception as e:
            print(f"Index error for {json_file}: {e}")

    index["total"] = len(index["careers"])

    output = OUTPUT_DIR / "career_index.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex built: {index['total']} careers saved")


def test_one():
    """Test with single career before running all"""
    client = Groq(api_key=GROQ_API_KEY)
    print("Testing with: Software Engineer")
    data = build_single_career(
        "SCI_001", "Software Engineer", "Science", client
    )
    if data:
        print(json.dumps(data, indent=2))
    else:
        print("FAILED — check your GROQ_API_KEY")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"

    if cmd == "test":
        test_one()
    elif cmd == "build":
        build_all()
    elif cmd == "index":
        build_index()
