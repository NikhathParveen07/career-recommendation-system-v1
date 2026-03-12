# CounselAI Knowledge Pipeline
## Setup and Running Instructions

---

## What This Is

A continuous monitoring pipeline that:
- Watches 30+ Indian government, news, and data sources
- Detects new content every 6 hours automatically
- Collects pure force signals about India's economy and society
- Stores everything in a structured database
- Ready for LLM extraction in Stage 2

---

## Local Setup (First Time)

### Step 1 — Install Python dependencies
```bash
pip install feedparser requests schedule python-dotenv groq chromadb
```

### Step 2 — Set up environment variables
```bash
cp .env.template .env
```
Open `.env` and fill in:
- `GROQ_API_KEY` — get free at console.groq.com
- `NEWS_API_KEY` — get free at newsapi.org (100 requests/day)

### Step 3 — Test the pipeline (single run)
```bash
python scheduler.py --once
```
This runs one complete monitoring cycle and exits.
Check the output to see what was collected.

### Step 4 — Run continuously
```bash
python scheduler.py
```
Pipeline runs every 6 hours automatically.
Press Ctrl+C to stop.

### Step 5 — Check dashboard anytime
```bash
python scheduler.py --dashboard
```

---

## Cloud Deployment on Render (Free Tier)

### Why Render
- Free tier supports background workers
- Runs 24/7 without your laptop
- Persistent disk for database storage
- Easy GitHub integration

### Steps

1. Push this project to GitHub
```bash
git init
git add .
git commit -m "Initial CounselAI pipeline"
git remote add origin https://github.com/YOUR_USERNAME/counselai-pipeline
git push -u origin main
```

2. Go to render.com
   - Create free account
   - Click "New" → "Blueprint"
   - Connect your GitHub repo
   - Render reads render.yaml automatically

3. Set environment variables in Render dashboard
   - GROQ_API_KEY
   - NEWS_API_KEY

4. Deploy — pipeline starts running automatically

---

## Project Structure

```
counselai_pipeline/
├── config/
│   └── sources.json          All 30+ sources configured
├── monitoring/
│   ├── rss_monitor.py        RSS feed polling
│   ├── news_monitor.py       NewsAPI polling
│   └── worldbank_monitor.py  World Bank API
├── storage/
│   └── database.py           SQLite database manager
├── logs/
│   └── pipeline.log          Auto-created
├── scheduler.py              Main entry point
├── render.yaml               Cloud deployment config
├── requirements.txt          Python dependencies
└── .env.template             Environment variables template
```

---

## What Gets Collected

### Force 1 — Technology
Sources: NASSCOM, ET Tech, PIB Science, News API
What: Technology changes affecting Indian workforce

### Force 2 — Government Policy  
Sources: PIB (all ministries), PRS India, NITI Aayog
What: Policy signals, budget allocations, schemes

### Force 3 — Demographics
Sources: ET Economy, PIB Census, World Bank Population
What: Population shifts, urbanisation, middle class

### Force 4 — Global Market
Sources: RBI, Ministry of External Affairs, World Bank Trade
What: Export demand, migration opportunities, trade flows

### Force 5 — Education Supply
Sources: Ministry of Education PIB, UGC, World Bank Education
What: Graduate output, enrollment trends, skill gaps

### Force 6 — Climate
Sources: MNRE, PIB Environment, Carbon Brief, ET Energy
What: Green commitments, renewable targets, transition signals

### Force 7 — Social Shifts
Sources: The Hindu Society, Live Mint Consumer, ET News
What: Behaviour changes, lifestyle shifts, social awareness

---

## Checking What Was Collected

```python
from storage.database import get_pipeline_stats, get_db_connection

# Quick stats
stats = get_pipeline_stats()
print(stats)

# See recent raw content
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT force, title, fetched_at 
    FROM raw_content 
    ORDER BY fetched_at DESC 
    LIMIT 20
""")
for row in cursor.fetchall():
    print(row['force'], '|', row['title'][:60])
conn.close()
```

---

## Important Notes

1. RSS feeds are free — no limits
2. NewsAPI free tier: 100 requests/day — used carefully
3. World Bank API: free, no key needed
4. All content stored locally in SQLite
5. No career mentions in stored signals — pure force data only
6. Pipeline is idempotent — safe to restart anytime

---

## Next Steps After Pipeline is Running

Once pipeline collects data for 1-2 weeks:

Stage 2 — LLM Extraction
Run LLM on collected raw content
Extract pure force signals
Store in force_signals table

Stage 3 — Validation
Check signal quality
Score confidence
Filter noise

Stage 4 — Knowledge Graph Integration
Build career knowledge base
Use semantic matching to connect
force signals to careers
No hardcoding — discovered automatically
