# Image Compression Paper Daily

This repository is configured to collect recent papers in:

- Learned Image Compression
- Generative Image Compression
- Extreme Image Compression
- Vector Quantization

The final README is generated automatically by `paper_daily.py`. Once the first GitHub Actions run completes, this file will contain the current paper tables.

## Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python paper_daily.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python paper_daily.py
```

## Configuration

Edit `TOPICS` near the top of `paper_daily.py` to change the research vocabulary.

Useful environment variables:

```text
PAPER_TIMEZONE=Asia/Tokyo
LOOKBACK_DAYS=60
PER_QUERY=25
PER_TOPIC=15
CONTACT_EMAIL=you@example.com
```

`CONTACT_EMAIL` is optional and is used as an API contact address for the requests.

## GitHub Actions

The workflow in `.github/workflows/daily.yml` runs every day at 09:10 Asia/Tokyo and can also be started manually with **Run workflow**.

The Actions job updates:

- `README.md` — human-readable latest results
- `papers.json` — current structured results
- `history.json` — all papers seen by this collector
