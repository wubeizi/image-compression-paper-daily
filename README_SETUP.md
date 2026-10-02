# Setup

This repository uses arXiv plus Google Scholar search results obtained through SerpApi. OpenAlex is not used.

## 1. Configure SerpApi

Create a SerpApi account and copy the API key from the SerpApi dashboard.

Do not put the key in `paper_daily.py`, `README.md`, `papers.json`, or any committed file.

## 2. Add the GitHub Actions secret

Open:

`Repository -> Settings -> Secrets and variables -> Actions -> New repository secret`

Use:

`Name: SERPAPI_API_KEY`

`Value: <your SerpApi API key>`

## 3. Enable GitHub Pages

Open:

`Repository -> Settings -> Pages`

Under **Build and deployment**, set **Source** to **GitHub Actions**.

The repository already contains the Pages deployment logic in `.github/workflows/daily.yml`.

## 4. Run it manually

Open:

`Actions -> Update and Deploy Image Compression Papers -> Run workflow`

The workflow will:

1. collect papers from arXiv and Google Scholar;
2. update `README.md`, `papers.json`, `history.json`, and `docs/index.md`;
3. build the Jekyll site from `docs/`;
4. deploy the result to GitHub Pages.

## 5. Automatic schedule

The default schedule is 09:10 Asia/Tokyo every day.
