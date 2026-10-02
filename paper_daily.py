#!/usr/bin/env python3
"""Daily paper collector for learned/generative/extreme image compression and VQ.

Sources:
- arXiv official Atom API
- OpenAlex Works API

The script is intentionally dependency-light and keeps structured JSON as the
source of truth. README.md is generated from that structured data.
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PAPERS_FILE = BASE_DIR / "papers.json"
HISTORY_FILE = BASE_DIR / "history.json"
README_FILE = BASE_DIR / "README.md"

TIMEZONE = ZoneInfo(os.getenv("PAPER_TIMEZONE", "Asia/Tokyo"))
RUN_AT = datetime.now(TIMEZONE)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "60"))
PER_QUERY = int(os.getenv("PER_QUERY", "25"))
PER_TOPIC = int(os.getenv("PER_TOPIC", "15"))
ARXIV_DELAY = float(os.getenv("ARXIV_DELAY", "3.0"))
OPENALEX_DELAY = float(os.getenv("OPENALEX_DELAY", "0.5"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# Optional: set this as a GitHub Actions repository variable or local env var.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"

# The search vocabulary is deliberately redundant. The relevance filter below
# removes many broad/noisy matches, while the multiple queries improve recall.
TOPICS: dict[str, dict[str, Any]] = {
    "Learned Image Compression": {
        "queries": [
            'learned image compression',
            'learned image coding',
            'neural image compression',
            'deep image compression',
            'end-to-end image compression',
            'learned lossy image compression',
        ],
        "required_any": [
            "compression", "compress", "coding", "codec", "rate-distortion",
            "entropy model", "entropy coding",
        ],
        "exclude_any": [],
    },
    "Generative Image Compression": {
        "queries": [
            'generative image compression',
            'generative compression image',
            'generative learned image compression',
            'generative image coding',
            'diffusion image compression',
            'generative compression diffusion',
        ],
        "required_any": [
            "compression", "compress", "coding", "codec", "rate-distortion",
        ],
        "exclude_any": [
            "text compression", "audio compression", "video compression only",
        ],
    },
    "Extreme Image Compression": {
        "queries": [
            'extreme image compression',
            'extreme compression image',
            'ultra low bitrate image compression',
            'very low bitrate image compression',
            'low bitrate learned image compression',
            'perceptual image compression low bitrate',
        ],
        "required_any": [
            "image", "compression", "compress", "codec", "coding", "bitrate",
        ],
        "exclude_any": [
            "text compression", "model compression only", "video compression only",
        ],
    },
    "Vector Quantization": {
        "queries": [
            'vector quantization image compression',
            'vector-quantized image compression',
            'vector quantised image compression',
            'vq image compression',
            'residual vector quantization image compression',
            'rvq image compression',
            'vq-vae image compression',
        ],
        "required_any": [
            "image", "compression", "compress", "codec", "coding", "rate-distortion",
        ],
        "exclude_any": [
            "text compression", "audio compression", "speech compression", "video compression only",
        ],
    },
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "image-compression-paper-daily/1.0"
        + (f" (mailto:{CONTACT_EMAIL})" if CONTACT_EMAIL else "")
    )
})


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else default
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to load {path}: {exc}")
        return default


def save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def clean_text(text: str | None) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    return normalize_text(title)


def compact_authors(authors: list[str], limit: int = 3) -> str:
    authors = [clean_text(x) for x in authors if clean_text(x)]
    if not authors:
        return "-"
    shown = authors[:limit]
    result = ", ".join(shown)
    if len(authors) > limit:
        result += " et al."
    return result


def extract_github_url(text: str) -> str | None:
    match = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text or "")
    return match.group(0).rstrip(".,)") if match else None


def paper_id(paper: dict[str, Any]) -> str:
    """Cross-source stable ID based primarily on normalized title.

    arXiv and OpenAlex frequently represent the same work with different IDs.
    A title-based key therefore gives us a simple cross-source deduplication
    mechanism while still retaining arXiv IDs and DOIs as metadata.
    """
    title_key = normalize_title(paper.get("title", ""))
    if title_key:
        return "title:" + hashlib.sha1(title_key.encode("utf-8")).hexdigest()

    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"

    if paper.get("doi"):
        doi = paper["doi"].lower().strip()
        doi = re.sub(r"^https?://doi.org/", "", doi)
        return f"doi:{doi}"

    url = paper.get("paper_url") or ""
    if url:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        url_key = f"{parsed.netloc.lower()}{path.lower()}"
        if url_key:
            return f"url:{url_key}"

    return "paper:unknown"


def score_relevance(paper: dict[str, Any], topic: dict[str, Any]) -> int:
    """Small deterministic score used only to remove obvious noise."""
    title = normalize_text(paper.get("title", ""))
    abstract = normalize_text(paper.get("abstract", ""))
    combined = f"{title} {abstract}"

    if any(phrase in combined for phrase in topic.get("exclude_any", [])):
        return -100

    required = [normalize_text(x) for x in topic.get("required_any", [])]
    hits = sum(1 for item in required if item and item in combined)

    score = 0
    if any(item and item in title for item in required):
        score += 3
    score += min(hits, 5)

    compression_terms = [
        "image compression", "image coding", "image codec", "neural compression",
        "learned compression", "rate distortion", "rate-distortion", "entropy model",
    ]
    score += sum(2 for item in compression_terms if item in combined)

    return score


def merge_paper(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if not existing:
        return new

    merged = dict(existing)
    for key, value in new.items():
        if value not in (None, "", [], {}):
            merged[key] = value

    merged["topics"] = sorted(set(existing.get("topics", [])) | set(new.get("topics", [])))
    merged["matched_queries"] = sorted(
        set(existing.get("matched_queries", [])) | set(new.get("matched_queries", []))
    )
    return merged


# -----------------------------------------------------------------------------
# arXiv
# -----------------------------------------------------------------------------

ATOM = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def arxiv_text(element: ET.Element | None, path: str = "") -> str:
    if element is None:
        return ""
    child = element.find(path) if path else element
    return clean_text(child.text if child is not None else "")


def search_arxiv(query: str, max_results: int = PER_QUERY) -> list[dict[str, Any]]:
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = SESSION.get(ARXIV_API, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    papers: list[dict[str, Any]] = []

    for entry in root.findall(f"{{{ATOM}}}entry"):
        title = arxiv_text(entry, f"{{{ATOM}}}title")
        summary = arxiv_text(entry, f"{{{ATOM}}}summary")
        published = arxiv_text(entry, f"{{{ATOM}}}published")
        arxiv_url = arxiv_text(entry, f"{{{ATOM}}}id")

        arxiv_id = arxiv_url.rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        authors = []
        for author in entry.findall(f"{{{ATOM}}}author"):
            name = arxiv_text(author, f"{{{ATOM}}}name")
            if name:
                authors.append(name)

        pdf_url = None
        for link in entry.findall(f"{{{ATOM}}}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        comment = arxiv_text(entry, f"{{{ARXIV_NS}}}comment")
        code_url = extract_github_url(comment)

        try:
            date = datetime.fromisoformat(published.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            date = RUN_AT.date().isoformat()

        papers.append({
            "title": clean_text(title),
            "authors": authors,
            "date": date,
            "paper_url": arxiv_url,
            "pdf_url": pdf_url or arxiv_url,
            "code_url": code_url,
            "abstract": summary,
            "source": "arXiv",
            "arxiv_id": arxiv_id,
            "doi": None,
        })

    time.sleep(ARXIV_DELAY)
    return papers


# -----------------------------------------------------------------------------
# OpenAlex
# -----------------------------------------------------------------------------


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""

    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions.append((index, word))

    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def search_openalex(query: str, max_results: int = PER_QUERY) -> list[dict[str, Any]]:
    since = (RUN_AT.date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    params = {
        "search": query,
        "per-page": max_results,
        "sort": "publication_date:desc",
        "filter": f"from_publication_date:{since}",
    }
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL

    response = SESSION.get(OPENALEX_API, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()

    results = response.json().get("results", [])
    papers: list[dict[str, Any]] = []

    for item in results:
        authors = []
        for authorship in item.get("authorships", []):
            name = (authorship.get("author") or {}).get("display_name")
            if name:
                authors.append(name)

        primary = item.get("primary_location") or {}
        paper_url = primary.get("landing_page_url")
        if not paper_url:
            doi = item.get("doi")
            if doi:
                paper_url = doi
        if not paper_url:
            paper_url = item.get("id")

        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        open_access = item.get("open_access") or {}
        if not paper_url:
            paper_url = open_access.get("oa_url")

        papers.append({
            "title": clean_text(item.get("title")),
            "authors": authors,
            "date": item.get("publication_date"),
            "paper_url": paper_url,
            "pdf_url": (primary.get("pdf_url") if primary else None),
            "code_url": None,
            "abstract": abstract,
            "source": "OpenAlex",
            "arxiv_id": None,
            "doi": item.get("doi"),
            "openalex_id": item.get("id"),
            "cited_by_count": item.get("cited_by_count", 0),
        })

    time.sleep(OPENALEX_DELAY)
    return papers


# -----------------------------------------------------------------------------
# Collection + deduplication
# -----------------------------------------------------------------------------


def collect() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    history: dict[str, dict[str, Any]] = load_json(HISTORY_FILE, {})
    by_topic: dict[str, list[dict[str, Any]]] = {}
    current_run: dict[str, dict[str, Any]] = {}

    for topic_name, topic in TOPICS.items():
        candidates: dict[str, dict[str, Any]] = {}
        print(f"\n=== {topic_name} ===")

        for query in topic["queries"]:
            print(f"Searching arXiv: {query}")
            try:
                source_papers = search_arxiv(query, PER_QUERY)
            except Exception as exc:
                print(f"  arXiv failed: {exc}")
                source_papers = []

            for paper in source_papers:
                score = score_relevance(paper, topic)
                if score < 5:
                    continue
                pid = paper_id(paper)
                paper["matched_queries"] = [query]
                paper["topics"] = [topic_name]
                paper["relevance_score"] = score
                candidates[pid] = merge_paper(candidates.get(pid), paper)

        for query in topic["queries"]:
            print(f"Searching OpenAlex: {query}")
            try:
                source_papers = search_openalex(query, PER_QUERY)
            except Exception as exc:
                print(f"  OpenAlex failed: {exc}")
                source_papers = []

            for paper in source_papers:
                score = score_relevance(paper, topic)
                if score < 5:
                    continue
                pid = paper_id(paper)
                paper["matched_queries"] = [query]
                paper["topics"] = [topic_name]
                paper["relevance_score"] = score
                candidates[pid] = merge_paper(candidates.get(pid), paper)

        papers = list(candidates.values())
        papers.sort(
            key=lambda p: (
                p.get("date") or "0000-00-00",
                p.get("relevance_score", 0),
            ),
            reverse=True,
        )
        papers = papers[:PER_TOPIC]
        by_topic[topic_name] = papers

        for paper in papers:
            pid = paper_id(paper)
            current_run[pid] = merge_paper(current_run.get(pid), paper)

    return by_topic, current_run


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------


def update_history(current_run: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    history: dict[str, dict[str, Any]] = load_json(HISTORY_FILE, {})
    previously_seen = set(history.keys())

    for pid, paper in current_run.items():
        history[pid] = merge_paper(history.get(pid), paper)
        history[pid]["last_seen"] = RUN_AT.date().isoformat()
        history[pid]["first_seen"] = history[pid].get("first_seen", RUN_AT.date().isoformat())

    save_json(HISTORY_FILE, history)
    return history, previously_seen


# -----------------------------------------------------------------------------
# Markdown generation
# -----------------------------------------------------------------------------


def md_escape(text: str) -> str:
    return clean_text(text).replace("|", "\\|")


def format_paper_row(paper: dict[str, Any], is_new: bool = False) -> str:
    title = md_escape(paper.get("title", "Untitled"))
    title = f"**NEW** {title}" if is_new else title

    authors = md_escape(compact_authors(paper.get("authors", [])))
    date = paper.get("date") or "-"
    source = paper.get("source") or "-"

    paper_url = paper.get("paper_url") or paper.get("pdf_url")
    paper_cell = f"[Paper]({paper_url})" if paper_url else "-"

    code_url = paper.get("code_url")
    code_cell = f"[GitHub]({code_url})" if code_url else "-"

    return f"| {date} | {title} | {authors} | {paper_cell} | {code_cell} | {source} |\n"


def generate_readme(by_topic: dict[str, list[dict[str, Any]]], current_run: dict[str, dict[str, Any]], previously_seen: set[str]) -> None:
    new_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic_name, papers in by_topic.items():
        for paper in papers:
            if paper_id(paper) not in previously_seen:
                new_by_topic[topic_name].append(paper)

    total_new = sum(len(v) for v in new_by_topic.values())
    total_current = len(current_run)

    lines: list[str] = [
        "# Image Compression Paper Daily",
        "",
        "> Automatically updated from arXiv and OpenAlex.",
        "> Focus: learned image compression, generative compression, extreme/low-bitrate compression, and vector quantization.",
        "",
        f"**Last updated:** {RUN_AT.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"**Papers collected this run:** {total_current}  |  **New since previous run:** {total_new}",
        "",
        "## Topics",
        "",
    ]

    for topic_name in TOPICS:
        anchor = topic_name.lower().replace(" ", "-")
        lines.append(f"- [{topic_name}](#{anchor})")

    lines += ["", "## New papers", ""]

    if total_new == 0:
        lines.append("No previously unseen papers were detected in the latest run.")
        lines.append("")
    else:
        for topic_name in TOPICS:
            papers = new_by_topic.get(topic_name, [])
            if not papers:
                continue
            lines += [f"### {topic_name}", "", "| Date | Title | Authors | Paper | Code | Source |", "|---|---|---|---|---|---|"]
            for paper in papers[:PER_TOPIC]:
                lines.append(format_paper_row(paper, is_new=True).rstrip())
            lines.append("")

    lines += ["## Latest papers by topic", ""]

    for topic_name in TOPICS:
        papers = by_topic.get(topic_name, [])
        anchor = topic_name.lower().replace(" ", "-")
        lines += [f"## {topic_name}", "", "| Date | Title | Authors | Paper | Code | Source |", "|---|---|---|---|---|---|"]
        if not papers:
            lines.append("| - | No matching papers found | - | - | - | - |")
        else:
            for paper in papers:
                lines.append(format_paper_row(paper).rstrip())
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- arXiv is queried through its official API rather than scraping the HTML search page. The API supports structured queries and date sorting. See the [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html).",
        "- OpenAlex is used as a second index to improve recall beyond arXiv-only searches.",
        "- `history.json` stores papers already observed so that later runs can mark newly discovered items.",
        "- Search vocabulary is intentionally broad; the script applies a deterministic title/abstract relevance filter before publishing results.",
        "",
    ]

    README_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_current_run(by_topic: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        "generated_at": RUN_AT.isoformat(),
        "timezone": str(TIMEZONE),
        "lookback_days": LOOKBACK_DAYS,
        "topics": by_topic,
    }
    save_json(PAPERS_FILE, payload)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print(f"Running paper collector at {RUN_AT.isoformat()}")
    print(f"Lookback={LOOKBACK_DAYS}d, per_query={PER_QUERY}, per_topic={PER_TOPIC}")

    by_topic, current_run = collect()
    _, previously_seen = update_history(current_run)
    save_current_run(by_topic)
    generate_readme(by_topic, current_run, previously_seen)

    print(f"\nFinished: {len(current_run)} unique papers in current run.")


if __name__ == "__main__":
    main()
