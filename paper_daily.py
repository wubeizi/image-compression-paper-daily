#!/usr/bin/env python3
"""High-precision daily paper collector for image compression research.

Sources:
  1) arXiv official API
  2) Google Scholar search results via SerpApi

Why SerpApi instead of scraping scholar.google.com directly?
Direct HTML scraping is brittle and can trigger Google Scholar anti-bot measures.
SerpApi exposes structured Google Scholar organic results and handles the
retrieval/parsing layer. The API key is read from the SERPAPI_API_KEY
environment variable and should never be committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

BASE_DIR = Path(__file__).resolve().parent
PAPERS_FILE = BASE_DIR / "papers.json"
HISTORY_FILE = BASE_DIR / "history.json"
README_FILE = BASE_DIR / "README.md"
DOCS_DIR = BASE_DIR / "docs"
DOCS_INDEX_FILE = DOCS_DIR / "index.md"

TZ = ZoneInfo(os.getenv("PAPER_TIMEZONE", "Asia/Tokyo"))
RUN_AT = datetime.now(TZ)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))
SCHOLAR_NUM = int(os.getenv("SCHOLAR_NUM", "20"))
PER_TOPIC = int(os.getenv("PER_TOPIC", "12"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
ARXIV_DELAY = float(os.getenv("ARXIV_DELAY", "3.0"))
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()

ARXIV_API = "https://export.arxiv.org/api/query"
SERPAPI_API = "https://serpapi.com/search.json"

ARXIV_CATEGORIES = ["cs.CV", "cs.MM", "eess.IV", "cs.IT", "cs.LG"]

# Keep the daily Google Scholar request count small. Four Scholar searches per
# run stay comfortably below SerpApi's current 250-search/month free quota.
# Each topic uses one carefully constructed query instead of many broad queries.
TOPICS: dict[str, dict[str, Any]] = {
    "Learned Image Compression": {
        "scholar_query": '"learned image compression" OR "neural image compression" OR "learned image coding" OR "end-to-end image compression"',
        "arxiv_queries": [
            '(ti:"learned image compression" OR abs:"learned image compression")',
            '(ti:"neural image compression" OR abs:"neural image compression")',
        ],
        "positive": [
            "image compression", "image coding", "image codec", "learned image compression",
            "neural image compression", "rate distortion", "rate-distortion", "entropy model",
            "hyperprior", "latent representation", "learned image coding", "neural image coding",
        ],
        "method_signals": [
            "hyperprior", "entropy model", "entropy coding", "autoregressive", "context model",
            "variational", "vae", "transformer", "attention", "latent", "quantization",
            "rate-distortion", "rate distortion",
        ],
        "exclude": [
            "video compression", "speech compression", "audio compression", "text compression",
            "model compression", "scientific data compression", "point cloud compression",
            "gaussian splatting", "3d gaussian", "mesh compression", "data compression",
        ],
        "gate": "learned",
    },
    "Generative Image Compression": {
        "scholar_query": '"generative image compression" OR "diffusion image compression" OR "generative image coding" OR "generative compression" image',
        "arxiv_queries": [
            '(ti:"generative image compression" OR abs:"generative image compression")',
            '(ti:"diffusion image compression" OR abs:"diffusion image compression")',
        ],
        "positive": [
            "generative image compression", "diffusion image compression", "generative image coding",
            "generative compression", "semantic image compression", "perceptual image compression",
            "semantic coding", "generative codec", "diffusion codec",
        ],
        "method_signals": [
            "diffusion", "generative", "latent diffusion", "score model", "text prompt",
            "semantic", "perceptual", "adversarial", "gan", "text-guided", "generative model",
        ],
        "exclude": [
            "video compression", "speech compression", "audio compression", "text compression",
            "medical image registration", "3d gaussian", "mesh compression", "scientific data",
        ],
        "gate": "generative",
    },
    "Extreme Image Compression": {
        "scholar_query": '"extreme image compression" OR "ultra low bitrate" image compression OR "very low bitrate" image compression OR "low bitrate image compression"',
        "arxiv_queries": [
            '(ti:"extreme image compression" OR abs:"extreme image compression")',
            '(ti:"low bitrate" OR abs:"low bitrate") AND (ti:"image compression" OR abs:"image compression")',
        ],
        "positive": [
            "extreme image compression", "ultra low bitrate", "very low bitrate",
            "low bitrate image compression", "low-rate image compression", "low rate image compression",
            "perceptual image compression", "bits per pixel", "bpp", "rate-distortion",
        ],
        "method_signals": [
            "bpp", "bits per pixel", "psnr", "ms-ssim", "lpips", "dists", "perceptual",
            "semantic", "generative", "diffusion", "rate-distortion", "low bitrate",
        ],
        "exclude": [
            "video compression", "speech compression", "audio compression", "text compression",
            "model compression", "scientific data compression", "point cloud compression",
            "3d gaussian", "mesh compression", "hyperspectral", "medical image", "hardware compression",
        ],
        "gate": "extreme",
    },
    "Vector Quantization": {
        "scholar_query": '("vector quantized" OR "vector-quantized" OR "vector quantization" OR "residual vector quantization" OR RVQ OR "VQ-VAE") AND ("image compression" OR "image coding" OR "image codec")',
        "arxiv_queries": [
            '(ti:"vector quantized" OR abs:"vector quantized" OR ti:"vector-quantization" OR abs:"vector-quantization") AND (ti:"image compression" OR abs:"image compression")',
            '(ti:"residual vector quantization" OR abs:"residual vector quantization") AND (ti:"image compression" OR abs:"image compression")',
        ],
        "positive": [
            "vector quantization", "vector-quantized", "vector quantized", "residual vector quantization",
            "rvq", "vq-vae", "codebook", "codebooks", "quantizer", "quantization", "image compression",
            "image coding", "image codec",
        ],
        "method_signals": [
            "vector quantization", "vector-quantized", "residual vector quantization", "rvq",
            "codebook", "codebooks", "vq-vae", "product quantization", "multi-codebook", "quantizer",
        ],
        "exclude": [
            "video compression", "speech coding", "audio codec", "text compression", "llm compression",
            "language model", "model compression", "point cloud compression", "3d gaussian",
            "mesh compression", "neural audio codec", "speech codec",
        ],
        "gate": "vq",
    },
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "image-compression-paper-daily/3.0"
    + (f" (mailto:{CONTACT_EMAIL})" if CONTACT_EMAIL else "")
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
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+/-]+", " ", text)).strip()


def normalize_title(title: str) -> str:
    title = normalize_text(title)
    title = re.sub(r"\b(v\d+)\b$", "", title).strip()
    return title


def stable_id(paper: dict[str, Any]) -> str:
    title = normalize_title(paper.get("title", ""))
    if title:
        return "title:" + hashlib.sha1(title.encode("utf-8")).hexdigest()
    if paper.get("arxiv_id"):
        return "arxiv:" + paper["arxiv_id"].lower()
    result_id = paper.get("scholar_result_id")
    if result_id:
        return "scholar:" + str(result_id)
    url = paper.get("paper_url") or ""
    parsed = urlparse(url)
    return "url:" + (parsed.netloc + parsed.path).lower()


def compact_authors(authors: list[str], limit: int = 3) -> str:
    names = [clean_text(x) for x in authors if clean_text(x)]
    if not names:
        return "-"
    text = ", ".join(names[:limit])
    return text + (" et al." if len(names) > limit else "")


def extract_github_url(text: str) -> str | None:
    match = re.search(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text or "")
    return match.group(0).rstrip(".,)") if match else None


def github_from_resources(resources: list[dict[str, Any]] | None) -> str | None:
    for resource in resources or []:
        link = resource.get("link", "")
        found = extract_github_url(link)
        if found:
            return found
    return None


def category_query() -> str:
    return " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)


def arxiv_date_range() -> str:
    now_utc = RUN_AT.astimezone(timezone.utc)
    start = now_utc - timedelta(days=LOOKBACK_DAYS)
    return f"[{start.strftime('%Y%m%d%H%M')} TO {now_utc.strftime('%Y%m%d%H%M')}]"


def search_arxiv(query: str) -> list[dict[str, Any]]:
    search_query = f"({query}) AND ({category_query()}) AND submittedDate:{arxiv_date_range()}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": 20,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = SESSION.get(ARXIV_API, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    papers: list[dict[str, Any]] = []
    atom = "http://www.w3.org/2005/Atom"
    arxiv_ns = "http://arxiv.org/schemas/atom"

    for entry in root.findall(f"{{{atom}}}entry"):
        title_el = entry.find(f"{{{atom}}}title")
        summary_el = entry.find(f"{{{atom}}}summary")
        id_el = entry.find(f"{{{atom}}}id")
        published_el = entry.find(f"{{{atom}}}published")
        title = clean_text(title_el.text if title_el is not None else "")
        abstract = clean_text(summary_el.text if summary_el is not None else "")
        arxiv_url = clean_text(id_el.text if id_el is not None else "")
        published = clean_text(published_el.text if published_el is not None else "")
        if not title or not arxiv_url:
            continue

        arxiv_id = re.sub(r"v\d+$", "", arxiv_url.rsplit("/", 1)[-1])
        authors: list[str] = []
        for author in entry.findall(f"{{{atom}}}author"):
            name_el = author.find(f"{{{atom}}}name")
            if name_el is not None and name_el.text:
                authors.append(clean_text(name_el.text))

        pdf_url = None
        for link in entry.findall(f"{{{atom}}}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        comment_el = entry.find(f"{{{arxiv_ns}}}comment")
        journal_el = entry.find(f"{{{arxiv_ns}}}journal_ref")
        comment = clean_text(comment_el.text if comment_el is not None else "")
        journal_ref = clean_text(journal_el.text if journal_el is not None else "")
        categories = [x.attrib.get("term", "") for x in entry.findall(f"{{{atom}}}category")]
        primary_category_el = entry.find(f"{{{arxiv_ns}}}primary_category")
        primary_category = primary_category_el.attrib.get("term") if primary_category_el is not None else None

        try:
            date = datetime.fromisoformat(published.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            date = RUN_AT.date().isoformat()

        papers.append({
            "title": title,
            "authors": authors,
            "date": date,
            "paper_url": arxiv_url,
            "pdf_url": pdf_url or arxiv_url,
            "code_url": extract_github_url(comment),
            "abstract": abstract,
            "source": "arXiv",
            "arxiv_id": arxiv_id,
            "doi": None,
            "comment": comment,
            "journal_ref": journal_ref,
            "categories": categories,
            "primary_category": primary_category,
        })

    time.sleep(ARXIV_DELAY)
    return papers


def search_google_scholar(query: str) -> list[dict[str, Any]]:
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY is not set; add it as a GitHub Actions secret.")

    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "as_sdt": "0,5",
        "as_vis": "0",
        "scisbd": "1",
        "num": SCHOLAR_NUM,
        "filter": "1",
    }
    response = SESSION.get(SERPAPI_API, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    papers: list[dict[str, Any]] = []
    for item in payload.get("organic_results", []):
        title = clean_text(item.get("title"))
        if not title:
            continue

        publication_info = item.get("publication_info") or {}
        authors: list[str] = []
        for author in publication_info.get("authors", []) or []:
            if isinstance(author, dict) and author.get("name"):
                authors.append(clean_text(author["name"]))

        if not authors:
            summary = clean_text(publication_info.get("summary"))
            if summary:
                # Scholar commonly presents "A. Author, B. Author - Venue - year".
                author_part = summary.split(" - ", 1)[0]
                authors = [x.strip() for x in author_part.split(",") if x.strip()]

        paper_url = item.get("link")
        snippet = clean_text(item.get("snippet"))
        year = None
        summary = clean_text(publication_info.get("summary"))
        years = re.findall(r"\b(?:19|20)\d{2}\b", summary)
        if years:
            year = years[-1]

        # Google Scholar gives publication year reliably in many cases, but not
        # a precise publication date for all records. Keep the year explicit.
        date = f"{year}" if year else RUN_AT.date().isoformat()

        resources = item.get("resources") or []
        cited_by = (item.get("inline_links") or {}).get("cited_by") or {}
        versions = (item.get("inline_links") or {}).get("versions") or {}

        papers.append({
            "title": title,
            "authors": authors,
            "date": date,
            "year": year,
            "paper_url": paper_url,
            "pdf_url": None,
            "code_url": github_from_resources(resources),
            "abstract": snippet,
            "source": "Google Scholar",
            "scholar_result_id": item.get("result_id"),
            "scholar_position": item.get("position"),
            "scholar_cited_by": cited_by.get("total") or cited_by.get("value"),
            "scholar_versions": versions.get("total") or versions.get("value"),
            "publication": summary,
        })

    return papers


def merge_papers(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if not old:
        result = dict(new)
        result["sources"] = sorted(set(new.get("sources", [new.get("source", "")])) - {""})
        return result

    merged = dict(old)
    old_source = old.get("source")
    new_source = new.get("source")

    # arXiv is preferred as the primary link/date when the same work is found
    # by both sources. Scholar remains useful for citation/versions metadata.
    if new_source == "arXiv":
        for key in ["paper_url", "pdf_url", "arxiv_id", "primary_category", "categories", "comment", "journal_ref"]:
            if new.get(key):
                merged[key] = new[key]
        merged["date"] = new.get("date") or merged.get("date")
        merged["source"] = "arXiv"
    elif old_source != "arXiv":
        for key in ["paper_url", "scholar_result_id", "scholar_position", "publication", "year"]:
            if new.get(key) and not merged.get(key):
                merged[key] = new[key]

    if len(new.get("abstract", "")) > len(old.get("abstract", "")):
        merged["abstract"] = new["abstract"]
    if not merged.get("code_url") and new.get("code_url"):
        merged["code_url"] = new["code_url"]

    for key in ["doi", "arxiv_id", "scholar_result_id"]:
        if new.get(key):
            merged[key] = merged.get(key) or new[key]

    merged["sources"] = sorted(
        (set(old.get("sources", [old_source] if old_source else []))
         | set(new.get("sources", [new_source] if new_source else []))) - {""}
    )
    merged["topics"] = sorted(set(old.get("topics", [])) | set(new.get("topics", [])))
    merged["matched_queries"] = sorted(set(old.get("matched_queries", [])) | set(new.get("matched_queries", [])))
    merged["signals"] = list(dict.fromkeys(old.get("signals", []) + new.get("signals", [])))[:8]
    for key in ["scholar_cited_by", "scholar_versions"]:
        try:
            merged[key] = max(int(old.get(key) or 0), int(new.get(key) or 0)) or None
        except (TypeError, ValueError):
            pass
    merged["relevance_score"] = max(old.get("relevance_score", 0), new.get("relevance_score", 0))
    return merged


def score_relevance(paper: dict[str, Any], topic: dict[str, Any]) -> tuple[int, list[str]]:
    title = normalize_text(paper.get("title", ""))
    abstract = normalize_text(paper.get("abstract", ""))
    combined = f"{title} {abstract}"

    for bad in topic["exclude"]:
        bad_n = normalize_text(bad)
        if bad_n and bad_n in title:
            return -999, [f"excluded:{bad_n}"]

    title_is_image = any(x in title for x in ["image", "images", "visual"])
    compression_title = any(x in title for x in ["compression", "compress", "coding", "codec", "bitrate", "bpp"])
    image_context = any(x in combined for x in ["image compression", "image coding", "image codec", "image compression"])
    if not title_is_image and not image_context:
        return -100, ["no-image-scope"]
    if not compression_title and not image_context:
        return -100, ["no-compression-scope"]

    score = 0
    reasons: list[str] = []
    positive_hits = 0
    for term in topic["positive"]:
        term_n = normalize_text(term)
        if term_n and term_n in combined:
            score += 4 if term_n in title else 1
            positive_hits += 1

    strong_phrases = [
        "image compression", "image coding", "image codec", "learned image compression",
        "neural image compression", "generative image compression", "diffusion image compression",
        "vector quantized image compression", "residual vector quantization", "extreme image compression",
        "low bitrate image compression", "perceptual image compression", "semantic image compression",
    ]
    for phrase in strong_phrases:
        if phrase in title:
            score += 8
            reasons.append(phrase)
        elif phrase in abstract:
            score += 3
            reasons.append(phrase)

    gate = topic["gate"]
    if gate == "learned":
        if not any(x in combined for x in ["learned", "neural", "deep learning", "transformer", "variational", "vae", "latent", "autoencoder", "hyperprior", "entropy model", "context model", "rate-distortion", "quantization"]):
            return -100, ["learned-gate-failed"]
    elif gate == "generative":
        if not any(x in combined for x in ["generative", "diffusion", "semantic", "gan", "score model"]):
            return -100, ["generative-gate-failed"]
    elif gate == "extreme":
        if not any(x in combined for x in ["extreme", "ultra low bitrate", "very low bitrate", "low bitrate", "low rate", "perceptual", "bpp", "bits per pixel"]):
            return -100, ["extreme-gate-failed"]
    elif gate == "vq":
        if not any(x in combined for x in ["vector quantization", "vector-quantized", "vector quantized", "residual vector quantization", "rvq", "vq-vae", "codebook", "quantizer"]):
            return -100, ["vq-gate-failed"]

    if any(x in title for x in ["video", "speech", "audio", "point cloud", "mesh", "3d gaussian", "gaussian splatting"]) and not title_is_image:
        score -= 15

    if "rate-distortion" in combined or "rate distortion" in combined:
        score += 3
        reasons.append("rate-distortion")
    if any(x in combined for x in ["bpp", "bits per pixel", "bit/pixel", "bits-per-pixel"]):
        score += 3
        reasons.append("bpp")
    if any(x in combined for x in ["psnr", "ms-ssim", "lpips", "dists"]):
        score += 2
        reasons.append("rd-metrics")

    score += min(positive_hits, 5)
    return score, sorted(set(reasons))


def extract_signals(paper: dict[str, Any]) -> list[str]:
    text = normalize_text(paper.get("title", "") + " " + paper.get("abstract", ""))
    signals: list[str] = []

    patterns = [
        (r"(?:^|\s)(\d+(?:\.\d+)?)\s*(?:bpp|bits per pixel)(?:\s|$)", "BPP"),
        (r"(?:psnr)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*d?b?", "PSNR"),
        (r"(?:ms-ssim)\s*(?:of|=|:)?\s*(0?\.\d+|\d+(?:\.\d+)?)", "MS-SSIM"),
        (r"(?:lpips)\s*(?:of|=|:)?\s*(0?\.\d+|\d+(?:\.\d+)?)", "LPIPS"),
        (r"(?:dists)\s*(?:of|=|:) ?\s*(0?\.\d+|\d+(?:\.\d+)?)", "DISTS"),
    ]
    for pattern, label in patterns:
        match = re.search(pattern, text)
        if match:
            signals.append(f"{label}={match.group(1)}")

    method_groups = [
        ("Diffusion", ["diffusion", "latent diffusion", "score model"]),
        ("Generative", ["generative image compression", "generative compression", "generative model"]),
        ("VQ/RVQ", ["vector quantization", "vector-quantized", "residual vector quantization", "rvq", "vq-vae"]),
        ("Hyperprior", ["hyperprior"]),
        ("Entropy model", ["entropy model", "context model", "autoregressive"]),
        ("Transformer", ["transformer", "self-attention", "attention"]),
        ("Semantic", ["semantic image compression", "semantic compression", "semantic coding"]),
        ("Perceptual", ["perceptual image compression", "perceptual quality", "perceptual loss"]),
        ("Low bitrate", ["low bitrate", "ultra low bitrate", "very low bitrate", "low-rate"]),
    ]
    for label, terms in method_groups:
        if any(t in text for t in terms):
            signals.append(label)

    datasets = ["kodak", "clic", "tecnick", "imagenet", "div2k", "flickr2k", "ffhq", "mscoco"]
    found = [d.upper() if d != "mscoco" else "MS COCO" for d in datasets if d in text]
    if found:
        signals.append("Data=" + ", ".join(found[:3]))

    return list(dict.fromkeys(signals))[:8]


def collect() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_topic: dict[str, list[dict[str, Any]]] = {}
    current_run: dict[str, dict[str, Any]] = {}

    for topic_name, topic in TOPICS.items():
        print(f"\n=== {topic_name} ===")
        candidates: dict[str, dict[str, Any]] = {}

        for query in topic["arxiv_queries"]:
            print(f"arXiv: {query}")
            try:
                papers = search_arxiv(query)
            except Exception as exc:
                print(f"  failed: {exc}")
                papers = []
            for paper in papers:
                score, _ = score_relevance(paper, topic)
                if score < 11:
                    continue
                paper["topics"] = [topic_name]
                paper["matched_queries"] = [query]
                paper["relevance_score"] = score
                paper["signals"] = extract_signals(paper)
                pid = stable_id(paper)
                candidates[pid] = merge_papers(candidates.get(pid), paper)

        print(f"Google Scholar: {topic['scholar_query']}")
        try:
            scholar_papers = search_google_scholar(topic["scholar_query"])
        except Exception as exc:
            print(f"  failed: {exc}")
            scholar_papers = []
        for paper in scholar_papers:
            score, _ = score_relevance(paper, topic)
            if score < 11:
                continue
            paper["topics"] = [topic_name]
            paper["matched_queries"] = [topic["scholar_query"]]
            paper["relevance_score"] = score
            paper["signals"] = extract_signals(paper)
            pid = stable_id(paper)
            candidates[pid] = merge_papers(candidates.get(pid), paper)

        papers = list(candidates.values())
        papers.sort(key=lambda p: (str(p.get("date") or "0000"), p.get("relevance_score", 0)), reverse=True)
        by_topic[topic_name] = papers[:PER_TOPIC]

        for paper in by_topic[topic_name]:
            pid = stable_id(paper)
            current_run[pid] = merge_papers(current_run.get(pid), paper)

    return by_topic, current_run


def update_history(current_run: dict[str, dict[str, Any]]) -> set[str]:
    history: dict[str, dict[str, Any]] = load_json(HISTORY_FILE, {})
    previously_seen = set(history.keys())
    for pid, paper in current_run.items():
        history[pid] = merge_papers(history.get(pid), paper)
        history[pid]["first_seen"] = history[pid].get("first_seen", RUN_AT.date().isoformat())
        history[pid]["last_seen"] = RUN_AT.date().isoformat()
    save_json(HISTORY_FILE, history)
    return previously_seen


def md_escape(text: str) -> str:
    return clean_text(text).replace("|", "\\|")


def format_row(paper: dict[str, Any], is_new: bool = False) -> str:
    title = md_escape(paper.get("title", "Untitled"))
    if is_new:
        title = "**NEW** " + title
    signals = ", ".join(paper.get("signals", [])[:6]) or "-"
    paper_url = paper.get("paper_url") or paper.get("pdf_url")
    paper_cell = f"[Paper]({paper_url})" if paper_url else "-"
    code = paper.get("code_url")
    code_cell = f"[GitHub]({code})" if code else "-"
    cites = paper.get("scholar_cited_by")
    citation_cell = str(cites) if cites is not None else "-"
    sources = ", ".join(paper.get("sources", [paper.get("source", "-")])) or "-"
    return (
        f"| {paper.get('date', '-')} | {title} | {md_escape(compact_authors(paper.get('authors', [])))} "
        f"| {md_escape(signals)} | {paper_cell} | {code_cell} | {citation_cell} | {sources} |"
    )


def generate_readme(
    by_topic: dict[str, list[dict[str, Any]]],
    current_run: dict[str, dict[str, Any]],
    previously_seen: set[str],
) -> None:
    new_pids = set(current_run) - previously_seen
    new_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic_name, papers in by_topic.items():
        for paper in papers:
            if stable_id(paper) in new_pids:
                new_by_topic[topic_name].append(paper)

    lines: list[str] = [
        "# Image Compression Paper Daily",
        "",
        "> High-precision daily reading list for learned, generative, low-bitrate image compression and vector quantization.",
        "> Sources: arXiv + Google Scholar (via SerpApi). OpenAlex is not used.",
        "",
        f"**Last updated:** {RUN_AT.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"**Unique papers this run:** {len(current_run)}  |  **New papers:** {len(new_pids)}",
        "",
        "## Collection policy",
        "",
        "- Google Scholar is used for broad scholarly discovery; arXiv is used for recent preprints and canonical arXiv links.",
        "- Candidates must show explicit image-compression/image-coding evidence in title or abstract/snippet.",
        "- Strong exclusions remove video, audio, speech, text, point-cloud, mesh, 3D Gaussian and generic model/data compression results.",
        "- Each paper is deduplicated globally by normalized title, so one paper is not counted four times merely because it matches multiple topics or sources.",
        "- Signals highlight BPP, PSNR, MS-SSIM, LPIPS, VQ/RVQ, diffusion, hyperprior, entropy models, transformers and common datasets when present in available text.",
        "",
        "## Topics",
        "",
    ]
    for topic_name in TOPICS:
        lines.append(f"- [{topic_name}](#{topic_name.lower().replace(' ', '-')})")

    lines += ["", "## New papers", ""]
    if not new_pids:
        lines += ["No previously unseen papers were detected.", ""]
    else:
        for topic_name in TOPICS:
            papers = new_by_topic.get(topic_name, [])
            if not papers:
                continue
            lines += [
                f"### {topic_name}",
                "",
                "| Date/Year | Title | Authors | Signals | Paper | Code | GS Cites | Source |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for paper in papers[:PER_TOPIC]:
                lines.append(format_row(paper, is_new=True))
            lines.append("")

    lines += ["## Latest papers by topic", ""]
    for topic_name in TOPICS:
        lines += [
            f"## {topic_name}",
            "",
            "| Date/Year | Title | Authors | Signals | Paper | Code | GS Cites | Source |",
            "|---|---|---|---|---|---|---|---|",
        ]
        papers = by_topic.get(topic_name, [])
        if not papers:
            lines.append("| - | No high-confidence matches | - | - | - | - | - | - |")
        else:
            for paper in papers:
                lines.append(format_row(paper))
        lines.append("")

    lines += [
        "## Notes",
        "",
        "Google Scholar does not expose a uniform exact publication date for every result. For Scholar-only records, the table therefore uses the publication year when available. arXiv records retain their submission date.",
        "",
        "Google Scholar API via SerpApi: https://serpapi.com/google-scholar-api",
        "arXiv API: https://info.arxiv.org/help/api/user-manual.html",
        "",
    ]

    readme_text = "\n".join(lines)
    README_FILE.write_text(readme_text, encoding="utf-8")

    # Keep the GitHub Pages source synchronized with README.md.  The Pages
    # workflow builds docs/ with Jekyll, so the generated site and repository
    # README always describe exactly the same paper snapshot.
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    front_matter = "---\nlayout: default\ntitle: Image Compression Paper Daily\n---\n\n"
    DOCS_INDEX_FILE.write_text(front_matter + readme_text, encoding="utf-8")


def save_current_run(by_topic: dict[str, list[dict[str, Any]]]) -> None:
    save_json(PAPERS_FILE, {
        "generated_at": RUN_AT.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "topics": by_topic,
    })


def main() -> None:
    if not SERPAPI_API_KEY:
        raise SystemExit(
            "SERPAPI_API_KEY is missing. Create a SerpApi key and store it in GitHub "
            "Settings -> Secrets and variables -> Actions -> New repository secret."
        )

    print(f"Run time: {RUN_AT.isoformat()}")
    print(f"lookback={LOOKBACK_DAYS}d, scholar_num={SCHOLAR_NUM}, per_topic={PER_TOPIC}")
    by_topic, current_run = collect()
    previously_seen = update_history(current_run)
    save_current_run(by_topic)
    generate_readme(by_topic, current_run, previously_seen)
    print(f"Finished: {len(current_run)} unique high-confidence papers.")
    print(f"New papers: {len(set(current_run) - previously_seen)}")


if __name__ == "__main__":
    main()
