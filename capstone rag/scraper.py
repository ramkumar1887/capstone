"""
Quality-gated knowledge base scraper for eggplant disease remedies.

Runs per disease class until a target number of VALIDATED sources is
collected (not just "found") — quality filters run at scrape time, not
as an afterthought. Falls back to broader/alternate query phrasing and
scientific names if the initial queries don't yield enough good sources.

Usage:
    pip install ddgs beautifulsoup4 requests pdfplumber
    python3 scraper.py
"""

import os
import re
import json
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
from difflib import SequenceMatcher

from bs4 import BeautifulSoup
from ddgs import DDGS

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Note: pdfplumber not installed — PDF sources will be skipped. "
          "Run: pip install pdfplumber")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("knowledge_base")
HEADERS = {"User-Agent": "Mozilla/5.0 (research data collection; contact: student project)"}

TARGET_SOURCES_PER_CLASS = 6      # stop once this many validated sources are collected
MAX_QUERY_ROUNDS = 4              # how many query-expansion rounds to attempt before giving up
MAX_RESULTS_PER_QUERY = 8
MIN_WORD_COUNT = 150              # reject thin/boilerplate content
NEAR_DUPLICATE_THRESHOLD = 0.85
REQUEST_DELAY = 1.5               # seconds between fetches (politeness)
QUERY_DELAY = 1.0                 # seconds between search queries

# Must appear in the text for a source to be considered on-topic at all
CROP_KEYWORDS = ["eggplant", "brinjal", "aubergine", "solanum melongena"]

# Trusted domains scored higher / preferred when ranking candidates
TRUSTED_DOMAINS = [
    "icar.gov.in", "icar.org.in", "agritech.tnau.ac.in", "vikaspedia.in",
    "plant-pest-advisory.rutgers.edu", "extension.umn.edu", "fao.org",
    "krishijagran.com", "agricoop.gov.in", "aginfobank.co", "apeda.gov.in",
    "extension.psu.edu", "ipm.ucanr.edu", "cabi.org", "plantvillage.psu.edu",
    "koppert.com", "apsnet.org",
]

# Domains known to be low-signal for this purpose (content farms, forums, etc.)
# Sources from these are rejected outright regardless of keyword match.
BLOCKED_DOMAINS = [
    "pinterest.com", "quora.com", "reddit.com", "healthline.com",
    "wikihow.com", "amazon.com", "facebook.com", "instagram.com",
]

# Disease classes with primary + fallback query sets.
# Fallback queries (scientific names, alternate phrasing) are only used
# if the primary set doesn't reach TARGET_SOURCES_PER_CLASS.
DISEASE_QUERIES = {
    "Wilt Disease": {
        "primary": [
            "wilt disease eggplant brinjal treatment management",
            "bacterial wilt brinjal control organic chemical",
        ],
        "fallback": [
            "Ralstonia solanacearum brinjal management",
            "Fusarium wilt eggplant control fungicide",
            "wilt disease solanum melongena treatment",
        ],
    },
    "Phytophthora Blight": {
        "primary": [
            "phytophthora blight eggplant brinjal treatment",
            "phytophthora blight brinjal fungicide management",
        ],
        "fallback": [
            "Phytophthora capsici eggplant control",
            "phytophthora blight solanum melongena remedy",
        ],
    },
    "Fruit rot": {
        "primary": [
            "fruit rot eggplant brinjal treatment management",
            "fruit rot brinjal fungicide organic control",
        ],
        "fallback": [
            "Phomopsis fruit rot eggplant treatment",
            "brinjal fruit rot disease control chemical",
        ],
    },
    "Fruit borer": {
        "primary": [
            "brinjal fruit and shoot borer control management",
            "eggplant fruit borer pesticide organic control",
        ],
        "fallback": [
            "Leucinodes orbonalis control brinjal management",
            "fruit and shoot borer IPM brinjal",
        ],
    },
    "Melon thrips": {
        "primary": [
            "thrips control brinjal eggplant management",
            "melon thrips eggplant pesticide organic control",
        ],
        "fallback": [
            "Thrips palmi management vegetable crops brinjal",
            "thrips brinjal eggplant IPM control",
        ],
    },
    "Tobacco Mosaic Virus": {
        "primary": [
            "tobacco mosaic virus eggplant brinjal management",
            "mosaic virus control eggplant brinjal treatment",
        ],
        "fallback": [
            "TMV management vegetable crops treatment",
            "tobacco mosaic virus symptoms control solanaceous crops",
        ],
    },
    "Cercospora Leaf Spot": {
        "primary": [
            "cercospora leaf spot brinjal eggplant treatment",
            "cercospora leaf spot fungicide vegetable management",
        ],
        "fallback": [
            "cercospora leaf spot solanum melongena control",
            "leaf spot disease brinjal fungal management",
        ],
    },
    "Phomopsis blight": {
        "primary": [
            "phomopsis blight eggplant brinjal treatment",
            "phomopsis blight brinjal fungicide management",
        ],
        "fallback": [
            "Phomopsis vexans control brinjal",
            "phomopsis blight solanum melongena remedy",
        ],
    },
}


# ---------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------

def domain_of(url):
    return urlparse(url).netloc.lower()


def is_blocked(url):
    domain = domain_of(url)
    return any(b in domain for b in BLOCKED_DOMAINS)


def is_trusted(url):
    domain = domain_of(url)
    return any(t in domain for t in TRUSTED_DOMAINS)


def run_queries(queries, seen_urls):
    """Run a list of queries, return new candidate results not already seen."""
    candidates = []
    with DDGS() as ddgs:
        for query in queries:
            try:
                hits = ddgs.text(query, max_results=MAX_RESULTS_PER_QUERY)
            except Exception as e:
                print(f"    [search error] '{query}': {e}")
                continue
            for hit in hits:
                url = hit.get("href") or hit.get("link")
                if not url or url in seen_urls:
                    continue
                if is_blocked(url):
                    continue
                seen_urls.add(url)
                candidates.append({
                    "url": url,
                    "title": hit.get("title", ""),
                    "snippet": hit.get("body", ""),
                })
            time.sleep(QUERY_DELAY)
    # Trusted domains first
    candidates.sort(key=lambda c: 0 if is_trusted(c["url"]) else 1)
    return candidates


# ---------------------------------------------------------------------
# FETCH + EXTRACT
# ---------------------------------------------------------------------

def extract_html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    if main is None:
        return ""
    text = main.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if len(l) > 25 or l.endswith((".", "?", "!"))]
    return "\n".join(lines)


def extract_pdf_text(content_bytes):
    if not PDF_SUPPORT:
        return ""
    import io
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for page in pdf.pages[:15]:  # cap pages for speed/safety
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"    [pdf parse error] {e}")
        return ""
    return "\n".join(text_parts)


def fetch_content(url):
    """Fetch a URL and return (text, content_type) or (None, None) on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [fetch failed] {url}: {e}")
        return None, None

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = extract_pdf_text(resp.content)
        return text, "pdf"
    elif "text/html" in content_type:
        text = extract_html_text(resp.text)
        return text, "html"
    else:
        print(f"    [unsupported content type: {content_type}] {url}")
        return None, None


# ---------------------------------------------------------------------
# QUALITY GATES (run at scrape time, not just validated afterward)
# ---------------------------------------------------------------------

def passes_quality_gates(text, disease, existing_texts):
    """Returns (passed: bool, reason: str)."""
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        return False, f"too short ({word_count} words)"

    text_lower = text.lower()

    if not any(kw in text_lower for kw in CROP_KEYWORDS):
        return False, "no eggplant/brinjal mention — likely off-topic"

    disease_words = [w for w in disease.lower().split() if len(w) > 3]
    hits = sum(1 for w in disease_words if w in text_lower)
    if disease_words and hits == 0:
        return False, "disease name not mentioned — likely off-topic"

    # Near-duplicate check against already-accepted sources (any class)
    for other_text in existing_texts:
        similarity = SequenceMatcher(None, text[:1500], other_text[:1500]).ratio()
        if similarity > NEAR_DUPLICATE_THRESHOLD:
            return False, f"near-duplicate of existing source ({similarity:.0%} similar)"

    return True, "ok"


# ---------------------------------------------------------------------
# MAIN COLLECTION LOOP
# ---------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def collect_for_disease(disease, queries_config, all_texts_flat):
    disease_slug = slugify(disease)
    out_dir = OUTPUT_DIR / "extracted_text" / disease_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_urls = set()
    accepted = []
    accepted_texts = []  # for dedup checks within this class + globally

    query_rounds = [queries_config["primary"]] + [
        [q] for q in queries_config["fallback"]
    ]

    round_num = 0
    while len(accepted) < TARGET_SOURCES_PER_CLASS and round_num < len(query_rounds) and round_num < MAX_QUERY_ROUNDS:
        queries = query_rounds[round_num]
        round_num += 1
        print(f"  round {round_num}: querying {queries}")
        candidates = run_queries(queries, seen_urls)
        print(f"    {len(candidates)} new candidate URLs found")

        for cand in candidates:
            if len(accepted) >= TARGET_SOURCES_PER_CLASS:
                break

            url = cand["url"]
            text, kind = fetch_content(url)
            time.sleep(REQUEST_DELAY)

            if not text:
                continue

            passed, reason = passes_quality_gates(text, disease, accepted_texts + all_texts_flat)
            if not passed:
                print(f"    [REJECTED: {reason}] {url}")
                continue

            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{disease_slug}_{len(accepted)}_{url_hash}.txt"
            filepath = out_dir / filename
            filepath.write_text(text, encoding="utf-8")

            meta = {
                "id": f"{disease_slug}_{len(accepted)}_{url_hash}",
                "disease_class": disease,
                "title": cand["title"],
                "url": url,
                "file_path": str(filepath),
                "content_type": kind,
                "word_count": len(text.split()),
                "date_collected": time.strftime("%Y-%m-%d"),
                "trusted_domain": is_trusted(url),
            }
            accepted.append(meta)
            accepted_texts.append(text)
            all_texts_flat.append(text)
            print(f"    [ACCEPTED] {cand['title'][:55]} ({meta['word_count']} words, "
                  f"{'trusted' if meta['trusted_domain'] else 'untrusted'} domain)")

        if len(accepted) < TARGET_SOURCES_PER_CLASS and round_num < len(query_rounds):
            print(f"    only {len(accepted)}/{TARGET_SOURCES_PER_CLASS} so far — "
                  f"expanding to next query round")

    if len(accepted) < TARGET_SOURCES_PER_CLASS:
        print(f"  ⚠ only reached {len(accepted)}/{TARGET_SOURCES_PER_CLASS} for {disease} "
              f"after exhausting all query rounds — consider manual sourcing for this class")
    else:
        print(f"  ✓ reached target: {len(accepted)}/{TARGET_SOURCES_PER_CLASS}")

    return accepted


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_metadata = []
    all_texts_flat = []  # for cross-class dedup

    for disease, queries_config in DISEASE_QUERIES.items():
        print(f"\n{'=' * 60}\nCollecting: {disease}\n{'=' * 60}")
        results = collect_for_disease(disease, queries_config, all_texts_flat)
        all_metadata.extend(results)

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"Total accepted sources: {len(all_metadata)}")
    from collections import Counter
    counts = Counter(m["disease_class"] for m in all_metadata)
    for disease in DISEASE_QUERIES:
        n = counts.get(disease, 0)
        status = "✓" if n >= TARGET_SOURCES_PER_CLASS else f"⚠ short by {TARGET_SOURCES_PER_CLASS - n}"
        print(f"  {disease:<25} {n}/{TARGET_SOURCES_PER_CLASS}  {status}")
    print(f"\nMetadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
