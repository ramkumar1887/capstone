"""
Knowledge base validation script.
Run this after scraping to check coverage, content quality, relevance,
near-duplicates, and retrieval sanity — before wiring the RAG pipeline.

Usage:
    python3 validate_knowledge_base.py
"""

import json
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# pip install sentence-transformers --break-system-packages
from sentence_transformers import SentenceTransformer, util

KB_DIR = Path("knowledge_base")
METADATA_PATH = KB_DIR / "metadata.json"

DISEASE_CLASSES = [
    "Wilt Disease", "Phytophthora Blight", "Fruit rot", "Fruit borer",
    "Melon thrips", "Tobacco Mosaic Virus", "Cercospora Leaf Spot", "Phomopsis blight",
]

SHORT_TEXT_THRESHOLD = 100      # words
NEAR_DUPLICATE_THRESHOLD = 0.8  # similarity ratio
TEST_QUERIES = [
    ("Wilt Disease", "treatment for wilt disease in eggplant"),
    ("Fruit rot", "organic remedy for fruit rot"),
    ("Phytophthora Blight", "fungicide dosage for phytophthora blight"),
    ("Cercospora Leaf Spot", "control measures for cercospora leaf spot"),
]


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} not found — run the scraper first.")
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_texts(metadata):
    texts = {}
    for m in metadata:
        path = Path(m["file_path"])
        if not path.exists():
            print(f"  ⚠ missing file for {m['id']}: {path}")
            continue
        texts[m["id"]] = path.read_text(encoding="utf-8")
    return texts


def check_coverage(metadata):
    print("\n" + "=" * 60)
    print("1. COVERAGE — sources per disease class")
    print("=" * 60)
    counts = Counter(m["disease_class"] for m in metadata)
    for disease in DISEASE_CLASSES:
        n = counts.get(disease, 0)
        flag = "  ⚠ LOW COVERAGE" if n < 2 else ""
        print(f"  {disease:<25} {n} sources{flag}")


def check_length(metadata, texts):
    print("\n" + "=" * 60)
    print("2. CONTENT LENGTH — flagging thin/junk extractions")
    print("=" * 60)
    flagged = 0
    for m in metadata:
        text = texts.get(m["id"], "")
        word_count = len(text.split())
        if word_count < SHORT_TEXT_THRESHOLD:
            print(f"  ⚠ {m['id']}: {word_count} words — {m['title'][:55]}")
            flagged += 1
    print(f"\n  {flagged} of {len(metadata)} sources flagged as too short.")


def check_relevance(metadata, texts):
    print("\n" + "=" * 60)
    print("3. RELEVANCE — does the text actually mention the disease?")
    print("=" * 60)
    flagged = 0
    for m in metadata:
        text = texts.get(m["id"], "").lower()
        disease_words = [w for w in m["disease_class"].lower().split() if len(w) > 3]
        hits = sum(1 for w in disease_words if w in text)
        if disease_words and hits < len(disease_words):
            print(f"  ⚠ {m['id']} may be off-topic — {m['title'][:55]}")
            flagged += 1
    print(f"\n  {flagged} of {len(metadata)} sources flagged as possibly off-topic.")


def check_duplicates(metadata, texts):
    print("\n" + "=" * 60)
    print("4. NEAR-DUPLICATES — same content pulled in by multiple queries")
    print("=" * 60)
    ids = list(texts.keys())
    found = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = texts[ids[i]][:1000], texts[ids[j]][:1000]
            similarity = SequenceMatcher(None, a, b).ratio()
            if similarity > NEAR_DUPLICATE_THRESHOLD:
                print(f"  ⚠ {ids[i]} vs {ids[j]} — {similarity:.0%} similar")
                found += 1
    if found == 0:
        print("  none found.")


def check_retrieval_sanity(metadata, texts):
    print("\n" + "=" * 60)
    print("5. RETRIEVAL SANITY — does semantic search surface the right class?")
    print("=" * 60)
    print("  loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pre-embed all sources once (first 2000 chars each, for a quick check)
    doc_embeddings = []
    doc_info = []
    for m in metadata:
        text = texts.get(m["id"], "")[:2000]
        if not text.strip():
            continue
        doc_embeddings.append(model.encode(text, convert_to_tensor=True))
        doc_info.append((m["title"], m["disease_class"]))

    for expected_disease, query in TEST_QUERIES:
        print(f"\n  Query: \"{query}\"  (expecting: {expected_disease})")
        query_emb = model.encode(query, convert_to_tensor=True)
        scores = []
        for emb, (title, disease) in zip(doc_embeddings, doc_info):
            sim = util.cos_sim(query_emb, emb).item()
            scores.append((sim, title, disease))
        scores.sort(reverse=True)

        top3_diseases = [d for _, _, d in scores[:3]]
        match = "✓" if expected_disease in top3_diseases else "⚠ expected disease not in top 3"
        print(f"    {match}")
        for sim, title, disease in scores[:3]:
            print(f"    {sim:.3f}  [{disease}] {title[:55]}")


def print_summary(metadata):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total sources: {len(metadata)}")
    counts = Counter(m["disease_class"] for m in metadata)
    missing = [d for d in DISEASE_CLASSES if counts.get(d, 0) == 0]
    if missing:
        print(f"  Classes with ZERO sources: {', '.join(missing)}")
    else:
        print("  All disease classes have at least one source.")
    print("\n  Review all ⚠ flags above manually before building the RAG index.")
    print("  Automated checks catch obvious problems; they don't replace")
    print("  actually reading a few sources per class yourself.")


def main():
    print("Loading knowledge base...")
    metadata = load_metadata()
    texts = load_texts(metadata)
    print(f"Loaded {len(metadata)} metadata entries, {len(texts)} text files found.")

    check_coverage(metadata)
    check_length(metadata, texts)
    check_relevance(metadata, texts)
    check_duplicates(metadata, texts)
    check_retrieval_sanity(metadata, texts)
    print_summary(metadata)


if __name__ == "__main__":
    main()
