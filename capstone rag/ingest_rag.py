"""
RAG ingestion pipeline for the eggplant disease remedy knowledge base.

Reads knowledge_base/metadata.json + extracted text files, chunks each
source into paragraph-sized pieces, embeds them locally, and stores them
in a persistent ChromaDB collection tagged by disease class.

Usage:
    pip install chromadb sentence-transformers
    python3 ingest_rag.py
"""

import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

KB_DIR = Path("knowledge_base")
METADATA_PATH = KB_DIR / "metadata.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "eggplant_remedies"

CHUNK_SIZE_WORDS = 250       # target chunk size
CHUNK_OVERLAP_WORDS = 40     # overlap between consecutive chunks
MIN_CHUNK_WORDS = 40         # drop chunks smaller than this (too little context)


def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    # Collapse excessive blank lines / whitespace left over from HTML/PDF extraction
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Simple word-based sliding window chunking with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= MIN_CHUNK_WORDS else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(metadata):
    """Turn every source into a list of chunk records ready for embedding."""
    all_chunks = []
    for m in metadata:
        path = Path(m["file_path"])
        if not path.exists():
            print(f"  ⚠ missing file, skipping: {m['id']}")
            continue

        raw_text = path.read_text(encoding="utf-8")
        text = clean_text(raw_text)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{m['id']}_chunk{i}",
                "text": chunk,
                "disease_class": m["disease_class"],
                "source_title": m["title"],
                "source_url": m["url"],
                "trusted_domain": m.get("trusted_domain", False),
            })
    return all_chunks


def ingest(chunks, model, collection):
    print(f"Embedding {len(chunks)} chunks...")
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "disease_class": c["disease_class"],
                "source_title": c["source_title"],
                "source_url": c["source_url"],
                "trusted_domain": c["trusted_domain"],
            } for c in batch],
        )
        print(f"  ingested {min(i + batch_size, len(chunks))}/{len(chunks)}")


def main():
    print("Loading metadata...")
    metadata = load_metadata()
    print(f"  {len(metadata)} sources")

    print("\nChunking sources...")
    chunks = build_chunks(metadata)
    print(f"  {len(chunks)} chunks produced")

    # Report chunk distribution per class — sanity check before embedding
    from collections import Counter
    counts = Counter(c["disease_class"] for c in chunks)
    print("\nChunks per disease class:")
    for disease, n in counts.items():
        print(f"  {disease:<25} {n} chunks")

    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\nInitializing ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Fresh start each time this script is run — drop and recreate the collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ingest(chunks, model, collection)

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    print(f"Persisted to: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
