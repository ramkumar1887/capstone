"""
Remedy pipeline: rule-based table + RAG supporting references + weather advisories.

Flow for get_remedy():
  1. Low classifier confidence -> safety fallback (no guessing).
  2. Rule table lookup for (disease, severity) -> structured recommendation.
  3. RAG retrieval from ChromaDB (filtered to the disease) -> cited source passages.
     Used as supporting references, and as the main answer if no rule exists.
  4. Weather advisories for the farmer's location -> urgency + spray timing.

Requires: chroma_db/ built by ingest_rag.py, and remedy_module.py (for RULE_BASE).
"""

import chromadb
from sentence_transformers import SentenceTransformer

from remedy_module import RULE_BASE
from weather_service import fetch_weather, weather_advisories, canonical_disease
from hard_rules import evaluate_rules, apply_rag_rules

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "eggplant_remedies"
MIN_SIMILARITY = 0.35          # cosine similarity floor for retrieved chunks
CONFIDENCE_THRESHOLD = 0.5


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, disease, query, k=3):
        emb = self.model.encode(query).tolist()
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            where={"disease_class": disease},
        )
        if not res["documents"] or not res["documents"][0]:
            return []
        out, seen_sources = [], set()
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            # Chroma default is squared L2 on normalized embeddings: cos = 1 - dist/2
            similarity = 1 - dist / 2
            if similarity < MIN_SIMILARITY:
                continue
            if meta["source_url"] in seen_sources:   # one passage per source
                continue
            seen_sources.add(meta["source_url"])
            out.append({
                "text": doc,
                "source_title": meta["source_title"],
                "source_url": meta["source_url"],
                "similarity": round(similarity, 3),
                "trusted": bool(meta.get("trusted_domain", False)),
            })
        return out


def get_remedy(disease, severity, confidence=None, location=None, retriever=None):
    """Main entry point. Returns a dict the UI can render directly."""

    # Model class names (e.g. "Fruit_borer") -> canonical names used everywhere else
    disease = canonical_disease(disease)

    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "low_confidence",
            "message": "Prediction confidence is low. Please retake the photo or consult "
                       "your local agricultural extension officer before acting.",
        }

    result = {"status": "ok", "disease": disease, "severity": severity}

    # Weather first, because hard rules can depend on it
    weather = fetch_weather(location) if location else None
    result["weather"] = weather
    result["weather_advice"] = weather_advisories(disease, weather)

    # Hard rules (safety + conditional)
    rule_result = evaluate_rules(disease, severity, confidence, weather)
    result["hard_rules"] = rule_result
    result["recommendation"] = RULE_BASE.get((disease, severity))

    # RAG chunks, gated by hard rules
    raw_refs = []
    if retriever is not None:
        raw_refs = retriever.retrieve(
            disease, f"{disease} management and treatment in eggplant brinjal")
    result["references"], result["rag_info"] = apply_rag_rules(raw_refs)

    if result["recommendation"] is None and not result["references"]:
        result["status"] = "no_data"
        result["message"] = ("No verified recommendation available. Please consult your local "
                             "agricultural extension officer.")
    return result


if __name__ == "__main__":
    r = Retriever()
    out = get_remedy("Fruit rot", "Moderate", confidence=0.93, location="Bengaluru", retriever=r)
    print("Status:", out["status"])
    if out.get("recommendation"):
        for k, v in out["recommendation"].items():
            print(f"  {k}: {v}")
    if out.get("weather"):
        print("Weather:", out["weather"])
        print("Weather advice:", out["weather_advice"])
    print("\nHard rules fired:")
    for rule in out["hard_rules"]["fired"]:
        print(f"  [{rule['id']}/{rule['level']}] {rule['text']}")
    print("RAG info:", out["rag_info"])
    for ref in out.get("references", []):
        print(f"\n[{ref['similarity']}] {ref['source_title']} flags={ref['flags']}\n  {ref['text'][:250]}...\n  {ref['source_url']}")
