"""
Hybrid remedy recommendation module: rule-based lookup first, RAG retrieval
as fallback/supplement, with a hard fallback to "consult an expert" when
confidence is low or no data is available.

Usage:
    python3 remedy_module.py
    (or import get_recommendation() into your inference pipeline)
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "eggplant_remedies"

# ---------------------------------------------------------------------
# 1. RULE-BASED KNOWLEDGE BASE
# Fill this in with verified info from your validated sources.
# Structure: (disease_class, severity) -> treatment dict
# Severity: "Mild" | "Moderate" | "Severe"
# ---------------------------------------------------------------------

DOSAGE_NOTE = (
    "Exact dosage/frequency not filled in — verify against current product "
    "label and local agricultural extension guidance before use."
)

CHEMICAL_NOTE = (
    DOSAGE_NOTE + " Observe the label's pre-harvest interval before picking sprayed fruit."
)

RULE_BASE = {
    # ---------------- Wilt Disease ----------------
    ("Wilt Disease", "Mild"): {
        "immediate": "Improve field drainage; avoid overwatering.",
        "preventive": "Rotate crops with non-solanaceous species for at least 2 seasons.",
        "organic": f"Apply Trichoderma viride to soil around root zone. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor closely.",
    },
    ("Wilt Disease", "Moderate"): {
        "immediate": "Remove and destroy visibly wilting plants to limit spread.",
        "preventive": "Use resistant/tolerant rootstock varieties in future planting.",
        "organic": f"Soil solarization before next planting season. {DOSAGE_NOTE}",
        "chemical": f"Soil drench with a carbendazim-based fungicide. {CHEMICAL_NOTE}",
    },
    ("Wilt Disease", "Severe"): {
        "immediate": "Remove and destroy all affected plants; avoid replanting susceptible varieties in the same plot this season.",
        "preventive": "Switch to resistant rootstock varieties; improve field drainage long-term.",
        "organic": f"Soil solarization plus Trichoderma application before next season. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer regarding soil fumigation options — this stage often needs professional assessment.",
    },

    # ---------------- Phytophthora Blight ----------------
    ("Phytophthora Blight", "Mild"): {
        "immediate": "Improve drainage; avoid overhead irrigation and waterlogging.",
        "preventive": "Raise planting beds; avoid dense planting to improve airflow.",
        "organic": f"Neem-based foliar spray as a preventive measure. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor closely.",
    },
    ("Phytophthora Blight", "Moderate"): {
        "immediate": "Remove and destroy infected plant parts; avoid working in wet fields to limit spread.",
        "preventive": "Improve field drainage; rotate with non-solanaceous crops.",
        "organic": f"Copper-based fungicide spray (organic-approved formulations exist). {DOSAGE_NOTE}",
        "chemical": f"Metalaxyl or a phosphonate-based fungicide, applied per label. {CHEMICAL_NOTE}",
    },
    ("Phytophthora Blight", "Severe"): {
        "immediate": "Remove and destroy severely affected plants; isolate affected area from irrigation runoff to other plants.",
        "preventive": "Switch to resistant varieties; long-term improvement of field drainage.",
        "organic": f"Copper-based fungicide combined with soil solarization. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for a combination fungicide program — severe blight often needs professional management.",
    },

    # ---------------- Fruit rot ----------------
    ("Fruit rot", "Mild"): {
        "immediate": "Remove visibly infected fruit promptly.",
        "preventive": "Avoid overhead irrigation; maintain adequate plant spacing for airflow.",
        "organic": f"Neem oil spray, applied preventively. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor closely.",
    },
    ("Fruit rot", "Moderate"): {
        "immediate": "Remove and destroy infected fruit; avoid harvesting wet fruit.",
        "preventive": "Mulch to reduce soil splash onto fruit; prune lower foliage.",
        "organic": f"Copper oxychloride spray, organic-approved formulation. {DOSAGE_NOTE}",
        "chemical": f"Copper oxychloride or a mancozeb-based fungicide. {CHEMICAL_NOTE}",
    },
    ("Fruit rot", "Severe"): {
        "immediate": "Remove and destroy all visibly affected fruit and nearby debris.",
        "preventive": "Rotate crops; avoid planting in poorly drained soil next season.",
        "organic": f"Repeated copper-based spray program. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for a fungicide rotation program to manage resistance risk.",
    },

    # ---------------- Fruit borer ----------------
    ("Fruit borer", "Mild"): {
        "immediate": "Hand-pick and destroy visibly damaged shoots/fruit.",
        "preventive": "Use pheromone traps to monitor adult moth activity.",
        "organic": f"Neem-based biopesticide spray. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor with traps.",
    },
    ("Fruit borer", "Moderate"): {
        "immediate": "Remove and destroy infested shoots and fruit; increase trap density.",
        "preventive": "Intercrop with a trap crop (where locally recommended); maintain field sanitation.",
        "organic": f"Bacillus thuringiensis (Bt)-based spray. {DOSAGE_NOTE}",
        "chemical": f"A recommended insecticide labeled for Leucinodes orbonalis. {CHEMICAL_NOTE}",
    },
    ("Fruit borer", "Severe"): {
        "immediate": "Remove and destroy all infested plant material; increase monitoring frequency.",
        "preventive": "Rotate insecticide classes to reduce resistance risk in future seasons.",
        "organic": f"Bt-based spray combined with pheromone trap mass trapping. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for an integrated pest management (IPM) program — severe infestation often needs a multi-pronged approach.",
    },

    # ---------------- Melon thrips ----------------
    ("Melon thrips", "Mild"): {
        "immediate": "Monitor with blue/yellow sticky traps.",
        "preventive": "Remove weeds around field margins that can host thrips.",
        "organic": f"Neem oil spray, applied preventively. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor with traps.",
    },
    ("Melon thrips", "Moderate"): {
        "immediate": "Increase sticky trap density; remove heavily infested leaves.",
        "preventive": "Reflective mulch can help deter thrips (where locally used).",
        "organic": f"Spinosad-based organic-approved spray. {DOSAGE_NOTE}",
        "chemical": f"A recommended insecticide labeled for Thrips palmi. {CHEMICAL_NOTE}",
    },
    ("Melon thrips", "Severe"): {
        "immediate": "Remove and destroy heavily infested plant material.",
        "preventive": "Rotate insecticide classes across the season to manage resistance.",
        "organic": f"Spinosad rotation with neem-based sprays. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for an IPM program — thrips develop resistance quickly, so professional guidance on rotation matters here.",
    },

    # ---------------- Tobacco Mosaic Virus ----------------
    # Note: this is viral — there is no curative chemical treatment.
    # Management is entirely preventive/vector control/sanitation.
    ("Tobacco Mosaic Virus", "Mild"): {
        "immediate": "Remove and destroy visibly infected plants to reduce spread.",
        "preventive": "Disinfect tools between plants; avoid handling healthy plants after infected ones.",
        "organic": "No curative organic treatment exists for viral infection — focus on sanitation and removal.",
        "chemical": "No chemical cure exists for viral infection — management is preventive only.",
    },
    ("Tobacco Mosaic Virus", "Moderate"): {
        "immediate": "Remove and destroy infected plants promptly; disinfect tools and hands.",
        "preventive": "Use certified virus-free seed/seedlings in future planting; control aphid vectors if present.",
        "organic": "No curative organic treatment — vector (aphid) control via neem spray can slow further spread.",
        "chemical": "No chemical cure exists — management is preventive and vector-control focused.",
    },
    ("Tobacco Mosaic Virus", "Severe"): {
        "immediate": "Remove and destroy all infected plants; consider whether replanting the affected area this season is worthwhile.",
        "preventive": "Source resistant varieties for future seasons; strict tool/hand sanitation protocol.",
        "organic": "No curative treatment exists — focus entirely on removal and preventing further spread.",
        "chemical": "No chemical cure exists — consult local agricultural officer regarding resistant variety options for next season.",
    },

    # ---------------- Cercospora Leaf Spot ----------------
    ("Cercospora Leaf Spot", "Mild"): {
        "immediate": "Remove visibly spotted lower leaves.",
        "preventive": "Improve airflow via plant spacing; avoid overhead irrigation.",
        "organic": f"Neem oil or copper-based organic spray. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor closely.",
    },
    ("Cercospora Leaf Spot", "Moderate"): {
        "immediate": "Remove and destroy heavily spotted leaves; avoid working in wet foliage.",
        "preventive": "Rotate crops; avoid overhead irrigation going forward.",
        "organic": f"Copper-based fungicide spray. {DOSAGE_NOTE}",
        "chemical": f"A mancozeb or chlorothalonil-based fungicide. {CHEMICAL_NOTE}",
    },
    ("Cercospora Leaf Spot", "Severe"): {
        "immediate": "Remove and destroy severely affected foliage; consider defoliation if spread is extensive.",
        "preventive": "Longer crop rotation cycle; resistant varieties for future planting.",
        "organic": f"Repeated copper-based spray program. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for a fungicide rotation program.",
    },

    # ---------------- Phomopsis blight ----------------
    ("Phomopsis blight", "Mild"): {
        "immediate": "Remove visibly infected plant parts.",
        "preventive": "Use disease-free seed; avoid working in wet fields.",
        "organic": f"Neem-based or copper-based organic spray. {DOSAGE_NOTE}",
        "chemical": "Not typically required at mild stage — monitor closely.",
    },
    ("Phomopsis blight", "Moderate"): {
        "immediate": "Remove and destroy infected plant material; avoid overhead irrigation.",
        "preventive": "Rotate crops with non-solanaceous species; improve field sanitation.",
        "organic": f"Copper-based fungicide spray. {DOSAGE_NOTE}",
        "chemical": f"A mancozeb-based fungicide, per Phomopsis vexans label guidance. {CHEMICAL_NOTE}",
    },
    ("Phomopsis blight", "Severe"): {
        "immediate": "Remove and destroy all severely affected plant material and debris.",
        "preventive": "Extended crop rotation; resistant varieties for future seasons.",
        "organic": f"Repeated copper-based spray program combined with strict field sanitation. {DOSAGE_NOTE}",
        "chemical": "Consult local agricultural officer for a fungicide rotation program to manage resistance risk.",
    },
}


# ---------------------------------------------------------------------
# 2. RAG RETRIEVAL LAYER
# ---------------------------------------------------------------------

class RemedyRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, disease_class, query, k=3, min_score=0.35):
        query_emb = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            where={"disease_class": disease_class},
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # ChromaDB returns distances (lower = more similar) by default with
        # cosine — convert to a similarity-like score for filtering.
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 - dist  # approximate; fine for relative filtering
            if similarity >= min_score:
                chunks.append({
                    "text": doc,
                    "source_title": meta["source_title"],
                    "source_url": meta["source_url"],
                    "similarity": round(similarity, 3),
                })
        return chunks


# ---------------------------------------------------------------------
# 3. HYBRID DECISION LOGIC
# ---------------------------------------------------------------------

def get_recommendation(disease_class, severity, retriever=None, classifier_confidence=None,
                        confidence_threshold=0.5):
    """
    Main entry point. Returns a structured recommendation dict.

    disease_class: predicted disease name (must match your class names)
    severity: "Mild" | "Moderate" | "Severe" (from your severity estimation module)
    retriever: a RemedyRetriever instance (created once, reused across calls)
    classifier_confidence: optional float from your classification model's
        top1 confidence — used for the safety fallback below
    """

    # Safety fallback: don't guess on low-confidence predictions
    if classifier_confidence is not None and classifier_confidence < confidence_threshold:
        return {
            "status": "low_confidence",
            "message": (
                "Prediction confidence is low. Please consult your local "
                "agricultural extension officer before taking action."
            ),
            "confidence": classifier_confidence,
        }

    rule_entry = RULE_BASE.get((disease_class, severity))
    if rule_entry:
        return {
            "status": "rule_based",
            "disease_class": disease_class,
            "severity": severity,
            "recommendation": rule_entry,
            "source": "internal knowledge base",
        }

    # No rule entry — fall back to RAG retrieval
    if retriever is not None:
        query = f"treatment and management for {disease_class} at {severity} severity"
        chunks = retriever.retrieve(disease_class, query)
        if chunks:
            return {
                "status": "rag_retrieved",
                "disease_class": disease_class,
                "severity": severity,
                "retrieved_context": chunks,
                "note": (
                    "No pre-verified rule found for this exact case. The "
                    "following is retrieved from agricultural reference "
                    "documents — cross-check before acting."
                ),
            }

    # Nothing found anywhere — hard fallback
    return {
        "status": "no_data",
        "message": (
            f"No verified recommendation available for {disease_class} "
            f"({severity}). Please consult your local agricultural "
            "extension officer."
        ),
    }


# ---------------------------------------------------------------------
# DEMO / QUICK TEST
# ---------------------------------------------------------------------

def print_recommendation(result):
    print(f"\nStatus: {result['status']}")
    if result["status"] == "rule_based":
        print(f"Disease: {result['disease_class']} | Severity: {result['severity']}")
        for tier, text in result["recommendation"].items():
            print(f"  {tier.capitalize()}: {text}")
        print(f"  Source: {result['source']}")
    elif result["status"] == "rag_retrieved":
        print(f"Disease: {result['disease_class']} | Severity: {result['severity']}")
        print(f"  Note: {result['note']}")
        for i, chunk in enumerate(result["retrieved_context"], 1):
            print(f"\n  [{i}] {chunk['source_title']} (similarity: {chunk['similarity']})")
            print(f"      {chunk['text'][:300]}...")
            print(f"      Source: {chunk['source_url']}")
    else:
        print(f"  {result['message']}")


if __name__ == "__main__":
    retriever = RemedyRetriever()

    # Test 1: covered by rule base
    print("=" * 60)
    print("TEST 1: Wilt Disease, Severe (in rule base)")
    print("=" * 60)
    result = get_recommendation("Wilt Disease", "Severe", retriever=retriever, classifier_confidence=0.94)
    print_recommendation(result)

    # Test 2: NOT in rule base -> should fall back to RAG
    print("\n" + "=" * 60)
    print("TEST 2: Cercospora Leaf Spot, Moderate (not in rule base)")
    print("=" * 60)
    result = get_recommendation("Cercospora Leaf Spot", "Moderate", retriever=retriever, classifier_confidence=0.88)
    print_recommendation(result)

    # Test 3: low confidence -> should trigger safety fallback
    print("\n" + "=" * 60)
    print("TEST 3: Fruit rot, Mild, LOW confidence (0.35)")
    print("=" * 60)
    result = get_recommendation("Fruit rot", "Mild", retriever=retriever, classifier_confidence=0.35)
    print_recommendation(result)
