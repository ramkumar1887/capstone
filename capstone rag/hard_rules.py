"""
Hard-set rules layer. Deterministic, reviewable, and independent of retrieval.

Three parts:
  1. ALWAYS_RULES       - safety rules shown for every recommendation.
  2. CONDITIONAL_RULES  - fire based on disease type, severity, classifier
                          confidence, and weather.
  3. apply_rag_rules()  - gates retrieved RAG chunks (blocklist, dosage tagging,
                          trusted-source ranking).

No dosages or product claims are made here: rules are general good practice.
Edit the lists below to add or change rules; each rule has a stable ID.
"""

import re
from weather_service import DISEASE_TYPE

# ---------------------------------------------------------------------
# 1. ALWAYS-ON SAFETY RULES
# ---------------------------------------------------------------------
ALWAYS_RULES = [
    ("S1", "Read the full product label and follow the registered crop, dose and interval."),
    ("S2", "Wear gloves, a mask and long sleeves when handling any spray; wash hands and clothes afterwards."),
    ("S3", "Respect the label's pre-harvest interval before picking any sprayed fruit."),
    ("S4", "Do not mix products or exceed the label dose unless the label explicitly allows it."),
    ("S5", "Keep sprays away from water sources, children and animals; dispose of empty containers safely."),
]

# ---------------------------------------------------------------------
# 2. CONDITIONAL RULES
# ctx keys: disease, severity, dtype, confidence, weather
# Each rule: (id, level, condition(ctx) -> bool, text)
# level: "MUST" (important, shown as a warning) or "ADVISE"
# ---------------------------------------------------------------------

def _wet(ctx):
    w = ctx.get("weather")
    return bool(w) and (w["rain_prob_next_48h"] >= 60 or w["rain_mm_next_48h"] >= 5)


CONDITIONAL_RULES = [
    ("C1", "ADVISE",
     lambda c: c["severity"] == "Severe" and c["dtype"] in ("fungal", "insect"),
     "Severe case: do not repeat the same chemical class back-to-back. Rotate classes to "
     "prevent resistance, and consult your local agricultural officer for a spray programme."),

    ("C2", "ADVISE",
     lambda c: c["dtype"] == "insect",
     "To protect pollinators and beneficial insects, apply insecticides in the evening, "
     "not while bees are active."),

    ("C3", "MUST",
     lambda c: c["dtype"] == "viral",
     "Mosaic viruses cannot be cured by sprays. Do not use fungicides as a treatment. Focus on "
     "removing infected plants and sanitising tools and hands; where the virus type is spread "
     "by aphids or whiteflies, control those insects."),

    ("C4", "ADVISE",
     lambda c: c["dtype"] == "fungal" and _wet(c),
     "Avoid entering the field while foliage is wet; this spreads spores between plants. "
     "Avoid overhead irrigation."),

    ("C5", "ADVISE",
     lambda c: c["severity"] == "Severe",
     "Isolate the affected area: clean tools, boots and hands before moving to healthy plants."),

    ("C6", "MUST",
     lambda c: c.get("confidence") is not None and 0.5 <= c["confidence"] < 0.7,
     "Moderate model confidence: confirm the symptoms with a second photo or your extension "
     "officer before applying any chemical."),

    ("C7", "ADVISE",
     lambda c: c["dtype"] == "soilborne" and c["severity"] in ("Moderate", "Severe"),
     "Do not move soil, water or tools from this plot to healthy plots, and do not replant "
     "eggplant or other solanaceous crops here this season."),
]

def evaluate_rules(disease, severity, confidence=None, weather=None):
    """Return dict with always-on rules and the conditional rules that fired."""
    ctx = {
        "disease": disease,
        "severity": severity,
        "dtype": DISEASE_TYPE.get(disease, "fungal"),
        "confidence": confidence,
        "weather": weather,
    }
    fired = [{"id": rid, "level": lvl, "text": text}
             for rid, lvl, cond, text in CONDITIONAL_RULES if cond(ctx)]
    return {
        "always": [{"id": rid, "text": t} for rid, t in ALWAYS_RULES],
        "fired": fired,
    }


# ---------------------------------------------------------------------
# 3. RAG CHUNK GATING
# ---------------------------------------------------------------------
# Extend with products banned/restricted where your farmers are. Verify against
# current national regulations before relying on this list.
BLOCKED_TERMS = ["endosulfan"]

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:ml|mL|g|gm|kg|l|L|litre|liter|oz|lb|lbs|gal|gallon|ppm)\b"
    r"(?:\s?(?:/|per)\s?(?:ha|acre|l|L|litre|liter|gal|gallon|plant|100\s?L))?",
)

DOSAGE_BANNER = ("Dosages in source documents may not match products registered in your "
                 "region. Follow your local product label.")


def apply_rag_rules(chunks):
    """
    Gate retrieved chunks. Returns (kept_chunks, info) where info reports what was
    dropped. Kept chunks get a 'flags' list and are ordered trusted-first, then by
    similarity.
    """
    kept, dropped_blocked = [], 0
    for ch in chunks:
        text_lower = ch["text"].lower()
        if any(term in text_lower for term in BLOCKED_TERMS):
            dropped_blocked += 1
            continue
        flags = []
        if DOSAGE_PATTERN.search(ch["text"]):
            flags.append("contains dosage figures: verify against local label")
        if not ch.get("trusted", False):
            flags.append("source not on trusted-domain list")
        ch = dict(ch)
        ch["flags"] = flags
        kept.append(ch)

    kept.sort(key=lambda c: (0 if c.get("trusted") else 1, -c["similarity"]))
    any_dosage = any("contains dosage figures" in f for c in kept for f in c["flags"])
    return kept, {
        "dropped_blocked": dropped_blocked,
        "dosage_banner": DOSAGE_BANNER if any_dosage else None,
    }
