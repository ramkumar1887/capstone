import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from remedy_pipeline import Retriever, get_remedy

st.set_page_config(page_title="Eggplant Disease Advisor", layout="centered")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


@st.cache_resource
def load_retriever():
    return Retriever()


def severity_from_percent(pct):
    # Thresholds as stated in the LLD; to be calibrated against annotated ground truth.
    if pct < 15:
        return "Mild"
    if pct <= 40:
        return "Moderate"
    return "Severe"


model = load_model()
retriever = load_retriever()

st.title("🍆 Eggplant Disease Advisor")
st.write("Upload a leaf or fruit photo to get a diagnosis and weather-aware treatment advice.")

location = st.text_input("Your location (village / town / city)", value="Bengaluru")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    res = model.predict(source=np.array(image), verbose=False)[0]
    names = res.names
    disease = names[res.probs.top1]
    confidence = float(res.probs.top1conf)

    st.subheader("1. Diagnosis")
    st.success(f"**{disease}** ({confidence:.1%} confidence)")
    probs = pd.DataFrame({
        "Class": [names[i] for i in range(len(res.probs.data))],
        "Probability": res.probs.data.tolist(),
    }).sort_values("Probability", ascending=False)
    st.bar_chart(probs.set_index("Class"))

    st.subheader("2. Severity")
    st.caption("Segmentation-based severity is in development. For now, estimate the "
               "percentage of the plant/fruit area that looks affected.")
    pct = st.slider("Affected area (%)", 0, 100, 20)
    severity = severity_from_percent(pct)
    st.info(f"Severity level: **{severity}**")

    st.subheader("3. Recommendation")
    out = get_remedy(disease, severity, confidence=confidence,
                     location=location, retriever=retriever)

    if out["status"] == "low_confidence":
        st.warning(out["message"])
    elif out["status"] == "no_data":
        st.warning(out["message"])
    else:
        wa = out.get("weather_advice")
        w = out.get("weather")
        if w:
            st.markdown(f"**Weather in {w['location']}:** {w['temp_c']}°C, "
                        f"{w['humidity']}% humidity, wind {w['wind_kmh']} km/h, "
                        f"rain chance next 48h: {w['rain_prob_next_48h']}%")
            if wa["urgency"] == "Elevated":
                st.error("⚠️ Weather is favouring this problem: act sooner rather than later.")
            for note in wa["risk_notes"]:
                st.write("• " + note)
        elif location:
            st.caption("Weather unavailable right now. Showing general advice only.")

        rec = out.get("recommendation")
        if rec:
            st.markdown("#### Action plan")
            labels = {"immediate": "Do now", "preventive": "Prevent spread / future",
                      "organic": "Organic option", "chemical": "Chemical option"}
            for key in ["immediate", "preventive", "organic", "chemical"]:
                if key in rec:
                    st.markdown(f"**{labels[key]}:** {rec[key]}")

        fired = out["hard_rules"]["fired"]
        if fired:
            st.markdown("#### Rules that apply to your situation")
            for rule in fired:
                text = f"**{rule['id']}** {rule['text']}"
                if rule["level"] == "MUST":
                    st.warning(text)
                else:
                    st.write("• " + text)

        if wa and wa["spray_advice"]:
            st.markdown("#### When to spray")
            for line in wa["spray_advice"]:
                st.write("• " + line)

        with st.expander("Safety rules (always apply)"):
            for rule in out["hard_rules"]["always"]:
                st.write(f"• **{rule['id']}** {rule['text']}")

        if out["references"]:
            with st.expander("Supporting references from agricultural sources"):
                if out["rag_info"]["dosage_banner"]:
                    st.warning(out["rag_info"]["dosage_banner"])
                for ref in out["references"]:
                    st.markdown(f"**{ref['source_title']}**  (relevance {ref['similarity']})")
                    for flag in ref["flags"]:
                        st.caption("⚑ " + flag)
                    st.write(ref["text"][:500] + "...")
                    st.markdown(f"[Source]({ref['source_url']})")
                    st.divider()

        st.caption("Always confirm product choice and dosage against the product label and "
                   "your local agricultural extension officer.")
