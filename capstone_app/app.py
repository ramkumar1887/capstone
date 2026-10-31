import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Eggplant Disease Classifier", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # path to your downloaded best.pt

model = load_model()

st.title("🍆 Eggplant Disease Classifier")
st.write("Upload a leaf or fruit image to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        results = model.predict(source=np.array(image), verbose=False)[0]

    names = results.names
    probs = results.probs.data.tolist()

    predicted_class = names[results.probs.top1]
    confidence = results.probs.top1conf.item()

    st.success(f"**Prediction: {predicted_class}** ({confidence:.1%} confidence)")

    df = pd.DataFrame({
        "Class": [names[i] for i in range(len(probs))],
        "Confidence": probs
    }).sort_values("Confidence", ascending=False)

    st.bar_chart(df.set_index("Class"))
    st.dataframe(df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)