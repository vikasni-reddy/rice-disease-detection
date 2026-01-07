import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import os
import gdown

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model_and_classes():
    MODEL_PATH = "rice_disease_model.keras"
    MODEL_URL = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- RADAR CHART ----------------
def radar_chart(pred, class_names):
    labels = [v.replace("_", " ").title() for v in class_names.values()]
    fig = go.Figure(
        go.Scatterpolar(
            r=pred * 100,
            theta=labels,
            fill="toself"
        )
    )
    fig.update_layout(showlegend=False)
    return fig

# ---------------- APP ----------------
st.set_page_config("🌾 Rice Disease Detection", layout="wide")

st.title("🌾 Rice Disease Detection")

uploaded = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with st.spinner("🔍 Analyzing..."):
        model, class_names = load_model_and_classes()
        x = preprocess_image(image)
        preds = model.predict(x)[0]

    idx = np.argmax(preds)
    disease = class_names[idx]
    confidence = preds[idx] * 100

    st.subheader(f"🦠 {disease.replace('_',' ').title()}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.plotly_chart(radar_chart(preds, class_names), use_container_width=True)
