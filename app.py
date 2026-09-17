# =========================================
# MORINGA LEAF DISEASE CLASSIFIER WEB APP
# =========================================

import os
import numpy as np
import streamlit as st
import gdown
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Moringa Disease Classifier", page_icon="leaf", layout="centered")

MODEL_PATH = "best_model.keras"
# Set this as an environment variable / Streamlit secret — never hardcode it in source.
MODEL_DRIVE_URL = os.environ.get("MODEL_DRIVE_URL", "")

if not os.path.exists(MODEL_PATH):
    if not MODEL_DRIVE_URL:
        st.error("MODEL_DRIVE_URL is not set. Configure it as an environment variable or secret.")
        st.stop()
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)


@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)


model = load_my_model()

CLASS_LABELS = ["Bacterial Leaf Spot", "Cercospora Leaf Spot", "Healthy", "Yellow"]

st.title("Moringa Leaf Disease Classifier")
st.markdown("""
Upload a Moringa leaf image and get an instant disease prediction.

**Supported classes:** Healthy, Yellow (Chlorosis), Bacterial Leaf Spot, Cercospora Leaf Spot
""")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])


def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return np.expand_dims(arr, axis=0)


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        processed = preprocess_image(img)
        prediction = model.predict(processed)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    predicted_class = CLASS_LABELS[predicted_index]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")
    for label, prob in zip(CLASS_LABELS, prediction):
        st.write(f"{label}: {prob*100:.2f}%")
        st.progress(float(prob))
