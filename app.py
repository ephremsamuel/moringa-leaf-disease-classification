# =========================================
# MORINGA LEAF DISEASE CLASSIFIER WEB APP
# =========================================

import os
import numpy as np
import streamlit as st
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as imagenet_preprocess,
    decode_predictions,
)

st.set_page_config(page_title="Moringa Disease Classifier", page_icon="leaf", layout="centered")

MODEL_PATH = "best_model.keras"
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


@st.cache_resource
def load_gate_model():
    return MobileNetV2(weights="imagenet")


model = load_my_model()
gate_model = load_gate_model()

CLASS_LABELS = ["Bacterial Leaf Spot", "Cercospora Leaf Spot", "Healthy", "Yellow"]

PLANT_KEYWORDS = ["leaf", "plant", "tree", "flower", "vegetable", "fruit", "herb", "flora", "bush", "seed"]
PERSON_KEYWORDS = ["person", "man", "woman", "face", "groom", "bride"]

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


def is_probably_leaf(img, green_yellow_brown_thresh=0.15, min_std_thresh=8.0, max_skin_ratio=0.35):
    """
    Lightweight heuristic pre-check — NOT a trained classifier.
    Flags images that are very unlikely to be a leaf photo based on color statistics:
      1. Too little green/yellow/brown ("plant-like") hue content.
      2. Significant skin-toned content (hands, faces).
      3. Near-uniform color (e.g. a blank wall, solid background, screenshot).
    Returns (passed: bool, reason: str, diagnostics: dict)
    """
    small = img.convert("RGB").resize((128, 128))
    hsv = np.array(small.convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    plant_hue_mask = (h >= 20) & (h <= 110)
    saturated_mask = s > 25
    plant_mask = plant_hue_mask & saturated_mask
    plant_ratio = float(plant_mask.mean())

    skin_hue_mask = (h >= 5) & (h <= 25)
    skin_sat_mask = (s > 30) & (s < 180)
    skin_bright_mask = v > 100
    skin_mask = skin_hue_mask & skin_sat_mask & skin_bright_mask
    skin_ratio = float(skin_mask.mean())

    color_std = float(np.std(np.array(small)))

    diagnostics = {
        "plant_ratio": round(plant_ratio, 3),
        "skin_ratio": round(skin_ratio, 3),
        "color_std": round(color_std, 2),
    }

    if skin_ratio > max_skin_ratio:
        return False, "Image appears to contain significant skin-toned content rather than leaf material.", diagnostics
    if plant_ratio < green_yellow_brown_thresh:
        return False, "Image doesn't contain enough green/yellow leaf-like coloring.", diagnostics
    if color_std < min_std_thresh:
        return False, "Image looks too flat/uniform to be a leaf photo.", diagnostics

    return True, "", diagnostics


def imagenet_gate(img):
    """
    Semantic pre-check using a pretrained ImageNet model (no training required).
    Returns (is_plant_like: bool, is_person_like: bool, top_labels: list[str])
    """
    gate_img = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(gate_img).astype(np.float32), axis=0)
    arr = imagenet_preprocess(arr)
    preds = gate_model.predict(arr, verbose=0)
    top5 = decode_predictions(preds, top=5)[0]
    labels = [label.lower().replace("_", " ") for (_, label, _) in top5]

    is_plant_like = any(any(kw in label for kw in PLANT_KEYWORDS) for label in labels)
    is_person_like = any(any(kw in label for kw in PERSON_KEYWORDS) for label in labels)

    return is_plant_like, is_person_like, labels


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    color_passed, color_reason, color_diag = is_probably_leaf(img)
    is_plant_like, is_person_like, top_labels = imagenet_gate(img)

    if is_person_like:
        st.error("🚫 This image appears to contain a person, not a Moringa leaf. Please upload a leaf photo instead.")
        with st.expander("Debug info"):
            st.write("Top predicted categories:", top_labels)
        st.stop()

    if not color_passed or not is_plant_like:
        st.warning(
            "⚠️ This doesn't look like a clear Moringa leaf image.\n\n"
            "Please upload a close-up photo of a single leaf, ideally against a plain background."
        )
        with st.expander("Why was this flagged? (debug info)"):
            st.write(color_diag)
            st.write("Top predicted categories:", top_labels)
        proceed_anyway = st.checkbox("I'm sure this is a leaf — analyze it anyway")
        if not proceed_anyway:
            st.stop()

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
