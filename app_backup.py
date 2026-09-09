import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease & Health Detection",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌱 Plant Disease & Health Detection")
st.write("Upload a plant leaf image to detect its health condition.")

# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = "models/best_plant_model.keras"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.error(str(e))
    st.stop()

# -----------------------------
# Class Names
# -----------------------------
class_names = ["Healthy", "Powdery", "Rust"]

# -----------------------------
# Care Tips
# -----------------------------
care_tips = {
    "Healthy": "🌿 The plant appears healthy. Continue proper watering, sunlight and regular care.",
    "Powdery": "🍃 Remove affected leaves and improve air circulation. Avoid excessive moisture on leaves.",
    "Rust": "🍂 Remove infected leaves and keep the plant area clean. Improve air circulation and avoid wet foliage."
}

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -----------------------------
    # Open Image
    # -----------------------------
    image = Image.open(uploaded_file).convert("RGB")

    # Display Image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Image Quality Check
    # -----------------------------
    image_array = np.array(image)

    if image_array.size == 0:
        st.error("Invalid image.")
        st.stop()

    # Calculate image sharpness
    gray_image = np.mean(image_array, axis=2)

    gradient_x = np.gradient(gray_image, axis=0)
    gradient_y = np.gradient(gray_image, axis=1)

    sharpness_score = (
        np.var(gradient_x) +
        np.var(gradient_y)
    )

    st.subheader("🖼️ Image Quality")

    st.write(
        f"Image Sharpness Score: **{sharpness_score:.2f}**"
    )

    if sharpness_score < 100:
        st.warning(
            "⚠️ Image quality is low. Please upload a clearer image."
        )
    else:
        st.success(
            "✅ Image quality is good."
        )

    # -----------------------------
    # Prepare Image
    # -----------------------------
    resized_image = image.resize((224, 224))

    img_array = np.array(resized_image)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔍 Detect Disease"):

        with st.spinner("Analyzing image..."):

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_class = np.argmax(
                prediction[0]
            )

            confidence = (
                float(prediction[0][predicted_class])
                * 100
            )

            disease_name = class_names[
                predicted_class
            ]

        # -----------------------------
        # Detection Result
        # -----------------------------
        st.subheader("📊 Detection Result")

        if disease_name == "Healthy":
            st.success(
                f"🌿 Prediction: {disease_name}"
            )
        else:
            st.warning(
                f"🍃 Prediction: {disease_name}"
            )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

        # -----------------------------
        # Plant Health Status
        # -----------------------------
        st.subheader("🩺 Plant Health Status")

        if disease_name == "Healthy":
            st.success(
                "The plant appears healthy."
            )
        elif disease_name == "Powdery":
            st.warning(
                "The plant may be affected by Powdery disease."
            )
        elif disease_name == "Rust":
            st.warning(
                "The plant may be affected by Rust disease."
            )

        # -----------------------------
        # Care Tips
        # -----------------------------
        st.subheader("🌿 Care Tips")

        st.info(
            care_tips[disease_name]
        )

        # -----------------------------
        # Save Prediction History
        # -----------------------------
        history_folder = "history"

        os.makedirs(
            history_folder,
            exist_ok=True
        )

        history_file = os.path.join(
            history_folder,
            "prediction_history.csv"
        )

        new_record = pd.DataFrame({
            "Date": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ],
            "Prediction": [
                disease_name
            ],
            "Confidence": [
                round(confidence, 2)
            ],
            "Image Quality": [
                round(sharpness_score, 2)
            ]
        })

        if os.path.exists(history_file):

            old_history = pd.read_csv(
                history_file
            )

            updated_history = pd.concat(
                [
                    old_history,
                    new_record
                ],
                ignore_index=True
            )

        else:

            updated_history = new_record

        updated_history.to_csv(
            history_file,
            index=False
        )

        st.success(
            "✅ Prediction saved to history."
        )

# -----------------------------
# Previous Results
# -----------------------------
st.subheader("📜 Previous Results")

history_file = (
    "history/prediction_history.csv"
)

if os.path.exists(history_file):

    history_data = pd.read_csv(
        history_file
    )

    if len(history_data) > 0:

        st.dataframe(
            history_data,
            use_container_width=True
        )

    else:

        st.write(
            "No previous predictions."
        )

else:

    st.write(
        "No previous predictions yet."
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Plant Disease & Health Detection using Deep Learning"
)