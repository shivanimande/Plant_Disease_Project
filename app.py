import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Plant Disease & Health Detection",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.feature-card {
    padding: 22px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
    min-height: 150px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 20px;
}

.small-text {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATHS
# -------------------------------------------------

MODEL_PATH = "models/best_plant_model_improved.keras"
HISTORY_FOLDER = "history"
HISTORY_FILE = os.path.join(
    HISTORY_FOLDER,
    "prediction_history.csv"
)

os.makedirs(HISTORY_FOLDER, exist_ok=True)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.error(str(e))
    st.stop()

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------

class_names = [
    "Healthy",
    "Powdery",
    "Rust"
]

# -------------------------------------------------
# CARE INFORMATION
# -------------------------------------------------

care_tips = {

    "Healthy":
        "🌿 The plant appears healthy. Continue proper watering, "
        "adequate sunlight and regular plant care.",

    "Powdery":
        "🍃 Remove affected leaves and improve air circulation. "
        "Avoid excessive moisture on the leaves.",

    "Rust":
        "🍂 Remove affected leaves and keep the plant area clean. "
        "Improve air circulation and avoid wet foliage."
}

disease_info = {

    "Healthy": {
        "description":
            "The leaf does not show the disease patterns represented "
            "by the trained classes.",
        "signs":
            "Normal green appearance and no obvious powdery or rust-like symptoms.",
        "care":
            "Maintain suitable sunlight, watering and general plant care."
    },

    "Powdery": {
        "description":
            "Powdery disease commonly appears as a white or powder-like "
            "layer on plant surfaces.",
        "signs":
            "White powder-like patches may appear on leaves.",
        "care":
            "Remove affected leaves and maintain good air circulation."
    },

    "Rust": {
        "description":
            "Rust is a fungal disease that can produce rust-coloured "
            "spots or patches on leaves.",
        "signs":
            "Orange, brown or rust-like spots may appear on leaves.",
        "care":
            "Remove affected leaves and keep the plant area clean."
    }
}

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("🌱 Plant Health")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Detection",
        "📊 Statistics",
        "📜 History",
        "📚 Disease Information",
        "⚙️ Preferences",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Plant Disease & Health Detection")
st.sidebar.caption("Deep Learning Project")

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🌱 Plant Disease & Health Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An AI-based system for detecting plant leaf health conditions'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1416879595882-3373a0480b5b",
        caption="Healthy Plants",
        width="stretch"
    )

    st.header("🌿 Welcome")

    st.write(
        "This project uses Deep Learning to analyze plant leaf images "
        "and predict whether the leaf belongs to one of the trained "
        "health conditions."
    )

    st.write(
        "The current model is trained to recognize three classes: "
        "**Healthy, Powdery and Rust**."
    )

    st.markdown("---")

    st.header("✨ Project Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
            <h3>🔍 Disease Detection</h3>
            <p>Upload a leaf image and get a predicted plant condition
            with confidence.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
            <h3>🖼️ Image Quality</h3>
            <p>The application checks image sharpness before making
            a prediction.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
            <h3>🌿 Care Tips</h3>
            <p>Basic care information is displayed according to
            the predicted condition.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            <div class="feature-card">
            <h3>📜 Prediction History</h3>
            <p>Previous predictions can be stored and reviewed.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            """
            <div class="feature-card">
            <h3>📊 Statistics</h3>
            <p>View prediction counts and confidence statistics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:
        st.markdown(
            """
            <div class="feature-card">
            <h3>📚 Disease Information</h3>
            <p>Learn basic information about the supported conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.info(
        "💡 For best results, upload a clear and close-up image "
        "of a plant leaf."
    )

# -------------------------------------------------
# DETECTION PAGE
# -------------------------------------------------

elif page == "🔍 Detection":

    st.title("🔍 Plant Disease Detection")

    st.write(
        "Upload a clear plant leaf image to analyze its condition."
    )

    uploaded_file = st.file_uploader(
        "📷 Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            caption="Uploaded Plant Image",
            width="stretch"
        )

        image_array = np.array(image)

        # ---------------------------------------------
        # IMAGE QUALITY
        # ---------------------------------------------

        gray_image = np.mean(image_array, axis=2)

        gradient_x = np.gradient(
            gray_image,
            axis=0
        )

        gradient_y = np.gradient(
            gray_image,
            axis=1
        )

        sharpness_score = (
            np.var(gradient_x) +
            np.var(gradient_y)
        )

        st.subheader("🖼️ Image Quality")

        st.write(
            f"Image Sharpness Score: "
            f"**{sharpness_score:.2f}**"
        )

        if sharpness_score < 100:
            st.warning(
                "⚠️ Image quality is low. "
                "A clearer image is recommended."
            )
        else:
            st.success(
                "✅ Image quality is good."
            )

        st.markdown("---")

        # ---------------------------------------------
        # DETECTION BUTTON
        # ---------------------------------------------

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            width="stretch"
        ):

            with st.spinner("Analyzing image..."):

                resized_image = image.resize(
                    (224, 224)
                )

                img_array = np.array(
                    resized_image
                )

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction[0]
                )

                confidence = float(
                    prediction[0][predicted_class]
                )

                disease_name = class_names[
                    predicted_class
                ]

            st.subheader("📊 Detection Result")

            # -----------------------------------------
            # LOW CONFIDENCE
            # -----------------------------------------

            if confidence < 0.70:

                st.warning(
                    "⚠️ The model is not confident enough "
                    "to give a reliable prediction."
                )

                st.write(
                    f"Model confidence: "
                    f"**{confidence * 100:.2f}%**"
                )

                st.info(
                    "Please upload a clear, close-up "
                    "image of a plant leaf."
                )

            # -----------------------------------------
            # HIGH CONFIDENCE
            # -----------------------------------------

            else:

                if disease_name == "Healthy":

                    st.success(
                        f"🌿 Prediction: {disease_name}"
                    )

                else:

                    st.warning(
                        f"🍃 Prediction: {disease_name}"
                    )

                st.write(
                    f"Confidence: "
                    f"**{confidence * 100:.2f}%**"
                )

                st.progress(
                    min(confidence, 1.0)
                )

                # -------------------------------------
                # HEALTH STATUS
                # -------------------------------------

                st.subheader(
                    "🩺 Plant Health Status"
                )

                if disease_name == "Healthy":

                    st.success(
                        "🌿 The plant appears healthy."
                    )

                elif disease_name == "Powdery":

                    st.warning(
                        "🍃 The plant may be affected "
                        "by Powdery disease."
                    )

                elif disease_name == "Rust":

                    st.warning(
                        "🍂 The plant may be affected "
                        "by Rust disease."
                    )

                # -------------------------------------
                # CARE TIPS
                # -------------------------------------

                st.subheader(
                    "🌿 Care Tips"
                )

                st.info(
                    care_tips[disease_name]
                )

                # -------------------------------------
                # SAVE HISTORY
                # -------------------------------------

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
                        round(
                            confidence * 100,
                            2
                        )
                    ],
                    "Image Quality": [
                        round(
                            sharpness_score,
                            2
                        )
                    ]
                })

                if os.path.exists(
                    HISTORY_FILE
                ):

                    old_history = pd.read_csv(
                        HISTORY_FILE
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
                    HISTORY_FILE,
                    index=False
                )

                st.success(
                    "✅ Prediction saved to history."
                )

# -------------------------------------------------
# STATISTICS PAGE
# -------------------------------------------------

elif page == "📊 Statistics":

    st.title("📊 Prediction Statistics")

    if os.path.exists(HISTORY_FILE):

        history_data = pd.read_csv(
            HISTORY_FILE
        )

        if len(history_data) > 0:

            total_predictions = len(
                history_data
            )

            healthy_count = len(
                history_data[
                    history_data["Prediction"] == "Healthy"
                ]
            )

            powdery_count = len(
                history_data[
                    history_data["Prediction"] == "Powdery"
                ]
            )

            rust_count = len(
                history_data[
                    history_data["Prediction"] == "Rust"
                ]
            )

            average_confidence = (
                history_data["Confidence"].mean()
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Predictions",
                    total_predictions
                )

            with col2:
                st.metric(
                    "Healthy",
                    healthy_count
                )

            with col3:
                st.metric(
                    "Powdery",
                    powdery_count
                )

            with col4:
                st.metric(
                    "Rust",
                    rust_count
                )

            st.markdown("---")

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )

            st.subheader(
                "📈 Prediction Distribution"
            )

            counts = history_data[
                "Prediction"
            ].value_counts()

            st.bar_chart(counts)

        else:

            st.info(
                "No prediction statistics available yet."
            )

    else:

        st.info(
            "No prediction history available yet."
        )

# -------------------------------------------------
# HISTORY PAGE
# -------------------------------------------------

elif page == "📜 History":

    st.title("📜 Prediction History")

    if os.path.exists(HISTORY_FILE):

        history_data = pd.read_csv(
            HISTORY_FILE
        )

        if len(history_data) > 0:

            st.dataframe(
                history_data,
                width="stretch",
                hide_index=True
            )

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                csv_data = history_data.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download History",
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with col2:

                if st.button(
                    "🗑️ Clear History",
                    width="stretch"
                ):

                    os.remove(HISTORY_FILE)

                    st.success(
                        "✅ Prediction history cleared."
                    )

                    st.rerun()

        else:

            st.info(
                "No previous predictions."
            )

    else:

        st.info(
            "No previous predictions yet."
        )

# -------------------------------------------------
# DISEASE INFORMATION
# -------------------------------------------------

elif page == "📚 Disease Information":

    st.title("📚 Disease Information")

    st.write(
        "Basic information about the conditions supported "
        "by the current model."
    )

    selected_disease = st.selectbox(
        "Select a condition",
        class_names
    )

    info = disease_info[
        selected_disease
    ]

    st.subheader(
        f"🌿 {selected_disease}"
    )

    st.write(
        f"**Description:** {info['description']}"
    )

    st.write(
        f"**Common signs:** {info['signs']}"
    )

    st.write(
        f"**Basic care:** {info['care']}"
    )

    st.info(
        "⚠️ This information is for educational purposes. "
        "The model prediction should not be treated as a professional "
        "plant diagnosis."
    )

# -------------------------------------------------
# PREFERENCES
# -------------------------------------------------

elif page == "⚙️ Preferences":

    st.title("⚙️ Preferences")

    st.subheader(
        "🎯 Prediction Settings"
    )

    confidence_display = st.checkbox(
        "Show confidence percentage",
        value=True
    )

    quality_display = st.checkbox(
        "Show image quality score",
        value=True
    )

    st.markdown("---")

    st.subheader(
        "ℹ️ Current Model"
    )

    st.write(
        "Model: **MobileNetV2-based Deep Learning Model**"
    )

    st.write(
        "Classes: **Healthy, Powdery, Rust**"
    )

    st.write(
        "Input image size: **224 × 224 pixels**"
    )

    st.markdown("---")

    if st.button(
        "🔄 Reset Preferences"
    ):

        st.success(
            "✅ Preferences restored to default."
        )

# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.header(
        "🌱 Plant Disease & Health Detection"
    )

    st.write(
        "Plant Disease & Health Detection is a Deep Learning-based "
        "project designed to analyze plant leaf images and predict "
        "their health condition."
    )

    st.markdown("---")

    st.subheader("🎯 Objective")

    st.write(
        "The main objective is to develop an easy-to-use system "
        "that can analyze plant leaf images and provide a predicted "
        "health condition along with confidence and basic care tips."
    )

    st.subheader("🧠 Technology Used")

    st.write(
        """
        • Python  
        • TensorFlow  
        • Keras  
        • MobileNetV2  
        • NumPy  
        • Pandas  
        • Streamlit  
        • Pillow
        """
    )

    st.subheader("📂 Dataset")

    st.write(
        "The disease model was trained using plant leaf images "
        "belonging to three classes: Healthy, Powdery and Rust."
    )

    st.subheader("✨ Main Features")

    st.write(
        """
        • Plant leaf image upload  
        • Disease prediction  
        • Confidence score  
        • Image quality checking  
        • Care tips  
        • Prediction history  
        • Clear history  
        • Statistics  
        • Disease information  
        • Preferences  
        • About section
        """
    )

    st.subheader("⚠️ Limitation")

    st.write(
        "The model cannot guarantee 100% correct predictions for "
        "every possible image. Performance can change when images "
        "differ significantly from the training data."
    )

    st.subheader("🚀 Future Scope")

    st.write(
        "Future improvements can include a dedicated plant/non-plant "
        "classifier, more diverse plant species and disease images, "
        "and evaluation using external images."
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "🌱 Plant Disease & Health Detection | "
    "Deep Learning Project"
)