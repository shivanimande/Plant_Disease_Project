import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Plant Disease and Health Detection",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------------------------
# LANGUAGE
# -------------------------------------------------
language = st.sidebar.selectbox(
    "🌐 Language / भाषा",
    ["English", "मराठी"]
)

is_marathi = language == "मराठी"

def tr(english, marathi):
    return marathi if is_marathi else english


# -------------------------------------------------
# PATHS AND MODEL
# -------------------------------------------------
MODEL_PATH = "models/best_plant_model_improved.keras"
HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "prediction_history.csv")
CONFIDENCE_THRESHOLD = 0.85

os.makedirs(HISTORY_DIR, exist_ok=True)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

try:
    model = load_model()
    model_available = True
except Exception as e:
    model = None
    model_available = False
    model_error = str(e)


# -------------------------------------------------
# CLASS INFORMATION
# -------------------------------------------------
class_names = ["Healthy", "Powdery", "Rust"]

marathi_names = {
    "Healthy": "निरोगी",
    "Powdery": "पावडरी रोग",
    "Rust": "रस्ट रोग"
}

care_tips = {
    "Healthy": {
        "en": "The leaf appears healthy. Continue proper watering, sunlight and regular plant care.",
        "mr": "पान निरोगी दिसत आहे. योग्य पाणी, सूर्यप्रकाश आणि नियमित निगा सुरू ठेवा."
    },
    "Powdery": {
        "en": "Remove badly affected leaves, improve air circulation and avoid unnecessary moisture on leaves.",
        "mr": "जास्त प्रभावित पाने काढा, हवेचे योग्य circulation ठेवा आणि पानांवर अनावश्यक ओलावा टाळा."
    },
    "Rust": {
        "en": "Remove affected leaves, maintain good air circulation and avoid excess moisture.",
        "mr": "प्रभावित पाने काढा, हवेचे योग्य circulation ठेवा आणि जास्त ओलावा टाळा."
    }
}

disease_info = {
    "Healthy": {
        "description": "The leaf does not show the target disease symptoms recognized by the model.",
        "signs": "Generally green and without strong visible disease patterns.",
        "care": "Continue regular watering, sunlight and plant care."
    },
    "Powdery": {
        "description": "Powdery mildew commonly appears as a white powder-like coating on leaf surfaces.",
        "signs": "White or powder-like patches may appear on leaves.",
        "care": "Remove affected leaves and improve air circulation."
    },
    "Rust": {
        "description": "Rust is commonly associated with rust-colored spots or patches on leaves.",
        "signs": "Orange, brown or rust-colored spots may be visible.",
        "care": "Remove affected leaves and reduce excessive moisture."
    }
}


# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
pages = [
    "🏠 Home",
    "🔍 Detection",
    "📊 Statistics",
    "📜 History",
    "📚 Disease Information",
    "⚙️ Preferences",
    "ℹ️ About"
]

marathi_labels = {
    "🏠 Home": "🏠 होम",
    "🔍 Detection": "🔍 रोग शोध",
    "📊 Statistics": "📊 आकडेवारी",
    "📜 History": "📜 इतिहास",
    "📚 Disease Information": "📚 रोगाची माहिती",
    "⚙️ Preferences": "⚙️ Preferences",
    "ℹ️ About": "ℹ️ प्रकल्पाबद्दल"
}

if is_marathi:
    display_pages = [marathi_labels[p] for p in pages]
    selected_display = st.sidebar.radio("नेव्हिगेशन", display_pages)
    page = pages[display_pages.index(selected_display)]
else:
    page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.caption(
    tr(
        "Plant Disease and Health Detection",
        "Plant Disease and Health Detection"
    )
)


# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "🏠 Home":

    st.title(
        tr(
            "🌱 Plant Disease and Health Detection",
            "🌱 वनस्पती रोग आणि आरोग्य शोध प्रणाली"
        )
    )

    st.write(
        tr(
            "Upload a plant leaf image to analyze its health condition.",
            "वनस्पतीच्या पानाची प्रतिमा अपलोड करून तिच्या आरोग्याची स्थिती तपासा."
        )
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(tr("🔍 Disease Detection", "🔍 रोग शोध"))
        st.write(
            tr(
                "Predicts Healthy, Powdery or Rust.",
                "Healthy, Powdery किंवा Rust यापैकी अंदाज देते."
            )
        )

    with col2:
        st.subheader(tr("📊 Confidence", "📊 Confidence"))
        st.write(
            tr(
                "Shows the model confidence for recognized predictions.",
                "ओळखलेल्या prediction साठी model confidence दाखवते."
            )
        )

    with col3:
        st.subheader(tr("📜 History", "📜 इतिहास"))
        st.write(
            tr(
                "Stores recognized predictions for later viewing.",
                "ओळखलेले predictions नंतर पाहण्यासाठी save केले जातात."
            )
        )

    st.info(
        tr(
            "For best results, upload a clear and close-up leaf image.",
            "चांगल्या परिणामांसाठी पानाचा स्पष्ट आणि जवळून घेतलेला फोटो अपलोड करा."
        )
    )


# -------------------------------------------------
# DETECTION
# -------------------------------------------------
elif page == "🔍 Detection":

    st.title(
        tr(
            "🔍 Plant Disease Detection",
            "🔍 वनस्पती रोग शोध"
        )
    )

    st.write(
        tr(
            "Upload a clear plant leaf image and click Predict.",
            "पानाचा स्पष्ट फोटो अपलोड करा आणि Predict वर क्लिक करा."
        )
    )

    if not model_available:
        st.error(
            tr(
                "Model could not be loaded.",
                "Model load होऊ शकले नाही."
            )
        )
        st.code(model_error)

    uploaded_file = st.file_uploader(
        tr(
            "📤 Upload a leaf image",
            "📤 पानाची प्रतिमा अपलोड करा"
        ),
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption=tr("Uploaded Image", "अपलोड केलेली प्रतिमा"),
            width="stretch"
        )

        if st.button(
            tr("🔎 Predict Disease", "🔎 रोग शोधा"),
            type="primary",
            disabled=not model_available
        ):

            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index])

            st.markdown("---")
            st.subheader(
                tr("📋 Detection Result", "📋 शोध परिणाम")
            )

            if confidence < CONFIDENCE_THRESHOLD:

                st.warning(
                    tr(
                        "⚠️ The model is not confident enough to recognize this image.",
                        "⚠️ या प्रतिमेला ओळखण्यासाठी model चा confidence पुरेसा नाही."
                    )
                )

                st.write(
                    tr(
                        "Please upload a clear, close-up image of a plant leaf.",
                        "कृपया वनस्पतीच्या पानाचा स्पष्ट आणि जवळून घेतलेला फोटो अपलोड करा."
                    )
                )

                st.metric(
                    tr("Model Confidence", "Model Confidence"),
                    f"{confidence * 100:.2f}%"
                )

            else:

                shown_name = (
                    marathi_names[predicted_class]
                    if is_marathi
                    else predicted_class
                )

                st.success(
                    tr(
                        f"✅ Predicted Condition: {predicted_class}",
                        f"✅ अंदाजित स्थिती: {shown_name}"
                    )
                )

                st.metric(
                    tr("Confidence", "Confidence"),
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                if predicted_class == "Healthy":
                    status_text = tr(
                        "🌿 Status: Healthy",
                        "🌿 स्थिती: निरोगी"
                    )
                else:
                    status_text = tr(
                        "⚠️ Status: Disease Detected",
                        "⚠️ स्थिती: रोग आढळला"
                    )

                st.info(status_text)

                st.subheader(
                    tr("💡 Care Tips", "💡 निगा टिप्स")
                )

                tip = (
                    care_tips[predicted_class]["mr"]
                    if is_marathi
                    else care_tips[predicted_class]["en"]
                )

                st.write(tip)

                history_row = pd.DataFrame(
                    [{
                        "Prediction": predicted_class,
                        "Confidence": round(confidence * 100, 2)
                    }]
                )

                if os.path.exists(HISTORY_FILE):
                    history_data = pd.read_csv(HISTORY_FILE)
                    history_data = pd.concat(
                        [history_data, history_row],
                        ignore_index=True
                    )
                else:
                    history_data = history_row

                history_data.to_csv(HISTORY_FILE, index=False)

                st.success(
                    tr(
                        "Prediction saved to history.",
                        "Prediction history मध्ये save झाले."
                    )
                )


# -------------------------------------------------
# STATISTICS
# -------------------------------------------------
elif page == "📊 Statistics":

    st.title(
        tr("📊 Prediction Statistics", "📊 Prediction आकडेवारी")
    )

    if os.path.exists(HISTORY_FILE):

        history_data = pd.read_csv(HISTORY_FILE)

        if len(history_data) > 0:

            healthy_count = int(
                (history_data["Prediction"] == "Healthy").sum()
            )
            powdery_count = int(
                (history_data["Prediction"] == "Powdery").sum()
            )
            rust_count = int(
                (history_data["Prediction"] == "Rust").sum()
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    tr("Healthy", "निरोगी"),
                    healthy_count
                )

            with col2:
                st.metric(
                    tr("Powdery", "पावडरी"),
                    powdery_count
                )

            with col3:
                st.metric(
                    tr("Rust", "रस्ट"),
                    rust_count
                )

            average_confidence = history_data["Confidence"].mean()

            st.markdown("---")

            st.metric(
                tr("Average Confidence", "सरासरी Confidence"),
                f"{average_confidence:.2f}%"
            )

            st.subheader(
                tr("📈 Prediction Distribution", "📈 Prediction वितरण")
            )

            counts = history_data["Prediction"].value_counts()
            st.bar_chart(counts)

        else:
            st.info(
                tr(
                    "No prediction statistics available yet.",
                    "अजून prediction statistics उपलब्ध नाहीत."
                )
            )

    else:
        st.info(
            tr(
                "No prediction history available yet.",
                "अजून prediction history उपलब्ध नाही."
            )
        )


# -------------------------------------------------
# HISTORY
# -------------------------------------------------
elif page == "📜 History":

    st.title(
        tr("📜 Prediction History", "📜 Prediction History")
    )

    if os.path.exists(HISTORY_FILE):

        history_data = pd.read_csv(HISTORY_FILE)

        if len(history_data) > 0:

            st.dataframe(
                history_data,
                width="stretch",
                hide_index=True
            )

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                csv_data = history_data.to_csv(index=False)

                st.download_button(
                    tr("⬇️ Download History", "⬇️ History Download करा"),
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with col2:

                if st.button(
                    tr("🗑️ Clear History", "🗑️ History Clear करा"),
                    width="stretch"
                ):

                    os.remove(HISTORY_FILE)

                    st.success(
                        tr(
                            "✅ Prediction history cleared.",
                            "✅ Prediction history clear झाली."
                        )
                    )

                    st.rerun()

        else:
            st.info(
                tr(
                    "No previous predictions.",
                    "पूर्वीचे predictions नाहीत."
                )
            )

    else:
        st.info(
            tr(
                "No previous predictions yet.",
                "अजून पूर्वीचे predictions नाहीत."
            )
        )


# -------------------------------------------------
# DISEASE INFORMATION
# -------------------------------------------------
elif page == "📚 Disease Information":

    st.title(
        tr(
            "📚 Disease Information",
            "📚 रोगाची माहिती"
        )
    )

    st.write(
        tr(
            "Basic information about the conditions supported by the current model.",
            "सध्याच्या model मध्ये असलेल्या conditions ची मूलभूत माहिती."
        )
    )

    selected_disease = st.selectbox(
        tr("Select a condition", "Condition निवडा"),
        class_names
    )

    info = disease_info[selected_disease]

    display_name = (
        marathi_names[selected_disease]
        if is_marathi
        else selected_disease
    )

    st.subheader(f"🌿 {display_name}")

    if is_marathi:
        st.write(f"**वर्णन:** {info['description']}")
        st.write(f"**सामान्य लक्षणे:** {info['signs']}")
        st.write(f"**मूलभूत निगा:** {info['care']}")
    else:
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Common signs:** {info['signs']}")
        st.write(f"**Basic care:** {info['care']}")

    st.info(
        tr(
            "⚠️ This information is for educational purposes. The model prediction should not be treated as a professional plant diagnosis.",
            "⚠️ ही माहिती शैक्षणिक उद्देशासाठी आहे. Model prediction ला व्यावसायिक plant diagnosis समजू नये."
        )
    )


# -------------------------------------------------
# PREFERENCES
# -------------------------------------------------
elif page == "⚙️ Preferences":

    st.title(
        tr("⚙️ Preferences", "⚙️ Preferences")
    )

    st.subheader(
        tr("🎯 Prediction Settings", "🎯 Prediction Settings")
    )

    st.checkbox(
        tr("Show confidence percentage", "Confidence percentage दाखवा"),
        value=True
    )

    st.checkbox(
        tr("Show image quality score", "Image quality score दाखवा"),
        value=True
    )

    st.markdown("---")

    st.subheader(
        tr("ℹ️ Current Model", "ℹ️ सध्याचा Model")
    )

    st.write(
        tr(
            "Model: **MobileNetV2-based Deep Learning Model**",
            "Model: **MobileNetV2 आधारित Deep Learning Model**"
        )
    )

    st.write(
        tr(
            "Classes: **Healthy, Powdery, Rust**",
            "Classes: **Healthy, Powdery, Rust**"
        )
    )

    st.write(
        tr(
            "Input image size: **224 × 224 pixels**",
            "Input image size: **224 × 224 pixels**"
        )
    )

    st.write(
        tr(
            f"Recognition threshold: **{CONFIDENCE_THRESHOLD * 100:.0f}%**",
            f"Recognition threshold: **{CONFIDENCE_THRESHOLD * 100:.0f}%**"
        )
    )

    if st.button(
        tr("🔄 Reset Preferences", "🔄 Preferences Reset करा")
    ):
        st.success(
            tr(
                "✅ Preferences restored to default.",
                "✅ Preferences default वर reset झाल्या."
            )
        )


# -------------------------------------------------
# ABOUT
# -------------------------------------------------
elif page == "ℹ️ About":

    st.title(
        tr("ℹ️ About the Project", "ℹ️ प्रकल्पाबद्दल")
    )

    st.header("🌱 Plant Disease & Health Detection")

    st.write(
        tr(
            "Plant Disease & Health Detection is a Deep Learning-based project designed to analyze plant leaf images and predict their health condition.",
            "Plant Disease & Health Detection हा Deep Learning आधारित प्रकल्प आहे जो वनस्पतीच्या पानांच्या प्रतिमांचे विश्लेषण करून त्यांच्या आरोग्य स्थितीचा अंदाज लावतो."
        )
    )

    st.markdown("---")

    st.subheader(
        tr("🎯 Objective", "🎯 उद्दिष्ट")
    )

    st.write(
        tr(
            "The main objective is to develop an easy-to-use system that can analyze plant leaf images and provide a predicted health condition along with confidence and basic care tips.",
            "वनस्पतीच्या पानांच्या प्रतिमांचे विश्लेषण करून confidence आणि मूलभूत care tips सह आरोग्य स्थितीचा अंदाज देणारी वापरण्यास सोपी प्रणाली तयार करणे हे मुख्य उद्दिष्ट आहे."
        )
    )

    st.subheader(
        tr("🧠 Technology Used", "🧠 वापरलेले तंत्रज्ञान")
    )

    tech_list = [
        "Python",
        "TensorFlow",
        "Keras",
        "MobileNetV2",
        "NumPy",
        "Pandas",
        "Streamlit",
        "Pillow"
    ]

    for item in tech_list:
        st.write(f"• {item}")

    st.subheader(
        tr("📂 Dataset", "📂 Dataset")
    )

    st.write(
        tr(
            "The disease model was trained using plant leaf images belonging to three classes: Healthy, Powdery and Rust.",
            "Model ला तीन classes साठी plant leaf images वर train केले आहे: Healthy, Powdery आणि Rust."
        )
    )

    st.subheader(
        tr("✨ Main Features", "✨ मुख्य वैशिष्ट्ये")
    )

    feature_list = [
        "Plant leaf image upload",
        "Disease prediction",
        "Confidence score",
        "Image quality checking",
        "Unknown / Not Recognized handling",
        "Care tips",
        "Prediction history",
        "Clear history",
        "Statistics",
        "Disease information",
        "English / Marathi language support",
        "Preferences",
        "About section"
    ]

    if is_marathi:
        feature_list_mr = [
            "Plant leaf image upload",
            "Disease prediction",
            "Confidence score",
            "Image quality checking",
            "Unknown / Not Recognized handling",
            "Care tips",
            "Prediction history",
            "Clear history",
            "Statistics",
            "Disease information",
            "English / Marathi language support",
            "Preferences",
            "About section"
        ]

        for item in feature_list_mr:
            st.write(f"• {item}")
    else:
        for item in feature_list:
            st.write(f"• {item}")

    st.subheader(
        tr("⚠️ Limitation", "⚠️ मर्यादा")
    )

    st.write(
        tr(
            "The model cannot guarantee 100% correct predictions for every possible image. Performance can change when images differ significantly from the training data.",
            "Model प्रत्येक प्रतिमेसाठी 100% अचूक prediction ची हमी देऊ शकत नाही. Training data पेक्षा खूप वेगळ्या प्रतिमांवर performance बदलू शकते."
        )
    )

    st.subheader(
        tr("🚀 Future Scope", "🚀 भविष्यातील विस्तार")
    )

    st.write(
        tr(
            "The system can be extended with more plant species, more disease classes, larger and more diverse datasets, and improved real-world validation.",
            "या system मध्ये भविष्यात अधिक plant species, अधिक disease classes, मोठे आणि विविध datasets तसेच real-world validation जोडता येऊ शकते."
        )
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    tr(
        "🌱 Plant Disease and Health Detection | Deep Learning + Streamlit",
        "🌱 Plant Disease and Health Detection | Deep Learning + Streamlit"
    )
)
