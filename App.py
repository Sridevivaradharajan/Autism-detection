import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Setup
st.set_page_config(page_title="Autisense", layout="centered")
np.set_printoptions(suppress=True)

# Sidebar Navigation
pages = ["Home", "Image analysis", "Form"]
page = st.sidebar.radio("🔍 Choose Page", pages)

# ----------------- HOME PAGE -----------------
if page == "Home":
    st.markdown("""
        <style>
        .main-title {
            font-size:40px;
            font-weight:bold;
            color:#4A90E2;
            text-align:center;
            margin-bottom:10px;
        }
        .subtext {
            font-size:18px;
            text-align:center;
            color:#555;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🧠 Autisense: Autism Detection Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Assess Autism Spectrum Disorder using Machine Learning & Image Analysis</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Use the sidebar to navigate to:")
    st.markdown("- 📷 **Image-based Detection**")
    st.markdown("- 📋 **Questionnaire-based Test (SVM)**")

    st.markdown("---")
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        This app uses:
        - A **CNN model** trained with Teachable Machine for image classification.
        - An **SVM model** trained on questionnaire data to classify ASD likelihood.
        """)

# ----------------- IMAGE-BASED CNN PAGE -----------------
elif page == "Image analysis":
    st.title("📷 Image-based Autism Prediction")

    cnn_model = load_model(r"C:\Confidential\Autism_detection\keras_Model.h5", compile=False)
    class_names = open(r"C:\Confidential\Autism_detection\labels.txt", "r").readlines()

    uploaded_file = st.file_uploader("Upload a child's image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = cnn_model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.markdown("### 🔍 Prediction Result")
        st.success(f"**Class**: {class_name}")
        st.info(f"**Confidence Score**: {confidence_score:.2f}")

# ----------------- FORM-BASED SVM PAGE -----------------
elif page == "Form":
    st.title("📋 Questionnaire-Based Prediction")

    # Train SVM on dataset
    autism_dataset = pd.read_csv(r'C:\Confidential\Autism_detection\asd_data_csv.csv')
    X = autism_dataset.drop(columns='Outcome', axis=1)
    Y = autism_dataset['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # Form Inputs
    def ValueCount(val): return 1 if val == "Yes" else 0
    def Sex(val): return 1 if val == "Female" else 0

    val10 = Sex(st.selectbox("Gender", ["Female", "Male"]))
    val2 = st.selectbox("Age", list(range(2, 19)))
    val1 = st.selectbox("Social Responsiveness", list(range(0, 11)))
    val3 = ValueCount(st.selectbox("Speech Delay", ["No", "Yes"]))
    val4 = ValueCount(st.selectbox("Learning Disorder", ["No", "Yes"]))
    val5 = ValueCount(st.selectbox("Genetic Disorders", ["No", "Yes"]))
    val6 = ValueCount(st.selectbox("Depression", ["No", "Yes"]))
    val7 = ValueCount(st.selectbox("Intellectual Disability", ["No", "Yes"]))
    val8 = ValueCount(st.selectbox("Social/Behavioural Issues", ["No", "Yes"]))
    val9 = ValueCount(st.selectbox("Anxiety Disorder", ["No", "Yes"]))
    val11 = ValueCount(st.selectbox("Suffers from Jaundice", ["No", "Yes"]))
    val12 = ValueCount(st.selectbox("Family History of ASD", ["No", "Yes"]))

    # Make Prediction
    input_data = [val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]
    input_np = np.asarray(input_data).reshape(1, -1)
    std_input = scaler.transform(input_np)
    prediction = classifier.predict(std_input)

    with st.expander("🧾 View Result"):
        if prediction[0] == 0:
            st.success(" The child is not having Autism Spectrum disorder.")
        else:
            st.warning(" The child is having Autism Spectrum disorder")
