# Autism Detection Using Teachable Machine & Streamlit Interface

This project integrates a Streamlit-based web interface with an image classification model trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/). The goal is to assist in the early detection of Autism Spectrum Disorder (ASD) by analyzing uploaded images.

## 📋 Key Features

- Upload images directly through a Streamlit web app
- CNN-based autism detection model trained via Teachable Machine
- Real-time prediction results displayed on the interface
- Fully local deployment (runs using Python + Streamlit)

## How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/autism-detection-teachable.git
   cd autism-detection-teachable
2. Install Required Libraries
pip install -r requirements.txt
3. Run the Streamlit App
streamlit run autism_app.py

## Model Info

- Built using Google Teachable Machine (image project)
- Classes: `Autistic`, `Non-Autistic`
- Files used:
  - `keras_model.h5`:Trained CNN model
  - `labels.txt`:Class label definitions

## Acknowledgements

This project’s form interface is based on publicly available code downloaded from GitHub.  
The original repository and author information are currently unknown.  
Modifications include integrating a CNN model from Google Teachable Machine for autism detection.

If you are the original author, please contact me so I can provide proper attribution.

---

