# Autism Detection Using Teachable Machine & Streamlit Interface

This project integrates a Streamlit-based web interface with an image classification model trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/). The goal is to assist in the early detection of Autism Spectrum Disorder (ASD) by analyzing uploaded images.

---

## Key Features

- Upload images directly through a Streamlit web app
- CNN-based autism detection model trained via Teachable Machine
- Real-time prediction results displayed on the interface
- Fully local deployment (runs using Python + Streamlit)

---

## How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/Sridevivaradharajan/Autism-detection.git
   cd Autism-detection
2. Install Required Libraries
pip install -r requirements.txt
3. Run the Streamlit App
streamlit run autism_app.py

---

## Model Info

- Built using Google Teachable Machine (image project)
- Classes: `Autistic`, `Non-Autistic`
- Files used:
  - `keras_model.h5`:Trained CNN model
  - `labels.txt`:Class label definitions

---

## Acknowledgements
This project integrates a Streamlit-based web interface for autism prediction. The image classification model was developed using Google’s Teachable Machine with a custom dataset. While the model architecture and training were completed on the platform, the integration and interface development are entirely my original work. The model predicts autism-related traits in children with ASD using both image input and Autism Spectrum Quotient (AQ)-based behavioral indicators.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.
You are free to use, share, and adapt the contents of this repository for **non-commercial purposes**, provided that proper **attribution is given**.
🔗 [View License Details](https://creativecommons.org/licenses/by-nc/4.0/)
