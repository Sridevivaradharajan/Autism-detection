# Autism Detection Using Teachable Machine & Streamlit Interface

This project integrates a Streamlit-based web interface with an image classification model trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/). The goal is to assist in the early detection of Autism Spectrum Disorder (ASD) by analyzing uploaded images.

## Key Features

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

This project integrates a Streamlit-based web interface that was adapted from publicly available code.
The original author and source are unknown. I do not claim ownership of the interface and have used it solely for educational and demonstration purposes.
The image classification model for autism detection was created using Google's Teachable Machine, with a custom dataset.
The model architecture and training were carried out through the Teachable Machine platform, and the integration into this project is entirely my original work.


