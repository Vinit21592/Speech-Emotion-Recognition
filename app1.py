import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from preprocess_audio import preprocess_audio  # Import the preprocessing function
import os

# Load the trained model and scaler
model_path = 'model.pkl'
model = joblib.load(model_path)

sc_path = 'scaler.pkl'
scaler = joblib.load(sc_path)

# Define the emotions your model predicts
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']  # Modify based on your model

# Streamlit UI setup
st.title("Speech Emotion Recognition")

st.write("Upload an audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Save the file temporarily
    file_path = f"temp/{uploaded_file.name}"
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format='audio/wav')

    if st.button('Predict Emotion'):
        try:
            # Preprocess the audio file
            audio_features = preprocess_audio(file_path)

            # Debugging: Check the shape of the features
            #st.write(f"Preprocessed features shape: {audio_features.shape}")

            # Use the loaded scaler for transforming new data
            features = scaler.transform(audio_features)

            # Predict using the loaded model
            prediction = model.predict(features)

            # Debugging: Check the prediction output
            #st.write(f"Raw prediction output: {prediction}")

            # Get the emotion with the highest probability
            predicted_emotion = emotions[int(prediction[0])]

            st.success(f"Predicted Emotion: {predicted_emotion}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

        finally:
            # Optionally, remove the uploaded file after processing
            if os.path.exists(file_path):
                os.remove(file_path)