from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model # type: ignore
from preprocess_audio import preprocess_audio  # Import the preprocessing function
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('model\ser_lstm_model.h5')

# Define the emotions your model predicts
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']  # Modify based on your model

# Set the new folder path for saving uploaded files
new_upload_folder = 'uploads'

# Ensure the new folder exists
if not os.path.exists(new_upload_folder):
    os.makedirs(new_upload_folder)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    file_path = os.path.join(new_upload_folder, file.filename)
    file.save(file_path)

    try:
        # Preprocess the audio file
        features = preprocess_audio(file_path)

        # Debugging: Check the shape of the features
        print(f"Preprocessed features shape: {features.shape}")

        # Predict using the loaded model
        prediction = model.predict(features)

        # Debugging: Check the prediction output
        print(f"Raw prediction output: {prediction}")

        # Get the emotion with the highest probability
        predicted_emotion = emotions[np.argmax(prediction)]
        print(f"Predicted Emotion: {predicted_emotion}")

        return jsonify({'emotion': predicted_emotion})
    except Exception as e:
        print(f"Error during prediction: {e}")
        # If there's an error during preprocessing or prediction, return an error message
        return jsonify({'error': str(e)}), 500
    #finally:
        # Optionally, remove the uploaded file after processing
        #if os.path.exists(file_path):
            #os.remove(file_path)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)