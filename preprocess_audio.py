import librosa
import numpy as np

def preprocess_audio(file_path):
    """
    Preprocess the audio file for prediction.
    
    Parameters:
    - file_path: str, path to the audio file.
    
    Returns:
    - mfccs: np.array, preprocessed MFCC features with shape (1, 128, 128).
    """
    # Load the audio file
    y, sr = librosa.load(file_path, duration=3, offset=0.5, res_type='kaiser_fast')

    # Extract MFCC features from the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Transpose to get shape (time_steps, n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    return mfcc_scaled