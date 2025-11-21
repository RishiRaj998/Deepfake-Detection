import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import os

N_MFCC = 13
N_CHROMA = 12
N_SPECTRAL = 7
N_TONNETZ = 6
MAX_PAD_LEN = 862
FEATURE_DIM = N_MFCC + N_CHROMA + N_SPECTRAL + N_TONNETZ  # 38

print("Loading LSTM model...")
lstm_model = load_model(r'D:\Major Project\Rishi_Raj\Audio_detection\model.keras')  # Adjust path as needed
print("LSTM loaded.")

print("Loading KNN model...")
knn = joblib.load(r'D:\Major Project\Rishi_Raj\KNN.joblib')  # Adjust path as needed
print("KNN loaded.\n")

def extract_features(file_path):
    print("Step 1: Extracting features...")
    y, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(S=librosa.power_to_db(stft), sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(S=librosa.power_to_db(stft), sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    def pad_feature(feature, n_rows):
        if feature.shape[0] < n_rows:
            feature = np.pad(feature, ((0, n_rows - feature.shape[0]), (0, 0)), mode='constant')
        elif feature.shape[0] > n_rows:
            feature = feature[:n_rows, :]
        if feature.shape[1] < MAX_PAD_LEN:
            feature = np.pad(feature, ((0, 0), (0, MAX_PAD_LEN - feature.shape[1])), mode='constant')
        elif feature.shape[1] > MAX_PAD_LEN:
            feature = feature[:, :MAX_PAD_LEN]
        return feature

    mfccs = pad_feature(mfccs, N_MFCC)
    chroma = pad_feature(chroma, N_CHROMA)
    spectral_contrast = pad_feature(spectral_contrast, N_SPECTRAL)
    tonnetz = pad_feature(tonnetz, N_TONNETZ)

    combined = np.concatenate((mfccs, chroma, spectral_contrast, tonnetz), axis=0).T
    print(f"Features shape: {combined.shape}")
    return combined

def extract_lstm_features(model, features):
    print("Step 2: Extracting LSTM features...")
    inputs = np.expand_dims(features, axis=0)  # (1, 862, 38)
    if inputs.shape[1:] != (MAX_PAD_LEN, FEATURE_DIM):
        raise ValueError(f"Input shape {inputs.shape[1:]} mismatch, expected ({MAX_PAD_LEN}, {FEATURE_DIM})")
    preds = model.predict(inputs)
    print(f"LSTM output shape: {preds.shape}")
    return preds.flatten().reshape(1, -1)

def predict(audio_path):
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        return
    print(f"\nProcessing file: {audio_path}")
    features = extract_features(audio_path)
    lstm_features = extract_lstm_features(lstm_model, features)
    print("Step 3: Predict using KNN...")
    pred = knn.predict(lstm_features)[0]
    label = "REAL" if pred == 1 else "FAKE"
    print(f"Prediction: The audio is classified as {label}")

def interactive():
    print("Audio Deepfake Detection - Input a WAV file path or type 'exit' to quit.")
    while True:
        path = input("> ").strip()
        if path.lower() == "exit":
            print("Exiting.")
            break
        if path == "":
            print("Please enter a valid file path.")
            continue
        try:
            predict(path)
        except Exception as e:
            print("Error during prediction:", e)

if __name__ == "__main__":
    interactive()