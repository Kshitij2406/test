import pandas as pd
import streamlit as st
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

labelencoder = LabelEncoder()
metadata = pd.read_csv('UrbanSound8K.csv')
classes = metadata['class']
class1 = pickle.load((open('class.pkl', 'rb')))
y = to_categorical(labelencoder.fit_transform(class1))

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except (FileNotFoundError, ValueError):
    model = None  # Assign None if the model loading fails


def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# Function to detect sound
def detect_sound(filename):
    # Load the audio file
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    if model is not None:
        predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
        prediction_class = labelencoder.inverse_transform(predicted_label)
        return prediction_class[0]
    else:
        return "Model not found. Unable to perform sound detection."


# Streamlit app
def main():
    st.title("Sound Detection App")

    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Play the audio file
        st.audio(audio_file)

        # Perform sound detection
        sound = detect_sound(audio_file)

        # Display the detection result
        st.header("Detected Sound: " + sound)


# Run the app
if __name__ == "__main__":
    main()
