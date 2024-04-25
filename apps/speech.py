
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import noisereduce as nr

from audiorecorder import audiorecorder

def preprocess_audio(audio_path, target_frames=400):
    x, sr = librosa.load(audio_path, sr=None)
    final_x = nr.reduce_noise(y=x, sr=sr)
    
    f1 = librosa.feature.rms(y=final_x, frame_length=2048, hop_length=512, center=True, pad_mode='reflect').T
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=2048, hop_length=512, center=True).T
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length=512).T
    
    X = np.concatenate((f1, f2, f3), axis=1)
    
    if X.shape[0] < target_frames:
        pad_width = target_frames - X.shape[0]
        X_padded = np.pad(X, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    else:
        X_padded = X[:target_frames, :]
    
    X_3D = np.expand_dims(X_padded, axis=0)
    return X_3D

def speech_page():
    saved_model_path = 'models/smodel.json'
    saved_weights_path = 'models/smodel_weights.h5'

    with open(saved_model_path, 'r') as json_file:
        json_savedModel = json_file.read()

    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)

    class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    

    option = st.selectbox("Choose an option:", ("Record Audio", "Upload Audio"))

    if option == "Record Audio":
        st.warning("Click the 'Start Recording' button to begin recording audio.")

        audio = audiorecorder("Start recording", "Stop recording")

        if len(audio) > 0:
        
            audio.export("misc/output.wav", format="wav")

            st.audio(audio.export().read(), format="audio/wav")

            audio_features = preprocess_audio("misc/output.wav")
            prediction = model.predict(audio_features)
            predicted_class = class_labels[np.argmax(prediction)]

            st.write("Predicted Emotion:", predicted_class)

    elif option == "Upload Audio":
        # Only works with WAV files
        file = st.file_uploader("Upload a WAV audio file", type=["wav"])

        if file is not None:
            st.write("File Uploaded!")

            with open("misc/uploaded.wav", "wb") as f:
                f.write(file.getvalue())

            st.audio("misc/uploaded.wav", format="audio/wav")

            audio_features = preprocess_audio("misc/uploaded.wav")
            prediction = model.predict(audio_features)
            predicted_class = class_labels[np.argmax(prediction)]

            st.write("Predicted Emotion:", predicted_class)

if __name__ == "__main__":
    speech_page()