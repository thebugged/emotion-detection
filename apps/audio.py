
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import noisereduce as nr

from st_audiorec import st_audiorec

from streamlit_option_menu import option_menu

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

def audio_page():
    saved_model_path = 'models/smodel.json'
    saved_weights_path = 'models/smodel_weights.h5'

    with open(saved_model_path, 'r') as json_file:
        json_savedModel = json_file.read()

    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)

    class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    st.caption("switch between facial & audio detection from sidebar ⬅️")
    selected_page = option_menu(
            menu_title = None,
            options = ["Audio Upload", "Audio Recording"],
            icons=['file-earmark-music', 'record-circle'],
            orientation="horizontal",
        )
    

    

    if selected_page == "Audio Upload":
        st.caption("sample [audio](https://pixabay.com/sound-effects/)")
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


    if selected_page == "Audio Recording":
        st.warning("Click the 'Start Recording' button to begin recording audio.")

        wav_audio_data = st_audiorec()
        st.divider()

        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')

            audio_features = preprocess_audio("misc/output.wav")
            prediction = model.predict(audio_features)
            predicted_class = class_labels[np.argmax(prediction)]
            st.write("Predicted Emotion:", predicted_class)

if __name__ == "__main__":
    audio_page()