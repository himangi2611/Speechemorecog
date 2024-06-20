import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input

custom_objects = {
    'tf': tf,
    'StandardScaler': StandardScaler, 
    'Input': Input 
    
}


model = load_model('ser (1).h5', custom_objects=custom_objects, compile=False)


def process_audio(file_path):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
   
    mfccs_resized = np.pad(mfccs, ((0, 0), (0, 162 - mfccs.shape[1])))
   
    mfccs_processed = np.expand_dims(mfccs_resized, axis=-1)
    return mfccs_processed


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


st.title('Speech Emotion Recognition')

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    processed_data = process_audio(uploaded_file)
    prediction = model.predict(processed_data)
    emotion_label = emotion_labels[np.argmax(prediction)]
    st.write(f'Predicted Emotion: {emotion_label}')
