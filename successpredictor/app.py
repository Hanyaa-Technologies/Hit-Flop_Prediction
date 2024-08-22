import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os

# Feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.effects.percussive(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)

        features = {
            'chroma_stft': np.mean(chroma_stft),
            'rmse': np.mean(rmse),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'rolloff': np.mean(rolloff),
            'zero_crossing_rate': np.mean(zero_crossing_rate),
            'harmony': np.mean(harmony),
            'perceptr': np.mean(perceptr),
            'tempo': tempo
        }

        for i in range(1, 21):
            features[f'mfcc{i}'] = np.mean(mfccs[i-1])

        return features
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None

# Convert and preprocess features
def preprocess_features(features, scaler):
    features_df = pd.DataFrame([features])
    X_scaled = scaler.transform(features_df)
    return X_scaled

# Train and fine-tune the model
def train_model(features_df):
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameters grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Set up the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search
    grid_search.fit(X_scaled, y)

    # Train the model with the best parameters
    best_rf = grid_search.best_estimator_

    return best_rf, scaler

# Directory processing function
def process_directory(directory, label, features_list, labels_list):
    for file_name in os.listdir(directory):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

# Build dataset function
def build_dataset(features_list, labels_list):
    features_df = pd.DataFrame(features_list)
    features_df['label'] = labels_list
    return features_df

# Streamlit app
def main():
    st.title("Hit or Flop Song Predictor")
    
    # Initialize session state
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        with st.spinner("Processing dataset and training model..."):
            # Process directories
            dataset_path = r"C:\Users\AkshithaKochika\Downloads\hit or flop predictor\Audio_Dataset"
            hit_dir = os.path.join(dataset_path, 'hit songs')
            flop_dir = os.path.join(dataset_path, 'flop songs')
            features_list = []
            labels_list = []
            process_directory(hit_dir, 1, features_list, labels_list)
            process_directory(flop_dir, 0, features_list, labels_list)
            
            # Build dataset and train model
            features_df = build_dataset(features_list, labels_list)
            model, scaler = train_model(features_df)
            
            # Store model and scaler in session state
            st.session_state.model = model
            st.session_state.scaler = scaler
        st.success("Dataset processed and model trained!")

    # Upload file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if uploaded_file is not None:
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.button('Check Song'):
            features = extract_features("temp_audio_file")
            if features:
                # Use the trained model and scaler
                X_scaled = preprocess_features(features, st.session_state.scaler)
                prediction = st.session_state.model.predict(X_scaled)
                st.write(f'The song is predicted to be a {"Hit" if prediction[0] == 1 else "Flop"}')
            else:
                st.write("Error processing the song.")

if __name__ == "__main__":
    main()
