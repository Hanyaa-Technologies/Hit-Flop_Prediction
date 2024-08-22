import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Feature extraction function
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.effects.percussive(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Note the underscore for unused variable
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
        print(f"Error processing {file_path}: {e}")
        return None

# Directory processing function
def process_directory(directory, label, features_list, labels_list):
    for file_name in os.listdir(directory):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            print(f"Processing {file_path}")
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

# Build dataset function
def build_dataset(features_list, labels_list):
    features_df = pd.DataFrame(features_list)
    features_df['label'] = labels_list
    return features_df

# Convert string lists to numeric
def convert_string_lists_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == object:
            try:
                df[column] = df[column].apply(eval).apply(np.mean)
            except Exception as e:
                print(f"Could not convert column {column} to numeric values. Error: {e}")
    return df

# Preprocess data function
def preprocess_data(features_df):
    # Convert string lists to numeric values
    features_df = convert_string_lists_to_numeric(features_df)
    
    # Handle missing values if any
    features_df.fillna(features_df.mean(), inplace=True)
    
    # Scale the features
    scaler = StandardScaler()
    X = features_df.drop(columns=['label'])
    X_scaled = scaler.fit_transform(X)
    y = features_df['label']
    
    return X_scaled, y

# Train and fine-tune the model
def train_model(X, y):
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
    grid_search.fit(X, y)

    # Print best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Train the model with the best parameters
    best_rf = grid_search.best_estimator_
    best_rf.fit(X, y)

    # Save the trained model
    joblib.dump(best_rf, 'finetuned_audio_hit_flop_model.pkl')

    return best_rf

# Evaluate the model
def evaluate_model(model, X, y):
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", np.mean(cv_scores))

    # Predict and evaluate on the training data
    y_pred = model.predict(X)
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# Prediction function for new songs
def predict_song(file_path, model_path='finetuned_audio_hit_flop_model.pkl'):
    model = joblib.load(model_path)
    features = extract_features(file_path)
    if features is not None:
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        return 'Hit' if prediction[0] == 1 else 'Flop'
    else:
        return "Error processing the song."

# Main script execution
if __name__ == "__main__":
    dataset_path = r"C:\Users\k.akshitha\Hit-Flop_Prediction\hit or flop predictor\Audio_Dataset"
    hit_dir = os.path.join(dataset_path, 'hit songs')
    flop_dir = os.path.join(dataset_path, 'flop songs')
    features_list = []
    labels_list = []

    # Process directories
    process_directory(hit_dir, 1, features_list, labels_list)
    process_directory(flop_dir, 0, features_list, labels_list)

    # Build dataset and save to CSV
    features_df = build_dataset(features_list, labels_list)
    features_df.to_csv('audio_features.csv', index=False)

    # Load and preprocess data
    features_df = pd.read_csv('audio_features.csv')
    X, y = preprocess_data(features_df)

    # Train and evaluate the model
    model = train_model(X, y)
    evaluate_model(model, X, y)

    # Example prediction
    new_song_path = r"C:\Users\k.akshitha\Downloads\[iSongs.info] 03 - Yenduko.mp3"
    prediction = predict_song(new_song_path)
    print(f'The song is predicted to be a {prediction}')
