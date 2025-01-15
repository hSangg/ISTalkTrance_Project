import os
import numpy as np
import librosa
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from hmmlearn.hmm import GaussianHMM

# Feature extraction
def extract_features(audio_path, start, end, sr=16000):
    """
    Extract MFCC and related features from an audio segment.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, offset=start, duration=(end - start))
        if len(y) < sr * 0.5:  # Skip segments shorter than 0.5 seconds
            return None

        mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(delta)

        # Combine features and normalize
        features = np.vstack([mfcc, delta, delta_delta]).T
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        return features
    except Exception as e:
        print(f"Error extracting features for {audio_path} ({start}-{end}s): {e}")
        return None

# Train HMM model
def train_hmm_model(features_list, n_components=5):
    """
    Train an HMM using provided features.
    """
    try:
        features = np.vstack(features_list)
        model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)
        model.fit(features)
        return model
    except Exception as e:
        print(f"Error training HMM: {e}")
        return None

def time_to_seconds(time_str):
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except Exception as e:
        print(f"Error parsing time string '{time_str}': {e}")
        return None

def process_audio_segments(script_path, audio_path, speaker_map, models):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    predictions, true_labels, errors = [], [], []

    with open(script_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            start_str, end_str, true_speaker = parts[0], parts[1], parts[2]

            # Convert time strings to seconds
            start = time_to_seconds(start_str)
            end = time_to_seconds(end_str)

            if start is None or end is None:
                errors.append(f"Invalid time format in line: {line.strip()}")
                continue

            if true_speaker not in speaker_map:
                errors.append(f"Speaker {true_speaker} not in speaker map")
                continue

            features = extract_features(audio_path, start, end, sr)
            if features is None:
                errors.append(f"Invalid segment {start}-{end}s for speaker {true_speaker}")
                continue

            # Score with models
            try:
                scores = {speaker: models[speaker].score(features) for speaker in models.keys()}
                predicted_speaker = max(scores, key=scores.get)
            except Exception as e:
                errors.append(f"Error scoring segment {start}-{end}s: {e}")
                continue

            predictions.append(predicted_speaker)
            true_labels.append(true_speaker)

    return predictions, true_labels, errors

def evaluate_speaker_recognition(data_dir, model_dir):
    """
    Evaluate the speaker recognition system using data and models.
    """
    speaker_map = {filename.split('.')[0]: filename for filename in os.listdir(model_dir) if filename.endswith('.hmm')}
    models = {}

    # Load HMM models
    for speaker, model_file in speaker_map.items():
        model_path = os.path.join(model_dir, model_file)
        try:
            model = GaussianHMM(n_components=5, covariance_type='diag', n_iter=100)
            model = model.fromfile(model_path)
            models[speaker] = model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")

    true_labels, predictions = [], []
    errors = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        script_path = os.path.join(folder_path, 'script.txt')
        audio_path = os.path.join(folder_path, 'raw.wav')

        if not os.path.isfile(script_path) or not os.path.isfile(audio_path):
            errors.append(f"Missing script or audio file in {folder_path}")
            continue

        preds, trues, errs = process_audio_segments(script_path, audio_path, speaker_map, models)
        predictions.extend(preds)
        true_labels.extend(trues)
        errors.extend(errs)

    # Debugging: Print the speaker map and label list
    print("Speaker Map:", speaker_map)
    print("True Labels:", true_labels)
    print("Predictions:", predictions)

    # Calculate metrics if we have labels and predictions
    if true_labels and predictions:
        labels = list(speaker_map.keys())
        print("Labels used for confusion matrix:", labels)
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

        print("\nEvaluation Results:")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)

# Example usage
if __name__ == "__main__":
    data_directory = "train_voice"
    model_directory = "models"
    evaluate_speaker_recognition(data_directory, model_directory)
