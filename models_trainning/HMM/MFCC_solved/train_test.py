import os

import librosa
import numpy as np
import torchaudio
from hmmlearn import hmm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 3
N_MFCC = 13

def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, speaker = parts
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers

def extract_mfcc_features(audio_path, segments, speakers, n_mfcc=N_MFCC):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.numpy().squeeze()

    feature_list = []

    for (start, end), speaker in zip(segments, speakers):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]

        if len(segment_waveform) < 400:
            continue

        mfcc = librosa.feature.mfcc(y=segment_waveform, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # shape: (n_frames, n_mfcc)
        feature_list.append((mfcc, speaker))

    return feature_list

def train_hmm(sequences):
    X = np.vstack(sequences)
    lengths = [len(seq) for seq in sequences]

    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    std[std < 1e-10] = 1.0
    X_norm = (X - mean) / std

    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
    model.fit(X_norm, lengths)

    return model, mean, std

def predict_segment(mfcc, models):
    best_speaker, best_score = None, float("-inf")
    for speaker, (model, mean, std) in models.items():
        try:
            norm = (mfcc - mean) / std
            score = model.score(norm)
            if score > best_score:
                best_speaker, best_score = speaker, score
        except Exception:
            continue
    return best_speaker

def cross_validate(features_with_labels, n_splits=N_SPLITS):
    X = [feat for feat, label in features_with_labels]
    y = [label for feat, label in features_with_labels]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{n_splits}")
        train_data = [(X[i], y[i]) for i in train_idx]
        test_data = [(X[i], y[i]) for i in test_idx]

        speaker_sequences = {}
        for feat, label in train_data:
            speaker_sequences.setdefault(label, []).append(feat)

        models = {}
        for speaker, sequences in speaker_sequences.items():
            model, mean, std = train_hmm(sequences)
            models[speaker] = (model, mean, std)

        y_true, y_pred = [], []
        for feat, label in test_data:
            pred = predict_segment(feat, models)
            y_true.append(label)
            y_pred.append(pred)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(classification_report(y_true, y_pred, zero_division=0))

    print("\n🔚 Final Evaluation Across All Folds:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    print("✅ Accuracy:", accuracy_score(all_y_true, all_y_pred))

if __name__ == "__main__":
    all_features = []
    train_root = "train_voice"

    for folder in os.listdir(train_root):
        subfolder = os.path.join(train_root, folder)
        audio_file = os.path.join(subfolder, "raw.WAV")
        script_file = os.path.join(subfolder, "script.txt")

        if not os.path.exists(audio_file) or not os.path.exists(script_file):
            continue

        print(f"🎙️ Processing {subfolder}")
        segments, speakers = load_script(script_file)
        features = extract_mfcc_features(audio_file, segments, speakers)
        all_features.extend(features)

    if len(all_features) >= N_SPLITS:
        cross_validate(all_features)
    else:
        print("❌ Not enough data for cross-validation.")
