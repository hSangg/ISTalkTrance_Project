import os

import numpy as np
import torch
import torchaudio
from hmmlearn import hmm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from speechbrain.inference.speaker import SpeakerRecognition

spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="speechbrain_models/spkrec-ecapa-voxceleb"
)

N_SPLITS = 3

def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[:3]
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers

def extract_dvectors(audio_path, segments, speakers):
    waveform, sample_rate = torchaudio.load(audio_path)
    features = []
    for (start, end), speaker in zip(segments, speakers):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        if segment_waveform.shape[1] < 1600:  # Skip too short segments
            continue
        with torch.no_grad():
            embedding = spk_model.encode_batch(segment_waveform).squeeze().numpy()
        features.append((embedding, speaker))
    return features

def train_hmm(features):
    X = np.vstack(features)
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    std[std < 1e-10] = 1.0
    X = (X - mean) / std
    model = hmm.GaussianHMM(n_components=min(3, len(X)), covariance_type="diag", n_iter=100)
    model.fit(X)
    return model, mean, std

def predict_segment(feature, models):
    scores = {}
    for speaker, (model, mean, std) in models.items():
        norm_feat = (feature - mean) / std
        try:
            score = model.score([norm_feat])
            scores[speaker] = score
        except:
            scores[speaker] = -np.inf
    return max(scores, key=scores.get)

def cross_validate(all_features, n_splits=N_SPLITS):
    X = np.array([feat for feat, label in all_features])
    y = np.array([label for feat, label in all_features])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{n_splits}")
        train_data = [(X[i], y[i]) for i in train_idx]
        test_data = [(X[i], y[i]) for i in test_idx]

        speaker_feats = {}
        for feat, label in train_data:
            speaker_feats.setdefault(label, []).append(feat)

        models = {}
        for speaker, feats in speaker_feats.items():
            try:
                model, mean, std = train_hmm(feats)
                models[speaker] = (model, mean, std)
            except Exception as e:
                print(f"❌ Error training HMM for {speaker}: {e}")

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
        folder_path = os.path.join(train_root, folder)
        if not os.path.isdir(folder_path):
            continue

        audio_path = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')), None)
        script_path = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.txt')), None)

        if not audio_path or not script_path:
            print(f"❌ Thiếu file .wav hoặc .txt trong {folder_path}")
            continue

        segments, speakers = load_script(script_path)
        print(f"🎙️ Processing {folder_path}")
        features = extract_dvectors(audio_path, segments, speakers)
        all_features.extend(features)

    if len(all_features) >= N_SPLITS:
        cross_validate(all_features)
    else:
        print("❌ Not enough data for cross-validation.")
