import os
from collections import defaultdict

import joblib
import numpy as np
import torch
import torchaudio
from hmmlearn import hmm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from speechbrain.inference.classifiers import EncoderClassifier

TRAIN_ROOT = "test_voice"

def cross_validate(k_folds=5):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})

    data = []

    for subfolder in os.listdir(TRAIN_ROOT):
        print("✨ process at: ", subfolder)
        subfolder_path = os.path.join(TRAIN_ROOT, subfolder)
        audio_path = os.path.join(subfolder_path, "raw.WAV")
        script_path = os.path.join(subfolder_path, "script.txt")

        if os.path.exists(audio_path) and os.path.exists(script_path):
            segments, speakers = load_script(script_path)
            xvector_dict = extract_xvectors(audio_path, segments, speakers, classifier)

            for speaker, xvectors in xvector_dict.items():
                for vec in xvectors:
                    data.append((vec, speaker))

    print(f"📦 Loaded {len(data)} samples for cross-validation")

    X = [x for x, _ in data]
    y = [label for _, label in data]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_preds = []
    all_truth = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n🔁 Fold {fold}/{k_folds}")

        train_dict = defaultdict(list)
        for idx in train_idx:
            train_dict[y[idx]].append(X[idx])

        hmm_models = {}
        for speaker, xvectors in train_dict.items():
            model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
            model.fit(np.vstack(xvectors))
            hmm_models[speaker] = model

        fold_preds = []
        fold_truth = []

        for idx in test_idx:
            true_speaker = y[idx]
            xvector = X[idx]
            predicted = predict_speaker(hmm_models, xvector)

            fold_preds.append(predicted)
            fold_truth.append(true_speaker)

        all_preds.extend(fold_preds)
        all_truth.extend(fold_truth)

        print(classification_report(fold_truth, fold_preds, zero_division=0))

    print("\n==== 📊 Overall Evaluation ====")
    print(classification_report(all_truth, all_preds, zero_division=0))
    print(f"🎯 Accuracy: {accuracy_score(all_truth, all_preds):.4f}")

def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def predict_speaker(hmm_models, xvector):
    max_score = float('-inf')
    predicted_speaker = None

    for speaker, model in hmm_models.items():
        try:
            score = model.score(xvector)
            if score > max_score:
                max_score = score
                predicted_speaker = speaker
        except:
            continue
    return predicted_speaker

def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            start, end, speaker = line.strip().split()
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers


def process_embedding(embedding):
    """Process embedding to ensure consistent shape for HMM"""
    if embedding.ndim == 3:
        embedding = embedding.reshape(embedding.shape[0], -1)

    elif embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    if embedding.shape[1] == 1024:
        first_half = embedding[:, :512]
        return first_half

    return embedding


def extract_xvectors(audio_path, segments, speakers, classifier):
    waveform, sample_rate = torchaudio.load(audio_path)
    xvector_dict = {}

    for (start, end), speaker in zip(segments, speakers):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        with torch.no_grad():
            embedding = classifier.encode_batch(segment_waveform).squeeze().numpy()

        embedding = process_embedding(embedding)

        if speaker not in xvector_dict:
            xvector_dict[speaker] = []
        xvector_dict[speaker].append(embedding)

    return xvector_dict


def train_hmm_for_speaker(speaker, xvectors):
    xvectors = np.vstack(xvectors)

    print(f"Training {speaker} model with data shape: {xvectors.shape}")

    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
    model.fit(xvectors)

    os.makedirs("hmm_xvector_models", exist_ok=True)
    joblib.dump(model, f"hmm_xvector_models/{speaker}_model.pkl")
    print(f"✅ Saved HMM model for {speaker}")

    with open(f"hmm_xvector_models/{speaker}_features.txt", "w") as f:
        f.write(str(xvectors.shape[1]))

    return xvectors.shape[1]


def train_all():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})

    train_root = TRAIN_ROOT
    feature_dim = None

    for subfolder in os.listdir(train_root):
        subfolder_path = os.path.join(train_root, subfolder)
        audio_path = os.path.join(subfolder_path, "raw.WAV")
        script_path = os.path.join(subfolder_path, "script.txt")

        if os.path.exists(audio_path) and os.path.exists(script_path):
            print(f"🔍 Processing folder: {subfolder}")

            segments, speakers = load_script(script_path)
            xvector_dict = extract_xvectors(audio_path, segments, speakers, classifier)

            for speaker, xvectors in xvector_dict.items():
                dim = train_hmm_for_speaker(speaker, xvectors)
                if feature_dim is None:
                    feature_dim = dim
                elif feature_dim != dim:
                    print(f"⚠️ Warning: Inconsistent feature dimensions. Found {dim}, expected {feature_dim}")
        else:
            print(f"⚠️ Skipping {subfolder} because raw.WAV or script.txt is missing")

    with open("hmm_xvector_models/feature_dim.txt", "w") as f:
        f.write(str(feature_dim))

if __name__ == "__main__":
    print("==== Step 1: Training hmm_wavelet_models ====")
    cross_validate()