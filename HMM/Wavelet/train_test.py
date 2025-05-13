import os

import numpy as np
import pywt
import torchaudio
from hmmlearn import hmm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

WAVELET = "db4"
LEVEL = 1
N_SPLITS = 3


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


def extract_wavelet_features(audio_path, segments, speakers, wavelet=WAVELET, level=LEVEL):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    waveform = waveform.numpy().squeeze()

    feature_list = []

    def safe_stat(func, arr, default=0.0):
        return func(arr) if len(arr) > 0 else default

    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]

        if len(segment_waveform) < 2 ** level:
            continue

        max_level = min(level, pywt.dwt_max_level(len(segment_waveform), pywt.Wavelet(wavelet).dec_len))
        coeffs = pywt.wavedec(segment_waveform, wavelet, level=max_level)

        features = []
        for coeff in coeffs:
            features.extend([
                safe_stat(np.mean, coeff),
                safe_stat(np.std, coeff, 1.0),
                safe_stat(np.max, coeff),
                safe_stat(np.min, coeff),
                safe_stat(np.median, coeff),
                safe_stat(lambda x: np.sum(x ** 2), coeff)
            ])
        feature_list.append((np.array(features), speakers[i]))

    return feature_list


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
        try:
            norm_feat = (feature - mean) / std
            score = model.score([norm_feat])
            scores[speaker] = score
        except ValueError as e:
            continue

    if not scores:
        return None  # fallback nếu không có mô hình nào hợp lệ

    return max(scores, key=scores.get)

def cross_validate(features_with_labels, n_splits=N_SPLITS):
    X = np.array([feat for feat, label in features_with_labels])
    y = np.array([label for feat, label in features_with_labels])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{n_splits}")
        train_data = [(X[i], y[i]) for i in train_idx]
        test_data = [(X[i], y[i]) for i in test_idx]

        speaker_feats = {}
        for feat, label in train_data:
            speaker_feats.setdefault(label, []).append(feat)

        models = {}
        for speaker, feats in speaker_feats.items():
            model, mean, std = train_hmm(feats)
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
        features = extract_wavelet_features(audio_file, segments, speakers)
        all_features.extend(features)

    if len(all_features) >= N_SPLITS:
        cross_validate(all_features)
    else:
        print("❌ Not enough data for cross-validation.")
