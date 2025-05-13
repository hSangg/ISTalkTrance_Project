import os

import joblib
import numpy as np
import pywt
import torchaudio
from hmmlearn import hmm


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def load_script(script_path, is_training=True):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if is_training:
                if len(parts) != 3:
                    print(f"⚠️ Bỏ qua dòng sai định dạng trong train: {line.strip()}")
                    continue
                start, end, speaker = parts
                segments.append((time_to_seconds(start), time_to_seconds(end)))
                speakers.append(speaker)
            else:
                if len(parts) < 2:
                    print(f"⚠️ Bỏ qua dòng sai định dạng trong test: {line.strip()}")
                    continue
                start, end = parts[:2]
                segments.append((time_to_seconds(start), time_to_seconds(end)))
    return segments, speakers


def extract_wavelet_features(audio_path, segments, speakers=None, wavelet='db4', level=1):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    waveform = waveform.numpy().squeeze()
    feature_dict = {} if speakers else []

    def safe_stat(func, arr, default=0.0):
        if len(arr) == 0:
            return default
        return func(arr)

    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]

        if len(segment_waveform) < 2 ** level:
            continue

        coeffs = pywt.wavedec(segment_waveform, wavelet, level=min(level, pywt.dwt_max_level(len(segment_waveform),
                                                                                             pywt.Wavelet(
                                                                                                 wavelet).dec_len)))

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

        features = np.array(features)

        if speakers:
            speaker = speakers[i]
            if speaker not in feature_dict:
                feature_dict[speaker] = []
            feature_dict[speaker].append(features)
        else:
            feature_dict.append(features)

    return feature_dict


def train_hmm_for_speaker(speaker, features):
    if not features:
        print(f"⚠️ Không có đặc trưng để huấn luyện cho người nói {speaker}")
        return

    features = np.vstack(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std < 1e-10] = 1.0
    features = (features - mean) / std

    n_components = min(3, len(features))
    model = hmm.GaussianHMM(n_components, covariance_type="diag", n_iter=100)
    model.fit(features)

    os.makedirs("hmm_wavelet_models", exist_ok=True)
    joblib.dump(model, f"hmm_wavelet_models/{speaker}_model.pkl")
    joblib.dump((mean, std), f"hmm_wavelet_models/{speaker}_norm_params.pkl")
    print(f"✅ Đã lưu mô hình HMM cho {speaker}")


def load_hmm_model(speaker):
    try:
        model = joblib.load(f"hmm_wavelet_models/{speaker}_model.pkl")
        mean, std = joblib.load(f"hmm_wavelet_models/{speaker}_norm_params.pkl")
        return model, (mean, std)
    except:
        return None, None


def predict_speaker_all_models(features):
    predictions = []
    all_speakers = [f.split('_model.pkl')[0] for f in os.listdir("hmm_wavelet_models") if f.endswith("_model.pkl")]

    for feat in features:
        max_score = float("-inf")
        best_speaker = "Unknown"
        for speaker in all_speakers:
            model, (mean, std) = load_hmm_model(speaker)
            std[std < 1e-10] = 1.0
            norm_feat = (feat - mean) / std
            try:
                score = model.score([norm_feat])
                if score > max_score:
                    max_score = score
                    best_speaker = speaker
            except:
                continue
        predictions.append(best_speaker)
    return predictions


train_root = "train_voice"
speaker_features = {}

for folder in os.listdir(train_root):
    subfolder = os.path.join(train_root, folder)
    audio_file = os.path.join(subfolder, "raw.WAV")
    script_file = os.path.join(subfolder, "script.txt")
    print("🏃‍♂️ load wavelet at: ", subfolder)

    segments, speakers = load_script(script_file, is_training=True)
    feature_dict = extract_wavelet_features(audio_file, segments, speakers)

    for speaker, feats in feature_dict.items():
        if speaker not in speaker_features:
            speaker_features[speaker] = []
        speaker_features[speaker].extend(feats)

for speaker, feats in speaker_features.items():
    print("🏃‍♂️ train hmm for: ", speaker)
    train_hmm_for_speaker(speaker, feats)

test_root = "test_voice"

for folder in os.listdir(test_root):
    subfolder = os.path.join(test_root, folder)
    audio_file = os.path.join(subfolder, "raw.WAV")
    script_file = os.path.join(subfolder, "script.txt")

    segments, _ = load_script(script_file, is_training=False)
    features = extract_wavelet_features(audio_file, segments, speakers=None)

    predictions = predict_speaker_all_models(features)

    output_file = os.path.join(subfolder, "script_predicted.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for (start, end), speaker in zip(segments, predictions):
            start_ts = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02}"
            end_ts = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02}"
            f.write(f"{start_ts} {end_ts} {speaker}\n")

    print(f"📄 Đã lưu file dự đoán: {output_file}")
