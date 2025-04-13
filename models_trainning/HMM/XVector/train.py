import os

import joblib
import numpy as np
import torch
import torchaudio
from hmmlearn import hmm
from speechbrain.inference.classifiers import EncoderClassifier

TEST_ROOT = "test_voice"
TRAIN_ROOT = "train_voice"


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


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


def predict_speakers():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})

    models = {}
    model_dir = "hmm_xvector_models"

    feature_dim = 512
    try:
        with open(os.path.join(model_dir, "feature_dim.txt"), "r") as f:
            feature_dim = int(f.read().strip())
            print(f"Using feature dimension: {feature_dim}")
    except:
        print(f"Using default feature dimension: {feature_dim}")

    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory '{model_dir}' not found. Run training first.")
        return

    for model_file in os.listdir(model_dir):
        if model_file.endswith("_model.pkl"):
            speaker = model_file.replace("_model.pkl", "")
            model_path = os.path.join(model_dir, model_file)
            try:
                models[speaker] = joblib.load(model_path)
                print(f"✅ Loaded model for speaker: {speaker}")
            except Exception as e:
                print(f"❌ Failed to load model for {speaker}: {e}")

    if not models:
        print("❌ No hmm_wavelet_models found. Please train hmm_wavelet_models first.")
        return

    test_root = TEST_ROOT
    if not os.path.exists(test_root):
        print(f"❌ Test directory '{test_root}' not found.")
        return

    for subfolder in os.listdir(test_root):
        subfolder_path = os.path.join(test_root, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        audio_path = os.path.join(subfolder_path, "raw.WAV")
        script_path = os.path.join(subfolder_path, "script.txt")

        if os.path.exists(audio_path) and os.path.exists(script_path):
            print(f"🔍 Processing folder: {subfolder}")

            waveform, sample_rate = torchaudio.load(audio_path)

            segments, _ = load_script(script_path)

            predictions = []

            for i, (start, end) in enumerate(segments):
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]

                with torch.no_grad():
                    embedding = classifier.encode_batch(segment_waveform).squeeze().numpy()

                embedding = process_embedding(embedding)

                print(
                    f"  Segment {i + 1}/{len(segments)}: {seconds_to_time(start)} - {seconds_to_time(end)}, embedding shape: {embedding.shape}")

                max_score = float('-inf')
                predicted_speaker = None

                for speaker, model in models.items():
                    try:
                        score = model.score(embedding)
                        if score > max_score:
                            max_score = score
                            predicted_speaker = speaker
                    except Exception as e:
                        print(f"  Error with {speaker} model: {str(e)}")

                predictions.append((seconds_to_time(start), seconds_to_time(end), predicted_speaker))
                print(f"  Predicted: {predicted_speaker}")

            output_path = os.path.join(subfolder_path, "script_predicted.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for start, end, speaker in predictions:
                    f.write(f"{start} {end} {speaker or 'unknown'}\n")

            print(f"✅ Created {output_path}")
        else:
            print(f"⚠️ Skipping {subfolder} because raw.WAV or script.txt is missing")
if __name__ == "__main__":
    print("==== Step 1: Training hmm_wavelet_models ====")
    train_all()
    print("\n==== Step 2: Predicting speakers ====")
    predict_speakers()