import os

import joblib
import numpy as np
import torch
import torchaudio
from hmmlearn import hmm
from speechbrain.inference.classifiers import EncoderClassifier


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            start, end, speaker = line.strip().split()
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers

def extract_xvectors(audio_path, segments, classifier):
    waveform, sample_rate = torchaudio.load(audio_path)
    xvector_dict = {}
    
    for (start, end), speaker in segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        
        with torch.no_grad():
            embedding = classifier.encode_batch(segment_waveform).squeeze().numpy()
        
        if speaker not in xvector_dict:
            xvector_dict[speaker] = []
        xvector_dict[speaker].append(embedding)
    
    return xvector_dict

def train_hmm_for_speaker(speaker, xvectors):
    xvectors = np.vstack(xvectors)  # Convert list to numpy array
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
    model.fit(xvectors)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{speaker}_model.pkl")
    print(f"✅ Đã lưu mô hình HMM cho {speaker}")

def load_hmm_model(speaker):
    model_path = f"models/hmm_{speaker}.pkl"
    return joblib.load(model_path) if os.path.exists(model_path) else None

def predict_speaker(speaker, xvectors):
    model = load_hmm_model(speaker)
    if model is None:
        print(f"⚠️ Không tìm thấy mô hình cho {speaker}")
        return []
    return model.predict(np.vstack(xvectors))

# ======= Chạy thử nghiệm =======
audio_file = "train_voice/vnoi_talkshow/raw.WAV"
script_file = "train_voice/vnoi_talkshow/script.txt"

# Load SpeechBrain XVector model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})

# Load script và trích xuất đặc trưng
segments, speakers = load_script(script_file)
xvector_dict = extract_xvectors(audio_file, zip(segments, speakers), classifier)

# Huấn luyện mô hình HMM
for speaker, xvectors in xvector_dict.items():
    train_hmm_for_speaker(speaker, xvectors)
