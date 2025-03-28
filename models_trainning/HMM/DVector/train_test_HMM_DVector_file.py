import numpy as np
import torchaudio
import joblib
from hmmlearn import hmm
import os
import torch
from speechbrain.inference.speaker import SpeakerRecognition

spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                            savedir="speechbrain_models/spkrec-ecapa-voxceleb")


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

def extract_dvectors(audio_path, segments):
    waveform, sample_rate = torchaudio.load(audio_path)
    dvector_dict = {}

    for (start, end), speaker in segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        with torch.no_grad():
            embedding = spk_model.encode_batch(segment_waveform).squeeze().numpy()

        if speaker not in dvector_dict:
            dvector_dict[speaker] = []
        dvector_dict[speaker].append(embedding)

    return dvector_dict
def train_hmm_for_speaker(speaker, dvectors):
    dvectors = np.vstack(dvectors)
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
    model.fit(dvectors)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{speaker}_dvector_model.pkl")
    print(f"✅ Đã lưu mô hình HMM cho {speaker}")

def load_hmm_model(speaker):
    model_path = f"models/{speaker}_dvector_model.pkl"
    return joblib.load(model_path) if os.path.exists(model_path) else None

def predict_speaker(speaker, dvectors):
    model = load_hmm_model(speaker)
    if model is None:
        print(f"⚠️ Không tìm thấy mô hình cho {speaker}")
        return []
    return model.predict(np.vstack(dvectors))

# ======= Chạy thử nghiệm =======
audio_file = "train_voice/vnoi_talkshow/ezyZip.wav"
script_file = "train_voice/vnoi_talkshow/script.txt"

# Load script và trích xuất đặc trưng
segments, speakers = load_script(script_file)
dvector_dict = extract_dvectors(audio_file, zip(segments, speakers))

# Huấn luyện mô hình HMM
for speaker, dvectors in dvector_dict.items():
    train_hmm_for_speaker(speaker, dvectors)
