import os

import joblib
import numpy as np
import torch
import torchaudio
from hmmlearn import hmm
from speechbrain.inference.speaker import SpeakerRecognition

spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="speechbrain_models/spkrec-ecapa-voxceleb"
)

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

def extract_dvectors(audio_path, segments_speakers):
    waveform, sample_rate = torchaudio.load(audio_path)
    dvector_dict = {}
    for (start, end), speaker in segments_speakers:
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
    try:
        dvectors = np.vstack(dvectors)
        if dvectors.shape[0] < 10:
            return
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(dvectors)
        os.makedirs("hmm_dvector_models", exist_ok=True)
        joblib.dump(model, f"hmm_dvector_models/{speaker}_model.pkl")
    except Exception as e:
        print(f"Failed to train HMM for {speaker}: {e}")

def train_all_from_folder(train_root="train_voice"):
    all_dvectors = {}

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
        print("extract for: ", script_path)
        dvector_dict = extract_dvectors(audio_path, zip(segments, speakers))

        for speaker, vecs in dvector_dict.items():
            if speaker not in all_dvectors:
                all_dvectors[speaker] = []
            all_dvectors[speaker].extend(vecs)

    for speaker, dvectors in all_dvectors.items():
        print("🏃‍ train for : ️", speaker)
        train_hmm_for_speaker(speaker, dvectors)

if __name__ == "__main__":
    train_all_from_folder("test_voice")
