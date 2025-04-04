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
        os.makedirs("hmm_wavelet_models", exist_ok=True)
        joblib.dump(model, f"hmm_wavelet_models/{speaker}_dvector_model.pkl")
    except Exception as e:
        print(f"Failed to train HMM for {speaker}: {e}")
def train_all_from_folder(train_root="train_voice"):
    all_dvectors = {}

    for folder in os.listdir(train_root):
        folder_path = os.path.join(train_root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Tìm file .wav và .txt trong folder
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

    # Huấn luyện HMM sau khi gom đủ dữ liệu
    for speaker, dvectors in all_dvectors.items():
        print("🏃‍ train for : ️", speaker)
        train_hmm_for_speaker(speaker, dvectors)


def predict_speaker_for_segment(dvector):
    scores = {}
    for model_file in os.listdir("hmm_wavelet_models"):
        speaker_name = model_file.split("_dvector_model.pkl")[0]
        model = joblib.load(os.path.join("hmm_wavelet_models", model_file))
        try:
            log_likelihood = model.score(dvector.reshape(1, -1))
            scores[speaker_name] = log_likelihood
        except:
            continue
    if not scores:
        return "unknown"
    return max(scores, key=scores.get)

def predict_test_folder(test_folder="test_voice"):
    for folder in os.listdir(test_folder):
        folder_path = os.path.join(test_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        audio_path = os.path.join(folder_path, "audio.wav")
        script_path = os.path.join(folder_path, "script.txt")
        output_path = os.path.join(folder_path, "script_predicted.txt")
        if not os.path.exists(audio_path) or not os.path.exists(script_path):
            continue
        segments = []
        with open(script_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    start, end = parts[:2]
                    segments.append((time_to_seconds(start), time_to_seconds(end)))
        waveform, sample_rate = torchaudio.load(audio_path)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for (start, end) in segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                with torch.no_grad():
                    embedding = spk_model.encode_batch(segment_waveform).squeeze().numpy()
                speaker = predict_speaker_for_segment(embedding)
                out_f.write(f"{start//3600:02}:{(start%3600)//60:02}:{start%60:02} "
                            f"{end//3600:02}:{(end%3600)//60:02}:{end%60:02} {speaker}\n")

if __name__ == "__main__":
    train_all_from_folder("train_voice")
    predict_test_folder("test_voice")
