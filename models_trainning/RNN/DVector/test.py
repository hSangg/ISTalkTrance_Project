import os

import torch
import torch.nn as nn
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir="speechbrain_models/spkrec-ecapa-voxceleb")


class SpeakerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


def load_script(script_path):
    segments = []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 2:
                print(f"⚠️ Lỗi: Dòng không đủ dữ liệu -> {line.strip()}")
                continue

            start, end = parts[:2]
            segments.append((time_to_seconds(start), time_to_seconds(end)))

    return segments


def extract_dvectors(audio_path, segments):
    waveform, sample_rate = torchaudio.load(audio_path)
    dvectors = []

    for (start, end) in segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        with torch.no_grad():
            embedding = spk_model.encode_batch(segment_waveform).squeeze().numpy()

        dvectors.append(embedding)

    return dvectors


def predict_speaker(dvectors, speaker_models):
    predictions = []
    for dvector in dvectors:
        dvector_tensor = torch.tensor(dvector, dtype=torch.float32).unsqueeze(0)
        best_speaker = None
        best_score = float("-inf")

        for speaker, model in speaker_models.items():
            with torch.no_grad():
                score = model(dvector_tensor).item()

            if score > best_score:
                best_score = score
                best_speaker = speaker

        predictions.append(best_speaker)

    return predictions


def load_speaker_models(model_folder):
    speaker_models = {}

    for model_file in os.listdir(model_folder):
        speaker = model_file.replace(".pth", "")
        model_path = os.path.join(model_folder, model_file)

        checkpoint = torch.load(model_path)
        input_dim = checkpoint['input_dim']

        model = SpeakerRNN(input_dim, 128, 1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        speaker_models[speaker] = model

    return speaker_models


test_folder = "test_voice"

speaker_models = load_speaker_models("rnn_dvector_models")

for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)
    audio_path = os.path.join(subfolder_path, "raw.WAV")
    script_path = os.path.join(subfolder_path, "script.txt")

    if os.path.exists(audio_path) and os.path.exists(script_path):
        print(f"🔍 Dự đoán speaker cho {subfolder}...")

        segments = load_script(script_path)
        dvectors = extract_dvectors(audio_path, segments)

        predicted_speakers = predict_speaker(dvectors, speaker_models)

        output_script_path = os.path.join(subfolder_path, "script_predicted.txt")
        with open(output_script_path, "w", encoding="utf-8") as file:
            for (start, end), speaker in zip(segments, predicted_speakers):
                file.write(f"{seconds_to_time(start)} {seconds_to_time(end)} {speaker}\n")

        print(f"✅ Đã lưu kết quả vào {output_script_path}")
