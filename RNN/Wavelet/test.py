import os

import numpy as np
import pywt
import torch
import torch.nn as nn
import torchaudio


class SpeakerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}"


def extract_wavelet_features(audio_path, segments, wavelet='db4', level=1):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy().squeeze()
    feature_list = []

    for start, end in segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]

        if len(segment_waveform) < 2 ** level:
            continue

        coeffs = pywt.wavedec(segment_waveform, wavelet, level=min(level, pywt.dwt_max_level(len(segment_waveform),
                                                                                             pywt.Wavelet(
                                                                                                 wavelet).dec_len)))
        features = np.hstack([np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs])
        feature_list.append(features)

    return feature_list


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s


def load_script(script_path):
    segments = []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start, end = parts[0], parts[1]
            segments.append((time_to_seconds(start), time_to_seconds(end)))
    return segments


def predict_speakers(model, features, idx_to_speaker):
    model.eval()
    predictions = []
    with torch.no_grad():
        for feature in features:
            input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predictions.append(idx_to_speaker[predicted_label])
    return predictions


def process_audio_and_script(audio_path, script_path, model_path, speaker_map_path, output_script):
    segments = load_script(script_path)
    features = extract_wavelet_features(audio_path, segments)

    if not features:
        print("Không có đặc trưng hợp lệ để phân loại.")
        return

    speaker_to_idx = torch.load(speaker_map_path)
    idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}

    input_size = len(features[0])
    output_size = len(speaker_to_idx)
    model = SpeakerRNN(input_size=input_size, hidden_size=64, output_size=output_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    predicted_speakers = predict_speakers(model, features, idx_to_speaker)

    with open(output_script, "w", encoding="utf-8") as file:
        for (start, end), speaker in zip(segments, predicted_speakers):
            file.write(f"{format_time(start)} {format_time(end)} {speaker}\n")
    print(f"Script mới được lưu tại: {output_script}")


def process_all_subfolders(test_voice_folder, model_path, speaker_map_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(speaker_map_path):
        print(f"Error: Speaker mapping file not found at {speaker_map_path}")
        return

    for subfolder in os.listdir(test_voice_folder):
        subfolder_path = os.path.join(test_voice_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nProcessing folder: {subfolder}")

        audio_file = os.path.join(subfolder_path, "raw.WAV")
        if not os.path.exists(audio_file):
            for ext in ["wav", "mp3", "flac"]:
                alt_audio = os.path.join(subfolder_path, f"raw.{ext}")
                if os.path.exists(alt_audio):
                    audio_file = alt_audio
                    break

        script_file = os.path.join(subfolder_path, "script.txt")
        output_script = os.path.join(subfolder_path, "script_predicted.txt")

        process_audio_and_script(
            audio_file,
            script_file,
            model_path,
            speaker_map_path,
            output_script
        )


if __name__ == "__main__":
    test_voice_folder = "test_voice"
    model_path = "rnn_wavelet_models/model.pth"
    speaker_map_path = "rnn_wavelet_models/speaker_to_idx.pth"

    process_all_subfolders(test_voice_folder, model_path, speaker_map_path)
