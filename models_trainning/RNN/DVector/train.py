import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def train_rnn_for_speaker(speaker, dvectors, num_epochs=50):
    dvectors = np.mean(dvectors, axis=1)
    input_dim = dvectors.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = SpeakerRNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(dvectors, dtype=torch.float32).unsqueeze(1)
    y_train = torch.ones((len(dvectors), 1), dtype=torch.float32)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    os.makedirs("rnn_dvector_models", exist_ok=True)

    torch.save({
        'input_dim': input_dim,
        'model_state_dict': model.state_dict()
    }, f"rnn_dvector_models/{speaker}.pth")

    print(f"✅ Đã lưu mô hình RNN cho {speaker}, input_dim = {input_dim}")

root_folder = "train_voice"
speaker_dvectors = {}

print("start.........")

for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    audio_path = os.path.join(subfolder_path, "raw.WAV")
    script_path = os.path.join(subfolder_path, "script.txt")

    if os.path.exists(audio_path) and os.path.exists(script_path):
        segments, speakers = load_script(script_path)
        print("🏃‍♂️ at: ", subfolder_path)
        dvector_dict = extract_dvectors(audio_path, zip(segments, speakers))
        for speaker, dvectors in dvector_dict.items():
            print("extract dvectors for speaker", speaker)
            if speaker not in speaker_dvectors:
                speaker_dvectors[speaker] = []
            speaker_dvectors[speaker].extend(dvectors)

for speaker, dvectors in speaker_dvectors.items():
    print("train for: ", speaker)
    train_rnn_for_speaker(speaker, np.array(dvectors))