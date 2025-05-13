import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


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

    print(f"🔍 Extracted xvectors: {xvector_dict.keys()}")
    return xvector_dict


def train_rnn_for_speaker(speaker, xvectors, num_epochs=50):
    if len(xvectors) == 0:
        print(f"❌ Không có dữ liệu xvector cho speaker: {speaker}")
        return

    xvectors = np.mean(xvectors, axis=1)
    input_dim = xvectors.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = SpeakerRNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.tensor(xvectors, dtype=torch.float32).unsqueeze(1)
    print(f"x_train shape: {x_train.shape}")

    y_train = torch.ones((len(xvectors), 1), dtype=torch.float32)  # Dummy labels

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    os.makedirs("rnn_xvector_models", exist_ok=True)

    torch.save({
        "input_dim": input_dim,
        "model_state_dict": model.state_dict()
    }, f"rnn_xvector_models/{speaker}.pth")

    print(f"✅ Đã lưu mô hình RNN cho {speaker}")

def load_rnn_model(speaker, input_dim):
    model_path = f"rnn_xvector_models/{speaker}.pth"
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy mô hình cho {speaker}")
        return None

    checkpoint = torch.load(model_path)
    input_dim = checkpoint["input_dim"]

    model = SpeakerRNN(input_dim, hidden_dim=128, output_dim=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})
train_voice_dir = "train_voice"
speaker_xvectors = {}

for subdir in os.listdir(train_voice_dir):
    subdir_path = os.path.join(train_voice_dir, subdir)
    audio_file = os.path.join(subdir_path, "raw.WAV")
    script_file = os.path.join(subdir_path, "script.txt")

    if os.path.isfile(audio_file) and os.path.isfile(script_file):
        print(f"📂 Processing: {subdir}")

        segments, speakers = load_script(script_file)
        xvector_dict = extract_xvectors(audio_file, zip(segments, speakers), classifier)

        for speaker, xvectors in xvector_dict.items():
            if len(xvectors) == 0:
                print(f"⚠️ Không có xvector cho speaker: {speaker}")
                continue  # Bỏ qua nếu không có dữ liệu

            if speaker not in speaker_xvectors:
                speaker_xvectors[speaker] = []
            speaker_xvectors[speaker].extend(xvectors)

for speaker, xvectors in speaker_xvectors.items():
    train_rnn_for_speaker(speaker, xvectors)