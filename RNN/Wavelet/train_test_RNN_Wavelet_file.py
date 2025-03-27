import numpy as np
import torchaudio
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader

# Mô hình LSTM để phân loại người nói
class SpeakerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Dataset tùy chỉnh
class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Tiền xử lý dữ liệu
# Hàm này trích xuất đặc trưng wavelet giống như mã gốc

def extract_wavelet_features(audio_path, segments_speakers, wavelet='db4', level=1):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy().squeeze()
    feature_list, label_list = [], []
    
    for (start, end), speaker in segments_speakers:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[start_sample:end_sample]
        
        if len(segment_waveform) < 2**level:
            continue
        
        coeffs = pywt.wavedec(segment_waveform, wavelet, level=min(level, pywt.dwt_max_level(len(segment_waveform), pywt.Wavelet(wavelet).dec_len)))
        features = np.hstack([np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs])
        feature_list.append(features)
        label_list.append(speaker)
    
    return feature_list, label_list

# Huấn luyện mô hình RNN
def train_rnn(train_features, train_labels, num_epochs=50, lr=0.001, hidden_size=64):
    speakers = list(set(train_labels))
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    labels = [speaker_to_idx[s] for s in train_labels]
    
    dataset = SpeakerDataset(train_features, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_size = len(train_features[0])
    model = SpeakerRNN(input_size, hidden_size, len(speakers))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)  # LSTM yêu cầu (batch, seq, feature)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    
    os.makedirs("models", exist_ok=True)
    print(f"Save models")
    torch.save(model.state_dict(), "models/speaker_rnn.pth")
    return model, speaker_to_idx

# ======= Chạy thử nghiệm =======
audio_file = "train_voice/extraordinary_strategic/raw.WAV"
script_file = "train_voice/extraordinary_strategic/script.txt"

def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s

def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Bỏ qua dòng không hợp lệ
            start, end, speaker = parts[0], parts[1], parts[2]
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers

segments, speakers = load_script(script_file)
features, labels = extract_wavelet_features(audio_file, zip(segments, speakers))
model, speaker_to_idx = train_rnn(features, labels)