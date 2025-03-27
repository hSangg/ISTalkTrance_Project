import numpy as np
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from speechbrain.inference.speaker import SpeakerRecognition
import os

# Load D-Vector model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir="speechbrain_models/spkrec-ecapa-voxceleb")

# RNN Model
class SpeakerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Lấy kết quả cuối cùng
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
    dvectors = np.mean(dvectors, axis=1)  # Trung bình đặc trưng
    input_dim = dvectors.shape[1]
    hidden_dim = 128
    output_dim = 1  # 1 lớp đầu ra cho mỗi người nói

    model = SpeakerRNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(dvectors, dtype=torch.float32).unsqueeze(1)
    print(f"x_train shape: {x_train.shape}")  # Kiểm tra định dạng trước khi đưa vào model

    y_train = torch.ones((len(dvectors), 1), dtype=torch.float32)  # Dummy labels

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{speaker}_rnn.pth")
    print(f"✅ Đã lưu mô hình RNN cho {speaker}")

def load_rnn_model(speaker, input_dim):
    model_path = f"models/{speaker}_rnn.pth"
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy mô hình cho {speaker}")
        return None
    
    model = SpeakerRNN(input_dim, hidden_dim=128, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_speaker(speaker, dvectors):
    model = load_rnn_model(speaker, dvectors.shape[1])
    if model is None:
        return []

    x_test = torch.tensor(np.array(dvectors), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        predictions = model(x_test).numpy()
    
    return predictions

# ======= Chạy thử nghiệm =======
audio_file = "train_voice/vnoi_talkshow/raw.wav"
script_file = "train_voice/vnoi_talkshow/script.txt"

# Load script và trích xuất đặc trưng D-Vector
segments, speakers = load_script(script_file)
dvector_dict = extract_dvectors(audio_file, zip(segments, speakers))

# Huấn luyện mô hình RNN
for speaker, dvectors in dvector_dict.items():
    train_rnn_for_speaker(speaker, dvectors)
