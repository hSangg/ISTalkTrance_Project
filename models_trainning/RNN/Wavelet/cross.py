import os

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader


# Define your SpeakerRNN and SpeakerDataset as before

class SpeakerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_wavelet_features(audio_path, segments_speakers, wavelet='db4', level=1):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy().squeeze()
    feature_list, label_list = [], []

    for (start, end), speaker in segments_speakers:
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
        label_list.append(speaker)

    return feature_list, label_list


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s


def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[0], parts[1], parts[2]
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers


def load_all_data(data_folder):
    all_features, all_labels = [], []
    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if os.path.isdir(subdir_path):
            audio_file = os.path.join(subdir_path, "raw.WAV")
            script_file = os.path.join(subdir_path, "script.txt")
            if os.path.exists(audio_file) and os.path.exists(script_file):
                segments, speakers = load_script(script_file)
                features, labels = extract_wavelet_features(audio_file, zip(segments, speakers))
                all_features.extend(features)
                all_labels.extend(labels)
    return all_features, all_labels


def train_rnn(train_features, train_labels, num_epochs=20, lr=0.001, hidden_size=64):
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
        model.train()  # Set model to training mode
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)  # Make sure the input has the right shape
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

    return model, speaker_to_idx

from sklearn.model_selection import StratifiedKFold

def cross_validate(features, labels, num_splits=5):
    # Convert labels to indices
    speakers = list(set(labels))
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    labels_idx = [speaker_to_idx[label] for label in labels]

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    fold = 1
    for train_idx, val_idx in skf.split(features, labels_idx):
        print(f"\n🔁 Fold {fold}/{num_splits}")

        # Split the data into training and validation sets
        train_features = np.array(features)[train_idx]
        val_features = np.array(features)[val_idx]
        train_labels = np.array(labels)[train_idx]
        val_labels = np.array(labels)[val_idx]

        # Convert train labels to indices for training
        train_labels_idx = [speaker_to_idx[s] for s in train_labels]

        # Train the model
        model, _ = train_rnn(train_features, train_labels_idx)

        # Make predictions on the validation set
        y_true = [speaker_to_idx[s] for s in val_labels]
        y_pred = []

        for inputs in val_features:
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted.item())

        # Print classification report with speaker names
        idx_to_speaker = {i: s for s, i in speaker_to_idx.items()}
        y_true_names = [idx_to_speaker[i] for i in y_true]
        y_pred_names = [idx_to_speaker[i] for i in y_pred]

        print(f"📊 Fold {fold} Classification Report:")
        print(classification_report(y_true_names, y_pred_names, zero_division=0))

        fold += 1



# Main code to load data, perform cross-validation and print reports

data_folder = "train_voice"
features, labels = load_all_data(data_folder)

if features and labels:
    cross_validate(features, labels)
else:
    print("Không có dữ liệu hợp lệ để huấn luyện.")
