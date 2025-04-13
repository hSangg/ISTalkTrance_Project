import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from speechbrain.inference.speaker import SpeakerRecognition

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load speaker embedding model
spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="speechbrain_models/spkrec-ecapa-voxceleb"
)


class SpeakerClassifierRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeakerClassifierRNN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, speaker = parts
                segments.append((time_to_seconds(start), time_to_seconds(end)))
                speakers.append(speaker)
    return segments, speakers


def extract_dvectors(audio_path, segments, speakers):
    waveform, sample_rate = torchaudio.load(audio_path)
    dvectors, labels = [], []

    for (start, end), speaker in zip(segments, speakers):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        if end_sample > waveform.shape[1]:  # Prevent overflow
            end_sample = waveform.shape[1]

        if end_sample <= start_sample:
            continue

        segment_waveform = waveform[:, start_sample:end_sample]

        embedding = spk_model.encode_batch(segment_waveform)
        embedding = embedding.squeeze().flatten().numpy()

        dvectors.append(embedding)
        labels.append(speaker)

    return dvectors, labels


def cross_validate_multiclass(dvectors, labels, num_folds=5, num_epochs=20):
    if not dvectors:
        print("⚠️ No dvectors to train on!")
        return

    dvectors = np.array(dvectors)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    input_dim = dvectors.shape[1]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_true_labels = []
    all_pred_labels = []

    print(f"\n🔁 Starting {num_folds}-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(dvectors, encoded_labels), 1):
        print(f"\n📂 Fold {fold}/{num_folds}")

        x_train = torch.tensor(dvectors[train_idx], dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(encoded_labels[train_idx], dtype=torch.long)

        x_test = torch.tensor(dvectors[test_idx], dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(encoded_labels[test_idx], dtype=torch.long)

        model = SpeakerClassifierRNN(input_dim, hidden_dim=128, output_dim=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(x_test)
            predicted_classes = torch.argmax(predictions, dim=1).numpy()

        true_labels = label_encoder.inverse_transform(y_test.numpy())
        pred_labels = label_encoder.inverse_transform(predicted_classes)

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

        print(classification_report(true_labels, pred_labels, zero_division=0))

    print("\n📊 Overall classification report across all folds:")
    print(classification_report(all_true_labels, all_pred_labels, zero_division=0))


# === Load all data ===
root_folder = "train_voice"
all_dvectors = []
all_labels = []

print("🚀 Start processing...")

for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    audio_path = os.path.join(subfolder_path, "raw.WAV")
    script_path = os.path.join(subfolder_path, "script.txt")

    if os.path.exists(audio_path) and os.path.exists(script_path):
        print("🎙 Processing:", subfolder_path)
        segments, speakers = load_script(script_path)
        dvectors, labels = extract_dvectors(audio_path, segments, speakers)
        all_dvectors.extend(dvectors)
        all_labels.extend(labels)

# === Cross-validate multi-class speaker model ===
cross_validate_multiclass(all_dvectors, all_labels)
