import os
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from speechbrain.inference.classifiers import EncoderClassifier
from torch import nn
from torch.utils.data import Dataset, DataLoader

TRAIN_ROOT = "train_voice"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20


# ==== Dataset and Model ====
class XvectorDataset(Dataset):
    def __init__(self, xvectors, labels, label_to_idx):
        self.xvectors = xvectors
        self.labels = labels
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.xvectors)

    def __getitem__(self, idx):
        x = torch.tensor(self.xvectors[idx], dtype=torch.float32)
        # Ensure x is a flat vector
        if x.dim() > 1:
            x = torch.mean(x, dim=0) if x.size(0) > 1 else x.reshape(-1)
        y = torch.tensor(self.label_to_idx[self.labels[idx]], dtype=torch.long)
        return x, y

def custom_collate(batch):
    """Custom collate function to handle tensors of different shapes."""
    xs = []
    ys = []
    for x, y in batch:
        # Ensure x is flat and consistent
        if x.dim() > 1:
            x = torch.mean(x, dim=0) if x.size(0) > 1 else x.reshape(-1)
        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.stack(ys)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h_n, _) = self.rnn(x)
        out = self.fc(h_n[-1])
        return out


# ==== Utils ====
def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            start, end, speaker = line.strip().split()
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers


def process_embedding(embedding):
    if embedding.ndim == 3:
        embedding = embedding.reshape(embedding.shape[0], -1)
    elif embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    if embedding.shape[1] == 1024:
        embedding = embedding[:, :512]
    return embedding

def extract_xvectors(audio_path, segments, speakers, classifier):
    waveform, sample_rate = torchaudio.load(audio_path)
    xvector_dict = defaultdict(list)

    for (start, end), speaker in zip(segments, speakers):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        with torch.no_grad():
            embedding = classifier.encode_batch(segment_waveform).squeeze().numpy()
        embedding = process_embedding(embedding)
        xvector_dict[speaker].append(embedding)
    return xvector_dict

# ==== Training Logic ====
def train_rnn_model(X, y, label_to_idx, input_size):
    num_classes = len(label_to_idx)

    # Preprocess to ensure consistent shapes
    standardized_X = []
    for x in X:
        if x.ndim == 2:
            if x.shape[0] > 1:
                standardized_X.append(np.mean(x, axis=0))
            else:
                standardized_X.append(x.reshape(-1))
        else:
            standardized_X.append(x)

    dataset = XvectorDataset(standardized_X, y, label_to_idx)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    # Update input size to match the standardized vectors
    if standardized_X[0].ndim == 1:
        input_size = standardized_X[0].shape[0]
    else:
        input_size = standardized_X[0].shape[1]

    model = RNNClassifier(input_size=input_size, hidden_size=128, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📚 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    return model


def evaluate_model(model, X_test, y_test, label_to_idx):
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for x, label in zip(X_test, y_test):
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = model(x_tensor)
            pred_idx = output.argmax(dim=1).item()
            preds.append(idx_to_label[pred_idx])
            true.append(label)

    print("\n==== 📊 Evaluation Report ====")
    print(classification_report(true, preds, zero_division=0))
    print(f"🎯 Accuracy: {accuracy_score(true, preds):.4f}")


# ==== Main Pipeline ====
def run_rnn_pipeline():
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": "cpu"}
    )

    xvectors, labels = [], []

    for folder in os.listdir(TRAIN_ROOT):
        subfolder_path = os.path.join(TRAIN_ROOT, folder)
        audio_path = os.path.join(subfolder_path, "raw.WAV")
        script_path = os.path.join(subfolder_path, "script.txt")

        print("🥺 at: ", audio_path)

        if not os.path.exists(audio_path) or not os.path.exists(script_path):
            continue

        segments, speakers = load_script(script_path)
        xvector_dict = extract_xvectors(audio_path, segments, speakers, classifier)

        for speaker, vecs in xvector_dict.items():
            for v in vecs:
                xvectors.append(v)
                labels.append(speaker)

    print(f"📦 Loaded {len(xvectors)} xvectors")

    input_size = xvectors[0].shape[1]
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(xvectors, labels)):
        print(f"\n=== Fold {fold + 1} ===")
        X_train = [xvectors[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test = [xvectors[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]

        model = train_rnn_model(X_train, y_train, label_to_idx, input_size)
        evaluate_model(model, X_test, y_test, label_to_idx)

        # Calculate accuracy manually
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for x, label in zip(X_test, y_test):
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                output = model(x_tensor)
                pred_idx = output.argmax(dim=1).item()
                preds.append(pred_idx)
                true.append(label_to_idx[label])

        acc = accuracy_score(true, preds)
        accuracies.append(acc)

    print("\n🎉 Final 3-Fold Cross Validation Accuracy:")
    print("📊 Accuracies per fold:", ["{:.4f}".format(a) for a in accuracies])
    print("🔁 Mean Accuracy: {:.4f}".format(np.mean(accuracies)))


if __name__ == "__main__":
    print("==== 🚀 Training RNN for Speaker Recognition ====")
    run_rnn_pipeline()
