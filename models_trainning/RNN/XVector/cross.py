import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
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

    return xvector_dict


def train_rnn_for_speaker(speaker, xvectors, num_epochs=50):
    if len(xvectors) == 0:
        return None

    xvectors = np.mean(xvectors, axis=1)
    input_dim = xvectors.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = SpeakerRNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.tensor(xvectors, dtype=torch.float32).unsqueeze(1)

    y_train = torch.ones((len(xvectors), 1), dtype=torch.float32)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model
def cross_validate_rnn(xvectors_dict, n_splits=5):
    all_xvectors = []
    all_speakers = []

    for speaker, xvectors in xvectors_dict.items():
        all_xvectors.extend(xvectors)
        all_speakers.extend([speaker] * len(xvectors))

    X = np.array(all_xvectors)
    y = np.array(all_speakers)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = {}
        for speaker in np.unique(y_train):
            speaker_xvectors = X_train[y_train == speaker]
            model = train_rnn_for_speaker(speaker, speaker_xvectors)
            models[speaker] = model

        y_true, y_pred = [], []
        for i in range(len(X_test)):
            speaker = y_test[i]
            model = models.get(speaker, None)

            if model is None:
                continue

            x_test = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = model(x_test).squeeze().numpy()

            distances = []
            for stored_speaker, stored_model in models.items():
                stored_embeddings = np.array([np.mean(stored_model.fc.weight.detach().numpy(), axis=0)])
                distance = np.linalg.norm(pred - stored_embeddings)
                distances.append((stored_speaker, distance))

            predicted_speaker = min(distances, key=lambda x: x[1])[0]

            y_true.append(speaker)
            y_pred.append(predicted_speaker)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(classification_report(y_true, y_pred))

    print("\n🔚 Final Evaluation Across All Folds:")
    print(classification_report(all_y_true, all_y_pred))


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
                continue

            if speaker not in speaker_xvectors:
                speaker_xvectors[speaker] = []
            speaker_xvectors[speaker].extend(xvectors)

cross_validate_rnn(speaker_xvectors)
