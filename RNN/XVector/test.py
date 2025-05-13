import os

import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

SCRIPT_PREDICTED_TXT = "script_predicted.txt"

class SpeakerRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SpeakerRNN, self).__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def load_script(script_path):
    segments = []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            start, end, _ = line.strip().split()
            segments.append((time_to_seconds(start), time_to_seconds(end)))
    return segments


def load_rnn_model(speaker):
    model_path = f"rnn_xvector_models/{speaker}.pth"
    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location="cpu")
    input_dim = checkpoint["input_dim"]

    model = SpeakerRNN(input_dim, hidden_dim=128, output_dim=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_xvector(audio_path, start, end, classifier):
    waveform, sample_rate = torchaudio.load(audio_path)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment_waveform = waveform[:, start_sample:end_sample]

    with torch.no_grad():
        embedding = classifier.encode_batch(segment_waveform).squeeze().numpy()

    return embedding


def predict_speaker(audio_path, script_path, classifier, speaker_models):
    segments = load_script(script_path)
    predictions = []

    for start, end in segments:
        xvector = extract_xvector(audio_path, start, end, classifier)
        xvector_tensor = torch.tensor(xvector, dtype=torch.float32).unsqueeze(0)
        best_speaker = "unknown"
        best_score = float("-inf")

        for speaker, model in speaker_models.items():
            if model is None:
                continue

            with torch.no_grad():
                score = model(xvector_tensor).item()

            if score > best_score:
                best_score = score
                best_speaker = speaker

        predictions.append((start, end, best_speaker))

    return predictions


def write_predictions(output_path, predictions):
    with open(output_path, "w", encoding="utf-8") as file:
        for start, end, speaker in predictions:
            start_time = f"{start // 3600:02}:{(start % 3600) // 60:02}:{start % 60:02}"
            end_time = f"{end // 3600:02}:{(end % 3600) // 60:02}:{end % 60:02}"
            file.write(f"{start_time} {end_time} {speaker}\n")


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})
test_voice_dir = "test_voice"
speaker_models = {sp[:-4]: load_rnn_model(sp[:-4]) for sp in os.listdir("rnn_xvector_models") if sp.endswith(".pth")}

for subdir in os.listdir(test_voice_dir):
    subdir_path = os.path.join(test_voice_dir, subdir)
    audio_file = os.path.join(subdir_path, "raw.WAV")
    script_file = os.path.join(subdir_path, "script.txt")
    predict_file = os.path.join(subdir_path, SCRIPT_PREDICTED_TXT)

    if os.path.isfile(audio_file) and os.path.isfile(script_file):
        print(f"🎤 Đang dự đoán người nói trong: {subdir}")
        predictions = predict_speaker(audio_file, script_file, classifier, speaker_models)
        write_predictions(predict_file, predictions)
        print(f"✅ Đã lưu kết quả vào: {predict_file}")
