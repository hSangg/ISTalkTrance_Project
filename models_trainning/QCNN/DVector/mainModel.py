import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

N_QUBITS = 4
N_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
DVECTOR_DIM = 256
SEGMENT_DURATION = 1.0
SAMPLE_RATE = 16000
NUM_BOOSTING_ESTIMATORS = 1

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Quantum circuit for the QNN.
    
    Args:
        inputs: Input features
        weights: Trainable weights
    """
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)
            qml.RZ(weights[l][i][1], wires=i)
        
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class DVectorExtractor(nn.Module):
    """
    D-Vector speaker embedding extractor based on a deep neural network
    """
    def __init__(self, input_dim=40, hidden_dim=128, dvector_dim=DVECTOR_DIM):
        super(DVectorExtractor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.encoder_output_size = input_dim // 8 * 128
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        
        self.dvector_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, dvector_dim),
            nn.BatchNorm1d(dvector_dim),
            nn.ReLU()
        )
        
        self.compressor = nn.Sequential(
            nn.Linear(dvector_dim, N_QUBITS),
            nn.Tanh()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.transpose(1, 2)
        
        x = x.unsqueeze(1)
        
        x = x.reshape(batch_size, 1, -1)
        
        x = self.encoder(x)
        
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        dvector = self.dvector_projector(context)
        
        compressed = self.compressor(dvector)
        
        return compressed

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        x = self.qlayer(x)
        return x

class QDVectorHybrid(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, input_dim=40):
        super(QDVectorHybrid, self).__init__()
        
        self.dvector = DVectorExtractor(input_dim=input_dim, dvector_dim=DVECTOR_DIM)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        self.post_processing = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        x = self.dvector(x)
        x = self.qnn(x)
        x = self.post_processing(x)
        return x

class BoostingQDVector:
    def __init__(self, n_estimators=NUM_BOOSTING_ESTIMATORS, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, input_dim=40, learning_rate=LEARNING_RATE):
        self.n_estimators = n_estimators
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.models = []
        self.weights = np.ones(n_estimators) / n_estimators
        
    def fit(self, train_loader, test_loader, epochs=EPOCHS):
        """Train multiple QDVector models and combine them using a boosting approach"""
        
        sample_weights = np.ones(len(train_loader.dataset)) / len(train_loader.dataset)
        
        for i in range(self.n_estimators):
            print(f"\nTraining model {i+1}/{self.n_estimators}")
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_loader.dataset),
                replacement=True
            )
            
            weighted_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                drop_last=True
            )
            
            model = QDVectorHybrid(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=self.n_classes,
                input_dim=self.input_dim
            )
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            train_losses = []
            test_accuracies = []
            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                
                for batch_features, batch_labels in weighted_train_loader:
                    optimizer.zero_grad()
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in test_loader:
                        outputs = model(batch_features)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_labels.size(0)
                        correct += (predicted == batch_labels).sum().item()
                
                test_accuracy = 100 * correct / total
                avg_loss = running_loss / len(weighted_train_loader)
                train_losses.append(avg_loss)
                test_accuracies.append(test_accuracy)
                
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            
            self.models.append(model)
            
            model.eval()
            all_preds = []
            all_labels = []
            dataset_indices = []
            
            idx = 0
            with torch.no_grad():
                for features, labels in train_loader:
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    
                    for j in range(len(labels)):
                        all_preds.append(predicted[j].item())
                        all_labels.append(labels[j].item())
                        dataset_indices.append(idx)
                        idx += 1
            
            errors = np.array(all_preds) != np.array(all_labels)
            error_rate = np.mean(errors)
            
            error_rate = max(min(error_rate, 0.999), 0.001)
            
            model_weight = 0.5 * np.log((1 - error_rate) / error_rate)
            self.weights[i] = model_weight
            
            for j, idx in enumerate(dataset_indices):
                if idx < len(sample_weights):
                    if errors[j]:
                        sample_weights[idx] *= np.exp(model_weight)
                    else:
                        sample_weights[idx] *= np.exp(-model_weight)
            
            if len(sample_weights) > 0:
                sample_weights = sample_weights / np.sum(sample_weights)
            
            print(f"Model {i+1} error rate: {error_rate:.4f}, weight: {model_weight:.4f}")
        
        self.weights = self.weights / np.sum(self.weights)
        print(f"Final model weights: {self.weights}")
        
        return self
    
    def predict(self, features):
        """Make predictions using the ensemble of models"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(features)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities)
        
        ensemble_pred = torch.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            ensemble_pred += self.weights[i] * pred
        
        _, predicted = torch.max(ensemble_pred, 1)
        return predicted

def extract_features_from_file(audio_file, sr=SAMPLE_RATE):
    """
    Extract spectral features from an audio file for d-vector processing.
    Returns mel-spectrograms instead of MFCCs.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, 
            fmin=20, fmax=8000,
            hop_length=int(sr/100)
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        log_mel_spec = log_mel_spec.T
        
        return log_mel_spec, y, sr
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        raise

def time_to_seconds(time_str):
    """
    Convert time string in format 'H:M:S' to seconds.
    """
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_script(script_file):
    """
    Parse the script file to get segments with labels.
    Returns a list of (start_time, end_time, label) tuples.
    """
    segments = []
    try:
        with open(script_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = time_to_seconds(parts[0])
                    end_time = time_to_seconds(parts[1])
                    label = parts[2]
                    segments.append((start_time, end_time, label))
    except Exception as e:
        print(f"Error parsing script file {script_file}: {e}")
        raise
        
    return segments

def create_dataset_from_audio(audio_file, script_file, segment_duration=SEGMENT_DURATION):
    """
    Create a dataset from audio file and script file.
    Returns a list of (spectrogram_segment, label) tuples.
    """
    specs, y, sr = extract_features_from_file(audio_file)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds, Spectrogram shape: {specs.shape}")
    
    segments = parse_script(script_file)
    print(f"Script parsed: {len(segments)} segments found")
    
    frames_per_second = specs.shape[0] / (len(y) / sr)
    print(f"Spectrogram frames per second: {frames_per_second:.2f}")
    
    dataset = []
    for start_time, end_time, label in segments:
        start_frame = int(start_time * frames_per_second)
        end_frame = int(end_time * frames_per_second)
        
        spec_segment = specs[start_frame:end_frame]
        
        if len(spec_segment) < 1:
            continue
        
        frames_per_segment = int(segment_duration * frames_per_second)
        
        for i in range(0, len(spec_segment), frames_per_segment):
            segment = spec_segment[i:i+frames_per_segment]
            if len(segment) < frames_per_segment // 2:
                continue
                
            if len(segment) < frames_per_segment:
                padded = np.zeros((frames_per_segment, specs.shape[1]))
                padded[:len(segment)] = segment
                segment = padded
            elif len(segment) > frames_per_segment:
                segment = segment[:frames_per_segment]
            
            dataset.append((segment, label))
    
    print(f"Dataset created: {len(dataset)} samples")
    return dataset

class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_speaker_recognition_qdvector(audio_file, script_file):
    """
    Train a QDVector model for speaker recognition using spectral features.
    """
    print("Creating dataset from audio and script...")
    dataset = create_dataset_from_audio(audio_file, script_file)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset. Check the audio file and script.")
    
    features = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    input_dim = features.shape[2]
    
    print(f"Initializing Boosting QDVector with {NUM_BOOSTING_ESTIMATORS} estimators...")
    boosting_model = BoostingQDVector(
        n_estimators=NUM_BOOSTING_ESTIMATORS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        input_dim=input_dim,
        learning_rate=LEARNING_RATE
    )
    
    boosting_model.fit(train_loader, test_loader, epochs=EPOCHS)
    
    print("\nFinal evaluation on test set:")
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for features, labels in test_loader:
        predictions = boosting_model.predict(features)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    final_accuracy = 100 * correct / total
    print(f"Ensemble test accuracy: {final_accuracy:.2f}%")
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('qdvector_confusion_matrix.png')
    
    print("\nComparing with best individual model:")
    best_model_idx = np.argmax(boosting_model.weights)
    best_model = boosting_model.models[best_model_idx]
    
    best_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = best_model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    best_model_accuracy = 100 * correct / total
    print(f"Best individual model (#{best_model_idx+1}) accuracy: {best_model_accuracy:.2f}%")
    print(f"Ensemble improvement: {final_accuracy - best_model_accuracy:.2f}%")
    
    return boosting_model, label_encoder

def predict_speaker(model, audio_file, label_encoder, segment_duration=1.0):
    """
    Predict speaker for each segment in an audio file using the boosting ensemble.
    """
    specs, y, sr = extract_features_from_file(audio_file)
    
    frames_per_second = specs.shape[0] / (len(y) / sr)
    frames_per_segment = int(segment_duration * frames_per_second)
    
    segments = []
    predictions = []
    times = []
    
    for i in range(0, len(specs), frames_per_segment):
        segment = specs[i:i+frames_per_segment]
        if len(segment) < frames_per_segment // 2:
            continue
            
        if len(segment) < frames_per_segment:
            padded = np.zeros((frames_per_segment, specs.shape[1]))
            padded[:len(segment)] = segment
            segment = padded
        elif len(segment) > frames_per_segment:
            segment = segment[:frames_per_segment]
        
        feature = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        
        predicted = model.predict(feature)
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
        
        start_time = i / frames_per_second
        end_time = min((i + frames_per_segment) / frames_per_second, len(y) / sr)
        
        segments.append((start_time, end_time))
        predictions.append(predicted_label)
        times.append(f"{int(start_time // 60):02d}:{int(start_time % 60):02d}")
    
    results = pd.DataFrame({
        'Start Time': [f"{int(s[0] // 60):02d}:{int(s[0] % 60):02d}" for s in segments],
        'End Time': [f"{int(s[1] // 60):02d}:{int(s[1] % 60):02d}" for s in segments],
        'Predicted Speaker': predictions
    })
    
    return results

def save_qdvector_ensemble(model, label_encoder, filename='qdvector_boosting_model.pth'):
    """Save the QDVector ensemble model to disk"""
    model_data = {
        'n_estimators': model.n_estimators,
        'n_qubits': model.n_qubits,
        'n_layers': model.n_layers,
        'n_classes': model.n_classes,
        'input_dim': model.input_dim,
        'weights': model.weights,
        'classes': label_encoder.classes_,
    }
    
    model_states = []
    for i, qdvector_model in enumerate(model.models):
        model_states.append(qdvector_model.state_dict())
    
    model_data['model_states'] = model_states
    
    torch.save(model_data, filename)
    print(f"Model saved to {filename}")

def load_qdvector_ensemble(filename='qdvector_boosting_model.pth'):
    """Load the QDVector ensemble model from disk"""
    model_data = torch.load(filename)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['classes']
    
    boosting_model = BoostingQDVector(
        n_estimators=model_data['n_estimators'],
        n_qubits=model_data['n_qubits'],
        n_layers=model_data['n_layers'],
        n_classes=model_data['n_classes'],
        input_dim=model_data['input_dim']
    )
    
    boosting_model.weights = model_data['weights']
    
    for i, state_dict in enumerate(model_data['model_states']):
        if i < len(boosting_model.models):
            qdvector_model = QDVectorHybrid(
                n_qubits=model_data['n_qubits'],
                n_layers=model_data['n_layers'],
                n_classes=model_data['n_classes'],
                input_dim=model_data['input_dim']
            )
            qdvector_model.load_state_dict(state_dict)
            boosting_model.models[i] = qdvector_model
    
    print(f"Model loaded from {filename}")
    return boosting_model, label_encoder

if __name__ == "__main__":
    audio_file = os.path.join("..", "..", "train_data", "meeting_1", "raw.wav")
    script_file = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")
        
    model, label_encoder = train_speaker_recognition_qdvector(audio_file, script_file)
    
    save_qdvector_ensemble(model, label_encoder, 'qdvector_boosting_model.pth')