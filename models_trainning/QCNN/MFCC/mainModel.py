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
MFCC_FEATURES = 13
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

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, mfcc_features=MFCC_FEATURES):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, N_QUBITS, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        return x.squeeze(-1)

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        x = self.qlayer(x)
        return x

class QCNNHybrid(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, mfcc_features=MFCC_FEATURES):
        super(QCNNHybrid, self).__init__()
        
        self.cnn = CNNFeatureExtractor(input_channels=1, mfcc_features=mfcc_features)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        self.post_processing = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        
        x = self.qnn(x)
        
        x = self.post_processing(x)
        
        return x

class BoostingQCNN:
    def __init__(self, n_estimators=NUM_BOOSTING_ESTIMATORS, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, mfcc_features=MFCC_FEATURES, learning_rate=LEARNING_RATE):
        self.n_estimators = n_estimators
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.mfcc_features = mfcc_features
        self.learning_rate = learning_rate
        self.models = []
        self.weights = np.ones(n_estimators) / n_estimators
        
    def fit(self, train_loader, test_loader, epochs=EPOCHS):
        """Train multiple QCNN models and combine them using a boosting approach"""
        
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
            
            model = QCNNHybrid(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=self.n_classes,
                mfcc_features=self.mfcc_features
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
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(features)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities)
        
        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            ensemble_pred += self.weights[i] * pred
        
        # Get the class with highest probability
        _, predicted = torch.max(ensemble_pred, 1)
        return predicted

# Audio processing functions
def extract_mfcc_from_file(audio_file, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES):
    """
    Extract MFCC features from an audio file.
    Returns a matrix of MFCC features with shape (n_frames, n_mfcc).
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Transpose to get (n_frames, n_mfcc) shape
        mfcc = mfcc.T
        
        return mfcc, y, sr
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
    Returns a list of (mfcc_segment, label) tuples.
    """
    mfcc, y, sr = extract_mfcc_from_file(audio_file)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds, MFCC shape: {mfcc.shape}")
    
    segments = parse_script(script_file)
    print(f"Script parsed: {len(segments)} segments found")
    
    frames_per_second = mfcc.shape[0] / (len(y) / sr)
    print(f"MFCC frames per second: {frames_per_second:.2f}")
    
    dataset = []
    for start_time, end_time, label in segments:
        start_frame = int(start_time * frames_per_second)
        end_frame = int(end_time * frames_per_second)
        
        mfcc_segment = mfcc[start_frame:end_frame]
        
        if len(mfcc_segment) < 1:
            continue
        
        frames_per_segment = int(segment_duration * frames_per_second)
        for i in range(0, len(mfcc_segment), frames_per_segment):
            segment = mfcc_segment[i:i+frames_per_segment]
            if len(segment) < frames_per_segment // 2:
                continue
                
            if len(segment) < frames_per_segment:
                padded = np.zeros((frames_per_segment, mfcc.shape[1]))
                padded[:len(segment)] = segment
                segment = padded
            elif len(segment) > frames_per_segment:
                segment = segment[:frames_per_segment]
            
            avg_segment = np.mean(segment, axis=0)
            
            dataset.append((avg_segment, label))
    
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

def train_speaker_recognition_qcnn(audio_file, script_file):
    """
    Train a QCNN for speaker recognition using MFCC features.
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
    
    print(f"Initializing Boosting QCNN with {NUM_BOOSTING_ESTIMATORS} estimators...")
    boosting_model = BoostingQCNN(
        n_estimators=NUM_BOOSTING_ESTIMATORS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        mfcc_features=MFCC_FEATURES,
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
    plt.savefig('qcnn_confusion_matrix.png')
    
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
    mfcc, y, sr = extract_mfcc_from_file(audio_file)
    
    frames_per_second = mfcc.shape[0] / (len(y) / sr)
    frames_per_segment = int(segment_duration * frames_per_second)
    
    segments = []
    predictions = []
    times = []
    
    for i in range(0, len(mfcc), frames_per_segment):
        segment = mfcc[i:i+frames_per_segment]
        if len(segment) < frames_per_segment // 2:
            continue
            
        if len(segment) < frames_per_segment:
            padded = np.zeros((frames_per_segment, mfcc.shape[1]))
            padded[:len(segment)] = segment
            segment = padded
        elif len(segment) > frames_per_segment:
            segment = segment[:frames_per_segment]
        
        avg_segment = np.mean(segment, axis=0)
        
        feature = torch.tensor(avg_segment, dtype=torch.float32).unsqueeze(0)
        
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

def save_qcnn_ensemble(model, label_encoder, filename='qcnn_boosting_model.pth'):
    """Save the QCNN ensemble model to disk"""
    model_data = {
        'n_estimators': model.n_estimators,
        'n_qubits': model.n_qubits,
        'n_layers': model.n_layers,
        'n_classes': model.n_classes,
        'mfcc_features': model.mfcc_features,
        'weights': model.weights,
        'classes': label_encoder.classes_,
    }
    
    model_states = []
    for i, qcnn_model in enumerate(model.models):
        model_states.append(qcnn_model.state_dict())
    
    model_data['model_states'] = model_states
    
    torch.save(model_data, filename)
    print(f"Model saved to {filename}")

def load_qcnn_ensemble(filename='qcnn_boosting_model.pth'):
    """Load the QCNN ensemble model from disk"""
    model_data = torch.load(filename)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['classes']
    
    boosting_model = BoostingQCNN(
        n_estimators=model_data['n_estimators'],
        n_qubits=model_data['n_qubits'],
        n_layers=model_data['n_layers'],
        n_classes=model_data['n_classes'],
        mfcc_features=model_data['mfcc_features']
    )
    
    boosting_model.weights = model_data['weights']
    
    for i, state_dict in enumerate(model_data['model_states']):
        if i < len(boosting_model.models):
            qcnn_model = QCNNHybrid(
                n_qubits=model_data['n_qubits'],
                n_layers=model_data['n_layers'],
                n_classes=model_data['n_classes'],
                mfcc_features=model_data['mfcc_features']
            )
            qcnn_model.load_state_dict(state_dict)
            boosting_model.models[i] = qcnn_model
    
    print(f"Model loaded from {filename}")
    return boosting_model, label_encoder

if __name__ == "__main__":
    audio_file = os.path.join("..", "..", "train_data", "meeting_1", "raw.wav")
    script_file = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")
        
    model, label_encoder = train_speaker_recognition_qcnn(audio_file, script_file)
    
    save_qcnn_ensemble(model, label_encoder, 'qcnn_boosting_model.pth')
