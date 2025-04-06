import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from speechbrain.pretrained import SpeakerRecognition
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# Configuration parameters
N_QUBITS = 4
N_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
SEGMENT_DURATION = 1.0
SAMPLE_RATE = 16000

# Initialize quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Quantum circuit for the QNN.
    
    Args:
        inputs: Input features (compressed d-vectors)
        weights: Trainable weights
    """
    # Encode the input features into quantum states
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum gates
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)
            qml.RZ(weights[l][i][1], wires=i)
        
        # Apply entanglement gates
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    # Return expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class DVectorCompressor(nn.Module):
    """
    Compresses pretrained d-vectors to feed into the quantum circuit
    """
    def __init__(self, input_dim=256, output_dim=N_QUBITS):
        super(DVectorCompressor, self).__init__()
        
        self.compression = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
            nn.Tanh()  # Bound values between -1 and 1 for quantum circuit
        )
        
    def forward(self, x):
        return self.compression(x)

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        return self.qlayer(x)

class QDVectorModel(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, dvector_dim=256):
        super(QDVectorModel, self).__init__()
        
        self.compressor = DVectorCompressor(input_dim=dvector_dim, output_dim=n_qubits)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        self.post_processing = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        x = self.compressor(x)
        x = self.qnn(x)
        x = self.post_processing(x)
        return x

class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def evaluate_model(model, test_loader, label_encoder):
    """
    Evaluate model performance with comprehensive metrics
    """
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predictions = torch.max(outputs.data, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert numeric labels back to original class names
    original_labels = label_encoder.inverse_transform(all_labels)
    original_preds = label_encoder.inverse_transform(all_preds)
    
    # Calculate metrics
    precision = precision_score(original_labels, original_preds, average='macro')
    recall = recall_score(original_labels, original_preds, average='macro')
    f1 = f1_score(original_labels, original_preds, average='macro')
    f1_weighted = f1_score(original_labels, original_preds, average='weighted')
    accuracy = accuracy_score(original_labels, original_preds)

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                  'F1-Score (Weighted)', 'Accuracy'],
        'Value': [precision, recall, f1, f1_weighted, accuracy]
    })
    
    # Generate classification report
    report = classification_report(original_labels, original_preds, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(original_labels, original_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return metrics_df, report

def extract_dvectors(audio_segments, speechbrain_model):
    """
    Extract speaker embeddings (d-vectors) using pretrained SpeechBrain model
    """
    dvectors = []
    
    for segment in audio_segments:
        with torch.no_grad():
            # Get embedding from speechbrain model
            embedding = speechbrain_model.encode_batch(torch.tensor(segment).unsqueeze(0))
            dvectors.append(embedding.squeeze().numpy())
    
    return np.array(dvectors)

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

def extract_audio_segments(audio_file, script_file, segment_duration=SEGMENT_DURATION):
    """
    Extract audio segments from the audio file based on the script.
    Returns a list of (audio_segment, label) tuples.
    """
    # Load audio file
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds")
    
    # Parse script
    segments = parse_script(script_file)
    print(f"Script parsed: {len(segments)} segments found")
    
    dataset = []
    for start_time, end_time, label in segments:
        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        segment_audio = y[start_sample:end_sample]
        
        if len(segment_audio) < 1:
            continue
        
        # Split into smaller segments for uniform processing
        samples_per_segment = int(segment_duration * sr)
        
        for i in range(0, len(segment_audio), samples_per_segment):
            segment = segment_audio[i:i+samples_per_segment]
            
            # Skip if segment is too short
            if len(segment) < samples_per_segment // 2:
                continue
                
            # Pad or trim to ensure uniform length
            if len(segment) < samples_per_segment:
                padded = np.zeros(samples_per_segment)
                padded[:len(segment)] = segment
                segment = padded
            elif len(segment) > samples_per_segment:
                segment = segment[:samples_per_segment]
            
            dataset.append((segment, label))
    
    print(f"Dataset created: {len(dataset)} samples")
    return dataset

def train_speaker_recognition_system(audio_file, script_file):
    """
    Train the speaker recognition system using quantum neural network.
    """
    print("Loading SpeechBrain pretrained speaker recognition model...")
    speechbrain_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    print("Extracting audio segments from audio and script...")
    dataset = extract_audio_segments(audio_file, script_file)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset. Check the audio file and script.")
    
    # Extract audio segments and labels
    audio_segments = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    # Extract d-vectors using pretrained model
    print("Extracting d-vectors using SpeechBrain model...")
    dvectors = extract_dvectors(audio_segments, speechbrain_model)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    # Convert to torch tensors
    dvectors = torch.tensor(dvectors, dtype=torch.float32)
    labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        dvectors, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # Get d-vector dimension
    dvector_dim = dvectors.shape[1]
    
    # Initialize Quantum QCNN model
    print(f"Initializing Quantum Speaker Recognition model...")
    quantum_model = QDVectorModel(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        dvector_dim=dvector_dim
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quantum_model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    test_accuracies = []
    
    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        quantum_model.train()
        running_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = quantum_model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluation
        quantum_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = quantum_model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    metrics_df, report = evaluate_model(quantum_model, test_loader, label_encoder)
    
    # Save metrics to CSV
    metrics_df.to_csv('speaker_recognition_metrics.csv', index=False)
    print("\nMetrics saved to speaker_recognition_metrics.csv")
    
    # Return models
    return {
        'quantum_model': quantum_model,
        'speechbrain_model': speechbrain_model,
        'label_encoder': label_encoder
    }

def predict_speaker(models, audio_file, segment_duration=1.0):
    """
    Predict speaker for each segment in an audio file using the quantum model.
    """
    # Unpack models
    quantum_model = models['quantum_model']
    speechbrain_model = models['speechbrain_model']
    label_encoder = models['label_encoder']
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    # Split into segments
    samples_per_segment = int(segment_duration * sr)
    
    segments = []
    times = []
    audio_segments = []
    
    for i in range(0, len(y), samples_per_segment):
        segment = y[i:i+samples_per_segment]
        if len(segment) < samples_per_segment // 2:
            continue
            
        if len(segment) < samples_per_segment:
            padded = np.zeros(samples_per_segment)
            padded[:len(segment)] = segment
            segment = padded
        elif len(segment) > samples_per_segment:
            segment = segment[:samples_per_segment]
        
        start_time = i / sr
        end_time = min((i + samples_per_segment) / sr, len(y) / sr)
        
        segments.append((start_time, end_time))
        times.append(f"{int(start_time // 60):02d}:{int(start_time % 60):02d}")
        audio_segments.append(segment)
    
    # Extract d-vectors
    dvectors = extract_dvectors(audio_segments, speechbrain_model)
    
    # Make quantum model predictions
    quantum_model.eval()
    predictions = []
    
    for dvector in dvectors:
        dvector_tensor = torch.tensor(dvector, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = quantum_model(dvector_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
            predictions.append(predicted_label)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Start Time': [f"{int(s[0] // 60):02d}:{int(s[0] % 60):02d}" for s in segments],
        'End Time': [f"{int(s[1] // 60):02d}:{int(s[1] % 60):02d}" for s in segments],
        'Speaker': predictions
    })
    
    return results

def save_model(models, filename='quantum_speaker_recognition_model.pth'):
    """Save the trained model to disk"""
    torch.save({
        'quantum_model_state': models['quantum_model'].state_dict(),
        'label_encoder_classes': models['label_encoder'].classes_
    }, filename)
    
    print(f"Model saved to {filename}")

def load_model(filename='quantum_speaker_recognition_model.pth'):
    """Load the trained model from disk"""
    model_data = torch.load(filename)
    
    # Recreate label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['label_encoder_classes']
    
    # Load SpeechBrain model
    speechbrain_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    # Recreate quantum model
    quantum_model = QDVectorModel(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=len(label_encoder.classes_)
    )
    
    quantum_model.load_state_dict(model_data['quantum_model_state'])
    
    print(f"Model loaded from {filename}")
    
    return {
        'quantum_model': quantum_model,
        'speechbrain_model': speechbrain_model,
        'label_encoder': label_encoder
    }

if __name__ == "__main__":
    audio_file = "train_data/gitlab_public_meeting/raw.wav"
    script_file = "train_data/gitlab_public_meeting/script.txt"
        
    # Train the speaker recognition system
    models = train_speaker_recognition_system(audio_file, script_file)
    
    # Save the trained model
    save_model(models, 'quantum_speaker_recognition_model.pth')
    
    # Test prediction on the same file
    results = predict_speaker(models, audio_file)
    results.to_csv('speaker_predictions.csv', index=False)
    print("\nPrediction results saved to speaker_predictions.csv")