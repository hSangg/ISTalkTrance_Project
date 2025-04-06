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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from speechbrain.pretrained import EncoderClassifier
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)

# Configuration parameters
N_QUBITS = 4
N_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 30
SEGMENT_DURATION = 1.0
SAMPLE_RATE = 16000

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Quantum circuit for the QNN.
    
    Args:
        inputs: Input features (compressed x-vectors)
        weights: Trainable weights
    """
    # Encode the input features into quantum states
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

class XVectorCompressor(nn.Module):
    """
    Compresses pretrained x-vectors to feed into the quantum circuit
    """
    def __init__(self, input_dim=512, output_dim=N_QUBITS):
        super(XVectorCompressor, self).__init__()
        
        self.compression = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
            nn.Tanh()
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

class QXVectorCNN(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, xvector_dim=512):
        super(QXVectorCNN, self).__init__()
        
        self.compressor = XVectorCompressor(input_dim=xvector_dim, output_dim=n_qubits)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        # Add a small convolutional layer after quantum processing
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        
        # Calculate the size after convolution and pooling
        self.fc_input_size = self._get_conv_output_size(n_qubits)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )
    
    def _get_conv_output_size(self, n_qubits):
        dummy_input = torch.zeros(1, 1, n_qubits)
        dummy_output = self.conv_layer(dummy_input)
        return dummy_output.numel()
        
    def forward(self, x):
        x = self.compressor(x)
        
        x = self.qnn(x)
        
        x = x.unsqueeze(1)
        
        x = self.conv_layer(x)
        x = x.view(-1, self.fc_input_size)
        
        x = self.classifier(x)
        
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
    
    Args:
        model: The neural classifier model
        test_loader: DataLoader containing test data
        label_encoder: LabelEncoder for converting between numeric and string labels
    
    Returns:
        metrics_df: DataFrame containing performance metrics
        report: Classification report string
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
    
    original_labels = label_encoder.inverse_transform(all_labels)
    original_preds = label_encoder.inverse_transform(all_preds)
    
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
    
    metrics_df = pd.DataFrame({
        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                  'F1-Score (Weighted)', 'Accuracy'],
        'Value': [precision, recall, f1, f1_weighted, accuracy]
    })
    
    report = classification_report(original_labels, original_preds, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(report)

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

def extract_xvectors(audio_segments, xvector_model):
    """
    Extract speaker embeddings (x-vectors) using pretrained SpeechBrain model
    """
    xvectors = []
    
    for segment in audio_segments:
        with torch.no_grad():
            embedding = xvector_model.encode_batch(torch.tensor(segment).unsqueeze(0))
            xvectors.append(embedding.squeeze().cpu().numpy())
    
    return np.array(xvectors)

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
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds")
    
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
        
        samples_per_segment = int(segment_duration * sr)
        
        for i in range(0, len(segment_audio), samples_per_segment):
            segment = segment_audio[i:i+samples_per_segment]
            
            if len(segment) < samples_per_segment // 2:
                continue
                
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
    Train the speaker recognition system using QCNN with XVector embeddings.
    """
    print("Loading SpeechBrain pretrained X-Vector model...")
    xvector_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    
    print("Extracting audio segments from audio and script...")
    dataset = extract_audio_segments(audio_file, script_file)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset. Check the audio file and script.")
    
    audio_segments = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    print("Extracting X-Vectors using SpeechBrain model...")
    xvectors = extract_xvectors(audio_segments, xvector_model)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    xvectors = torch.tensor(xvectors, dtype=torch.float32)
    labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(
        xvectors, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    xvector_dim = xvectors.shape[1]
    
    print(f"Initializing QCNN model for {n_classes} speakers...")
    qcnn_model = QXVectorCNN(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        xvector_dim=xvector_dim
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qcnn_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training QCNN model for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        qcnn_model.train()
        running_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = qcnn_model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        qcnn_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = qcnn_model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('qcnn_training_history.png')
    plt.close()
    
    print("QCNN model training complete!")
    print("\nFinal evaluation on test set:")
    metrics_df, report = evaluate_model(qcnn_model, test_loader, label_encoder)
    
    metrics_df.to_csv('qcnn_speaker_recognition_metrics.csv', index=False)
    print("\nMetrics saved to qcnn_speaker_recognition_metrics.csv")
    
    print("\nVisualizing quantum embeddings...")
    visualize_quantum_embeddings(qcnn_model, test_loader, label_encoder)
    
    return {
        'qcnn_model': qcnn_model,
        'xvector_model': xvector_model,
        'label_encoder': label_encoder
    }

def visualize_quantum_embeddings(model, data_loader, label_encoder):
    """
    Visualize the quantum embeddings using t-SNE.
    """
    from sklearn.manifold import TSNE
    
    # Collect embeddings and labels
    quantum_embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for features, batch_labels in data_loader:
            # Get compressed embeddings
            compressed = model.compressor(features)
            
            # Get quantum embeddings
            quantum_outputs = model.qnn(compressed)
            
            quantum_embeddings.extend(quantum_outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    quantum_embeddings = np.array(quantum_embeddings)
    labels = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(quantum_embeddings)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            color=colors[i],
            label=label_encoder.classes_[label]
        )
    
    plt.legend()
    plt.title('t-SNE Visualization of Quantum Embeddings')
    plt.savefig('quantum_embeddings_tsne.png')
    plt.close()

def predict_speaker(models, audio_file, segment_duration=1.0):
    """
    Predict speaker for each segment in an audio file using the QCNN model.
    """
    # Unpack models
    qcnn_model = models['qcnn_model']
    xvector_model = models['xvector_model']
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
    
    # Extract x-vectors
    xvectors = extract_xvectors(audio_segments, xvector_model)
    
    # Make predictions
    qcnn_model.eval()
    predictions = []
    confidence_scores = []
    
    for xvector in xvectors:
        xvector_tensor = torch.tensor(xvector, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Get model prediction
            outputs = qcnn_model(xvector_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction and confidence
            conf_value, predicted = torch.max(probabilities, 1)
            predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
            
            predictions.append(predicted_label)
            confidence_scores.append(conf_value.item())
    
    # Apply temporal smoothing using a sliding window majority vote
    window_size = 5
    smoothed_predictions = []
    
    for i in range(len(predictions)):
        if i < window_size // 2 or i >= len(predictions) - window_size // 2:
            # Keep original prediction at boundaries
            smoothed_predictions.append(predictions[i])
        else:
            # Use majority vote in the middle with context
            window = predictions[i - window_size // 2:i + window_size // 2 + 1]
            from collections import Counter
            most_common = Counter(window).most_common(1)[0][0]
            smoothed_predictions.append(most_common)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Start Time': [f"{int(s[0] // 60):02d}:{int(s[0] % 60):02d}" for s in segments],
        'End Time': [f"{int(s[1] // 60):02d}:{int(s[1] % 60):02d}" for s in segments],
        'Prediction': predictions,
        'Confidence': confidence_scores,
        'Smoothed': smoothed_predictions
    })
    
    return results

def save_models(models, filename='qcnn_speaker_recognition_models.pth'):
    """Save the trained models to disk"""
    model_data = {
        'qcnn_model': models['qcnn_model'].state_dict(),
        'label_encoder_classes': models['label_encoder'].classes_,
    }
    
    torch.save(model_data, filename)
    print(f"Models saved to {filename}")

def load_models(filename='qcnn_speaker_recognition_models.pth'):
    """Load the trained models from disk"""
    model_data = torch.load(filename)
    
    # Recreate label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['label_encoder_classes']
    
    # Load XVector model
    xvector_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    
    # Recreate QCNN model
    xvector_dim = 512  # Default XVector dimension
    qcnn_model = QXVectorCNN(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=len(label_encoder.classes_),
        xvector_dim=xvector_dim
    )
    qcnn_model.load_state_dict(model_data['qcnn_model'])
    
    print(f"Models loaded from {filename}")
    
    return {
        'qcnn_model': qcnn_model,
        'xvector_model': xvector_model,
        'label_encoder': label_encoder
    }

def analyze_quantum_circuit(model):
    """
    Analyze the quantum circuit used in the model.
    Prints circuit statistics and visualizes circuit structure.
    """
    print("\nQuantum Circuit Analysis:")
    print(f"Number of qubits: {N_QUBITS}")
    print(f"Number of layers: {N_LAYERS}")
    
    # Count number of quantum gates
    n_gates = N_QUBITS * 2 * N_LAYERS  # RY and RZ gates
    n_cnot = N_QUBITS * N_LAYERS  # CNOT gates
    print(f"Total parameterized gates: {n_gates}")
    print(f"Total CNOT gates: {n_cnot}")
    
    # Draw the circuit if supported by PennyLane version
    try:
        # Create a dummy circuit just for visualization
        dummy_inputs = torch.zeros(N_QUBITS)
        dummy_weights = torch.zeros((N_LAYERS, N_QUBITS, 2))
        
        # Draw the circuit
        fig, ax = qml.draw_mpl(quantum_circuit)(dummy_inputs, dummy_weights)
        plt.savefig('quantum_circuit_visualization.png')
        plt.close()
        print("Circuit visualization saved to 'quantum_circuit_visualization.png'")
    except Exception as e:
        print(f"Could not visualize circuit due to: {str(e)}")

if __name__ == "__main__":
    audio_file = "train_data/gitlab_public_meeting/raw.wav"
    script_file = "train_data/gitlab_public_meeting/script.txt"
        
    models = train_speaker_recognition_system(audio_file, script_file)
    
    analyze_quantum_circuit(models['qcnn_model'])
    
    save_models(models, 'qcnn_speaker_recognition_models.pth')
