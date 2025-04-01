import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define constants
N_QUBITS = 4  # Number of qubits
N_LAYERS = 2  # Number of quantum layers
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
MFCC_FEATURES = 13  # Typical number of MFCC features
SEGMENT_DURATION = 1.0  # Duration of audio segments in seconds for MFCC extraction
SAMPLE_RATE = 16000  # Sample rate for audio processing
NUM_BOOSTING_ESTIMATORS = 1  # Number of models in the boosting ensemble

# Define the quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

# Define the quantum circuit as a QNode
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Quantum circuit for the QNN.
    
    Args:
        inputs: Input features (size must match N_QUBITS)
        weights: Trainable weights of shape (n_layers, n_qubits, 3)
    """
    # Encode the classical input data
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum gates
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)  # First parameter
            qml.RZ(weights[l][i][1], wires=i)  # Second parameter
            # Removed RX gate to match weight dimensions
        
        # Entanglement
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        if N_QUBITS > 1:  # Only add circular entanglement if more than 1 qubit
            qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    # Measure the expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, mfcc_features=MFCC_FEATURES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Additional dense layer for better feature transformation
        self.dense = nn.Linear(128, N_QUBITS)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dense(x)  # Additional transformation
        return x

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        # Initialize the weights for the quantum circuit
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Create the quantum layer
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        # Apply the quantum layer
        x = self.qlayer(x)
        return x

# QCNN Hybrid Model
class QCNNHybrid(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, mfcc_features=MFCC_FEATURES):
        super(QCNNHybrid, self).__init__()
        
        self.cnn = CNNFeatureExtractor(input_channels=1, mfcc_features=mfcc_features)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        # Linear layer to map from quantum outputs to class probabilities
        self.post_processing = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        # Extract features using CNN
        x = self.cnn(x)
        
        # Apply the quantum layer
        x = self.qnn(x)
        
        # Apply post-processing
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
        self.weights = np.ones(n_estimators) / n_estimators  # Equal weights initially
        
    def fit(self, train_loader, test_loader, epochs=EPOCHS):
        """Train multiple QCNN models and combine them using a boosting approach"""
        
        sample_weights = np.ones(len(train_loader.dataset)) / len(train_loader.dataset)
        
        for i in range(self.n_estimators):
            print(f"\nTraining model {i+1}/{self.n_estimators}")
            
            # Create a weighted sampler for this iteration
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_loader.dataset),
                replacement=True
            )
            
            # Create a new data loader with the weighted sampler
            weighted_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                drop_last=True
            )
            
            # Initialize a new model
            model = QCNNHybrid(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=self.n_classes,
                mfcc_features=self.mfcc_features
            )
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            train_losses = []
            test_accuracies = []
            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                
                for batch_features, batch_labels in weighted_train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                # Evaluate on test set
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
            
            # Add model to ensemble
            self.models.append(model)
            
            # Update sample weights based on misclassifications
            model.eval()
            all_preds = []
            all_labels = []
            dataset_indices = []
            
            idx = 0
            with torch.no_grad():
                for features, labels in train_loader:  # Use original loader to get ordered samples
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    
                    for j in range(len(labels)):
                        all_preds.append(predicted[j].item())
                        all_labels.append(labels[j].item())
                        dataset_indices.append(idx)
                        idx += 1
            
            # Only update weights for samples that were processed
            errors = np.array(all_preds) != np.array(all_labels)
            error_rate = np.mean(errors)
            
            # Avoid division by zero or negative log
            error_rate = max(min(error_rate, 0.999), 0.001)
            
            # Calculate model weight using AdaBoost formula
            model_weight = 0.5 * np.log((1 - error_rate) / error_rate)
            self.weights[i] = model_weight
            
            # Update sample weights
            for j, idx in enumerate(dataset_indices):
                if idx < len(sample_weights):
                    if errors[j]:
                        sample_weights[idx] *= np.exp(model_weight)
                    else:
                        sample_weights[idx] *= np.exp(-model_weight)
            
            # Normalize weights
            if len(sample_weights) > 0:
                sample_weights = sample_weights / np.sum(sample_weights)
            
            print(f"Model {i+1} error rate: {error_rate:.4f}, weight: {model_weight:.4f}")
        
        # Normalize model weights
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
    # Extract MFCC features from audio
    mfcc, y, sr = extract_mfcc_from_file(audio_file)
    print(f"Audio loaded: {len(y)/sr:.2f} seconds, MFCC shape: {mfcc.shape}")
    
    # Parse script file
    segments = parse_script(script_file)
    print(f"Script parsed: {len(segments)} segments found")
    
    # Calculate frames per second based on MFCC extraction
    frames_per_second = mfcc.shape[0] / (len(y) / sr)
    print(f"MFCC frames per second: {frames_per_second:.2f}")
    
    # Create dataset
    dataset = []
    for start_time, end_time, label in segments:
        # Convert time to frame indices
        start_frame = int(start_time * frames_per_second)
        end_frame = int(end_time * frames_per_second)
        
        # Get MFCC segment
        mfcc_segment = mfcc[start_frame:end_frame]
        
        # Skip segments that are too short
        if len(mfcc_segment) < 1:
            continue
        
        # Divide into smaller segments if needed
        frames_per_segment = int(segment_duration * frames_per_second)
        for i in range(0, len(mfcc_segment), frames_per_segment):
            segment = mfcc_segment[i:i+frames_per_segment]
            if len(segment) < frames_per_segment // 2:  # Skip if too short
                continue
                
            # Pad or truncate to ensure consistent size
            if len(segment) < frames_per_segment:
                # Pad with zeros
                padded = np.zeros((frames_per_segment, mfcc.shape[1]))
                padded[:len(segment)] = segment
                segment = padded
            elif len(segment) > frames_per_segment:
                # Truncate
                segment = segment[:frames_per_segment]
            
            # Average across time dimension to get a fixed-size feature vector
            avg_segment = np.mean(segment, axis=0)
            
            dataset.append((avg_segment, label))
    
    print(f"Dataset created: {len(dataset)} samples")
    return dataset

# Dataset class for PyTorch
class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
def evaluate_model(model, test_loader, label_encoder):
    all_preds = []
    all_labels = []
    
    for features, labels in test_loader:
        predictions = model.predict(features)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Convert labels back to original labels
    all_labels = label_encoder.inverse_transform(all_labels)
    all_preds = label_encoder.inverse_transform(all_preds)
    
    # Calculate Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"Precision (Macro): {precision:.2f}")
    print(f"Recall (Macro): {recall:.2f}")
    print(f"F1-Score (Macro): {f1:.2f}")
    print(f"F1-Score (Weighted): {f1_weighted:.2f}")
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.2f}")
    metrics_df = pd.DataFrame({
        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                   'F1-Score (Weighted)', 'Accuracy'],
        'Value': [precision, recall, f1, f1_weighted, accuracy]
    })
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return metrics_df, report

# Main function
def train_speaker_recognition_qcnn(audio_file, script_file):
    """
    Train a QCNN for speaker recognition using MFCC features.
    """
    print("Creating dataset from audio and script...")
    # Create dataset
    dataset = create_dataset_from_audio(audio_file, script_file)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset. Check the audio file and script.")
    
    # Extract features and labels
    features = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    # Convert to PyTorch tensors
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(encoded_labels, dtype=torch.long)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create data loaders
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # Initialize and train the boosting ensemble
    print(f"Initializing Boosting QCNN with {NUM_BOOSTING_ESTIMATORS} estimators...")
    boosting_model = BoostingQCNN(
        n_estimators=NUM_BOOSTING_ESTIMATORS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        mfcc_features=MFCC_FEATURES,
        learning_rate=LEARNING_RATE
    )
    
    # Train the ensemble
    boosting_model.fit(train_loader, test_loader, epochs=EPOCHS)
    
    # Final evaluation
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
    
    # Plot confusion matrix
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
    
    # Compare with individual models
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
    
    print("\nDetailed evaluation on test set:")
    metrics_df, report = evaluate_model(boosting_model, test_loader, label_encoder)
    
    print("\nClassification Report:")
    print(report)
    
    print("\nDetailed Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics to CSV
    metrics_df.to_csv('speaker_recognition_metrics.csv', index=False)
    print("\nMetrics saved to speaker_recognition_metrics.csv")
    
    return boosting_model, label_encoder, metrics_df

def predict_speaker(model, audio_file, label_encoder, segment_duration=1.0):
    """
    Predict speaker for each segment in an audio file using the boosting ensemble.
    """
    # Extract MFCC features
    mfcc, y, sr = extract_mfcc_from_file(audio_file)
    
    # Calculate frames per second
    frames_per_second = mfcc.shape[0] / (len(y) / sr)
    frames_per_segment = int(segment_duration * frames_per_second)
    
    # Process segments
    segments = []
    predictions = []
    times = []
    
    for i in range(0, len(mfcc), frames_per_segment):
        segment = mfcc[i:i+frames_per_segment]
        if len(segment) < frames_per_segment // 2:  # Skip if too short
            continue
            
        # Pad or truncate
        if len(segment) < frames_per_segment:
            padded = np.zeros((frames_per_segment, mfcc.shape[1]))
            padded[:len(segment)] = segment
            segment = padded
        elif len(segment) > frames_per_segment:
            segment = segment[:frames_per_segment]
        
        # Average across time dimension
        avg_segment = np.mean(segment, axis=0)
        
        # Convert to tensor
        feature = torch.tensor(avg_segment, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        predicted = model.predict(feature)
        # Fix: Handle single prediction properly
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
        
        # Calculate time
        start_time = i / frames_per_second
        end_time = min((i + frames_per_segment) / frames_per_second, len(y) / sr)
        
        segments.append((start_time, end_time))
        predictions.append(predicted_label)
        times.append(f"{int(start_time // 60):02d}:{int(start_time % 60):02d}")
    
    # Create a DataFrame for easier visualization
    results = pd.DataFrame({
        'Start Time': [f"{int(s[0] // 60):02d}:{int(s[0] % 60):02d}" for s in segments],
        'End Time': [f"{int(s[1] // 60):02d}:{int(s[1] % 60):02d}" for s in segments],
        'Predicted Speaker': predictions
    })
    
    return results

# Function to save the QCNN ensemble model
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
    
    # Save model states
    model_states = []
    for i, qcnn_model in enumerate(model.models):
        model_states.append(qcnn_model.state_dict())
    
    model_data['model_states'] = model_states
    
    torch.save(model_data, filename)
    print(f"Model saved to {filename}")

# Function to load the QCNN ensemble model
def load_qcnn_ensemble(filename='qcnn_boosting_model.pth'):
    """Load the QCNN ensemble model from disk"""
    model_data = torch.load(filename)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['classes']
    
    # Create boosting model
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
    
    # Train the model
    model, label_encoder = train_speaker_recognition_qcnn(audio_file, script_file)
    
    save_qcnn_ensemble(model, label_encoder, 'qcnn_boosting_model.pth')