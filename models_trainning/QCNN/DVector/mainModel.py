import os

import hmmlearn.hmm as hmm
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
NUM_BOOSTING_ESTIMATORS = 1

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

class QDVectorHybrid(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, dvector_dim=256):
        super(QDVectorHybrid, self).__init__()
        
        self.compressor = DVectorCompressor(input_dim=dvector_dim, output_dim=n_qubits)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        self.post_processing = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        x = self.compressor(x)
        x = self.qnn(x)
        x = self.post_processing(x)
        return x

class BoostingQDVector:
    def __init__(self, n_estimators=NUM_BOOSTING_ESTIMATORS, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, dvector_dim=256, learning_rate=LEARNING_RATE):
        self.n_estimators = n_estimators
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.dvector_dim = dvector_dim
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
                dvector_dim=self.dvector_dim
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
            
            # Update sample weights based on errors
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
            
            # Avoid division by zero or log(0)
            error_rate = max(min(error_rate, 0.999), 0.001)
            
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
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            ensemble_pred += self.weights[i] * pred
        
        _, predicted = torch.max(ensemble_pred, 1)
        return predicted

class HMMModel:
    def __init__(self, n_components=3, n_features=N_QUBITS):
        self.models = {}
        self.n_components = n_components
        self.n_features = n_features
    
    def fit(self, features_dict):
        """Train HMM models for each speaker"""
        for speaker, features in features_dict.items():
            # Convert to numpy array if not already
            features = np.array(features)
            
            # Initialize and train HMM model
            hmm_model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            hmm_model.fit(features)
            self.models[speaker] = hmm_model
    
    def predict(self, features):
        """Predict speaker based on features"""
        scores = {}
        
        for speaker, model in self.models.items():
            score = model.score(features)
            scores[speaker] = score
        
        # Return speaker with highest log likelihood
        return max(scores.items(), key=lambda x: x[1])[0]

class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def evaluate_model(model, test_loader, label_encoder, is_boosting_model=False):
    """
    Evaluate model performance with comprehensive metrics
    
    Args:
        model: The model to evaluate (either standard PyTorch model or BoostingQDVector)
        test_loader: DataLoader containing test data
        label_encoder: LabelEncoder for converting between numeric and string labels
        is_boosting_model: Boolean flag to indicate if model is a BoostingQDVector model
    
    Returns:
        metrics_df: DataFrame containing performance metrics
        report: Classification report string
    """
    all_preds = []
    all_labels = []
    
    for features, labels in test_loader:
        # Get model predictions based on model type
        if is_boosting_model:
            # For BoostingQDVector model
            predictions = model.predict(features)
        else:
            # For standard PyTorch models with forward method
            with torch.no_grad():
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
    Train the speaker recognition system using quantum neural network and HMM.
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
    
    # Train Quantum-Enhanced Boosting model
    print(f"Initializing Boosting QDVector with {NUM_BOOSTING_ESTIMATORS} estimators...")
    boosting_model = BoostingQDVector(
        n_estimators=NUM_BOOSTING_ESTIMATORS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_classes=n_classes,
        dvector_dim=dvector_dim,
        learning_rate=LEARNING_RATE
    )
    
    boosting_model.fit(train_loader, test_loader, epochs=EPOCHS)
    
    # Final evaluation on test set
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
    
    # Confusion matrix
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
    
    # Train HMM for temporal modeling
    print("\nTraining HMM models for temporal sequence modeling...")
    
    # Extract compressed features for HMM
    speaker_features = {speaker: [] for speaker in label_encoder.classes_}
    
    # Process train set to get features per speaker
    for i, model in enumerate(boosting_model.models):
        model.eval()
        with torch.no_grad():
            for features, labels in train_loader:
                # Get compressed features before quantum processing
                compressed = model.compressor(features)
                
                # Group by speaker
                for j, label in enumerate(labels):
                    speaker = label_encoder.classes_[label.item()]
                    speaker_features[speaker].append(compressed[j].cpu().numpy())
    
    # Train HMM models
    hmm_model = HMMModel(n_components=3, n_features=N_QUBITS)
    hmm_model.fit(speaker_features)
    
    print("Speaker recognition system training complete!")
    print("\nFinal evaluation on test set:")
    metrics_df, report = evaluate_model(boosting_model, test_loader, label_encoder, is_boosting_model=True)
    
    # Save metrics to CSV
    metrics_df.to_csv('speaker_recognition_metrics.csv', index=False)
    print("\nMetrics saved to speaker_recognition_metrics.csv")
    
    return {
        'quantum_model': boosting_model,
        'hmm_model': hmm_model,
        'speechbrain_model': speechbrain_model,
        'label_encoder': label_encoder
    }

def predict_speaker(models, audio_file, segment_duration=1.0):
    """
    Predict speaker for each segment in an audio file using the hybrid system.
    """
    # Unpack models
    quantum_model = models['quantum_model']
    hmm_model = models['hmm_model']
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
    predictions = []
    quantum_features = []
    
    for dvector in dvectors:
        dvector_tensor = torch.tensor(dvector, dtype=torch.float32).unsqueeze(0)
        
        # Get best model from ensemble
        best_model_idx = np.argmax(quantum_model.weights)
        best_model = quantum_model.models[best_model_idx]
        
        # Get compressed features for HMM
        with torch.no_grad():
            compressed = best_model.compressor(dvector_tensor)
            quantum_features.append(compressed.squeeze().cpu().numpy())
            
            # Get quantum model prediction
            outputs = best_model(dvector_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
            predictions.append(predicted_label)
    
    # Refine predictions using HMM for temporal coherence
    quantum_features = np.array(quantum_features)
    
    # Apply a sliding window for HMM prediction
    window_size = 5
    refined_predictions = []
    
    for i in range(len(quantum_features)):
        if i < window_size // 2 or i >= len(quantum_features) - window_size // 2:
            # Keep quantum prediction at boundaries
            refined_predictions.append(predictions[i])
        else:
            # Use HMM in the middle with context
            window = quantum_features[i - window_size // 2:i + window_size // 2 + 1]
            hmm_prediction = hmm_model.predict(window)
            refined_predictions.append(hmm_prediction)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Start Time': [f"{int(s[0] // 60):02d}:{int(s[0] % 60):02d}" for s in segments],
        'End Time': [f"{int(s[1] // 60):02d}:{int(s[1] % 60):02d}" for s in segments],
        'Quantum Model': predictions,
        'HMM Refined': refined_predictions
    })
    
    return results

def save_models(models, filename='speaker_recognition_models.pth'):
    """Save the trained models to disk"""
    model_data = {
        'quantum_model': {
            'n_estimators': models['quantum_model'].n_estimators,
            'n_qubits': models['quantum_model'].n_qubits,
            'n_layers': models['quantum_model'].n_layers,
            'n_classes': models['quantum_model'].n_classes,
            'dvector_dim': models['quantum_model'].dvector_dim,
            'weights': models['quantum_model'].weights,
        },
        'label_encoder_classes': models['label_encoder'].classes_,
    }
    
    # Save quantum model states
    model_states = []
    for i, qdvector_model in enumerate(models['quantum_model'].models):
        model_states.append(qdvector_model.state_dict())
    
    model_data['quantum_model_states'] = model_states
    
    # Save HMM models
    hmm_models = {}
    for speaker, model in models['hmm_model'].models.items():
        hmm_models[speaker] = {
            'means': model.means_,
            'covars': model.covars_,
            'transmat': model.transmat_,
            'startprob': model.startprob_
        }
    
    model_data['hmm_models'] = hmm_models
    
    torch.save(model_data, filename)
    print(f"Models saved to {filename}")

def load_models(filename='speaker_recognition_models.pth'):
    """Load the trained models from disk"""
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
    boosting_model = BoostingQDVector(
        n_estimators=model_data['quantum_model']['n_estimators'],
        n_qubits=model_data['quantum_model']['n_qubits'],
        n_layers=model_data['quantum_model']['n_layers'],
        n_classes=model_data['quantum_model']['n_classes'],
        dvector_dim=model_data['quantum_model']['dvector_dim']
    )
    
    boosting_model.weights = model_data['quantum_model']['weights']
    
    for i, state_dict in enumerate(model_data['quantum_model_states']):
        if i < len(boosting_model.models):
            qdvector_model = QDVectorHybrid(
                n_qubits=model_data['quantum_model']['n_qubits'],
                n_layers=model_data['quantum_model']['n_layers'],
                n_classes=model_data['quantum_model']['n_classes'],
                dvector_dim=model_data['quantum_model']['dvector_dim']
            )
            qdvector_model.load_state_dict(state_dict)
            boosting_model.models[i] = qdvector_model
    
    # Recreate HMM models
    hmm_model = HMMModel(n_features=model_data['quantum_model']['n_qubits'])
    hmm_model.models = {}
    
    for speaker, model_params in model_data['hmm_models'].items():
        model = hmm.GaussianHMM(
            n_components=model_params['means'].shape[0], 
            covariance_type="diag"
        )
        model.means_ = model_params['means']
        model.covars_ = model_params['covars']
        model.transmat_ = model_params['transmat']
        model.startprob_ = model_params['startprob']
        hmm_model.models[speaker] = model
    
    print(f"Models loaded from {filename}")
    
    return {
        'quantum_model': boosting_model,
        'hmm_model': hmm_model,
        'speechbrain_model': speechbrain_model,
        'label_encoder': label_encoder
    }

if __name__ == "__main__":
    audio_file = os.path.join("..", "..", "train_data", "meeting_1", "raw.WAV")
    script_file = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")
        
    # Train the speaker recognition system
    models = train_speaker_recognition_system(audio_file, script_file)
    
    # Save the trained models
    save_models(models, 'speaker_recognition_models.pth')

    