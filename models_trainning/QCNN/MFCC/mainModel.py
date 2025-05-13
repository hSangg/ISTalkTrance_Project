import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)
torch.manual_seed(42)

N_QUBITS = 7
N_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 10
MFCC_FEATURES = 13
SEGMENT_DURATION = 1.0
SAMPLE_RATE = 16000

dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)
            qml.RZ(weights[l][i][1], wires=i)
        
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        if N_QUBITS > 1:
            qml.CNOT(wires=[N_QUBITS - 1, 0])
    
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
        
        self.dense = nn.Linear(128, N_QUBITS)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dense(x)  # This should output a tensor of size N_QUBITS
        return x

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        # Make sure x has the right shape for the quantum circuit
        batch_size = x.shape[0]
        results = []
        
        # Process each sample in the batch individually
        for i in range(batch_size):
            # Get a single input vector
            x_i = x[i]
            # Ensure it has the right dimension (N_QUBITS)
            if x_i.dim() == 0:  # If it's a scalar
                x_i = x_i.unsqueeze(0).repeat(N_QUBITS)
            elif x_i.size(0) < N_QUBITS:  # If it has fewer elements than needed
                padding = torch.zeros(N_QUBITS - x_i.size(0), device=x_i.device)
                x_i = torch.cat([x_i, padding])
            elif x_i.size(0) > N_QUBITS:  # If it has more elements than needed
                x_i = x_i[:N_QUBITS]
            
            # Apply quantum layer to this sample
            result = self.qlayer(x_i)
            results.append(result)
        
        # Stack results back into a batch
        return torch.stack(results)

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

def extract_mfcc_from_file(audio_file, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES):
    try:
        print(f"Processing file: {audio_file}")
        
        y, sr = librosa.load(audio_file, sr=sr)
        print(f"Loaded audio with sample rate: {sr}, duration: {librosa.get_duration(y=y, sr=sr)} seconds")

        n_mels = max(n_mfcc, 40)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
        
        mfcc = mfcc.T
        return mfcc, y, sr

    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        raise

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_script(script_file):
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

def evaluate_model(model, test_loader, label_encoder, device):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = label_encoder.inverse_transform(all_labels)
    all_preds = label_encoder.inverse_transform(all_preds)
    
    # Macro
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Weighted
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Logging
    logging.info(f"Precision (Macro): {precision_macro:.2f}")
    logging.info(f"Recall (Macro): {recall_macro:.2f}")
    logging.info(f"F1-Score (Macro): {f1_macro:.2f}")

    logging.info(f"Precision (Weighted): {precision_weighted:.2f}")
    logging.info(f"Recall (Weighted): {recall_weighted:.2f}")
    logging.info(f"F1-Score (Weighted): {f1_weighted:.2f}")

    logging.info(f"Accuracy: {accuracy:.2f}")

    
    metrics_df = pd.DataFrame({
        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                   'F1-Score (Weighted)', 'Accuracy'],
        'Value': [precision_macro, recall_macro, f1_macro, f1_weighted, accuracy]
    })

    # report = classification_report(
    #     all_labels,
    #     all_preds,
    #     labels=label_encoder.transform(label_encoder.classes_),
    #     target_names=label_encoder.classes_,
    #     zero_division=0
    # )
    
    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=label_encoder.classes_,
    #             yticklabels=label_encoder.classes_)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('qcnn_confusion_matrix.png')
    # plt.close()
    
    return metrics_df, accuracy

def load_all_datasets(root_dir, min_segments=3):
    all_datasets = []
    speaker_counts = {}
    
    print(f"Scanning {root_dir} for datasets...")
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            audio_file = os.path.join(subdir_path, "raw.WAV")
            if not os.path.exists(audio_file):
                audio_file = os.path.join(subdir_path, "raw.wav")
            
            script_file = os.path.join(subdir_path, "script.txt")
            
            if os.path.exists(audio_file) and os.path.exists(script_file):
                print(f"\nProcessing dataset in {subdir_path}")
                try:
                    dataset = create_dataset_from_audio(audio_file, script_file)
                    if dataset:
                        for _, speaker in dataset:
                            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                        all_datasets.append(dataset)
                except Exception as e:
                    print(f"Error processing {subdir_path}: {e}")
    
    valid_speakers = {speaker for speaker, count in speaker_counts.items() 
                     if count >= min_segments}
    
    filtered_dataset = []
    for dataset in all_datasets:
        filtered_segments = [(features, speaker) for features, speaker in dataset
                            if speaker in valid_speakers]
        if filtered_segments:
            filtered_dataset.extend(filtered_segments)
    
    print(f"\nFound {len(filtered_dataset)} valid segments from {len(valid_speakers)} speakers "
          f"(after filtering speakers with <{min_segments} segments)")
    
    return filtered_dataset

def train_speaker_recognition_with_cv(root_dir, n_folds=3, min_segments=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    dataset = load_all_datasets(root_dir, min_segments=min_segments)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset. Check the audio files and scripts.")
    
    features = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    # Calculate class weights ensuring we have weights for all classes
    unique_classes = np.unique(encoded_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=encoded_labels)
    
    # Verify we have weights for all classes
    assert len(class_weights) == n_classes, \
        f"Class weights length ({len(class_weights)}) doesn't match number of classes ({n_classes})"
    
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    features = np.array(features)
    encoded_labels = np.array(encoded_labels)
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    start_fold = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):

        if fold < start_fold:
            print(f"Skipping fold {fold+1}")
            continue
            
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{n_folds}")
        logging.info(f"\n{'='*50}")
        logging.info(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*50}")
        
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = encoded_labels[train_idx], encoded_labels[test_idx]
        
        # Recompute class weights for this specific fold's training data
        fold_unique_classes = np.unique(y_train)
        fold_class_weights = compute_class_weight('balanced', classes=fold_unique_classes, y=y_train)
        
        # Create a full weight vector for all classes
        full_weights = np.ones(n_classes)
        for cls, weight in zip(fold_unique_classes, fold_class_weights):
            full_weights[cls] = weight
            
        weights = torch.tensor(full_weights, dtype=torch.float).to(device)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = SpeakerDataset(X_train_tensor, y_train_tensor)
        test_dataset = SpeakerDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
        model = QCNNHybrid(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=n_classes)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
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
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            test_accuracy = 100 * correct / total
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            test_accuracies.append(test_accuracy)
            
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            logging.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        
        print(f"\nFinal evaluation for fold {fold+1}:")
        logging.info(f"\nFinal evaluation for fold {fold+1}:")
        torch.save(model.state_dict(), f'qcnn_model_fold_{fold+1}.pth')

        metrics_df, accuracy = evaluate_model(model, test_loader, label_encoder, device)
        
        print("\nClassification Report:")
        logging.info("\nClassification Report:")
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'model': model,
            'metrics': metrics_df
        })
        
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    
    accuracies = [result['accuracy'] for result in fold_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    for fold, result in enumerate(fold_results):
        print(f"Fold {fold+1} Accuracy: {result['accuracy']:.4f}")
    
    best_fold_idx = np.argmax(accuracies)
    best_model = fold_results[best_fold_idx]['model']
    
    print(f"\nBest model from fold {best_fold_idx+1} with accuracy: {accuracies[best_fold_idx]:.4f}")
    torch.save(best_model.state_dict(), 'qcnn_best_model.pth')
    
    return best_model, label_encoder, fold_results

def predict_speaker(model, audio_file, label_encoder, segment_duration=1.0, device='cuda'):
    model.to(device)
    model.eval()
    
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
        
        feature = torch.tensor(avg_segment, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(feature)
            _, predicted = torch.max(output.data, 1)
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

def save_model(model, label_encoder, filename='qcnn_model.pth'):
    model_data = {
        'n_qubits': N_QUBITS,
        'n_layers': N_LAYERS,
        'n_classes': len(label_encoder.classes_),
        'mfcc_features': MFCC_FEATURES,
        'classes': label_encoder.classes_,
        'state_dict': model.state_dict()
    }
    
    torch.save(model_data, filename)
    print(f"Model saved to {filename}")

def load_model(filename='qcnn_model.pth'):
    model_data = torch.load(filename)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = model_data['classes']
    
    model = QCNNHybrid(
        n_qubits=model_data['n_qubits'],
        n_layers=model_data['n_layers'],
        n_classes=model_data['n_classes'],
        mfcc_features=model_data['mfcc_features']
    )
    
    model.load_state_dict(model_data['state_dict'])
    
    print(f"Model loaded from {filename}")
    return model, label_encoder

if __name__ == "__main__":
    train_dir = "../../train_voice"
    
    model, label_encoder, fold_results = train_speaker_recognition_with_cv(train_dir, n_folds=3)
    
    save_model(model, label_encoder, 'mfcc_qcnn_speaker_recognition_model.pth')